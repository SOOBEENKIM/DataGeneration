"""Paired oracle-label rollouts isolate feeding generated intervals into the clock."""
from datetime import datetime,timezone
import json
import numpy as np
import pandas as pd
import torch
from run_argn_time_density import ROOT,OUT,DOCS,META,CFG,verify,digest,write
from benchmarks.argn_time_density import TimeDensity,PastClock,clock_features
from benchmarks.argn_clock_regression import ClockRegression
from diagnose_argn_residual import w1


@torch.no_grad()
def main():
    verify();dest=DOCS/'clock_intervention';dest.mkdir(exist_ok=True)
    p=torch.load(OUT/'worker_20260930/history_relative/time_head.pt',map_location='cpu',weights_only=True)
    q=torch.load(OUT/'worker_20261001/history_relative/time_head.pt',map_location='cpu',weights_only=True)
    for name in ['prior_mean','prior_logits','prior_raw_scale']:
        torch.testing.assert_close(p['state_dict'][name],q['state_dict'][name],rtol=0,atol=0)
    for payload in [p,q]:
        for name in ['net.2.weight','net.2.bias']:assert payload['state_dict'][name].eq(0).all()
    relative=TimeDensity(p['dim'],p['arm'],p['components'],p['hidden']);relative.load_state_dict(p['state_dict']);relative.eval()
    path=ROOT/'artifacts/argn_clock_regression_v1/time_head.pt'
    q=torch.load(path,map_location='cpu',weights_only=True);gmr=ClockRegression(q['dim'],q['arm'],q['components'],q['hidden']);gmr.load_state_dict(q['state_dict']);gmr.eval()
    d=pd.read_parquet(META/'metadata_development.parquet');clock,valid=clock_features(d,p['fallback']);d=d.assign(clock=clock,valid=valid)
    customers=[g.iloc[:525] for _,g in d.groupby('entity_id',sort=False) if len(g)>=525]
    gaps=np.stack([g.gap.to_numpy() for g in customers]);labels=np.stack([g.label.to_numpy() for g in customers]);real_clock=np.stack([g.clock.to_numpy() for g in customers]);real_valid=np.stack([g.valid.to_numpy() for g in customers])
    codec=PastClock(p['fallback']);initial=codec.empty(len(customers))
    for t in range(25):initial=codec.advance(initial,gaps[:,t])
    rows=[]
    for name,head in [('history_relative',relative),('clock_joint',gmr)]:
        base=torch.zeros(len(customers),head.dim)
        for gs in CFG['sampling_seeds']:
            for arm in ['actual_clock','recursive_clock']:
                memory=initial.copy();rng=np.random.default_rng(gs);generated=[];used=[]
                for t in range(25,525):
                    c,v=(real_clock[:,t],real_valid[:,t]) if arm=='actual_clock' else codec.features(memory)
                    phase=torch.as_tensor(2*labels[:,t-1]+labels[:,t],dtype=torch.long)
                    y,_=head.sample(base,phase,torch.as_tensor(c,dtype=torch.float32),torch.as_tensor(v,dtype=torch.float32),rng,p['limits'],p['quantum'])
                    generated.append(y);used.append(c.copy());memory=codec.advance(memory,y)
                generated=np.stack(generated,axis=1);used=np.stack(used,axis=1)
                for horizon in [20,100,500]:
                    truth=gaps[:,25:25+horizon];refclock=real_clock[:,25:25+horizon];c=used[:,:horizon];y=generated[:,:horizon]
                    rows.append(dict(model=name,clock_source=arm,sampling_seed=gs,horizon=horizon,customers=len(customers),
                        events=truth.size,gap_w1=w1(truth.ravel(),y.ravel()),clock_log_mae=float(np.abs(c-refclock).mean()),
                        real_clock_median=float(np.median(np.expm1(refclock))),used_clock_median=float(np.median(np.expm1(c)))))
    data=pd.DataFrame(rows);data.to_csv(dest/'metrics.csv',index=False)
    mean=data.groupby(['model','clock_source','horizon']).mean(numeric_only=True);mean.to_csv(dest/'means.csv');print(mean.round(5).to_string())
    write(dest/'COMPLETE.json',dict(customers=len(customers),real_prefix=25,oracle_label_horizon=500,sampling_seeds=CFG['sampling_seeds'],
        not_free_generation_benchmark=True,new_fits=0,test_events_read=False,source_sha256=digest(__file__),
        plan_sha256=digest(DOCS/'CLOCK_INTERVENTION_PROTOCOL.md'),relative_head_sha256=digest(OUT/'worker_20260930/history_relative/time_head.pt'),
        gmr_head_sha256=digest(path),completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
