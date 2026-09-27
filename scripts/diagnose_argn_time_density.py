"""Frozen head audit and paired-customer clock trajectories (descriptive only)."""
from datetime import datetime,timezone
import json
import numpy as np
import pandas as pd
import torch
from run_argn_time_density import ROOT,OUT,DOCS,CFG,META,verify,digest,write
from report_argn_time_density import ARMS,locate
from run_argn_state_first import SOURCE
from diagnose_argn_residual import decorate


def main():
    verify();dest=DOCS/'diagnostics';dest.mkdir(exist_ok=True);heads=[]
    for fs in CFG['fit_seeds']:
        p=OUT/f'worker_{fs}/history_relative'
        f=json.loads((p/'FIT_COMPLETE.json').read_text());h=torch.load(p/'time_head.pt',map_location='cpu',weights_only=True)
        zero=bool(h['state_dict']['net.2.weight'].eq(0).all() and h['state_dict']['net.2.bias'].eq(0).all())
        heads.append(dict(fit_seed=fs,selected_step=f['selected_step'],last_layer_exactly_zero=zero,
            output_depends_on_native_history=not zero,head_sha256=digest(p/'time_head.pt')))
    pd.DataFrame(heads).to_csv(dest/'relative_head_audit.csv',index=False)
    train=pd.read_parquet(META/'metadata_optimization.parquet');relations=[]
    for phase,g in train[train.past_gap.gt(0)].groupby('phase'):
        clock=np.log1p(g.past_gap.to_numpy());residual=np.log1p(g.gap.to_numpy())-clock
        relations.append(dict(phase=phase,events=len(g),clock_residual_correlation=float(np.corrcoef(clock,residual)[0,1]),
            residual_mean=float(np.mean(residual)),residual_median=float(np.median(residual))))
    pd.DataFrame(relations).to_csv(dest/'train_clock_dependence.csv',index=False)
    real=decorate(pd.read_parquet(SOURCE/'prepared/validation.parquet'));lengths=real.groupby('entity_id').size();rows=[]
    bins=[6,50,200,500,1000]
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            p,_=locate('frozen',fs,gs);base=pd.read_parquet(p);slen=base.groupby('entity_id').size()
            common=lengths.index[(lengths.ge(1000)&slen.reindex(lengths.index).ge(1000))]
            for arm in ['real']+ARMS:
                if arm=='real':d=real
                else:
                    p,_=locate(arm,fs,gs);d=decorate(pd.read_parquet(p))
                d=d[d.entity_id.isin(common)&d.event_index.ge(6)&d.event_index.lt(1000)&d.past_gap.gt(0)].copy()
                d['clock']=np.log1p(d.past_gap)
                anchor=d[d.event_index.lt(50)].groupby('entity_id').clock.median()
                d['delta']=d.clock-d.entity_id.map(anchor);d['abs_delta']=d.delta.abs()
                for a,b in zip(bins[:-1],bins[1:]):
                    g=d[d.event_index.ge(a)&d.event_index.lt(b)]
                    customer=g.groupby('entity_id')[['clock','delta','abs_delta']].mean()
                    rows.append(dict(arm=arm,fit_seed=fs,generation_seed=gs,start=a,end_exclusive=b,customers=len(customer),events=len(g),
                        mean_log_clock=float(customer.clock.mean()),mean_clock_change=float(customer.delta.mean()),
                        mean_absolute_clock_change=float(customer.abs_delta.mean()),gap_median=float(g.gap.median())))
    result=pd.DataFrame(rows);result.to_csv(dest/'clock_trajectories.csv',index=False)
    result.groupby(['arm','start','end_exclusive'])[['customers','mean_log_clock','mean_clock_change','mean_absolute_clock_change','gap_median']].mean().to_csv(dest/'clock_trajectory_means.csv')
    write(dest/'COMPLETE.json',dict(source_sha256=digest(__file__),plan_sha256=digest(DOCS/'DIAGNOSTIC_PLAN.md'),
        test_events_read=False,new_training=0,new_generation=0,completed_utc=datetime.now(timezone.utc).isoformat()))
    print(pd.DataFrame(heads).to_string(index=False));print(result.groupby(['arm','start']).mean(numeric_only=True).to_string())


if __name__=='__main__':main()
