"""Local bounded diagnostics of expressiveness, optimizer budget and cohort mix."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize,minimize_scalar
from scipy.stats import wasserstein_distance
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'artifacts/argn_clock_full_length_v1/local_source'),str(ROOT/'scripts'),str(ROOT)]
from run_full_length import ActualPrefix,CFG,verify,digest,write,objective,META
from report_full_length import locate
from run_argn_state_first import SOURCE
OUT=Path(__file__).resolve().parent
BINS=[(1,50),(50,200),(200,500),(500,1024),(1024,2048),(2048,np.inf)]


def payloads():
    p=torch.load(ROOT/'artifacts/argn_clock_regression_v1/time_head.pt',map_location='cpu',weights_only=True)
    pars={}
    for name,path in [('clock_prefix',ROOT/'docs/argn_clock_location_v1/clock_prefix_fit.json'),
                      ('clock_full_prefix',ROOT/'artifacts/argn_clock_full_length_v1/report/clock_full_prefix_fit.json')]:
        d=json.loads(path.read_text())['selected'];pars[name]=[d['a'],d['b']]
    return p,pars


def optimization():
    verify();p,pars=payloads();sim=ActualPrefix('optimization',p,None,CFG['fit_noise_seed'])
    ref=sim.run([0,0]);curve=[]
    def evaluate(theta):
        metrics=sim.run(theta);value=objective(metrics,ref)
        curve.append(dict(evaluation=len(curve),a=float(theta[0]),b=float(theta[1]),objective=value,**metrics))
        if len(curve)%16==0: print('BUDGET',len(curve),value,flush=True)
        return value
    evaluate([0,0]);fit=minimize(evaluate,np.zeros(2),method='Powell',bounds=CFG['bounds'],
        options=dict(maxfev=191,xtol=.001,ftol=.001))
    prior=pd.read_csv(ROOT/'artifacts/argn_clock_full_length_v1/report/clock_full_prefix_learning_curve.csv')
    replay=pd.DataFrame(curve)
    np.testing.assert_allclose(replay.iloc[:48][prior.columns],prior,rtol=0,atol=1e-9)
    replay.to_csv(OUT/'budget_curve.csv',index=False)
    best=min(curve,key=lambda r:r['objective']);pars['budget192']=[best['a'],best['b']]
    write(OUT/'budget_fit.json',dict(selected=best,success=bool(fit.success),message=str(fit.message),
                                  evaluations=len(curve),first48_reproduced=True))
    m=pd.read_parquet(META/'metadata_optimization.parquet');m=m[m.event_index.gt(0)]
    assert len(m)==len(sim.base_z)
    local=[];offsets=[]
    for lo,hi in BINS:
        mask=sim.nn&m.event_index.ge(lo).to_numpy()&m.event_index.lt(hi).to_numpy()
        base=np.sort(sim.base_z[mask]);real=np.sort(np.log1p(m.gap.to_numpy()[mask]))
        def score(delta):
            q=np.floor(np.expm1(np.clip(base+delta,0,np.log1p(1045691.)))+1e-8)
            return float(np.mean(np.abs(real-np.log1p(q))))
        fit=minimize_scalar(score,bounds=(-.35,.35),method='bounded',options=dict(maxiter=64,xatol=.0001))
        candidates=[0.,-.35,.35,float(fit.x)];delta=min(candidates,key=score);offsets.append(delta)
        local.append(dict(start=lo,end=-1 if not np.isfinite(hi) else int(hi),events=int(mask.sum()),
                          delta=delta,unadjusted_w1=score(0.),adjusted_w1=score(delta),success=bool(fit.success)))
    pd.DataFrame(local).to_csv(OUT/'local_offset_diagnostic.csv',index=False)
    rows=[];bybin=[]
    for split,seeds in [('optimization',[CFG['fit_noise_seed']]),('internal_validation',CFG['diagnostic_seeds'])]:
        meta=pd.read_parquet(META/f'metadata_{split}.parquet');meta=meta[meta.event_index.gt(0)]
        for ns in seeds:
            s=ActualPrefix(split,p,None,ns);base=s.base_z.copy()
            for arm in list(pars)+['local_offsets_diagnostic']:
                if arm=='local_offsets_diagnostic':
                    delta=np.zeros(len(meta))
                    for (lo,hi),value in zip(BINS,offsets):
                        ix=meta.event_index.ge(lo).to_numpy()&meta.event_index.lt(hi).to_numpy()&s.nn
                        delta[ix]=value
                    s.base_z=base+delta;theta=[0.,0.]
                else:s.base_z=base;theta=pars[arm];delta=(theta[0]+theta[1]*s.decay)*s.nn
                rows.append(dict(split=split,noise_seed=ns,arm=arm,**s.run(theta)))
                y=np.floor(np.expm1(np.clip(base+delta,0,np.log1p(1045691.)))+1e-8)
                for lo,hi in BINS:
                    mask=s.nn&meta.event_index.ge(lo).to_numpy()&meta.event_index.lt(hi).to_numpy()
                    bybin.append(dict(split=split,noise_seed=ns,arm=arm,start=lo,
                        events=int(mask.sum()),gap_w1=wasserstein_distance(np.log1p(meta.gap.to_numpy()[mask]),np.log1p(y[mask]))))
    pd.DataFrame(rows).to_csv(OUT/'conditional_diagnostics.csv',index=False)
    pd.DataFrame(bybin).to_csv(OUT/'conditional_position.csv',index=False)
    values=[]
    for n in [1,20,50,100,200,500,1024,2048]:
        values.append(dict(position=n,decaying_basis=float(np.exp(-n/50)),
                           **{arm:a+b*np.exp(-n/50) for arm,(a,b) in pars.items()}))
    pd.DataFrame(values).to_csv(OUT/'correction_shape.csv',index=False)
    verify();write(OUT/'OPTIMIZATION_COMPLETE.json',dict(diagnostic_only=True,new_native_fits=0,new_generations=0,
        test_events_read=False,source_sha256=digest(__file__)))
    print(pd.DataFrame(local).round(5).to_string(index=False),flush=True)
    print(pd.DataFrame(rows).groupby(['split','arm'])[['normal','personal']].mean().round(5).to_string(),flush=True)


def times(path):
    d=pd.read_parquet(path,columns=['entity_id','event_index','event_is_fraud','gap'])
    d=d.sort_values(['entity_id','event_index']).reset_index(drop=True)
    d['label']=pd.to_numeric(d.event_is_fraud).astype(int);d['gap']=pd.to_numeric(d.gap).astype(float)
    d.loc[d.event_index.eq(0),'gap']=np.nan
    d['clock']=d.groupby('entity_id').gap.transform(lambda x:x.shift().rolling(20,min_periods=5).median())
    return d


def cohorts():
    verify();real=times(SOURCE/'prepared/validation.parquet');lengths=real.groupby('entity_id').size();rows=[]
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            samples={a:times(locate(a,fs,gs)[0]) for a in ['clock_joint','clock_prefix','clock_full_prefix']}
            lens=samples['clock_joint'].groupby('entity_id').size()
            common=pd.concat([lengths,lens],axis=1).dropna().min(axis=1)
            for scope in ['all_matched','same_long_customers']:
                keep=common.index if scope=='all_matched' else common.index[common.ge(2048)]
                r=real[real.entity_id.isin(keep)&real.event_index.lt(real.entity_id.map(common))]
                for arm,d in samples.items():
                    d=d[d.entity_id.isin(keep)&d.event_index.lt(d.entity_id.map(common))]
                    for lo,hi in [(6,50),(50,200),(200,500),(500,1024),(1024,2048),(2048,np.inf)]:
                        a=r[r.label.eq(0)&r.clock.gt(0)&r.event_index.ge(lo)&r.event_index.lt(hi)]
                        b=d[d.label.eq(0)&d.clock.gt(0)&d.event_index.ge(lo)&d.event_index.lt(hi)]
                        if not len(a) or not len(b):continue
                        both=np.intersect1d(a.entity_id.unique(),b.entity_id.unique())
                        a=a[a.entity_id.isin(both)];b=b[b.entity_id.isin(both)]
                        wa=1/a.groupby('entity_id').entity_id.transform('size').to_numpy()
                        wb=1/b.groupby('entity_id').entity_id.transform('size').to_numpy()
                        x,y=np.log1p(a.gap),np.log1p(b.gap)
                        ca,cb=np.log1p(a.clock),np.log1p(b.clock)
                        rows.append(dict(scope=scope,arm=arm,fit_seed=fs,generation_seed=gs,start=lo,
                            customers=len(both),real_events=len(a),generated_events=len(b),
                            event_weighted_gap_w1=wasserstein_distance(x,y),
                            equal_customer_gap_w1=wasserstein_distance(x,y,u_weights=wa,v_weights=wb),
                            log_gap_bias=float(np.average(y,weights=wb)-np.average(x,weights=wa)),
                            log_clock_bias=float(np.average(cb,weights=wb)-np.average(ca,weights=wa))))
            print('COHORT',fs,gs,flush=True)
    result=pd.DataFrame(rows);result.to_csv(OUT/'cohort_diagnostic.csv',index=False)
    means=result.groupby(['scope','arm','start'])[['customers','event_weighted_gap_w1','equal_customer_gap_w1','log_gap_bias','log_clock_bias']].mean()
    means.to_csv(OUT/'cohort_means.csv');print(means.round(5).to_string(),flush=True)
    verify();write(OUT/'COHORT_COMPLETE.json',dict(posthoc_descriptive=True,new_fits=0,new_generations=0,
        test_events_read=False,source_sha256=digest(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['optimization','cohorts']);a=p.parse_args()
    path=OUT/f'{a.mode.upper()}_START.json';assert not path.exists()
    write(path,dict(source_sha256=digest(__file__),plan_sha256=digest(OUT/'PLAN.md'),
        utc=datetime.now(timezone.utc).isoformat(),test_events_read=False))
    optimization() if a.mode=='optimization' else cohorts()
