"""One preregistered progress/rollout time-output experiment; no test loading."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
import torch
import run_argn_time_density as temporal
from run_argn_clock_regression_amended import verify as verify_prior
from benchmarks.argn_clock_evolution import (ProgressClock, ProgressDensity, NumpyDensity,
    progress_features, evolution_generation)
from run_argn_time_density import ROOT, META, digest, write, seed, inputs, load_parent

OUT=ROOT/'artifacts/argn_clock_evolution_v1'; DOCS=ROOT/'docs/argn_clock_evolution_v1'
CFG=dict(arms=['progress_joint','progress_prefix','progress_rollout'],
    fit_seeds=temporal.CFG['fit_seeds'],generation_seeds=temporal.CFG['generation_seeds'],
    allowed_gpu_uuids=temporal.CFG['allowed_gpu_uuids'],horizon=1024,fit_noise_seed=20261201,
    diagnostic_seeds=[20261202,20261203],maxfev=48,bounds=[[-.35,.35],[-1.,1.]])


def verify():
    verify_prior(); m=json.loads((OUT/'MANIFEST.json').read_text()); assert m['config']==CFG
    for p,h in m['hashes'].items(): assert digest(p)==h,p


class ConditionalSimulation:
    """Actual label paths are training conditions, never copied into free generation."""
    def __init__(self, split, fallback):
        m=pd.read_parquet(META/f'metadata_{split}.parquet')
        m=m[m.event_index.lt(CFG['horizon'])].copy(); x,v=progress_features(m,fallback)
        ids=pd.factorize(m.entity_id,sort=False)[0]; t=m.event_index.to_numpy()
        shape=(int(t.max())+1,int(ids.max())+1); self.shape=shape
        self.mask=np.zeros(shape,bool); self.mask[t,ids]=t>0
        self.phase=np.zeros(shape,np.int64); self.phase[t,ids]=2*m.previous.clip(lower=0).to_numpy()+m.label.to_numpy()
        self.gaps=np.zeros(shape); self.gaps[t,ids]=m.gap.fillna(0).to_numpy()
        self.actual=np.zeros((*shape,2)); self.actual[t,ids]=x
        self.actual_raw=np.full(shape,fallback); self.actual_raw[t,ids]=m.past_gap.fillna(fallback).to_numpy()
        self.valid=np.zeros(shape,bool); self.valid[t,ids]=v>.5
        self.fallback=fallback; self.clock=ProgressClock(fallback)
        self.refz=np.log1p(self.gaps)
        self.refrel=np.log1p(self.gaps/np.maximum(self.actual_raw,1e-12))
        self.normal=self.mask&(self.phase==0); self.fraud=self.mask&(self.phase%2==1)
        self.onset=self.mask&(self.phase==1)&self.valid
        self.personal=self.normal&self.valid
        self.bins=[self.normal&(np.arange(shape[0])[:,None]>=a)&(np.arange(shape[0])[:,None]<b)
                   for a,b in [(1,50),(50,200),(200,CFG['horizon'])]]
        self.refsort={}
        for name,value,mask in [('normal',self.refz,self.normal),('fraud',self.refz,self.fraud),
            ('clock',self.actual[:,:,0],self.personal),('personal',self.refrel,self.personal),
            ('onset',self.refrel,self.onset)]+[(f'bin{i}',self.refz,k) for i,k in enumerate(self.bins)]:
            assert mask.any(),(split,name)
            self.refsort[name]=np.sort(value[mask])

    def run(self, density, correction, source, noise_seed):
        rng=np.random.default_rng(noise_seed)
        # Identical exogenous noise at each index and customer, even for terminated records.
        uniforms=rng.random(self.shape); normals=rng.standard_normal(self.shape)
        values=np.zeros(self.shape); used=np.zeros(self.shape); personal_denominator=np.full(self.shape,self.fallback)
        memory=self.clock.empty(self.shape[1]); memory=self.clock.advance(memory,np.zeros(self.shape[1]))
        raw_ring=np.full((self.shape[1],20),np.nan)
        for t in range(1,self.shape[0]):
            x=self.actual[t] if source=='actual' else self.clock.features(memory)[0]
            if source=='actual': personal_denominator[t]=self.actual_raw[t]
            elif t>=6: personal_denominator[t]=np.nanmedian(raw_ring,axis=1)
            p,mu,sigma=density.parameters(self.phase[t],x,correction)
            comp=(uniforms[t,:,None]>np.cumsum(p,axis=1)).sum(-1).clip(max=p.shape[-1]-1)
            rows=np.arange(len(comp)); z=mu[rows,comp]+sigma[rows,comp]*normals[t]
            values[t]=np.floor(np.expm1(np.clip(z,0,np.log1p(1045691.)))+1e-8)
            used[t]=x[:,0]; memory=self.clock.advance(memory,values[t])
            raw_ring[:,(t-1)%20]=values[t]
        z=np.log1p(values); relative=np.log1p(values/np.maximum(personal_denominator,1e-12))
        metrics={}
        for name,value,mask in [('normal',z,self.normal),('fraud',z,self.fraud),
            ('clock',used,self.personal),('personal',relative,self.personal),('onset',relative,self.onset)]+[
            (f'bin{i}',z,k) for i,k in enumerate(self.bins)]:
            metrics[name]=float(np.mean(np.abs(self.refsort[name]-np.sort(value[mask]))))
        return metrics


def objective(metrics, reference):
    value=.5*(metrics['normal']+np.mean([metrics[f'bin{i}'] for i in range(3)]))+.25*metrics['clock']
    for k,tolerance in [('personal',.01),('onset',.01),('fraud',.02)]:
        value+=4*max(metrics[k]-reference[k]-tolerance,0.)
    return float(value)


def save_head(payload, arm, correction):
    head=ProgressDensity(1,arm); head.load_state_dict(payload['state_dict'])
    head.correction.copy_(torch.as_tensor(correction,dtype=torch.float32))
    p={**payload,'arm':arm,'state_dict':head.state_dict()}
    dest=OUT/arm; dest.mkdir(exist_ok=True); torch.save(p,dest/'time_head.pt')
    return p


def prepare():
    verify_prior(); assert not (OUT/'MANIFEST.json').exists()
    paths=[Path(__file__),ROOT/'benchmarks/argn_clock_evolution.py',ROOT/'tests/test_argn_clock_evolution.py',
        DOCS/'PROTOCOL.md',ROOT/'artifacts/argn_clock_regression_v1/time_head.pt',
        ROOT/'artifacts/argn_clock_regression_v1/MANIFEST.json',ROOT/'artifacts/argn_clock_regression_v1/AMENDMENT_01.json']
    record=dict(config=CFG,created_utc=datetime.now(timezone.utc).isoformat(),test_events_read=False,
        hashes={str(p):digest(p) for p in paths})
    write(OUT/'MANIFEST.json',record); write(DOCS/'MANIFEST.json',record)
    old=torch.load(ROOT/'artifacts/argn_clock_regression_v1/time_head.pt',map_location='cpu',weights_only=True)
    m=pd.read_parquet(META/'metadata_optimization.parquet'); x,_=progress_features(m,old['fallback'])
    phases=2*m.previous.clip(lower=0).to_numpy()+m.label.to_numpy(); head=ProgressDensity(1,'progress_joint'); stats=[]
    for p in range(4):
        mask=m.event_index.gt(0).to_numpy()&(phases==p)
        values=np.c_[x[mask],np.log1p(m.gap.to_numpy()[mask])]; count=len(values)
        if len(values)>50000: values=values[np.random.default_rng(20261115).choice(len(values),50000,replace=False)]
        fit=GaussianMixture(3,random_state=20261115,n_init=3,reg_covar=1e-4,covariance_type='full').fit(values)
        head.weights[p]=torch.tensor(fit.weights_); head.means[p]=torch.tensor(fit.means_); head.covariances[p]=torch.tensor(fit.covariances_)
        stats.append(dict(phase=p,positions=count,fitted_positions=len(values),converged=bool(fit.converged_),iterations=fit.n_iter_))
    payload={**old,'arm':'progress_joint','state_dict':head.state_dict()}
    save_head(payload,'progress_joint',[0.,0.]); write(OUT/'GMM_COMPLETE.json',dict(phases=stats,statistical_phase_fits=4))
    print('PROGRESS_GMM_FIT',flush=True)
    simulator=ConditionalSimulation('optimization',old['fallback']); density=NumpyDensity(payload); baseline=NumpyDensity(old)
    for arm,source in [('progress_prefix','actual'),('progress_rollout','recursive')]:
        reference=simulator.run(baseline,[0.,0.],source,CFG['fit_noise_seed']); curve=[]
        def evaluate(theta):
            start=time.monotonic(); metrics=simulator.run(density,theta,source,CFG['fit_noise_seed'])
            score=objective(metrics,reference); curve.append(dict(evaluation=len(curve),a=float(theta[0]),b=float(theta[1]),
                objective=score,seconds=time.monotonic()-start,**metrics))
            print('CALIBRATION',arm,len(curve),np.round(theta,5),round(score,6),flush=True)
            return score
        evaluate([0.,0.])
        result=minimize(evaluate,np.zeros(2),method='Powell',bounds=CFG['bounds'],
            options=dict(maxfev=CFG['maxfev']-1,xtol=.001,ftol=.001))
        best=min(curve,key=lambda v:v['objective']); chosen=[best['a'],best['b']]
        save_head(payload,arm,chosen); pd.DataFrame(curve).to_csv(DOCS/f'{arm}_learning_curve.csv',index=False)
        write(OUT/arm/'FIT_COMPLETE.json',dict(selected=best,initial=curve[0],source=source,
            optimizer_success=bool(result.success),optimizer_message=str(result.message),evaluations=len(curve),
            bounded_budget=True,reference=reference,head_sha256=digest(OUT/arm/'time_head.pt')))
    diagnostics=[]
    for split in ['internal_validation']:
        simulator=ConditionalSimulation(split,old['fallback'])
        for gs in CFG['diagnostic_seeds']:
            for arm in ['clock_joint']+CFG['arms']:
                p=old if arm=='clock_joint' else torch.load(OUT/arm/'time_head.pt',map_location='cpu',weights_only=True)
                theta=p['state_dict'].get('correction',torch.zeros(2)).tolist()
                for source in ['actual','recursive']:
                    metrics=simulator.run(NumpyDensity(p),theta,source,gs)
                    diagnostics.append(dict(arm=arm,split=split,source=source,noise_seed=gs,**metrics))
    pd.DataFrame(diagnostics).to_csv(DOCS/'internal_conditional_diagnostics.csv',index=False)
    for arm in CFG['arms']:
        p=OUT/arm/'time_head.pt'; write(DOCS/f'{arm}_head.json',dict(sha256=digest(p),
            correction=torch.load(p,map_location='cpu',weights_only=True)['state_dict']['correction'].tolist()))
    verify(); write(OUT/'FIT_COMPLETE.json',dict(statistical_phase_fits=4,calibration_fits=2,
        native_argn_fits=0,test_events_read=False,heads={arm:digest(OUT/arm/'time_head.pt') for arm in CFG['arms']}))
    print('EVOLUTION_PREPARED',flush=True)


def worker(fs, arm):
    verify(); ready=json.loads((OUT/'FIT_COMPLETE.json').read_text()); path=OUT/arm/'time_head.pt'
    assert digest(path)==ready['heads'][arm]
    assert os.environ['CUDA_VISIBLE_DEVICES'] in CFG['allowed_gpu_uuids']; device=torch.device('cuda:0'); seed(fs)
    dest=OUT/f'worker_{fs}/{arm}'; dest.mkdir(parents=True,exist_ok=False)
    write(dest/'START.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],arm=arm,fit_seed=fs))
    _,codec,frames,metas=inputs(); ws,native=load_parent(fs,device)
    with patch.object(temporal,'time_density_generation',evolution_generation):
        temporal.integration(ws,native,frames['development'],metas['development'],codec,fs,path,dest)
        del frames,metas,native; torch.cuda.empty_cache()
        with patch.object(temporal,'OUT',OUT),patch.object(temporal,'DOCS',DOCS):
            temporal.generate_arm(ws,fs,arm,path,device)
    verify(); write(dest/'COMPLETE.json',dict(generations=4,head_sha256=digest(path),test_events_read=False))
    print('EVOLUTION_WORKER_COMPLETE',fs,arm,flush=True)


def dispatch():
    from dispatch_argn_label_first_control import gpu_snapshot
    verify(); assert (OUT/'FIT_COMPLETE.json').exists() and not (OUT/'LAUNCH.json').exists()
    (OUT/'logs').mkdir(exist_ok=True); pending=[(fs,arm) for arm in CFG['arms'] for fs in CFG['fit_seeds']]
    running={}; finished={}; launched={}; idle={}
    while pending or running:
        for key,job in list(running.items()):
            code=job['process'].poll()
            if code is not None:
                finished[key]=dict(pid=job['process'].pid,exit_code=code); del running[key]
        for gpu in gpu_snapshot():
            uuid=gpu['uuid']; occupied=any(j['gpu']==uuid for j in running.values())
            free=uuid in CFG['allowed_gpu_uuids'] and gpu['memory_mib']<500 and gpu['utilization']<=5 and not gpu['has_compute_process'] and not occupied
            idle[uuid]=idle.get(uuid,0)+1 if free else 0
            if not pending or idle[uuid]<2: continue
            now=next(x for x in gpu_snapshot() if x['uuid']==uuid)
            if now['memory_mib']>=500 or now['utilization']>5 or now['has_compute_process']: continue
            fs,arm=pending.pop(0); key=f'{arm}_{fs}'
            cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--seed',str(fs),'--arm',arm]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (OUT/'logs'/f'{key}.log').open('x') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            running[key]=dict(process=process,gpu=uuid)
            launched[key]=dict(pid=process.pid,gpu=uuid,command=cmd,started_utc=datetime.now(timezone.utc).isoformat())
            write(OUT/'LAUNCH.json',launched); write(DOCS/'LAUNCH.json',launched)
            print('EVOLUTION_LAUNCHED',key,process.pid,gpu['index'],flush=True)
        write(OUT/'STATUS.json',dict(pending=pending,running={k:dict(pid=v['process'].pid,gpu=v['gpu']) for k,v in running.items()},finished=finished))
        if pending or running: time.sleep(5)
    result=dict(success=all(j['exit_code']==0 for j in finished.values()),jobs=finished,completed_utc=datetime.now(timezone.utc).isoformat())
    write(DOCS/'DISPATCH_COMPLETE.json',result); assert result['success'],result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','worker','dispatch'])
    parser.add_argument('--seed',type=int,choices=CFG['fit_seeds']); parser.add_argument('--arm',choices=CFG['arms']); args=parser.parse_args()
    prepare() if args.mode=='prepare' else worker(args.seed,args.arm) if args.mode=='worker' else dispatch()
