"""Frozen-head early-history interventions. No fitting or native data generation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import torch

ROOT=Path(__file__).resolve().parents[3]
CURRENT=ROOT/'artifacts/argn_clock_stratified_loss_v1'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(CURRENT/'local_source')]
import run_stratified as frozen
from adaptive_clock import AdaptiveClock,basis_numpy,LIMIT
from benchmarks.argn_clock_evolution import NumpyDensity,ProgressClock,progress_features
from benchmarks.argn_clock_location import ClockLocation
from benchmarks.argn_clock_regression import ClockRegression
from benchmarks.argn_past_state import DURATION_CAP
from run_argn_clock_evolution import digest,write,META
from run_argn_state_first import SOURCE
from run_adaptive import equal_weights
import report_adaptive as previous_report

OUT=ROOT/'artifacts/argn_early_history_diagnostic_v1'; DOCS=OUT/'report'
HEADS={
 'clock_prefix':ROOT/'artifacts/argn_clock_location_v1/clock_prefix/time_head.pt',
 'coarse_personal':ROOT/'artifacts/argn_clock_adaptive_v1/personal_six/time_head.pt',
 'refined_personal':CURRENT/'personal_six/time_head.pt',
 'refined_position':CURRENT/'position_six/time_head.pt',
 'clock_joint':ROOT/'artifacts/argn_clock_regression_v1/time_head.pt'}
CORE=['clock_prefix','coarse_personal','refined_personal']
MODES=['RR','GG','RG','GR','seed5','reset21','prefix_delta_first5']
BINS=[(1,6),(6,21),(21,50),(6,50),(50,100),(100,200)]
CFG=dict(horizon=200,fit_seeds=frozen.CFG['fit_seeds'],generation_seeds=frozen.CFG['generation_seeds'],
 noise_seeds=[20261202,20261203,20261204,20261205],optimization_noise_count=2,
 generated_path_noise_count=2,modes=MODES,bins=BINS,models=list(HEADS),max_cpu_jobs=4)
CFG=json.loads(json.dumps(CFG))


def locate(name,fs,gs):
    if name=='clock_prefix': folder=ROOT/'artifacts/argn_clock_location_v1';arm='clock_prefix'
    elif name=='clock_joint':folder=ROOT/'artifacts/argn_clock_regression_v1';arm='clock_joint'
    elif name=='coarse_personal':folder=ROOT/'artifacts/argn_clock_adaptive_v1';arm='personal_six'
    else:folder=CURRENT;arm='personal_six' if name=='refined_personal' else 'position_six'
    return folder/f'runs/{arm}_{fs}/generated_validation_{gs}.parquet'


def verify():
    frozen.verify()
    m=json.loads((OUT/'MANIFEST.json').read_text());assert m['config']==CFG
    for p,sha in m['hashes'].items():assert digest(p)==sha,p


class Kernel:
    def __init__(self,name):
        self.name=name;self.payload=torch.load(HEADS[name],map_location='cpu',weights_only=True)
        self.density=NumpyDensity(self.payload);self.fallback=self.payload['fallback']
        self.state=self.payload['state_dict'];self.theta=self.state.get('correction',torch.zeros(2)).double().numpy()
        self.limits=self.state.get('clock_limits',torch.tensor([0.,1.])).double().numpy()
        self.arm='position_six' if name=='refined_position' else 'personal_six'
        self.prefix=torch.load(HEADS['clock_prefix'],map_location='cpu',weights_only=True)['state_dict']['correction'].double().numpy()

    def delta(self,x,valid,prefix=False):
        if self.name=='clock_joint' and not prefix:return np.zeros(len(x))
        if self.name=='clock_prefix' or prefix:
            a,b=self.prefix if prefix else self.theta
            return a+b*np.exp(-np.expm1(x[:,1])/50)
        return np.clip(basis_numpy(self.arm,x,valid,self.limits)@self.theta,-LIMIT,LIMIT)

    def parameters(self,phase,xb,xd,valid,prefix=False):
        p,mu,scale=self.density.parameters(phase,xb)
        delta=self.delta(xd,valid,prefix)*(phase==0)
        return p,mu+delta[:,None],scale,delta


def frame(path):
    d=pd.read_parquet(path,columns=['entity_id','event_index','event_is_fraud','gap'])
    d=d.sort_values(['entity_id','event_index']).reset_index(drop=True)
    d['label']=pd.to_numeric(d.event_is_fraud).astype(int)
    d['gap']=pd.to_numeric(d.gap).astype(float);d.loc[d.event_index.eq(0),'gap']=np.nan
    d['previous']=d.groupby('entity_id').label.shift().fillna(-1).astype(int)
    d['phase_num']=2*d.previous.clip(lower=0)+d.label
    d['raw_clock']=d.groupby('entity_id').gap.transform(lambda x:x.shift().rolling(20,min_periods=5).median())
    return d


class Data:
    def __init__(self,split,fs=None,gs=None):
        m=pd.read_parquet(META/f'metadata_{split}.parquet')
        codes,ids=pd.factorize(m.entity_id,sort=False)
        n=len(ids);h=CFG['horizon'];self.ids=np.asarray(ids);self.shape=(h,n)
        self.noise_shape=(min(int(m.event_index.max())+1,1024),n)
        old=Kernel('clock_joint');self.fallback=old.fallback
        x,valid=progress_features(m,self.fallback)
        keep=m.event_index.lt(h).to_numpy();t=m.event_index.to_numpy()[keep];j=codes[keep]
        self.mask=np.zeros((h,n),bool);self.mask[t,j]=True
        self.real=np.zeros((h,n));self.real[t,j]=m.gap.fillna(0).to_numpy()[keep]
        self.phase=np.zeros((h,n),int);self.phase[t,j]=(2*m.previous.clip(lower=0)+m.label).to_numpy()[keep]
        self.real_phase=self.phase.copy()
        self.actual=np.zeros((h,n,2));self.actual[:,:,0]=np.log1p(self.fallback)
        self.actual[:,:,1]=np.log1p(np.arange(h))[:,None];self.actual[t,j]=x[keep]
        self.valid=np.zeros((h,n));self.valid[t,j]=valid[keep]
        self.raw_clock=np.full((h,n),np.nan);self.raw_clock[t,j]=m.past_gap.to_numpy()[keep]
        self.tag=dict(split=split,path_source='actual_labels',fit_seed=-1,generation_seed=-1)
        if fs is not None:
            generated=frame(locate('refined_personal',fs,gs))
            generated=generated[generated.event_index.lt(h)]
            mapping={eid:i for i,eid in enumerate(ids)}
            gj=generated.entity_id.map(mapping).to_numpy();gt=generated.event_index.to_numpy()
            assert np.isfinite(gj).all();gj=gj.astype(int)
            present=np.zeros((h,n),bool);present[gt,gj]=True
            self.mask &= present
            self.phase=np.zeros((h,n),int);self.phase[gt,gj]=generated.phase_num.to_numpy()
            self.tag.update(path_source='generated_labels',fit_seed=fs,generation_seed=gs)
        self.mask[0]=False
        self.flat_ids=np.broadcast_to(np.arange(n),(h,n))
        codec=ProgressClock(self.fallback);memory=codec.advance(codec.empty(n),np.zeros(n))
        self.snapshots={}
        for t in range(1,h):
            if t in [6,21]:self.snapshots[t]=memory.copy()
            got,v=codec.features(memory);active=self.mask[t]
            np.testing.assert_allclose(got[active],self.actual[t,active],atol=1e-12,rtol=0)
            np.testing.assert_array_equal(v[active],self.valid[t,active])
            memory=codec.advance(memory,self.real[t])

    def noise(self,seed):
        r=np.random.default_rng(seed)
        u=r.random(self.noise_shape);z=r.standard_normal(self.noise_shape)
        return u[:self.shape[0]],z[:self.shape[0]]


def simulate(data,kernel,mode,seed):
    h,n=data.shape;u,z=data.noise(seed);codec=ProgressClock(kernel.fallback)
    memory=codec.advance(codec.empty(n),np.zeros(n))
    out={key:np.zeros((h,n)) for key in ['gap','own_clock','base_clock','delta_clock','delta','valid','clipped']}
    for t in range(1,h):
        if mode=='reset21' and t==21:memory=data.snapshots[21].copy()
        own,ov=codec.features(memory)
        if mode=='seed5' and t==6:
            np.testing.assert_allclose(own[data.mask[t]],data.actual[t,data.mask[t]],atol=1e-12,rtol=0)
        xb=data.actual[t] if mode in ['RR','RG'] else own
        xd=data.actual[t] if mode in ['RR','GR'] else own
        vd=data.valid[t] if mode in ['RR','GR'] else ov
        p,mu,scale,delta=kernel.parameters(data.phase[t],xb,xd,vd,
            prefix=(mode=='prefix_delta_first5' and t<6))
        c=(u[t,:,None]>np.cumsum(p,axis=-1)).sum(-1).clip(max=2);ix=np.arange(n)
        raw=mu[ix,c]+scale[ix,c]*z[t]
        lo,hi=np.log1p(kernel.payload['limits'])
        y=np.floor(np.expm1(np.clip(raw,lo,hi))+1e-8)
        out['gap'][t]=y;out['own_clock'][t]=own[:,0];out['base_clock'][t]=xb[:,0]
        out['delta_clock'][t]=xd[:,0];out['delta'][t]=delta;out['valid'][t]=ov
        out['clipped'][t]=(raw<lo)|(raw>hi)
        feed=data.real[t] if mode=='seed5' and t<6 else y
        memory=codec.advance(memory,feed)
    return out


def distances(real,generated,ids_real,ids_generated):
    x=np.log1p(real);y=np.log1p(generated)
    wa=equal_weights(ids_real);wb=equal_weights(ids_generated)
    return dict(gap_w1=float(wasserstein_distance(x,y)),
        customer_gap_w1=float(wasserstein_distance(x,y,u_weights=wa,v_weights=wb)),
        log_gap_bias=float(y.mean()-x.mean()),real_log_gap_sd=float(x.std()),generated_log_gap_sd=float(y.std()),
        real_gap_median=float(np.median(real)),generated_gap_median=float(np.median(generated)),
        real_gap_q90=float(np.quantile(real,.9)),generated_gap_q90=float(np.quantile(generated,.9)))


def metrics(data,out,model,mode,noise):
    rows=[];t=np.arange(data.shape[0])[:,None]
    lim=Kernel('refined_personal').limits
    for lo,hi in BINS:
        if mode=='seed5' and lo<6:continue
        for phase_group in ['normal','NN']:
            target=(data.phase%2==0)&(data.real_phase%2==0) if phase_group=='normal' else (data.phase==0)&(data.real_phase==0)
            mask=data.mask & target & (t>=lo)&(t<hi)
            if not mask.any():continue
            x=data.actual[:,:,0][mask];own=out['own_clock'][mask];base=out['base_clock'][mask]
            ids=data.flat_ids[mask];real=data.real[mask];gen=out['gap'][mask]
            r=dict(**data.tag,model=model,mode=mode,noise_seed=noise,start=lo,end=hi,group=phase_group,
                events=int(mask.sum()),customers=len(np.unique(ids)),**distances(real,gen,ids,ids),
                own_clock_w1=float(wasserstein_distance(x,own)),base_clock_w1=float(wasserstein_distance(x,base)),
                own_log_clock_bias=float((own-x).mean()),real_log_clock_sd=float(x.std()),
                own_log_clock_sd=float(own.std()),invalid_fraction=float(np.mean(out['valid'][mask]<.5)),
                own_clock_below_endpoint=float(np.mean(own<lim[0])),own_clock_above_endpoint=float(np.mean(own>lim[1])),
                actual_clock_below_endpoint=float(np.mean(x<lim[0])),actual_clock_above_endpoint=float(np.mean(x>lim[1])),
                mean_delta=float(out['delta'][mask].mean()),clipped_fraction=float(out['clipped'][mask].mean()))
            # Compare gap relative to the clock actually available before each output.
            r['personal_gap_w1']=float(wasserstein_distance(np.log1p(real/np.expm1(x)),np.log1p(gen/np.expm1(base))))
            rows.append(r)
    return rows


def job(spec):
    split,fs,gs=spec;data=Data(split,fs,gs)
    noises=CFG['noise_seeds'][:2] if split=='optimization' or fs is not None else CFG['noise_seeds']
    rows=[];trajectories=0;values=0
    for name in HEADS:
        if fs is not None and name not in CORE:continue
        modes=MODES if name in CORE and (fs is None or name=='refined_personal') else ['RR','GG']
        kernel=Kernel(name)
        for ns in noises:
            for mode in modes:
                out=simulate(data,kernel,mode,ns)
                rows+=metrics(data,out,name,mode,ns);trajectories+=1;values+=int(data.mask.sum())
        print('DIAGNOSED',data.tag,name,flush=True)
    tag=split if fs is None else f'generated_path_{fs}_{gs}'
    df=pd.DataFrame(rows);df.to_csv(DOCS/f'{tag}.csv',index=False)
    write(DOCS/f'{tag}_COMPLETE.json',dict(trajectories=trajectories,time_values=values,new_model_fits=0,
        full_transaction_generations=0,test_events_read=False))
    return tag,trajectories,values


def checks():
    rng=np.random.default_rng(2909);x=np.c_[rng.uniform(6,15,500),np.log1p(rng.integers(1,200,500))]
    valid=rng.integers(0,2,500).astype(float);phase=rng.integers(0,4,500)
    errors={}
    for name in HEADS:
        k=Kernel(name)
        if name=='clock_joint':head=ClockRegression(k.payload['dim'],'clock_joint').double()
        elif name=='clock_prefix':head=ClockLocation(k.payload['dim'],'clock_prefix').double()
        else:head=AdaptiveClock(k.payload['dim'],k.arm).double()
        head.load_state_dict(k.state)
        clock=torch.tensor(x if name!='clock_joint' else x[:,0],dtype=torch.float64)
        with torch.no_grad():o=head(torch.zeros(500,k.payload['dim'],dtype=torch.float64),torch.tensor(phase),clock,torch.tensor(valid))
        p,mu,scale,_=k.parameters(phase,x,x,valid)
        error=max(np.max(np.abs(a-b.detach().numpy())) for a,b in zip([p,mu,scale],[o[0].softmax(-1),o[1],o[2]]))
        assert error<1e-11,(name,error);errors[name]=float(error)
        for key in ['weights','means','covariances']:
            torch.testing.assert_close(k.state[key],Kernel('clock_joint').state[key],rtol=0,atol=0)
    data=Data('internal_validation');ns=CFG['noise_seeds'][0]
    for name in ['clock_prefix','refined_position']:
        k=Kernel(name);rr=simulate(data,k,'RR',ns);rg=simulate(data,k,'RG',ns)
        gg=simulate(data,k,'GG',ns);gr=simulate(data,k,'GR',ns)
        np.testing.assert_array_equal(rr['gap'],rg['gap']);np.testing.assert_array_equal(gg['gap'],gr['gap'])
    k=Kernel('refined_personal');rr=simulate(data,k,'RR',ns);gg=simulate(data,k,'GG',ns)
    reset=simulate(data,k,'reset21',ns);warm=simulate(data,k,'seed5',ns)
    np.testing.assert_array_equal(reset['gap'][:21],gg['gap'][:21])
    np.testing.assert_array_equal(rr['gap'][:6],gg['gap'][:6])
    np.testing.assert_allclose(reset['base_clock'][21,data.mask[21]],data.actual[21,data.mask[21],0],atol=1e-12,rtol=0)
    np.testing.assert_allclose(warm['base_clock'][6,data.mask[6]],data.actual[6,data.mask[6],0],atol=1e-12,rtol=0)
    swap=simulate(data,k,'prefix_delta_first5',ns);prefix=simulate(data,Kernel('clock_prefix'),'GG',ns)
    np.testing.assert_array_equal(swap['gap'][:6],prefix['gap'][:6])
    # Actual-prefix output parity against the pre-existing independently vectorized simulator.
    full=frozen.prior.full.ActualPrefix('internal_validation',frozen.payload(),None,ns)
    m=pd.read_parquet(META/'metadata_internal_validation.parquet');non=m.event_index.gt(0)
    xm,vm=progress_features(m[non],k.fallback)
    y=np.floor(np.expm1(np.clip(full.base_z+k.delta(xm,vm)*full.nn,0,np.log1p(k.payload['limits'][1])))+1e-8)
    codes=pd.Categorical(m.loc[non,'entity_id'],categories=data.ids).codes
    tt=m.loc[non,'event_index'].to_numpy();keep=tt<CFG['horizon']
    np.testing.assert_array_equal(rr['gap'][tt[keep],codes[keep]],y[keep])
    # Future values cannot change earlier state; first-event gap is ignored.
    codec=ProgressClock(k.fallback);a=codec.advance(codec.empty(3),[0,10,100000]);b=codec.advance(codec.empty(3),[99,0,1])
    np.testing.assert_array_equal(a,b)
    for step in range(35):a=codec.advance(a,rng.uniform(1,1000,3))
    xa,va=codec.features(a);xb,vb=codec.features(a[[2,0]])
    np.testing.assert_array_equal(xb,xa[[2,0]]);np.testing.assert_array_equal(vb,va[[2,0]])
    changed=m.copy();changed.loc[changed.event_index.ge(30),'gap']*=4
    xa,va=progress_features(m,k.fallback);xb,vb=progress_features(changed,k.fallback)
    np.testing.assert_array_equal(xa[m.event_index.lt(30)],xb[m.event_index.lt(30)])
    write(DOCS/'IMPLEMENTATION_CHECK.json',dict(numpy_torch_parameter_errors=errors,unchanged_gmr_buffers=True,
        recurrent_vectorized_history_parity=True,known_noop_interventions_exact=True,reset_boundaries_exact=True,
        first5_prefix_delta_swap_exact=True,actual_prefix_previous_simulator_exact=True,
        first_gap_ignored=True,future_invariance=True,reorder_invariance=True))


def snapshots():
    real=frame(SOURCE/'prepared/validation.parquet');rows=[]
    for fs in CFG['fit_seeds']:
        for gs in CFG['generation_seeds']:
            base=frame(locate('clock_prefix',fs,gs))
            common=pd.concat([real.groupby('entity_id').size(),base.groupby('entity_id').size()],axis=1).min(axis=1)
            r=real[real.event_index.lt(real.entity_id.map(common))]
            for name in HEADS:
                d=frame(locate(name,fs,gs));cols=['entity_id','event_index','event_is_fraud']
                assert d[cols].equals(base[cols])
                d=d[d.event_index.lt(d.entity_id.map(common))]
                for lo,hi in BINS:
                    a=r[r.event_index.ge(lo)&r.event_index.lt(hi)&r.label.eq(0)]
                    b=d[d.event_index.ge(lo)&d.event_index.lt(hi)&d.label.eq(0)]
                    if lo>=6:a=a[a.raw_clock.gt(0)];b=b[b.raw_clock.gt(0)]
                    row=dict(model=name,fit_seed=fs,generation_seed=gs,start=lo,end=hi,
                        real_events=len(a),generated_events=len(b),real_customers=a.entity_id.nunique(),
                        generated_customers=b.entity_id.nunique(),**distances(a.gap,b.gap,a.entity_id,b.entity_id))
                    if lo>=6:
                        x=np.log1p(a.raw_clock);y=np.log1p(b.raw_clock)
                        row.update(clock_w1=wasserstein_distance(x,y),log_clock_bias=float(y.mean()-x.mean()),
                            real_log_clock_sd=float(x.std(ddof=0)),generated_log_clock_sd=float(y.std(ddof=0)))
                    rows.append(row)
            print('SAVED_GENERATION_AUDITED',fs,gs,flush=True)
    result=pd.DataFrame(rows);result.to_csv(DOCS/'saved_generation_positions.csv',index=False)
    previous=pd.read_csv(CURRENT/'report/horizon_diagnostic.csv')
    for name,oldname in [('clock_prefix','clock_prefix'),('refined_personal','personal_six'),('refined_position','position_six'),('clock_joint','clock_joint')]:
        a=result[result.model.eq(name)&result.start.eq(6)].sort_values(['fit_seed','generation_seed'])
        b=previous[previous.arm.eq(oldname)&previous.start.eq(6)].sort_values(['fit_seed','generation_seed'])
        np.testing.assert_allclose(a.gap_w1,b.gap_w1,rtol=0,atol=1e-12)
    write(DOCS/'SAVED_RESULTS_CHECK.json',dict(previous_6_49_metrics_reproduced=32,outputs_read=40,
        rows_in_generated_files_unchanged=True))


def initialize():
    frozen.verify();assert not (OUT/'MANIFEST.json').exists();checks()
    paths=[Path(__file__),DOCS/'PROTOCOL.md',DOCS/'IMPLEMENTATION_CHECK.json',CURRENT/'MANIFEST.json',
        CURRENT/'report/SUMMARY_METRICS.csv',CURRENT/'report/horizon_diagnostic.csv',
        ROOT/'benchmarks/argn_time_density.py',ROOT/'benchmarks/argn_clock_evolution.py',
        ROOT/'benchmarks/argn_clock_location.py',ROOT/'benchmarks/argn_clock_regression.py',
        ROOT/'artifacts/argn_clock_adaptive_v1/local_source/adaptive_clock.py',
        SOURCE/'prepared/validation.parquet',*HEADS.values()]
    paths += [META/f'metadata_{s}.parquet' for s in ['optimization','internal_validation','development']]
    paths += [locate(m,fs,gs) for m in HEADS for fs in CFG['fit_seeds'] for gs in CFG['generation_seeds']]
    write(OUT/'MANIFEST.json',dict(config=CFG,hashes={str(p):digest(p) for p in paths},
        created_utc=datetime.now(timezone.utc).isoformat(),test_events_read=False,new_model_fits=0))
    verify();print('INITIALIZED',flush=True)


def run():
    verify();assert not (DOCS/'COMPLETE.json').exists();snapshots()
    specs=[(s,None,None) for s in ['optimization','internal_validation','development']]
    specs += [('development',fs,gs) for fs in CFG['fit_seeds'] for gs in CFG['generation_seeds']]
    with ProcessPoolExecutor(max_workers=CFG['max_cpu_jobs']) as pool:results=list(pool.map(job,specs))
    frames=[pd.read_csv(DOCS/f'{name}.csv') for name,_,_ in results]
    all_data=pd.concat(frames,ignore_index=True);all_data.to_csv(DOCS/'all_diagnostics.csv',index=False)
    keys=['split','path_source','model','mode','start','end','group']
    all_data.groupby(keys).mean(numeric_only=True).to_csv(DOCS/'mean_diagnostics.csv')
    verify();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    write(DOCS/'COMPLETE.json',dict(trajectories=sum(x[1] for x in results),time_values=sum(x[2] for x in results),
        new_model_fits=0,full_transaction_generations=0,original_heads_and_outputs_unchanged=True,
        test_events_read=False,pushed=False,completed_utc=datetime.now(timezone.utc).isoformat()))
    print('DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initialize','run']);a=p.parse_args()
    initialize() if a.mode=='initialize' else run()
