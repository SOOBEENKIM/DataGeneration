"""Independent refits, paired deletion controls, and native generation. No test reads."""
import argparse,copy,json,os,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/argn_init_validation_v1';DOCS=OUT/'report'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(OUT/'local_source'),
    str(ROOT/'artifacts/argn_clock_initial_coupling_v1/local_source'),
    str(ROOT/'artifacts/argn_clock_stratified_loss_v1/local_source')]
import run_initial as original
import run_stratified as strat
import run_argn_time_density as temporal
from run_argn_amount_learning import digest,write,seed,inputs,load_parent,build_cache
from benchmarks.argn_clock_evolution import progress_features,NumpyDensity
from benchmarks.argn_clock_regression import ClockRegression
from benchmarks.argn_time_density import TimeDensity
from adaptive_clock import basis_numpy,LIMIT
from initial_clock import InitialClock
from validation_head import HistoryClock,factory,validation_generation

META=ROOT/'artifacts/argn_residual_v1'
SEEDS=[73001,73002,73003];PARENTS=[20260930,20261001];DRAWS=[20261011,20261012]
ARMS=['init_only','no_initial','no_calibration','gmr','phase_gmm','history_only','history_init_cal']
CFG=dict(time_fit_seeds=SEEDS,parents=PARENTS,draws=DRAWS,arms=ARMS,
    maxfev=1536,mixture_n_init=3,mixture_cap=50000,gmr_max_iter=100,initial_max_iter=300,
    history_config=temporal.CFG,new_native_argn_fits=0,new_full_generations=84)


def now():return datetime.now(timezone.utc).isoformat()
def verify():
    m=json.loads((OUT/'MANIFEST.json').read_text())
    assert m['config']==json.loads(json.dumps(CFG))
    for p,h in m['hashes'].items():assert digest(p)==h,p
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==m['head']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()


def parameters():
    return torch.load(ROOT/'artifacts/argn_clock_regression_v1/time_head.pt',map_location='cpu',weights_only=True)


def save_head(head,arm,rs,parent=0):
    path=OUT/f'heads/{rs}/{parent}/{arm}';path.mkdir(parents=True,exist_ok=True)
    p={**parameters(),'dim':head.dim,'arm':arm,'state_dict':{k:v.cpu() for k,v in head.state_dict().items()}}
    torch.save(p,path/'time_head.pt')
    write(path/'HEAD.json',dict(arm=arm,time_fit_seed=rs,parent=parent,sha256=digest(path/'time_head.pt')))
    return path/'time_head.pt'


def load_cache(parent,split):
    path=(ROOT/'artifacts/argn_joint_clock_density_v1/cache'/f'{split}.pt' if parent==PARENTS[0]
          else OUT/f'cache/{parent}/{split}.pt')
    ledger=(ROOT/'artifacts/argn_joint_clock_density_v1/CACHE_COMPLETE.json' if parent==PARENTS[0]
            else OUT/f'cache/{parent}/CACHE_COMPLETE.json')
    record=json.loads(ledger.read_text())['records'][split]
    assert digest(path)==record['sha256']
    return torch.load(path,map_location='cpu',weights_only=True)


def make_cache(parent):
    verify();assert parent==PARENTS[1] and torch.cuda.is_available()
    dest=OUT/f'cache/{parent}';dest.mkdir(parents=True,exist_ok=False)
    _,codec,frames,metas=inputs();ws,model=load_parent(parent,torch.device('cuda:0'));records={}
    keys=[k for k in model.tgt_cardinalities if k.startswith(codec.prefixes['gap']+'__')]
    p=parameters()
    for split in ['optimization','internal_validation']:
        meta=metas[split];m=pd.read_parquet(META/f'metadata_{split}.parquet')
        assert np.array_equal(meta.record,m.entity_id) and np.array_equal(meta.event_index,m.event_index)
        assert np.array_equal(meta.label,m.label)
        cache,checks=build_cache(model,frames[split],meta,codec,keys,model.device)
        y=codec.numeric({k:cache['targets'][:,j].numpy() for j,k in enumerate(keys)},'gap')
        keep=m.event_index.gt(0).to_numpy();np.testing.assert_array_equal(y[keep],m.gap.to_numpy()[keep])
        x,v=progress_features(m,p['fallback'])
        data=dict(base=cache['base'],phase=torch.tensor((2*m.previous.clip(lower=0)+m.label).to_numpy(),dtype=torch.long),
            gap=torch.tensor(np.nan_to_num(m.gap.to_numpy()),dtype=torch.float32),
            clock=torch.tensor(x,dtype=torch.float32),valid=torch.tensor(v,dtype=torch.float32),
            position=torch.tensor(m.event_index.to_numpy()),keys=keys)
        path=dest/f'{split}.pt';torch.save(data,path)
        records[split]=dict(rows=len(m),customers=m.entity_id.nunique(),native_logits_checked=checks,
            mapping_exact=True,sha256=digest(path))
        print('CACHE_READY',parent,split,len(m),flush=True)
    write(dest/'CACHE_COMPLETE.json',dict(records=records,parent_seed=parent,test_events_read=False))


def initial_parameters(head,mix):
    head.initial_weights.copy_(torch.tensor(mix['weights']))
    head.initial_location.copy_(torch.tensor(mix['location']))
    head.initial_scale.copy_(torch.tensor(mix['scale']))
    limits=json.loads((ROOT/'artifacts/argn_clock_stratified_loss_v1/report/CLOCK_ENDPOINTS.json').read_text())['values']
    head.clock_limits.copy_(torch.tensor(limits))
    return head


def calibrate(payload,rs,dest,base_z=None):
    # Full-length, actual-prefix objective from the preserved selected method.
    sim=strat.Simulation('optimization',payload,rs+100,'customer')
    if base_z is not None:
        assert base_z.shape==sim.base_z.shape
        sim.base_z=base_z;sim.constant={}
    limits=json.loads((ROOT/'artifacts/argn_clock_stratified_loss_v1/report/CLOCK_ENDPOINTS.json').read_text())['values']
    basis=basis_numpy('personal_six',sim.x,sim.valid,limits)
    reference=sim.measure(np.zeros(len(sim.nn)));curve=[];cache={};started=time.monotonic()
    def objective(theta):
        key=tuple(map(float,theta))
        if key not in cache:
            metrics=sim.measure(np.clip(basis@theta,-LIMIT,LIMIT))
            cache[key]=dict(objective=strat.objective(metrics,reference),**metrics)
        row=dict(evaluation=len(curve),theta=list(key),seconds=time.monotonic()-started,**cache[key]);curve.append(row)
        if len(curve)%32==0:
            pd.DataFrame(curve).to_csv(dest/'CALIBRATION_CURVE.csv',index=False)
            print('CALIBRATION',dest.name,rs,len(curve),row['objective'],flush=True)
        return row['objective']
    objective(np.zeros(6))
    fit=minimize(objective,np.zeros(6),method='Powell',bounds=[(-LIMIT,LIMIT)]*6,
        options=dict(maxfev=CFG['maxfev']-1,xtol=.001,ftol=.001))
    selected=min(curve,key=lambda r:r['objective'])
    pd.DataFrame(curve).to_csv(dest/'CALIBRATION_CURVE.csv',index=False)
    write(dest/'CALIBRATION.json',dict(selected=selected,initial=curve[0],support=sim.info,
        optimizer_success=bool(fit.success),optimizer_message=str(fit.message),evaluations=len(curve)))
    return selected['theta']


def fit_statistics(rs):
    verify();dest=OUT/f'fits/statistics_{rs}';dest.mkdir(parents=True,exist_ok=False)
    p=parameters();m=pd.read_parquet(META/'metadata_optimization.parquet');x,v=progress_features(m,p['fallback'])
    phase=(2*m.previous.clip(lower=0)+m.label).to_numpy();keep=m.event_index.gt(0).to_numpy();z=np.log1p(m.gap.to_numpy())
    gmr=ClockRegression(1,'clock_joint');phase_head=ClockRegression(1,'phase_gmm');records=[];priors={}
    for ph in range(4):
        mask=keep&(phase==ph);values=np.c_[x[mask,0],z[mask]];n=len(values)
        if n>CFG['mixture_cap']:values=values[np.random.default_rng(rs).choice(n,CFG['mixture_cap'],replace=False)]
        for kind,obs in [('gmr',values),('phase',values[:,1:])]:
            fit=GaussianMixture(3,random_state=rs,n_init=3,reg_covar=1e-4,max_iter=100,covariance_type='full').fit(obs)
            records.append(dict(kind=kind,phase=ph,positions=n,fitted_positions=len(obs),converged=bool(fit.converged_),iterations=fit.n_iter_))
            if kind=='gmr':
                gmr.weights[ph]=torch.tensor(fit.weights_);gmr.means[ph]=torch.tensor(fit.means_);gmr.covariances[ph]=torch.tensor(fit.covariances_)
            else:
                phase_head.weights[ph]=torch.tensor(fit.weights_)
                phase_head.means[ph,:,0]=0;phase_head.means[ph,:,1]=torch.tensor(fit.means_[:,0])
                phase_head.covariances[ph]=torch.eye(2).repeat(3,1,1)
                phase_head.covariances[ph,:,1,1]=torch.tensor(fit.covariances_[:,0,0])
                sd=np.sqrt(fit.covariances_[:,0,0]).clip(.051,4.99)
                priors[str(ph)]=dict(weights=fit.weights_.tolist(),means=fit.means_[:,0].tolist(),raw_scales=np.log(np.expm1(sd-.05)).tolist())
    mask=keep&(phase==0)&m.event_index.le(5).to_numpy()
    init=GaussianMixture(3,random_state=rs,n_init=3,reg_covar=1e-4,max_iter=300).fit(z[mask,None])
    mix=dict(weights=init.weights_.tolist(),location=init.means_[:,0].tolist(),scale=np.sqrt(init.covariances_[:,0,0]).tolist(),
        rows=int(mask.sum()),customers=m.loc[mask,'entity_id'].nunique(),converged=bool(init.converged_))
    write(dest/'MIXTURES.json',dict(phases=records,initial=mix,priors=priors))
    fresh={**p,'state_dict':gmr.state_dict()};theta=calibrate(fresh,rs,dest)
    hashes={}
    for arm in ARMS[:5]:
        head=InitialClock(1,arm);head.load_state_dict((phase_head if arm=='phase_gmm' else gmr).state_dict(),strict=False)
        initial_parameters(head,mix)
        head.use_initial.fill_(arm in ['init_only','no_calibration'])
        if arm in ['init_only','no_initial']:head.correction.copy_(torch.tensor(theta))
        path=save_head(head,arm,rs);hashes[arm]=digest(path)
    write(dest/'COMPLETE.json',dict(time_fit_seed=rs,mixture_fits=9,calibration_fits=1,head_hashes=hashes,finished_utc=now()))
    verify();print('STATISTICS_COMPLETE',rs,flush=True)


def common_noise(metadata,rs):
    codes=pd.factorize(metadata.entity_id,sort=False)[0];t=metadata.event_index.to_numpy()
    h,n=int(t.max())+1,int(codes.max())+1; rng=np.random.default_rng(rs+100);shape=(min(h,1024),n)
    u,z=rng.random(shape),rng.standard_normal(shape)
    if h>1024:u=np.r_[u,rng.random((h-1024,n))];z=np.r_[z,rng.standard_normal((h-1024,n))]
    keep=t>0
    return u[t[keep],codes[keep]],z[t[keep],codes[keep]]


def fit_history(parent,rs):
    verify();assert torch.cuda.is_available();device=torch.device('cuda:0')
    dest=OUT/f'fits/history_{parent}_{rs}';dest.mkdir(parents=True,exist_ok=False)
    stats=json.loads((OUT/f'fits/statistics_{rs}/MIXTURES.json').read_text())
    params={**parameters(),'priors':{'absolute':stats['priors']}}
    data={};keys=None
    for split in ['optimization','internal_validation']:
        c=load_cache(parent,split);keep=c['position']>0;keys=c['keys']
        data[split]={k:c[k][keep].to(device) for k in ['base','phase','valid','gap']}
        data[split]['clock']=c['clock'][keep,0].to(device)
    # Explicit new initialization/mini-batch seed, parent is a frozen representation.
    with patch.object(temporal,'DOCS',DOCS/f'fit_details/{parent}'):
        head=temporal.fit(data,keys,params,rs,'history_only',dest)
    meta=pd.read_parquet(META/'metadata_optimization.parquet');u,z=common_noise(meta,rs);values=[]
    tr=data['optimization']
    with torch.no_grad():
        for lo in range(0,len(u),8192):
            ix=slice(lo,lo+8192);lg,mu,sd=head(*(tr[k][ix] for k in ['base','phase','clock','valid']))
            prob=lg.softmax(-1).cpu().numpy();mu=mu.cpu().numpy();sd=sd.cpu().numpy()
            comp=(u[ix,None]>np.cumsum(prob,-1)).sum(-1).clip(max=2);row=np.arange(len(comp))
            values.append(mu[row,comp]+sd[row,comp]*z[ix])
    wrapped=initial_parameters(HistoryClock(head.dim,'history_only'),stats['initial'])
    wrapped.history_head.load_state_dict({k:v.cpu() for k,v in head.state_dict().items()})
    save_head(wrapped,'history_only',rs,parent)
    np.save(dest/'CONDITIONAL_Z.npy',np.concatenate(values))
    write(dest/'TRAIN_COMPLETE.json',dict(parent=parent,time_fit_seed=rs,neural_fits=1,
        conditional_z_sha256=digest(dest/'CONDITIONAL_Z.npy'),finished_utc=now()))
    verify();print('HISTORY_TRAIN_COMPLETE',parent,rs,flush=True)


def calibrate_history(parent,rs):
    verify();dest=OUT/f'fits/history_{parent}_{rs}'
    rec=json.loads((dest/'TRAIN_COMPLETE.json').read_text())
    assert digest(dest/'CONDITIONAL_Z.npy')==rec['conditional_z_sha256']
    p=torch.load(head_path('history_only',rs,parent),map_location='cpu',weights_only=True)
    wrapped=factory(p['dim'],p['arm']);wrapped.load_state_dict(p['state_dict'])
    fresh={**parameters(),'state_dict':torch.load(OUT/f'heads/{rs}/0/gmr/time_head.pt',map_location='cpu',weights_only=True)['state_dict']}
    theta=calibrate(fresh,rs,dest,np.load(dest/'CONDITIONAL_Z.npy'))
    wrapped.correction.copy_(torch.tensor(theta));wrapped.use_initial.fill_(True)
    save_head(wrapped,'history_init_cal',rs,parent)
    write(dest/'COMPLETE.json',dict(parent=parent,time_fit_seed=rs,neural_fits=1,calibration_fits=1,finished_utc=now()))
    verify();print('HISTORY_COMPLETE',parent,rs,flush=True)


def head_path(arm,rs,parent):return OUT/f'heads/{rs}/{parent if arm.startswith("history") else 0}/{arm}/time_head.pt'


def generate(parent,rs,arm):
    verify();path=head_path(arm,rs,parent);p=path.parent
    assert digest(path)==json.loads((p/'HEAD.json').read_text())['sha256']
    dest=OUT/f'runs/{arm}_{rs}_{parent}';dest.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda:0');assert torch.cuda.is_available()
    _,codec,frames,metas=inputs();ws,native=load_parent(parent,device)
    tag=f'{arm}_{rs}'
    with patch.object(temporal,'time_density_generation',validation_generation):
        temporal.integration(ws,native,frames['development'],metas['development'],codec,parent,path,dest)
        del frames,metas,native;torch.cuda.empty_cache()
        # Existing generator uses runs/<tag>_<parent>; the registered target already
        # exists for routing audit, so preserve it separately before generation.
        checkdir=OUT/f'routing/{arm}_{rs}_{parent}';checkdir.parent.mkdir(exist_ok=True)
        dest.rename(checkdir)
        with patch.object(temporal,'OUT',OUT),patch.object(temporal,'DOCS',DOCS), \
             patch.object(temporal,'CFG',{**temporal.CFG,'generation_seeds':DRAWS}):
            temporal.generate_arm(ws,parent,tag,path,device)
    # Legacy helper's COMPLETE count is fixed at4; correct it explicitly in NEW
    # output only. Actual files/records and this wrapper define the registered2.
    write(dest/'COMPLETE.json',dict(generations=len(DRAWS),parent=parent,time_fit_seed=rs,arm=arm,
        exact_labels_lengths=True,test_events_read=False,finished_utc=now()))
    verify();print('GENERATION_COMPLETE',arm,rs,parent,flush=True)


def initialize():
    assert not (OUT/'MANIFEST.json').exists()
    old=json.loads((ROOT/'artifacts/research_process_audit_20260930/VERIFIED_HASHES.json').read_text())
    for p,h in old.items():assert digest(p)==h,p
    files=list((OUT/'local_source').glob('*.py'))+[DOCS/'PROTOCOL.md',
        ROOT/'artifacts/research_process_audit_20260930/VERIFIED_HASHES.json',
        ROOT/'artifacts/argn_clock_stratified_loss_v1/report/CLOCK_ENDPOINTS.json',
        META/'metadata_optimization.parquet',META/'metadata_internal_validation.parquet']
    files+=list((ROOT/'benchmarks').glob('argn_*.py'))
    files+=[ROOT/'scripts'/n for n in ['run_argn_time_density.py','run_argn_amount_learning.py','run_argn_clock_evolution.py']]
    files+=[ROOT/f'artifacts/{folder}/local_source/{name}' for folder,name in [
        ('argn_clock_initial_coupling_v1','initial_clock.py'),('argn_clock_adaptive_v1','adaptive_clock.py'),
        ('argn_clock_adaptive_v1','run_adaptive.py'),('argn_clock_stratified_loss_v1','run_stratified.py')]]
    write(OUT/'MANIFEST.json',dict(created_utc=now(),config=CFG,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        prior_hashes_verified=len(old),hashes={str(p):digest(p) for p in files},test_events_read=False))
    verify();print('INITIALIZED',len(old),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['initialize','cache','statistics','history','calibrate_history','generate'])
    a.add_argument('--parent',type=int,choices=PARENTS);a.add_argument('--seed',type=int,choices=SEEDS);a.add_argument('--arm',choices=ARMS);v=a.parse_args()
    {'initialize':initialize,'cache':lambda:make_cache(v.parent),'statistics':lambda:fit_statistics(v.seed),
     'history':lambda:fit_history(v.parent,v.seed),'calibrate_history':lambda:calibrate_history(v.parent,v.seed),
     'generate':lambda:generate(v.parent,v.seed,v.arm)}[v.mode]()
