"""Refit the preserved selected pipeline for each fresh native representation."""
import argparse,copy,gc,json,os,shutil,sys,time
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from native import ROOT,OUT,SEEDS,BASE,digest,write,seed,now,verify as verify_native
sys.path[:0]=[str(ROOT/'artifacts/argn_init_validation_v1/local_source'),str(ROOT/'artifacts/argn_calibration_specificity_v1/local_source')]
import experiment as prior
import study as specificity
import run_argn_amount_learning as amount
import run_argn_joint_preservation as joint
import run_argn_onset_output as onset
import run_argn_time_density as temporal
from benchmarks.argn_time_density import time_density_generation
from benchmarks.argn_clock_evolution import ProgressClock,progress_features
from validation_head import HistoryClock
from adaptive_clock import basis_numpy,LIMIT
ARMS=['history_only','history_position6','history_calibration_only']
DRAWS=[20261031,20261032]
DOCS=OUT/'report'
META=ROOT/'artifacts/argn_residual_v1'
def verify():
    verify_native();m=json.loads((OUT/'STAGES_MANIFEST.json').read_text())
    for p,h in m['hashes'].items():assert digest(p)==h,p

def folder(arm,fs):
    assert arm=='B_event_label_first' and fs in SEEDS
    return OUT/f'backbones/{fs}'
def load_parent(fs,device):
    with patch.object(amount,'folder',folder):return amount.load_parent(fs,device)
def head_path(arm,fs):return OUT/f'heads/{fs}/{arm}/time_head.pt'
def params():return prior.parameters()
def factory(dim,arm,components=3,hidden=64):
    cls=specificity.ControlledHistory if arm=='history_position6' else HistoryClock
    return cls(dim,arm,components,hidden)
@contextmanager
def generation(*args,**kwargs):
    with patch('benchmarks.argn_time_density.TimeDensity',factory),patch('benchmarks.argn_time_density.PastClock',ProgressClock):
        with time_density_generation(*args,**kwargs) as audit:yield audit

def generation_args(ws,fs):
    f=OUT/f'foundation/{fs}'
    return (ws.tgt_stats.read(),json.loads(joint.OLD.read_text()),f/'amount_head.pt',
      json.loads((joint.PRIOR/'episode_parameters.json').read_text()),
      json.loads((joint.OUT/'hazard_parameters.json').read_text()),f/'count/count_head.pt',True),dict(onset_payload=f/'onset/onset_head.pt')

def save_time(h,arm,fs):
    path=head_path(arm,fs);path.parent.mkdir(parents=True,exist_ok=False)
    torch.save({**params(),'dim':h.dim,'arm':arm,'state_dict':{k:v.cpu() for k,v in h.state_dict().items()}},path)
    return path

def foundation(fs):
    verify();assert (OUT/f'backbones/{fs}/FIT.json').exists()
    dest=OUT/f'foundation/{fs}';dest.mkdir(parents=True,exist_ok=False)
    write(dest/'START.json',dict(pid=os.getpid(),utc=now(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    device=torch.device('cuda:0');torch.set_num_threads(2);seed(fs)
    _,codec,frames,metas=amount.inputs();ws,model=load_parent(fs,device)
    splits=['optimization','internal_validation'];combined={};combined_keys=[];evidence={}
    configs={'category':json.loads((ROOT/'configs/argn_category_amount_v1.json').read_text()),
      'amount_or_numeric_value':copy.deepcopy(amount.CFG),'gap':json.loads((ROOT/'configs/argn_gap_episode_v1.json').read_text())}
    for field in ['category','amount_or_numeric_value','gap']:
        keys=[k for k in model.tgt_cardinalities if k.startswith(codec.prefixes[field]+'__')]
        caches={};checks={};raw_gap=None
        for split in splits:
            cache,checks[split]=amount.build_cache(model,frames[split],metas[split],codec,keys,device)
            if field=='gap':
                m=pd.read_parquet(META/f'metadata_{split}.parquet');meta=metas[split]
                assert np.array_equal(meta.record,m.entity_id) and np.array_equal(meta.event_index,m.event_index)
                assert np.array_equal(meta.label,m.label)
                decoded=codec.numeric({k:cache['targets'][:,j].numpy() for j,k in enumerate(keys)},'gap')
                keep=m.event_index.gt(0).to_numpy();np.testing.assert_array_equal(decoded[keep],m.gap.to_numpy()[keep])
                x,v=progress_features(m,params()['fallback'])
                data=dict(base=cache['base'],phase=torch.tensor((2*m.previous.clip(lower=0)+m.label).to_numpy(),dtype=torch.long),
                    gap=torch.tensor(np.nan_to_num(m.gap.to_numpy()),dtype=torch.float32),clock=torch.tensor(x,dtype=torch.float32),
                    valid=torch.tensor(v,dtype=torch.float32),position=torch.tensor(m.event_index.to_numpy()),keys=keys)
                cp=OUT/f'cache/{fs}/{split}.pt';cp.parent.mkdir(parents=True,exist_ok=True);torch.save(data,cp);del data
                ix=torch.as_tensor(keep);cache={k:v[ix] for k,v in cache.items()}
            caches[split]={k:v.to(device) for k,v in cache.items()};del cache
        run=dest/field;run.mkdir()
        cfg={**amount.CFG,**configs[field],'arms':{'selected':{'kind':'shared','balanced':True}}}
        with patch.object(amount,'CFG',cfg):trained=amount.fit(model,caches,keys,'selected',fs,run)
        payload=torch.load(run/'amount_head.pt',map_location='cpu',weights_only=True)
        assert not set(combined).intersection(payload['state_dict']);combined.update(payload['state_dict']);combined_keys+=keys
        evidence[field]=dict(native_replay_checks=checks,fit=json.loads((run/'FIT_COMPLETE.json').read_text()))
        if field=='amount_or_numeric_value':
            sub={}
            for split in splits:
                m=metas[split];sub[split]={}
                for group,mask in [('fraud',m.label.eq(1)),('onset',m.label.eq(1)&m.previous_label.eq(0))]:
                    ix=torch.as_tensor(np.flatnonzero(mask),device=device)
                    sub[split][group]={k:v[ix] for k,v in caches[split].items()}
            op=dest/'onset';op.mkdir();onset.fit(model,sub,keys,payload,fs,'onset_fit',op);del sub
        del caches,trained,payload;gc.collect();torch.cuda.empty_cache()
        print('FRESH_OUTPUT_FIT',fs,field,flush=True)
    # Original combined payload order was category,gap,amount (state keys are independent).
    ordered=[k for f in ['category','gap','amount_or_numeric_value'] for k in combined_keys if k.startswith(codec.prefixes[f]+'__')]
    torch.save(dict(keys=ordered,kind='shared',state_dict=combined),dest/'amount_head.pt')
    countcache={s:joint.context_cache(model,frames[s],metas[s]) for s in splits}
    cp=dest/'count';cp.mkdir();joint.fit_count(countcache,ws.tgt_stats.read()['seq_len'],fs,False,cp)
    write(dest/'FIT_COMPLETE.json',dict(seed=fs,output_fits=3,onset_fits=1,count_fits=1,evidence=evidence,
      fresh_representation=True,native_sha256=digest(ws.model_tabular_weights_path),utc=now(),test_events_read=False))
    rec={s:dict(sha256=digest(OUT/f'cache/{fs}/{s}.pt')) for s in splits}
    write(OUT/f'cache/{fs}/CACHE_COMPLETE.json',dict(records=rec))
    verify()

def history(fs):
    verify();device=torch.device('cuda:0');torch.set_num_threads(2)
    dest=OUT/f'time_fits/{fs}';dest.mkdir(parents=True,exist_ok=False)
    p=params();m=pd.read_parquet(META/'metadata_optimization.parquet');phase=(2*m.previous.clip(lower=0)+m.label).to_numpy()
    keep=m.event_index.gt(0).to_numpy();z=np.log1p(m.gap.to_numpy());priors={};fits=[]
    for ph in range(4):
        values=z[keep&(phase==ph),None];n=len(values)
        if n>50000:values=values[np.random.default_rng(fs).choice(n,50000,replace=False)]
        mix=GaussianMixture(3,random_state=fs,n_init=3,reg_covar=1e-4,max_iter=100,covariance_type='full').fit(values)
        sd=np.sqrt(mix.covariances_[:,0,0]).clip(.051,4.99)
        priors[str(ph)]=dict(weights=mix.weights_.tolist(),means=mix.means_[:,0].tolist(),raw_scales=np.log(np.expm1(sd-.05)).tolist())
        fits.append(dict(phase=ph,positions=n,fitted_positions=len(values),converged=bool(mix.converged_),iterations=int(mix.n_iter_)))
    write(dest/'PRIORS.json',dict(phases=fits,priors=priors))
    data={};keys=None
    rec=json.loads((OUT/f'cache/{fs}/CACHE_COMPLETE.json').read_text())
    for split in ['optimization','internal_validation']:
        path=OUT/f'cache/{fs}/{split}.pt';assert digest(path)==rec['records'][split]['sha256']
        c=torch.load(path,weights_only=True,map_location='cpu');ok=c['position']>0;keys=c['keys']
        data[split]={k:c[k][ok].to(device) for k in ['base','phase','valid','gap']};data[split]['clock']=c['clock'][ok,0].to(device)
        del c
    with patch.object(temporal,'DOCS',dest):h=temporal.fit(data,keys,{**p,'priors':{'absolute':priors}},fs,'history_only',dest)
    # Initial/GMR buffers inherited by the wrapper are inactive in ALL three arms.
    wrapped=HistoryClock(h.dim,'history_only');wrapped.history_head.load_state_dict({k:v.cpu() for k,v in h.state_dict().items()})
    limits=json.loads((ROOT/'artifacts/argn_clock_stratified_loss_v1/report/CLOCK_ENDPOINTS.json').read_text())['values']
    wrapped.clock_limits.copy_(torch.tensor(limits));wrapped.use_initial.fill_(False);assert not wrapped.correction.any()
    hp=save_time(wrapped,'history_only',fs)
    u,n=prior.common_noise(m,fs);vals=[];tr=data['optimization']
    with torch.no_grad():
        for lo in range(0,len(u),8192):
            ix=slice(lo,lo+8192);lg,mu,sd=h(*(tr[k][ix] for k in ['base','phase','clock','valid']))
            prob=lg.softmax(-1).cpu().numpy();mu=mu.cpu().numpy();sd=sd.cpu().numpy()
            comp=(u[ix,None]>np.cumsum(prob,-1)).sum(-1).clip(max=2);row=np.arange(len(comp));vals.append(mu[row,comp]+sd[row,comp]*n[ix])
    np.save(dest/'CONDITIONAL_Z.npy',np.concatenate(vals))
    write(dest/'COMPLETE.json',dict(neural_fits=1,phase_mixture_fits=4,head_sha256=digest(hp),conditional_z_sha256=digest(dest/'CONDITIONAL_Z.npy'),utc=now()))
    verify()

def calibrate(fs,arm):
    verify();assert arm in ARMS[1:]
    dest=OUT/f'calibration/{fs}/{arm}';dest.mkdir(parents=True,exist_ok=False)
    zp=OUT/f'time_fits/{fs}/CONDITIONAL_Z.npy'
    assert digest(zp)==json.loads((zp.parent/'COMPLETE.json').read_text())['conditional_z_sha256']
    sim=prior.strat.Simulation('optimization',params(),fs+100,'customer');z=np.load(zp)
    assert z.shape==sim.base_z.shape;sim.base_z=z;sim.constant={}
    p=torch.load(head_path('history_only',fs),map_location='cpu',weights_only=True)
    b=(specificity.control_basis(arm,sim.x) if arm=='history_position6' else basis_numpy('personal_six',sim.x,sim.valid,p['state_dict']['clock_limits'].numpy()))
    ref=sim.measure(np.zeros(len(sim.nn)));curve=[];cache={};start=time.monotonic()
    def objective(theta):
        key=tuple(map(float,theta))
        if key not in cache:
            v=sim.measure(np.clip(b@theta,-LIMIT,LIMIT));cache[key]=dict(objective=prior.strat.objective(v,ref),**v)
        row=dict(evaluation=len(curve),theta=list(key),seconds=time.monotonic()-start,**cache[key]);curve.append(row)
        if len(curve)%32==0:
            pd.DataFrame(curve).to_csv(dest/'CURVE.csv',index=False);print('CALIBRATION',fs,arm,len(curve),row['objective'],flush=True)
        return row['objective']
    objective(np.zeros(6));opt=minimize(objective,np.zeros(6),method='Powell',bounds=[(-LIMIT,LIMIT)]*6,options=dict(maxfev=1535,xtol=.001,ftol=.001))
    chosen=min(curve,key=lambda v:v['objective']);pd.DataFrame(curve).to_csv(dest/'CURVE.csv',index=False)
    h=factory(p['dim'],arm);h.load_state_dict(p['state_dict']);h.correction.copy_(torch.tensor(chosen['theta']));h.use_initial.fill_(False)
    hp=save_time(h,arm,fs)
    write(dest/'COMPLETE.json',dict(selected=chosen,initial=curve[0],evaluations=len(curve),optimizer_success=bool(opt.success),optimizer_message=str(opt.message),head_sha256=digest(hp),utc=now()))
    verify()

def generate(fs,arm):
    verify();assert (OUT/'EVALUATION_READY.json').exists();torch.set_num_threads(2)
    from mostlyai.engine import generate as native_generate
    import run_argn_label_first_control as evaluation
    run=OUT/f'runs/{arm}_{fs}';run.mkdir(parents=True,exist_ok=False)
    shutil.copytree(folder('B_event_label_first',fs)/'workspace',run/'workspace',ignore=shutil.ignore_patterns('SyntheticData'))
    ws=amount.Workspace(run/'workspace');before=digest(ws.model_tabular_weights_path);hp=head_path(arm,fs)
    # Native hook/strict-past/state-compaction check on every new time head.
    _,codec,frames,metas=amount.inputs();_,model=load_parent(fs,torch.device('cuda:0'))
    with patch.object(temporal,'generation_args',generation_args),patch.object(temporal,'time_density_generation',generation):
        temporal.integration(ws,model,frames['development'],metas['development'],codec,fs,hp,run)
    del frames,metas,model;gc.collect();torch.cuda.empty_cache()
    context=pd.read_parquet(BASE/'prepared/validation_context.parquet');args,kw=generation_args(ws,fs)
    with generation(*args,**kw,time_payload=hp) as audit:
        for gs in DRAWS:
            seed(gs);start=time.monotonic();i=len(audit);native_generate(ctx_data=context,device='cuda:0',workspace_dir=run/'workspace')
            raw=pd.read_parquet(run/'workspace/SyntheticData');raw.to_parquet(run/f'native_validation_{gs}.parquet',index=False)
            d=raw.rename(columns={'customer_id':'entity_id'}).copy();d['event_index']=d.groupby('entity_id',sort=False).cumcount();d.loc[d.event_index.eq(0),'gap']=np.nan
            path=run/f'generated_validation_{gs}.parquet';d.to_parquet(path,index=False)
            assert digest(ws.model_tabular_weights_path)==before
            write(run/f'GENERATION_{gs}.json',dict(rows=len(d),customers=d.entity_id.nunique(),seconds=time.monotonic()-start,
                 sha256=digest(path),head_sha256=digest(hp),native_sha256=before,sampling_audit=audit[i:],future_labels_or_lengths=False))
            print('GENERATED',fs,arm,gs,len(d),flush=True)
    with patch.object(evaluation,'DOCS',DOCS),patch.object(evaluation,'CONTROL',{**evaluation.CONTROL,'generation_seeds':DRAWS}):evaluation.evaluate(run,arm,fs)
    verify();write(run/'COMPLETE.json',dict(generations=2,utc=now(),test_events_read=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['foundation','history','calibrate','generate']);p.add_argument('--seed',type=int,required=True);p.add_argument('--arm');a=p.parse_args()
    if a.mode in ['foundation','history']:globals()[a.mode](a.seed)
    else:globals()[a.mode](a.seed,a.arm)
