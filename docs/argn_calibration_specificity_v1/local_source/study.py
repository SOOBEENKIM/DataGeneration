"""Frozen history output: global, nested position, equal-capacity position controls."""
import argparse,copy,json,subprocess,sys,time
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch

ROOT=Path(__file__).resolve().parents[3]
PRIOR=ROOT/'artifacts/argn_init_validation_v1'
ABLATION=ROOT/'artifacts/argn_history_component_ablation_v1'
OUT=ROOT/'artifacts/argn_calibration_specificity_v1';DOCS=OUT/'report'
sys.path[:0]=[str(PRIOR/'local_source'),str(ROOT),str(ROOT/'scripts')]
import experiment as prior
from validation_head import HistoryClock
from adaptive_clock import hats_numpy,hats_torch,basis_numpy,POSITION_KNOTS,JOINT_KNOTS,LIMIT
from benchmarks.argn_time_density import time_density_generation
from benchmarks.argn_clock_evolution import ProgressClock
digest,write,now=prior.digest,prior.write,prior.now
SEEDS,PARENTS,DRAWS=prior.SEEDS,prior.PARENTS,prior.DRAWS
NEW_ARMS=['history_global','history_position3','history_position6']
ARMS=['history_only',*NEW_ARMS,'history_calibration_only','init_only']
SIZES=dict(zip(NEW_ARMS,[1,3,6]))
CFG=dict(time_fit_seeds=SEEDS,parents=PARENTS,draws=DRAWS,new_arms=NEW_ARMS,
    coefficient_counts=SIZES,new_calibration_fits=18,new_generations=36,reused_generations=36,
    maxfev=1536,xtol=.001,ftol=.001,coefficient_limit=LIMIT,
    position3_knots=JOINT_KNOTS,position6_knots=POSITION_KNOTS)

def origin(arm):return OUT if arm in NEW_ARMS else ABLATION if arm=='history_calibration_only' else PRIOR
def head_path(arm,rs,parent):
    return origin(arm)/f'heads/{rs}/{parent if arm.startswith("history") else 0}/{arm}/time_head.pt'
def verify():
    m=json.loads((OUT/'MANIFEST.json').read_text());assert m['config']==CFG
    for p,h in m['hashes'].items():assert digest(p)==h,p
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==m['head']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
def preserve():
    old=json.loads((OUT/'PREVIOUS_HASHES.json').read_text())
    for p,h in old.items():assert digest(p)==h,p
    return len(old)
def control_basis(arm,x):
    if arm=='history_global':return np.ones((len(x),1))
    return hats_numpy(x[:,1],np.log1p(JOINT_KNOTS if arm=='history_position3' else POSITION_KNOTS))

class ControlledHistory(HistoryClock):
    def __init__(self,dim,arm,components=3,hidden=64):
        super().__init__(dim,arm,components,hidden)
        self.control_arm=arm
        self.correction=torch.zeros(SIZES[arm])
    def basis(self,clock,valid):
        if self.control_arm=='history_global':return clock.new_ones((len(clock),1))
        knots=JOINT_KNOTS if self.control_arm=='history_position3' else POSITION_KNOTS
        return hats_torch(clock[:,1],clock.new_tensor(knots).log1p())
def factory(dim,arm,components=3,hidden=64):return ControlledHistory(dim,arm,components,hidden)
@contextmanager
def generation(*args,**kwargs):
    with patch('benchmarks.argn_time_density.TimeDensity',factory),patch('benchmarks.argn_time_density.PastClock',ProgressClock):
        with time_density_generation(*args,**kwargs) as audit:yield audit

@torch.no_grad()
def preflight():
    records=[]
    for rs in SEEDS:
        for parent in PARENTS:
            p=torch.load(head_path('history_only',rs,parent),map_location='cpu',weights_only=True)
            original=HistoryClock(p['dim'],p['arm']);original.load_state_dict(p['state_dict']);original.eval()
            c=torch.load(head_path('history_calibration_only',rs,parent),map_location='cpu',weights_only=True)
            assert not c['state_dict']['use_initial']
            for k,v in p['state_dict'].items():
                if k!='correction':assert torch.equal(v,c['state_dict'][k]),k
            torch.manual_seed(rs+parent)
            n=torch.tensor([0,1,5,6,25,50,125,350,750,1024,1500,2500,3106]).repeat_interleave(4)
            phase=torch.arange(4).repeat(len(n)//4)
            clock=torch.stack([torch.linspace(7,14,len(n)),n.float().log1p()],1)
            valid=n.ge(6).float();neural=original.history_head
            base=neural.center+torch.randn(len(n),neural.dim)*neural.scale
            reference=original(base,phase,clock,valid)
            for arm in NEW_ARMS:
                head=factory(p['dim'],arm);state=copy.deepcopy(p['state_dict']);state['correction']=torch.zeros(SIZES[arm])
                head.load_state_dict(state);head.eval()
                for x,y in zip(reference,head(base,phase,clock,valid)):torch.testing.assert_close(x,y,rtol=0,atol=0)
                # NumPy/Torch coordinates agree including boundary extrapolation.
                a=control_basis(arm,clock.double().numpy());b=head.double().basis(clock.double(),valid.double()).numpy()
                np.testing.assert_allclose(a,b,rtol=0,atol=2e-15)
                np.testing.assert_allclose(a.sum(1),1,atol=2e-15);assert np.all(a>=0)
                head.float();head.correction.copy_(torch.linspace(-LIMIT,LIMIT,SIZES[arm]))
                modified=head(base,phase,clock,valid)
                for j in [0,2]:torch.testing.assert_close(reference[j],modified[j],rtol=0,atol=0)
                for x,y in zip(reference,modified):torch.testing.assert_close(x[phase.ne(0)],y[phase.ne(0)],rtol=0,atol=0)
                assert (modified[1]-reference[1]).abs().max()<=LIMIT+1e-6
            limits=original.clock_limits.numpy();x=clock.double().numpy();v=valid.numpy()
            coeff=np.array([-.2,.1,.3])
            np.testing.assert_allclose(basis_numpy('personal_six',x,v,limits)@np.repeat(coeff,2),
                control_basis('history_position3',x)@coeff,atol=2e-15,rtol=0)
            records.append(dict(seed=rs,parent=parent,zero_exact=True,non_normal_laws_exact=True,
                weights_scales_exact=True,nested_personal_position_parity=True,numpy_torch_basis_parity=True))
    write(DOCS/'PREFLIGHT.json',dict(checks=records,new_fit=False))

def prepare():
    assert not (OUT/'MANIFEST.json').exists();prior.verify()
    old=json.loads((ABLATION/'PREVIOUS_HASHES.json').read_text())
    for f in (ABLATION/'runs').glob('*/GENERATION_*.json'):
        r=json.loads(f.read_text());gs=f.stem.split('_')[-1]
        old[str(f.parent/f'generated_validation_{gs}.parquet')]=r['sha256']
    paths=list((ABLATION/'heads').glob('**/time_head.pt'))+list((ABLATION/'local_source').glob('*.py'))
    paths+=list((ABLATION/'report').glob('*.*'))
    paths+=list((ROOT/'artifacts/research_direction_20261001').glob('*.md'))
    old.update({str(p):digest(p) for p in paths if p.is_file()})
    write(OUT/'PREVIOUS_HASHES.json',old);count=preserve();preflight()
    files=list((OUT/'local_source').glob('*.py'))+[DOCS/'PROTOCOL.md',DOCS/'PREFLIGHT.json',OUT/'PREVIOUS_HASHES.json']
    # Include all prior registered transitive implementation/input dependencies.
    frozen=json.loads((PRIOR/'MANIFEST.json').read_text())['hashes']
    for p,h in frozen.items():assert digest(p)==h,p
    files+=list((PRIOR/'heads').glob('**/time_head.pt'))+list((ABLATION/'heads').glob('**/time_head.pt'))
    files+=list((PRIOR/'fits').glob('history_*/CONDITIONAL_Z.npy'))
    files+=list((PRIOR/'fits').glob('history_*/CALIBRATION.json'))
    files+=list((PRIOR/'local_source').glob('*.py'))
    frozen.update({str(p):digest(p) for p in files})
    write(OUT/'MANIFEST.json',dict(config=CFG,created_utc=now(),hashes=frozen,
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),test_events_read=False))
    verify();write(OUT/'READY.json',dict(prepared_utc=now(),previous_hashes_verified=count));print('READY',count,flush=True)

def fit(parent,rs,arm):
    verify();dest=OUT/f'fits/{arm}_{parent}_{rs}';dest.mkdir(parents=True,exist_ok=False)
    previous=PRIOR/f'fits/history_{parent}_{rs}';zpath=previous/'CONDITIONAL_Z.npy'
    assert digest(zpath)==json.loads((previous/'TRAIN_COMPLETE.json').read_text())['conditional_z_sha256']
    p=prior.parameters();p['state_dict']=torch.load(PRIOR/f'heads/{rs}/0/gmr/time_head.pt',map_location='cpu',weights_only=True)['state_dict']
    sim=prior.strat.Simulation('optimization',p,rs+100,'customer')
    z=np.load(zpath);assert z.shape==sim.base_z.shape;sim.base_z=z;sim.constant={}
    ref=sim.measure(np.zeros(len(sim.nn)));limits=json.loads((ROOT/'artifacts/argn_clock_stratified_loss_v1/report/CLOCK_ENDPOINTS.json').read_text())['values']
    old=json.loads((previous/'CALIBRATION.json').read_text())
    expected=sim.measure(np.clip(basis_numpy('personal_six',sim.x,sim.valid,limits)@old['selected']['theta'],-LIMIT,LIMIT))
    expected_value=prior.strat.objective(expected,ref)
    assert abs(expected_value-old['selected']['objective'])<1e-12
    assert abs(prior.strat.objective(ref,ref)-old['initial']['objective'])<1e-12
    b=control_basis(arm,sim.x);curve=[];cache={};started=time.monotonic()
    def objective(theta):
        key=tuple(map(float,theta))
        if key not in cache:
            values=sim.measure(np.clip(b@theta,-LIMIT,LIMIT));cache[key]=dict(objective=prior.strat.objective(values,ref),**values)
        row=dict(evaluation=len(curve),theta=list(key),seconds=time.monotonic()-started,**cache[key]);curve.append(row)
        if len(curve)%32==0:
            pd.DataFrame(curve).to_csv(dest/'CALIBRATION_CURVE.csv',index=False);print('CALIBRATION',arm,parent,rs,len(curve),row['objective'],flush=True)
        return row['objective']
    n=SIZES[arm];objective(np.zeros(n))
    opt=minimize(objective,np.zeros(n),method='Powell',bounds=[(-LIMIT,LIMIT)]*n,
        options=dict(maxfev=CFG['maxfev']-1,xtol=CFG['xtol'],ftol=CFG['ftol']))
    chosen=min(curve,key=lambda v:v['objective'])
    pd.DataFrame(curve).to_csv(dest/'CALIBRATION_CURVE.csv',index=False)
    write(dest/'CALIBRATION.json',dict(selected=chosen,initial=curve[0],support=sim.info,
        optimizer_success=bool(opt.success),optimizer_message=str(opt.message),evaluations=len(curve),
        frozen_personal_objective=expected_value,frozen_personal_objective_parity=True,
        conditional_sample_sha256=digest(zpath),time_fit_seed=rs,parent=parent,arm=arm))
    payload=torch.load(head_path('history_only',rs,parent),map_location='cpu',weights_only=True)
    original=copy.deepcopy(payload['state_dict']);payload['arm']=arm
    payload['state_dict']['correction']=torch.tensor(chosen['theta'],dtype=original['correction'].dtype)
    h=factory(payload['dim'],arm);h.load_state_dict(payload['state_dict']);h.eval()
    for k,v in original.items():
        if k!='correction':assert torch.equal(v,payload['state_dict'][k]),k
    assert not h.use_initial
    path=head_path(arm,rs,parent);path.parent.mkdir(parents=True,exist_ok=False);torch.save(payload,path)
    write(path.parent/'HEAD.json',dict(arm=arm,parent=parent,time_fit_seed=rs,sha256=digest(path),
        source_sha256=digest(head_path('history_only',rs,parent)),frozen_neural_exact=True))
    verify();write(dest/'COMPLETE.json',dict(completed_utc=now(),calibration_fits=1,neural_fits=0,head_sha256=digest(path)))
    print('FIT_COMPLETE',arm,parent,rs,len(curve),chosen['objective'],flush=True)

def generate(parent,rs,arm):
    verify()
    with patch.object(prior,'OUT',OUT),patch.object(prior,'DOCS',DOCS),patch.object(prior,'verify',verify), \
         patch.object(prior,'head_path',head_path),patch.object(prior,'validation_generation',generation):
        prior.generate(parent,rs,arm)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fit','generate'])
    p.add_argument('--parent',type=int,choices=PARENTS);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--arm',choices=NEW_ARMS)
    a=p.parse_args();{'prepare':prepare,'fit':lambda:fit(a.parent,a.seed,a.arm),'generate':lambda:generate(a.parent,a.seed,a.arm)}[a.mode]()
