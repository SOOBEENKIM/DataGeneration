"""Small frozen-head feedback-gain controls; generated histories throughout fitting."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch

ROOT=Path(__file__).resolve().parents[3]
PRIOR=ROOT/'artifacts/argn_clock_initial_coupling_v1'
sys.path[:0]=[str(PRIOR/'local_source'),str(ROOT),str(ROOT/'scripts')]
import run_initial as prior
from feedback_clock import FeedbackClock,feedback_generation,envelope
from initial_clock import InitialClock
from adaptive_clock import basis_numpy
from run_adaptive import Distance,equal_weights
from run_argn_clock_evolution import digest,write,META

OUT=ROOT/'artifacts/argn_clock_feedback_gain_v1';DOCS=OUT/'report'
ARMS=['correction_gain','regression_gain','joint_gain','output_gain_shift']
PARAMS={'correction_gain':([1.],[(0.,1.5)]),'regression_gain':([0.],[(0.,1.)]),
 'joint_gain':([0.,1.],[(0.,1.),(0.,1.5)]),'output_gain_shift':([1.,0.],[(0.,1.5),(-.35,.35)])}
CFG=json.loads(json.dumps({**prior.CFG,'arms':ARMS,'max_parallel_fits':4,'maxfev':192,
 'gain_parameters':PARAMS,'feedback_knots':[50,125,350],'additional_mean_bound':.35,
 'mid_loss_refinement':[[50,100],[100,200]],'mid_personal_tolerance':.005,'mid_personal_penalty':4.}))


def payload():return torch.load(PRIOR/'init_only/time_head.pt',map_location='cpu',weights_only=True)


def decode(arm,theta):
    if arm=='correction_gain':return np.array([0.,theta[0],0.])
    if arm=='regression_gain':return np.array([theta[0],1.,0.])
    if arm=='joint_gain':return np.array([theta[0],theta[1],0.])
    if arm=='output_gain_shift':return np.array([0.,theta[0],theta[1]])
    raise ValueError(arm)


def verify():
    prior.verify();m=json.loads((OUT/'MANIFEST.json').read_text());assert m['config']==CFG
    for p,h in m['hashes'].items():assert digest(p)==h,p


def build_head(arm,theta):
    old=payload();head=FeedbackClock(old['dim'],arm);head.load_state_dict(old['state_dict'],strict=False)
    head.feedback_parameters.copy_(torch.tensor(decode(arm,theta)))
    for k,v in old['state_dict'].items():torch.testing.assert_close(head.state_dict()[k],v,rtol=0,atol=0)
    return {**old,'arm':arm,'state_dict':head.state_dict()}


class Kernel(prior.Kernel):
    def __init__(self,p):
        super().__init__(p);self.arm=p['arm'];self.feedback=p['state_dict']['feedback_parameters'].double().numpy()

    def parameters(self,phase,x,valid,theta=None):
        p,mu,sd=super().parameters(phase,x,valid)
        a,g,b=self.feedback if theta is None else decode(self.arm,theta)
        regression=(self.density.beta[phase]*(x[:,None,:1]-self.density.means[phase,:,:1])).sum(-1)
        delta=np.clip(basis_numpy('personal_six',x,valid,self.limits)@self.theta,-.35,.35)
        weight=envelope(x)*(phase==0)
        change=np.clip(weight[:,None]*(-a*regression+(g-1)*delta[:,None]+b),-.35,.35)
        return p,mu+change,sd


class Simulation(prior.Simulation):
    def __init__(self,split,noise):
        super().__init__(split,noise)
        m=pd.read_parquet(META/f'metadata_{split}.parquet');m=m[m.event_index.gt(0)]
        parent=self.target.masks['bin1'];weights=np.zeros(len(m));weights[parent]=equal_weights(self.ids[parent])
        self.mid_groups=[];restored=np.zeros(len(m));self.real_gap=np.log1p(m.gap.to_numpy())
        for lo,hi in [(50,100),(100,200)]:
            mask=parent&(self.t>=lo)&(self.t<hi);mass=float(weights[mask].sum()/weights[parent].sum())
            self.mid_groups.append((lo,mask,mass,Distance(self.real_gap[mask],weights[mask])))
            restored[mask]=mass*weights[mask]/weights[mask].sum()
        np.testing.assert_allclose(restored,weights/weights.sum(),rtol=1e-13,atol=1e-16)
        self.mid_mask=self.target.masks['personal']&(self.t>=100)&(self.t<200)
        denominator=self.target.denominator
        real_ratio=np.log1p(m.gap.to_numpy()/np.maximum(denominator,1e-12))
        self.mid_personal=Distance(real_ratio[self.mid_mask],equal_weights(self.ids[self.mid_mask]))

    def sample(self,kernel,theta=None,horizon=None):
        result=super().sample(kernel,theta,horizon);self.last=result;return result

    def run(self,kernel,theta=None):
        metrics=super().run(kernel,theta);gap,_,denominator=self.last
        z=np.log1p(gap[self.t,self.j]);metrics['bin1_refined']=0.
        for lo,mask,mass,distance in self.mid_groups:
            metrics[f'mid_{lo}']=distance(z[mask]);metrics['bin1_refined']+=mass*metrics[f'mid_{lo}']
        ratio=np.log1p(gap[self.t,self.j]/np.maximum(denominator[self.t,self.j],1e-12))
        metrics['mid_personal']=self.mid_personal(ratio[self.mid_mask])
        return metrics


def objective(metrics,reference):
    value=prior.prior.objective({**metrics,'bin1':metrics['bin1_refined']},reference)
    return value+4*max(metrics['mid_personal']-reference['mid_personal']-.005,0.)


def checks():
    old=payload();rng=np.random.default_rng(20261208);errors=[]
    n=rng.choice([1,5,6,49,50,51,99,100,125,199,200,349,350,351,1024],500)
    x=np.c_[rng.uniform(5,14,500),np.log1p(n)];v=np.ones(500);phase=rng.integers(0,4,500)
    original=InitialClock(old['dim'],'init_only').double();original.load_state_dict(old['state_dict'])
    args=(torch.zeros(500,old['dim'],dtype=torch.double),torch.tensor(phase),torch.tensor(x),torch.tensor(v))
    with torch.no_grad():
        ref=original(*args)
        for arm,(initial,bounds) in PARAMS.items():
            for theta in [initial,[(a+b)/2 for a,b in bounds]]:
                p=build_head(arm,theta);model=FeedbackClock(p['dim'],arm).double();model.load_state_dict(p['state_dict'])
                actual=model(*args);want=Kernel(p).parameters(phase,x,v)
                err=max(np.max(abs(a-b.detach().numpy())) for a,b in zip(want,[actual[0].softmax(-1),actual[1],actual[2]]));errors.append(err);assert err<1e-11
                for i in [0,2]:torch.testing.assert_close(actual[i],ref[i],rtol=0,atol=0)
                untouched=(phase!=0)|(n<=50)|(n>=350)
                torch.testing.assert_close(actual[1][untouched],ref[1][untouched],rtol=0,atol=0)
                assert (actual[1]-ref[1]).abs().max()<=.35000000001
                if theta==initial:torch.testing.assert_close(actual[1],ref[1],rtol=0,atol=0)
        for arm,theta,joint in [('correction_gain',[.8],[0.,.8]),('regression_gain',[.3],[.3,1.])]:
            for a,b in zip(Kernel(build_head(arm,theta)).parameters(phase,x,v),Kernel(build_head('joint_gain',joint)).parameters(phase,x,v)):
                np.testing.assert_array_equal(a,b)
    sim=Simulation('internal_validation',CFG['diagnostic_seeds'][0]);identity=Kernel(build_head('joint_gain',[0.,1.]))
    a=sim.sample(identity,horizon=200)[0]
    b=prior.Simulation('internal_validation',CFG['diagnostic_seeds'][0]).sample(prior.Kernel(old),horizon=200)[0]
    np.testing.assert_array_equal(a,b)
    changed=sim.sample(Kernel(build_head('joint_gain',[.3,.8])),horizon=200)[0]
    np.testing.assert_array_equal(a[:51],changed[:51])
    write(DOCS/'IMPLEMENTATION_CHECK.json',dict(numpy_torch_max_error=max(errors),zero_change_exact=True,
        probabilities_and_variances_same_input_exact=True,non_target_law_exact=True,additional_mean_bound=True,
        scalar_ablation_nested_exact=True,first50_outputs_exact=True,previous_replay_identity_exact=True,
        mid_loss_measure_preserved=True))


def initialize():
    prior.verify();assert not (OUT/'MANIFEST.json').exists();checks()
    files=list((OUT/'local_source').glob('*.py'))+[DOCS/'PROTOCOL.md',DOCS/'IMPLEMENTATION_CHECK.json',
        PRIOR/'init_only/time_head.pt',PRIOR/'MANIFEST.json',PRIOR/'report/ARTIFACT_HASHES.json',
        PRIOR/'local_source/run_initial.py',PRIOR/'local_source/initial_clock.py']
    write(OUT/'MANIFEST.json',dict(config=CFG,hashes={str(p):digest(p) for p in files},
        created_utc=datetime.now(timezone.utc).isoformat(),test_events_read=False,local_only=True))
    verify();print('INITIALIZED',flush=True)


def fit(arm):
    verify();initial,bounds=PARAMS[arm];sim=Simulation('optimization',CFG['fit_noise_seed'])
    kernel=Kernel(build_head(arm,initial));ref=sim.run(kernel);curve=[];cache={};start=time.monotonic()
    def evaluate(theta):
        key=tuple(map(float,theta))
        if key not in cache:
            m=sim.run(kernel,theta);cache[key]=dict(objective=objective(m,ref),**m)
        row=dict(evaluation=len(curve),theta=list(key),seconds=time.monotonic()-start,**cache[key]);curve.append(row)
        pd.DataFrame(curve).to_csv(DOCS/f'{arm}_learning_curve.csv',index=False)
        print('FIT',arm,len(curve),np.round(theta,5),round(row['objective'],6),round(row['seconds'],1),flush=True)
        return row['objective']
    evaluate(initial)
    opt=minimize(evaluate,initial,method='Powell',bounds=bounds,options=dict(maxfev=191,xtol=.001,ftol=.001))
    best=min(curve,key=lambda r:r['objective']);p=build_head(arm,best['theta']);dest=OUT/arm;dest.mkdir(exist_ok=True)
    torch.save(p,dest/'time_head.pt')
    write(dest/'FIT_COMPLETE.json',dict(selected=best,initial=curve[0],reference=ref,evaluations=len(curve),
        optimizer_success=bool(opt.success),optimizer_message=str(opt.message),support=sim.target.info,
        decoded_parameters=decode(arm,best['theta']).tolist(),head_sha256=digest(dest/'time_head.pt')))
    verify();return arm


def prepare():
    verify();assert not (OUT/'FIT_COMPLETE.json').exists()
    with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(fit,ARMS))
    rows=[]
    for noise in CFG['diagnostic_seeds']:
        sim=Simulation('internal_validation',noise);ref=sim.run(prior.Kernel(payload()))
        rows.append(dict(arm='init_only',noise_seed=noise,objective=objective(ref,ref),**ref))
        for arm in ARMS:
            p=torch.load(OUT/arm/'time_head.pt',map_location='cpu',weights_only=True);m=sim.run(Kernel(p))
            rows.append(dict(arm=arm,noise_seed=noise,objective=objective(m,ref),**m))
    pd.DataFrame(rows).to_csv(DOCS/'internal_diagnostics.csv',index=False)
    verify();write(OUT/'FIT_COMPLETE.json',dict(calibration_fits=4,new_native_argn_fits=0,
        heads={a:digest(OUT/a/'time_head.pt') for a in ARMS},test_events_read=False))
    print('ALL_FITS_COMPLETE',flush=True)


def worker(fs,arm):
    verify()
    with patch.object(prior.study,'OUT',OUT),patch.object(prior.study,'DOCS',DOCS),patch.object(prior.study,'CFG',CFG), \
         patch.object(prior.study,'verify',verify),patch.object(prior.study,'evolution_generation',feedback_generation):
        prior.study.worker(fs,arm)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initialize','prepare','worker'])
    p.add_argument('--seed',type=int,choices=CFG['fit_seeds']);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    initialize() if a.mode=='initialize' else prepare() if a.mode=='prepare' else worker(a.seed,a.arm)
