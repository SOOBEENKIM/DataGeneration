"""Registered 2x2 initial-output / recursive-handoff study. Never load test events."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
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

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT/'artifacts/argn_clock_stratified_loss_v1'
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(PRIOR/'local_source')]
import run_stratified as prior
import run_argn_clock_evolution as study
from adaptive_clock import AdaptiveClock, basis_numpy, LIMIT
from initial_clock import InitialClock, handoff_basis, initial_generation
from benchmarks.argn_clock_evolution import NumpyDensity, ProgressClock, progress_features
from run_adaptive import Distance, equal_weights
from run_argn_clock_evolution import digest, write, META

OUT = ROOT/'artifacts/argn_clock_initial_coupling_v1'; DOCS = OUT/'report'
ARMS = ['init_only', 'handoff_only', 'init_handoff']
CFG = {**prior.CFG, 'arms': ARMS, 'maxfev': 192, 'max_parallel_fits': 2,
       'mixture_seed': 20261206, 'mixture_components': 3, 'mixture_n_init': 3,
       'mixture_reg_covar': 1e-4, 'mixture_max_iter': 300,
       'handoff_knots': [6,50,200], 'handoff_bound': .35,
       'maximum_own_workers_per_gpu': 2,
       'fit_condition': 'actual_labels_lengths_and_own_generated_time_history',
       'initial_scope': 'phase0_and_positions1to5', 'fit_support': 'full_sequence'}
CFG = json.loads(json.dumps(CFG))


def current():
    return torch.load(PRIOR/'personal_six/time_head.pt', map_location='cpu', weights_only=True)


def verify():
    prior.verify()
    m = json.loads((OUT/'MANIFEST.json').read_text())
    assert m['config'] == CFG
    for p, h in m['hashes'].items():
        assert digest(p) == h, p
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()


def build_head(arm, mixture, theta):
    p = current(); head = InitialClock(p['dim'], arm)
    head.load_state_dict(p['state_dict'], strict=False)
    head.initial_weights.copy_(torch.tensor(mixture['weights']))
    head.initial_location.copy_(torch.tensor(mixture['location']))
    head.initial_scale.copy_(torch.tensor(mixture['scale']))
    head.use_initial.fill_(arm in ['init_only','init_handoff'])
    head.handoff.copy_(torch.tensor(theta))
    for k, v in p['state_dict'].items():
        torch.testing.assert_close(head.state_dict()[k],v,rtol=0,atol=0)
    return {**p,'arm':arm,'state_dict':head.state_dict()}


class Kernel:
    def __init__(self, payload):
        self.p = payload; self.density = NumpyDensity(payload)
        s = payload['state_dict']; self.theta = s['correction'].double().numpy()
        self.limits = s['clock_limits'].double().numpy()
        self.init = bool(s['use_initial'])
        self.iw = s['initial_weights'].double().numpy(); self.iw /= self.iw.sum()
        self.im = s['initial_location'].double().numpy()
        self.isd = s['initial_scale'].double().numpy()
        self.handoff = s['handoff'].double().numpy()

    def parameters(self, phase, x, valid, theta=None):
        p, mu, sd = self.density.parameters(phase, x)
        handoff = self.handoff if theta is None else np.asarray(theta)
        delta = np.clip(basis_numpy('personal_six',x,valid,self.limits)@self.theta + handoff_basis(x)@handoff,-LIMIT,LIMIT)
        mu += (delta*(phase==0))[:,None]
        n = np.rint(np.expm1(x[:,1])); initial = self.init & (phase==0) & (n>=1) & (n<=5)
        p[initial] = self.iw; mu[initial] = self.im; sd = sd.copy(); sd[initial] = self.isd
        return p, mu, sd


class Simulation:
    def __init__(self, split, noise):
        p = current()
        self.target = prior.Simulation(split, prior.payload(), noise, 'customer')
        m = pd.read_parquet(META/f'metadata_{split}.parquet')
        codes, ids = pd.factorize(m.entity_id, sort=False)
        self.shape = (int(m.event_index.max())+1, len(ids)); h,n = self.shape
        t = m.event_index.to_numpy(); keep = t>0
        self.t = t[keep]; self.j = codes[keep]; self.ids = m.entity_id.to_numpy()[keep]
        self.phase = np.zeros(self.shape,dtype=int)
        self.phase[t,codes] = (2*m.previous.clip(lower=0)+m.label).to_numpy()
        self.fallback = p['fallback']; self.gap_limits = p['limits']
        self.x, self.valid = progress_features(m,p['fallback'])
        self.real_clock = self.x[keep,0]
        self.clock_distance = Distance(self.real_clock[self.target.masks['personal']], equal_weights(self.ids[self.target.masks['personal']]))
        rng = np.random.default_rng(noise); prefix = (min(h,1024),n)
        self.u, self.z = rng.random(prefix), rng.standard_normal(prefix)
        if h>1024:
            extra = (h-1024,n)
            self.u = np.concatenate([self.u,rng.random(extra)])
            self.z = np.concatenate([self.z,rng.standard_normal(extra)])

    def sample(self, kernel, theta=None, horizon=None):
        h,n = self.shape; h = h if horizon is None else min(h,horizon)
        codec = ProgressClock(self.fallback)
        memory = codec.advance(codec.empty(n),np.zeros(n))
        raw = np.full((n,20),np.nan); gaps = np.zeros((h,n)); clocks = np.zeros((h,n))
        denominator = np.full((h,n),self.fallback)
        for t in range(1,h):
            x,valid = codec.features(memory); clocks[t] = x[:,0]
            if t>=6:
                denominator[t] = np.median(raw[:,:min(t-1,20)],axis=1)
            probs,loc,scale = kernel.parameters(self.phase[t],x,valid,theta)
            component = (self.u[t,:,None]>np.cumsum(probs,axis=-1)).sum(-1).clip(max=2)
            ix = np.arange(n); z = loc[ix,component]+scale[ix,component]*self.z[t]
            gaps[t] = np.floor(np.expm1(np.clip(z,*np.log1p(self.gap_limits)))+1e-8)
            memory = codec.advance(memory,gaps[t]); raw[:,(t-1)%20] = gaps[t]
        return gaps,clocks,denominator

    def run(self,kernel,theta=None):
        gaps,clocks,denominator = self.sample(kernel,theta)
        z = np.log1p(gaps[self.t,self.j]); rel = np.log1p(gaps[self.t,self.j]/np.maximum(denominator[self.t,self.j],1e-12))
        metrics = {}
        for key,mask in self.target.masks.items():
            values = rel if key in ['personal','onset'] else z
            metrics[key] = self.target.distances[key](values[mask])
        mask = self.target.masks['personal']
        metrics['clock'] = self.clock_distance(clocks[self.t,self.j][mask])
        metrics['bin2_refined'] = 0.
        for lo,hi,mask,mass,distance in self.target.groups:
            metrics[f'fine_{lo}'] = distance(z[mask])
            metrics['bin2_refined'] += mass*metrics[f'fine_{lo}']
        for lo,hi in [(1,6),(6,50),(100,200)]:
            mask = self.target.nn & (self.t>=lo) & (self.t<hi)
            # Explicit diagnostics use the same row weights; not extra objective terms.
            actual = self.actual_gap if hasattr(self,'actual_gap') else None
            if actual is None:
                m = pd.read_parquet(META/f'metadata_{self.target.info["split"]}.parquet')
                self.actual_gap = np.log1p(m.loc[m.event_index.gt(0),'gap'].to_numpy())
            metrics[f'position_{lo}_{hi}'] = Distance(self.actual_gap[mask],equal_weights(self.ids[mask]))(z[mask])
        return metrics


def checks():
    rng = np.random.default_rng(21092); p = current()
    mix = dict(weights=[.2,.5,.3],location=[8.,10.,12.],scale=[.3,.5,.8])
    n = rng.choice([0,1,5,6,20,25,49,50,199,200,201,1024],600)
    x = np.c_[rng.uniform(5,14,600),np.log1p(n)]; valid = (n>=6).astype(float)
    phase = rng.integers(0,4,len(n)); errors = []
    baseline = AdaptiveClock(p['dim'],'personal_six').double(); baseline.load_state_dict(p['state_dict'])
    with torch.no_grad():
        original = baseline(torch.zeros(len(n),p['dim'],dtype=torch.double),torch.tensor(phase),torch.tensor(x),torch.tensor(valid))
        for arm in ARMS+['identity']:
            payload = build_head(arm,mix,[0.,0.] if arm in ['init_only','identity'] else [-.17,.11])
            model = InitialClock(p['dim'],arm).double(); model.load_state_dict(payload['state_dict'])
            got = model(torch.zeros(len(n),p['dim'],dtype=torch.double),torch.tensor(phase),torch.tensor(x),torch.tensor(valid))
            want = Kernel(payload).parameters(phase,x,valid)
            errors.append(max(np.max(abs(a.detach().numpy()-b)) for a,b in zip([got[0].softmax(-1),got[1],got[2]],want)))
            assert errors[-1]<1e-11,(arm,errors[-1])
            untouched = (phase!=0)|(n>=200)|(n==0)
            if arm=='identity': untouched = np.ones(len(n),bool)
            for a,b in zip(got,original): torch.testing.assert_close(a[untouched],b[untouched],rtol=0,atol=0)
            assert np.allclose(want[0].sum(-1),1) and (want[2]>0).all()
    # Reproduce the frozen diagnostic's own-clock path on the same 1024-prefix RNG.
    sys.path.insert(0,str(ROOT/'artifacts/argn_early_history_diagnostic_v1/local_source'))
    import diagnose as diagnostic
    sim = Simulation('internal_validation',CFG['diagnostic_seeds'][0])
    our = sim.sample(Kernel(build_head('identity',mix,[0.,0.])),horizon=200)[0]
    old = diagnostic.simulate(diagnostic.Data('internal_validation'),diagnostic.Kernel('refined_personal'),'GG',CFG['diagnostic_seeds'][0])['gap']
    np.testing.assert_array_equal(our,old)
    write(DOCS/'IMPLEMENTATION_CHECK.json',dict(numpy_torch_max_errors=errors,
        frozen_own_clock_first200_exact=True,identity_head_exact=True,non_target_phases_exact=True,
        later_conditional_head_unchanged=True,mixtures_normalized=True,first_gap_native_unchanged=True))


def initialize():
    prior.verify(); assert not (OUT/'MANIFEST.json').exists(); checks()
    files = list((OUT/'local_source').glob('*.py'))+[DOCS/'PROTOCOL.md',DOCS/'IMPLEMENTATION_CHECK.json',
        PRIOR/'personal_six/time_head.pt',PRIOR/'MANIFEST.json',PRIOR/'local_source/run_stratified.py',
        META/'metadata_optimization.parquet',META/'metadata_internal_validation.parquet',
        ROOT/'artifacts/argn_early_history_diagnostic_v1/local_source/diagnose.py']
    write(OUT/'MANIFEST.json',dict(config=CFG,hashes={str(p):digest(p) for p in files},
        created_utc=datetime.now(timezone.utc).isoformat(),local_only=True,test_events_read=False,
        baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    verify(); print('INITIALIZED',flush=True)


def fit_handoff(arm):
    verify(); mixture = json.loads((OUT/'INITIAL_MIXTURE.json').read_text())
    p = build_head(arm,mixture,[0.,0.]); kernel = Kernel(p)
    sim = Simulation('optimization',CFG['fit_noise_seed'])
    reference = sim.run(Kernel(build_head('identity',mixture,[0.,0.])))
    cache = {}; curve = []; started = time.monotonic()
    def evaluate(theta):
        key = tuple(map(float,theta))
        if key not in cache:
            metrics = sim.run(kernel,theta)
            cache[key] = dict(objective=prior.objective(metrics,reference),**metrics)
        row = dict(evaluation=len(curve),theta=list(key),seconds=time.monotonic()-started,**cache[key]);curve.append(row)
        pd.DataFrame(curve).to_csv(DOCS/f'{arm}_learning_curve.csv',index=False)
        print('FIT',arm,len(curve),np.round(theta,5),round(row['objective'],6),round(row['seconds'],1),flush=True)
        return row['objective']
    evaluate([0.,0.])
    opt = minimize(evaluate,[0.,0.],method='Powell',bounds=[(-.35,.35)]*2,
        options=dict(maxfev=CFG['maxfev']-1,xtol=.001,ftol=.001))
    best = min(curve,key=lambda r:r['objective'])
    output = build_head(arm,mixture,best['theta']); dest=OUT/arm;dest.mkdir(exist_ok=True)
    torch.save(output,dest/'time_head.pt')
    write(dest/'FIT_COMPLETE.json',dict(selected=best,reference=reference,initial=curve[0],evaluations=len(curve),
        optimizer_success=bool(opt.success),optimizer_message=str(opt.message),fit_support=sim.target.info,
        head_sha256=digest(dest/'time_head.pt'),new_native_argn_fits=0))
    verify();return arm


def prepare():
    verify(); assert not (OUT/'FIT_COMPLETE.json').exists()
    m = pd.read_parquet(META/'metadata_optimization.parquet')
    target = m[m.event_index.between(1,5)&m.previous.eq(0)&m.label.eq(0)]
    fit = GaussianMixture(3,covariance_type='full',random_state=CFG['mixture_seed'],n_init=3,
        max_iter=CFG['mixture_max_iter'],reg_covar=CFG['mixture_reg_covar']).fit(np.log1p(target.gap.to_numpy())[:,None])
    assert fit.converged_
    mixture = dict(weights=fit.weights_.tolist(),location=fit.means_[:,0].tolist(),scale=np.sqrt(fit.covariances_[:,0,0]).tolist(),
        rows=len(target),customers=target.entity_id.nunique(),converged=True,iterations=fit.n_iter_,
        seed=CFG['mixture_seed'],fit_split='optimization',first_positions=[1,5],phase='normal_stay')
    write(OUT/'INITIAL_MIXTURE.json',mixture)
    dest=OUT/'init_only';dest.mkdir(exist_ok=True);torch.save(build_head('init_only',mixture,[0.,0.]),dest/'time_head.pt')
    write(dest/'FIT_COMPLETE.json',dict(initial_mixture=mixture,handoff=[0.,0.],head_sha256=digest(dest/'time_head.pt')))
    print('INITIAL_MIXTURE_FIT',len(target),target.entity_id.nunique(),flush=True)
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(fit_handoff,['handoff_only','init_handoff']))
    diagnostics=[]
    for ns in CFG['diagnostic_seeds']:
        sim=Simulation('internal_validation',ns)
        for arm in ['identity']+ARMS:
            p=build_head('identity',mixture,[0.,0.]) if arm=='identity' else torch.load(OUT/arm/'time_head.pt',map_location='cpu',weights_only=True)
            diagnostics.append(dict(arm=arm,noise_seed=ns,**sim.run(Kernel(p))))
    pd.DataFrame(diagnostics).to_csv(DOCS/'internal_diagnostics.csv',index=False)
    verify();write(OUT/'FIT_COMPLETE.json',dict(initial_mixture_fits=1,calibration_fits=2,new_native_argn_fits=0,
        heads={a:digest(OUT/a/'time_head.pt') for a in ARMS},test_events_read=False))
    print('ALL_FITS_COMPLETE',flush=True)


def worker(fs,arm):
    verify()
    with patch.object(study,'OUT',OUT),patch.object(study,'DOCS',DOCS),patch.object(study,'CFG',CFG), \
         patch.object(study,'verify',verify),patch.object(study,'evolution_generation',initial_generation):
        study.worker(fs,arm)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['initialize','prepare','worker'])
    parser.add_argument('--seed',type=int,choices=CFG['fit_seeds']);parser.add_argument('--arm',choices=ARMS)
    a=parser.parse_args()
    initialize() if a.mode=='initialize' else prepare() if a.mode=='prepare' else worker(a.seed,a.arm)
