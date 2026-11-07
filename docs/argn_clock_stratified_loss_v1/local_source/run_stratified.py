"""Loss-only refinement of the frozen 200+ position group; no reweighting."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance
import torch

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT/'artifacts/argn_clock_adaptive_v1'
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(PRIOR/'local_source')]
import run_adaptive as prior
import run_argn_clock_evolution as study
from run_adaptive import Distance, equal_weights, payload, digest, write, META
from adaptive_clock import ARMS, AdaptiveClock, adaptive_generation, basis_numpy, LIMIT

OUT = ROOT/'artifacts/argn_clock_stratified_loss_v1'
DOCS = OUT/'report'
FINE_BINS = [(200,500), (500,1024), (1024,2048), (2048,None)]
CFG = {**prior.CFG, 'loss_refinement': FINE_BINS,
       'subgroup_mass': 'inherited_original_bin2_row_weights',
       'maximum_own_workers_per_gpu': 4}
# JSON uses lists instead of tuples; normalize once for exact manifest equality.
CFG = json.loads(json.dumps(CFG))


def verify():
    prior.verify()
    m = json.loads((OUT/'MANIFEST.json').read_text())
    assert m['config'] == CFG
    for path, sha in m['hashes'].items():
        assert digest(path) == sha, path


class Simulation(prior.Simulation):
    def __init__(self, split, old, noise_seed, weighting):
        super().__init__(split, old, noise_seed, weighting)
        m = pd.read_parquet(META/f'metadata_{split}.parquet')
        m = m[m.event_index.gt(0)]
        parent = self.masks['bin2']
        # Inherit the ORIGINAL parent measure. Recomputing inverse customer
        # counts per child would change the intervention to loss + reweighting.
        weights = np.zeros(len(m))
        weights[parent] = equal_weights(self.ids[parent]) if weighting == 'customer' else 1.
        target = np.log1p(m.gap.to_numpy())
        position = m.event_index.to_numpy()
        self.groups = []
        restored = np.zeros(len(m))
        for lo, hi in FINE_BINS:
            mask = parent & (position >= lo)
            if hi is not None:
                mask &= position < hi
            mass = float(weights[mask].sum()/weights[parent].sum())
            assert mask.any(), (split, lo, hi)
            distance = Distance(target[mask], weights[mask])
            self.groups.append((lo, hi, mask, mass, distance))
            restored[mask] = mass*weights[mask]/weights[mask].sum()
        np.testing.assert_allclose(restored, weights/weights.sum(), rtol=1e-13, atol=1e-16)
        assert abs(sum(g[3] for g in self.groups)-1) < 1e-12
        self.group_support = [dict(start=lo, end=hi, rows=int(mask.sum()),
            customers=int(pd.Series(self.ids[mask]).nunique()), mass=mass)
            for lo,hi,mask,mass,_ in self.groups]

    def measure(self, delta):
        result = super().measure(delta)
        y = np.floor(np.expm1(np.clip(self.base_z+delta*self.nn, 0, np.log1p(1045691.)))+1e-8)
        z = np.log1p(y)
        refined = 0.
        for lo,hi,mask,mass,distance in self.groups:
            value = distance(z[mask])
            result[f'fine_{lo}'] = value
            refined += mass*value
        result['bin2_refined'] = refined
        result['pooling_gap'] = refined-result['bin2']
        # A conditional coupling is feasible for the pooled OT problem.
        assert result['pooling_gap'] >= -1e-10, result
        return result


def objective(metrics, reference):
    return prior.objective({**metrics, 'bin2': metrics['bin2_refined']}, reference)


def checks():
    # Opposite errors cancel in the pooled marginal but not within position.
    x = np.array([0.,0.,10.,10.]); y = np.array([10.,10.,0.,0.])
    assert Distance(x)(y) == 0
    refined = .5*Distance(x[:2])(y[:2])+.5*Distance(x[2:])(y[2:])
    assert refined == 10
    rng = np.random.default_rng(822)
    weights = rng.uniform(.1,1,40); a = rng.normal(size=40); b = rng.normal(size=40)
    np.testing.assert_allclose(Distance(a,weights)(b), wasserstein_distance(a,b,u_weights=weights,v_weights=weights), atol=1e-12)
    # A single child with full mass exactly recovers the parent distance.
    assert weights.sum()/weights.sum()*Distance(a,weights)(b) == Distance(a,weights)(b)
    old = payload(); limits = json.loads((DOCS/'CLOCK_ENDPOINTS.json').read_text())['values']
    records=[]
    for weighting in ['event','customer']:
        sim = Simulation('internal_validation',old,CFG['diagnostic_seeds'][0],weighting)
        for arm in ARMS:
            oldfit = json.loads((PRIOR/arm/'FIT_COMPLETE.json').read_text())['selected']['theta']
            delta = np.clip(basis_numpy(arm,sim.x,sim.valid,limits)@oldfit,-LIMIT,LIMIT)
            old_metrics = prior.Simulation.measure(sim,delta)
            new_metrics = sim.measure(delta)
            assert all(old_metrics[k] == new_metrics[k] for k in old_metrics)
            ref = sim.measure(np.zeros(len(delta)))
            change = objective(new_metrics,ref)-prior.objective(old_metrics,ref)
            np.testing.assert_allclose(change,new_metrics['pooling_gap']/6,rtol=0,atol=1e-12)
            records.append(dict(weighting=weighting,arm=arm,unchanged_terms_exact=True,
                objective_change=change,pooling_gap=new_metrics['pooling_gap'],support=sim.group_support))
    write(DOCS/'IMPLEMENTATION_CHECK.json',dict(toy_pooled=0,toy_refined=10,
        exact_weighted_w1=True,collapsed_partition_parity=True,
        original_row_and_customer_mass_preserved=True,checks=records))


def fit_arm(arm):
    verify(); old=payload()
    weighting='event' if arm=='event_two' else 'customer'
    sim=Simulation('optimization',old,CFG['fit_noise_seed'],weighting)
    limits=json.loads((DOCS/'CLOCK_ENDPOINTS.json').read_text())['values']
    basis=basis_numpy(arm,sim.x,sim.valid,limits)
    reference=sim.measure(np.zeros(len(sim.nn)))
    curve=[]; cache={}; start=time.monotonic()
    def evaluate(theta):
        key=tuple(theta)
        if key not in cache:
            raw=basis@theta; metrics=sim.measure(np.clip(raw,-LIMIT,LIMIT))
            cache[key]=dict(objective=objective(metrics,reference),
                coarse_objective=prior.objective(metrics,reference),**metrics,
                clipped_offset_fraction=float(np.mean(np.abs(raw)>LIMIT)))
        rec=dict(evaluation=len(curve),theta=list(map(float,theta)),seconds=time.monotonic()-start,**cache[key])
        curve.append(rec)
        if len(curve)%16==0:
            print('FIT',arm,len(curve),round(rec['objective'],8),flush=True)
            pd.DataFrame(curve).to_csv(DOCS/f'{arm}_learning_curve.csv',index=False)
        return rec['objective']
    size=basis.shape[1]; bounds=CFG['bounds'] if size==2 else [(-LIMIT,LIMIT)]*size
    evaluate(np.zeros(size))
    result=minimize(evaluate,np.zeros(size),method='Powell',bounds=bounds,
        options=dict(maxfev=CFG['maxfev']-1,xtol=.001,ftol=.001))
    best=min(curve,key=lambda x:x['objective'])
    pd.DataFrame(curve).to_csv(DOCS/f'{arm}_learning_curve.csv',index=False)
    # Report the former fit under BOTH objectives; never use it to initialize
    # or replace the registered zero-initialized Powell fit.
    oldtheta=json.loads((PRIOR/arm/'FIT_COMPLETE.json').read_text())['selected']['theta']
    oldmetrics=sim.measure(np.clip(basis@oldtheta,-LIMIT,LIMIT))
    head=AdaptiveClock(1,arm)
    head.load_state_dict({**old['state_dict'],'clock_limits':torch.tensor(limits,dtype=torch.float32),
        'correction':torch.tensor(best['theta'],dtype=torch.float32)})
    for k,v in old['state_dict'].items(): torch.testing.assert_close(head.state_dict()[k],v,rtol=0,atol=0)
    dest=OUT/arm;dest.mkdir(exist_ok=True)
    torch.save({**old,'arm':arm,'state_dict':head.state_dict()},dest/'time_head.pt')
    record=dict(arm=arm,selected=best,initial=curve[0],support=sim.info,subgroup_support=sim.group_support,
        weighting=weighting,evaluations=len(curve),distinct_evaluations=len(cache),
        optimizer_success=bool(result.success),optimizer_message=str(result.message),
        old_coefficients_on_same_noise=dict(theta=oldtheta,**oldmetrics,
            objective=objective(oldmetrics,reference),coarse_objective=prior.objective(oldmetrics,reference)),
        head_sha256=digest(dest/'time_head.pt'),original_gmr_buffers_unchanged=True,new_native_argn_fits=0)
    write(dest/'FIT_COMPLETE.json',record);write(DOCS/f'{arm}_fit.json',record)
    verify();print('FIT_COMPLETE',arm,len(curve),best['theta'],flush=True)
    return record


def initialize():
    prior.verify();assert not (OUT/'MANIFEST.json').exists()
    shutil.copyfile(PRIOR/'report/CLOCK_ENDPOINTS.json',DOCS/'CLOCK_ENDPOINTS.json')
    checks()
    paths=list((OUT/'local_source').glob('*.py'))+[DOCS/'PROTOCOL.md',DOCS/'CLOCK_ENDPOINTS.json',
        DOCS/'IMPLEMENTATION_CHECK.json',PRIOR/'MANIFEST.json',PRIOR/'AMENDMENT_01.json',
        PRIOR/'local_source/run_adaptive.py',PRIOR/'local_source/adaptive_clock.py',
        PRIOR/'local_source/report_adaptive.py',META/'metadata_optimization.parquet',
        META/'metadata_internal_validation.parquet']
    for arm in ARMS: paths.extend([PRIOR/arm/'FIT_COMPLETE.json',PRIOR/arm/'time_head.pt'])
    write(OUT/'MANIFEST.json',dict(config=CFG,hashes={str(p):digest(p) for p in paths},
        created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        test_events_read=False,local_only=True,only_intervention='within_200plus_OT_coupling'))
    evaluator=OUT/'local_source/report_stratified.py'
    write(OUT/'EVALUATION_MANIFEST.json',dict(source=str(evaluator),source_sha256=digest(evaluator),
        created_utc=datetime.now(timezone.utc).isoformat(),registered_before_new_development_generations=True))
    verify();write(OUT/'INITIALIZED.json',dict(checks_passed=True));print('INITIALIZED',flush=True)


def prepare():
    verify();assert (OUT/'INITIALIZED.json').exists() and not (OUT/'FIT_COMPLETE.json').exists()
    with ProcessPoolExecutor(max_workers=4) as pool: fits=list(pool.map(fit_arm,ARMS))
    rows=[];old=payload();limits=json.loads((DOCS/'CLOCK_ENDPOINTS.json').read_text())['values']
    for ns in CFG['diagnostic_seeds']:
        for weighting in ['event','customer']:
            sim=Simulation('internal_validation',old,ns,weighting)
            reference=sim.measure(np.zeros(len(sim.nn)))
            for loss in ['coarse','refined']:
                for arm in ARMS:
                    source=PRIOR if loss=='coarse' else OUT
                    theta=json.loads((source/arm/'FIT_COMPLETE.json').read_text())['selected']['theta']
                    metrics=sim.measure(np.clip(basis_numpy(arm,sim.x,sim.valid,limits)@theta,-LIMIT,LIMIT))
                    rows.append(dict(arm=arm,loss=loss,noise_seed=ns,weighting=weighting,**metrics,
                        refined_objective=objective(metrics,reference),coarse_objective=prior.objective(metrics,reference)))
    pd.DataFrame(rows).to_csv(DOCS/'internal_diagnostics.csv',index=False)
    verify();write(OUT/'FIT_COMPLETE.json',dict(new_calibration_fits=4,new_native_argn_fits=0,
        test_events_read=False,heads={a:digest(OUT/a/'time_head.pt') for a in ARMS}))
    print('ALL_FITS_COMPLETE',flush=True)


def worker(fs,arm):
    verify()
    with patch.object(study,'OUT',OUT),patch.object(study,'DOCS',DOCS),patch.object(study,'CFG',CFG), \
         patch.object(study,'verify',verify),patch.object(study,'evolution_generation',adaptive_generation):
        study.worker(fs,arm)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initialize','prepare','worker'])
    p.add_argument('--seed',type=int,choices=CFG['fit_seeds']);p.add_argument('--arm',choices=ARMS)
    a=p.parse_args()
    if a.mode=='initialize':initialize()
    elif a.mode=='prepare':prepare()
    else:worker(a.seed,a.arm)
