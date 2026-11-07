"""Isolated full-support control. Never modifies or promotes the prior candidate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import run_argn_clock_evolution as study
import run_argn_clock_location as previous
from benchmarks.argn_clock_evolution import NumpyDensity, progress_features
from benchmarks.argn_clock_location import ClockLocation, location_generation
from run_argn_clock_evolution import digest, write, objective, META

OUT = ROOT/'artifacts/argn_clock_full_length_v1'
DOCS = OUT/'report'
CFG = {**previous.CFG, 'arms': ['clock_full_prefix'], 'horizon': None,
       'prefix_horizon': 1024, 'noise_extension': 'preserve_prefix_then_append'}


def verify():
    previous.verify()
    m = json.loads((OUT/'MANIFEST.json').read_text())
    assert m['config'] == CFG
    for p, sha in m['hashes'].items():
        assert digest(p) == sha, p
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == m['baseline_commit']
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip()


class ActualPrefix:
    """Exact event-sparse counterpart of the frozen actual-prefix simulator."""
    def __init__(self, split, payload, horizon, noise_seed):
        all_m = pd.read_parquet(META/f'metadata_{split}.parquet')
        if horizon is None:
            horizon = int(all_m.event_index.max()) + 1
        m = all_m[all_m.event_index.lt(horizon)].copy()
        ids = pd.factorize(m.entity_id, sort=False)[0]
        t = m.event_index.to_numpy()
        shape = (int(t.max())+1, int(ids.max())+1)
        rng = np.random.default_rng(noise_seed)
        prefix_shape = (min(shape[0], 1024), shape[1])
        u, z = rng.random(prefix_shape), rng.standard_normal(prefix_shape)
        if shape[0] > 1024:
            extra = (shape[0]-1024, shape[1])
            u = np.concatenate([u, rng.random(extra)], axis=0)
            z = np.concatenate([z, rng.standard_normal(extra)], axis=0)
        keep = t > 0
        u, z = u[t[keep], ids[keep]], z[t[keep], ids[keep]]
        m = m[keep].copy()
        x, valid = progress_features(m, payload['fallback'])
        phase = 2*m.previous.clip(lower=0).to_numpy()+m.label.to_numpy()
        density = NumpyDensity(payload)
        probs, loc, scale = density.parameters(phase, x)
        comp = (u[:,None] > np.cumsum(probs, axis=1)).sum(-1).clip(max=probs.shape[-1]-1)
        rows = np.arange(len(comp))
        self.base_z = loc[rows,comp]+scale[rows,comp]*z
        self.nn = phase == 0
        self.decay = np.exp(-m.event_index.to_numpy()/50)
        self.denominator = m.past_gap.fillna(payload['fallback']).to_numpy()
        self.masks = dict(normal=self.nn, fraud=phase%2==1,
                          personal=self.nn&(valid>.5), onset=(phase==1)&(valid>.5))
        for i,(lo,hi) in enumerate([(1,50),(50,200),(200,horizon)]):
            self.masks[f'bin{i}'] = self.nn & m.event_index.ge(lo).to_numpy() & m.event_index.lt(hi).to_numpy()
        ref_z = np.log1p(m.gap.to_numpy())
        ref_rel = np.log1p(m.gap.to_numpy()/np.maximum(self.denominator,1e-12))
        self.refsort = {k: np.sort((ref_rel if k in ['personal','onset'] else ref_z)[mask])
                        for k,mask in self.masks.items()}
        assert all(len(v) for v in self.refsort.values())
        self.info = dict(split=split,horizon=horizon,customers=m.entity_id.nunique(),events=len(m),
                         tail_events=int(m.event_index.ge(1024).sum()),
                         counts={k:int(v.sum()) for k,v in self.masks.items()})

    def run(self, theta):
        shifted = self.base_z+(theta[0]+theta[1]*self.decay)*self.nn
        values = np.floor(np.expm1(np.clip(shifted,0,np.log1p(1045691.)))+1e-8)
        z = np.log1p(values)
        rel = np.log1p(values/np.maximum(self.denominator,1e-12))
        metrics = {'clock':0.}
        for k,mask in self.masks.items():
            values = rel if k in ['personal','onset'] else z
            metrics[k] = float(np.mean(np.abs(self.refsort[k]-np.sort(values[mask]))))
        return metrics


def fit(sim, name):
    reference = sim.run([0.,0.]); curve = []
    def evaluate(theta):
        metrics = sim.run(theta); value = objective(metrics,reference)
        curve.append(dict(evaluation=len(curve),a=float(theta[0]),b=float(theta[1]),objective=value,**metrics))
        print('FIT',name,len(curve),np.round(theta,7),round(value,8),flush=True)
        return value
    evaluate([0.,0.])
    result = minimize(evaluate,np.zeros(2),method='Powell',bounds=CFG['bounds'],
                      options=dict(maxfev=CFG['maxfev']-1,xtol=.001,ftol=.001))
    best = min(curve,key=lambda row:row['objective'])
    pd.DataFrame(curve).to_csv(DOCS/f'{name}_learning_curve.csv',index=False)
    record = dict(selected=best,initial=curve[0],reference=reference,source='actual',
                  optimizer_success=bool(result.success),optimizer_message=str(result.message),
                  evaluations=len(curve),support=sim.info)
    write(DOCS/f'{name}_fit.json',record)
    return record


def prepare():
    previous.verify(); assert not (OUT/'MANIFEST.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    paths = [Path(__file__),DOCS/'PROTOCOL.md',ROOT/'artifacts/argn_clock_location_v1/MANIFEST.json',
             ROOT/'artifacts/argn_clock_location_v1/clock_prefix/time_head.pt',
             ROOT/'artifacts/argn_clock_regression_v1/time_head.pt',
             ROOT/'scripts/run_argn_clock_evolution.py',ROOT/'scripts/run_argn_clock_location.py',
             META/'metadata_optimization.parquet',META/'metadata_internal_validation.parquet']
    write(OUT/'MANIFEST.json',dict(config=CFG,hashes={str(p):digest(p) for p in paths},
          baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          created_utc=datetime.now(timezone.utc).isoformat(),test_events_read=False,local_only=True))
    payload = torch.load(ROOT/'artifacts/argn_clock_regression_v1/time_head.pt',map_location='cpu',weights_only=True)
    old_fit = json.loads((previous.DOCS/'clock_prefix_fit.json').read_text())
    old_theta = [old_fit['selected']['a'],old_fit['selected']['b']]
    sim = ActualPrefix('optimization',payload,1024,CFG['fit_noise_seed'])
    frozen = study.ConditionalSimulation('optimization',payload['fallback'])
    checks=[]
    for theta in [[0.,0.],old_theta,[-.03,.17]]:
        expected = frozen.run(NumpyDensity(payload),theta,'actual',CFG['fit_noise_seed'])
        actual = sim.run(theta)
        error = max(abs(actual[k]-expected[k]) for k in expected)
        assert error < 1e-12,(theta,error,actual,expected)
        checks.append(dict(theta=theta,max_metric_error=error))
    replay = fit(sim,'prefix_1024_replay')
    original_curve = pd.read_csv(previous.DOCS/'clock_prefix_learning_curve.csv')
    replay_curve = pd.read_csv(DOCS/'prefix_1024_replay_learning_curve.csv')
    np.testing.assert_allclose(replay_curve[original_curve.columns],original_curve,rtol=0,atol=1e-9)
    write(DOCS/'IMPLEMENTATION_CHECK.json',dict(parity=checks,all_48_evaluations_reproduced=True,
          selected_coefficients_reproduced=True,prefix_noise_unchanged=True))
    del frozen,sim
    full = ActualPrefix('optimization',payload,None,CFG['fit_noise_seed'])
    record = fit(full,'clock_full_prefix')
    chosen = [record['selected']['a'],record['selected']['b']]
    head = ClockLocation(1,'clock_full_prefix')
    head.load_state_dict({**payload['state_dict'],'correction':torch.tensor(chosen,dtype=torch.float32)})
    for k,v in payload['state_dict'].items():
        torch.testing.assert_close(head.state_dict()[k],v,rtol=0,atol=0)
    dest = OUT/'clock_full_prefix'; dest.mkdir()
    torch.save({**payload,'arm':'clock_full_prefix','state_dict':head.state_dict()},dest/'time_head.pt')
    record.update(head_sha256=digest(dest/'time_head.pt'),original_gmr_buffers_unchanged=True)
    write(dest/'FIT_COMPLETE.json',record)
    rows=[]
    for horizon in [1024,None]:
        for ns in CFG['diagnostic_seeds']:
            sim = ActualPrefix('internal_validation',payload,horizon,ns)
            for name,theta in [('clock_joint',[0.,0.]),('clock_prefix',old_theta),('clock_full_prefix',chosen)]:
                rows.append(dict(arm=name,noise_seed=ns,support='full' if horizon is None else '1024',
                                 **sim.info,**sim.run(theta)))
    pd.DataFrame(rows).to_csv(DOCS/'internal_diagnostics.csv',index=False)
    verify()
    write(OUT/'FIT_COMPLETE.json',dict(new_calibration_fits=1,reproduction_fits=1,native_argn_fits=0,
          test_events_read=False,heads={'clock_full_prefix':digest(dest/'time_head.pt')}))
    print('FULL_LENGTH_PREPARED',chosen,flush=True)


def execute(mode, fs, arm):
    verify()
    with patch.object(study,'OUT',OUT),patch.object(study,'DOCS',DOCS),patch.object(study,'CFG',CFG), \
         patch.object(study,'verify',verify),patch.object(study,'evolution_generation',location_generation), \
         patch.object(study,'__file__',str(Path(__file__).resolve())):
        study.worker(fs,arm) if mode=='worker' else study.dispatch()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','dispatch'])
    p.add_argument('--seed',type=int,choices=CFG['fit_seeds']);p.add_argument('--arm',choices=CFG['arms']);a=p.parse_args()
    prepare() if a.mode=='prepare' else execute(a.mode,a.seed,a.arm)
