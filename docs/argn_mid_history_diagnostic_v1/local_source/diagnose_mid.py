"""Frozen-head local clock/correction interventions; no fitting or final-test reads."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import torch

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / 'artifacts/argn_clock_initial_coupling_v1'
OLD = ROOT / 'artifacts/argn_early_history_diagnostic_v1'
sys.path[:0] = [str(PRIOR/'local_source'), str(OLD/'local_source'), str(ROOT), str(ROOT/'scripts')]
import run_initial as frozen
import diagnose as old
import diagnose_amendment2 as mapped
from adaptive_clock import basis_numpy, LIMIT
from initial_clock import InitialClock
from benchmarks.argn_clock_evolution import ProgressClock, progress_features
from run_argn_clock_evolution import digest, write, META

OUT = ROOT / 'artifacts/argn_mid_history_diagnostic_v1'
DOCS = OUT / 'report'
HEAD = PRIOR / 'init_only/time_head.pt'
MODES = ['GG_on', 'RG_on', 'GR_on', 'RR_on', 'GG_off', 'RR_off']
BINS = [(6,50),(50,100),(100,150),(150,200),(100,200),(200,500),(500,1024)]
CFG = dict(horizon=1024, intervention=[100,200], modes=MODES, bins=BINS,
    noise_seeds=[20261202,20261203,20261204,20261205], max_cpu_jobs=4,
    fit_seeds=frozen.CFG['fit_seeds'], generation_seeds=frozen.CFG['generation_seeds'])
CFG = json.loads(json.dumps(CFG))


def verify():
    frozen.verify()
    manifest = json.loads((OUT/'MANIFEST.json').read_text())
    assert manifest['config'] == CFG
    for path, sha in manifest['hashes'].items():
        assert digest(path) == sha, path


class Kernel(frozen.Kernel):
    def __init__(self):
        super().__init__(torch.load(HEAD, map_location='cpu', weights_only=True))
        assert self.init and not np.any(self.handoff)
        self.fallback = self.p['fallback']
        self.gap_limits = self.p['limits']

    def split_parameters(self, phase, own, ov, actual, av, mode, t):
        xb, xd, vd = own.copy(), own.copy(), ov.copy()
        active = (phase == 0) & (100 <= t < 200)
        if mode[0] == 'R': xb[active] = actual[active]
        if mode[1] == 'R': xd[active] = actual[active]; vd[active] = av[active]
        p, mu, sd = self.density.parameters(phase, xb)
        delta = np.clip(basis_numpy('personal_six', xd, vd, self.limits)@self.theta, -LIMIT, LIMIT)*(phase == 0)
        if mode.endswith('off'): delta[active] = 0.
        loc = mu + delta[:,None]
        initial = (phase == 0) & (1 <= t <= 5)
        if initial.any():
            p[initial] = self.iw; loc[initial] = self.im; sd = sd.copy(); sd[initial] = self.isd
            mu = mu.copy(); mu[initial] = self.im; delta[initial] = 0.
        return p, loc, sd, delta, xb[:,0], (p*mu).sum(-1)


class Data(old.Data):
    def __init__(self, split, fs=None, gs=None):
        with patch.object(old, 'CFG', {**old.CFG, 'horizon': CFG['horizon']}):
            super().__init__(split)
        self.raw_clock = np.where(np.isfinite(self.raw_clock) & (self.raw_clock > 0), self.raw_clock, self.fallback)
        self.saved = None
        if fs is None: return
        assert split == 'development'
        mapping, audit = mapped.identity_map()
        path = PRIOR / f'runs/init_only_{fs}/generated_validation_{gs}.parquet'
        g = old.frame(path)
        record = g.entity_id.map(mapping)
        assert record.notna().all() and record.nunique() == 147
        column = {rid: j for j, rid in enumerate(self.ids)}
        cols = record.map(column).to_numpy(dtype=int)
        x, v = progress_features(g, self.fallback)
        keep = g.event_index.lt(CFG['horizon']).to_numpy()
        t, j = g.event_index.to_numpy()[keep], cols[keep]
        present = np.zeros(self.shape, bool); present[t,j] = True
        self.mask &= present
        self.phase = np.zeros(self.shape, int); self.phase[t,j] = g.phase_num.to_numpy()[keep]
        self.saved = dict(x=self.actual.copy(), valid=np.zeros(self.shape), raw=np.full(self.shape,self.fallback))
        self.saved['x'][t,j] = x[keep]; self.saved['valid'][t,j] = v[keep]
        raw = g.raw_clock.where(g.raw_clock.gt(0), self.fallback).to_numpy()[keep]
        self.saved['raw'][t,j] = raw
        self.tag.update(path_source='generated_labels', fit_seed=fs, generation_seed=gs)


def draw(kernel, parameters, uniform, normal):
    p, location, scale = parameters[:3]
    c = (uniform[:,None] > np.cumsum(p,axis=-1)).sum(-1).clip(max=2)
    idx = np.arange(len(c)); z = location[idx,c] + scale[idx,c]*normal
    lo, hi = np.log1p(kernel.gap_limits)
    return np.floor(np.expm1(np.clip(z,lo,hi))+1e-8), (z<lo)|(z>hi)


def simulate(data, kernel, mode, noise):
    h,n = data.shape; u,z = data.noise(noise)
    codec = ProgressClock(kernel.fallback); memory = codec.advance(codec.empty(n), np.zeros(n))
    out = {name: np.zeros((h,n)) for name in ['gap','clock','base_clock','base_mean','delta','clipped','raw_clock']}
    raw = np.full((n,20), np.nan)
    for t in range(1,h):
        own, ov = codec.features(memory)
        denom = np.median(raw[:,:min(t-1,20)],axis=1) if t>=6 else np.full(n,kernel.fallback)
        denom = np.where(denom>0,denom,kernel.fallback)
        params = kernel.split_parameters(data.phase[t],own,ov,data.actual[t],data.valid[t],mode,t)
        gap, clipped = draw(kernel,params,u[t],z[t])
        for key,value in [('gap',gap),('clock',own[:,0]),('base_clock',params[4]),('base_mean',params[5]),
                          ('delta',params[3]),('clipped',clipped),('raw_clock',denom)]: out[key][t] = value
        memory = codec.advance(memory,gap); raw[:,(t-1)%20] = gap
    return out


def fixed_prefix(data, kernel, mode, noise):
    assert data.saved is not None
    h,n = data.shape; u,z = data.noise(noise)
    out = {name: np.zeros((h,n)) for name in ['gap','clock','base_clock','base_mean','delta','clipped','raw_clock']}
    for t in range(100,200):
        own, ov = data.saved['x'][t], data.saved['valid'][t]
        params = kernel.split_parameters(data.phase[t],own,ov,data.actual[t],data.valid[t],mode,t)
        gap, clipped = draw(kernel,params,u[t],z[t])
        for key,value in [('gap',gap),('clock',own[:,0]),('base_clock',params[4]),('base_mean',params[5]),
                          ('delta',params[3]),('clipped',clipped),('raw_clock',data.saved['raw'][t])]:out[key][t] = value
    return out


def metrics(data, out, mode, noise, style):
    rows = []; t = np.arange(data.shape[0])[:,None]
    bins = BINS if style == 'recursive' else [(100,150),(150,200),(100,200)]
    for lo,hi in bins:
        for group in ['normal','NN']:
            labels = ((data.phase%2 == 0)&(data.real_phase%2 == 0)) if group=='normal' else ((data.phase==0)&(data.real_phase==0))
            mask = data.mask & labels & (t>=lo) & (t<hi)
            if not mask.any(): continue
            real, gen = data.real[mask], out['gap'][mask]; ids=data.flat_ids[mask]
            x, own = data.actual[:,:,0][mask], out['clock'][mask]
            delta=out['delta'][mask]
            r = dict(**data.tag,style=style,mode=mode,noise_seed=noise,start=lo,end=hi,group=group,
                events=int(mask.sum()),customers=len(np.unique(ids)),**old.distances(real,gen,ids,ids),
                own_clock_w1=wasserstein_distance(x,own),own_log_clock_bias=float((own-x).mean()),
                base_clock_w1=wasserstein_distance(x,out['base_clock'][mask]),
                base_expected_log_gap_bias=float((out['base_mean'][mask]-np.log1p(real)).mean()),
                mean_delta=float(delta.mean()),q10_delta=float(np.quantile(delta,.1)),q90_delta=float(np.quantile(delta,.9)),
                saturated_delta_fraction=float((np.abs(delta)>=LIMIT-1e-12).mean()),
                clipped_fraction=float(out['clipped'][mask].mean()),
                personal_gap_w1=wasserstein_distance(np.log1p(real/data.raw_clock[mask]),np.log1p(gen/out['raw_clock'][mask])))
            rows.append(r)
    return rows


def job(spec):
    split,fs,gs = spec; data=Data(split,fs,gs); kernel=Kernel()
    noises=CFG['noise_seeds'][:2] if split=='optimization' or fs is not None else CFG['noise_seeds']
    rows=[]; recursive=0; fixed=0; values=0; early_checks=0
    for ns in noises:
        baseline=None
        for mode in MODES:
            out=simulate(data,kernel,mode,ns)
            if baseline is None:baseline=out['gap'][:100].copy()
            else:np.testing.assert_array_equal(out['gap'][:100],baseline);early_checks+=1
            rows+=metrics(data,out,mode,ns,'recursive');recursive+=1;values+=int(data.mask.sum())
            if fs is not None:
                out=fixed_prefix(data,kernel,mode,ns)
                rows+=metrics(data,out,mode,ns,'saved_prefix');fixed+=1
                values+=int(data.mask[100:200].sum())
        print('NOISE_COMPLETE',data.tag,ns,flush=True)
    name=split if fs is None else f'generated_{fs}_{gs}'
    pd.DataFrame(rows).to_csv(DOCS/f'{name}.csv',index=False)
    write(DOCS/f'{name}_COMPLETE.json',dict(recursive_replays=recursive,fixed_prefix_resamples=fixed,
        evaluated_time_values=values,early_prefix_checks=early_checks,test_events_read=False))
    return name,recursive,fixed,values


def checks():
    frozen.verify(); kernel=Kernel(); data=Data('internal_validation'); noise=CFG['noise_seeds'][0]
    original=frozen.Simulation('internal_validation',noise).sample(frozen.Kernel(kernel.p),horizon=1024)[0]
    ours=simulate(data,kernel,'GG_on',noise)
    np.testing.assert_array_equal(ours['gap'],original)
    rng=np.random.default_rng(20261207); maxerr=0.
    model=InitialClock(kernel.p['dim'],'init_only').double();model.load_state_dict(kernel.p['state_dict'])
    for t in [1,5,6,99,100,149,199,200,1023]:
        x=np.c_[rng.uniform(6,14,40),np.full(40,np.log1p(t))];v=np.ones(40);phase=rng.integers(0,4,40)
        p=kernel.split_parameters(phase,x,v,x,v,'GG_on',t)
        with torch.no_grad(): ref=model(torch.zeros(40,kernel.p['dim'],dtype=torch.double),torch.tensor(phase),torch.tensor(x),torch.tensor(v))
        err=max(np.max(abs(a-b.detach().numpy())) for a,b in zip(p[:3],[ref[0].softmax(-1),ref[1],ref[2]]));maxerr=max(maxerr,err)
        assert err<1e-11
        off=kernel.split_parameters(phase,x,v,x,v,'GG_off',t)
        untouched=(phase!=0) | (t<100) | (t>=200)
        for a,b in zip(p[:3],off[:3]):np.testing.assert_array_equal(a[untouched],b[untouched])
        np.testing.assert_array_equal(p[0],off[0]);np.testing.assert_array_equal(p[2],off[2])
    for mode in MODES[1:]:
        other=simulate(data,kernel,mode,noise)
        np.testing.assert_array_equal(other['gap'][:100],ours['gap'][:100])
    _,mapping_check=mapped.identity_map()
    mapped_data=Data('development',CFG['fit_seeds'][0],CFG['generation_seeds'][0])
    assert np.isfinite(mapped_data.saved['x'][mapped_data.mask]).all()
    write(DOCS/'IMPLEMENTATION_CHECK.json',dict(numpy_torch_max_error=maxerr,
        unmodified_replay_matches_previous_exact=True,all_interventions_preserve_first100_exact=True,
        non_target_same_input_law_exact=True,fixed_prefix_probabilities_scales_unchanged_when_removing_delta=True,
        entity_mapping=mapping_check,previous_strict_past_checks_preserved=True))


def initialize():
    assert not (OUT/'MANIFEST.json').exists();checks()
    paths=list((OUT/'local_source').glob('*.py'))+[DOCS/'PROTOCOL.md',DOCS/'IMPLEMENTATION_CHECK.json',HEAD,
        PRIOR/'MANIFEST.json',PRIOR/'report/ARTIFACT_HASHES.json',PRIOR/'local_source/run_initial.py',
        PRIOR/'local_source/initial_clock.py',OLD/'local_source/diagnose.py',OLD/'local_source/diagnose_amendment2.py',
        mapped.POSITIONS,mapped.INPUTS]
    paths += [META/f'metadata_{s}.parquet' for s in ['optimization','internal_validation','development']]
    paths += [PRIOR/f'runs/init_only_{fs}/generated_validation_{gs}.parquet' for fs in CFG['fit_seeds'] for gs in CFG['generation_seeds']]
    write(OUT/'MANIFEST.json',dict(config=CFG,hashes={str(p):digest(p) for p in paths},
        created_utc=datetime.now(timezone.utc).isoformat(),new_fits=0,test_events_read=False))
    verify();print('INITIALIZED',flush=True)


def run():
    verify();assert not (DOCS/'COMPLETE.json').exists()
    write(OUT/'START.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),cpu_workers=4,config=CFG))
    specs=[(s,None,None) for s in ['optimization','internal_validation','development']]
    specs += [('development',fs,gs) for fs in CFG['fit_seeds'] for gs in CFG['generation_seeds']]
    with ProcessPoolExecutor(max_workers=4) as pool: results=list(pool.map(job,specs))
    frames=pd.concat([pd.read_csv(DOCS/f'{name}.csv') for name,*_ in results],ignore_index=True)
    frames.to_csv(DOCS/'all_diagnostics.csv',index=False)
    keys=['split','path_source','style','mode','start','end','group']
    frames.groupby(keys).mean(numeric_only=True).to_csv(DOCS/'mean_diagnostics.csv')
    verify();write(DOCS/'COMPLETE.json',dict(recursive_replays=sum(r[1] for r in results),
        fixed_prefix_resamples=sum(r[2] for r in results),evaluated_time_values=sum(r[3] for r in results),
        new_model_fits=0,new_native_full_generations=0,test_events_read=False,pushed=False,
        completed_utc=datetime.now(timezone.utc).isoformat()))
    print('DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initialize','run']);a=p.parse_args()
    initialize() if a.mode=='initialize' else run()
