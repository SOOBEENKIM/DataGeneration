"""Four frozen, comparable fits and native free-generation experiments."""
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
from scipy.stats import wasserstein_distance
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'artifacts/argn_clock_full_length_v1/local_source')]
import run_full_length as full
import run_argn_clock_evolution as study
from run_argn_clock_evolution import digest, write, objective, META
from benchmarks.argn_clock_evolution import progress_features, ProgressClock
from benchmarks.argn_clock_regression import ClockRegression
from adaptive_clock import ARMS, AdaptiveClock, adaptive_generation, basis_numpy, LIMIT

OUT = ROOT/'artifacts/argn_clock_adaptive_v1'
DOCS = OUT/'report'
CFG = {**full.CFG, 'arms': ARMS, 'maxfev': 1536, 'max_parallel_fits': 4,
       'delta_limit': LIMIT, 'weighting': 'equal_customer_within_each_objective_group'}


def verify():
    full.verify()
    m = json.loads((OUT/'MANIFEST.json').read_text())
    assert m['config'] == CFG
    amendment = json.loads((OUT/'AMENDMENT_01.json').read_text()) if (OUT/'AMENDMENT_01.json').exists() else {}
    for p, sha in m['hashes'].items():
        if p in amendment.get('updates', {}):
            change = amendment['updates'][p]
            assert sha == change['original_sha256']
            assert digest(change['archived_original']) == sha
            sha = change['updated_sha256']
        assert digest(p) == sha, p


def payload():
    return torch.load(ROOT/'artifacts/argn_clock_regression_v1/time_head.pt', map_location='cpu', weights_only=True)


def equal_weights(ids):
    _, inverse, count = np.unique(ids, return_inverse=True, return_counts=True)
    return 1./count[inverse]


class Distance:
    """Exact CDF integration with a cached real sort (not quantile-grid approximation)."""
    def __init__(self, real, weights=None):
        i = np.argsort(real, kind='stable')
        self.x = np.asarray(real)[i]
        self.w = None if weights is None else np.asarray(weights)
        if weights is not None:
            self.cdf = np.r_[0., np.cumsum(self.w[i])/self.w.sum()]

    def __call__(self, generated):
        if self.w is None:
            return float(np.mean(np.abs(self.x-np.sort(generated))))
        j = np.argsort(generated, kind='stable')
        y = generated[j]
        cy = np.r_[0., np.cumsum(self.w[j])/self.w.sum()]
        z = np.sort(np.r_[self.x, y], kind='mergesort')
        dx = np.diff(z)
        ix = np.searchsorted(self.x, z[:-1], side='right')
        iy = np.searchsorted(y, z[:-1], side='right')
        return float(np.sum(np.abs(self.cdf[ix]-cy[iy])*dx))


class Simulation(full.ActualPrefix):
    def __init__(self, split, old, noise_seed, weighting):
        super().__init__(split, old, None, noise_seed)
        m = pd.read_parquet(META/f'metadata_{split}.parquet')
        m = m[m.event_index.gt(0)].copy()
        assert len(m) == len(self.base_z)
        self.x, self.valid = progress_features(m, old['fallback'])
        self.ids = m.entity_id.to_numpy()
        real = np.log1p(m.gap.to_numpy())
        relative = np.log1p(m.gap.to_numpy()/np.maximum(self.denominator, 1e-12))
        self.distances = {}
        for k, mask in self.masks.items():
            w = equal_weights(self.ids[mask]) if weighting == 'customer' else None
            target = relative if k in ['personal', 'onset'] else real
            self.distances[k] = Distance(target[mask], w)
        self.constant = {}

    def measure(self, delta):
        shifted = self.base_z+delta*self.nn
        y = np.floor(np.expm1(np.clip(shifted, 0, np.log1p(1045691.)))+1e-8)
        z = np.log1p(y)
        relative = np.log1p(y/np.maximum(self.denominator, 1e-12))
        metrics = {'clock': 0.}
        for k, mask in self.masks.items():
            if k in self.constant:
                metrics[k] = self.constant[k]
            else:
                v = relative if k in ['personal', 'onset'] else z
                metrics[k] = self.distances[k](v[mask])
                if k in ['fraud', 'onset']:
                    self.constant[k] = metrics[k]
        return metrics


def clock_endpoints(sim):
    mask = sim.nn & (sim.valid > .5)
    x = sim.x[mask, 0]
    w = equal_weights(sim.ids[mask]); order = np.argsort(x)
    cdf = np.cumsum(w[order])/w.sum()
    return x[order][np.searchsorted(cdf, [.1, .9])].tolist()


def checks(old, limits):
    rng = np.random.default_rng(724)
    errors = []
    for n in [7, 61, 900]:
        x = np.round(rng.normal(size=n), 1)
        y = np.round(rng.normal(size=n), 1)
        ids = rng.integers(0, 9, n); w = equal_weights(ids)
        expected = wasserstein_distance(x, y, u_weights=w, v_weights=w)
        actual = Distance(x, w)(y)
        assert abs(actual-expected) < 1e-12
        np.testing.assert_allclose(pd.Series(w).groupby(ids).sum(), 1., rtol=0, atol=1e-12)
        errors.append(abs(actual-expected))
    x = np.c_[rng.uniform(0, 15, 400), np.log1p(rng.integers(0, 7000, 400))]
    valid = rng.integers(0, 2, 400).astype(float)
    phase = rng.integers(0, 4, 400)
    base = torch.zeros(400, 1)
    xt, vt, pt = torch.tensor(x, dtype=torch.float64), torch.tensor(valid), torch.tensor(phase)
    original = ClockRegression(1, 'clock_joint').double()
    original.load_state_dict(old['state_dict'])
    reference = original(base, pt, xt[:, 0], vt)
    for arm in ARMS:
        head = AdaptiveClock(1, arm).double()
        head.load_state_dict({**old['state_dict'], 'clock_limits': torch.tensor(limits, dtype=torch.float64),
                             'correction': torch.zeros_like(head.correction)})
        b = basis_numpy(arm, x, valid, limits)
        np.testing.assert_allclose(head.basis(xt, vt).numpy(), b, rtol=0, atol=1e-12)
        if arm in ARMS[2:]:
            np.testing.assert_allclose(b.sum(1), 1., rtol=0, atol=1e-12)
            assert b.min() >= 0
        for a, ref in zip(head(base, pt, xt, vt), reference):
            torch.testing.assert_close(a, ref, rtol=0, atol=0)
        theta = rng.uniform(-.35, .35, len(head.correction))
        head.correction.copy_(torch.tensor(theta))
        actual = head(base, pt, xt, vt)
        delta = np.clip(b@theta, -LIMIT, LIMIT)*(phase == 0)
        np.testing.assert_allclose((actual[1]-reference[1]).numpy(), np.repeat(delta[:, None], 3, axis=1), atol=1e-12)
        for k in [0, 2]:
            torch.testing.assert_close(actual[k], reference[k], rtol=0, atol=0)
        torch.testing.assert_close(actual[1][pt.ne(0)], reference[1][pt.ne(0)], rtol=0, atol=0)
    # Real strict-past features vs recurrent state, plus reorder and future mutation.
    m = pd.read_parquet(META/'metadata_internal_validation.parquet')
    ids = m.entity_id.unique()[:5]
    subset = m[m.entity_id.isin(ids)&m.event_index.lt(60)].copy()
    expected, v = progress_features(subset, old['fallback'])
    expected = pd.DataFrame({'entity_id': subset.entity_id, 'event_index': subset.event_index,
                             'clock': expected[:, 0], 'position': expected[:, 1], 'valid': v})
    codec = ProgressClock(old['fallback']); count = 0
    for eid in ids:
        rows = subset[subset.entity_id.eq(eid)]
        target = expected[expected.entity_id.eq(eid)]
        state = codec.empty(1)
        for r, e in zip(rows.itertuples(), target.itertuples()):
            f, valid = codec.features(state)
            np.testing.assert_allclose(f[0], [e.clock, e.position], atol=1e-12)
            assert valid[0] == e.valid
            state = codec.advance(state, [r.gap]); count += 1
    changed = subset.copy(); changed.loc[changed.event_index.ge(30), 'gap'] *= 10
    later, lv = progress_features(changed, old['fallback'])
    original_x, ov = progress_features(subset, old['fallback'])
    np.testing.assert_allclose(later[subset.event_index.lt(30)], original_x[subset.event_index.lt(30)], rtol=0, atol=0)
    state = codec.empty(5)
    for t in range(10):
        state = codec.advance(state, rng.uniform(1, 10000, 5))
    order = np.array([3, 1, 4]); a, av = codec.features(state); b, bv = codec.features(state[order])
    np.testing.assert_array_equal(b, a[order]); np.testing.assert_array_equal(bv, av[order])
    # Exact parity with original all-event actual-prefix objective while unclipped.
    sim = Simulation('internal_validation', old, CFG['diagnostic_seeds'][0], 'event')
    theta = [-.01, .3]
    computed = sim.measure(basis_numpy('event_two', sim.x, sim.valid, limits)@theta)
    legacy = sim.run(theta)
    assert max(abs(computed[k]-legacy[k]) for k in computed) < 1e-12
    write(DOCS/'IMPLEMENTATION_CHECK.json', dict(weighted_w1_max_error=max(errors),
          numpy_torch_and_phase_parity=True, unmodified_gmr_at_zero=True,
          strict_past_events=count, reorder_and_future_invariance=True, legacy_uniform_objective_parity=True))


def fit_arm(arm):
    verify()
    old = payload()
    weighting = 'event' if arm == 'event_two' else 'customer'
    sim = Simulation('optimization', old, CFG['fit_noise_seed'], weighting)
    limits = json.loads((DOCS/'CLOCK_ENDPOINTS.json').read_text())['values']
    basis = basis_numpy(arm, sim.x, sim.valid, limits)
    reference = sim.measure(np.zeros(len(sim.nn)))
    curve = []; cache = {}; start = time.monotonic()
    def evaluate(theta):
        key = tuple(theta)
        if key not in cache:
            delta = np.clip(basis@theta, -LIMIT, LIMIT)
            metrics = sim.measure(delta)
            cache[key] = dict(objective=objective(metrics, reference), **metrics,
                              clipped_offset_fraction=float(np.mean(np.abs(basis@theta) > LIMIT)))
        record = dict(evaluation=len(curve), theta=list(map(float, theta)),
                      seconds=time.monotonic()-start, **cache[key])
        curve.append(record)
        if len(curve) % 16 == 0:
            print('FIT', arm, len(curve), round(record['objective'], 8), flush=True)
            pd.DataFrame(curve).to_csv(DOCS/f'{arm}_learning_curve.csv', index=False)
        return record['objective']
    size = basis.shape[1]
    bounds = CFG['bounds'] if size == 2 else [(-LIMIT, LIMIT)]*size
    evaluate(np.zeros(size))
    result = minimize(evaluate, np.zeros(size), method='Powell', bounds=bounds,
                      options=dict(maxfev=CFG['maxfev']-1, xtol=.001, ftol=.001))
    best = min(curve, key=lambda r: r['objective'])
    pd.DataFrame(curve).to_csv(DOCS/f'{arm}_learning_curve.csv', index=False)
    head = AdaptiveClock(1, arm)
    head.load_state_dict({**old['state_dict'], 'clock_limits': torch.tensor(limits, dtype=torch.float32),
                         'correction': torch.tensor(best['theta'], dtype=torch.float32)})
    for k, v in old['state_dict'].items():
        torch.testing.assert_close(head.state_dict()[k], v, rtol=0, atol=0)
    dest = OUT/arm; dest.mkdir(exist_ok=True)
    torch.save({**old, 'arm': arm, 'state_dict': head.state_dict()}, dest/'time_head.pt')
    record = dict(arm=arm, selected=best, initial=curve[0], support=sim.info,
                  weighting=weighting, evaluations=len(curve), distinct_evaluations=len(cache),
                  optimizer_success=bool(result.success), optimizer_message=str(result.message),
                  head_sha256=digest(dest/'time_head.pt'), original_gmr_buffers_unchanged=True,
                  new_native_argn_fits=0)
    if arm == 'event_two':
        prior = json.loads((ROOT/'artifacts/argn_clock_recheck_v1/budget_fit.json').read_text())['selected']
        record['previous_unclipped_converged_control'] = prior
        record['selected_coefficient_difference'] = [best['theta'][0]-prior['a'], best['theta'][1]-prior['b']]
    write(dest/'FIT_COMPLETE.json', record); write(DOCS/f'{arm}_fit.json', record)
    verify()
    print('FIT_COMPLETE', arm, len(curve), result.message, best['theta'], flush=True)
    return record


def initialize():
    full.verify(); assert not (OUT/'MANIFEST.json').exists()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    paths = [Path(__file__), Path(__file__).with_name('adaptive_clock.py'), DOCS/'PROTOCOL.md',
             ROOT/'artifacts/argn_clock_full_length_v1/MANIFEST.json',
             ROOT/'artifacts/argn_clock_recheck_v1/REVIEW_AND_NEXT_MODEL.md',
             META/'metadata_optimization.parquet', META/'metadata_internal_validation.parquet']
    write(OUT/'MANIFEST.json', dict(config=CFG, hashes={str(p): digest(p) for p in paths},
          baseline_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          created_utc=datetime.now(timezone.utc).isoformat(), test_events_read=False, local_only=True))
    old = payload(); sim = Simulation('optimization', old, CFG['fit_noise_seed'], 'event')
    limits = clock_endpoints(sim)
    write(DOCS/'CLOCK_ENDPOINTS.json', dict(values=limits, fit_split='optimization', quantiles=[.1, .9]))
    checks(old, limits); verify()
    write(OUT/'INITIALIZED.json', dict(checks_passed=True, clock_limits=limits))
    print('INITIALIZED', limits, flush=True)


def prepare():
    verify(); assert (OUT/'INITIALIZED.json').exists() and not (OUT/'FIT_COMPLETE.json').exists()
    with ProcessPoolExecutor(max_workers=CFG['max_parallel_fits']) as pool:
        fits = list(pool.map(fit_arm, ARMS))
    rows = []; old = payload()
    limits = json.loads((DOCS/'CLOCK_ENDPOINTS.json').read_text())['values']
    for ns in CFG['diagnostic_seeds']:
        for weighting in ['event', 'customer']:
            sim = Simulation('internal_validation', old, ns, weighting)
            old_fit = json.loads((ROOT/'docs/argn_clock_location_v1/clock_prefix_fit.json').read_text())['selected']
            for arm, theta in [('clock_joint', [0., 0.]), ('clock_prefix', [old_fit['a'], old_fit['b']])] + [(f['arm'], f['selected']['theta']) for f in fits]:
                kind = arm if arm in ARMS else 'event_two'
                delta = np.clip(basis_numpy(kind, sim.x, sim.valid, limits)@theta, -LIMIT, LIMIT)
                rows.append(dict(arm=arm, noise_seed=ns, weighting=weighting, **sim.measure(delta)))
    pd.DataFrame(rows).to_csv(DOCS/'internal_diagnostics.csv', index=False)
    verify()
    write(OUT/'FIT_COMPLETE.json', dict(new_calibration_fits=4, new_native_argn_fits=0,
          test_events_read=False, heads={arm: digest(OUT/arm/'time_head.pt') for arm in ARMS}))
    print('ALL_FITS_COMPLETE', flush=True)


def execute(mode, fs, arm):
    verify()
    with patch.object(study, 'OUT', OUT), patch.object(study, 'DOCS', DOCS), patch.object(study, 'CFG', CFG), \
         patch.object(study, 'verify', verify), patch.object(study, 'evolution_generation', adaptive_generation), \
         patch.object(study, '__file__', str(Path(__file__).resolve())):
        study.worker(fs, arm) if mode == 'worker' else study.dispatch()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['initialize', 'prepare', 'worker', 'dispatch'])
    p.add_argument('--seed', type=int, choices=CFG['fit_seeds']); p.add_argument('--arm', choices=ARMS)
    a = p.parse_args()
    if a.mode == 'initialize': initialize()
    elif a.mode == 'prepare': prepare()
    else: execute(a.mode, a.seed, a.arm)
