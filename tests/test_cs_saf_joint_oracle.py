from dataclasses import replace
import numpy as np
import pytest
import torch

from benchmarks.cs_saf_oracle import SemiMarkovCopyOracle
from benchmarks.temporal_coupling_v2 import BenchmarkConfig
from benchmarks import cs_saf_joint_oracle as joint


def oracle(duration=3):
    cfg = replace(BenchmarkConfig(scenario='joint_semimarkov_v2b', kappa=1),
                  duration_max=duration, duration_normal_mean=2.4 if duration == 3 else 14.,
                  duration_burst_mean=1.6 if duration == 3 else 6., pi_burst=.4 if duration == 3 else .3)
    return SemiMarkovCopyOracle(cfg, np.array([.4, 1.2, np.inf]))


def data(n=16, steps=8):
    rng = np.random.default_rng(2026091901)
    gap = torch.from_numpy(rng.exponential(1.5, (n, steps)).astype(np.float32)); gap[:, 0] = float('nan')
    marks = torch.from_numpy(rng.integers(3, 67, (n, steps)))
    return {'gap': gap, 'receiver': marks, 'numeric_value': torch.zeros((n, steps)),
            'valid_mask': torch.ones((n, steps), dtype=torch.bool), 'lengths': torch.full((n,), steps),
            'codes': torch.tensor([3]*(n//2)+[4]*(n-n//2))}


def dense_transition(o):
    d = o.config.duration_max; matrix = np.zeros((2*d, 2*d))
    for z in (0, 1):
        for remaining in range(1, d+1):
            row = z*d+remaining-1
            if remaining > 1: matrix[row, row-1] = 1
            else: matrix[row, (1-z)*d:(2-z)*d] = o.pmfs[1-z]
    return matrix


def test_stationary_residual_initialization():
    o = oracle(128); initial = o.initial(3)
    np.testing.assert_allclose(initial.sum((1, 2)), 1, atol=1e-12)
    np.testing.assert_allclose(o.advance(initial), initial, atol=1e-12, rtol=0)
    np.testing.assert_allclose(o.bin_mass.sum(1), 1, atol=1e-12)


@pytest.mark.parametrize('info', ['CONT', 'BIN', 'MODELINFO'])
def test_predictions_match_independent_dense_hidden_recursion(info):
    o = oracle(); x = data(4, 6); d = 3; transition = dense_transition(o)
    result = joint.predict(o, x, 1, info)
    for i in range(4):
        prior = o.initial(1).ravel()@transition
        for t in range(1, 6):
            gap = float(x['gap'][i, t]); b = np.searchsorted(o.upper, gap)
            density = np.exp(-gap/o.scales)/o.scales
            current = o.bin_mass[:, b] if info != 'CONT' else density
            if x['codes'][i] == 3: current = np.ones(2)
            state = prior.reshape(2, d).sum(1)*current; state /= state.sum()
            expected = state@o.repeat_by_state
            assert result[i, t] == pytest.approx(expected, abs=1e-12)
            equal = x['receiver'][i, t] == x['receiver'][i, t-1]
            emission = o.repeat_by_state if equal else 1-o.repeat_by_state
            if x['codes'][i] == 4: emission = emission*(o.bin_mass[:, b] if info == 'BIN' else density)
            posterior = prior*np.repeat(emission, d)
            prior = (posterior/posterior.sum())@transition


@pytest.mark.parametrize('binned', [False, True])
def test_joint_probability_factorization_equals_direct_state_marginal(binned):
    o = oracle(); rng = np.random.default_rng(81)
    prior = rng.random((5, 2, 3)); prior /= prior.sum((1, 2), keepdims=True)
    gaps = np.array([.1, .3, .9, 2., 4.]); previous = np.arange(5)
    probabilities, _ = joint.mark_probabilities(o, prior, gaps, previous, np.ones(5, bool), binned)
    for i in range(5):
        emission = o.bin_mass[:, np.searchsorted(o.upper, gaps[i])] if binned else np.exp(-gaps[i]/o.scales)/o.scales
        prior_state = prior[i].sum(1); pgap = prior_state@emission
        for m in range(64):
            likelihood = o.copy_by_state*(m == previous[i])+(1-o.copy_by_state)/64
            direct_joint = (prior_state*emission)@likelihood
            assert pgap*probabilities[i, m] == pytest.approx(direct_joint, abs=1e-12)


def test_model_information_matches_frozen_oracle_and_no_future_leakage():
    o = oracle(); x = data()
    p = joint.predict(o, x, 1, 'MODELINFO')
    for label in (0, 1):
        ids = np.flatnonzero(x['codes'].numpy() == label+3)
        old = o.filter_batch(x['gap'][ids].numpy(), x['receiver'][ids].numpy()-3, x['lengths'][ids].numpy(), active=bool(label))
        np.testing.assert_allclose(p[ids[old['entity_index']], old['event_index']], old['repeat_probability'], atol=1e-12, rtol=0)
    changed = {k: v.clone() for k, v in x.items()}; changed['receiver'][:, 4:] = 3; changed['gap'][:, 5:] = 9
    np.testing.assert_array_equal(joint.predict(o, changed, 1, 'MODELINFO')[:, :5], p[:, :5])


@pytest.mark.parametrize('mode', joint.MODES)
def test_sampler_replay_and_fixed_input_contract(mode):
    o = oracle(); x = data(); original = {k: v.clone() for k, v in x.items()}; reps = [.2, .7, 2.]
    gen, during = joint.sample(o, x, 1, reps, mode, 2026091800)
    again, repeat = joint.sample(o, x, 1, reps, mode, 2026091800)
    info = 'MODELINFO' if mode == 'FIX_MODELINFO' else ('BIN' if mode.endswith('BIN') else 'CONT')
    np.testing.assert_array_equal(during, joint.predict(o, gen, 1, info))
    np.testing.assert_array_equal(during, repeat)
    for key in gen: torch.testing.assert_close(gen[key], again[key], equal_nan=True, atol=0, rtol=0)
    for key in x: torch.testing.assert_close(x[key], original[key], equal_nan=True, atol=0, rtol=0)
    torch.testing.assert_close(gen['receiver'][:, 0], x['receiver'][:, 0], atol=0, rtol=0)
    torch.testing.assert_close(gen['numeric_value'], x['numeric_value'], atol=0, rtol=0)
    if mode in ('FIX_CONT', 'FIX_MODELINFO'): torch.testing.assert_close(gen['gap'], x['gap'], equal_nan=True, atol=0, rtol=0)
    if mode.endswith('BIN'):
        b = np.searchsorted(o.upper, gen['gap'].numpy()[:, 1:])
        np.testing.assert_array_equal(gen['gap'].numpy()[:, 1:], np.asarray(reps, np.float32)[b])


def test_independent_null_mark_process_ignores_gap_values():
    o = oracle(); x = data(); x['codes'][:] = 3
    first = joint.predict(o, x, 1, 'CONT'); x['gap'][:, 1:] = 100
    np.testing.assert_array_equal(joint.predict(o, x, 1, 'CONT'), first)
    np.testing.assert_array_equal(joint.predict(o, x, 1, 'BIN'), first)


def test_joint_samplers_against_analytic_stationary_moments():
    from experiments import cs_saf_rollout_audit as parent
    from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
    import yaml
    payload, _, _, _ = parent.load_cell(.05, 1)
    support = SAFTensorizerState.from_dict(payload['tensorizer_state']).gap_support
    cfg = BenchmarkConfig.from_mapping(yaml.safe_load((parent.ROOT/'configs/benchmark_v2/full_v2_5.yaml').read_text()), 'joint_semimarkov_v2b', 1)
    o = SemiMarkovCopyOracle(cfg, np.asarray(support.upper_bounds, np.float32).astype(float))
    result = joint.monte_carlo_gate(o, support.representatives, n_per_group=2048, seed=2026091901)
    assert len(result) == 4 and all(item['PASS'] for item in result.values())
