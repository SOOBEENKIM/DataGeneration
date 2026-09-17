"""Functional gates for a fixed-checkpoint measurement, not training tests."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch

from benchmarks.cs_saf_oracle import SemiMarkovCopyOracle
from benchmarks.temporal_coupling_v2 import BenchmarkConfig
from benchmarks.cof_seqgen_saf_metrics import MetricState, _repeat_curve_error, _mutual_information
from experiments.cs_saf_rollout_audit import (
    load_cell, folder_for, make_model, select, make_oracle, oracle_arrays,
    initial_hidden, stream_event, rollout, evaluate_histories, quantize,
    inverse_cdf, reference_metrics, summarize_predictions, state_digest,
)


@pytest.fixture(scope='module')
def setup():
    torch.set_num_threads(1)
    payload, edges, panel, _ = load_cell(.05, 1)
    ids = torch.tensor([0, 1, 2, 128, 129, 130])
    small = select(panel, ids)
    model = make_model(payload, 'E', torch.device('cpu'), 0)
    checkpoint = torch.load(folder_for(.05, 1, 0, 'E')/'checkpoint_best.pt', weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint['model_state']); model.eval()
    return model, small, make_oracle(model, 1), edges


def test_streamed_encoder_matches_original_and_is_strict_past(setup):
    model, data, _, _ = setup
    with torch.no_grad():
        native = model.encoder(data['gap'], data['receiver'], data['numeric_value'], data['valid_mask'], static_categorical=(data['codes'],))
        hidden, state = initial_hidden(model, data['codes']); steps = [hidden]
        for t in range(data['gap'].shape[1]-1):
            hidden, state = stream_event(model, data['gap'][:, t], data['receiver'][:, t], data['numeric_value'][:, t], data['valid_mask'][:, t], state)
            steps.append(hidden)
        torch.testing.assert_close(torch.stack(steps, 1), native, atol=2e-6, rtol=0)
        changed = {k: v.clone() for k, v in data.items()}
        changed['receiver'][:, 5:] = 3
        changed['numeric_value'][:, 5:] = 99
        changed['gap'][:, 5:] = 99
        out = model.encoder(changed['gap'], changed['receiver'], changed['numeric_value'], changed['valid_mask'], static_categorical=(data['codes'],))
        torch.testing.assert_close(out[:, :6], native[:, :6], atol=0, rtol=0)


def test_observable_oracle_matches_existing_filter_and_has_no_future_leakage(setup):
    _, data, oracle, _ = setup
    for kappa in (0, 1):
        priors, repeats = oracle_arrays(oracle, data, kappa)
        for label in (0, 1):
            ids = np.flatnonzero(data['codes'].numpy() == label+3)
            original = oracle.filter_batch(data['gap'][ids].numpy(), data['receiver'][ids].numpy()-3,
                                           data['lengths'][ids].numpy(), active=bool(kappa*label))
            for key, result in (('prior_burst', priors), ('repeat_probability', repeats)):
                np.testing.assert_allclose(result[ids[original['entity_index']], original['event_index']], original[key], atol=1e-12, rtol=0)
        if not kappa:
            binned = oracle_arrays(oracle, data, kappa, binned_history=True)
            np.testing.assert_array_equal(binned[1], repeats)
    changed = {k: v.clone() for k, v in data.items()}
    changed['receiver'][:, 5:] = 3; changed['gap'][:, 6:] = 10
    np.testing.assert_array_equal(oracle_arrays(oracle, changed, 1)[1][:, :6], repeats[:, :6])


def test_binned_oracle_matches_independent_dense_transition():
    cfg = replace(BenchmarkConfig(scenario='joint_semimarkov_v2b', kappa=1), duration_max=3)
    oracle = SemiMarkovCopyOracle(cfg, np.array([.4, 1.2, np.inf]))
    data = {'gap': torch.tensor([[float('nan'), .2, 2., .6]]),
            'receiver': torch.tensor([[4, 4, 6, 6]]), 'codes': torch.tensor([4]),
            'lengths': torch.tensor([4]), 'valid_mask': torch.ones((1, 4), dtype=torch.bool)}
    transition = np.zeros((6, 6))
    for state in (0, 1):
        for remaining in (1, 2, 3):
            index = state*3+remaining-1
            if remaining > 1: transition[index, index-1] = 1
            else:
                for duration in (1, 2, 3):
                    transition[index, (1-state)*3+duration-1] = oracle.pmfs[1-state, duration-1]
    prior = oracle.initial(1).ravel()@transition; expected = []
    for t in (1, 2, 3):
        b = np.searchsorted(oracle.upper, data['gap'][0, t])
        expected.append(prior[3:].sum())
        equal = data['receiver'][0, t] == data['receiver'][0, t-1]
        likelihood = oracle.bin_mass[:, b]*(oracle.repeat_by_state if equal else 1-oracle.repeat_by_state)
        posterior = prior*np.repeat(likelihood, 3)
        prior = (posterior/posterior.sum())@transition
    observed, _ = oracle_arrays(oracle, data, 1, binned_history=True)
    np.testing.assert_allclose(observed[0, 1:], expected, atol=1e-12, rtol=0)


def test_inverse_cdf_reserved_zero_and_exact_mass():
    u = (torch.arange(10000, dtype=torch.float64)+.5)/10000
    p = torch.tensor([0., 0., 0., .1, .2, .7], dtype=torch.float64).expand(10000, -1)
    samples = inverse_cdf(p, u)
    np.testing.assert_array_equal(np.bincount(samples, minlength=6), [0, 0, 0, 1000, 2000, 7000])


@pytest.mark.parametrize('arm', ['OBS', 'QUANT', 'GAP', 'FULL'])
def test_rollout_controls_probabilities_and_frozen_checkpoint(setup, arm):
    model, data, oracle, _ = setup; before = state_digest(model)
    gen, during = rollout(model, data, arm, 2026091800, torch.device('cpu'))
    repeated, twice = rollout(model, data, arm, 2026091800, torch.device('cpu'))
    np.testing.assert_array_equal(during, twice)
    for key in gen: torch.testing.assert_close(gen[key], repeated[key], equal_nan=True, atol=0, rtol=0)
    for key in ('receiver', 'numeric_value'): torch.testing.assert_close(gen[key][:, 0], data[key][:, 0], atol=0, rtol=0)
    if arm != 'FULL': torch.testing.assert_close(gen['numeric_value'], data['numeric_value'], atol=0, rtol=0)
    if arm == 'OBS': torch.testing.assert_close(gen['gap'], data['gap'], equal_nan=True, atol=0, rtol=0)
    if arm == 'QUANT': torch.testing.assert_close(gen['gap'], quantize(model, data['gap']), equal_nan=True, atol=0, rtol=0)
    if arm in ('QUANT', 'GAP', 'FULL'):
        torch.testing.assert_close(quantize(model, gen['gap']), gen['gap'], equal_nan=True, atol=0, rtol=0)
    pred = evaluate_histories(model, gen, oracle, 1, model.reference_probabilities, torch.device('cpu'))
    mask = gen['valid_mask'].numpy().copy(); mask[:, 0] = False
    np.testing.assert_allclose(during[mask], pred['repeat'][mask], atol=2e-6, rtol=0)
    assert state_digest(model) == before


def test_aggregate_metric_matches_independent_pandas_implementation(setup):
    model, data, oracle, edges = setup
    gen, _ = rollout(model, data, 'FULL', 2026091800, torch.device('cpu'))
    pred = evaluate_histories(model, gen, oracle, 1, model.reference_probabilities, torch.device('cpu'), grid=True)
    reference = reference_metrics(data, edges)
    result = summarize_predictions(gen, pred, reference, edges)
    def frame(x, label):
        valid = x['valid_mask'].numpy() & (x['codes'].numpy() == label+3)[:, None]
        row, col = np.where(valid)
        return pd.DataFrame({'entity_id': row, 'event_index': col, 'gap': x['gap'].numpy()[valid],
                             'receiver_or_mark': x['receiver'].numpy()[valid].astype(str)})
    for label in (0, 1):
        state = MetricState(gap_bin_edges=tuple(edges[str(label)]), mark_groups=(), gap_scale=1, amount_scale=1, length_scale=1, fit_entity_count=6)
        ref, can = frame(data, label), frame(gen, label)
        assert result[str(label)]['empirical_repeat_L1'] == pytest.approx(_repeat_curve_error(ref, can, state), abs=1e-12)
        assert result[str(label)]['mi_error'] == pytest.approx(abs(_mutual_information(ref, state)-_mutual_information(can, state)), abs=1e-12)
        scores = result[str(label)]['conditional']
        assert scores['grid_mark_TV'] == pytest.approx(scores['grid_repeat_L1']+scores['grid_excess_TV'], abs=1e-12)
        assert scores['grid_excess_TV'] >= -2e-6
