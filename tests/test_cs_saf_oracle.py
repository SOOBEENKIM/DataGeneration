from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from benchmarks.cs_saf_oracle import (
    SemiMarkovCopyOracle, decide_oracle, repeat_probability, summarize_context,
)
from benchmarks.temporal_coupling_v2 import BenchmarkConfig, _split
from scripts.audit_cs_saf_oracle import load_train_cell


def config(**kwargs):
    return replace(BenchmarkConfig(scenario="joint_semimarkov_v2b", kappa=1), **kwargs)


def test_copy_is_not_observed_repeat():
    np.testing.assert_allclose(repeat_probability(np.array([0.0, 0.5, 1.0]), 4), [0.25, 0.625, 1.0])


def test_bin_likelihood_integrates_density_instead_of_evaluating_representative():
    oracle = SemiMarkovCopyOracle(config(), np.array([0.5, 2.0, np.inf]))
    np.testing.assert_allclose(oracle.bin_mass.sum(1), [1, 1])
    prior = 0.3
    normal_mass = 1-np.exp(-0.5/2.0)
    burst_mass = 1-np.exp(-0.5/0.5)
    posterior = prior*burst_mass/((1-prior)*normal_mass+prior*burst_mass)
    assert oracle.copy_curve(np.array([prior]), True)[0, 0] == pytest.approx(0.05+0.85*posterior)
    point_posterior = (prior/0.5)/((1-prior)/2.0+prior/0.5)
    assert abs(posterior-point_posterior) > 0.05


def test_residual_initialization_matches_duration_inspection_paradox():
    oracle = SemiMarkovCopyOracle(config(duration_max=3), np.array([1.0, np.inf]))
    oracle.pmfs = np.array([[0.2, 0.3, 0.5], [0.4, 0.5, 0.1]])
    state = oracle.initial(1)[0]
    for z in (0, 1):
        pmf = oracle.pmfs[z]
        expected = np.zeros(3)
        for duration in range(1, 4):
            size_biased = duration*pmf[duration-1]/(pmf @ np.arange(1, 4))
            for age in range(duration):
                expected[duration-age-1] += size_biased/duration
        np.testing.assert_allclose(state[z]/state[z].sum(), expected)


def test_filter_matches_independent_dense_hidden_state_recursion():
    oracle = SemiMarkovCopyOracle(config(duration_max=3), np.array([0.4, 1.2, np.inf]))
    # Construct the transition by enumerating (state, residual), independently
    # of the vectorized shift/renewal implementation.
    transition = np.zeros((6, 6))
    for state in (0, 1):
        for remaining in (1, 2, 3):
            index = state*3+remaining-1
            if remaining > 1:
                transition[index, index-1] = 1
            else:
                for duration in (1, 2, 3):
                    transition[index, (1-state)*3+duration-1] = oracle.pmfs[1-state, duration-1]
    gaps = np.array([[np.nan, 0.2, 2.0, 0.6]])
    marks = np.array([[1, 1, 3, 3]])
    result = oracle.filter_batch(gaps, marks, np.array([4]), active=True)
    prior = oracle.initial(1).reshape(-1) @ transition
    expected = []
    for t in (1, 2, 3):
        burst_prior = prior[3:].sum()
        expected.append(burst_prior)
        repeat = marks[0, t] == marks[0, t-1]
        emission = np.exp(-gaps[0, t]/oracle.scales)/oracle.scales
        emission *= oracle.repeat_by_state if repeat else 1-oracle.repeat_by_state
        posterior = prior*np.repeat(emission, 3)
        prior = (posterior/posterior.sum()) @ transition
    np.testing.assert_allclose(result["prior_burst"], expected, atol=1e-12)


def test_no_current_mark_or_future_leakage_and_first_gap_excluded():
    oracle = SemiMarkovCopyOracle(config(), np.array([0.3, 2, np.inf]))
    gaps = np.array([[np.nan, 0.1, 0.8, 3.0]])
    marks = np.array([[0, 0, 1, 1]])
    before = oracle.filter_batch(gaps, marks, np.array([4]), active=True)
    changed = marks.copy(); changed[:, 2:] = 0
    future = gaps.copy(); future[:, 3] = 100
    after = oracle.filter_batch(future, changed, np.array([4]), active=True)
    np.testing.assert_array_equal(before["copy_probability"][:2], after["copy_probability"][:2])
    assert before["prior_burst"][2] != after["prior_burst"][2]
    with pytest.raises(ValueError, match="first gap"):
        oracle.filter_batch(np.nan_to_num(gaps), marks, np.array([4]), active=True)


def test_null_response_is_exactly_zero_while_active_response_is_material():
    oracle = SemiMarkovCopyOracle(config(), np.array([0.1, 1, 5, np.inf]))
    priors = np.array([0.1, 0.3, 0.8])
    assert np.ptp(oracle.copy_curve(priors, False), axis=1).max() == 0
    assert np.ptp(oracle.copy_curve(priors, True), axis=1).min() > 0.05


@pytest.mark.parametrize("kappa,label", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_oracle_calibrates_against_production_dgp(kappa, label):
    cfg = config(kappa=kappa, fraud_rate=label, min_length=16, max_length=16)
    values, latent = _split(cfg, 1000, 7123, 0)
    gaps = latent["raw_gap"].astype(float); gaps[:, 0] = np.nan
    oracle = SemiMarkovCopyOracle(cfg, np.array([0.1, 0.3, 0.7, 1.5, 3, 6, np.inf]))
    predictions = oracle.filter_batch(gaps, values["x_cat"][:, :, 0], values["lengths"], active=bool(kappa*label))
    summary = summarize_context(predictions, 1000, cfg.n_receiver_categories)
    assert abs(summary["repeat_calibration_mean_residual"]) < max(0.025, 5*summary["repeat_calibration_entity_cluster_se"])


def test_reader_requests_only_train_rows(tmp_path, monkeypatch):
    import json
    static = pd.DataFrame({"entity_id": ["a", "b", "c"], "entity_label": [0, 1, 1]})
    splits = pd.DataFrame({"entity_id": ["a", "b", "c"], "split": ["train", "validation", "test"]})
    events = pd.DataFrame([
        {"entity_id": entity, "event_id": f"{entity}{i}", "event_index": i,
         "timestamp": float(i), "gap": np.nan if i == 0 else 1.0,
         "receiver_or_mark": "receiver-0", "amount_or_numeric_value": 1.0}
        for entity in ["a", "b", "c"] for i in range(2)])
    schema = {"dataset_id": "controlled_coupling_toy", "time_representation": "absolute",
              "timestamp_unit": "step", "static_context_columns": ["entity_label"],
              "auxiliary_numeric_columns": [], "auxiliary_categorical_columns": []}
    (tmp_path/"schema.json").write_text(json.dumps(schema))
    for name, frame in [("static_context", static), ("entity_splits", splits), ("events", events)]:
        frame.to_parquet(tmp_path/f"{name}.parquet", index=False)
    real_read = pd.read_parquet
    seen = []
    def guarded_read(path, **kwargs):
        if path.name in ("events.parquet", "static_context.parquet"):
            assert kwargs["filters"] == [("entity_id", "in", ["a"])]
            seen.append(path.name)
        return real_read(path, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    loaded = load_train_cell(tmp_path)
    assert loaded.events.entity_id.unique().tolist() == ["a"]
    assert set(seen) == {"events.parquet", "static_context.parquet"}


def test_gate_does_not_pass_missing_context_or_calibration_failure():
    criteria = {"active_cell_min_mean_range": .05, "noncausal_cell_max_mean_range": .05,
                "active_to_largest_noncausal_min_ratio": 2, "minimum_entities_per_context": 100,
                "repeat_calibration_absolute_tolerance": .02, "repeat_calibration_cluster_se_multiplier": 5}
    cells = {str(k): {str(y): {"mean_copy_probability_range": .5 if k*y else 0,
                              "entity_count": 200, "repeat_calibration_mean_residual": 0.0,
                              "repeat_calibration_entity_cluster_se": .001}
                      for y in (0, 1)} for k in (0, 1)}
    assert decide_oracle(cells, criteria)["decision"] == "PASS"
    cells["1"]["1"]["repeat_calibration_mean_residual"] = .2
    assert decide_oracle(cells, criteria)["decision"] == "FAIL"
    del cells["0"]["1"]
    with pytest.raises(ValueError, match="mandatory"):
        decide_oracle(cells, criteria)
