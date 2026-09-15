from dataclasses import replace

import numpy as np

from benchmarks.semi_markov import calibrated_duration_pmf, residual_pmf
from benchmarks.temporal_coupling_v2 import (
    BenchmarkConfig,
    generate_benchmark,
    generate_receiver_only,
    generate_single_row_audit,
    transition_matrix,
)


SMALL = BenchmarkConfig(n_train=2000, n_test=500, min_length=16, max_length=24)


def test_transition_rows_sum_to_one():
    assert np.allclose(transition_matrix(0.3, 0.9).sum(1), 1)


def test_stationary_distribution_exact():
    pi = np.array([0.7, 0.3])
    assert np.allclose(pi @ transition_matrix(0.3, 0.7), pi)


def test_kappa_zero_same_transition_for_labels():
    assert np.array_equal(transition_matrix(0.3, 0), transition_matrix(0.3, 0))


def test_kappa_controls_persistence_not_marginal():
    low, high = transition_matrix(0.3, 0), transition_matrix(0.3, 0.9)
    assert high.trace() > low.trace()
    assert np.allclose(np.array([0.7, 0.3]) @ high, [0.7, 0.3])


def test_right_padding_contract():
    bundle = generate_benchmark(SMALL, 42)
    assert np.array_equal(bundle.train.valid_mask.sum(1), bundle.train.lengths)
    assert np.all(bundle.train.x_num[~bundle.train.valid_mask] == 0)


def test_generation_deterministic_for_same_seed():
    a, b = generate_benchmark(SMALL, 42), generate_benchmark(SMALL, 42)
    assert np.array_equal(a.train.x_num, b.train.x_num)
    assert np.array_equal(a.train.dt_bin, b.train.dt_bin)


def test_generation_changes_for_different_seed():
    a, b = generate_benchmark(SMALL, 42), generate_benchmark(SMALL, 43)
    assert not np.array_equal(a.train.x_num, b.train.x_num)


def test_semimarkov_duration_mean_calibration():
    _, pmf = calibrated_duration_pmf(6, 0.6, 128)
    assert abs(pmf @ np.arange(1, 129) - 6) < 1e-7


def test_semimarkov_residual_matches_survival_formula():
    _, pmf = calibrated_duration_pmf(14, 0.6, 128)
    residual = residual_pmf(pmf)
    assert np.isclose(residual.sum(), 1)
    assert np.all(residual >= 0)


def test_shared_binning_across_kappa():
    reference = generate_benchmark(SMALL, 42)
    other = generate_benchmark(replace(SMALL, kappa=1, bin_edges=np.asarray(reference.metadata["bin_edges"])), 42)
    assert other.metadata["bin_edges"] == reference.metadata["bin_edges"]


def test_no_fraud_exclusive_category():
    bundle = generate_benchmark(replace(SMALL, kappa=1), 42)
    cats0 = set(bundle.train.x_cat[bundle.train.valid_mask & (bundle.train.y_entity[:, None] == 0), 0])
    cats1 = set(bundle.train.x_cat[bundle.train.valid_mask & (bundle.train.y_entity[:, None] == 1), 0])
    assert len(cats0 & cats1) > 50


def test_v2b_cross_channel_alignment_increases_with_kappa():
    base = replace(SMALL, scenario="joint_semimarkov_v2b", n_train=5000)
    low, high = generate_benchmark(base, 42), generate_benchmark(replace(base, kappa=1), 42)
    def alignment(bundle):
        latent = bundle.metadata["train_latent"]
        valid = bundle.train.valid_mask
        label = bundle.train.y_entity
        return np.mean(latent["gap_state"][valid & (label[:, None] == 1)] * latent["receiver_repeat"][valid & (label[:, None] == 1)])
    assert alignment(high) > alignment(low)


def test_receiver_only_monte_carlo_uses_production_path():
    for scenario in ("markov_persistence_v2a", "joint_semimarkov_v2b"):
        config = replace(SMALL, scenario=scenario, kappa=1, n_train=128)
        full = generate_benchmark(config, 42).train
        receiver = generate_receiver_only(
            config, n_entities=config.n_train, seed=42, split_id=0
        )
        assert np.array_equal(full.y_entity, receiver["labels"])
        assert np.array_equal(full.lengths, receiver["lengths"])
        assert np.array_equal(full.x_cat[..., 0], receiver["categories"])


def test_single_row_audit_matches_production_features():
    for scenario in ("markov_persistence_v2a", "joint_semimarkov_v2b"):
        base = replace(SMALL, scenario=scenario, kappa=1.0, n_train=128)
        reference = generate_benchmark(base, 42)
        config = replace(
            base,
            bin_edges=np.asarray(reference.metadata["bin_edges"]),
        )
        full = generate_benchmark(config, 42).train
        audit = generate_single_row_audit(
            config,
            n_entities=128,
            seed=42,
            split_id=0,
            position_seed=99,
        )
        rng = np.random.default_rng(99)
        positions = np.asarray(
            [rng.integers(0, int(length)) for length in full.lengths]
        )
        rows = np.arange(len(full.lengths))
        expected = np.column_stack(
            [
                full.x_num[rows, positions, 0],
                full.dt_bin[rows, positions],
                full.x_cat[rows, positions, 0],
            ]
        )
        np.testing.assert_allclose(audit["features"], expected)
        np.testing.assert_array_equal(audit["labels"], full.y_entity)
