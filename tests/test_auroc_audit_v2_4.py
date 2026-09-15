from dataclasses import replace

import numpy as np

from benchmarks.auroc_audit_v2_4 import (
    generate_stationary_row_audit,
    inject_leakage,
    leakage_operator_for_trial,
)
from benchmarks.temporal_coupling_v2 import BenchmarkConfig


BASE = BenchmarkConfig(
    scenario="joint_semimarkov_v2b",
    kappa=1.0,
    fraud_rate=0.2,
)


def test_stationary_row_audit_has_fixed_continuous_marginals():
    audit = generate_stationary_row_audit(
        BASE,
        n_entities=100_000,
        audit_seed=1000,
        split_id=0,
    )
    x, y = audit["features"], audit["labels"]
    assert x.shape == (100_000, 3)
    assert x.dtype == np.float32
    assert abs(y.mean() - BASE.fraud_rate) < 0.01
    assert abs(x[:, 0].mean() - BASE.amount_mean) < 0.02
    expected_gap = (
        BASE.pi_burst * BASE.gap_burst_scale
        + (1 - BASE.pi_burst) * BASE.gap_normal_scale
    )
    assert abs(x[:, 1].mean() - expected_gap) < 0.03
    assert set(np.unique(x[:, 2])).issubset(
        set(range(BASE.n_receiver_categories))
    )


def test_stationary_row_marginal_does_not_change_with_kappa():
    low = generate_stationary_row_audit(
        replace(BASE, kappa=0.0),
        n_entities=20_000,
        audit_seed=7,
        split_id=1,
    )
    high = generate_stationary_row_audit(
        replace(BASE, kappa=1.0),
        n_entities=20_000,
        audit_seed=7,
        split_id=1,
    )
    # Streams differ by cell, but population row-marginal contracts agree.
    for index in (0, 1):
        assert abs(low["features"][:, index].mean() - high["features"][:, index].mean()) < 0.06
    assert abs(low["labels"].mean() - high["labels"].mean()) < 0.02


def test_fixed_operator_schedule_is_independent_of_numeric_seed():
    amount = [
        leakage_operator_for_trial(
            feature="amount",
            magnitude=0.02,
            validation_trial_ordinal=index,
        ).target_label
        for index in range(4)
    ]
    receiver = [
        (
            leakage_operator_for_trial(
                feature="receiver_category",
                magnitude=0.05,
                validation_trial_ordinal=index,
            ).target_label,
            leakage_operator_for_trial(
                feature="receiver_category",
                magnitude=0.05,
                validation_trial_ordinal=index,
            ).receiver_category,
        )
        for index in range(4)
    ]
    assert amount == [1, 0, 1, 0]
    assert receiver == [(1, 0), (1, 31), (0, 1), (0, 63)]


def test_receiver_injection_has_exact_realised_probability_change():
    audit = generate_stationary_row_audit(
        BASE,
        n_entities=10_000,
        audit_seed=2000,
        split_id=0,
    )
    operator = leakage_operator_for_trial(
        feature="receiver_category",
        magnitude=0.02,
        validation_trial_ordinal=1,
    )
    changed, metadata = inject_leakage(
        audit["features"],
        audit["labels"],
        operator=operator,
        config=BASE,
        audit_seed=2000,
        split_id=0,
    )
    expected = metadata["changed_rows"] / metadata["target_class_rows"]
    assert metadata["realised_probability_change"] == expected
    assert np.array_equal(audit["features"][:, :2], changed[:, :2])
    assert not np.array_equal(audit["features"][:, 2], changed[:, 2])


def test_continuous_injection_is_deterministic_and_uses_fixed_anchor():
    audit = generate_stationary_row_audit(
        BASE,
        n_entities=10_000,
        audit_seed=2000,
        split_id=2,
    )
    operator = leakage_operator_for_trial(
        feature="gap",
        magnitude=0.05,
        validation_trial_ordinal=0,
    )
    first, first_meta = inject_leakage(
        audit["features"],
        audit["labels"],
        operator=operator,
        config=BASE,
        audit_seed=2000,
        split_id=2,
    )
    second, second_meta = inject_leakage(
        audit["features"],
        audit["labels"],
        operator=operator,
        config=BASE,
        audit_seed=2000,
        split_id=2,
    )
    np.testing.assert_array_equal(first, second)
    assert first_meta == second_meta
    expected_anchor = -BASE.gap_normal_scale * np.log(0.01)
    assert first_meta["replacement_anchor"] == expected_anchor
