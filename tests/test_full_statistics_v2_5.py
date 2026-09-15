import numpy as np

from eval.full_statistics_v2_5 import (
    PRIMARY_COMPARATORS,
    SECONDARY_COMPARATORS,
    analyze_full_experiment,
    floor_proximity_interpretation,
    hedges_g_baseline_minus_cof,
    holm_adjust,
)


PLAN_HASH = "0" * 64


def generator_records(values, *, invalid_seed=None, plan_hash=PLAN_HASH):
    output = {}
    for seed, value in enumerate(values, 1):
        output[seed] = {
            "status": "COMPLETE",
            "sampling_plan_hash": plan_hash,
            "hard_guards": {"mask": "PASS", "support": "PASS"},
            "association_recovery_error": value,
        }
    if invalid_seed is not None:
        output[invalid_seed]["hard_guards"]["support"] = "FAIL"
    return output


def complete_family():
    records = {
        "cof_seqgen": generator_records(
            [0.010, 0.012, 0.011, 0.013, 0.009]
        ),
    }
    for index, generator in enumerate(
        PRIMARY_COMPARATORS + SECONDARY_COMPARATORS
    ):
        records[generator] = generator_records(
            np.asarray([0.050, 0.052, 0.051, 0.053, 0.049])
            + index * 0.001
        )
    records["block_2"] = generator_records(
        [0.05, 0.06, 0.055, 0.05, 0.06]
    )
    records["block_4"] = generator_records(
        [0.04, 0.05, 0.045, 0.04, 0.05]
    )
    records["block_8"] = generator_records(
        [0.03, 0.04, 0.035, 0.03, 0.04]
    )
    records["full_sequence_reference"] = generator_records(
        [0.005, 0.006, 0.004, 0.005, 0.006]
    )
    return records


def test_full_analysis_uses_exact_primary_and_secondary_holm_families():
    result = analyze_full_experiment(
        complete_family(),
        bootstrap_resamples=200,
        bootstrap_seed=7,
        c0_mean_error=0.008,
    )
    assert result["status"] == "COMPLETE"
    assert tuple(result["primary_family"]) == PRIMARY_COMPARATORS
    assert tuple(result["secondary_family"]) == SECONDARY_COMPARATORS
    assert result["primary_endpoint"] == (
        "continuous_association_recovery_error"
    )
    assert result["sole_primary_endpoint"] is True
    assert result["support_diagnostic_bins"] == [4, 8]
    assert result["c2_supported"] is True
    for generator in PRIMARY_COMPARATORS:
        comparison = result["pairwise"][generator]
        assert comparison["mean_effect"] > 0
        assert comparison["hedges_g"] > 0.8
        assert comparison["holm_adjusted_p"] < 0.05
        assert comparison["multiplicity_family"] == "C2_primary"
        assert "unpaired_bootstrap_95_ci" in comparison
    for generator in SECONDARY_COMPARATORS:
        assert result["pairwise"][generator][
            "multiplicity_family"
        ] == "secondary"
    assert result["pairwise"]["full_sequence_reference"][
        "reference_only"
    ] is True


def test_any_primary_failure_uses_frozen_c2_not_supported_decision():
    records = complete_family()
    records["empirical_iid"] = generator_records(
        [0.009, 0.011, 0.010, 0.012, 0.008]
    )
    result = analyze_full_experiment(records, bootstrap_resamples=100)
    assert result["c2_supported"] is False
    assert result["c2_decision"] == (
        "이번 사전등록 실험에서 C2는 지지되지 않음"
    )


def test_one_failed_guard_invalidates_generator_without_nan_exclusion():
    records = complete_family()
    records["plug_in_hmm"] = generator_records(
        [0.03, 0.04, 0.035, 0.03, 0.04],
        invalid_seed=3,
    )
    result = analyze_full_experiment(
        records,
        bootstrap_resamples=20,
    )
    assert result["status"] == "INVALID"
    assert result["summaries"]["plug_in_hmm"]["status"] == "INVALID"
    assert result["summaries"]["plug_in_hmm"][
        "mean_association_recovery_error"
    ] is None
    assert "plug_in_hmm" in result["reason"]


def test_partial_five_seed_set_is_not_aggregated():
    records = complete_family()
    del records["neural_sequence"][5]
    result = analyze_full_experiment(
        records,
        bootstrap_resamples=20,
    )
    assert result["status"] == "INVALID"
    assert result["summaries"]["neural_sequence"]["status"] == "INCOMPLETE"


def test_sampling_plan_hash_mismatch_refuses_aggregation():
    records = complete_family()
    records["plug_in_hsmm"][5]["sampling_plan_hash"] = "1" * 64
    result = analyze_full_experiment(records, bootstrap_resamples=20)
    assert result["status"] == "INVALID"
    assert "SamplingPlan" in result["reason"]
    assert result["pairwise"] == {}


def test_floor_ratio_is_uninterpretable_at_or_below_fixed_epsilon():
    result = floor_proximity_interpretation(
        cof_mean_error=0.0,
        c0_mean_error=1e-6,
    )
    assert result["status"] == "RATIO_UNINTERPRETABLE"
    assert result["ratio"] is None
    assert result["floor_proximity_permitted"] is False
    permitted = floor_proximity_interpretation(
        cof_mean_error=0.0015,
        c0_mean_error=0.001,
    )
    assert permitted["floor_proximity_permitted"] is True


def test_holm_adjustment_is_monotone_and_hedges_direction_is_fixed():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.02})
    ordered = [adjusted[key] for key in ("a", "c", "b")]
    assert ordered == sorted(ordered)
    assert hedges_g_baseline_minus_cof(
        np.asarray([2, 3, 4, 5, 6], dtype=float),
        np.asarray([1, 2, 3, 4, 5], dtype=float),
    ) > 0
