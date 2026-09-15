import numpy as np

from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_benchmark
from benchmarks.types import SyntheticBatch
from eval.model_guards_v2_5 import (
    calibrate_row_guard_thresholds,
    evaluate_row_guards,
)
from generators.empirical_conditional_block import EmpiricalConditionalBlock
from generators.sampling_plan import SamplingPlan


def setup_problem():
    bundle = generate_benchmark(
        BenchmarkConfig(
            scenario="joint_semimarkov_v2b",
            kappa=1.0,
            n_train=160,
            n_test=60,
            min_length=5,
            max_length=8,
            fraud_rate=0.5,
            n_gap_bins=8,
            n_receiver_categories=8,
        ),
        42,
    )
    tau = np.asarray(bundle.metadata["tau"])
    return bundle, tau


def test_row_guard_calibration_is_train_only_max_order_statistic():
    bundle, tau = setup_problem()
    thresholds = calibrate_row_guard_thresholds(
        bundle.train,
        tau=tau,
        entity_count=50,
        trials=4,
        seed=100,
        receiver_practical_margin=0.02,
    )
    assert thresholds.fit_split == "train"
    assert thresholds.calibration_trials == 4
    assert thresholds.cutoff_rule == "maximum_rank_n_of_n_no_interpolation"
    assert thresholds.amount_ks > 0
    assert thresholds.receiver_max_abs_signed_frequency >= 0.02


def test_large_label_conditional_receiver_leak_fails_hard_guard():
    bundle, tau = setup_problem()
    thresholds = calibrate_row_guard_thresholds(
        bundle.train,
        tau=tau,
        entity_count=50,
        trials=4,
        seed=100,
        receiver_practical_margin=0.02,
    )
    plan = SamplingPlan.from_train_policy(
        bundle.train,
        entity_count=60,
        seed=200,
    )
    reference = EmpiricalConditionalBlock("full")
    reference.fit(bundle.train, config={}, seed=1)
    sample = reference.sample(plan, seed=2)
    categories = sample.x_cat.copy()
    for label in (0, 1):
        selected = (
            sample.valid_mask
            & (sample.y_entity[:, None] == label)
        )
        categories[..., 0][selected] = label
    leaked = SyntheticBatch(
        sample.x_num,
        sample.dt_bin,
        categories,
        sample.valid_mask,
        sample.y_entity,
        sample.lengths,
    )
    report = evaluate_row_guards(
        bundle.test,
        leaked,
        tau=tau,
        receiver_categories=8,
        thresholds=thresholds,
    )
    assert report["status"] == "FAIL"
    assert report["checks"][
        "receiver_max_abs_signed_frequency"
    ] == "FAIL"
