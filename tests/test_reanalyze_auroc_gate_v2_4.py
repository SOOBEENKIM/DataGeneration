import numpy as np

from scripts.reanalyze_auroc_gate_v2_4 import (
    exact_binomial_interval,
    gaussian_upper_prediction,
    maximum_order_statistic,
    modeled_power,
    order_statistic,
    order_statistic_rank,
)


def test_order_statistic_rank_is_preregistered_noninterpolated_rank():
    # Historical ranks are retained for provenance but superseded for v2.4.
    assert order_statistic_rank(50, 0.95) == 49
    assert order_statistic_rank(100, 0.95) == 96
    value, rank = order_statistic(range(1, 101), 0.95)
    assert rank == 96
    assert value == 96


def test_superseding_step_b_threshold_is_rank_200_maximum():
    value, rank = maximum_order_statistic(range(1, 201))
    assert rank == 200
    assert value == 200


def test_gaussian_threshold_proxy_is_eight_dimension_upper_bound():
    rng = np.random.default_rng(42)
    deviations = rng.normal(0, 0.01, size=(100, 8))
    bound, rows, z_value = gaussian_upper_prediction(
        deviations,
        target_calibration_seeds=200,
        mean_confidence=0.95,
        sd_confidence=0.95,
    )
    assert len(rows) == 8
    assert bound == max(row["upper_prediction_bound"] for row in rows)
    assert z_value > 3.0
    assert bound > np.max(np.std(deviations, axis=0, ddof=1))


def test_modeled_power_scales_all_clean_noise_but_not_leakage_effect():
    clean_noise = np.array(
        [
            [-0.02, 0.00, 0.00, 0.00],
            [0.00, 0.02, 0.00, 0.00],
        ]
    )
    effect = np.full(4, 0.01)
    low_n = modeled_power(
        clean_noise=clean_noise,
        effect=effect,
        current_threshold=0.03,
        multiplier=1,
    )
    high_n = modeled_power(
        clean_noise=clean_noise,
        effect=effect,
        current_threshold=0.03,
        multiplier=100,
    )
    assert low_n == 0.0
    assert high_n == 1.0


def test_exact_binomial_interval_has_expected_boundaries():
    assert exact_binomial_interval(0, 50)[0] == 0.0
    assert exact_binomial_interval(50, 50)[1] == 1.0
    assert exact_binomial_interval(0, 100)[1] < 0.05
