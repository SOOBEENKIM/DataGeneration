import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from benchmarks.argn_state_evaluation import (
    episode_features, hazard_table, scalar_summary, bootstrap_rates,
    histogram_by_customer, bootstrap_w1_hist, safe_rate,
)


def example():
    # A: left-boundary fraud ends normally, a new fraud then reaches window end.
    # B: normal -> two frauds -> normal: one complete known-start episode.
    labels = [[1, 1, 0, 1], [0, 1, 1, 0], [0, 0]]
    rows = []
    for entity, sequence in enumerate(labels):
        for index, label in enumerate(sequence):
            rows.append(dict(entity_id=entity, event_index=index, fraud=label,
                             gap=np.nan if index == 0 else 5,
                             amount_or_numeric_value=100, receiver_or_mark="m", seen_merchant=int(index > 0)))
    return episode_features(pd.DataFrame(rows))


def test_boundaries_do_not_count_as_normal_termination():
    events, runs = example()
    assert list(runs.length) == [2, 1, 2]
    assert list(runs.span_seconds) == [5, 0, 5]
    assert runs.left_boundary.sum() == 1
    assert runs.right_boundary.sum() == 1
    assert runs.completed_known_start.sum() == 1
    summary = scalar_summary(events, runs)
    assert summary["continuation_denominator"] == 4
    assert summary["continuation_rate"] == .5
    assert summary["termination_rate"] == .5
    hazards = pd.DataFrame(hazard_table(events, "example"))
    assert hazards.eligible_transitions.sum() == 2  # left run excluded, tail excluded
    assert hazards.censored_next.sum() == 1
    assert hazards.ends.sum() == 1


def test_bootstrap_preserves_whole_customer_transitions():
    events, _ = example()
    weights = np.array([[1, 1, 1], [2, 0, 0], [0, 0, 3]])
    rates = bootstrap_rates(events, [0, 1, 2], weights)
    np.testing.assert_allclose(rates["fraud_rate"], [.5, .75, 0])
    np.testing.assert_allclose(rates["continuation_rate"][:2], [.5, .5])
    assert np.isnan(rates["continuation_rate"][2])
    assert np.isnan(safe_rate(0, 0))


def test_histogram_bootstrap_w1_matches_weighted_exact_distance():
    support = np.array([1, 2, 4, 10])
    real = histogram_by_customer([1, 2, 4], [0, 0, 1], [0, 1], support)
    syn = histogram_by_customer([2, 10], [0, 1], [0, 1], support)
    weights = np.array([[1, 1], [3, 1]])
    results = bootstrap_w1_hist(real, syn, weights, support)
    for result, w in zip(results, weights):
        expected = wasserstein_distance(np.log1p([1, 2, 4]), np.log1p([2, 10]),
                                        u_weights=[w[0], w[0], w[1]], v_weights=w)
        np.testing.assert_allclose(result, expected)
