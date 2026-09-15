import numpy as np

from scripts.calibrate_receiver_gate_v2_2 import (
    category_counts,
    contaminated_counts,
    exact_binomial_interval,
)


def test_vectorized_category_counts_and_contamination_preserve_lengths():
    categories = np.array([[0, 1, 0], [1, 1, 0]])
    lengths = np.array([2, 3])
    labels = np.array([0, 1])
    counts = category_counts(categories, lengths, 2)
    np.testing.assert_array_equal(counts, [[1, 1], [1, 2]])
    changed = contaminated_counts(
        counts, categories, lengths, labels,
        target_category=0, probability=1.0, seed=3,
    )
    np.testing.assert_array_equal(changed, [[1, 1], [3, 0]])
    np.testing.assert_array_equal(changed.sum(1), lengths)


def test_exact_zero_failure_upper_bound_needs_at_least_100_seeds():
    assert exact_binomial_interval(0, 30)[1] > 0.05
    assert exact_binomial_interval(0, 100)[1] < 0.05
