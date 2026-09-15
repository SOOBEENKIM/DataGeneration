import numpy as np

from benchmarks.receiver_diagnostics import (
    entity_category_counts,
    permutation_null,
    pooled_and_balanced_tvd,
    tvd,
)


def test_tvd_hand_computed():
    assert np.isclose(tvd(np.array([3, 1]), np.array([1, 3])), 0.5)


def test_entity_balanced_differs_from_row_weighting():
    categories = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
    lengths = np.array([4, 1, 2, 1])
    labels = np.array([0, 0, 1, 1])
    counts = entity_category_counts(categories, lengths, 2)
    pooled, balanced = pooled_and_balanced_tvd(counts, labels)
    assert not np.isclose(pooled, balanced)


def test_permutation_is_entity_clustered_and_deterministic():
    rng = np.random.default_rng(4)
    counts = rng.integers(0, 8, size=(100, 4))
    labels = np.r_[np.zeros(80, dtype=int), np.ones(20, dtype=int)]
    first = permutation_null(counts, labels, n_permutations=100, seed=9)
    second = permutation_null(counts, labels, n_permutations=100, seed=9)
    assert first == second
    assert 0 < first["pooled_tail_probability"] <= 1
