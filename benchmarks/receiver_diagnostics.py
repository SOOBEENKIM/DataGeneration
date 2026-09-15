from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def tvd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.sum() <= 0 or b.sum() <= 0:
        return float("nan")
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def entity_category_counts(
    categories: np.ndarray,
    lengths: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    counts = np.zeros((len(lengths), n_categories), dtype=np.int64)
    for index, length in enumerate(lengths):
        counts[index] = np.bincount(
            categories[index, : int(length)], minlength=n_categories
        )
    return counts


def entity_category_frequencies(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)


def pooled_and_balanced_tvd(
    counts: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    frequencies = entity_category_frequencies(counts)
    pooled = tvd(counts[labels == 0].sum(0), counts[labels == 1].sum(0))
    balanced = tvd(frequencies[labels == 0].mean(0), frequencies[labels == 1].mean(0))
    return pooled, balanced


def receiver_run_statistics(
    categories: np.ndarray, lengths: np.ndarray, labels: np.ndarray
) -> list[dict[str, float]]:
    rows = []
    for label in (0, 1):
        run_lengths: list[int] = []
        repeats = 0
        transitions = 0
        per_entity_repeat = []
        for index in np.flatnonzero(labels == label):
            values = categories[index, : int(lengths[index])]
            if len(values) < 2:
                continue
            cuts = np.flatnonzero(values[1:] != values[:-1]) + 1
            run_lengths.extend(np.diff(np.r_[0, cuts, len(values)]).tolist())
            repeated = int(np.sum(values[1:] == values[:-1]))
            repeats += repeated
            transitions += len(values) - 1
            per_entity_repeat.append(repeated / (len(values) - 1))
        rows.append(
            {
                "label": label,
                "n_entities": int(np.sum(labels == label)),
                "pooled_repeat_rate": repeats / transitions,
                "entity_mean_repeat_rate": float(np.mean(per_entity_repeat)),
                "mean_run_length": float(np.mean(run_lengths)),
                "median_run_length": float(np.median(run_lengths)),
                "p95_run_length": float(np.quantile(run_lengths, 0.95)),
                "max_run_length": int(np.max(run_lengths)),
            }
        )
    return rows


def positionwise_tvd(
    categories: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    n_categories: int,
) -> list[dict[str, float]]:
    rows = []
    for position in range(categories.shape[1]):
        valid = lengths > position
        counts = [
            np.bincount(
                categories[valid & (labels == label), position],
                minlength=n_categories,
            )
            for label in (0, 1)
        ]
        rows.append(
            {
                "position": position,
                "n_y0": int(np.sum(valid & (labels == 0))),
                "n_y1": int(np.sum(valid & (labels == 1))),
                "receiver_tvd": tvd(counts[0], counts[1]),
            }
        )
    return rows


def segmentwise_tvd(
    categories: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    n_categories: int,
) -> list[dict[str, float]]:
    values: dict[tuple[str, int], list[int]] = {
        (segment, label): []
        for segment in ("early", "middle", "late")
        for label in (0, 1)
    }
    for index, length in enumerate(lengths):
        for position, category in enumerate(categories[index, : int(length)]):
            relative = (position + 0.5) / int(length)
            segment = "early" if relative < 1 / 3 else "middle" if relative < 2 / 3 else "late"
            values[(segment, int(labels[index]))].append(int(category))
    rows = []
    for segment in ("early", "middle", "late"):
        counts = [
            np.bincount(values[(segment, label)], minlength=n_categories)
            for label in (0, 1)
        ]
        rows.append(
            {
                "segment": segment,
                "n_y0": len(values[(segment, 0)]),
                "n_y1": len(values[(segment, 1)]),
                "receiver_tvd": tvd(counts[0], counts[1]),
            }
        )
    return rows


def permutation_null(
    counts: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int = 1000,
    seed: int = 20260728,
) -> dict[str, float]:
    observed_pooled, observed_balanced = pooled_and_balanced_tvd(counts, labels)
    frequencies = entity_category_frequencies(counts)
    rng = np.random.default_rng(seed)
    n_positive = int(labels.sum())
    n_negative = len(labels) - n_positive
    total_counts = counts.sum(0)
    total_frequencies = frequencies.sum(0)
    pooled_null = np.empty(n_permutations)
    balanced_null = np.empty(n_permutations)
    all_indices = np.arange(len(labels))
    for iteration in range(n_permutations):
        positive = rng.choice(all_indices, n_positive, replace=False)
        positive_counts = counts[positive].sum(0)
        positive_frequencies = frequencies[positive].sum(0)
        pooled_null[iteration] = tvd(
            total_counts - positive_counts, positive_counts
        )
        balanced_null[iteration] = tvd(
            (total_frequencies - positive_frequencies) / n_negative,
            positive_frequencies / n_positive,
        )
    def summary(observed: float, null: np.ndarray, prefix: str) -> dict[str, float]:
        median = float(np.median(null))
        return {
            f"observed_{prefix}_tvd": observed,
            f"{prefix}_permutation_median": median,
            f"{prefix}_permutation_q95": float(np.quantile(null, 0.95)),
            f"{prefix}_permutation_q99": float(np.quantile(null, 0.99)),
            f"{prefix}_excess_tvd": observed - median,
            f"{prefix}_tail_probability": float(
                (1 + np.sum(null >= observed)) / (n_permutations + 1)
            ),
        }
    return {
        "n_permutations": n_permutations,
        **summary(observed_pooled, pooled_null, "pooled"),
        **summary(observed_balanced, balanced_null, "entity_balanced"),
    }
