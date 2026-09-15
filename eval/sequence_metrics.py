from __future__ import annotations

import numpy as np


def class_contrast(values: np.ndarray, labels: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.mean(values[valid & (labels == 1)]) - np.mean(values[valid & (labels == 0)]))


def lag1_autocorrelation(values: np.ndarray) -> float:
    if len(values) < 3 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return float("nan")
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def receiver_run_lengths(categories: np.ndarray) -> np.ndarray:
    if not len(categories):
        return np.array([], dtype=int)
    cuts = np.flatnonzero(categories[1:] != categories[:-1]) + 1
    return np.diff(np.r_[0, cuts, len(categories)])
