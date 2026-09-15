from __future__ import annotations

import numpy as np


def fit_binning(gaps: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(gaps, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        raise ValueError("fewer than two effective gap bins")
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(gaps, edges[1:-1]).astype(np.int64)
    tau = np.array([np.median(gaps[bins == i]) for i in range(len(edges) - 1)])
    return edges, tau


def apply_binning(gaps: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(gaps, edges[1:-1]).astype(np.int64)
