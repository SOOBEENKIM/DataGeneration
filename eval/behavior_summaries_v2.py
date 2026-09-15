from __future__ import annotations

from typing import Mapping

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch


def fit_short_gap_threshold(
    train: SequenceBatch, *, tau: np.ndarray, quantile: float = 0.5
) -> float:
    transition_mask = train.valid_mask.copy()
    transition_mask[:, 0] = False
    return float(np.quantile(np.asarray(tau)[train.dt_bin[transition_mask]], quantile))


def compute_behavior_summaries_v2(
    batch: SequenceBatch | SyntheticBatch,
    *,
    tau: np.ndarray,
    short_gap_threshold: float,
    window_width: float,
    min_valid_transitions: int = 7,
) -> Mapping[str, np.ndarray]:
    tau = np.asarray(tau)
    n = len(batch.lengths)
    output = {
        key: np.full(n, np.nan, dtype=float)
        for key in ("velocity", "gap", "fanout", "amount", "joint_alignment")
    }
    for i, length in enumerate(batch.lengths):
        length = int(length)
        gaps = tau[batch.dt_bin[i, :length]]
        times = np.cumsum(gaps)
        counts = np.array([
            np.sum((times[:t] >= times[t] - window_width)) for t in range(length)
        ])
        cats = batch.x_cat[i, :length, 0] if batch.x_cat.shape[-1] else np.zeros(length)
        fanout = np.array([
            len(np.unique(cats[:t][times[:t] >= times[t] - window_width]))
            for t in range(length)
        ])
        output["velocity"][i] = counts.mean()
        output["gap"][i] = gaps.mean()
        output["fanout"][i] = fanout.mean()
        output["amount"][i] = batch.x_num[i, :length, 0].mean()
        if length - 1 >= min_valid_transitions:
            short = (gaps[1:] <= short_gap_threshold).astype(float)
            repeat = (cats[1:] == cats[:-1]).astype(float)
            output["joint_alignment"][i] = np.mean(
                (short - short.mean()) * (repeat - repeat.mean())
            )
    return output
