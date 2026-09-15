from __future__ import annotations

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.sequence_metrics import lag1_autocorrelation, receiver_run_lengths


def fixed_step_unique_receiver(categories: np.ndarray, width: int = 4) -> float:
    if len(categories) == 0:
        return float("nan")
    values = [
        len(np.unique(categories[max(0, end - width + 1) : end + 1]))
        for end in range(len(categories))
    ]
    return float(np.mean(values))


def channel_only_summaries(
    batch: SequenceBatch | SyntheticBatch,
    *,
    tau: np.ndarray,
    window_width: float = 7.0,
) -> dict[str, np.ndarray]:
    tau = np.asarray(tau)
    output = {
        key: np.empty(len(batch.lengths), dtype=float)
        for key in (
            "raw_gap_mean",
            "gap_lag1_autocorrelation",
            "gap_time_window_velocity",
            "receiver_repeat_rate",
            "receiver_mean_run_length",
            "fixed_step_unique_receiver",
            "raw_amount_mean",
        )
    }
    for index, length_value in enumerate(batch.lengths):
        length = int(length_value)
        gaps = tau[batch.dt_bin[index, :length]]
        times = np.cumsum(gaps)
        categories = batch.x_cat[index, :length, 0]
        amounts = batch.x_num[index, :length, 0]
        output["raw_gap_mean"][index] = gaps.mean()
        output["gap_lag1_autocorrelation"][index] = lag1_autocorrelation(gaps)
        output["gap_time_window_velocity"][index] = np.mean(
            [
                np.sum(times[:end] >= times[end] - window_width)
                for end in range(length)
            ]
        )
        output["receiver_repeat_rate"][index] = np.mean(
            categories[1:] == categories[:-1]
        )
        output["receiver_mean_run_length"][index] = receiver_run_lengths(
            categories
        ).mean()
        output["fixed_step_unique_receiver"][index] = (
            fixed_step_unique_receiver(categories)
        )
        output["raw_amount_mean"][index] = amounts.mean()
    return output
