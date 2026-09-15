"""External validation v1 fidelity and coherence metrics.

These functions contain no controlled-benchmark thresholds.  Thresholds are
fitted separately from the external training entities and frozen before a
validation candidate is evaluated.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch


class ExternalMetricError(RuntimeError):
    """Raised when external metric inputs violate the frozen contract."""


def _ks(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        raise ExternalMetricError("both classes must contain valid amount rows")
    support = np.sort(np.concatenate((left, right)))
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    left_cdf = np.searchsorted(left_sorted, support, side="right") / len(left)
    right_cdf = np.searchsorted(right_sorted, support, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _total_variation(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        raise ExternalMetricError("both classes must contain categorical rows")
    left_values, left_counts = np.unique(left, return_counts=True)
    right_values, right_counts = np.unique(right, return_counts=True)
    support = np.union1d(left_values, right_values)
    left_frequency = np.zeros(len(support), dtype=float)
    right_frequency = np.zeros(len(support), dtype=float)
    left_frequency[np.searchsorted(support, left_values)] = left_counts / len(left)
    right_frequency[np.searchsorted(support, right_values)] = right_counts / len(right)
    return float(0.5 * np.abs(left_frequency - right_frequency).sum())


def _rows_by_label(
    batch: SequenceBatch | SyntheticBatch, label: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = batch.valid_mask & (batch.y_entity[:, None] == label)
    return (
        batch.x_num[..., 0][selected],
        batch.dt_bin[selected],
        batch.x_cat[..., 0][selected],
    )


def _short_gap_repeat_rate(
    batch: SequenceBatch | SyntheticBatch,
    *,
    label: int,
    gap_tau: np.ndarray,
    short_gap_threshold: float,
) -> float:
    if len(gap_tau) <= int(batch.dt_bin.max(initial=0)):
        raise ExternalMetricError("gap tau does not cover the encoded bins")
    positions = np.arange(batch.valid_mask.shape[1])[None, :]
    eligible = (
        batch.valid_mask
        & (positions > 0)
        & (batch.y_entity[:, None] == label)
    )
    previous_valid = np.zeros_like(batch.valid_mask)
    previous_valid[:, 1:] = batch.valid_mask[:, :-1]
    eligible &= previous_valid
    if not eligible.any():
        raise ExternalMetricError("coherence has no eligible rows for a class")
    previous_receiver = np.zeros_like(batch.x_cat[..., 0])
    previous_receiver[:, 1:] = batch.x_cat[:, :-1, 0]
    repeated = batch.x_cat[..., 0] == previous_receiver
    short = gap_tau[batch.dt_bin] <= short_gap_threshold
    return float(np.mean((short & repeated)[eligible]))


def compute_external_validation_metrics(
    *,
    real: SequenceBatch,
    synthetic: SyntheticBatch,
    gap_tau: np.ndarray,
    short_gap_threshold: float,
) -> Mapping[str, Any]:
    """Compute preregistered class-conditional external metrics."""

    tau = np.asarray(gap_tau, dtype=float)
    if tau.ndim != 1 or not len(tau) or not np.isfinite(tau).all():
        raise ExternalMetricError("gap tau must be a finite one-dimensional array")
    if not np.isfinite(short_gap_threshold):
        raise ExternalMetricError("short-gap threshold must be finite")
    fidelity: dict[str, float] = {}
    coherence: dict[str, float] = {}
    for label in (0, 1):
        real_amount, real_gap, real_receiver = _rows_by_label(real, label)
        synthetic_amount, synthetic_gap, synthetic_receiver = _rows_by_label(
            synthetic, label
        )
        fidelity[f"amount_ks_y{label}"] = _ks(real_amount, synthetic_amount)
        fidelity[f"gap_total_variation_y{label}"] = _total_variation(
            real_gap, synthetic_gap
        )
        fidelity[f"receiver_total_variation_y{label}"] = _total_variation(
            real_receiver, synthetic_receiver
        )
        real_coherence = _short_gap_repeat_rate(
            real,
            label=label,
            gap_tau=tau,
            short_gap_threshold=short_gap_threshold,
        )
        synthetic_coherence = _short_gap_repeat_rate(
            synthetic,
            label=label,
            gap_tau=tau,
            short_gap_threshold=short_gap_threshold,
        )
        coherence[f"short_gap_receiver_repeat_error_y{label}"] = abs(
            real_coherence - synthetic_coherence
        )
    return {
        "schema_version": "external-validation-metrics-v1",
        "fidelity": fidelity,
        "coherence": coherence,
        "controlled_benchmark_thresholds_used": False,
    }


def validate_external_hard_contract(
    *,
    real_validation: SequenceBatch,
    synthetic: SyntheticBatch,
    gap_cardinality: int,
    receiver_cardinality: int,
) -> Mapping[str, bool]:
    """Validate only structural/padding/declared train-vocabulary support."""

    if not (
        np.array_equal(synthetic.y_entity, real_validation.y_entity)
        and np.array_equal(synthetic.lengths, real_validation.lengths)
        and np.array_equal(synthetic.valid_mask, real_validation.valid_mask)
    ):
        raise ExternalMetricError("synthetic validation Y/L/mask plan mismatch")
    valid = synthetic.valid_mask
    if np.any(synthetic.dt_bin[valid] < 0) or np.any(
        synthetic.dt_bin[valid] >= int(gap_cardinality)
    ):
        raise ExternalMetricError("synthetic gap support is outside train state")
    receiver = synthetic.x_cat[..., 0][valid]
    if np.any(receiver < 1) or np.any(receiver >= int(receiver_cardinality)):
        raise ExternalMetricError("synthetic receiver support is outside train vocabulary")
    padding = ~valid
    if (
        np.any(synthetic.x_num[padding] != 0)
        or np.any(synthetic.dt_bin[padding] != 0)
        or np.any(synthetic.x_cat[padding] != 0)
    ):
        raise ExternalMetricError("synthetic padding contract failed")
    return {
        "mask": True,
        "padding": True,
        "train_discrete_support": True,
    }
