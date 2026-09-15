from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy.stats import ks_2samp

from benchmarks.types import SequenceBatch, SyntheticBatch
from generators.empirical_conditional_block import EmpiricalConditionalBlock
from generators.sampling_plan import SamplingPlan


@dataclass(frozen=True)
class RowGuardThresholds:
    amount_ks: float
    gap_ks: float
    amount_abs_standardized_label_effect: float
    gap_abs_standardized_label_effect: float
    receiver_max_abs_signed_frequency: float
    calibration_trials: int
    calibration_seed: int
    cutoff_rule: str = "maximum_rank_n_of_n_no_interpolation"
    fit_split: str = "train"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _standardized_label_effect(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    groups = [np.asarray(values)[labels == label] for label in (0, 1)]
    if any(len(group) < 2 for group in groups):
        return float("inf")
    degrees = len(groups[0]) + len(groups[1]) - 2
    pooled_variance = (
        (len(groups[0]) - 1) * groups[0].var(ddof=1)
        + (len(groups[1]) - 1) * groups[1].var(ddof=1)
    ) / degrees
    if pooled_variance <= 0:
        difference = float(groups[1].mean() - groups[0].mean())
        return 0.0 if difference == 0 else float("inf")
    return float(
        abs(groups[1].mean() - groups[0].mean())
        / np.sqrt(pooled_variance)
    )


def _receiver_entity_frequencies(
    batch: SequenceBatch | SyntheticBatch,
    categories: int,
) -> np.ndarray:
    output = np.zeros((len(batch.lengths), categories), dtype=float)
    for index, length_value in enumerate(batch.lengths):
        length = int(length_value)
        counts = np.bincount(
            batch.x_cat[index, :length, 0],
            minlength=categories,
        )
        output[index] = counts / length
    return output


def receiver_max_abs_signed_frequency(
    batch: SequenceBatch | SyntheticBatch,
    *,
    categories: int,
) -> float:
    frequencies = _receiver_entity_frequencies(batch, categories)
    means = [
        frequencies[batch.y_entity == label].mean(0)
        for label in (0, 1)
    ]
    if any(not np.isfinite(value).all() for value in means):
        return float("inf")
    return float(np.max(np.abs(means[1] - means[0])))


def row_guard_statistics(
    reference: SequenceBatch | SyntheticBatch,
    candidate: SequenceBatch | SyntheticBatch,
    *,
    tau: np.ndarray,
    receiver_categories: int,
) -> Mapping[str, float]:
    reference_valid = reference.valid_mask
    candidate_valid = candidate.valid_mask
    amount_candidate = candidate.x_num[..., 0][candidate_valid]
    candidate_row_labels = np.repeat(
        candidate.y_entity,
        candidate.lengths,
    )
    gap_candidate = np.asarray(tau)[candidate.dt_bin[candidate_valid]]
    return {
        "amount_ks": float(
            ks_2samp(
                reference.x_num[..., 0][reference_valid],
                amount_candidate,
            ).statistic
        ),
        "gap_ks": float(
            ks_2samp(
                np.asarray(tau)[reference.dt_bin[reference_valid]],
                gap_candidate,
            ).statistic
        ),
        "amount_abs_standardized_label_effect": (
            _standardized_label_effect(
                amount_candidate,
                candidate_row_labels,
            )
        ),
        "gap_abs_standardized_label_effect": _standardized_label_effect(
            gap_candidate,
            candidate_row_labels,
        ),
        "receiver_max_abs_signed_frequency": (
            receiver_max_abs_signed_frequency(
                candidate,
                categories=receiver_categories,
            )
        ),
    }


def calibrate_row_guard_thresholds(
    train: SequenceBatch,
    *,
    tau: np.ndarray,
    entity_count: int,
    trials: int,
    seed: int,
    receiver_practical_margin: float = 0.02,
) -> RowGuardThresholds:
    """Calibrate all cutoffs from train-only empirical sequence resampling."""
    if trials < 1 or entity_count < 1:
        raise ValueError("calibration trials and entity count must be positive")
    if receiver_practical_margin < 0:
        raise ValueError("receiver practical margin cannot be negative")
    reference = EmpiricalConditionalBlock("full")
    reference.fit(train, config={}, seed=seed)
    receiver_categories = int(
        train.x_cat[..., 0][train.valid_mask].max()
    ) + 1
    rows = []
    for trial in range(trials):
        plan_left = SamplingPlan.from_train_policy(
            train,
            entity_count=entity_count,
            seed=seed + 4 * trial,
        )
        plan_right = SamplingPlan.from_train_policy(
            train,
            entity_count=entity_count,
            seed=seed + 4 * trial + 1,
        )
        left = reference.sample(plan_left, seed=seed + 4 * trial + 2)
        right = reference.sample(plan_right, seed=seed + 4 * trial + 3)
        rows.append(
            row_guard_statistics(
                left,
                right,
                tau=tau,
                receiver_categories=receiver_categories,
            )
        )
    maximum = {
        key: max(float(row[key]) for row in rows)
        for key in rows[0]
    }
    maximum["receiver_max_abs_signed_frequency"] = max(
        receiver_practical_margin,
        maximum["receiver_max_abs_signed_frequency"],
    )
    return RowGuardThresholds(
        **maximum,
        calibration_trials=trials,
        calibration_seed=seed,
    )


def evaluate_row_guards(
    real_test: SequenceBatch,
    synthetic: SyntheticBatch,
    *,
    tau: np.ndarray,
    receiver_categories: int,
    thresholds: RowGuardThresholds,
) -> Mapping[str, Any]:
    statistics = row_guard_statistics(
        real_test,
        synthetic,
        tau=tau,
        receiver_categories=receiver_categories,
    )
    limits = thresholds.to_dict()
    checks = {
        key: (
            "PASS"
            if np.isfinite(value) and value <= float(limits[key])
            else "FAIL"
        )
        for key, value in statistics.items()
    }
    return {
        "status": (
            "PASS" if all(value == "PASS" for value in checks.values())
            else "FAIL"
        ),
        "statistics": statistics,
        "thresholds": {
            key: limits[key]
            for key in statistics
        },
        "checks": checks,
        "fit_split": thresholds.fit_split,
        "calibration_trials": thresholds.calibration_trials,
        "calibration_seed": thresholds.calibration_seed,
        "cutoff_rule": thresholds.cutoff_rule,
    }
