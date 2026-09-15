from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .temporal_coupling_v2 import BenchmarkConfig


FEATURE_NAMES = ("log_amount", "continuous_gap", "receiver_category")
_SCENARIO_IDS = {
    "markov_persistence_v2a": 1,
    "joint_semimarkov_v2b": 2,
}
_FEATURE_IDS = {
    "amount": 1,
    "gap": 2,
    "receiver_category": 3,
}


@dataclass(frozen=True)
class LeakageOperator:
    feature: str
    target_label: int
    magnitude: float
    operator_index: int
    receiver_category: int | None = None


def generate_stationary_row_audit(
    config: BenchmarkConfig,
    *,
    n_entities: int,
    audit_seed: int,
    split_id: int,
) -> dict[str, np.ndarray | dict]:
    """Generate the exact stationary single-row marginal without sequences.

    Temporal coupling changes persistence/alignment, not these stationary row
    marginals. Direct generation avoids materialising 16--32 unused events per
    entity in the enlarged AUROC-only audit.
    """
    if n_entities <= 0:
        raise ValueError("n_entities must be positive")
    try:
        scenario_id = _SCENARIO_IDS[config.scenario]
    except KeyError as exc:
        raise ValueError(f"unknown scenario {config.scenario}") from exc
    root = np.random.SeedSequence(
        [
            2404,
            int(audit_seed),
            int(split_id),
            scenario_id,
            int(round(100 * config.kappa)),
            int(config.randomness_nonce),
        ]
    )
    label_ss, state_ss, gap_ss, amount_ss, receiver_ss = root.spawn(5)
    labels = (
        np.random.default_rng(label_ss).random(n_entities) < config.fraud_rate
    ).astype(np.int8)
    burst = (
        np.random.default_rng(state_ss).random(n_entities) < config.pi_burst
    )
    scales = np.where(
        burst,
        config.gap_burst_scale,
        config.gap_normal_scale,
    )
    gaps = np.random.default_rng(gap_ss).exponential(scales).astype(np.float32)
    amounts = np.random.default_rng(amount_ss).normal(
        config.amount_mean,
        config.amount_std,
        n_entities,
    ).astype(np.float32)
    receivers = np.random.default_rng(receiver_ss).integers(
        0,
        config.n_receiver_categories,
        n_entities,
        dtype=np.int64,
    )
    features = np.empty((n_entities, 3), dtype=np.float32)
    features[:, 0] = amounts
    features[:, 1] = gaps
    features[:, 2] = receivers
    return {
        "features": features,
        "labels": labels,
        "metadata": {
            "schema_version": "stationary-row-auroc-audit-v2.4",
            "feature_names": FEATURE_NAMES,
            "audit_seed": int(audit_seed),
            "split_id": int(split_id),
            "scenario": config.scenario,
            "kappa": float(config.kappa),
            "n_entities": int(n_entities),
            "generation": "direct_exact_stationary_row_marginal",
        },
    }


def leakage_operator_for_trial(
    *,
    feature: str,
    magnitude: float,
    validation_trial_ordinal: int,
) -> LeakageOperator:
    if validation_trial_ordinal < 0:
        raise ValueError("validation_trial_ordinal must be nonnegative")
    if feature in ("amount", "gap"):
        target_labels = (1, 0)
        index = validation_trial_ordinal % len(target_labels)
        return LeakageOperator(
            feature=feature,
            target_label=target_labels[index],
            magnitude=float(magnitude),
            operator_index=index,
        )
    if feature == "receiver_category":
        operators = ((1, 0), (1, 31), (0, 1), (0, 63))
        index = validation_trial_ordinal % len(operators)
        target_label, category = operators[index]
        return LeakageOperator(
            feature=feature,
            target_label=target_label,
            receiver_category=category,
            magnitude=float(magnitude),
            operator_index=index,
        )
    raise ValueError(f"unknown leakage feature {feature}")


def inject_leakage(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    operator: LeakageOperator,
    config: BenchmarkConfig,
    audit_seed: int,
    split_id: int,
) -> tuple[np.ndarray, dict]:
    """Apply one deterministic operator and report its exact realised change."""
    if features.shape != (len(labels), 3):
        raise ValueError("expected three audit features per label")
    if operator.feature not in _FEATURE_IDS:
        raise ValueError(f"unknown leakage feature {operator.feature}")
    target = np.flatnonzero(labels == operator.target_label)
    denominator = len(target)
    if denominator == 0:
        raise ValueError("target class is empty")
    requested_count = int(round(operator.magnitude * denominator))
    if requested_count <= 0:
        raise ValueError("leakage magnitude rounds to zero rows")
    result = features.copy()
    feature_id = _FEATURE_IDS[operator.feature]
    magnitude_code = int(round(1_000_000 * operator.magnitude))
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                2404,
                int(audit_seed),
                int(split_id),
                feature_id,
                int(operator.target_label),
                int(operator.operator_index),
                magnitude_code,
            ]
        )
    )

    if operator.feature == "receiver_category":
        if operator.receiver_category is None:
            raise ValueError("receiver operator requires a category")
        category = int(operator.receiver_category)
        if not 0 <= category < config.n_receiver_categories:
            raise ValueError("receiver category is outside support")
        eligible = target[result[target, 2].astype(np.int64) != category]
        if requested_count > len(eligible):
            raise ValueError("not enough non-target receiver rows")
        selected = rng.choice(eligible, requested_count, replace=False)
        before_probability = float(
            np.mean(result[target, 2].astype(np.int64) == category)
        )
        result[selected, 2] = category
        after_probability = float(
            np.mean(result[target, 2].astype(np.int64) == category)
        )
        realised_change = after_probability - before_probability
        anchor = category
    else:
        selected = rng.choice(target, requested_count, replace=False)
        column = 0 if operator.feature == "amount" else 1
        if operator.feature == "amount":
            anchor = config.amount_mean + 3 * config.amount_std
        else:
            anchor = -config.gap_normal_scale * math.log(0.01)
        result[selected, column] = anchor
        before_probability = 0.0
        after_probability = requested_count / denominator
        realised_change = after_probability

    expected_change = requested_count / denominator
    if not np.isclose(realised_change, expected_change, atol=1e-12):
        raise AssertionError("realised injection does not match selected count")
    metadata = {
        "feature": operator.feature,
        "target_label": int(operator.target_label),
        "receiver_category": operator.receiver_category,
        "magnitude": float(operator.magnitude),
        "operator_index": int(operator.operator_index),
        "target_class_rows": int(denominator),
        "requested_rows": int(requested_count),
        "changed_rows": int(len(selected)),
        "requested_probability_change": float(operator.magnitude),
        "realised_probability_change": float(realised_change),
        "before_target_probability": before_probability,
        "after_target_probability": after_probability,
        "replacement_anchor": float(anchor),
        "audit_seed": int(audit_seed),
        "split_id": int(split_id),
    }
    return result, metadata
