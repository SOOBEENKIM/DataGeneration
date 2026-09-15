from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.stats import ttest_ind


REQUIRED_SEEDS = (1, 2, 3, 4, 5)
COF_ID = "cof_seqgen"
PRIMARY_COMPARATORS = (
    "ctgan_separate_class",
    "tvae_separate_class",
    "empirical_iid",
)
SECONDARY_COMPARATORS = (
    "neural_sequence",
    "independent_markov",
    "joint_markov",
    "plug_in_hmm",
    "plug_in_hsmm",
)
STRUCTURAL_REFERENCES = (
    "empirical_iid",
    "block_2",
    "block_4",
    "block_8",
    "full_sequence_reference",
)
PRIMARY_ENDPOINT = "continuous_association_recovery_error"
SUPPORT_DIAGNOSTIC_BINS = (4, 8)
FLOOR_EPSILON = 1e-6


@dataclass(frozen=True)
class Aggregate:
    status: str
    errors: np.ndarray | None
    reason: str | None


def _aggregate_one(seed_records: Mapping[int, Mapping[str, Any]]) -> Aggregate:
    if set(seed_records) != set(REQUIRED_SEEDS):
        return Aggregate(
            "INCOMPLETE",
            None,
            "all seeds 1..5 must complete before aggregation",
        )
    errors = []
    for seed in REQUIRED_SEEDS:
        record = seed_records[seed]
        if record.get("status") != "COMPLETE":
            return Aggregate(
                "INVALID",
                None,
                f"seed {seed} status={record.get('status')}",
            )
        plan_hash = record.get("sampling_plan_hash")
        if not isinstance(plan_hash, str) or len(plan_hash) != 64:
            return Aggregate(
                "INVALID",
                None,
                f"seed {seed} has a missing or invalid SamplingPlan hash",
            )
        guards = record.get("hard_guards")
        if not isinstance(guards, Mapping) or any(
            value != "PASS" for value in guards.values()
        ):
            return Aggregate(
                "INVALID",
                None,
                f"seed {seed} has a failed or missing hard guard",
            )
        value = record.get("association_recovery_error")
        if value is None or not np.isfinite(float(value)):
            return Aggregate(
                "INVALID",
                None,
                f"seed {seed} has a non-finite primary endpoint",
            )
        errors.append(float(value))
    return Aggregate("VALID", np.asarray(errors, dtype=float), None)


def _shared_sampling_plan_hash(
    records: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> tuple[str | None, str | None]:
    hashes: set[str] = set()
    for seed_records in records.values():
        for record in seed_records.values():
            value = record.get("sampling_plan_hash")
            if not isinstance(value, str) or len(value) != 64:
                return None, "aggregation refused: missing SamplingPlan hash"
            hashes.add(value)
    if len(hashes) != 1:
        return (
            None,
            "aggregation refused: model runs do not share one SamplingPlan hash",
        )
    return next(iter(hashes)), None


def hedges_g_baseline_minus_cof(
    baseline: np.ndarray,
    cof: np.ndarray,
) -> float:
    baseline = np.asarray(baseline, dtype=float)
    cof = np.asarray(cof, dtype=float)
    degrees = len(baseline) + len(cof) - 2
    pooled_variance = (
        (len(baseline) - 1) * baseline.var(ddof=1)
        + (len(cof) - 1) * cof.var(ddof=1)
    ) / degrees
    pooled_sd = float(np.sqrt(pooled_variance))
    if pooled_sd == 0:
        difference = float(baseline.mean() - cof.mean())
        if difference == 0:
            return 0.0
        return float(np.copysign(np.inf, difference))
    correction = 1 - 3 / (4 * (len(baseline) + len(cof)) - 9)
    return float(
        correction * (baseline.mean() - cof.mean()) / pooled_sd
    )


def unpaired_bootstrap_ci(
    baseline: np.ndarray,
    cof: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    baseline = np.asarray(baseline, dtype=float)
    cof = np.asarray(cof, dtype=float)
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=float)
    for index in range(resamples):
        left = baseline[
            rng.integers(0, len(baseline), size=len(baseline))
        ]
        right = cof[rng.integers(0, len(cof), size=len(cof))]
        differences[index] = left.mean() - right.mean()
    low, high = np.quantile(differences, (0.025, 0.975))
    return float(low), float(high)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    for key, value in p_values.items():
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid p-value for {key}")
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(p_values[key]))
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def floor_proximity_interpretation(
    *,
    cof_mean_error: float,
    c0_mean_error: float | None,
    floor_epsilon: float = FLOOR_EPSILON,
) -> Mapping[str, Any]:
    if floor_epsilon != FLOOR_EPSILON:
        raise ValueError("v2.5 fixes floor_epsilon=1e-6")
    if c0_mean_error is None:
        return {
            "status": "UNAVAILABLE",
            "floor_epsilon": floor_epsilon,
            "ratio": None,
            "floor_proximity_permitted": False,
        }
    if c0_mean_error <= floor_epsilon:
        return {
            "status": "RATIO_UNINTERPRETABLE",
            "floor_epsilon": floor_epsilon,
            "c0_mean_error": float(c0_mean_error),
            "cof_mean_error": float(cof_mean_error),
            "ratio": None,
            "floor_proximity_permitted": False,
        }
    ratio = float(cof_mean_error / c0_mean_error)
    return {
        "status": (
            "FLOOR_PROXIMITY_PERMITTED"
            if cof_mean_error <= 2 * c0_mean_error
            else "NOT_FLOOR_PROXIMATE"
        ),
        "floor_epsilon": floor_epsilon,
        "c0_mean_error": float(c0_mean_error),
        "cof_mean_error": float(cof_mean_error),
        "ratio": ratio,
        "floor_proximity_permitted": bool(ratio <= 2),
    }


def analyze_full_experiment(
    records: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 25_000,
    c0_mean_error: float | None = None,
) -> Mapping[str, Any]:
    """Apply the frozen v2.5 five-seed, fail-closed C2 analysis."""
    shared_plan_hash, plan_error = _shared_sampling_plan_hash(records)
    aggregated = {
        generator: _aggregate_one(seed_records)
        for generator, seed_records in records.items()
    }
    summaries: dict[str, Any] = {}
    for generator, aggregate in aggregated.items():
        if aggregate.status != "VALID":
            summaries[generator] = {
                "status": aggregate.status,
                "reason": aggregate.reason,
                "mean_association_recovery_error": None,
            }
        else:
            assert aggregate.errors is not None
            summaries[generator] = {
                "status": "VALID",
                "seeds": list(REQUIRED_SEEDS),
                "errors": aggregate.errors.tolist(),
                "mean_association_recovery_error": float(
                    aggregate.errors.mean()
                ),
                "std_association_recovery_error": float(
                    aggregate.errors.std(ddof=1)
                ),
            }
    base_output = {
        "primary_endpoint": PRIMARY_ENDPOINT,
        "sole_primary_endpoint": True,
        "support_diagnostic_bins": list(SUPPORT_DIAGNOSTIC_BINS),
        "primary_family": list(PRIMARY_COMPARATORS),
        "secondary_family": list(SECONDARY_COMPARATORS),
        "structural_references": list(STRUCTURAL_REFERENCES),
        "shared_sampling_plan_hash": shared_plan_hash,
        "partial_seed_results_used_for_decisions": False,
    }
    if plan_error is not None:
        return {
            **base_output,
            "status": "INVALID",
            "reason": plan_error,
            "summaries": summaries,
            "pairwise": {},
            "c2_supported": False,
            "c2_decision": "이번 사전등록 실험에서 C2는 지지되지 않음",
        }
    if COF_ID not in aggregated or aggregated[COF_ID].status != "VALID":
        return {
            **base_output,
            "status": "INVALID",
            "reason": "CoF does not have five valid seeds",
            "summaries": summaries,
            "pairwise": {},
            "c2_supported": False,
            "c2_decision": "이번 사전등록 실험에서 C2는 지지되지 않음",
        }
    cof = aggregated[COF_ID].errors
    assert cof is not None
    comparisons: dict[str, Any] = {}
    raw_primary: dict[str, float] = {}
    raw_secondary: dict[str, float] = {}
    for generator, aggregate in aggregated.items():
        if generator == COF_ID:
            continue
        if aggregate.status != "VALID":
            comparisons[generator] = {
                "status": "INVALID",
                "reason": aggregate.reason,
            }
            continue
        assert aggregate.errors is not None
        error = aggregate.errors
        test = ttest_ind(error, cof, equal_var=False)
        low, high = unpaired_bootstrap_ci(
            error,
            cof,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed
            + sum(
                (index + 1) * ord(char)
                for index, char in enumerate(generator)
            ),
        )
        comparison = {
            "status": "VALID",
            "effect_definition": "baseline_mean_error_minus_cof_mean_error",
            "mean_effect": float(error.mean() - cof.mean()),
            "hedges_g": hedges_g_baseline_minus_cof(error, cof),
            "unpaired_bootstrap_95_ci": [low, high],
            "welch_t_two_sided": float(test.statistic),
            "welch_p_two_sided": float(test.pvalue),
            "reference_only": generator == "full_sequence_reference",
        }
        comparisons[generator] = comparison
        if generator in PRIMARY_COMPARATORS:
            raw_primary[generator] = float(test.pvalue)
            comparison["multiplicity_family"] = "C2_primary"
        elif generator in SECONDARY_COMPARATORS:
            raw_secondary[generator] = float(test.pvalue)
            comparison["multiplicity_family"] = "secondary"
    missing_primary = set(PRIMARY_COMPARATORS) - set(raw_primary)
    missing_secondary = set(SECONDARY_COMPARATORS) - set(raw_secondary)
    for generator, value in holm_adjust(raw_primary).items():
        comparisons[generator]["holm_adjusted_p"] = value
    for generator, value in holm_adjust(raw_secondary).items():
        comparisons[generator]["holm_adjusted_p"] = value
    for generator in ("plug_in_hmm", "plug_in_hsmm"):
        comparison = comparisons.get(generator)
        if (
            comparison
            and comparison.get("status") == "VALID"
            and comparison.get("holm_adjusted_p", 1.0) >= 0.05
        ):
            comparison["secondary_interpretation"] = (
                "planned sample did not detect a difference"
            )
    c2_checks = {
        generator: {
            "mean_effect_gt_zero": bool(
                comparisons.get(generator, {}).get("mean_effect", -np.inf)
                > 0
            ),
            "holm_adjusted_p_lt_0_05": bool(
                comparisons.get(generator, {}).get(
                    "holm_adjusted_p",
                    np.inf,
                )
                < 0.05
            ),
            "hedges_g_gt_0_8": bool(
                comparisons.get(generator, {}).get("hedges_g", -np.inf)
                > 0.8
            ),
        }
        for generator in PRIMARY_COMPARATORS
    }
    c2_supported = (
        not missing_primary
        and all(all(checks.values()) for checks in c2_checks.values())
    )
    missing = sorted(missing_primary | missing_secondary)
    overall = "COMPLETE" if not missing else "INVALID"
    return {
        **base_output,
        "status": overall,
        "reason": (
            None
            if not missing
            else f"missing/invalid preregistered comparisons: {missing}"
        ),
        "summaries": summaries,
        "pairwise": comparisons,
        "c2_checks": c2_checks,
        "c2_supported": c2_supported,
        "c2_decision": (
            "C2 supported in this preregistered experiment"
            if c2_supported
            else "이번 사전등록 실험에서 C2는 지지되지 않음"
        ),
        "floor_proximity": floor_proximity_interpretation(
            cof_mean_error=float(cof.mean()),
            c0_mean_error=c0_mean_error,
        ),
    }
