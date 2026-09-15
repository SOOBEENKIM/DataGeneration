from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.candidate_preparation_v2_7 import (
    FROZEN_SAMPLING_PLAN_SHA256,
    FROZEN_THRESHOLDS,
)
from eval.model_guards_v2_5 import row_guard_statistics
from generators.contracts_v2_5 import validate_synthetic_contract
from generators.sampling_plan import SamplingPlan


class EvaluationGuardContractError(RuntimeError):
    pass


def evaluate_validation_guards(
    *,
    train: SequenceBatch,
    validation: SequenceBatch,
    sample: SyntheticBatch,
    sampling_plan: SamplingPlan,
    tau: np.ndarray,
    receiver_categories: int,
    thresholds: Mapping[str, float],
) -> Mapping[str, Any]:
    if (
        dict(thresholds) != FROZEN_THRESHOLDS
        or sampling_plan.plan_hash != FROZEN_SAMPLING_PLAN_SHA256
    ):
        raise EvaluationGuardContractError(
            "frozen threshold or SamplingPlan mismatch"
        )
    contract = validate_synthetic_contract(
        sample,
        plan=sampling_plan,
        train=train,
    )
    statistics = {
        key: float(value)
        for key, value in row_guard_statistics(
            validation,
            sample,
            tau=np.asarray(tau, dtype=float),
            receiver_categories=receiver_categories,
        ).items()
    }
    checks = {
        key: (
            "PASS"
            if np.isfinite(statistics[key])
            and statistics[key] <= float(FROZEN_THRESHOLDS[key])
            else "FAIL"
        )
        for key in FROZEN_THRESHOLDS
    }
    return {
        "schema_version": "benchmark-v2.7-five-guard-evaluation-v1",
        "status": (
            "PASS"
            if all(value == "PASS" for value in checks.values())
            else "FAIL"
        ),
        "statistics": statistics,
        "thresholds": dict(FROZEN_THRESHOLDS),
        "checks": checks,
        "all_five_guards_pass": all(
            value == "PASS" for value in checks.values()
        ),
        "sampling_plan_sha256": sampling_plan.plan_hash,
        "sample_contract": dict(contract),
        "fit_split": "train",
        "evaluation_split": "validation",
        "validation_selection_executed": False,
        "test_split_read": False,
    }
