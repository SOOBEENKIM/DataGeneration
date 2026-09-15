from __future__ import annotations

from typing import Any, Mapping

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.evaluation_only_v2_7 import evaluate_validation_guards as _evaluate
from generators.sampling_plan import SamplingPlan


def evaluate_validation_guards(
    *,
    train: SequenceBatch,
    validation: SequenceBatch,
    sample: SyntheticBatch,
    sampling_plan: SamplingPlan,
    tau,
    receiver_categories: int,
    thresholds: Mapping[str, float],
) -> Mapping[str, Any]:
    result = _evaluate(
        train=train,
        validation=validation,
        sample=sample,
        sampling_plan=sampling_plan,
        tau=tau,
        receiver_categories=receiver_categories,
        thresholds=thresholds,
    )
    return {
        **result,
        "schema_version": "benchmark-v2.8-five-guard-evaluation-v1",
    }
