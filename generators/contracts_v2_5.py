from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch
from .sampling_plan import SamplingPlan


BaselineRole = Literal[
    "stochastic_baseline",
    "learned_baseline",
    "proposed",
    "structural_reference",
    "oracle_reference",
]


@dataclass(frozen=True)
class BaselineSpec:
    generator_id: str
    display_name: str
    role: BaselineRole
    factory: Callable[[], Any]
    training_device: Literal["cpu", "gpu"]
    stochastic: bool
    full_sequence_competitor: bool = True


def validate_adapter(adapter: Any) -> None:
    missing = [
        name for name in ("name", "fit", "sample")
        if not hasattr(adapter, name)
    ]
    if missing:
        raise TypeError(f"generator adapter is missing {missing}")


def validate_synthetic_contract(
    sample: SyntheticBatch,
    *,
    plan: SamplingPlan,
    train: SequenceBatch,
) -> Mapping[str, Any]:
    """Fail closed on the shared v2.5 label/length/mask/support contract."""
    errors: list[str] = []
    if not np.array_equal(sample.y_entity, plan.y_entity):
        errors.append("entity labels differ from the fixed sampling plan")
    if not np.array_equal(sample.lengths, plan.lengths):
        errors.append("sequence lengths differ from the fixed sampling plan")
    if not np.array_equal(sample.valid_mask, plan.valid_mask):
        errors.append("valid mask differs from the fixed sampling plan")
    if sample.x_num.shape[-1] != train.x_num.shape[-1]:
        errors.append("numerical channel count differs from train")
    if sample.x_cat.shape[-1] != train.x_cat.shape[-1]:
        errors.append("categorical channel count differs from train")
    valid = sample.valid_mask
    if not np.isfinite(sample.x_num[valid]).all():
        errors.append("valid numerical output contains non-finite values")
    train_gap_max = int(train.dt_bin[train.valid_mask].max())
    if np.any(sample.dt_bin[valid] < 0) or np.any(
        sample.dt_bin[valid] > train_gap_max
    ):
        errors.append("gap category lies outside train support")
    for channel in range(train.x_cat.shape[-1]):
        train_max = int(train.x_cat[..., channel][train.valid_mask].max())
        generated = sample.x_cat[..., channel][valid]
        if np.any(generated < 0) or np.any(generated > train_max):
            errors.append(
                f"categorical channel {channel} lies outside train support"
            )
    padding = ~valid
    if (
        np.any(sample.x_num[padding] != 0)
        or np.any(sample.dt_bin[padding] != 0)
        or np.any(sample.x_cat[padding] != 0)
    ):
        errors.append("padded values are not all zero")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "status": "PASS",
        "mask_contract": "PASS",
        "support_contract": "PASS",
        "label_prevalence": float(sample.y_entity.mean()),
        "entity_count": int(len(sample.lengths)),
        "valid_rows": int(valid.sum()),
        "sampling_plan_hash": plan.plan_hash,
    }
