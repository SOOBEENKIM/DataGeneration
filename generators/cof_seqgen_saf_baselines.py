"""Fail-closed compatibility contracts for the CoFSeqGen-SAF benchmark.

No third-party package is imported or installed here. The module defines
comparison roles, fair views, train-only generation plans, and raw-output
validation before implementation-specific wrappers are authorized.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.cof_seqgen_saf_contract import CanonicalEntitySequenceDataset


class BaselineCompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class BaselineSpec:
    baseline_id: str
    tier: str
    comparison_role: str
    input_view: str
    package_probe: Optional[str]
    supports_static_context: bool
    supports_irregular_time: bool
    exact_length_control: bool
    execution_status: str
    output_variant: str = "raw_primary"


BASELINE_SPECS: Mapping[str, BaselineSpec] = {
    "tabularargn": BaselineSpec("tabularargn", "primary_sequential", "current_generator", "parent_child", "mostlyai", True, True, False, "wrapper_source_ready_isolated_runtime_required"),
    "tabdit": BaselineSpec("tabdit", "reported_only", "published_reference_no_executable_generator", "sequence_tensor", None, True, True, False, "blocked_upstream_repository_omits_training_and_generation_code"),
    "cpar": BaselineSpec("cpar", "primary_sequential", "current_generator", "sdv_sequential", "sdv", True, True, True, "installed_wrapper_ready"),
    "realtabformer": BaselineSpec("realtabformer", "primary_sequential", "current_generator", "parent_child", "realtabformer", True, True, True, "installed_wrapper_ready"),
    "empirical_sequence_sampler": BaselineSpec("empirical_sequence_sampler", "primary_sequential", "nonparametric_memorization_control", "canonical", None, True, True, True, "wrapper_ready"),
    "ctgan": BaselineSpec("ctgan", "flattened_control", "marginal_control_only", "flat_event", "ctgan", False, False, True, "installed_wrapper_ready"),
    "tvae": BaselineSpec("tvae", "flattened_control", "marginal_control_only", "flat_event", "ctgan", False, False, True, "installed_wrapper_ready"),
    "gaussian_copula": BaselineSpec("gaussian_copula", "flattened_control", "marginal_control_only", "flat_event", "sdv", False, False, True, "installed_wrapper_ready"),
    "non_v3_cof": BaselineSpec("non_v3_cof", "historical_frozen", "lineage_context_only", "legacy_frozen", None, False, True, False, "frozen_artifact_only"),
    "ccmtpp_c1": BaselineSpec("ccmtpp_c1", "historical_frozen", "lineage_context_only", "legacy_frozen", None, True, True, False, "frozen_artifact_only"),
    "hcmttpp_h1": BaselineSpec("hcmttpp_h1", "historical_stopped", "failed_gate_context_only", "legacy_frozen", None, True, True, False, "stopped_failed_gate"),
}

PRIMARY_SEQUENTIAL = tuple(
    key for key, value in BASELINE_SPECS.items() if value.tier == "primary_sequential"
)
FLATTENED_CONTROLS = tuple(
    key for key, value in BASELINE_SPECS.items() if value.tier == "flattened_control"
)
HISTORICAL_ONLY = tuple(
    key for key, value in BASELINE_SPECS.items() if value.tier.startswith("historical")
)


@dataclass(frozen=True)
class SharedGenerationPlan:
    """Common child-conditional contexts and lengths sampled from train only."""

    entity_ids: Tuple[str, ...]
    source_train_entity_ids: Tuple[object, ...]
    lengths: Tuple[int, ...]
    static_context: pd.DataFrame
    fit_split: str
    seed: int

    def __post_init__(self) -> None:
        if self.fit_split != "train":
            raise BaselineCompatibilityError("generation plan must be train-only")
        if not self.entity_ids or len(self.entity_ids) != len(self.lengths):
            raise BaselineCompatibilityError("plan identities and lengths must align")
        if any(length < 1 for length in self.lengths):
            raise BaselineCompatibilityError("planned lengths must be positive")
        if tuple(self.static_context["entity_id"]) != self.entity_ids:
            raise BaselineCompatibilityError("static plan rows must align to entity IDs")


def build_shared_generation_plan(
    dataset: CanonicalEntitySequenceDataset,
    *,
    n_entities: int,
    seed: int,
    eligible_entity_ids: Optional[Sequence[object]] = None,
) -> SharedGenerationPlan:
    """Sample both context and sequence length only from canonical train entities."""

    if n_entities < 1:
        raise BaselineCompatibilityError("n_entities must be positive")
    all_train_ids = dataset.entity_ids_for_split("train")
    if eligible_entity_ids is None:
        train_ids = all_train_ids
    else:
        requested = tuple(eligible_entity_ids)
        if not requested or not set(requested) <= set(all_train_ids):
            raise BaselineCompatibilityError(
                "eligible generation entities must be a non-empty train subset"
            )
        train_ids = requested
    rng = np.random.default_rng(seed)
    positions = rng.integers(0, len(train_ids), size=n_entities)
    selected = tuple(train_ids[int(position)] for position in positions)
    length_map = dataset.events.groupby("entity_id").size().to_dict()
    synthetic_ids = tuple(f"saf-synthetic-{index:08d}" for index in range(n_entities))
    static_index = dataset.static_context.set_index("entity_id")
    rows = []
    for synthetic_id, source_id in zip(synthetic_ids, selected):
        row = static_index.loc[source_id].to_dict()
        rows.append({"entity_id": synthetic_id, **row})
    return SharedGenerationPlan(
        entity_ids=synthetic_ids,
        source_train_entity_ids=selected,
        lengths=tuple(int(length_map[source_id]) for source_id in selected),
        static_context=pd.DataFrame(rows, columns=dataset.static_context.columns),
        fit_split="train",
        seed=seed,
    )


def dependency_readiness() -> Dict[str, Dict[str, object]]:
    """Read-only package probe; this imports and installs nothing."""

    package_distributions = {
        "mostlyai": "mostlyai-engine",
        "sdv": "sdv",
        "realtabformer": "realtabformer",
        "ctgan": "ctgan",
    }
    result = {}
    for baseline_id, spec in BASELINE_SPECS.items():
        installed = (
            None
            if spec.package_probe is None
            else importlib.util.find_spec(spec.package_probe) is not None
        )
        version = None
        if installed:
            try:
                version = importlib.metadata.version(
                    package_distributions[spec.package_probe]
                )
            except importlib.metadata.PackageNotFoundError:
                version = "unknown"
        result[baseline_id] = {
            "package_probe": spec.package_probe,
            "installed": installed,
            "version": version,
            "exact_length_control": spec.exact_length_control,
            "execution_status": spec.execution_status,
            "installation_attempted": False,
        }
    return result


def materialize_compatibility_view(
    dataset: CanonicalEntitySequenceDataset,
    baseline_id: str,
    *,
    split: str = "train",
) -> Dict[str, pd.DataFrame]:
    if baseline_id not in BASELINE_SPECS:
        raise BaselineCompatibilityError(f"unknown baseline: {baseline_id}")
    if split not in {"train", "validation", "test"}:
        raise BaselineCompatibilityError(f"unknown split: {split}")
    spec = BASELINE_SPECS[baseline_id]
    ids = set(dataset.entity_ids_for_split(split))
    static = dataset.static_context[dataset.static_context["entity_id"].isin(ids)].copy()
    events = dataset.events[dataset.events["entity_id"].isin(ids)].copy()
    if spec.input_view == "parent_child":
        return {"parent": static, "child": events}
    if spec.input_view == "sdv_sequential":
        repeated = events.merge(static, on="entity_id", how="left", validate="many_to_one")
        return {"sequences": repeated}
    if spec.input_view == "sequence_tensor":
        return {"static": static, "events": events}
    if spec.input_view == "flat_event":
        flat = events.drop(columns=["entity_id", "event_id", "event_index", "timestamp"])
        return {"flat_events": flat}
    if spec.input_view == "canonical":
        return {"static": static, "events": events}
    raise BaselineCompatibilityError(
        "historical frozen artifacts are not rematerialized through the SAF view"
    )


def validate_raw_generated_events(
    generated: pd.DataFrame,
    plan: SharedGenerationPlan,
    *,
    atol: float = 1e-8,
) -> None:
    """Validate raw output before any optional temporal repair."""

    required = {
        "entity_id",
        "event_index",
        "timestamp",
        "gap",
        "receiver_or_mark",
        "amount_or_numeric_value",
    }
    missing = required - set(generated.columns)
    if missing:
        raise BaselineCompatibilityError(f"raw output missing columns: {sorted(missing)}")
    if set(generated["entity_id"]) != set(plan.entity_ids):
        raise BaselineCompatibilityError("raw output entity IDs differ from shared plan")
    lengths = generated.groupby("entity_id").size().reindex(plan.entity_ids).to_numpy()
    if not np.array_equal(lengths, np.asarray(plan.lengths)):
        raise BaselineCompatibilityError("raw output lengths differ from shared plan")
    for entity_id, group in generated.groupby("entity_id", sort=False):
        group = group.sort_values("event_index", kind="mergesort")
        indices = group["event_index"].to_numpy(int)
        if not np.array_equal(indices, np.arange(len(group))):
            raise BaselineCompatibilityError("event_index is not contiguous")
        gap = pd.to_numeric(group["gap"], errors="coerce").to_numpy(float)
        timestamp = pd.to_numeric(group["timestamp"], errors="coerce").to_numpy(float)
        if not np.isnan(gap[0]) or not np.isfinite(gap[1:]).all() or (gap[1:] < 0).any():
            raise BaselineCompatibilityError("raw gap support violates canonical contract")
        if not np.isfinite(timestamp).all() or not np.allclose(np.diff(timestamp), gap[1:], atol=atol, rtol=0):
            raise BaselineCompatibilityError("raw timestamps and gaps disagree")


def comparison_manifest() -> Dict[str, object]:
    return {
        "primary_sequential": PRIMARY_SEQUENTIAL,
        "flattened_marginal_controls": FLATTENED_CONTROLS,
        "historical_frozen_context": HISTORICAL_ONLY,
        "proposed_model": "SAF-O1",
        "tabpfn_role": "downstream_utility_evaluator_only_not_a_generator",
        "primary_output": "raw_unrepaired",
        "postprocessed_output": "secondary_separately_labeled_never_pooled",
        "common_plan": "train_only_parent_context_and_length_sampling",
        "exact_length_primary": tuple(
            key for key in PRIMARY_SEQUENTIAL if BASELINE_SPECS[key].exact_length_control
        ),
        "native_length_reported_separately": ("tabularargn",),
        "reported_only_not_executed": ("tabdit",),
    }
