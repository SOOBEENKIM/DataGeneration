"""Fail-closed source and preregistration contracts for ``cof_ccmtpp_v1``."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml


class CCMTPPContractError(RuntimeError):
    pass


DIAGNOSTIC_FIELDS = {
    "gap": (
        "nll", "class_0_nll", "class_1_nll", "finite_density_count",
        "sampled_min", "sampled_max",
    ),
    "receiver": (
        "overall_nll", "head_nll", "tail_nll", "unk_nll",
        "repeat_nll", "new_nll", "head_count", "tail_count",
        "unk_count", "repeat_count", "new_count",
    ),
    "amount": ("normalized_mse", "decode_contract_sha256"),
    "fidelity": (
        "classwise_gap_ks", "full_receiver_tv", "head_receiver_tv",
        "tail_receiver_tv", "unk_rate_error", "short_gap_repeat_error",
    ),
    "runtime": (
        "requested_updates", "actual_updates", "elapsed_seconds",
        "peak_memory_bytes",
    ),
}

EXPECTED_CANDIDATES = (
    ("C0", "frozen_non_v3_cof_control", False, "FROZEN_REFERENCE_ONLY"),
    ("C1", "cof_ccmtpp_v1_causal_continuous_flat", True, "IMPLEMENTED_SOURCE_ONLY"),
    ("C2", "cof_ccmtpp_v1_sampled_gap_copy", True, "IMPLEMENTED_SOURCE_ONLY_GATED_ON_C1"),
    ("C3", "cof_ccmtpp_v1_hierarchical_receiver", True, "IMPLEMENTED_SOURCE_ONLY_GATED_ON_C2"),
    ("C4", "cof_ccmtpp_v1_y_balanced", True, "IMPLEMENTED_SOURCE_ONLY_GATED_ON_C3"),
)
EXPECTED_C5 = {
    "status": "LOCKED_UNIMPLEMENTED",
    "implementation_allowed_only_after": "C4_PASS",
    "selected_loss": None,
    "mutually_exclusive_options": [
        "class_conditional_gap_quantile_x_receiver_frequency_bin_pair_loss",
        "class_conditional_short_gap_x_repeat_coherence_loss",
    ],
    "simultaneous_losses": "FORBIDDEN",
}
EXPECTED_STOP_CRITERIA = {
    "C1": {
        "required_parent": "C0",
        "all_datasets_and_classes_required": True,
        "gap_fidelity": "current_gap_ks_le_parent",
        "coherence": "current_short_gap_repeat_error_le_parent",
    },
    "C2": {
        "required_parent": "C1",
        "all_datasets_and_classes_required": True,
        "short_gap_repeat": "current_strictly_less_parent",
        "receiver_tv": "current_le_parent",
    },
    "C3": {
        "required_parent": "C2",
        "all_datasets_required": True,
        "full_receiver_tv": "current_strictly_less_parent",
        "decomposition": "head_tail_unk_each_le_parent_and_one_strictly_less",
    },
    "C4": {
        "required_parent": "C3",
        "all_datasets_required": True,
        "y1_composite": "current_strictly_less_parent",
        "y0_composite_relative_tolerance": 0.05,
        "y0_zero_parent_rule": "current_must_equal_zero",
    },
}
FALSE_SAFETY_FLAGS = (
    "execution_authorized", "authorization_creation_authorized",
    "gpu_query_authorized", "cuda_authorized", "model_fit_authorized",
    "checkpoint_write_authorized", "model_sample_authorized",
    "evaluation_authorized", "selection_authorized",
    "internal_test_authorized", "sparkov_fraud_test_authorized",
    "tstr_authorized", "privacy_authorized", "full_run_authorized",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CCMTPPContractError(f"cannot hash preserved file: {path}") from error
    return digest.hexdigest()


@dataclass(frozen=True)
class CCMTPPCandidate:
    candidate_id: str
    model_id: str
    implemented: bool
    execute: bool
    status: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class CCMTPPDefinition:
    config_path: Path
    config_sha256: str
    repository_root: Path
    future_artifact_root: Path
    future_data_root: Path
    candidates: tuple[CCMTPPCandidate, ...]
    raw: Mapping[str, Any]


def validate_ccmtpp_definition(
    raw: Mapping[str, Any],
) -> tuple[CCMTPPCandidate, ...]:
    safety = raw.get("safety")
    common = raw.get("common_contract")
    model = raw.get("model")
    if (
        raw.get("schema_version") != "cof-ccmtpp-v1-source-only-v1"
        or raw.get("mode") != "SOURCE_ONLY"
        or raw.get("family") != "cof_ccmtpp_v1"
        or not isinstance(safety, Mapping)
        or any(safety.get(flag) is not False for flag in FALSE_SAFETY_FLAGS)
        or safety.get("entity_id_input") != "FORBIDDEN"
        or any(safety.get(key) != 0 for key in (
            "validation_fit_rows", "internal_test_fit_rows", "fraud_test_fit_rows"
        ))
    ):
        raise CCMTPPContractError("source-only safety boundary changed")
    if not isinstance(common, Mapping) or any((
        common.get("amount_contract") != "frozen_non_v3_train_only_encode_inverse_decode_v1",
        common.get("amount_architecture_redesign") is not False,
        common.get("gap_representation") != "continuous_log1p",
        common.get("gap_density") != "mixture_logistic",
        common.get("gap_support_max_rule") != "train_observed_max_valid_gap",
        common.get("gap_bin_classification") != "FORBIDDEN",
        common.get("receiver_vocabulary_fit_split") != "train",
        common.get("hierarchy_fit_split") != "train",
        common.get("receiver_pad_code") != 0,
        common.get("receiver_unk_code") != 1,
        common.get("sampled_gap_receiver_conditioning") != "model_sample_only",
        common.get("structure_loss") is not None,
    )):
        raise CCMTPPContractError("common scientific contract changed")
    hierarchy = common.get("receiver_hierarchy_rule")
    if not isinstance(hierarchy, Mapping) or hierarchy != {
        "algorithm": "descending_frequency_then_code_round_robin_tail",
        "head_min_count": 32,
        "max_head_categories": 512,
        "tail_cluster_count": 64,
    }:
        raise CCMTPPContractError("receiver hierarchy rule changed")
    if not isinstance(model, Mapping) or model != {
        "seed": 4001,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "max_length": 64,
        "dropout": 0.0,
        "gap_mixture_components": 5,
        "gap_max_seconds": "train_state_bound",
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 128,
        "requested_updates": 20000,
        "max_wall_seconds": 7200,
        "checkpoint_interval_updates": 1000,
        "early_stopping": "FORBIDDEN",
        "update_sweep": "FORBIDDEN",
    }:
        raise CCMTPPContractError("fixed training contract changed")
    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, list) or len(candidates_raw) != 5:
        raise CCMTPPContractError("finite candidate family changed")
    observed = tuple(
        (
            candidate.get("candidate_id"), candidate.get("model_id"),
            candidate.get("implemented"), candidate.get("status"),
        )
        for candidate in candidates_raw
    )
    if observed != EXPECTED_CANDIDATES or any(
        candidate.get("execute") is not False for candidate in candidates_raw
    ):
        raise CCMTPPContractError("finite candidate family changed")
    if candidates_raw[0].get("changes") != []:
        raise CCMTPPContractError("C0 must remain a frozen hash reference")
    expected_changes = {
        "C1": ["causal_decoder", "continuous_gap_density"],
        "C2": ["sampled_gap_receiver_conditioning", "causal_copy_gate"],
        "C3": ["train_only_head_tail_hierarchy"],
        "C4": ["y_balanced_conditional_likelihood"],
    }
    if any(
        candidate.get("changes") != expected_changes[candidate["candidate_id"]]
        for candidate in candidates_raw[1:]
    ):
        raise CCMTPPContractError("candidate factor isolation changed")
    if raw.get("deferred_C5") != EXPECTED_C5:
        raise CCMTPPContractError("C5 is not locked and unimplemented")
    diagnostic = raw.get("diagnostic_schema")
    if not isinstance(diagnostic, Mapping) or any(
        tuple(diagnostic.get(key, ())) != value for key, value in DIAGNOSTIC_FIELDS.items()
    ):
        raise CCMTPPContractError("candidate diagnostic schema changed")
    if raw.get("stop_criteria") != EXPECTED_STOP_CRITERIA:
        raise CCMTPPContractError("sequential stop criteria changed")
    return tuple(
        CCMTPPCandidate(
            candidate_id=str(candidate["candidate_id"]),
            model_id=str(candidate["model_id"]),
            implemented=bool(candidate["implemented"]),
            execute=bool(candidate["execute"]),
            status=str(candidate["status"]),
            raw=dict(candidate),
        )
        for candidate in candidates_raw
    )


def load_ccmtpp_definition(config_path: Path) -> CCMTPPDefinition:
    config_path = config_path.resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CCMTPPContractError(f"cannot read config: {config_path}") from error
    if not isinstance(raw, Mapping):
        raise CCMTPPContractError("config root must be a mapping")
    candidates = validate_ccmtpp_definition(raw)
    root = config_path.parents[2]
    return CCMTPPDefinition(
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        repository_root=root,
        future_artifact_root=(root / str(raw["future_artifact_root"])).resolve(),
        future_data_root=(root / str(raw["future_data_root"])).resolve(),
        candidates=candidates,
        raw=raw,
    )


def evaluate_candidate_stop(
    candidate: str,
    *,
    parent: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> Mapping[str, Any]:
    if candidate not in {"C1", "C2", "C3", "C4"} or parent is None:
        raise CCMTPPContractError("sequential parent result is required")
    checks: dict[str, bool]
    try:
        if candidate == "C1":
            parent_gap = parent["gap_ks"]
            current_gap = current["gap_ks"]
            checks = {
                "classwise_gap_noninferior": set(parent_gap) == set(current_gap)
                and all(current_gap[key] <= parent_gap[key] for key in parent_gap),
                "coherence_noninferior": current["short_gap_repeat_error"]
                <= parent["short_gap_repeat_error"],
            }
        elif candidate == "C2":
            checks = {
                "short_gap_repeat_improved": current["short_gap_repeat_error"]
                < parent["short_gap_repeat_error"],
                "receiver_tv_noninferior": current["receiver_tv"] <= parent["receiver_tv"],
            }
        elif candidate == "C3":
            components = ("head_tv", "tail_tv", "unk_error")
            checks = {
                "full_receiver_tv_improved": current["receiver_tv"] < parent["receiver_tv"],
                "decomposition_noninferior": all(
                    current[key] <= parent[key] for key in components
                ),
                "decomposition_has_improvement": any(
                    current[key] < parent[key] for key in components
                ),
            }
        else:
            parent_y0 = parent["y0_composite"]
            y0_limit = 0.0 if parent_y0 == 0 else parent_y0 * 1.05
            checks = {
                "y1_improved": current["y1_composite"] < parent["y1_composite"],
                "y0_within_preregistered_tolerance": current["y0_composite"] <= y0_limit,
            }
    except (KeyError, TypeError) as error:
        raise CCMTPPContractError("stop criterion evidence is incomplete") from error
    return {"candidate_id": candidate, "checks": checks, "passed": all(checks.values())}


def validate_ccmtpp_io_path(
    *,
    repository_root: Path,
    path: Path,
    purpose: str,
    access: str,
    source_only: bool,
) -> Path:
    root = repository_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root).as_posix().lower()
    except ValueError as error:
        raise CCMTPPContractError("path escapes repository") from error
    if any(token in relative for token in (
        "internal_test", "fraudtest", "fraud_test", "fresh_test"
    )):
        raise CCMTPPContractError("test/fraudTest access is forbidden")
    if purpose == "fit" and not relative.endswith("train.npz"):
        raise CCMTPPContractError("fit and hierarchy construction are train-only")
    if source_only and access != "read":
        raise CCMTPPContractError("source-only mode forbids artifact/data writes")
    return resolved
