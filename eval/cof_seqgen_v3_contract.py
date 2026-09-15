"""Fail-closed source and provenance contracts for CoF-SeqGen v3."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
import yaml

from generators.sampling_plan import SamplingPlan
from models.cof_seqgen_v3 import JointStateCodec


FROZEN_AMOUNT_CONTRACT = {
    "algorithm": "train_fitted_centered_empirical_residual",
    "quantile_grid_size": 257,
    "fit_split": "train",
    "validation_rows_used": 0,
    "test_rows_used": 0,
    "post_sample_channels_changed": ["amount"],
}
FROZEN_THRESHOLDS = {
    "amount_ks": 0.006081138155655141,
    "gap_ks": 0.006387882975686154,
    "amount_abs_standardized_label_effect": 0.0363693454591819,
    "gap_abs_standardized_label_effect": 0.051540527275560376,
    "receiver_max_abs_signed_frequency": 0.02,
}
FROZEN_SAMPLING_PLAN_SHA256 = (
    "862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27"
)
EXPECTED_SELECTION_RULE = {
    "eligibility": "all_five_row_marginal_guards_PASS",
    "tie_break": [
        "minimize_max_amount_ks_gap_ks",
        "minimize_sum_amount_ks_gap_ks",
        "lexicographic_candidate_id",
    ],
}
EXPECTED_CANDIDATES = (
    (
        "cof_v3_ref_v28_frozen",
        "frozen_reference",
        "hash_reference_only",
        False,
    ),
    (
        "cof_v3_c01_direct_joint",
        "direct_joint",
        "future_train_validation_only",
        True,
    ),
    (
        "cof_v3_c02_factorized_joint",
        "factorized_joint",
        "future_train_validation_only",
        True,
    ),
)
REQUIRED_FALSE_FLAGS = (
    "execution_authorized",
    "authorization_creation_authorized",
    "gpu_query_authorized",
    "cuda_authorized",
    "model_fit_authorized",
    "checkpoint_write_authorized",
    "sampling_authorized",
    "validation_execution_authorized",
    "selection_authorized",
    "test_authorized",
    "tstr_authorized",
    "privacy_authorized",
    "five_seed_full_run_authorized",
)


class V3ContractError(RuntimeError):
    pass


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise V3ContractError(f"cannot hash frozen file: {path}") from error
    return digest.hexdigest()


@dataclass(frozen=True)
class V3Candidate:
    candidate_id: str
    architecture: str
    execution_kind: str
    training_required: bool
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class V3Definition:
    config_path: Path
    config_sha256: str
    repository_root: Path
    future_artifact_root: Path
    candidates: tuple[V3Candidate, ...]
    raw: Mapping[str, Any]


def validate_v3_definition(
    raw: Mapping[str, Any],
) -> tuple[V3Candidate, ...]:
    if (
        raw.get("schema_version")
        != "cof-seqgen-v3-source-preparation-v1"
        or raw.get("mode") != "SOURCE_ONLY_IMPLEMENTATION"
        or raw.get("test_split_access") != "FORBIDDEN"
        or any(raw.get(flag) is not False for flag in REQUIRED_FALSE_FLAGS)
    ):
        raise V3ContractError("invalid v3 source-only safety boundary")
    frozen = raw.get("frozen_contract")
    if not isinstance(frozen, Mapping) or (
        frozen.get("thresholds") != FROZEN_THRESHOLDS
        or frozen.get("sampling_plan_sha256")
        != FROZEN_SAMPLING_PLAN_SHA256
        or frozen.get("amount_contract") != FROZEN_AMOUNT_CONTRACT
        or frozen.get("selection_rule") != EXPECTED_SELECTION_RULE
    ):
        raise V3ContractError(
            "frozen five-guard, plan, amount, or selection contract changed"
        )
    model = raw.get("model_contract")
    if not isinstance(model, Mapping) or any(
        (
            model.get("gap_bins") != 16,
            model.get("receiver_classes") != 64,
            model.get("d_model") != 128,
            model.get("n_heads") != 4,
            model.get("n_layers") != 2,
            model.get("requested_updates") != 20000,
            model.get("max_wall_seconds") != 7200,
            model.get("paired_discrete_mask") is not True,
            model.get("coherence_lambda") != 0.0,
            model.get("post_hoc_calibration") != "FORBIDDEN",
            model.get("legacy_independent_heads") != "FORBIDDEN",
            model.get("early_stopping") != "FORBIDDEN",
            model.get("update_sweep") != "FORBIDDEN",
        )
    ):
        raise V3ContractError("v3 fixed model contract changed")
    raw_candidates = raw.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 3:
        raise V3ContractError("v3 finite candidate family changed")
    observed = tuple(
        (
            candidate.get("candidate_id"),
            candidate.get("architecture"),
            candidate.get("execution_kind"),
            candidate.get("training_required"),
        )
        for candidate in raw_candidates
    )
    if observed != EXPECTED_CANDIDATES:
        raise V3ContractError("v3 finite candidate family changed")
    reference, direct, factorized = raw_candidates
    if (
        reference.get("sampling_required") is not False
        or reference.get("validation_required") is not False
        or direct.get("changed_factor")
        != "joint_discrete_parameterization"
        or direct.get("joint_output") != "one_cartesian_joint_head"
        or direct.get("training_receiver_condition") != "none"
        or direct.get("sampling_receiver_condition") != "same_joint_state"
        or factorized.get("changed_factor")
        != "joint_discrete_parameterization"
        or factorized.get("joint_output")
        != "gap_head_then_gap_conditional_receiver_head"
        or factorized.get("training_receiver_condition") != "true_gap"
        or factorized.get("sampling_receiver_condition") != "sampled_gap"
    ):
        raise V3ContractError("v3 joint architecture contract changed")
    if any(
        candidate.get("joint_input")
        != "one_joint_embedding_and_one_joint_MASK"
        for candidate in (direct, factorized)
    ):
        raise V3ContractError("v3 paired joint input contract changed")
    return tuple(
        V3Candidate(
            candidate_id=str(candidate["candidate_id"]),
            architecture=str(candidate["architecture"]),
            execution_kind=str(candidate["execution_kind"]),
            training_required=bool(candidate["training_required"]),
            raw=dict(candidate),
        )
        for candidate in raw_candidates
    )


def load_v3_definition(config_path: Path) -> V3Definition:
    config_path = config_path.resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise V3ContractError(f"cannot read v3 config: {config_path}") from error
    if not isinstance(raw, Mapping):
        raise V3ContractError("v3 config root must be a mapping")
    candidates = validate_v3_definition(raw)
    repository_root = config_path.parents[2]
    return V3Definition(
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        repository_root=repository_root,
        future_artifact_root=(
            repository_root / str(raw["future_artifact_root"])
        ).resolve(),
        candidates=candidates,
        raw=raw,
    )


def _validate_sha256_mapping(provenance: Mapping[str, Any]) -> None:
    expected = {
        "train_file_sha256",
        "train_content_sha256",
        "sampling_plan_sha256",
    }
    if set(provenance) != expected or any(
        not isinstance(provenance[key], str)
        or len(provenance[key]) != 64
        for key in expected
    ):
        raise V3ContractError("train provenance is incomplete")


def build_train_joint_support_state(
    *,
    codec: JointStateCodec,
    gap: Tensor,
    receiver: Tensor,
    valid_mask: Tensor,
    fit_split: str,
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Fit the immutable global pair-support mask from valid train rows."""

    if fit_split != "train":
        raise V3ContractError("joint support fitting is train-only")
    _validate_sha256_mapping(provenance)
    if (
        valid_mask.dtype != torch.bool
        or gap.shape != receiver.shape
        or gap.shape != valid_mask.shape
    ):
        raise V3ContractError("joint support tensor shapes are invalid")
    joint = codec.encode(gap, receiver)
    observed = sorted(
        int(value)
        for value in torch.unique(joint[valid_mask]).cpu().tolist()
    )
    support_mask = [
        state in set(observed)
        for state in range(codec.state_count)
    ]
    payload = {
        "schema_version": "cof-seqgen-v3-joint-support-v1",
        "fit_split": "train",
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "gap_bins": codec.gap_bins,
        "receiver_classes": codec.receiver_classes,
        "observed_joint_states": observed,
        "support_mask": support_mask,
        "provenance": dict(provenance),
    }
    return {**payload, "state_sha256": canonical_sha256(payload)}


def bind_train_only_conditioning_contract(
    *,
    plan: SamplingPlan,
    expected_plan_sha256: str,
    amount_contract: Mapping[str, Any],
    fit_split: str,
) -> Mapping[str, Any]:
    """Bind Y/L/mask and the unchanged v2.8 amount path before training."""

    if fit_split != "train":
        raise V3ContractError("conditioning and amount fitting is train-only")
    if plan.plan_hash != expected_plan_sha256:
        raise V3ContractError("SamplingPlan hash mismatch")
    if dict(amount_contract) != FROZEN_AMOUNT_CONTRACT:
        raise V3ContractError("frozen amount contract changed")
    if (
        plan.y_entity.ndim != 1
        or plan.lengths.shape != plan.y_entity.shape
        or plan.valid_mask.ndim != 2
        or plan.valid_mask.shape[0] != len(plan.lengths)
        or not np.isin(plan.y_entity, (0, 1)).all()
    ):
        raise V3ContractError("SamplingPlan Y/L shapes are invalid")
    expected_mask = (
        np.arange(plan.valid_mask.shape[1])[None, :]
        < plan.lengths[:, None]
    )
    if not np.array_equal(plan.valid_mask, expected_mask):
        raise V3ContractError("SamplingPlan valid mask is not a prefix mask")
    payload = {
        "schema_version": "cof-seqgen-v3-conditioning-binding-v1",
        "fit_split": "train",
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "sampling_plan_sha256": plan.plan_hash,
        "entity_count": int(len(plan.lengths)),
        "labels": plan.y_entity.astype(np.int64).tolist(),
        "lengths": plan.lengths.astype(np.int64).tolist(),
        "valid_mask_sha256": hashlib.sha256(
            plan.valid_mask.astype(np.bool_).tobytes()
        ).hexdigest(),
        "amount_contract": dict(FROZEN_AMOUNT_CONTRACT),
    }
    return {**payload, "binding_sha256": canonical_sha256(payload)}


def validate_v3_io_path(
    *,
    repository_root: Path,
    path: Path,
    access: str,
    purpose: str,
    source_only: bool,
) -> Path:
    """Enforce split and append-only boundaries before any future I/O."""

    root = repository_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise V3ContractError("v3 path escapes the repository") from error
    lowered = tuple(part.lower() for part in relative.parts)
    if any(
        part in {"test", "tests", "fresh_test", "fresh-test"}
        or part.startswith("test.")
        for part in lowered
    ):
        raise V3ContractError("test and fresh-test access is forbidden")
    if access == "read":
        if purpose == "fit" and (
            relative.name != "train.npz"
            or "data" not in lowered
            or "frozen" not in lowered
        ):
            raise V3ContractError("v3 fit input is train-only")
        return resolved
    if access != "write":
        raise V3ContractError("v3 access must be read or write")
    if source_only:
        raise V3ContractError("source-only phase cannot write artifacts")
    future_root = (
        root / "artifacts/benchmark_v3/candidate_selection"
    ).resolve()
    try:
        resolved.relative_to(future_root)
    except ValueError as error:
        raise V3ContractError(
            "frozen data and earlier benchmark artifacts are read-only"
        ) from error
    return resolved
