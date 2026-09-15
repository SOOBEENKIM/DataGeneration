from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


MODEL_IDS = (
    "ctgan_separate_class",
    "tvae_separate_class",
    "cof_seqgen",
)
EXPECTED_CANDIDATES = {
    "ctgan_separate_class": {
        "ctgan_v28_c00_frozen_amount_inverse": "none",
        "ctgan_v28_c01_joint_gap_receiver_decoder": (
            "joint_gap_receiver_discrete_decoder"
        ),
    },
    "tvae_separate_class": {
        "tvae_v28_c00_frozen_selected_amount_inverse": "none",
    },
    "cof_seqgen": {
        "cof_v28_c00_frozen_empirical_residual": "none",
        "cof_v28_c01_gap_distribution_sampler": (
            "gap_distribution_sampler"
        ),
    },
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
REQUIRED_FALSE_FLAGS = (
    "execution_authorized",
    "authorization_creation_authorized",
    "gpu_query_authorized",
    "cuda_authorized",
    "training_authorized",
    "checkpoint_update_authorized",
    "sampling_authorized",
    "evaluation_authorized",
    "selection_authorized",
    "fresh_test_authorized",
    "tstr_authorized",
    "privacy_authorized",
    "five_seed_full_run_authorized",
)
EXPECTED_FIT_PARAMETERS = {
    "ctgan_v28_c01_joint_gap_receiver_decoder": {
        "rule": "class_conditional_joint_gap_receiver_logit_residual",
        "base_distribution": "parent_checkpoint_train_plan_output",
        "target_distribution": "valid_train_rows",
        "additive_smoothing": 0.5,
        "logit_offset_clip": [-2.0, 2.0],
        "support": "observed_train_joint_pairs",
    },
    "cof_v28_c01_gap_distribution_sampler": {
        "rule": "class_conditional_monotone_gap_distribution_transport",
        "base_distribution": "parent_checkpoint_train_plan_output",
        "target_distribution": "valid_train_rows",
        "support": "frozen_train_gap_bins",
        "interpolation": "discrete_monotone_quantile_transport",
        "tie_break": "lower_gap_bin",
    },
}
EXPECTED_FUTURE_EVALUATION_CONTRACT = {
    "checkpoint_access": "READ_ONLY",
    "existing_checkpoint_required": True,
    "optimizer_updates": 0,
    "training_calls": 0,
    "checkpoint_writes": 0,
    "validation_sample_generation": (
        "FUTURE_SEPARATE_AUTHORIZATION_ONLY"
    ),
    "evaluation": "FUTURE_SEPARATE_AUTHORIZATION_ONLY",
    "selection": "FUTURE_SEPARATE_AUTHORIZATION_ONLY",
    "append_only_artifacts": True,
}
EXPECTED_FIT_STATE_CONTRACT = {
    "fit_split": "train",
    "fit_input": "valid_train_rows_and_fixed_train_only_sampling_plan",
    "validation_rows_used": 0,
    "test_rows_used": 0,
    "hash_algorithm": "sha256_canonical_json",
    "source_config_data_plan_parent_provenance_required": True,
    "state_destination": "candidate_artifact",
}
EXPECTED_PARENT_LINEAGE = {
    "ctgan_separate_class": (
        "ctgan_v27_c01_amount_quantile_inverse",
        "amount_inverse_map",
    ),
    "tvae_separate_class": (
        "tvae_v27_c01_amount_inverse_decoder",
        "amount_inverse_decoder",
    ),
    "cof_seqgen": (
        "cof_v27_c01_empirical_residual",
        "amount_residual_sampler",
    ),
}


class V28PreparationContractError(RuntimeError):
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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class V28Candidate:
    model_id: str
    candidate_id: str
    execution_kind: str
    factor: str
    changed_dimensions: tuple[str, ...]
    baseline_dimensions: Mapping[str, Any]
    effective_dimensions: Mapping[str, Any]
    fit_parameters: Mapping[str, Any]
    frozen_parent: Mapping[str, Any]
    future_sampling_required: bool

    @property
    def is_control(self) -> bool:
        return self.execution_kind == "reuse_frozen_parent"


@dataclass(frozen=True)
class V28Definition:
    config_path: Path
    config_sha256: str
    repository_root: Path
    future_artifact_root: Path
    candidates: tuple[V28Candidate, ...]
    raw: Mapping[str, Any]

    @property
    def model_ids(self) -> tuple[str, ...]:
        return MODEL_IDS


def _changed_dimensions(
    baseline: Mapping[str, Any],
    effective: Mapping[str, Any],
) -> tuple[str, ...]:
    if set(baseline) != set(effective):
        raise V28PreparationContractError(
            "candidate dimension key set changed"
        )
    return tuple(
        key
        for key in baseline
        if canonical_sha256(baseline[key])
        != canonical_sha256(effective[key])
    )


def validate_v28_definition(
    raw: Mapping[str, Any],
) -> tuple[V28Candidate, ...]:
    if (
        raw.get("schema_version")
        != "benchmark-v2.8-source-amendment-v1"
        or raw.get("mode") != "SOURCE_ONLY_PREPARATION"
        or raw.get("test_split_access") != "FORBIDDEN"
        or any(raw.get(flag) is not False for flag in REQUIRED_FALSE_FLAGS)
    ):
        raise V28PreparationContractError(
            "invalid v2.8 source-amendment schema"
        )
    frozen = raw.get("frozen_contract")
    if (
        not isinstance(frozen, Mapping)
        or frozen.get("thresholds") != FROZEN_THRESHOLDS
        or frozen.get("sampling_plan_sha256")
        != FROZEN_SAMPLING_PLAN_SHA256
    ):
        raise V28PreparationContractError(
            "frozen threshold or SamplingPlan changed"
        )
    if (
        raw.get("future_evaluation_contract")
        != EXPECTED_FUTURE_EVALUATION_CONTRACT
    ):
        raise V28PreparationContractError(
            "future evaluation zero-update contract changed"
        )
    if raw.get("fit_state_contract") != EXPECTED_FIT_STATE_CONTRACT:
        raise V28PreparationContractError(
            "train-only fit-state contract changed"
        )
    models = raw.get("models")
    if not isinstance(models, Mapping) or tuple(models) != MODEL_IDS:
        raise V28PreparationContractError(
            "v2.8 model family changed"
        )
    candidates: list[V28Candidate] = []
    for model_id in MODEL_IDS:
        model = models[model_id]
        expected = EXPECTED_CANDIDATES[model_id]
        raw_candidates = model.get("candidates")
        if not isinstance(raw_candidates, list):
            raise V28PreparationContractError(
                f"candidate list missing: {model_id}"
            )
        observed = {
            str(item.get("candidate_id")): str(item.get("factor"))
            for item in raw_candidates
        }
        if observed != expected:
            raise V28PreparationContractError(
                f"finite candidate family changed: {model_id}"
            )
        baseline = model.get("baseline_dimensions")
        parent = model.get("parent")
        if not isinstance(baseline, Mapping) or not isinstance(
            parent, Mapping
        ):
            raise V28PreparationContractError(
                f"candidate lineage missing: {model_id}"
            )
        if (
            parent.get("candidate_id"),
            parent.get("factor"),
        ) != EXPECTED_PARENT_LINEAGE[model_id]:
            raise V28PreparationContractError(
                f"frozen parent lineage changed: {model_id}"
            )
        for item in raw_candidates:
            effective = item.get("effective_dimensions")
            if not isinstance(effective, Mapping):
                raise V28PreparationContractError(
                    f"effective dimensions missing: {model_id}"
                )
            factor = str(item["factor"])
            changed = _changed_dimensions(baseline, effective)
            expected_changed = () if factor == "none" else (factor,)
            if changed != expected_changed:
                raise V28PreparationContractError(
                    "candidate must change exactly its sole factor"
                )
            expected_parameters = EXPECTED_FIT_PARAMETERS.get(
                str(item["candidate_id"]),
                {},
            )
            if (
                item.get("fit_parameters", {}) != expected_parameters
                or (
                    factor != "none"
                    and item.get("calibration_components") != 1
                )
                or (
                    factor == "none"
                    and "calibration_components" in item
                )
            ):
                raise V28PreparationContractError(
                    "preregistered single-factor parameters changed"
                )
            candidates.append(
                V28Candidate(
                    model_id=model_id,
                    candidate_id=str(item["candidate_id"]),
                    execution_kind=str(item["execution_kind"]),
                    factor=factor,
                    changed_dimensions=changed,
                    baseline_dimensions=dict(baseline),
                    effective_dimensions=dict(effective),
                    fit_parameters=dict(item.get("fit_parameters", {})),
                    frozen_parent=dict(parent),
                    future_sampling_required=bool(
                        item.get("future_sampling_required")
                    ),
                )
            )
    return tuple(candidates)


def load_v28_definition(config_path: Path) -> V28Definition:
    config_path = config_path.resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise V28PreparationContractError(
            f"cannot read v2.8 amendment: {config_path}"
        ) from error
    if not isinstance(raw, Mapping):
        raise V28PreparationContractError(
            "v2.8 amendment root must be a mapping"
        )
    candidates = validate_v28_definition(raw)
    repository_root = config_path.parents[2]
    future_artifact_root = (
        repository_root / str(raw["future_artifact_root"])
    ).resolve()
    return V28Definition(
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        repository_root=repository_root,
        future_artifact_root=future_artifact_root,
        candidates=candidates,
        raw=raw,
    )


def build_v28_fit_state(
    *,
    candidate: V28Candidate,
    parameters: Mapping[str, Any],
    source_commit: str,
    config_sha256: str,
    train_file_sha256: str,
    train_content_sha256: str,
    sampling_plan_sha256: str,
) -> Mapping[str, Any]:
    if candidate.is_control:
        raise V28PreparationContractError(
            "frozen control cannot create fit state"
        )
    parent = candidate.frozen_parent
    payload = {
        "schema_version": "benchmark-v2.8-train-fit-state-v1",
        "candidate_id": candidate.candidate_id,
        "model_id": candidate.model_id,
        "factor": candidate.factor,
        "fit_split": "train",
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "parameters": dict(parameters),
        "parameter_sha256": canonical_sha256(parameters),
        "provenance": {
            "source_commit": source_commit,
            "config_sha256": config_sha256,
            "train_file_sha256": train_file_sha256,
            "train_content_sha256": train_content_sha256,
            "sampling_plan_sha256": sampling_plan_sha256,
        },
        "parent_provenance": {
            "candidate_id": str(parent["candidate_id"]),
            "checkpoint_sha256": str(parent["checkpoint_sha256"]),
            "validation_sample_sha256": str(
                parent["validation_sample_sha256"]
            ),
            "manifest_sha256": str(parent["manifest_sha256"]),
            "source_commit": str(parent["source_commit"]),
        },
    }
    return {**payload, "state_sha256": canonical_sha256(payload)}


def validate_v28_fit_state(
    *,
    state: Mapping[str, Any],
    candidate: V28Candidate,
    source_commit: str,
    config_sha256: str,
    train_file_sha256: str,
    train_content_sha256: str,
    sampling_plan_sha256: str,
) -> None:
    rebuilt = build_v28_fit_state(
        candidate=candidate,
        parameters=state.get("parameters", {}),
        source_commit=source_commit,
        config_sha256=config_sha256,
        train_file_sha256=train_file_sha256,
        train_content_sha256=train_content_sha256,
        sampling_plan_sha256=sampling_plan_sha256,
    )
    if canonical_sha256(state) != canonical_sha256(rebuilt):
        raise V28PreparationContractError(
            "v2.8 fit-state provenance or hash mismatch"
        )


def validate_v28_io_path(
    *,
    repository_root: Path,
    path: Path,
    access: str,
) -> Path:
    root = repository_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise V28PreparationContractError(
            "v2.8 path escapes repository"
        ) from error
    lowered = tuple(part.lower() for part in relative.parts)
    if any(
        part in {"test", "tests", "fresh_test", "fresh-test"}
        or part.startswith("test.")
        for part in lowered
    ):
        raise V28PreparationContractError(
            "test and fresh-test access is forbidden"
        )
    if access == "read":
        return resolved
    if access != "write":
        raise V28PreparationContractError(
            "v2.8 path access must be read or write"
        )
    future_root = (
        root / "artifacts/benchmark_v2_8/candidate_selection"
    ).resolve()
    try:
        resolved.relative_to(future_root)
    except ValueError as error:
        raise V28PreparationContractError(
            "frozen data and v2.5/v2.6/v2.7 artifacts are read-only"
        ) from error
    return resolved
