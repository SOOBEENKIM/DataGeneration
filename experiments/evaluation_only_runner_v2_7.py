from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import multiprocessing
from pathlib import Path
import queue
import subprocess
import time
import traceback
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from eval.candidate_preparation_v2_7 import (
    FROZEN_SAMPLING_PLAN_SHA256,
    FROZEN_THRESHOLDS,
    V27Candidate,
    V27PreparationPlan,
    canonical_sha256,
    load_and_validate_preparation,
    sha256_file,
    validate_candidate_io_path,
)


MODEL_IDS = (
    "ctgan_separate_class",
    "tvae_separate_class",
    "cof_seqgen",
)
EXPECTED_RESTORE_MODES = {
    "ctgan_separate_class": (
        "cpu_first_then_explicit_selected_device"
    ),
    "tvae_separate_class": (
        "cpu_first_then_explicit_selected_device"
    ),
    "cof_seqgen": "selected_device_restore",
}
RELEVANT_SOURCE_PATHS = (
    "configs/benchmark_v2/evaluation_only_v2_7.yaml",
    "configs/benchmark_v2/selection_v2_7_source_preparation.yaml",
    "eval/candidate_preparation_v2_7.py",
    "eval/evaluation_only_v2_7.py",
    "experiments/evaluation_only_runner_v2_7.py",
    "generators/evaluation_only_transforms_v2_7.py",
    "models/evaluation_only_components_v2_7.py",
    "scripts/run_evaluation_only_v2_7.py",
)
ZERO_EXECUTION_COUNTS = {
    "gpu_queries": 0,
    "cuda_calls": 0,
    "model_fit_calls": 0,
    "model_sample_calls": 0,
    "optimizer_updates": 0,
    "validation_selection_runs": 0,
    "fresh_test_runs": 0,
    "five_seed_full_runs": 0,
}


class EvaluationOnlyContractError(RuntimeError):
    pass


class EvaluationOnlyArtifactStore:
    def __init__(self, repository_root: Path):
        self.repository_root = repository_root.resolve()
        self.root = (
            self.repository_root
            / "artifacts/benchmark_v2_7/candidate_selection"
        )

    def claim_model(
        self,
        *,
        model_id: str,
        provenance_sha256: str,
    ) -> Path:
        if model_id not in MODEL_IDS or len(provenance_sha256) != 64:
            raise EvaluationOnlyContractError(
                "invalid evaluation worker ownership claim"
            )
        worker_root = self.root / "workers" / model_id
        worker_root.mkdir(parents=True, exist_ok=True)
        claim = worker_root / "ownership.lock"
        try:
            with claim.open("x", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": (
                            "benchmark-v2.7-evaluation-owner-v1"
                        ),
                        "model_id": model_id,
                        "provenance_sha256": provenance_sha256,
                    },
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
        except FileExistsError as error:
            raise EvaluationOnlyContractError(
                f"evaluation model is already owned: {model_id}"
            ) from error
        return claim

    def create_evaluation_attempt(
        self,
        *,
        model_id: str,
        candidate_id: str,
        seed: int,
    ) -> Path:
        if (
            model_id not in MODEL_IDS
            or not candidate_id.startswith(
                {
                    "ctgan_separate_class": "ctgan_v27_",
                    "tvae_separate_class": "tvae_v27_",
                    "cof_seqgen": "cof_v27_",
                }[model_id]
            )
            or not (
                self.root
                / "workers"
                / model_id
                / "ownership.lock"
            ).is_file()
        ):
            raise EvaluationOnlyContractError(
                "evaluation attempt requires its exclusive model owner"
            )
        parent = (
            self.root
            / "evaluations"
            / model_id
            / candidate_id
            / f"seed_{seed}"
        )
        parent.mkdir(parents=True, exist_ok=True)
        for number in range(1, 1_000_000):
            attempt = parent / f"attempt_{number:03d}"
            try:
                attempt.mkdir()
            except FileExistsError:
                continue
            return attempt
        raise EvaluationOnlyContractError(
            "append-only evaluation attempt namespace exhausted"
        )

    def write_json(
        self,
        path: Path,
        value: Mapping[str, Any],
    ) -> None:
        path = path.resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as error:
            raise EvaluationOnlyContractError(
                "evaluation artifact escapes its append-only root"
            ) from error
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(
                    value,
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
        except FileExistsError as error:
            raise EvaluationOnlyContractError(
                f"append-only evaluation artifact exists: {path}"
            ) from error


@dataclass(frozen=True)
class EvaluationOperation:
    model_id: str
    candidate_id: str
    factor: str
    execution_kind: str
    checkpoint_path: str
    checkpoint_sha256: str
    restore_mode: str
    definition_sha256: str
    training_required: bool
    sampling_required: bool
    fit_state_required: bool

    @property
    def is_control(self) -> bool:
        return self.execution_kind == "reuse_frozen_control"


@dataclass(frozen=True)
class EvaluationExecutionPlan:
    repository_root: Path
    runner_config_path: Path
    runner_config_sha256: str
    candidate_plan: V27PreparationPlan
    runtime_root: Path
    operations: tuple[EvaluationOperation, ...]
    execution_contract: Mapping[str, Any]
    validation_contract: Mapping[str, Any]


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EvaluationOnlyContractError(
            f"cannot read v2.7 evaluation runner config: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise EvaluationOnlyContractError(
            "v2.7 evaluation runner config root must be a mapping"
        )
    return value


def _operation_definition(candidate: V27Candidate) -> Mapping[str, Any]:
    return {
        "model_id": candidate.model_id,
        "candidate_id": candidate.candidate_id,
        "factor": candidate.factor,
        "execution_kind": candidate.execution_kind,
        "changed_dimensions": list(candidate.changed_dimensions),
        "fit_parameters": dict(candidate.fit_parameters),
        "checkpoint_sha256": candidate.frozen_control[
            "checkpoint_sha256"
        ],
    }


def build_evaluation_execution_plan(
    runner_config_path: Path,
) -> EvaluationExecutionPlan:
    runner_config_path = runner_config_path.resolve()
    root = runner_config_path.parents[2]
    raw = _read_yaml(runner_config_path)
    false_flags = (
        "execution_authorized",
        "authorization_creation_authorized",
        "gpu_query_authorized",
        "cuda_authorized",
        "training_authorized",
        "sampling_authorized",
        "validation_selection_authorized",
        "fresh_test_authorized",
        "tstr_authorized",
        "privacy_authorized",
        "five_seed_full_run_authorized",
    )
    if (
        raw.get("schema_version")
        != "benchmark-v2.7-evaluation-only-runner-v1"
        or raw.get("mode") != "SOURCE_ONLY_IMPLEMENTATION"
        or raw.get("authorization_required_for_execute") is not True
        or raw.get("test_split_access") != "FORBIDDEN"
        or any(raw.get(key) is not False for key in false_flags)
    ):
        raise EvaluationOnlyContractError(
            "v2.7 evaluation authorization boundary changed"
        )
    candidate_config = (
        root / str(raw["candidate_config_path"])
    ).resolve()
    if sha256_file(candidate_config) != raw.get(
        "candidate_config_sha256"
    ):
        raise EvaluationOnlyContractError(
            "v2.7 candidate config provenance mismatch"
        )
    candidate_plan = load_and_validate_preparation(candidate_config)
    restore_contract = raw.get("restore_contract")
    execution_contract = raw.get("execution_contract")
    validation_contract = raw.get("validation_contract")
    if (
        not isinstance(restore_contract, Mapping)
        or tuple(restore_contract) != MODEL_IDS
        or not isinstance(execution_contract, Mapping)
        or not isinstance(validation_contract, Mapping)
    ):
        raise EvaluationOnlyContractError(
            "v2.7 evaluation runner contract is incomplete"
        )
    if (
        validation_contract.get("sampling_plan_sha256")
        != FROZEN_SAMPLING_PLAN_SHA256
        or validation_contract.get("thresholds")
        != FROZEN_THRESHOLDS
        or validation_contract.get(
            "candidate_selection_inside_worker"
        )
        != "FORBIDDEN"
        or validation_contract.get(
            "result_contingent_candidate_change"
        )
        != "FORBIDDEN"
    ):
        raise EvaluationOnlyContractError(
            "v2.7 validation threshold, SamplingPlan, or selection "
            "boundary changed"
        )
    operations: list[EvaluationOperation] = []
    for candidate in candidate_plan.candidates:
        restore = restore_contract[candidate.model_id]
        frozen = candidate.frozen_control
        if (
            restore.get("mode")
            != EXPECTED_RESTORE_MODES[candidate.model_id]
            or restore.get("checkpoint_path")
            != frozen["checkpoint_path"]
            or restore.get("checkpoint_sha256")
            != frozen["checkpoint_sha256"]
        ):
            raise EvaluationOnlyContractError(
                f"restore provenance changed: {candidate.model_id}"
            )
        operations.append(
            EvaluationOperation(
                model_id=candidate.model_id,
                candidate_id=candidate.candidate_id,
                factor=candidate.factor,
                execution_kind=candidate.execution_kind,
                checkpoint_path=str(restore["checkpoint_path"]),
                checkpoint_sha256=str(
                    restore["checkpoint_sha256"]
                ),
                restore_mode=str(restore["mode"]),
                definition_sha256=canonical_sha256(
                    _operation_definition(candidate)
                ),
                training_required=False,
                sampling_required=not candidate.is_control,
                fit_state_required=not candidate.is_control,
            )
        )
    if (
        len(operations) != 9
        or sum(operation.is_control for operation in operations) != 3
        or sum(
            operation.sampling_required for operation in operations
        )
        != 6
        or any(operation.training_required for operation in operations)
    ):
        raise EvaluationOnlyContractError(
            "v2.7 execution operation family changed"
        )
    return EvaluationExecutionPlan(
        repository_root=root,
        runner_config_path=runner_config_path,
        runner_config_sha256=sha256_file(runner_config_path),
        candidate_plan=candidate_plan,
        runtime_root=(
            root / str(raw["future_runtime_root"])
        ).resolve(),
        operations=tuple(operations),
        execution_contract=dict(execution_contract),
        validation_contract=dict(validation_contract),
    )


def validate_checkpoint_provenance(
    plan: EvaluationExecutionPlan,
    operation: EvaluationOperation,
) -> Path:
    expected = [
        planned
        for planned in plan.operations
        if (
            planned.model_id,
            planned.candidate_id,
        )
        == (
            operation.model_id,
            operation.candidate_id,
        )
    ]
    if (
        len(expected) != 1
        or expected[0].checkpoint_path != operation.checkpoint_path
        or expected[0].checkpoint_sha256
        != operation.checkpoint_sha256
    ):
        raise EvaluationOnlyContractError(
            "checkpoint provenance differs from the frozen plan"
        )
    path = (
        plan.repository_root / operation.checkpoint_path
    ).resolve()
    try:
        path.relative_to(plan.repository_root)
    except ValueError as error:
        raise EvaluationOnlyContractError(
            "checkpoint provenance escapes the repository"
        ) from error
    if (
        not path.is_file()
        or sha256_file(path) != operation.checkpoint_sha256
    ):
        raise EvaluationOnlyContractError(
            "checkpoint provenance hash mismatch"
        )
    return path


def _candidate_definition(
    plan: EvaluationExecutionPlan,
    operation: EvaluationOperation,
) -> V27Candidate:
    matches = [
        candidate
        for candidate in plan.candidate_plan.candidates
        if (
            candidate.model_id,
            candidate.candidate_id,
        )
        == (
            operation.model_id,
            operation.candidate_id,
        )
    ]
    if len(matches) != 1:
        raise EvaluationOnlyContractError(
            "candidate definition does not resolve exactly once"
        )
    return matches[0]


def build_control_reference(
    *,
    plan: EvaluationExecutionPlan,
    operation: EvaluationOperation,
    source_commit: str,
    relevant_source_sha256: str,
) -> Mapping[str, Any]:
    if (
        not operation.is_control
        or len(source_commit) != 40
        or len(relevant_source_sha256) != 64
    ):
        raise EvaluationOnlyContractError(
            "invalid frozen-control reference request"
        )
    validate_checkpoint_provenance(plan, operation)
    candidate = _candidate_definition(plan, operation)
    frozen = dict(candidate.frozen_control)
    for role in ("candidate_result", "validation_sample"):
        path = (
            plan.repository_root / str(frozen[f"{role}_path"])
        ).resolve()
        if (
            not path.is_file()
            or sha256_file(path) != frozen[f"{role}_sha256"]
        ):
            raise EvaluationOnlyContractError(
                f"frozen control {role} hash mismatch"
            )
    return {
        "schema_version": "benchmark-v2.7-frozen-control-reference-v1",
        "status": "COMPLETE",
        "source_commit": source_commit,
        "relevant_source_sha256": relevant_source_sha256,
        "runner_config_sha256": plan.runner_config_sha256,
        "candidate_config_sha256": (
            plan.candidate_plan.config_sha256
        ),
        "model_id": operation.model_id,
        "candidate_id": operation.candidate_id,
        "operation_definition_sha256": (
            operation.definition_sha256
        ),
        "frozen_control": frozen,
        "training_calls": 0,
        "optimizer_updates": 0,
        "checkpoint_training_calls": 0,
        "sampling_calls": 0,
        "fit_state_created": False,
        "test_split_read": False,
        "validation_selection_executed": False,
    }


def validate_operation_single_factor(
    plan: EvaluationExecutionPlan,
    operation: EvaluationOperation,
) -> None:
    candidate = _candidate_definition(plan, operation)
    changed = candidate.changed_dimensions
    expected_definition = canonical_sha256(
        _operation_definition(candidate)
    )
    if operation.is_control:
        valid = (
            candidate.is_control
            and operation.factor == "none"
            and changed == ()
            and not operation.sampling_required
        )
    else:
        valid = (
            not candidate.is_control
            and changed == (candidate.factor,)
            and operation.factor == candidate.factor
            and operation.sampling_required
            and operation.fit_state_required
        )
    if (
        not valid
        or operation.training_required
        or operation.definition_sha256 != expected_definition
    ):
        raise EvaluationOnlyContractError(
            "evaluation operation is not the frozen single factor"
        )


def restore_checkpoint_for_evaluation(
    *,
    plan: EvaluationExecutionPlan,
    operation: EvaluationOperation,
    device: str,
    train: Any,
    loader: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    if not device.startswith("cuda:"):
        raise EvaluationOnlyContractError(
            "future evaluation restore requires an explicit CUDA device"
        )
    validate_operation_single_factor(plan, operation)
    checkpoint = validate_checkpoint_provenance(plan, operation)
    if loader is None:
        from experiments.candidate_runner_v2_6 import (
            restore_candidate_checkpoint_for_sampling,
        )

        loader = restore_candidate_checkpoint_for_sampling
    bundle = loader(
        checkpoint,
        model_id=operation.model_id,
        device=device,
        train=train,
    )
    if not isinstance(bundle, Mapping) or bundle.get("backend") is None:
        raise EvaluationOnlyContractError(
            "restored checkpoint bundle has no backend"
        )
    return {
        "checkpoint_bundle": bundle,
        "checkpoint_path": str(
            checkpoint.relative_to(plan.repository_root)
        ),
        "checkpoint_sha256": operation.checkpoint_sha256,
        "restore_mode": operation.restore_mode,
        "training_calls": 0,
        "optimizer_updates": 0,
        "checkpoint_training_calls": 0,
    }


def validate_evaluation_boundary(
    *,
    plan: EvaluationExecutionPlan,
    data_paths: tuple[Path, ...],
    thresholds: Mapping[str, float],
    sampling_plan_sha256: str,
    selection_requested: bool,
    candidate_overrides: Mapping[str, Any],
) -> None:
    if (
        dict(thresholds) != FROZEN_THRESHOLDS
        or sampling_plan_sha256 != FROZEN_SAMPLING_PLAN_SHA256
        or selection_requested
        or bool(candidate_overrides)
    ):
        raise EvaluationOnlyContractError(
            "threshold, SamplingPlan, selection, or candidate override "
            "request is forbidden"
        )
    expected = {
        (
            plan.repository_root
            / "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
            "kappa_1.00/train.npz"
        ).resolve(),
        (
            plan.repository_root
            / "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
            "kappa_1.00/validation.npz"
        ).resolve(),
    }
    resolved = set()
    for path in data_paths:
        try:
            validated = validate_candidate_io_path(
                repository_root=plan.repository_root,
                path=path,
                access="read",
            )
        except Exception as error:
            raise EvaluationOnlyContractError(
                "test or non-frozen data access is forbidden"
            ) from error
        resolved.add(validated)
    if resolved != expected:
        raise EvaluationOnlyContractError(
            "evaluation may read exactly frozen train and validation"
        )


def validate_execution_authorization(
    *,
    plan: EvaluationExecutionPlan,
    authorization: Mapping[str, Any],
    source_commit: str,
    relevant_source_sha256: str,
) -> None:
    frozen = plan.candidate_plan.frozen_contract
    expected_scalars = {
        "schema_version": (
            "benchmark-v2.7-evaluation-execution-authorization-v1"
        ),
        "status": "AUTHORIZED",
        "source_commit": source_commit,
        "relevant_source_sha256": relevant_source_sha256,
        "runner_config_sha256": plan.runner_config_sha256,
        "candidate_config_sha256": (
            plan.candidate_plan.config_sha256
        ),
        "development_manifest_sha256": frozen[
            "development_manifest_sha256"
        ],
        "train_file_sha256": frozen["train_file_sha256"],
        "train_content_sha256": frozen["train_content_sha256"],
        "validation_file_sha256": frozen[
            "validation_file_sha256"
        ],
        "validation_content_sha256": frozen[
            "validation_content_sha256"
        ],
        "sampling_plan_sha256": FROZEN_SAMPLING_PLAN_SHA256,
        "runtime_root": (
            "artifacts/benchmark_v2_7/candidate_selection"
        ),
    }
    expected_counts = {
        "controls": 3,
        "evaluation_only_candidates": 6,
        "training_trajectories": 0,
    }
    expected_scope = {
        "control_reference": True,
        "evaluation_only_sampling": True,
        "training": False,
        "optimizer_updates": False,
        "checkpoint_retraining": False,
        "validation_selection": False,
        "test_split_access": False,
        "fresh_test": False,
        "tstr": False,
        "privacy": False,
        "five_seed_full_run": False,
    }
    expected_checkpoints = {
        operation.model_id: operation.checkpoint_sha256
        for operation in plan.operations
    }
    mismatches = {
        key: {
            "expected": value,
            "actual": authorization.get(key),
        }
        for key, value in expected_scalars.items()
        if authorization.get(key) != value
    }
    if (
        mismatches
        or authorization.get("models")
        != list(plan.candidate_plan.model_ids)
        or authorization.get("candidate_ids")
        != [
            operation.candidate_id for operation in plan.operations
        ]
        or authorization.get("checkpoint_hashes")
        != expected_checkpoints
        or authorization.get("counts") != expected_counts
        or authorization.get("authorization") != expected_scope
    ):
        raise EvaluationOnlyContractError(
            f"evaluation authorization mismatch: {mismatches}"
        )


def evaluation_relevant_source_sha256(
    repository_root: Path,
) -> str:
    repository_root = repository_root.resolve()
    digest = hashlib.sha256()
    for relative in RELEVANT_SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise EvaluationOnlyContractError(
                f"v2.7 evaluation source is missing: {relative}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _current_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()


def evaluation_plan_report(
    plan: EvaluationExecutionPlan,
) -> Mapping[str, Any]:
    return {
        "schema_version": "benchmark-v2.7-evaluation-plan-v1",
        "status": "PASS",
        "mode": "PLAN_ONLY",
        "source_commit": _current_head(plan.repository_root),
        "relevant_source_sha256": evaluation_relevant_source_sha256(
            plan.repository_root
        ),
        "runner_config_path": str(
            plan.runner_config_path.relative_to(plan.repository_root)
        ),
        "runner_config_sha256": plan.runner_config_sha256,
        "candidate_config_sha256": (
            plan.candidate_plan.config_sha256
        ),
        "runtime_root": str(
            plan.runtime_root.relative_to(plan.repository_root)
        ),
        "models": list(MODEL_IDS),
        "counts": {
            "models": len(MODEL_IDS),
            "candidates": len(plan.operations),
            "frozen_controls": sum(
                operation.is_control for operation in plan.operations
            ),
            "evaluation_only_candidates": sum(
                not operation.is_control for operation in plan.operations
            ),
            "training_trajectories": 0,
        },
        "operations": [
            {
                "model_id": operation.model_id,
                "candidate_id": operation.candidate_id,
                "factor": operation.factor,
                "execution_kind": operation.execution_kind,
                "checkpoint_path": operation.checkpoint_path,
                "checkpoint_sha256": operation.checkpoint_sha256,
                "restore_mode": operation.restore_mode,
                "training_required": operation.training_required,
                "sampling_required": operation.sampling_required,
                "fit_state_required": operation.fit_state_required,
            }
            for operation in plan.operations
        ],
        "sampling_plan_sha256": FROZEN_SAMPLING_PLAN_SHA256,
        "thresholds": dict(FROZEN_THRESHOLDS),
        "authorization_created": False,
        "execution_authorized": False,
        "runtime_artifact_created": False,
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),
    }


def dry_run_evaluation(
    plan: EvaluationExecutionPlan,
) -> Mapping[str, Any]:
    from experiments.candidate_preparation_runner_v2_7 import (
        build_preparation_execution_plan,
        dry_run_preparation,
    )

    if (
        sha256_file(plan.runner_config_path)
        != plan.runner_config_sha256
    ):
        raise EvaluationOnlyContractError(
            "runner config changed after plan construction"
        )
    preparation = build_preparation_execution_plan(
        plan.candidate_plan.config_path
    )
    frozen_report = dry_run_preparation(preparation)
    controls = []
    for operation in plan.operations:
        validate_operation_single_factor(plan, operation)
        validate_checkpoint_provenance(plan, operation)
        if operation.is_control:
            controls.append(
                build_control_reference(
                    plan=plan,
                    operation=operation,
                    source_commit=_current_head(plan.repository_root),
                    relevant_source_sha256=(
                        evaluation_relevant_source_sha256(
                            plan.repository_root
                        )
                    ),
                )
            )
    report = evaluation_plan_report(plan)
    return {
        **report,
        "schema_version": "benchmark-v2.7-evaluation-dry-run-v1",
        "mode": "DRY_RUN",
        "frozen_provenance_verified": True,
        "frozen_inventory": frozen_report["inventory"],
        "checkpoint_hashes_verified": {
            operation.model_id: operation.checkpoint_sha256
            for operation in plan.operations
        },
        "control_reference_hashes_verified": {
            value["model_id"]: {
                "candidate_result_sha256": value["frozen_control"][
                    "candidate_result_sha256"
                ],
                "validation_sample_sha256": value["frozen_control"][
                    "validation_sample_sha256"
                ],
            }
            for value in controls
        },
        "test_split_read": False,
        "fit_state_created": False,
        "authorization_created": False,
        "runtime_artifact_created": False,
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),
    }


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationOnlyContractError(
            f"cannot read execution authorization: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise EvaluationOnlyContractError(
            "execution authorization root must be a mapping"
        )
    return value


def _save_synthetic_npz(path: Path, sample: Any) -> None:
    if path.exists():
        raise EvaluationOnlyContractError(
            f"append-only sample path exists: {path}"
        )
    np.savez_compressed(
        path,
        **{
            field: getattr(sample, field)
            for field in sample.__dataclass_fields__
        },
    )


def _sample_restored_bundle(
    bundle: Mapping[str, Any],
    *,
    sampling_plan: Any,
    seed: int,
) -> Any:
    backend = bundle.get("backend")
    if backend is None or not callable(getattr(backend, "sample", None)):
        raise EvaluationOnlyContractError(
            "restored checkpoint backend cannot sample"
        )
    sample = backend.sample(sampling_plan, seed=seed)
    transform = bundle.get("train_only_zscore")
    if transform is not None:
        sample = transform.inverse_synthetic(sample)
    return sample


def _set_tvae_temperature(backend: Any, value: float) -> None:
    models = getattr(backend, "models", None)
    if not isinstance(models, Mapping) or set(models) != {0, 1}:
        raise EvaluationOnlyContractError(
            "TVAE restore lacks separate class models"
        )
    for synthesizer in models.values():
        synthesizer.set_candidate_contract(
            channel_weights=dict(
                synthesizer.v2_6_channel_weights
            ),
            latent_scale=float(synthesizer.v2_6_latent_scale),
            categorical_temperature=float(value),
        )


def _apply_cof_frozen_variance_residual(
    *,
    sample: Any,
    train: Any,
    device: str,
    seed: int,
) -> tuple[Any, Mapping[str, Any]]:
    import torch

    from benchmarks.types import SyntheticBatch
    from models.single_factor_components_v2_6 import (
        variance_preserving_residual,
    )

    amount = torch.from_numpy(sample.x_num).to(device)
    mask = torch.from_numpy(sample.valid_mask).to(device)
    adjusted, state = variance_preserving_residual(
        amount,
        valid_mask=mask,
        train_variance=float(
            train.x_num[train.valid_mask, 0].var()
        ),
        seed=seed,
    )
    numerical = adjusted.detach().cpu().numpy().astype(np.float32)
    numerical[~sample.valid_mask] = 0
    return (
        SyntheticBatch(
            x_num=numerical,
            dt_bin=sample.dt_bin.copy(),
            x_cat=sample.x_cat.copy(),
            valid_mask=sample.valid_mask.copy(),
            y_entity=sample.y_entity.copy(),
            lengths=sample.lengths.copy(),
        ),
        dict(state),
    )


def _execute_evaluation_child(
    payload: Mapping[str, Any],
    *,
    completion_notifier: Callable[[], None] | None = None,
) -> None:
    import torch

    from eval.evaluation_only_v2_7 import (
        evaluate_validation_guards,
    )
    from eval.validation_selection_v2_6 import load_development_context
    from experiments.candidate_runner_v2_6 import (
        load_train_only_candidate_context,
    )
    from generators.evaluation_only_transforms_v2_7 import (
        apply_post_sample_intervention,
        configure_backend_for_intervention,
        fit_train_only_intervention,
    )

    root = Path(str(payload["repository_root"])).resolve()
    plan = build_evaluation_execution_plan(
        Path(str(payload["runner_config_path"])).resolve()
    )
    operation = next(
        (
            item
            for item in plan.operations
            if item.model_id == payload["model_id"]
            and item.candidate_id == payload["candidate_id"]
        ),
        None,
    )
    if operation is None or operation.is_control:
        raise EvaluationOnlyContractError(
            "evaluation child received a control or unknown operation"
        )
    source_commit = str(payload["source_commit"])
    relevant_hash = str(payload["relevant_source_sha256"])
    if (
        _current_head(root) != source_commit
        or evaluation_relevant_source_sha256(root)
        != relevant_hash
    ):
        raise EvaluationOnlyContractError(
            "source changed after execution authorization"
        )
    authorization_path = Path(
        str(payload["authorization_path"])
    ).resolve()
    authorization = _read_json_mapping(authorization_path)
    validate_execution_authorization(
        plan=plan,
        authorization=authorization,
        source_commit=source_commit,
        relevant_source_sha256=relevant_hash,
    )
    device = str(payload["device"])
    if not device.startswith("cuda:"):
        raise EvaluationOnlyContractError(
            "authorized evaluation child needs an explicit CUDA device"
        )
    if not torch.cuda.is_available():
        raise EvaluationOnlyContractError(
            "authorized CUDA device is unavailable"
        )
    torch.cuda.set_device(torch.device(device))
    train_context = load_train_only_candidate_context(
        repository_root=root,
        development_manifest_path=(
            root / "configs/benchmark_v2/development_data_v2_6.yaml"
        ),
    )
    frozen = plan.candidate_plan.frozen_contract
    if (
        train_context.manifest_sha256
        != frozen["development_manifest_sha256"]
        or train_context.train_file_sha256
        != frozen["train_file_sha256"]
        or train_context.train_content_sha256
        != frozen["train_content_sha256"]
        or train_context.validation_file_sha256
        != frozen["validation_file_sha256"]
        or train_context.validation_content_sha256
        != frozen["validation_content_sha256"]
        or train_context.plan.plan_hash
        != FROZEN_SAMPLING_PLAN_SHA256
    ):
        raise EvaluationOnlyContractError(
            "development data provenance changed"
        )
    candidate = _candidate_definition(plan, operation)
    restored = restore_checkpoint_for_evaluation(
        plan=plan,
        operation=operation,
        device=device,
        train=train_context.train,
    )
    bundle = restored["checkpoint_bundle"]
    backend = bundle["backend"]
    if operation.model_id == "tvae_separate_class":
        _set_tvae_temperature(backend, 0.75)
    if operation.factor == "categorical_temperature":
        calibration_samples = {}
        for value in candidate.fit_parameters["fixed_grid"]:
            _set_tvae_temperature(backend, float(value))
            calibration_samples[f"{float(value):g}"] = (
                _sample_restored_bundle(
                    bundle,
                    sampling_plan=train_context.plan,
                    seed=int(
                        plan.execution_contract["fit_sample_seed"]
                    ),
                )
            )
    else:
        calibration_samples = {
            "base": _sample_restored_bundle(
                bundle,
                sampling_plan=train_context.plan,
                seed=int(plan.execution_contract["fit_sample_seed"]),
            )
        }
    fit_state = fit_train_only_intervention(
        candidate=candidate,
        train=train_context.train,
        calibration_samples=calibration_samples,
        source_commit=source_commit,
        config_sha256=plan.candidate_plan.config_sha256,
        train_file_sha256=train_context.train_file_sha256,
        train_content_sha256=train_context.train_content_sha256,
        sampling_plan_sha256=train_context.plan.plan_hash,
    )
    intervention = configure_backend_for_intervention(
        backend=backend,
        candidate=candidate,
        fit_state=fit_state,
    )
    sample = _sample_restored_bundle(
        bundle,
        sampling_plan=train_context.plan,
        seed=int(plan.execution_contract["validation_sample_seed"]),
    )
    sample = apply_post_sample_intervention(
        candidate=candidate,
        sample=sample,
        fit_state=fit_state,
    )
    baseline_postprocess: Mapping[str, Any] | None = None
    if (
        operation.model_id == "cof_seqgen"
        and operation.factor == "gap_logit_calibration"
    ):
        sample, baseline_postprocess = (
            _apply_cof_frozen_variance_residual(
                sample=sample,
                train=train_context.train,
                device=device,
                seed=(
                    int(
                        plan.execution_contract[
                            "validation_sample_seed"
                        ]
                    )
                    + 60_001
                ),
            )
        )
    # Validation is deliberately opened only after the synthetic output is
    # complete. It never participates in intervention fitting or sampling.
    context = load_development_context(
        repository_root=root,
        manifest_path=(
            root / "configs/benchmark_v2/development_data_v2_6.yaml"
        ),
    )
    if (
        context.manifest_sha256 != train_context.manifest_sha256
        or context.train_file_sha256
        != train_context.train_file_sha256
        or context.train_content_sha256
        != train_context.train_content_sha256
        or context.validation_file_sha256
        != train_context.validation_file_sha256
        or context.validation_content_sha256
        != train_context.validation_content_sha256
        or context.plan.plan_hash != train_context.plan.plan_hash
    ):
        raise EvaluationOnlyContractError(
            "development data changed between generation and evaluation"
        )
    guards = evaluate_validation_guards(
        train=context.train,
        validation=context.validation,
        sample=sample,
        sampling_plan=context.plan,
        tau=context.tau,
        receiver_categories=context.receiver_categories,
        thresholds=FROZEN_THRESHOLDS,
    )
    attempt = Path(str(payload["attempt_path"])).resolve()
    expected_root = plan.runtime_root.resolve()
    try:
        attempt.relative_to(expected_root)
    except ValueError as error:
        raise EvaluationOnlyContractError(
            "evaluation child artifact path escapes v2.7 runtime"
        ) from error
    store = EvaluationOnlyArtifactStore(root)
    sample_path = attempt / "validation_sample.npz"
    _save_synthetic_npz(sample_path, sample)
    store.write_json(attempt / "fit_state.json", fit_state)
    store.write_json(
        attempt / "evaluation.json",
        {
            **guards,
            "candidate_id": operation.candidate_id,
            "model_id": operation.model_id,
        },
    )
    result = {
        "schema_version": (
            "benchmark-v2.7-evaluation-only-candidate-result-v1"
        ),
        "status": "COMPLETE",
        "model_id": operation.model_id,
        "candidate_id": operation.candidate_id,
        "factor": operation.factor,
        "source_commit": source_commit,
        "relevant_source_sha256": relevant_hash,
        "runner_config_sha256": plan.runner_config_sha256,
        "candidate_config_sha256": (
            plan.candidate_plan.config_sha256
        ),
        "authorization_sha256": sha256_file(
            authorization_path
        ),
        "development_manifest_sha256": context.manifest_sha256,
        "train_file_sha256": context.train_file_sha256,
        "train_content_sha256": context.train_content_sha256,
        "validation_file_sha256": context.validation_file_sha256,
        "validation_content_sha256": (
            context.validation_content_sha256
        ),
        "sampling_plan_sha256": context.plan.plan_hash,
        "checkpoint_path": operation.checkpoint_path,
        "checkpoint_sha256": operation.checkpoint_sha256,
        "restore_mode": operation.restore_mode,
        "fit_state_sha256": fit_state["state_sha256"],
        "validation_sample_path": str(
            sample_path.relative_to(root)
        ),
        "validation_sample_sha256": sha256_file(sample_path),
        "guard_status": guards["status"],
        "all_five_guards_pass": guards[
            "all_five_guards_pass"
        ],
        "intervention": dict(intervention),
        "frozen_baseline_postprocess": (
            dict(baseline_postprocess)
            if baseline_postprocess is not None
            else None
        ),
        "training_calls": 0,
        "optimizer_updates": 0,
        "checkpoint_training_calls": 0,
        "sampling_calls": (
            6
            if operation.factor == "categorical_temperature"
            else 2
        ),
        "validation_selection_executed": False,
        "test_split_read": False,
    }
    store.write_json(attempt / "candidate_result.json", result)
    store.write_json(
        attempt / "COMPLETE.json",
        {
            "schema_version": (
                "benchmark-v2.7-evaluation-attempt-terminal-v1"
            ),
            "status": "COMPLETE",
            "candidate_result_sha256": sha256_file(
                attempt / "candidate_result.json"
            ),
            "evaluation_sha256": sha256_file(
                attempt / "evaluation.json"
            ),
            "fit_state_sha256": sha256_file(
                attempt / "fit_state.json"
            ),
            "validation_sample_sha256": sha256_file(sample_path),
        },
    )
    if completion_notifier is not None:
        completion_notifier()


def _evaluation_child_entry(
    payload: Mapping[str, Any],
    result_queue: Any,
) -> None:
    completion_sent = False

    def notify_complete() -> None:
        nonlocal completion_sent
        result_queue.put({"status": "COMPLETE"})
        completion_sent = True

    try:
        _execute_evaluation_child(
            payload,
            completion_notifier=notify_complete,
        )
    except BaseException as error:
        result_queue.put(
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        return
    if not completion_sent:
        notify_complete()


def _stop_runner_owned_child(
    child: Any,
    *,
    termination_grace_seconds: float,
) -> bool:
    cleanup_forced = child.is_alive()
    if cleanup_forced:
        child.terminate()
        child.join(timeout=termination_grace_seconds)
    if child.is_alive():
        child.kill()
        child.join(timeout=termination_grace_seconds)
    return cleanup_forced


def run_evaluation_child_until_terminal(
    *,
    child_target: Callable[[Mapping[str, Any], Any], None],
    payload: Mapping[str, Any],
    attempt_path: Path,
    max_wall_seconds: float,
    termination_grace_seconds: float,
    on_target_complete: (
        Callable[[Mapping[str, Any]], None] | None
    ) = None,
    target_complete_cleanup_grace_seconds: float = 2.0,
) -> Mapping[str, Any]:
    if (
        max_wall_seconds <= 0
        or termination_grace_seconds <= 0
        or target_complete_cleanup_grace_seconds <= 0
    ):
        raise ValueError("evaluation child timeouts must be positive")
    attempt_path = attempt_path.resolve()
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    child = context.Process(
        target=child_target,
        args=(dict(payload), result_queue),
    )
    started = time.monotonic()
    child.start()
    deadline = started + max_wall_seconds
    terminal = None
    while terminal is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            terminal = result_queue.get(
                timeout=min(0.1, remaining)
            )
        except queue.Empty:
            if not child.is_alive():
                try:
                    terminal = result_queue.get_nowait()
                except queue.Empty:
                    break
    target_elapsed = time.monotonic() - started
    if terminal is None:
        was_alive = child.is_alive()
        _stop_runner_owned_child(
            child,
            termination_grace_seconds=termination_grace_seconds,
        )
        return {
            "status": "FAILED",
            "failure_class": (
                "wall_cap" if was_alive else "child_exit"
            ),
            "actual_wall_seconds": time.monotonic() - started,
            "child_pid": child.pid,
            "child_exitcode": child.exitcode,
            "error": (
                "wall cap reached"
                if was_alive
                else "child exited without terminal metadata"
            ),
            "cleanup_forced": was_alive,
        }
    if terminal.get("status") != "COMPLETE":
        cleanup_forced = _stop_runner_owned_child(
            child,
            termination_grace_seconds=termination_grace_seconds,
        )
        return {
            **dict(terminal),
            "status": "FAILED",
            "failure_class": terminal.get(
                "failure_class",
                "code",
            ),
            "actual_wall_seconds": time.monotonic() - started,
            "child_pid": child.pid,
            "child_exitcode": child.exitcode,
            "cleanup_forced": cleanup_forced,
        }
    if not (attempt_path / "COMPLETE.json").is_file():
        cleanup_forced = _stop_runner_owned_child(
            child,
            termination_grace_seconds=termination_grace_seconds,
        )
        return {
            "status": "FAILED",
            "failure_class": "terminal_contract",
            "actual_wall_seconds": time.monotonic() - started,
            "child_pid": child.pid,
            "child_exitcode": child.exitcode,
            "error": (
                "child reported COMPLETE without candidate COMPLETE.json"
            ),
            "cleanup_forced": cleanup_forced,
        }
    provisional = {
        "status": "COMPLETE",
        "failure_class": None,
        "actual_wall_seconds": target_elapsed,
        "child_pid": child.pid,
        "target_completed": True,
    }
    if on_target_complete is not None:
        on_target_complete(provisional)
    child.join(timeout=target_complete_cleanup_grace_seconds)
    cleanup_forced = _stop_runner_owned_child(
        child,
        termination_grace_seconds=termination_grace_seconds,
    )
    return {
        **provisional,
        "cleanup_forced": cleanup_forced,
        "cleanup_exitcode": child.exitcode,
        "cleanup_elapsed_seconds": (
            time.monotonic() - started - target_elapsed
        ),
    }


def finalize_evaluation_worker(
    *,
    store: EvaluationOnlyArtifactStore,
    model_id: str,
    provenance_sha256: str,
    terminals: list[Mapping[str, Any]],
    expected_candidate_ids: tuple[str, ...],
    finalization_kind: str,
) -> Path:
    if (
        model_id not in MODEL_IDS
        or len(provenance_sha256) != 64
        or finalization_kind not in {"normal", "recovery"}
        or not expected_candidate_ids
    ):
        raise EvaluationOnlyContractError(
            "invalid evaluation worker finalization request"
        )
    terminal_ids = [str(row.get("candidate_id")) for row in terminals]
    if (
        len(terminal_ids) != len(set(terminal_ids))
        or not set(terminal_ids).issubset(expected_candidate_ids)
        or any(
            row.get("status") not in {"COMPLETE", "FAILED"}
            for row in terminals
        )
    ):
        raise EvaluationOnlyContractError(
            "evaluation worker terminal rows are invalid"
        )
    status = (
        "COMPLETE"
        if tuple(terminal_ids) == expected_candidate_ids
        and all(row["status"] == "COMPLETE" for row in terminals)
        else "FAILED"
    )
    worker_root = (
        store.root / "workers" / model_id
    )
    if not (worker_root / "ownership.lock").is_file():
        raise EvaluationOnlyContractError(
            "evaluation worker ownership is missing"
        )
    marker = worker_root / (
        "WORKER_COMPLETE.json"
        if status == "COMPLETE"
        else "WORKER_FAILED.json"
    )
    store.write_json(
        marker,
        {
            "schema_version": "benchmark-v2.7-evaluation-worker-v1",
            "status": status,
            "model_id": model_id,
            "provenance_sha256": provenance_sha256,
            "terminals": [dict(row) for row in terminals],
            "finalization_kind": finalization_kind,
            "validation_selection_executed": False,
            "training_calls": 0,
            "optimizer_updates": 0,
            "test_split_read": False,
        },
    )
    return marker


def _tree_record(
    *,
    relative_root: Path,
    roots: tuple[Path, ...],
) -> Mapping[str, Any]:
    relative_root = relative_root.resolve()
    files: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root.is_file():
            files.add(root)
        elif root.is_dir():
            files.update(path for path in root.rglob("*") if path.is_file())
        else:
            raise EvaluationOnlyContractError(
                f"recovery preservation path is missing: {root}"
            )
    ordered = sorted(
        files,
        key=lambda path: path.relative_to(relative_root).as_posix(),
    )
    digest = hashlib.sha256()
    byte_count = 0
    for path in ordered:
        relative = path.relative_to(relative_root).as_posix()
        byte_count += path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return {
        "sha256": digest.hexdigest(),
        "files": len(ordered),
        "bytes": byte_count,
    }


def build_finalization_recovery_inventory(
    *,
    runtime_root: Path,
    model_id: str,
    expected_candidate_ids: tuple[str, ...],
    allow_existing_worker_complete: bool = False,
) -> Mapping[str, Any]:
    runtime_root = runtime_root.resolve()
    if model_id not in MODEL_IDS or not expected_candidate_ids:
        raise EvaluationOnlyContractError(
            "invalid finalization recovery inventory request"
        )
    worker_root = runtime_root / "workers" / model_id
    worker_complete_exists = (
        worker_root / "WORKER_COMPLETE.json"
    ).exists()
    worker_failed_exists = (
        worker_root / "WORKER_FAILED.json"
    ).exists()
    if worker_failed_exists or (
        worker_complete_exists
        and not allow_existing_worker_complete
    ):
        raise EvaluationOnlyContractError(
            "evaluation worker already has a terminal marker"
        )
    ownership = worker_root / "ownership.lock"
    if not ownership.is_file():
        raise EvaluationOnlyContractError(
            "finalization recovery ownership is missing"
        )
    attempts = []
    source_commits = set()
    authorization_hashes = set()
    authorization_paths = set()
    preservation_roots = [ownership]
    for candidate_id in expected_candidate_ids:
        seed_root = (
            runtime_root
            / "evaluations"
            / model_id
            / candidate_id
            / "seed_2601"
        )
        candidate_attempts = sorted(
            path
            for path in seed_root.glob("attempt_*")
            if path.is_dir()
        )
        if len(candidate_attempts) != 1:
            raise EvaluationOnlyContractError(
                "finalization recovery requires exactly one preserved "
                f"attempt: {candidate_id}"
            )
        attempt = candidate_attempts[0]
        manifest_path = attempt / "manifest.json"
        complete_path = attempt / "COMPLETE.json"
        if (
            not manifest_path.is_file()
            or not complete_path.is_file()
            or (attempt / "FAILED.json").exists()
        ):
            raise EvaluationOnlyContractError(
                "finalization recovery requires COMPLETE-only candidates"
            )
        manifest = _read_json_mapping(manifest_path)
        complete = _read_json_mapping(complete_path)
        if (
            manifest.get("model_id") != model_id
            or manifest.get("candidate_id") != candidate_id
            or complete.get("status") != "COMPLETE"
        ):
            raise EvaluationOnlyContractError(
                "finalization recovery candidate identity mismatch"
            )
        source_commits.add(str(manifest.get("source_commit")))
        authorization_hashes.add(
            str(manifest.get("authorization_sha256"))
        )
        authorization_paths.add(
            str(manifest.get("authorization_path"))
        )
        attempt_record = _tree_record(
            relative_root=runtime_root,
            roots=(attempt,),
        )
        attempts.append(
            {
                "candidate_id": candidate_id,
                "attempt_path": attempt.relative_to(
                    runtime_root
                ).as_posix(),
                "attempt_tree_sha256": attempt_record["sha256"],
                "attempt_files": attempt_record["files"],
                "attempt_bytes": attempt_record["bytes"],
                "manifest_sha256": sha256_file(manifest_path),
                "complete_sha256": sha256_file(complete_path),
            }
        )
        preservation_roots.append(attempt)
    if (
        len(source_commits) != 1
        or len(authorization_hashes) != 1
        or len(authorization_paths) != 1
        or any(len(value) != 40 for value in source_commits)
        or any(len(value) != 64 for value in authorization_hashes)
        or any(
            "authorization" not in value
            for value in authorization_paths
        )
    ):
        raise EvaluationOnlyContractError(
            "finalization recovery provenance is inconsistent"
        )
    preservation = _tree_record(
        relative_root=runtime_root,
        roots=tuple(preservation_roots),
    )
    value = {
        "schema_version": (
            "benchmark-v2.7-finalization-recovery-inventory-v1"
        ),
        "model_id": model_id,
        "candidate_ids": list(expected_candidate_ids),
        "candidate_source_commit": next(iter(source_commits)),
        "original_authorization_sha256": next(
            iter(authorization_hashes)
        ),
        "original_authorization_path": next(
            iter(authorization_paths)
        ),
        "ownership_lock_path": ownership.relative_to(
            runtime_root
        ).as_posix(),
        "ownership_lock_sha256": sha256_file(ownership),
        "candidate_attempts": attempts,
        "preservation_tree_sha256": preservation["sha256"],
        "preservation_files": preservation["files"],
        "preservation_bytes": preservation["bytes"],
        "recovery_required": not worker_complete_exists,
        "candidate_execution_required": False,
    }
    return {
        **value,
        "inventory_sha256": canonical_sha256(value),
    }


def validate_finalization_only_authorization(
    *,
    plan: EvaluationExecutionPlan,
    authorization: Mapping[str, Any],
    inventory: Mapping[str, Any],
    source_commit: str,
    relevant_source_sha256: str,
    original_authorization_sha256: str,
) -> None:
    model_id = str(inventory.get("model_id"))
    expected_candidate_ids = [
        operation.candidate_id
        for operation in plan.operations
        if operation.model_id == model_id
    ]
    expected_attempts = {
        row["candidate_id"]: {
            "attempt_path": row["attempt_path"],
            "attempt_tree_sha256": row["attempt_tree_sha256"],
            "manifest_sha256": row["manifest_sha256"],
            "complete_sha256": row["complete_sha256"],
        }
        for row in inventory.get("candidate_attempts", [])
    }
    expected_actions = {
        "write_worker_terminal_marker": True,
        "candidate_execution": False,
        "candidate_replay": False,
        "sampling": False,
        "evaluation": False,
        "training": False,
        "optimizer_updates": False,
        "selection": False,
        "fresh_test": False,
        "tstr": False,
        "privacy": False,
        "five_seed_full_run": False,
    }
    expected = {
        "schema_version": (
            "benchmark-v2.7-evaluation-finalization-only-authorization-v1"
        ),
        "status": "AUTHORIZED",
        "finalizer_source_commit": source_commit,
        "finalizer_relevant_source_sha256": relevant_source_sha256,
        "runner_config_sha256": plan.runner_config_sha256,
        "candidate_config_sha256": (
            plan.candidate_plan.config_sha256
        ),
        "model_id": model_id,
        "candidate_source_commit": inventory.get(
            "candidate_source_commit"
        ),
        "original_execution_authorization_sha256": (
            original_authorization_sha256
        ),
        "ownership_lock_sha256": inventory.get(
            "ownership_lock_sha256"
        ),
        "preservation_tree_sha256": inventory.get(
            "preservation_tree_sha256"
        ),
        "inventory_sha256": inventory.get("inventory_sha256"),
        "candidate_ids": expected_candidate_ids,
        "candidate_attempts": expected_attempts,
        "actions": expected_actions,
    }
    if any(
        authorization.get(key) != value
        for key, value in expected.items()
    ) or inventory.get(
        "original_authorization_sha256"
    ) != original_authorization_sha256:
        raise EvaluationOnlyContractError(
            "finalization-only authorization mismatch"
        )


def recover_evaluation_worker_finalization(
    *,
    plan: EvaluationExecutionPlan,
    store: EvaluationOnlyArtifactStore,
    authorization_path: Path,
    model_id: str,
    source_commit: str,
    relevant_source_sha256: str,
) -> Mapping[str, Any]:
    expected_candidate_ids = tuple(
        operation.candidate_id
        for operation in plan.operations
        if operation.model_id == model_id
    )
    inventory = build_finalization_recovery_inventory(
        runtime_root=store.root,
        model_id=model_id,
        expected_candidate_ids=expected_candidate_ids,
    )
    authorization_path = authorization_path.resolve()
    try:
        authorization_path.relative_to(store.repository_root)
    except ValueError as error:
        raise EvaluationOnlyContractError(
            "finalization-only authorization escapes the repository"
        ) from error
    if (
        not authorization_path.is_file()
        or "authorization" not in authorization_path.as_posix()
    ):
        raise EvaluationOnlyContractError(
            "finalization-only authorization is missing"
        )
    original_path = (
        store.repository_root
        / str(inventory["original_authorization_path"])
    ).resolve()
    try:
        original_path.relative_to(store.repository_root)
    except ValueError as error:
        raise EvaluationOnlyContractError(
            "original execution authorization escapes the repository"
        ) from error
    original_hash = (
        sha256_file(original_path)
        if original_path.is_file()
        else ""
    )
    if original_hash != inventory[
        "original_authorization_sha256"
    ]:
        raise EvaluationOnlyContractError(
            "original execution authorization hash mismatch"
        )
    authorization = _read_json_mapping(authorization_path)
    validate_finalization_only_authorization(
        plan=plan,
        authorization=authorization,
        inventory=inventory,
        source_commit=source_commit,
        relevant_source_sha256=relevant_source_sha256,
        original_authorization_sha256=original_hash,
    )
    terminals = [
        {
            "candidate_id": row["candidate_id"],
            "status": "COMPLETE",
            "attempt": (
                store.root / row["attempt_path"]
            ).relative_to(store.repository_root).as_posix(),
        }
        for row in inventory["candidate_attempts"]
    ]
    marker = finalize_evaluation_worker(
        store=store,
        model_id=model_id,
        provenance_sha256=canonical_sha256(
            {
                "finalization_authorization_sha256": sha256_file(
                    authorization_path
                ),
                "inventory_sha256": inventory["inventory_sha256"],
                "model_id": model_id,
            }
        ),
        terminals=terminals,
        expected_candidate_ids=expected_candidate_ids,
        finalization_kind="recovery",
    )
    for row in inventory["candidate_attempts"]:
        attempt = store.root / row["attempt_path"]
        after = _tree_record(
            relative_root=store.root,
            roots=(attempt,),
        )
        if after["sha256"] != row["attempt_tree_sha256"]:
            raise EvaluationOnlyContractError(
                "candidate artifact changed during finalization recovery"
            )
    return {
        "status": "COMPLETE",
        "model_id": model_id,
        "worker_terminal_path": marker.relative_to(
            store.repository_root
        ).as_posix(),
        "candidate_execution_calls": 0,
        "sampling_calls": 0,
        "evaluation_calls": 0,
        "training_calls": 0,
        "selection_calls": 0,
        "preservation_tree_sha256": inventory[
            "preservation_tree_sha256"
        ],
    }


def execute_model_worker(
    *,
    plan: EvaluationExecutionPlan,
    authorization_path: Path,
    model_id: str,
    device: str,
) -> Mapping[str, Any]:
    if model_id not in MODEL_IDS:
        raise EvaluationOnlyContractError(
            f"unknown v2.7 evaluation model: {model_id}"
        )
    source_commit = _current_head(plan.repository_root)
    relevant_hash = evaluation_relevant_source_sha256(
        plan.repository_root
    )
    authorization_path = authorization_path.resolve()
    try:
        authorization_relative = authorization_path.relative_to(
            plan.repository_root
        )
    except ValueError as error:
        raise EvaluationOnlyContractError(
            "execution authorization must be inside the repository"
        ) from error
    if (
        not authorization_path.is_file()
        or "authorization" not in authorization_relative.as_posix()
        or not device.startswith("cuda:")
    ):
        raise EvaluationOnlyContractError(
            "authorization path or explicit CUDA device is invalid"
        )
    authorization = _read_json_mapping(authorization_path)
    validate_execution_authorization(
        plan=plan,
        authorization=authorization,
        source_commit=source_commit,
        relevant_source_sha256=relevant_hash,
    )
    if sha256_file(plan.runner_config_path) != (
        plan.runner_config_sha256
    ):
        raise EvaluationOnlyContractError(
            "runner config changed after authorization"
        )
    provenance = canonical_sha256(
        {
            "source_commit": source_commit,
            "relevant_source_sha256": relevant_hash,
            "runner_config_sha256": plan.runner_config_sha256,
            "candidate_config_sha256": (
                plan.candidate_plan.config_sha256
            ),
            "authorization_sha256": sha256_file(
                authorization_path
            ),
            "model_id": model_id,
        }
    )
    store = EvaluationOnlyArtifactStore(plan.repository_root)
    store.claim_model(
        model_id=model_id,
        provenance_sha256=provenance,
    )
    operations = [
        operation
        for operation in plan.operations
        if operation.model_id == model_id
    ]
    expected_candidate_ids = tuple(
        operation.candidate_id for operation in operations
    )
    terminals: list[Mapping[str, Any]] = []
    worker_marker: Path | None = None
    for operation_index, operation in enumerate(operations):
        attempt = store.create_evaluation_attempt(
            model_id=model_id,
            candidate_id=operation.candidate_id,
            seed=int(
                plan.execution_contract["validation_sample_seed"]
            ),
        )
        manifest = {
            "schema_version": (
                "benchmark-v2.7-evaluation-attempt-manifest-v1"
            ),
            "status": "RUNNING",
            "source_commit": source_commit,
            "relevant_source_sha256": relevant_hash,
            "runner_config_sha256": plan.runner_config_sha256,
            "candidate_config_sha256": (
                plan.candidate_plan.config_sha256
            ),
            "authorization_path": str(
                authorization_relative
            ),
            "authorization_sha256": sha256_file(
                authorization_path
            ),
            "model_id": model_id,
            "candidate_id": operation.candidate_id,
            "factor": operation.factor,
            "operation_definition_sha256": (
                operation.definition_sha256
            ),
            "checkpoint_path": operation.checkpoint_path,
            "checkpoint_sha256": operation.checkpoint_sha256,
            "sampling_plan_sha256": FROZEN_SAMPLING_PLAN_SHA256,
            "training_calls": 0,
            "optimizer_updates": 0,
            "checkpoint_training_calls": 0,
            "test_split_read": False,
            "device": device,
        }
        store.write_json(attempt / "manifest.json", manifest)
        if operation.is_control:
            reference = build_control_reference(
                plan=plan,
                operation=operation,
                source_commit=source_commit,
                relevant_source_sha256=relevant_hash,
            )
            store.write_json(
                attempt / "control_reference.json",
                reference,
            )
            store.write_json(
                attempt / "COMPLETE.json",
                {
                    "schema_version": (
                        "benchmark-v2.7-evaluation-attempt-terminal-v1"
                    ),
                    "status": "COMPLETE",
                    "control_reference_sha256": sha256_file(
                        attempt / "control_reference.json"
                    ),
                },
            )
            terminals.append(
                {
                    "candidate_id": operation.candidate_id,
                    "status": "COMPLETE",
                    "attempt": str(attempt.relative_to(plan.repository_root)),
                }
            )
            continue
        payload = {
            "repository_root": str(plan.repository_root),
            "runner_config_path": str(plan.runner_config_path),
            "authorization_path": str(authorization_path),
            "attempt_path": str(attempt),
            "source_commit": source_commit,
            "relevant_source_sha256": relevant_hash,
            "model_id": model_id,
            "candidate_id": operation.candidate_id,
            "device": device,
        }
        terminal_row = {
            "candidate_id": operation.candidate_id,
            "status": "COMPLETE",
            "attempt": str(attempt.relative_to(plan.repository_root)),
        }
        final_operation = operation_index == len(operations) - 1

        def finalize_after_target(
            provisional: Mapping[str, Any],
        ) -> None:
            del provisional
            nonlocal worker_marker
            worker_marker = finalize_evaluation_worker(
                store=store,
                model_id=model_id,
                provenance_sha256=provenance,
                terminals=[*terminals, terminal_row],
                expected_candidate_ids=expected_candidate_ids,
                finalization_kind="normal",
            )

        result = run_evaluation_child_until_terminal(
            child_target=_evaluation_child_entry,
            payload=payload,
            attempt_path=attempt,
            max_wall_seconds=float(
                plan.execution_contract[
                    "max_wall_seconds_per_candidate"
                ]
            ),
            termination_grace_seconds=float(
                plan.execution_contract[
                    "termination_grace_seconds"
                ]
            ),
            on_target_complete=(
                finalize_after_target if final_operation else None
            ),
            target_complete_cleanup_grace_seconds=2.0,
        )
        if result["status"] != "COMPLETE":
            store.write_json(
                attempt / "FAILED.json",
                {
                    "schema_version": (
                        "benchmark-v2.7-evaluation-attempt-terminal-v1"
                    ),
                    "status": "FAILED",
                    "failure_class": result.get(
                        "failure_class",
                        "code",
                    ),
                    "actual_wall_seconds": result[
                        "actual_wall_seconds"
                    ],
                    "details": result,
                },
            )
            terminals.append(
                {
                    "candidate_id": operation.candidate_id,
                    "status": "FAILED",
                    "failure_class": result.get(
                        "failure_class",
                        "code",
                    ),
                    "attempt": str(attempt.relative_to(plan.repository_root)),
                }
            )
            break
        terminals.append(terminal_row)
    if worker_marker is None:
        worker_marker = finalize_evaluation_worker(
            store=store,
            model_id=model_id,
            provenance_sha256=provenance,
            terminals=terminals,
            expected_candidate_ids=expected_candidate_ids,
            finalization_kind="normal",
        )
    status = (
        "COMPLETE"
        if worker_marker.name == "WORKER_COMPLETE.json"
        else "FAILED"
    )
    return {
        "status": status,
        "model_id": model_id,
        "worker_terminal_path": str(
            worker_marker.relative_to(plan.repository_root)
        ),
        "terminals": terminals,
    }
