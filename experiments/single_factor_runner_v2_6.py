from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping

import numpy as np
import yaml

from benchmarks.types import SequenceBatch
from eval.single_factor_amendment_v2_6 import (
    MODEL_IDS,
    load_and_validate_amendment,
    sha256_file,
)
from eval.validation_selection_v2_6 import validate_selection_config
from experiments.provenance_v2_5 import hash_batch
from generators.sampling_plan import SamplingPlan


SINGLE_FACTOR_MODEL_IDS = MODEL_IDS
AUTHORIZATION_SCHEMA = (
    "benchmark-v2.6-single-factor-execution-authorization-v1"
)
RUNTIME_ROOT = "artifacts/benchmark_v2_6/selection_single_factor"
APPROVAL_SCOPE = (
    "validation-only single-factor candidate execution: exactly "
    "three frozen controls, three checkpoint-only evaluations, "
    "and three new trajectories; test/fresh-test/selection/"
    "five-seed/full execution remain forbidden"
)
BASE_SELECTION_CONFIG_SHA256 = (
    "0884a0144f74f6317ae9e636c0cf5657dedac21ff12cdf545a6b6e5e29be4c4d"
)
DEVELOPMENT_MANIFEST_SHA256 = (
    "31ddf45dcf7b4f982ed990ec94281c3e6bad0454ec3182e2434e42358006ade5"
)
TRAIN_FILE_SHA256 = (
    "c67a6fce4593317e11e1b6f36397bfb38f9607916bf8f5b32329d8fa99b700a8"
)
TRAIN_CONTENT_SHA256 = (
    "0c03f179930cd4d89ccc8a8b66283e620313e16d15fd53f0674c133bcb80c09d"
)
VALIDATION_FILE_SHA256 = (
    "68c15c05c55742bfce0f22560ffe8378d032a7a9fd9a1072e80ec8da5340fba5"
)
VALIDATION_CONTENT_SHA256 = (
    "aa1576b4ccae6a2ca30d0472f0782e1231ecc61ab3d224590a772dc3448b9e66"
)
SAMPLING_PLAN_SHA256 = (
    "862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27"
)
V2_5_CONFIG_SHA256 = (
    "81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3"
)
V2_5_FINAL_COMPLETE_SHA256 = (
    "e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a"
)
V2_5_FROZEN_MANIFEST_SHA256 = (
    "b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05"
)
PRIOR_SELECTION_COMPLETE_SHA256 = (
    "3cb9f91a00ea4b3bcb772527e2ff4f49b1a47c645618ce79995847f07337407d"
)

RELEVANT_SOURCE_PATHS = (
    "eval/single_factor_amendment_v2_6.py",
    "eval/validation_selection_v2_6.py",
    "experiments/candidate_runner_v2_6.py",
    "experiments/single_factor_runner_v2_6.py",
    "generators/candidate_adapters_v2_6.py",
    "generators/candidate_model_backends_v2_6.py",
    "generators/single_factor_backends_v2_6.py",
    "models/candidate_components_v2_6.py",
    "models/single_factor_components_v2_6.py",
    "experiments/single_factor_aggregate_v2_6.py",
    "scripts/aggregate_single_factor_selection_v2_6.py",
    "scripts/run_single_factor_candidates_v2_6.py",
)


class SingleFactorRunnerContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class SingleFactorOperation:
    model_id: str
    candidate_id: str
    execution_kind: str
    intervention_dimension: str
    changed_dimensions: tuple[str, ...]
    trajectory_id: str | None
    training_required: bool
    sampling_required: bool
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    frozen_control_paths: Mapping[str, str]
    frozen_control_hashes: Mapping[str, str]
    baseline_dimensions: Mapping[str, Any]
    effective_dimensions: Mapping[str, Any]
    definition_sha256: str


@dataclass(frozen=True)
class SingleFactorExecutionPlan:
    repository_root: Path
    config_path: Path
    config_sha256: str
    base_selection_config_path: Path
    base_selection_config_sha256: str
    development_manifest_path: Path
    model_ids: tuple[str, ...]
    operations: tuple[SingleFactorOperation, ...]
    max_training_gpu_hours: float
    max_evaluation_only_gpu_hours: float
    max_total_gpu_hours: float


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SingleFactorRunnerContractError(
            f"cannot read YAML: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise SingleFactorRunnerContractError(
            f"YAML root is not a mapping: {path}"
        )
    return value


def _read_sequence_batch(path: Path) -> SequenceBatch:
    if path.name not in {"train.npz", "validation.npz"}:
        raise SingleFactorRunnerContractError(
            "single-factor preflight may read only train/validation"
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {
                field: archive[field]
                for field in SequenceBatch.__dataclass_fields__
            }
    except (OSError, ValueError, KeyError) as error:
        raise SingleFactorRunnerContractError(
            f"cannot read frozen development split: {path}"
        ) from error
    return SequenceBatch(**values)


def _current_head(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise SingleFactorRunnerContractError(
            "current source commit cannot be resolved"
        )
    return value


def single_factor_relevant_source_sha256(
    repository_root: Path,
) -> str:
    root = repository_root.resolve()
    digest = hashlib.sha256()
    for relative in RELEVANT_SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise SingleFactorRunnerContractError(
                f"relevant single-factor source is missing: {relative}"
            )
        digest.update(relative.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def preserved_tree_record_sha256(root: Path) -> tuple[str, int, int]:
    root = root.resolve()
    if not root.is_dir():
        raise SingleFactorRunnerContractError(
            f"preserved tree is missing: {root}"
        )
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    byte_count = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                byte_count += len(chunk)
                file_digest.update(chunk)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(file_digest.hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest(), len(files), byte_count


def build_single_factor_execution_plan(
    *,
    config_path: Path,
    base_selection_config_path: Path,
) -> SingleFactorExecutionPlan:
    config_path = config_path.resolve()
    base_selection_config_path = base_selection_config_path.resolve()
    amendment = load_and_validate_amendment(config_path)
    repository_root = amendment.config_path.parents[2]
    expected_base = (
        repository_root / "configs/benchmark_v2/selection_v2_6.yaml"
    ).resolve()
    if base_selection_config_path != expected_base:
        raise SingleFactorRunnerContractError(
            "base selection config path changed"
        )
    base_hash = sha256_file(base_selection_config_path)
    if base_hash != BASE_SELECTION_CONFIG_SHA256:
        raise SingleFactorRunnerContractError(
            "base selection config hash changed"
        )
    try:
        validate_selection_config(_read_yaml(base_selection_config_path))
    except Exception as error:
        raise SingleFactorRunnerContractError(
            "base selection config contract changed"
        ) from error
    config = _read_yaml(config_path)
    models = config["models"]
    definitions = {
        (item.model_id, item.candidate_id): item
        for item in amendment.candidates
    }
    operations = []
    for model_id in amendment.model_ids:
        model = models[model_id]
        frozen = model["frozen_control"]
        frozen_hashes = {
            "candidate_result_sha256": str(
                frozen["candidate_result_sha256"]
            ),
            "checkpoint_sha256": str(frozen["checkpoint_sha256"]),
            "validation_sample_sha256": str(
                frozen["validation_sample_sha256"]
            ),
        }
        frozen_paths = {
            "candidate_result_path": str(
                frozen["candidate_result_path"]
            ),
            "checkpoint_path": str(frozen["checkpoint_path"]),
            "validation_sample_path": str(
                frozen["validation_sample_path"]
            ),
        }
        baseline = dict(model["baseline_dimensions"])
        for candidate in model["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            definition = definitions[(model_id, candidate_id)]
            execution_kind = definition.execution_kind
            checkpoint_path = None
            checkpoint_hash = None
            if execution_kind == "evaluate_existing_checkpoint":
                checkpoint_path = str(frozen["checkpoint_path"])
                checkpoint_hash = str(frozen["checkpoint_sha256"])
            operations.append(
                SingleFactorOperation(
                    model_id=model_id,
                    candidate_id=candidate_id,
                    execution_kind=execution_kind,
                    intervention_dimension=(
                        definition.intervention_dimension
                    ),
                    changed_dimensions=definition.changed_dimensions,
                    trajectory_id=definition.trajectory_id,
                    training_required=(
                        execution_kind == "train_new_trajectory"
                    ),
                    sampling_required=(
                        execution_kind
                        != "reuse_frozen_control"
                    ),
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_hash,
                    frozen_control_paths=frozen_paths,
                    frozen_control_hashes=frozen_hashes,
                    baseline_dimensions=baseline,
                    effective_dimensions=dict(
                        candidate["effective_dimensions"]
                    ),
                    definition_sha256=canonical_sha256(candidate),
                )
            )
    if len(operations) != 9:
        raise SingleFactorRunnerContractError(
            "single-factor operation count changed"
        )
    development = (
        repository_root
        / "configs/benchmark_v2/development_data_v2_6.yaml"
    ).resolve()
    return SingleFactorExecutionPlan(
        repository_root=repository_root,
        config_path=config_path,
        config_sha256=amendment.config_sha256,
        base_selection_config_path=base_selection_config_path,
        base_selection_config_sha256=base_hash,
        development_manifest_path=development,
        model_ids=amendment.model_ids,
        operations=tuple(operations),
        max_training_gpu_hours=amendment.max_training_gpu_hours,
        max_evaluation_only_gpu_hours=(
            amendment.max_evaluation_only_gpu_hours
        ),
        max_total_gpu_hours=amendment.max_total_gpu_hours,
    )


def _expected_control_hashes(
    plan: SingleFactorExecutionPlan,
) -> Mapping[str, Mapping[str, str]]:
    return {
        operation.model_id: dict(operation.frozen_control_hashes)
        for operation in plan.operations
        if operation.execution_kind == "reuse_frozen_control"
    }


def validate_single_factor_authorization(
    authorization: Mapping[str, Any],
    *,
    plan: SingleFactorExecutionPlan,
    source_commit: str,
    relevant_source_sha256: str,
    prior_candidate_tree_sha256: str,
) -> None:
    expected_scalars = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "AUTHORIZED",
        "approval_scope": APPROVAL_SCOPE,
        "source_commit": source_commit,
        "relevant_source_sha256": relevant_source_sha256,
        "single_factor_config_sha256": plan.config_sha256,
        "base_selection_config_sha256": (
            BASE_SELECTION_CONFIG_SHA256
        ),
        "development_manifest_sha256": (
            DEVELOPMENT_MANIFEST_SHA256
        ),
        "train_file_sha256": TRAIN_FILE_SHA256,
        "train_content_sha256": TRAIN_CONTENT_SHA256,
        "validation_file_sha256": VALIDATION_FILE_SHA256,
        "validation_content_sha256": VALIDATION_CONTENT_SHA256,
        "sampling_plan_sha256": SAMPLING_PLAN_SHA256,
        "v2_5_config_sha256": V2_5_CONFIG_SHA256,
        "v2_5_final_complete_sha256": (
            V2_5_FINAL_COMPLETE_SHA256
        ),
        "v2_5_frozen_manifest_sha256": (
            V2_5_FROZEN_MANIFEST_SHA256
        ),
        "prior_v2_6_selection_complete_sha256": (
            PRIOR_SELECTION_COMPLETE_SHA256
        ),
        "prior_v2_6_candidate_tree_sha256": (
            prior_candidate_tree_sha256
        ),
        "runtime_root": RUNTIME_ROOT,
    }
    mismatches = {
        key: {
            "expected": value,
            "actual": authorization.get(key),
        }
        for key, value in expected_scalars.items()
        if authorization.get(key) != value
    }
    if mismatches:
        raise SingleFactorRunnerContractError(
            f"single-factor authorization mismatch: {mismatches}"
        )
    expected_candidates = [
        operation.candidate_id for operation in plan.operations
    ]
    expected_trajectories = [
        operation.trajectory_id
        for operation in plan.operations
        if operation.execution_kind == "train_new_trajectory"
    ]
    expected_counts = {
        "candidates": 9,
        "reused_controls": 3,
        "evaluation_only": 3,
        "new_trajectories": 3,
    }
    expected_flags = {
        "candidate_execution": True,
        "gpu_training_for_three_new_trajectories": True,
        "gpu_sampling_for_three_evaluation_candidates": True,
        "validation_selection": False,
        "test_split_access": False,
        "fresh_test": False,
        "five_seed_full_run": False,
        "data_generation": False,
        "dgp_execution": False,
    }
    if (
        authorization.get("models") != list(plan.model_ids)
        or authorization.get("candidate_ids") != expected_candidates
        or authorization.get("new_trajectory_ids")
        != expected_trajectories
        or authorization.get("counts") != expected_counts
        or authorization.get("authorization") != expected_flags
        or authorization.get("frozen_control_hashes")
        != _expected_control_hashes(plan)
    ):
        raise SingleFactorRunnerContractError(
            "single-factor authorization scope changed"
        )


def _verify_development_provenance(
    plan: SingleFactorExecutionPlan,
) -> Mapping[str, Any]:
    repository_root = plan.repository_root
    manifest_path = plan.development_manifest_path
    if sha256_file(manifest_path) != DEVELOPMENT_MANIFEST_SHA256:
        raise SingleFactorRunnerContractError(
            "development manifest hash mismatch"
        )
    manifest = _read_yaml(manifest_path)
    if (
        manifest.get("test_split_access") != "FORBIDDEN"
        or set(manifest.get("splits", {})) != {
            "train",
            "validation",
        }
    ):
        raise SingleFactorRunnerContractError(
            "development split scope changed"
        )
    records = manifest["splits"]
    paths: dict[str, Path] = {}
    expected_hashes = {
        "train": (TRAIN_FILE_SHA256, TRAIN_CONTENT_SHA256),
        "validation": (
            VALIDATION_FILE_SHA256,
            VALIDATION_CONTENT_SHA256,
        ),
    }
    content_hashes = {}
    for split in ("train", "validation"):
        relative = Path(str(records[split]["path"]))
        path = (repository_root / relative).resolve()
        if (
            path.name != f"{split}.npz"
            or "test" in {part.lower() for part in path.parts}
            or not path.is_file()
        ):
            raise SingleFactorRunnerContractError(
                f"forbidden development split path: {path}"
            )
        file_hash, content_hash = expected_hashes[split]
        if (
            sha256_file(path) != file_hash
            or records[split].get("file_sha256") != file_hash
            or records[split].get("content_sha256")
            != content_hash
        ):
            raise SingleFactorRunnerContractError(
                f"{split} provenance mismatch"
            )
        batch = _read_sequence_batch(path)
        observed_content = hash_batch(batch)
        if observed_content != content_hash:
            raise SingleFactorRunnerContractError(
                f"{split} content hash mismatch"
            )
        paths[split] = path
        content_hashes[split] = observed_content
    plan_record = manifest.get("selection_sampling_plan")
    train_batch = _read_sequence_batch(paths["train"])
    sampling_plan = SamplingPlan.from_train_policy(
        train_batch,
        entity_count=int(plan_record["entity_count"]),
        seed=int(plan_record["seed"]),
    )
    if (
        plan_record.get("fit_split") != "train"
        or sampling_plan.plan_hash != SAMPLING_PLAN_SHA256
        or plan_record.get("content_sha256")
        != SAMPLING_PLAN_SHA256
    ):
        raise SingleFactorRunnerContractError(
            "train-only SamplingPlan provenance mismatch"
        )
    return {
        "development_manifest_sha256": (
            DEVELOPMENT_MANIFEST_SHA256
        ),
        "train_file_sha256": TRAIN_FILE_SHA256,
        "train_content_sha256": content_hashes["train"],
        "validation_file_sha256": VALIDATION_FILE_SHA256,
        "validation_content_sha256": content_hashes["validation"],
        "sampling_plan_sha256": sampling_plan.plan_hash,
        "test_split_accesses": 0,
        "fresh_test_accesses": 0,
    }


def _verify_frozen_external_provenance(
    repository_root: Path,
) -> Mapping[str, str]:
    paths = {
        "v2_5_config_sha256": (
            repository_root / "configs/benchmark_v2/full_v2_5.yaml",
            V2_5_CONFIG_SHA256,
        ),
        "v2_5_final_complete_sha256": (
            repository_root
            / "artifacts/benchmark_v2_5/full/FINAL_COMPLETE.json",
            V2_5_FINAL_COMPLETE_SHA256,
        ),
        "v2_5_frozen_manifest_sha256": (
            repository_root
            / (
                "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
                "kappa_1.00/data_manifest.json"
            ),
            V2_5_FROZEN_MANIFEST_SHA256,
        ),
        "prior_v2_6_selection_complete_sha256": (
            repository_root
            / (
                "artifacts/benchmark_v2_6/selection/"
                "aggregate_attempt_001/AGGREGATE_COMPLETE.json"
            ),
            PRIOR_SELECTION_COMPLETE_SHA256,
        ),
    }
    result = {}
    for key, (path, expected) in paths.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise SingleFactorRunnerContractError(
                f"frozen external provenance mismatch: {key}"
            )
        result[key] = expected
    return result


def dry_run_single_factor_worker(
    *,
    repository_root: Path,
    plan: SingleFactorExecutionPlan,
    authorization: Mapping[str, Any],
    source_commit: str,
    prior_candidate_tree_sha256: str,
    model_id: str,
    counters: MutableMapping[str, int] | None = None,
    verify_files: bool = True,
) -> Mapping[str, Any]:
    repository_root = repository_root.resolve()
    if model_id not in plan.model_ids:
        raise SingleFactorRunnerContractError(
            f"unknown single-factor model worker: {model_id}"
        )
    if repository_root != plan.repository_root:
        raise SingleFactorRunnerContractError(
            "single-factor repository root mismatch"
        )
    source_hash = single_factor_relevant_source_sha256(repository_root)
    validate_single_factor_authorization(
        authorization,
        plan=plan,
        source_commit=source_commit,
        relevant_source_sha256=source_hash,
        prior_candidate_tree_sha256=prior_candidate_tree_sha256,
    )
    provenance: Mapping[str, Any] = {
        "single_factor_config_sha256": plan.config_sha256,
        "base_selection_config_sha256": (
            plan.base_selection_config_sha256
        ),
    }
    if verify_files:
        if _current_head(repository_root) != source_commit:
            raise SingleFactorRunnerContractError(
                "authorization source commit is not current HEAD"
            )
        if sha256_file(plan.config_path) != plan.config_sha256:
            raise SingleFactorRunnerContractError(
                "single-factor config changed after plan"
            )
        provenance = {
            **provenance,
            **_verify_development_provenance(plan),
            **_verify_frozen_external_provenance(repository_root),
        }
        tree_hash, tree_files, tree_bytes = (
            preserved_tree_record_sha256(
                repository_root
                / "artifacts/benchmark_v2_6/selection"
            )
        )
        if tree_hash != prior_candidate_tree_sha256:
            raise SingleFactorRunnerContractError(
                "prior v2.6 candidate tree changed"
            )
        provenance = {
            **provenance,
            "prior_v2_6_candidate_tree_sha256": tree_hash,
            "prior_v2_6_candidate_tree_files": tree_files,
            "prior_v2_6_candidate_tree_bytes": tree_bytes,
        }
    operations = tuple(
        operation
        for operation in plan.operations
        if operation.model_id == model_id
    )
    if len(operations) != 3:
        raise SingleFactorRunnerContractError(
            "model worker operation count changed"
        )
    observed_counters = counters if counters is not None else {
        "test_split_reads": 0,
        "fresh_test_reads": 0,
        "gpu_queries": 0,
        "cuda_calls": 0,
        "dgp_calls": 0,
        "fit_calls": 0,
        "sample_calls": 0,
        "selection_calls": 0,
    }
    if any(value != 0 for value in observed_counters.values()):
        raise SingleFactorRunnerContractError(
            "dry-run observed an execution call"
        )
    return {
        "schema_version": (
            "benchmark-v2.6-single-factor-dry-run-v1"
        ),
        "status": "DRY_RUN_PASS",
        "model_id": model_id,
        "operation_count": len(operations),
        "candidate_ids": [
            operation.candidate_id for operation in operations
        ],
        "new_trajectory_count": sum(
            operation.training_required for operation in operations
        ),
        "evaluation_only_count": sum(
            operation.execution_kind
            == "evaluate_existing_checkpoint"
            for operation in operations
        ),
        "reused_control_count": sum(
            operation.execution_kind == "reuse_frozen_control"
            for operation in operations
        ),
        "new_trajectory_ids": [
            operation.trajectory_id
            for operation in operations
            if operation.training_required
        ],
        "frozen_controls_retrained": False,
        "evaluation_only_retrained": False,
        "test_split_accesses": 0,
        "fresh_test_accesses": 0,
        "thresholds_changed": False,
        "sampling_plan_changed": False,
        "dgp_changed": False,
        "guards_changed": False,
        "validation_selection_executed": False,
        "execution_calls": dict(observed_counters),
        "provenance": dict(provenance),
    }


def plan_report(plan: SingleFactorExecutionPlan) -> Mapping[str, Any]:
    return {
        "schema_version": (
            "benchmark-v2.6-single-factor-execution-plan-v1"
        ),
        "status": "PLAN_PASS",
        "config_sha256": plan.config_sha256,
        "base_selection_config_sha256": (
            plan.base_selection_config_sha256
        ),
        "models": list(plan.model_ids),
        "counts": {
            "candidates": len(plan.operations),
            "reused_controls": sum(
                operation.execution_kind == "reuse_frozen_control"
                for operation in plan.operations
            ),
            "evaluation_only": sum(
                operation.execution_kind
                == "evaluate_existing_checkpoint"
                for operation in plan.operations
            ),
            "new_trajectories": sum(
                operation.execution_kind == "train_new_trajectory"
                for operation in plan.operations
            ),
        },
        "operations": [
            {
                "model_id": operation.model_id,
                "candidate_id": operation.candidate_id,
                "execution_kind": operation.execution_kind,
                "intervention_dimension": (
                    operation.intervention_dimension
                ),
                "changed_dimensions": list(
                    operation.changed_dimensions
                ),
                "trajectory_id": operation.trajectory_id,
                "training_required": operation.training_required,
                "sampling_required": operation.sampling_required,
                "checkpoint_path": operation.checkpoint_path,
                "checkpoint_sha256": operation.checkpoint_sha256,
            }
            for operation in plan.operations
        ],
        "budget": {
            "max_training_gpu_hours": (
                plan.max_training_gpu_hours
            ),
            "max_evaluation_only_gpu_hours": (
                plan.max_evaluation_only_gpu_hours
            ),
            "max_total_gpu_hours": plan.max_total_gpu_hours,
            "three_gpu_max_wall_hours": 2.5,
        },
        "test_split_accesses": 0,
        "fresh_test_accesses": 0,
        "gpu_queries": 0,
        "cuda_calls": 0,
        "fit_calls": 0,
        "sample_calls": 0,
        "dgp_calls": 0,
        "selection_calls": 0,
    }


def read_authorization(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SingleFactorRunnerContractError(
            f"cannot read single-factor authorization: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise SingleFactorRunnerContractError(
            "single-factor authorization root must be a mapping"
        )
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(repository_root: Path, path: Path) -> str:
    return path.resolve().relative_to(
        repository_root.resolve()
    ).as_posix()


def _append_only_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 1_000_000):
        path = root / f"attempt_{number:03d}"
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise SingleFactorRunnerContractError(
        "append-only attempt namespace exhausted"
    )


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise SingleFactorRunnerContractError(
            f"append-only artifact exists: {path}"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_preflight_record(
    *,
    repository_root: Path,
    kind: str,
    value: Mapping[str, Any],
) -> Path:
    if kind not in {"plan", "dry_run"}:
        raise ValueError("preflight kind must be plan or dry_run")
    root = (
        repository_root.resolve()
        / RUNTIME_ROOT
        / "runner_preflight"
    )
    attempt = _append_only_attempt(root / kind)
    path = attempt / f"{kind}.json"
    _exclusive_json(path, value)
    return path


def _operation_by_id(
    plan: SingleFactorExecutionPlan,
    *,
    model_id: str,
    candidate_id: str,
) -> SingleFactorOperation:
    matches = [
        operation
        for operation in plan.operations
        if operation.model_id == model_id
        and operation.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise SingleFactorRunnerContractError(
            f"candidate operation does not resolve once: "
            f"{model_id}/{candidate_id}"
        )
    return matches[0]


def _single_factor_adapter_spec(operation: SingleFactorOperation):
    from generators.candidate_adapters_v2_6 import (
        CandidateAdapterSpec,
    )

    model_id = operation.model_id
    effective = operation.effective_dimensions
    class_updates = (
        {0: 10_000, 1: 10_000}
        if model_id
        in {"ctgan_separate_class", "tvae_separate_class"}
        else None
    )
    class_wall = (
        {0: 3_600.0, 1: 3_600.0}
        if class_updates is not None
        else None
    )
    if model_id == "ctgan_separate_class":
        numeric = {
            "amount": (
                "native_then_train_fitted_bayesian_gmm"
            ),
            "inverse": (
                "upstream_cluster_normalizer_reverse_transform"
            ),
        }
        weights: Mapping[str, float] = {}
        sampling = {
            "latent": "standard_normal",
            "conditional_vector": (
                "upstream_original_frequency"
            ),
            "categorical_activation": "gumbel_softmax",
            "categorical_temperature": float(
                effective["categorical_temperature"]
            ),
            "inverse_decode": "upstream",
        }
    elif model_id == "tvae_separate_class":
        numeric = {
            "amount": (
                "native_then_train_fitted_bayesian_gmm"
            ),
            "inverse": (
                "upstream_cluster_normalizer_reverse_transform_with_sigma"
            ),
        }
        channel_weights = effective["channel_weights"]
        weights = {
            "amount": float(channel_weights["amount"]),
            "gap": float(channel_weights["gap"]),
            "receiver": float(channel_weights["receiver"]),
        }
        sampling = {
            "latent_scale": 1.0,
            "continuous_decode": (
                "tanh_then_sigma_perturbed_inverse"
            ),
            "categorical_decode": str(
                effective["categorical_decode"]
            ),
        }
        if (
            effective["categorical_decode"]
            == "temperature_multinomial_gap_receiver_0.75"
        ):
            sampling["categorical_temperature"] = 0.75
    else:
        numeric = {
            "amount": "native_raw_gaussian_diffusion",
            "inverse": "identity",
        }
        weights = {
            "amount": 1.0,
            "gap": 1.0,
            "receiver": 1.0,
            "label": 1.0,
        }
        sampling = {
            "amount": (
                "epsilon_reverse_diffusion_50_steps"
                if effective[
                    "amount_diffusion_parameterization"
                ]
                == "epsilon_mse_matching_reverse_diffusion"
                else "deterministic_ddim_50_steps"
            ),
            "discrete_feedback_temperature": 1.0,
            "final_categorical_decode": "argmax",
            "guidance_scale": 2.0,
        }
    trajectory_id = operation.trajectory_id or (
        f"{model_id}_frozen_control_seed_2601"
    )
    return CandidateAdapterSpec(
        model_id=model_id,
        candidate_id=operation.candidate_id,
        shared_trajectory_id=trajectory_id,
        requested_updates=20_000,
        sampled_checkpoint_step=20_000,
        checkpoint_interval_steps=100,
        max_wall_seconds=(
            7_200.0
            if operation.training_required
            else 1_800.0
        ),
        numeric_representation=numeric,
        loss_weights=weights,
        sampling_rule=sampling,
        class_updates=class_updates,
        class_wall_seconds=class_wall,
        definition_sha256=operation.definition_sha256,
    )


def _base_adapter_config(
    *,
    plan: SingleFactorExecutionPlan,
    model_id: str,
    device: str,
    tau: np.ndarray,
) -> dict[str, Any]:
    base = _read_yaml(plan.base_selection_config_path)
    config = dict(
        base["models"][model_id]["adapter_base_config"]
    )
    config.pop("adapter", None)
    config["device"] = device
    if model_id == "cof_seqgen":
        config["tau"] = tau.tolist()
    return config


def _backend_factory(operation: SingleFactorOperation):
    if operation.candidate_id == "ctgan_sf_c01_shared_transformer":
        from generators.single_factor_backends_v2_6 import (
            ConditionalCTGANSharedTransformerV26,
        )

        return ConditionalCTGANSharedTransformerV26
    if operation.candidate_id == "cof_sf_c01_noise_prediction":
        from generators.single_factor_backends_v2_6 import (
            CoFNoisePredictionCandidateV26,
        )

        return CoFNoisePredictionCandidateV26
    return None


def _save_synthetic_npz(path: Path, sample) -> None:
    try:
        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                **{
                    field: getattr(sample, field)
                    for field in sample.__dataclass_fields__
                },
            )
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise SingleFactorRunnerContractError(
            f"append-only sample exists: {path}"
        ) from error


def _candidate_manifest(
    *,
    repository_root: Path,
    operation: SingleFactorOperation,
    source_commit: str,
    source_hash: str,
    plan: SingleFactorExecutionPlan,
    authorization_path: Path,
    authorization_hash: str,
    context,
    checkpoint_path: Path | None,
    checkpoint_hash: str | None,
    validation_sample_path: Path,
    actual_updates: int,
    actual_wall_seconds: float,
    trajectory_manifest_path: Path | None,
    trajectory_complete_path: Path | None,
    execution_details: Mapping[str, Any],
) -> Mapping[str, Any]:
    value: dict[str, Any] = {
        "schema_version": (
            "benchmark-v2.6-single-factor-candidate-result-v1"
        ),
        "status": "COMPLETE",
        "source_commit": source_commit,
        "relevant_source_sha256": source_hash,
        "single_factor_config_sha256": plan.config_sha256,
        "base_selection_config_sha256": (
            plan.base_selection_config_sha256
        ),
        "authorization_path": _relative(
            repository_root,
            authorization_path,
        ),
        "authorization_sha256": authorization_hash,
        "model_id": operation.model_id,
        "candidate_id": operation.candidate_id,
        "selection_seed": 2601,
        "execution_kind": operation.execution_kind,
        "intervention_dimension": (
            operation.intervention_dimension
        ),
        "changed_dimensions": list(
            operation.changed_dimensions
        ),
        "candidate_definition_sha256": (
            operation.definition_sha256
        ),
        "development_manifest_sha256": (
            context.manifest_sha256
        ),
        "train_file_sha256": context.train_file_sha256,
        "train_content_sha256": context.train_content_sha256,
        "validation_file_sha256": (
            context.validation_file_sha256
        ),
        "validation_content_sha256": (
            context.validation_content_sha256
        ),
        "selection_plan_sha256": context.plan.plan_hash,
        "requested_updates": (
            20_000 if operation.training_required else 0
        ),
        "actual_updates": int(actual_updates),
        "actual_wall_seconds": float(actual_wall_seconds),
        "checkpoint_path": (
            _relative(repository_root, checkpoint_path)
            if checkpoint_path is not None
            else None
        ),
        "checkpoint_sha256": checkpoint_hash,
        "validation_sample_path": _relative(
            repository_root,
            validation_sample_path,
        ),
        "validation_sample_sha256": sha256_file(
            validation_sample_path
        ),
        "trajectory_manifest_path": (
            _relative(
                repository_root,
                trajectory_manifest_path,
            )
            if trajectory_manifest_path is not None
            else None
        ),
        "trajectory_manifest_sha256": (
            sha256_file(trajectory_manifest_path)
            if trajectory_manifest_path is not None
            else None
        ),
        "trajectory_complete_path": (
            _relative(
                repository_root,
                trajectory_complete_path,
            )
            if trajectory_complete_path is not None
            else None
        ),
        "trajectory_complete_sha256": (
            sha256_file(trajectory_complete_path)
            if trajectory_complete_path is not None
            else None
        ),
        "test_split_read": False,
        "fresh_test_read": False,
        "validation_selection_executed": False,
        **dict(execution_details),
    }
    return value


def _write_candidate_completion(
    *,
    repository_root: Path,
    runtime_root: Path,
    operation: SingleFactorOperation,
    result: Mapping[str, Any],
    sample,
    diagnostics: Mapping[str, Any],
) -> Mapping[str, Any]:
    attempt = _append_only_attempt(
        runtime_root
        / "candidates"
        / operation.model_id
        / operation.candidate_id
        / "seed_2601"
    )
    sample_path = attempt / "validation_sample.npz"
    _save_synthetic_npz(sample_path, sample)
    result_with_sample = dict(result)
    result_with_sample.update(
        {
            "validation_sample_path": _relative(
                repository_root,
                sample_path,
            ),
            "validation_sample_sha256": sha256_file(
                sample_path
            ),
        }
    )
    _exclusive_json(attempt / "diagnostics.json", diagnostics)
    _exclusive_json(
        attempt / "candidate_result.json",
        result_with_sample,
    )
    _exclusive_json(
        attempt / "COMPLETE.json",
        {
            "schema_version": (
                "benchmark-v2.6-single-factor-candidate-terminal-v1"
            ),
            "status": "COMPLETE",
            "candidate_result_sha256": sha256_file(
                attempt / "candidate_result.json"
            ),
            "diagnostics_sha256": sha256_file(
                attempt / "diagnostics.json"
            ),
            "validation_sample_sha256": sha256_file(
                sample_path
            ),
        },
    )
    return {
        "candidate_attempt_path": _relative(
            repository_root,
            attempt,
        ),
        "candidate_result_sha256": sha256_file(
            attempt / "candidate_result.json"
        ),
        "complete_sha256": sha256_file(
            attempt / "COMPLETE.json"
        ),
        "validation_sample_sha256": sha256_file(sample_path),
    }


def _execute_single_factor_training_child(
    payload: Mapping[str, Any],
) -> None:
    import torch

    from eval.single_factor_amendment_v2_6 import (
        compute_model_diagnostics,
    )
    from experiments.candidate_runner_v2_6 import (
        _actual_backend_updates,
        _save_balanced_class_checkpoint,
        build_checkpoint_bundle,
        load_train_only_candidate_context,
        restore_candidate_checkpoint_for_sampling,
    )
    from generators.candidate_adapters_v2_6 import (
        build_candidate_adapter,
    )

    repository_root = Path(payload["repository_root"]).resolve()
    config_path = Path(payload["config_path"]).resolve()
    base_config_path = Path(
        payload["base_config_path"]
    ).resolve()
    authorization_path = Path(
        payload["authorization_path"]
    ).resolve()
    trajectory_attempt = Path(
        payload["trajectory_attempt"]
    ).resolve()
    runtime_root = Path(payload["runtime_root"]).resolve()
    source_commit = str(payload["source_commit"])
    source_hash = str(payload["source_hash"])
    device = str(payload["device"])
    model_id = str(payload["model_id"])
    candidate_id = str(payload["candidate_id"])
    if not device.startswith("cuda"):
        raise SingleFactorRunnerContractError(
            "single-factor learned execution requires CUDA"
        )
    if not torch.cuda.is_available():
        raise SingleFactorRunnerContractError(
            "authorized single-factor CUDA device is unavailable"
        )
    torch.cuda.set_device(torch.device(device))
    plan = build_single_factor_execution_plan(
        config_path=config_path,
        base_selection_config_path=base_config_path,
    )
    operation = _operation_by_id(
        plan,
        model_id=model_id,
        candidate_id=candidate_id,
    )
    if not operation.training_required:
        raise SingleFactorRunnerContractError(
            "training child received a nontraining operation"
        )
    if (
        single_factor_relevant_source_sha256(repository_root)
        != source_hash
    ):
        raise SingleFactorRunnerContractError(
            "single-factor source changed after authorization"
        )
    context = load_train_only_candidate_context(
        repository_root=repository_root,
        development_manifest_path=(
            plan.development_manifest_path
        ),
    )
    spec = _single_factor_adapter_spec(operation)
    factory = _backend_factory(operation)
    adapter = build_candidate_adapter(
        spec,
        backend_factory=factory,
    )
    adapter.bind_train_only_sampling_plan(context.plan)
    checkpoints = trajectory_attempt / "checkpoints"
    checkpoints.mkdir()
    progress_path = trajectory_attempt / "progress.jsonl"

    def progress(event):
        line = (
            json.dumps(
                dict(event),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode()
        descriptor = os.open(
            progress_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        with os.fdopen(descriptor, "ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def checkpoint(event, backend):
        if model_id in {
            "ctgan_separate_class",
            "tvae_separate_class",
        }:
            _save_balanced_class_checkpoint(
                checkpoints=checkpoints,
                backend=backend,
                event=event,
                candidate_steps={20_000},
                train_only_zscore=adapter._zscore,
                shared_trajectory_id=(
                    operation.trajectory_id or ""
                ),
                selection_plan_sha256=context.plan.plan_hash,
            )
            return
        step = int(event["step"])
        if step == 20_000:
            try:
                with (
                    checkpoints / "step_20000.pt"
                ).open("xb") as handle:
                    torch.save(
                        build_checkpoint_bundle(
                            backend=backend,
                            train_only_zscore=adapter._zscore,
                            shared_trajectory_id=(
                                operation.trajectory_id or ""
                            ),
                            sampled_checkpoint_step=20_000,
                            selection_plan_sha256=(
                                context.plan.plan_hash
                            ),
                        ),
                        handle,
                    )
            except FileExistsError as error:
                raise SingleFactorRunnerContractError(
                    "single-factor checkpoint already exists"
                ) from error

    trajectory_manifest = trajectory_attempt / "manifest.json"
    _exclusive_json(
        trajectory_manifest,
        {
            "schema_version": (
                "benchmark-v2.6-single-factor-trajectory-v1"
            ),
            "status": "RUNNING",
            "source_commit": source_commit,
            "relevant_source_sha256": source_hash,
            "single_factor_config_sha256": plan.config_sha256,
            "authorization_sha256": sha256_file(
                authorization_path
            ),
            "model_id": model_id,
            "candidate_id": candidate_id,
            "trajectory_id": operation.trajectory_id,
            "selection_seed": 2601,
            "requested_updates": 20_000,
            "max_wall_seconds": 7_200,
            "train_file_sha256": context.train_file_sha256,
            "train_content_sha256": context.train_content_sha256,
            "validation_file_sha256": (
                context.validation_file_sha256
            ),
            "validation_content_sha256": (
                context.validation_content_sha256
            ),
            "selection_plan_sha256": context.plan.plan_hash,
            "test_split_read": False,
            "device": device,
        },
    )
    adapter_config = _base_adapter_config(
        plan=plan,
        model_id=model_id,
        device=device,
        tau=context.tau,
    )
    adapter_config["progress_callback"] = progress
    adapter_config["checkpoint_callback"] = checkpoint
    adapter.fit_train_only(
        context.train,
        base_config=adapter_config,
        seed=2601,
    )
    backend = adapter._backend
    if backend is None:
        raise SingleFactorRunnerContractError(
            "single-factor backend disappeared"
        )
    actual_updates, wall_reached, training_seconds = (
        _actual_backend_updates(backend)
    )
    if wall_reached or actual_updates != 20_000:
        raise SingleFactorRunnerContractError(
            "single-factor trajectory did not finish its fixed budget"
        )
    checkpoint_path = checkpoints / "step_20000.pt"
    if not checkpoint_path.is_file():
        raise SingleFactorRunnerContractError(
            "single-factor final checkpoint is missing"
        )
    trajectory_complete = (
        trajectory_attempt / "TRAJECTORY_COMPLETE.json"
    )
    _exclusive_json(
        trajectory_complete,
        {
            "schema_version": (
                "benchmark-v2.6-single-factor-trajectory-v1"
            ),
            "status": "COMPLETE",
            "requested_updates": 20_000,
            "actual_updates": actual_updates,
            "wall_cap_reached": False,
            "actual_training_seconds": training_seconds,
            "checkpoint_path": _relative(
                repository_root,
                checkpoint_path,
            ),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
    )
    restored = restore_candidate_checkpoint_for_sampling(
        checkpoint_path,
        model_id=model_id,
        device=device,
        train=context.train,
    )
    sampling_adapter = build_candidate_adapter(spec)
    sampling_adapter.bind_train_only_sampling_plan(context.plan)
    sampling_adapter.attach_trained_backend_for_sampling(
        restored["backend"],
        train_only_zscore=restored["train_only_zscore"],
    )
    sample = sampling_adapter.sample_validation(
        context.plan,
        seed=2601,
    )
    diagnostics = compute_model_diagnostics(
        model_id=model_id,
        sample=sample,
        tau=context.tau,
        receiver_categories=(
            int(context.train.x_cat[..., 0].max()) + 1
        ),
    )
    temporary = trajectory_attempt / "validation_sample.npz"
    _save_synthetic_npz(temporary, sample)
    result = _candidate_manifest(
        repository_root=repository_root,
        operation=operation,
        source_commit=source_commit,
        source_hash=source_hash,
        plan=plan,
        authorization_path=authorization_path,
        authorization_hash=sha256_file(authorization_path),
        context=context,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=sha256_file(checkpoint_path),
        validation_sample_path=temporary,
        actual_updates=actual_updates,
        actual_wall_seconds=training_seconds,
        trajectory_manifest_path=trajectory_manifest,
        trajectory_complete_path=trajectory_complete,
        execution_details={
            "training_calls": 1,
            "sampling_calls": 1,
            "checkpoint_reused": False,
            "single_factor_enforced": True,
        },
    )
    artifact = _write_candidate_completion(
        repository_root=repository_root,
        runtime_root=runtime_root,
        operation=operation,
        result=result,
        sample=sample,
        diagnostics=diagnostics,
    )
    _exclusive_json(
        trajectory_attempt / "CANDIDATE_ARTIFACT.json",
        artifact,
    )


def _apply_evaluation_only_intervention(
    *,
    operation: SingleFactorOperation,
    backend: Any,
) -> None:
    if operation.candidate_id == "ctgan_sf_c02_temperature_only":
        for synthesizer in backend.models.values():
            synthesizer.set_categorical_temperature(0.5)
        return
    if operation.candidate_id == (
        "tvae_sf_c01_categorical_decode_only"
    ):
        for synthesizer in backend.models.values():
            synthesizer.set_candidate_contract(
                channel_weights={
                    "amount": 1.0,
                    "gap": 1.0,
                    "receiver": 1.0,
                },
                latent_scale=1.0,
                categorical_temperature=0.75,
            )
        return
    if operation.candidate_id == (
        "cof_sf_c02_variance_preserving_residual"
    ):
        return
    raise SingleFactorRunnerContractError(
        "unknown evaluation-only intervention"
    )


def _execute_single_factor_evaluation_child(
    payload: Mapping[str, Any],
) -> None:
    import torch

    from benchmarks.types import SyntheticBatch
    from eval.single_factor_amendment_v2_6 import (
        compute_model_diagnostics,
    )
    from experiments.candidate_runner_v2_6 import (
        load_train_only_candidate_context,
        restore_candidate_checkpoint_for_sampling,
    )
    from generators.candidate_adapters_v2_6 import (
        build_candidate_adapter,
    )
    from models.single_factor_components_v2_6 import (
        variance_preserving_residual,
    )

    repository_root = Path(payload["repository_root"]).resolve()
    config_path = Path(payload["config_path"]).resolve()
    base_config_path = Path(
        payload["base_config_path"]
    ).resolve()
    authorization_path = Path(
        payload["authorization_path"]
    ).resolve()
    evaluation_attempt = Path(
        payload["evaluation_attempt"]
    ).resolve()
    runtime_root = Path(payload["runtime_root"]).resolve()
    source_commit = str(payload["source_commit"])
    source_hash = str(payload["source_hash"])
    device = str(payload["device"])
    model_id = str(payload["model_id"])
    candidate_id = str(payload["candidate_id"])
    if not device.startswith("cuda"):
        raise SingleFactorRunnerContractError(
            "single-factor evaluation requires CUDA"
        )
    if not torch.cuda.is_available():
        raise SingleFactorRunnerContractError(
            "authorized single-factor CUDA device is unavailable"
        )
    torch.cuda.set_device(torch.device(device))
    plan = build_single_factor_execution_plan(
        config_path=config_path,
        base_selection_config_path=base_config_path,
    )
    operation = _operation_by_id(
        plan,
        model_id=model_id,
        candidate_id=candidate_id,
    )
    if operation.execution_kind != "evaluate_existing_checkpoint":
        raise SingleFactorRunnerContractError(
            "evaluation child received the wrong operation"
        )
    if (
        single_factor_relevant_source_sha256(repository_root)
        != source_hash
    ):
        raise SingleFactorRunnerContractError(
            "single-factor source changed after authorization"
        )
    context = load_train_only_candidate_context(
        repository_root=repository_root,
        development_manifest_path=(
            plan.development_manifest_path
        ),
    )
    checkpoint_path = (
        repository_root / str(operation.checkpoint_path)
    ).resolve()
    if (
        not checkpoint_path.is_file()
        or sha256_file(checkpoint_path)
        != operation.checkpoint_sha256
    ):
        raise SingleFactorRunnerContractError(
            "evaluation-only checkpoint hash mismatch"
        )
    evaluation_manifest = evaluation_attempt / "manifest.json"
    _exclusive_json(
        evaluation_manifest,
        {
            "schema_version": (
                "benchmark-v2.6-single-factor-evaluation-v1"
            ),
            "status": "RUNNING",
            "source_commit": source_commit,
            "relevant_source_sha256": source_hash,
            "single_factor_config_sha256": plan.config_sha256,
            "authorization_sha256": sha256_file(
                authorization_path
            ),
            "model_id": model_id,
            "candidate_id": candidate_id,
            "selection_seed": 2601,
            "checkpoint_path": _relative(
                repository_root,
                checkpoint_path,
            ),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_calls": 0,
            "test_split_read": False,
            "device": device,
        },
    )
    restored = restore_candidate_checkpoint_for_sampling(
        checkpoint_path,
        model_id=model_id,
        device=device,
        train=context.train,
    )
    backend = restored["backend"]
    _apply_evaluation_only_intervention(
        operation=operation,
        backend=backend,
    )
    spec = _single_factor_adapter_spec(operation)
    adapter = build_candidate_adapter(spec)
    adapter.bind_train_only_sampling_plan(context.plan)
    adapter.attach_trained_backend_for_sampling(
        backend,
        train_only_zscore=restored["train_only_zscore"],
    )
    sample = adapter.sample_validation(
        context.plan,
        seed=2601,
    )
    extra: dict[str, Any] = {}
    if operation.candidate_id == (
        "cof_sf_c02_variance_preserving_residual"
    ):
        amount_tensor = torch.from_numpy(sample.x_num).to(device)
        mask_tensor = torch.from_numpy(sample.valid_mask).to(device)
        adjusted, residual_state = variance_preserving_residual(
            amount_tensor,
            valid_mask=mask_tensor,
            train_variance=float(
                context.train.x_num[
                    context.train.valid_mask,
                    0,
                ].var()
            ),
            seed=2601 + 60_001,
        )
        x_num = adjusted.cpu().numpy().astype(np.float32)
        x_num[~sample.valid_mask] = 0
        sample = SyntheticBatch(
            x_num=x_num,
            dt_bin=sample.dt_bin.copy(),
            x_cat=sample.x_cat.copy(),
            valid_mask=sample.valid_mask.copy(),
            y_entity=sample.y_entity.copy(),
            lengths=sample.lengths.copy(),
        )
        extra["train_fitted_residual_state"] = dict(
            residual_state
        )
    diagnostics = compute_model_diagnostics(
        model_id=model_id,
        sample=sample,
        tau=context.tau,
        receiver_categories=(
            int(context.train.x_cat[..., 0].max()) + 1
        ),
    )
    temporary = evaluation_attempt / "validation_sample.npz"
    _save_synthetic_npz(temporary, sample)
    result = _candidate_manifest(
        repository_root=repository_root,
        operation=operation,
        source_commit=source_commit,
        source_hash=source_hash,
        plan=plan,
        authorization_path=authorization_path,
        authorization_hash=sha256_file(authorization_path),
        context=context,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=sha256_file(checkpoint_path),
        validation_sample_path=temporary,
        actual_updates=0,
        actual_wall_seconds=0.0,
        trajectory_manifest_path=None,
        trajectory_complete_path=None,
        execution_details={
            "training_calls": 0,
            "sampling_calls": 1,
            "checkpoint_reused": True,
            "single_factor_enforced": True,
            **extra,
        },
    )
    artifact = _write_candidate_completion(
        repository_root=repository_root,
        runtime_root=runtime_root,
        operation=operation,
        result=result,
        sample=sample,
        diagnostics=diagnostics,
    )
    _exclusive_json(
        evaluation_attempt / "CANDIDATE_ARTIFACT.json",
        artifact,
    )
    _exclusive_json(
        evaluation_attempt / "EVALUATION_COMPLETE.json",
        {
            "schema_version": (
                "benchmark-v2.6-single-factor-evaluation-v1"
            ),
            "status": "COMPLETE",
            "candidate_id": candidate_id,
            "checkpoint_reused": True,
            "training_calls": 0,
            "candidate_artifact": artifact,
        },
    )


def _write_frozen_control_reference(
    *,
    repository_root: Path,
    runtime_root: Path,
    operation: SingleFactorOperation,
    source_commit: str,
    source_hash: str,
    plan: SingleFactorExecutionPlan,
    authorization_path: Path,
) -> Mapping[str, Any]:
    if operation.execution_kind != "reuse_frozen_control":
        raise SingleFactorRunnerContractError(
            "frozen-control writer received the wrong operation"
        )
    for path_key, hash_key in (
        ("candidate_result_path", "candidate_result_sha256"),
        ("checkpoint_path", "checkpoint_sha256"),
        ("validation_sample_path", "validation_sample_sha256"),
    ):
        path = (
            repository_root
            / operation.frozen_control_paths[path_key]
        ).resolve()
        if (
            not path.is_file()
            or sha256_file(path)
            != operation.frozen_control_hashes[hash_key]
        ):
            raise SingleFactorRunnerContractError(
                f"frozen control changed: {operation.model_id}/{path_key}"
            )
    attempt = _append_only_attempt(
        runtime_root
        / "candidates"
        / operation.model_id
        / operation.candidate_id
        / "seed_2601"
    )
    reference = {
        "schema_version": (
            "benchmark-v2.6-single-factor-frozen-reference-v1"
        ),
        "status": "COMPLETE",
        "source_commit": source_commit,
        "relevant_source_sha256": source_hash,
        "single_factor_config_sha256": plan.config_sha256,
        "authorization_path": _relative(
            repository_root,
            authorization_path,
        ),
        "authorization_sha256": sha256_file(
            authorization_path
        ),
        "model_id": operation.model_id,
        "candidate_id": operation.candidate_id,
        "execution_kind": "reuse_frozen_control",
        "training_calls": 0,
        "sampling_calls": 0,
        "frozen_control_paths": dict(
            operation.frozen_control_paths
        ),
        "frozen_control_hashes": dict(
            operation.frozen_control_hashes
        ),
        "test_split_read": False,
        "validation_selection_executed": False,
    }
    _exclusive_json(attempt / "reference_manifest.json", reference)
    _exclusive_json(
        attempt / "COMPLETE.json",
        {
            "schema_version": (
                "benchmark-v2.6-single-factor-candidate-terminal-v1"
            ),
            "status": "COMPLETE",
            "reference_manifest_sha256": sha256_file(
                attempt / "reference_manifest.json"
            ),
        },
    )
    return {
        "candidate_attempt_path": _relative(
            repository_root,
            attempt,
        ),
        "reference_manifest_sha256": sha256_file(
            attempt / "reference_manifest.json"
        ),
        "complete_sha256": sha256_file(
            attempt / "COMPLETE.json"
        ),
    }


def _read_json_mapping(path: Path, *, role: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SingleFactorRunnerContractError(
            f"cannot read {role}: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise SingleFactorRunnerContractError(
            f"{role} is not a JSON object: {path}"
        )
    return value


def _path_below(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def verify_single_factor_worker_candidate_artifacts(
    *,
    repository_root: Path,
    runtime_root: Path,
    plan: SingleFactorExecutionPlan,
    model_id: str,
    results: list[Mapping[str, Any]],
    source_commit: str,
    source_hash: str,
    authorization_hash: str,
) -> Mapping[str, Mapping[str, Any]]:
    repository_root = repository_root.resolve()
    runtime_root = runtime_root.resolve()
    operations = [
        operation
        for operation in plan.operations
        if operation.model_id == model_id
    ]
    if (
        model_id not in plan.model_ids
        or len(operations) != 3
        or len(results) != 3
        or {
            str(result.get("candidate_id"))
            for result in results
        }
        != {operation.candidate_id for operation in operations}
        or any(result.get("status") != "COMPLETE" for result in results)
    ):
        raise SingleFactorRunnerContractError(
            "all three planned candidate results must be COMPLETE"
        )
    by_candidate = {
        str(result["candidate_id"]): result
        for result in results
    }
    verified: dict[str, Mapping[str, Any]] = {}
    for operation in operations:
        result = by_candidate[operation.candidate_id]
        if operation.execution_kind == "reuse_frozen_control":
            relative_attempt = result.get(
                "candidate_attempt_path"
            )
            artifact_record = result
        else:
            relative_operation_attempt = result.get("attempt_path")
            if not isinstance(relative_operation_attempt, str):
                raise SingleFactorRunnerContractError(
                    "candidate operation attempt path is missing"
                )
            operation_attempt = (
                repository_root / relative_operation_attempt
            ).resolve()
            if not _path_below(operation_attempt, runtime_root):
                raise SingleFactorRunnerContractError(
                    "candidate operation attempt escapes runtime root"
                )
            artifact_record = _read_json_mapping(
                operation_attempt / "CANDIDATE_ARTIFACT.json",
                role="candidate artifact pointer",
            )
            relative_attempt = artifact_record.get(
                "candidate_attempt_path"
            )
        if not isinstance(relative_attempt, str):
            raise SingleFactorRunnerContractError(
                "candidate attempt path is missing"
            )
        candidate_attempt = (
            repository_root / relative_attempt
        ).resolve()
        expected_candidate_root = (
            runtime_root
            / "candidates"
            / model_id
            / operation.candidate_id
            / "seed_2601"
        ).resolve()
        if (
            not _path_below(
                candidate_attempt,
                expected_candidate_root,
            )
            or not candidate_attempt.name.startswith("attempt_")
            or not candidate_attempt.is_dir()
        ):
            raise SingleFactorRunnerContractError(
                "candidate attempt path/provenance mismatch"
            )
        complete_path = candidate_attempt / "COMPLETE.json"
        complete_hash = sha256_file(complete_path)
        if complete_hash != artifact_record.get("complete_sha256"):
            raise SingleFactorRunnerContractError(
                f"candidate COMPLETE hash mismatch: "
                f"{operation.candidate_id}"
            )
        complete = _read_json_mapping(
            complete_path,
            role="candidate completion",
        )
        if complete.get("status") != "COMPLETE":
            raise SingleFactorRunnerContractError(
                f"candidate is not COMPLETE: {operation.candidate_id}"
            )
        record: dict[str, Any] = {
            "candidate_attempt_path": _relative(
                repository_root,
                candidate_attempt,
            ),
            "complete_path": _relative(
                repository_root,
                complete_path,
            ),
            "complete_sha256": complete_hash,
            "execution_kind": operation.execution_kind,
        }
        if operation.execution_kind == "reuse_frozen_control":
            reference_path = (
                candidate_attempt / "reference_manifest.json"
            )
            reference_hash = sha256_file(reference_path)
            if (
                reference_hash
                != artifact_record.get("reference_manifest_sha256")
                or complete.get("reference_manifest_sha256")
                != reference_hash
            ):
                raise SingleFactorRunnerContractError(
                    "frozen-control reference hash mismatch"
                )
            reference = _read_json_mapping(
                reference_path,
                role="frozen-control reference",
            )
            expected_reference = {
                "status": "COMPLETE",
                "source_commit": source_commit,
                "relevant_source_sha256": source_hash,
                "single_factor_config_sha256": plan.config_sha256,
                "authorization_sha256": authorization_hash,
                "model_id": model_id,
                "candidate_id": operation.candidate_id,
                "execution_kind": "reuse_frozen_control",
                "training_calls": 0,
                "sampling_calls": 0,
                "test_split_read": False,
                "validation_selection_executed": False,
            }
            if any(
                reference.get(key) != value
                for key, value in expected_reference.items()
            ):
                raise SingleFactorRunnerContractError(
                    "frozen-control reference provenance mismatch"
                )
            record.update(
                {
                    "reference_manifest_path": _relative(
                        repository_root,
                        reference_path,
                    ),
                    "reference_manifest_sha256": reference_hash,
                }
            )
        else:
            candidate_result_path = (
                candidate_attempt / "candidate_result.json"
            )
            diagnostics_path = (
                candidate_attempt / "diagnostics.json"
            )
            sample_path = (
                candidate_attempt / "validation_sample.npz"
            )
            hashes = {
                "candidate_result_sha256": sha256_file(
                    candidate_result_path
                ),
                "diagnostics_sha256": sha256_file(
                    diagnostics_path
                ),
                "validation_sample_sha256": sha256_file(
                    sample_path
                ),
            }
            if (
                hashes["candidate_result_sha256"]
                != artifact_record.get(
                    "candidate_result_sha256"
                )
                or hashes["validation_sample_sha256"]
                != artifact_record.get(
                    "validation_sample_sha256"
                )
                or any(
                    complete.get(key) != value
                    for key, value in hashes.items()
                )
            ):
                raise SingleFactorRunnerContractError(
                    f"candidate artifact hash mismatch: "
                    f"{operation.candidate_id}"
                )
            candidate_result = _read_json_mapping(
                candidate_result_path,
                role="candidate result",
            )
            expected_result = {
                "status": "COMPLETE",
                "source_commit": source_commit,
                "relevant_source_sha256": source_hash,
                "single_factor_config_sha256": plan.config_sha256,
                "authorization_sha256": authorization_hash,
                "model_id": model_id,
                "candidate_id": operation.candidate_id,
                "test_split_read": False,
                "validation_selection_executed": False,
                "validation_sample_sha256": hashes[
                    "validation_sample_sha256"
                ],
            }
            if any(
                candidate_result.get(key) != value
                for key, value in expected_result.items()
            ):
                raise SingleFactorRunnerContractError(
                    f"candidate result provenance mismatch: "
                    f"{operation.candidate_id}"
                )
            record.update(
                {
                    "candidate_result_path": _relative(
                        repository_root,
                        candidate_result_path,
                    ),
                    "diagnostics_path": _relative(
                        repository_root,
                        diagnostics_path,
                    ),
                    "validation_sample_path": _relative(
                        repository_root,
                        sample_path,
                    ),
                    **hashes,
                }
            )
        verified[operation.candidate_id] = record
    return verified


def finalize_single_factor_worker_attempt(
    *,
    repository_root: Path,
    runtime_root: Path,
    worker_attempt: Path,
    plan: SingleFactorExecutionPlan,
    model_id: str,
    results: list[Mapping[str, Any]],
    source_commit: str,
    source_hash: str,
    authorization_hash: str,
    finalization_kind: str,
    candidate_source_commit: str | None = None,
    candidate_source_hash: str | None = None,
    recovery_authorization_hash: str | None = None,
) -> Mapping[str, Any]:
    if finalization_kind not in {"normal", "recovery"}:
        raise ValueError("invalid worker finalization kind")
    worker_attempt = worker_attempt.resolve()
    if (
        (worker_attempt / "WORKER_COMPLETE.json").exists()
        or (worker_attempt / "WORKER_FAILED.json").exists()
    ):
        raise SingleFactorRunnerContractError(
            "worker attempt is already terminal"
        )
    candidate_artifacts = (
        verify_single_factor_worker_candidate_artifacts(
            repository_root=repository_root,
            runtime_root=runtime_root,
            plan=plan,
            model_id=model_id,
            results=results,
            source_commit=(
                candidate_source_commit or source_commit
            ),
            source_hash=candidate_source_hash or source_hash,
            authorization_hash=authorization_hash,
        )
    )
    terminal = {
        "schema_version": (
            "benchmark-v2.6-single-factor-model-worker-v2"
        ),
        "status": "COMPLETE",
        "model_id": model_id,
        "candidate_count": 3,
        "new_trajectory_count": 1,
        "evaluation_only_count": 1,
        "reused_control_count": 1,
        "results": [dict(result) for result in results],
        "candidate_artifacts": candidate_artifacts,
        "source_commit": source_commit,
        "relevant_source_sha256": source_hash,
        "single_factor_config_sha256": plan.config_sha256,
        "authorization_sha256": authorization_hash,
        "finalization_kind": finalization_kind,
        "selection_aggregate_created": False,
        "test_split_read": False,
        "training_calls_during_finalization": 0,
        "sampling_calls_during_finalization": 0,
        "completed_at": _utc_now(),
    }
    if finalization_kind == "recovery":
        if (
            candidate_source_commit is None
            or candidate_source_hash is None
            or recovery_authorization_hash is None
        ):
            raise SingleFactorRunnerContractError(
                "recovery finalization provenance is incomplete"
            )
        terminal.update(
            {
                "candidate_source_commit": (
                    candidate_source_commit
                ),
                "candidate_relevant_source_sha256": (
                    candidate_source_hash
                ),
                "recovery_authorization_sha256": (
                    recovery_authorization_hash
                ),
            }
        )
    _exclusive_json(
        worker_attempt / "WORKER_COMPLETE.json",
        terminal,
    )
    return terminal


def record_single_factor_worker_failure(
    *,
    worker_attempt: Path,
    model_id: str,
    results: list[Mapping[str, Any]],
    failure_class: str,
    exception: str,
    source_commit: str,
    source_hash: str,
    config_hash: str,
    authorization_hash: str,
) -> Mapping[str, Any]:
    worker_attempt = worker_attempt.resolve()
    if (
        (worker_attempt / "WORKER_COMPLETE.json").exists()
        or (worker_attempt / "WORKER_FAILED.json").exists()
    ):
        raise SingleFactorRunnerContractError(
            "worker attempt is already terminal"
        )
    terminal = {
        "schema_version": (
            "benchmark-v2.6-single-factor-model-worker-v2"
        ),
        "status": "FAILED",
        "model_id": model_id,
        "failure_class": failure_class,
        "exception": exception,
        "results": [dict(result) for result in results],
        "source_commit": source_commit,
        "relevant_source_sha256": source_hash,
        "single_factor_config_sha256": config_hash,
        "authorization_sha256": authorization_hash,
        "selection_aggregate_created": False,
        "test_split_read": False,
        "failed_at": _utc_now(),
    }
    _exclusive_json(
        worker_attempt / "WORKER_FAILED.json",
        terminal,
    )
    return terminal


def single_factor_model_tree_sha256(
    *,
    runtime_root: Path,
    model_id: str,
) -> tuple[str, int, int]:
    if model_id not in SINGLE_FACTOR_MODEL_IDS:
        raise SingleFactorRunnerContractError(
            f"unknown single-factor model: {model_id}"
        )
    runtime_root = runtime_root.resolve()
    roots = (
        runtime_root / "workers" / model_id,
        runtime_root / "candidates" / model_id,
        runtime_root / "trajectories" / model_id,
        runtime_root / "evaluations" / model_id,
    )
    files = sorted(
        {
            path
            for root in roots
            if root.exists()
            for path in (
                [root]
                if root.is_file()
                else root.rglob("*")
            )
            if path.is_file()
        },
        key=lambda path: path.relative_to(runtime_root).as_posix(),
    )
    if not files:
        raise SingleFactorRunnerContractError(
            "single-factor model tree is empty"
        )
    digest = hashlib.sha256()
    byte_count = 0
    for path in files:
        relative = path.relative_to(runtime_root).as_posix()
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(
                lambda: handle.read(1024 * 1024),
                b"",
            ):
                byte_count += len(chunk)
                file_digest.update(chunk)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(file_digest.hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest(), len(files), byte_count


def recover_single_factor_worker_finalization(
    *,
    repository_root: Path,
    runtime_root: Path,
    plan: SingleFactorExecutionPlan,
    authorization: Mapping[str, Any],
    recovery_authorization_hash: str,
    current_source_commit: str,
    current_source_hash: str,
) -> Mapping[str, Any]:
    repository_root = repository_root.resolve()
    runtime_root = runtime_root.resolve()
    model_id = str(authorization.get("model_id", ""))
    expected_flags = {
        "terminal_marker_write": True,
        "gpu_query": False,
        "training": False,
        "sampling": False,
        "selection": False,
        "test_split_access": False,
        "fresh_test": False,
        "full_run": False,
    }
    expected_scalars = {
        "schema_version": (
            "benchmark-v2.6-single-factor-finalization-recovery-"
            "authorization-v1"
        ),
        "status": "AUTHORIZED",
        "authorized_action": "FINALIZATION_ONLY",
        "recovery_source_commit": current_source_commit,
        "recovery_relevant_source_sha256": current_source_hash,
        "single_factor_config_sha256": plan.config_sha256,
    }
    if (
        model_id not in plan.model_ids
        or any(
            authorization.get(key) != value
            for key, value in expected_scalars.items()
        )
        or authorization.get("authorization") != expected_flags
    ):
        raise SingleFactorRunnerContractError(
            "finalization recovery authorization mismatch"
        )
    candidate_source_commit = authorization.get(
        "candidate_source_commit"
    )
    candidate_source_hash = authorization.get(
        "candidate_relevant_source_sha256"
    )
    execution_authorization_hash = authorization.get(
        "execution_authorization_sha256"
    )
    results = authorization.get("results")
    if (
        not isinstance(candidate_source_commit, str)
        or len(candidate_source_commit) != 40
        or not isinstance(candidate_source_hash, str)
        or len(candidate_source_hash) != 64
        or not isinstance(execution_authorization_hash, str)
        or len(execution_authorization_hash) != 64
        or not isinstance(results, list)
    ):
        raise SingleFactorRunnerContractError(
            "finalization recovery provenance is incomplete"
        )
    relative_prior = authorization.get(
        "prior_worker_attempt_path"
    )
    if not isinstance(relative_prior, str):
        raise SingleFactorRunnerContractError(
            "prior worker attempt path is missing"
        )
    prior_worker = (repository_root / relative_prior).resolve()
    expected_worker_root = (
        runtime_root / "workers" / model_id
    ).resolve()
    if (
        not _path_below(prior_worker, expected_worker_root)
        or not prior_worker.is_dir()
        or (prior_worker / "WORKER_COMPLETE.json").exists()
        or (prior_worker / "WORKER_FAILED.json").exists()
    ):
        raise SingleFactorRunnerContractError(
            "prior worker is absent or already terminal"
        )
    tree_hash, tree_files, tree_bytes = (
        single_factor_model_tree_sha256(
            runtime_root=runtime_root,
            model_id=model_id,
        )
    )
    if (
        authorization.get("prior_model_tree_sha256")
        != tree_hash
        or int(
            authorization.get("prior_model_tree_files", -1)
        )
        != tree_files
        or int(
            authorization.get("prior_model_tree_bytes", -1)
        )
        != tree_bytes
    ):
        raise SingleFactorRunnerContractError(
            "prior model tree preservation hash mismatch"
        )
    verified = verify_single_factor_worker_candidate_artifacts(
        repository_root=repository_root,
        runtime_root=runtime_root,
        plan=plan,
        model_id=model_id,
        results=results,
        source_commit=candidate_source_commit,
        source_hash=candidate_source_hash,
        authorization_hash=execution_authorization_hash,
    )
    worker_attempt = _append_only_attempt(expected_worker_root)
    _exclusive_json(
        worker_attempt / "recovery_manifest.json",
        {
            "schema_version": (
                "benchmark-v2.6-single-factor-finalization-recovery-v1"
            ),
            "status": "VERIFIED_FOR_FINALIZATION",
            "model_id": model_id,
            "prior_worker_attempt_path": relative_prior,
            "prior_model_tree_sha256": tree_hash,
            "prior_model_tree_files": tree_files,
            "prior_model_tree_bytes": tree_bytes,
            "candidate_artifacts": verified,
            "candidate_source_commit": candidate_source_commit,
            "candidate_relevant_source_sha256": (
                candidate_source_hash
            ),
            "recovery_source_commit": current_source_commit,
            "recovery_relevant_source_sha256": (
                current_source_hash
            ),
            "single_factor_config_sha256": plan.config_sha256,
            "execution_authorization_sha256": (
                execution_authorization_hash
            ),
            "recovery_authorization_sha256": (
                recovery_authorization_hash
            ),
            "training_calls": 0,
            "sampling_calls": 0,
            "gpu_queries": 0,
            "test_split_read": False,
            "selection_aggregate_created": False,
            "created_at": _utc_now(),
        },
    )
    return finalize_single_factor_worker_attempt(
        repository_root=repository_root,
        runtime_root=runtime_root,
        worker_attempt=worker_attempt,
        plan=plan,
        model_id=model_id,
        results=results,
        source_commit=current_source_commit,
        source_hash=current_source_hash,
        authorization_hash=execution_authorization_hash,
        finalization_kind="recovery",
        candidate_source_commit=candidate_source_commit,
        candidate_source_hash=candidate_source_hash,
        recovery_authorization_hash=(
            recovery_authorization_hash
        ),
    )


def execute_single_factor_worker(
    *,
    repository_root: Path,
    config_path: Path,
    base_selection_config_path: Path,
    authorization_path: Path,
    source_commit: str,
    model_id: str,
    device: str,
    termination_grace_seconds: float = 10.0,
) -> Mapping[str, Any]:
    from experiments.candidate_runner_v2_6 import (
        run_bounded_process,
    )

    repository_root = repository_root.resolve()
    authorization_path = authorization_path.resolve()
    plan = build_single_factor_execution_plan(
        config_path=config_path,
        base_selection_config_path=base_selection_config_path,
    )
    authorization = read_authorization(authorization_path)
    prior_tree_hash = str(
        authorization.get("prior_v2_6_candidate_tree_sha256", "")
    )
    dry_run_single_factor_worker(
        repository_root=repository_root,
        plan=plan,
        authorization=authorization,
        source_commit=source_commit,
        prior_candidate_tree_sha256=prior_tree_hash,
        model_id=model_id,
        verify_files=True,
    )
    if not device.startswith("cuda"):
        raise SingleFactorRunnerContractError(
            "execute mode requires an explicit CUDA device"
        )
    source_hash = single_factor_relevant_source_sha256(
        repository_root
    )
    authorization_hash = sha256_file(authorization_path)
    runtime_root = repository_root / RUNTIME_ROOT
    workers_root = runtime_root / "workers" / model_id
    workers_root.mkdir(parents=True, exist_ok=True)
    ownership = workers_root / "ownership.lock"
    _exclusive_json(
        ownership,
        {
            "schema_version": (
                "benchmark-v2.6-single-factor-worker-lock-v1"
            ),
            "model_id": model_id,
            "source_commit": source_commit,
            "relevant_source_sha256": source_hash,
            "single_factor_config_sha256": plan.config_sha256,
            "authorization_sha256": authorization_hash,
            "claimed_at": _utc_now(),
        },
    )
    worker_attempt = _append_only_attempt(workers_root)
    operations = [
        operation
        for operation in plan.operations
        if operation.model_id == model_id
    ]
    results: list[Mapping[str, Any]] = []
    try:
        worker_manifest = worker_attempt / "manifest.json"
        _exclusive_json(
            worker_manifest,
            {
                "schema_version": (
                    "benchmark-v2.6-single-factor-model-worker-v2"
                ),
                "status": "RUNNING",
                "model_id": model_id,
                "source_commit": source_commit,
                "relevant_source_sha256": source_hash,
                "single_factor_config_sha256": plan.config_sha256,
                "authorization_sha256": authorization_hash,
                "candidate_ids": [
                    operation.candidate_id
                    for operation in operations
                ],
                "operation_count": 3,
                "test_split_read": False,
                "selection_aggregate_created": False,
                "started_at": _utc_now(),
            },
        )
        for operation_index, operation in enumerate(operations):
            if operation.execution_kind == "reuse_frozen_control":
                results.append(
                    {
                        "candidate_id": operation.candidate_id,
                        "status": "COMPLETE",
                        "operation": "reuse_frozen_control",
                        **_write_frozen_control_reference(
                            repository_root=repository_root,
                            runtime_root=runtime_root,
                            operation=operation,
                            source_commit=source_commit,
                            source_hash=source_hash,
                            plan=plan,
                            authorization_path=authorization_path,
                        ),
                    }
                )
                continue
            common_payload = {
                "repository_root": str(repository_root),
                "config_path": str(plan.config_path),
                "base_config_path": str(
                    plan.base_selection_config_path
                ),
                "authorization_path": str(authorization_path),
                "runtime_root": str(runtime_root),
                "source_commit": source_commit,
                "source_hash": source_hash,
                "model_id": model_id,
                "candidate_id": operation.candidate_id,
                "device": device,
            }
            if operation.execution_kind == "train_new_trajectory":
                attempt = _append_only_attempt(
                    runtime_root
                    / "trajectories"
                    / model_id
                    / str(operation.trajectory_id)
                    / "seed_2601"
                )
                target = _execute_single_factor_training_child
                payload = {
                    **common_payload,
                    "trajectory_attempt": str(attempt),
                }
                max_wall_seconds = 7_200.0
                operation_name = "train_new_trajectory"
            else:
                attempt = _append_only_attempt(
                    runtime_root
                    / "evaluations"
                    / model_id
                    / operation.candidate_id
                    / "seed_2601"
                )
                target = _execute_single_factor_evaluation_child
                payload = {
                    **common_payload,
                    "evaluation_attempt": str(attempt),
                }
                max_wall_seconds = 1_800.0
                operation_name = "evaluate_existing_checkpoint"

            def enrich(
                value: Mapping[str, Any],
            ) -> Mapping[str, Any]:
                return {
                    **dict(value),
                    "candidate_id": operation.candidate_id,
                    "operation": operation_name,
                    "attempt_path": _relative(
                        repository_root,
                        attempt,
                    ),
                }

            final_operation = operation_index == len(operations) - 1

            def finalize_after_target(
                provisional: Mapping[str, Any],
            ) -> None:
                finalize_single_factor_worker_attempt(
                    repository_root=repository_root,
                    runtime_root=runtime_root,
                    worker_attempt=worker_attempt,
                    plan=plan,
                    model_id=model_id,
                    results=[
                        *results,
                        enrich(provisional),
                    ],
                    source_commit=source_commit,
                    source_hash=source_hash,
                    authorization_hash=authorization_hash,
                    finalization_kind="normal",
                )

            result = run_bounded_process(
                target=target,
                attempt_path=attempt,
                max_wall_seconds=max_wall_seconds,
                termination_grace_seconds=(
                    termination_grace_seconds
                ),
                args=(payload,),
                on_target_complete=(
                    finalize_after_target
                    if final_operation
                    else None
                ),
                target_complete_cleanup_grace_seconds=(
                    2.0 if final_operation else None
                ),
            )
            results.append(enrich(result))
            if result.get("status") != "COMPLETE":
                return record_single_factor_worker_failure(
                    worker_attempt=worker_attempt,
                    model_id=model_id,
                    results=results,
                    failure_class=str(
                        result.get(
                            "failure_class",
                            "infrastructure_or_code",
                        )
                    ),
                    exception=str(result.get("exception")),
                    source_commit=source_commit,
                    source_hash=source_hash,
                    config_hash=plan.config_sha256,
                    authorization_hash=authorization_hash,
                )
        complete_path = worker_attempt / "WORKER_COMPLETE.json"
        if complete_path.is_file():
            return _read_json_mapping(
                complete_path,
                role="worker completion",
            )
        return finalize_single_factor_worker_attempt(
            repository_root=repository_root,
            runtime_root=runtime_root,
            worker_attempt=worker_attempt,
            plan=plan,
            model_id=model_id,
            results=results,
            source_commit=source_commit,
            source_hash=source_hash,
            authorization_hash=authorization_hash,
            finalization_kind="normal",
        )
    except BaseException as error:
        complete_path = worker_attempt / "WORKER_COMPLETE.json"
        failed_path = worker_attempt / "WORKER_FAILED.json"
        if complete_path.is_file():
            return _read_json_mapping(
                complete_path,
                role="worker completion",
            )
        if not failed_path.exists():
            failure_class = (
                "interrupted"
                if isinstance(
                    error,
                    (KeyboardInterrupt, SystemExit),
                )
                else (
                    "contract"
                    if isinstance(
                        error,
                        SingleFactorRunnerContractError,
                    )
                    else "infrastructure_or_code"
                )
            )
            record_single_factor_worker_failure(
                worker_attempt=worker_attempt,
                model_id=model_id,
                results=results,
                failure_class=failure_class,
                exception=repr(error),
                source_commit=source_commit,
                source_hash=source_hash,
                config_hash=plan.config_sha256,
                authorization_hash=authorization_hash,
            )
        raise
