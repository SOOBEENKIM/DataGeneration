from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import subprocess
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from benchmarks.types import SequenceBatch
from eval.validation_selection_v2_6 import (
    CANDIDATE_RESULT_SCHEMA,
    MODEL_IDS,
    SECONDARY_MODEL_IDS,
    SelectionContractError,
    build_selection_freeze,
    read_yaml_mapping,
    run_validation_selection,
    sha256_file,
    validate_selection_config,
)
from experiments.provenance_v2_5 import hash_batch
from generators.candidate_adapters_v2_6 import (
    CandidateAdapterContractError,
    CandidateAdapterSpec,
    TrainOnlyZScore,
    build_candidate_adapter,
    candidate_adapter_spec,
)
from generators.sampling_plan import SamplingPlan


TRAJECTORY_SCHEMA = "benchmark-v2.6-candidate-trajectory-v1"
RUNNER_SCHEMA = "benchmark-v2.6-candidate-runner-v1"
WORKER_SCHEMA = "benchmark-v2.6-model-worker-v1"
CONTINUATION_WORKER_SCHEMA = "benchmark-v2.6-model-worker-continuation-v1"
_FORBIDDEN_PATH_PARTS = {
    "test",
    "test.npz",
    "test_split",
    "heldout_test",
    "fresh_test",
}


class CandidateRunnerContractError(RuntimeError):
    pass


class InvalidCandidateError(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateTrajectory:
    model_id: str
    shared_trajectory_id: str
    candidates: tuple[CandidateAdapterSpec, ...]
    requested_updates: int
    max_wall_seconds: float


@dataclass(frozen=True)
class CandidateTrajectoryPlan:
    selection_seed: int
    trajectories: tuple[CandidateTrajectory, ...]


@dataclass(frozen=True)
class CandidateContinuationPlan:
    model_id: str
    selection_seed: int
    evaluation_only_candidates: tuple[CandidateAdapterSpec, ...]
    training_trajectories: tuple[CandidateTrajectory, ...]
    native_training_trajectory_count: int = 0


@dataclass(frozen=True)
class TrainOnlyCandidateContext:
    train: SequenceBatch
    plan: SamplingPlan
    tau: np.ndarray
    manifest_sha256: str
    train_file_sha256: str
    train_content_sha256: str
    validation_file_sha256: str
    validation_content_sha256: str


def _path_is_below(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def validate_candidate_io_path(
    *,
    repository_root: Path,
    path: Path,
    role: str,
    access: str,
) -> Path:
    repository_root = repository_root.resolve()
    resolved = path.resolve()
    if not _path_is_below(resolved, repository_root):
        raise CandidateRunnerContractError(
            f"{role} is forbidden outside the repository: {resolved}"
        )
    relative = resolved.relative_to(repository_root)
    lowered = tuple(part.lower() for part in relative.parts)
    if set(lowered) & _FORBIDDEN_PATH_PARTS:
        raise CandidateRunnerContractError(
            f"{role} uses a forbidden test/fresh-test path: {resolved}"
        )
    if lowered[:2] == ("artifacts", "benchmark_v2_5"):
        raise CandidateRunnerContractError(
            f"{role} uses a forbidden v2.5 runtime path: {resolved}"
        )
    if (
        len(lowered) >= 3
        and lowered[:2] == ("artifacts", "benchmark_v2_6")
        and lowered[2] == "full"
    ):
        raise CandidateRunnerContractError(
            f"{role} uses a forbidden v2.6 full-run path: {resolved}"
        )
    if access == "write":
        allowed = (
            repository_root
            / "artifacts/benchmark_v2_6/selection/candidates"
        )
        if not _path_is_below(resolved, allowed):
            raise CandidateRunnerContractError(
                f"{role} write is forbidden outside the candidate root"
            )
    elif access != "read":
        raise ValueError("access must be read or write")
    return resolved


def build_trajectory_plan(
    config: Mapping[str, Any],
) -> CandidateTrajectoryPlan:
    try:
        validated = validate_selection_config(config)
    except SelectionContractError as error:
        raise CandidateRunnerContractError(str(error)) from error
    groups: dict[tuple[str, str], list[CandidateAdapterSpec]] = {}
    for model_id, candidates in validated["definitions"].items():
        for candidate in candidates:
            spec = candidate_adapter_spec(
                config,
                model_id=model_id,
                candidate_id=str(candidate["candidate_id"]),
            )
            groups.setdefault(
                (model_id, spec.shared_trajectory_id),
                [],
            ).append(spec)
    trajectories = []
    for (model_id, trajectory_id), candidates in groups.items():
        requested = {item.requested_updates for item in candidates}
        wall = {item.max_wall_seconds for item in candidates}
        if len(requested) != 1 or len(wall) != 1:
            raise CandidateRunnerContractError(
                f"shared trajectory resources differ: {trajectory_id}"
            )
        checkpoints = [
            item.sampled_checkpoint_step for item in candidates
        ]
        if len(checkpoints) != len(set(checkpoints)):
            raise CandidateRunnerContractError(
                f"shared trajectory duplicates an evaluation checkpoint: "
                f"{trajectory_id}"
            )
        trajectories.append(
            CandidateTrajectory(
                model_id=model_id,
                shared_trajectory_id=trajectory_id,
                candidates=tuple(
                    sorted(candidates, key=lambda item: item.candidate_id)
                ),
                requested_updates=next(iter(requested)),
                max_wall_seconds=next(iter(wall)),
            )
        )
    trajectories.sort(
        key=lambda item: (item.model_id, item.shared_trajectory_id)
    )
    if (
        len(trajectories) != validated["training_trajectory_count"]
        or sum(len(item.candidates) for item in trajectories)
        != validated["evaluation_candidate_count"]
    ):
        raise CandidateRunnerContractError(
            "trajectory/candidate resource accounting mismatch"
        )
    return CandidateTrajectoryPlan(
        selection_seed=int(validated["selection_seed"]),
        trajectories=tuple(trajectories),
    )


def build_model_trajectory_plan(
    config: Mapping[str, Any],
    *,
    model_id: str,
) -> CandidateTrajectoryPlan:
    if model_id not in MODEL_IDS:
        raise CandidateRunnerContractError(
            f"unknown v2.6 candidate worker model: {model_id}"
        )
    plan = build_trajectory_plan(config)
    trajectories = tuple(
        trajectory
        for trajectory in plan.trajectories
        if trajectory.model_id == model_id
    )
    if (
        len(trajectories) != 3
        or sum(len(item.candidates) for item in trajectories) != 4
    ):
        raise CandidateRunnerContractError(
            f"model worker resource accounting mismatch: {model_id}"
        )
    return CandidateTrajectoryPlan(
        selection_seed=plan.selection_seed,
        trajectories=trajectories,
    )


def build_model_continuation_plan(
    config: Mapping[str, Any],
    *,
    model_id: str,
) -> CandidateContinuationPlan:
    if model_id not in {
        "ctgan_separate_class",
        "tvae_separate_class",
    }:
        raise CandidateRunnerContractError(
            "continuation is limited to CTGAN/TVAE failed workers"
        )
    original = build_model_trajectory_plan(config, model_id=model_id)
    native = [
        trajectory
        for trajectory in original.trajectories
        if {
            candidate.candidate_id
            for candidate in trajectory.candidates
        }
        == {
            "c00_native_checkpoint_10000",
            "c01_native_checkpoint_20000",
        }
    ]
    if len(native) != 1:
        raise CandidateRunnerContractError(
            f"continuation native trajectory mismatch: {model_id}"
        )
    remaining = tuple(
        trajectory
        for trajectory in original.trajectories
        if trajectory is not native[0]
    )
    if (
        len(remaining) != 2
        or any(len(trajectory.candidates) != 1 for trajectory in remaining)
    ):
        raise CandidateRunnerContractError(
            f"continuation remaining trajectory mismatch: {model_id}"
        )
    return CandidateContinuationPlan(
        model_id=model_id,
        selection_seed=original.selection_seed,
        evaluation_only_candidates=native[0].candidates,
        training_trajectories=remaining,
    )


def dispatch_continuation_plan(
    plan: CandidateContinuationPlan,
    *,
    evaluate_native: Callable[
        [tuple[CandidateAdapterSpec, ...]],
        Mapping[str, Any],
    ],
    train_trajectory: Callable[
        [CandidateTrajectory],
        Mapping[str, Any],
    ],
) -> tuple[Mapping[str, Any], ...]:
    results = [evaluate_native(plan.evaluation_only_candidates)]
    if results[-1].get("status") != "COMPLETE":
        return tuple(results)
    for trajectory in plan.training_trajectories:
        results.append(train_trajectory(trajectory))
        if results[-1].get("status") != "COMPLETE":
            break
    return tuple(results)


def _load_sequence_batch(path: Path) -> SequenceBatch:
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {
                field: archive[field]
                for field in SequenceBatch.__dataclass_fields__
            }
    except (OSError, ValueError, KeyError) as error:
        raise CandidateRunnerContractError(
            f"cannot load frozen train split: {path}"
        ) from error
    return SequenceBatch(**values)


def load_train_only_candidate_context(
    *,
    repository_root: Path,
    development_manifest_path: Path,
) -> TrainOnlyCandidateContext:
    repository_root = repository_root.resolve()
    manifest_path = validate_candidate_io_path(
        repository_root=repository_root,
        path=development_manifest_path,
        role="development manifest",
        access="read",
    )
    manifest = read_yaml_mapping(manifest_path)
    if (
        manifest.get("schema_version")
        != "benchmark-v2.6-development-data-v1"
        or manifest.get("status") != "FROZEN_FOR_SELECTION"
        or manifest.get("test_split_access") != "FORBIDDEN"
    ):
        raise CandidateRunnerContractError(
            "development manifest is not frozen train/validation-only"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {
        "train",
        "validation",
    }:
        raise CandidateRunnerContractError(
            "development manifest must contain train and validation metadata"
        )
    train_record = splits["train"]
    validation_record = splits["validation"]
    if not isinstance(train_record, Mapping) or not isinstance(
        validation_record,
        Mapping,
    ):
        raise CandidateRunnerContractError("development split metadata is invalid")
    train_value = Path(str(train_record["path"]))
    train_path = validate_candidate_io_path(
        repository_root=repository_root,
        path=(
            train_value
            if train_value.is_absolute()
            else repository_root / train_value
        ),
        role="frozen train split",
        access="read",
    )
    relative = train_path.relative_to(repository_root)
    if (
        relative.parts[:3]
        != ("data", "benchmark_v2_5", "frozen")
        or train_path.name != "train.npz"
    ):
        raise CandidateRunnerContractError(
            "candidate train input must be the frozen v2.5 train.npz"
        )
    if sha256_file(train_path) != train_record.get("file_sha256"):
        raise CandidateRunnerContractError("frozen train file hash mismatch")
    train = _load_sequence_batch(train_path)
    train_content_hash = hash_batch(train)
    if train_content_hash != train_record.get("content_sha256"):
        raise CandidateRunnerContractError("frozen train content hash mismatch")
    if int(train_record.get("entities", -1)) != len(train.lengths):
        raise CandidateRunnerContractError("frozen train entity count mismatch")
    plan_record = manifest.get("selection_sampling_plan")
    if (
        not isinstance(plan_record, Mapping)
        or plan_record.get("fit_split") != "train"
    ):
        raise CandidateRunnerContractError(
            "SamplingPlan must be fitted from train only"
        )
    plan = SamplingPlan.from_train_policy(
        train,
        entity_count=int(plan_record["entity_count"]),
        seed=int(plan_record["seed"]),
    )
    if plan.plan_hash != plan_record.get("content_sha256"):
        raise CandidateRunnerContractError(
            "train-only SamplingPlan hash mismatch"
        )
    if int(validation_record.get("entities", -1)) != len(plan.lengths):
        raise CandidateRunnerContractError(
            "plan entity count differs from frozen validation metadata"
        )
    tau = np.asarray(manifest.get("tau"), dtype=np.float32)
    if tau.ndim != 1 or not len(tau) or not np.isfinite(tau).all():
        raise CandidateRunnerContractError("development tau is invalid")
    return TrainOnlyCandidateContext(
        train=train,
        plan=plan,
        tau=tau,
        manifest_sha256=sha256_file(manifest_path),
        train_file_sha256=sha256_file(train_path),
        train_content_sha256=train_content_hash,
        validation_file_sha256=str(
            validation_record["file_sha256"]
        ),
        validation_content_sha256=str(
            validation_record["content_sha256"]
        ),
    )


def claim_next_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 1_000_000):
        attempt = root / f"attempt_{number:03d}"
        try:
            attempt.mkdir()
        except FileExistsError:
            continue
        return attempt
    raise CandidateRunnerContractError("attempt namespace exhausted")


def claim_exact_attempt(root: Path, *, attempt_name: str) -> Path:
    if (
        not attempt_name.startswith("attempt_")
        or len(attempt_name) != len("attempt_000")
        or not attempt_name[-3:].isdigit()
    ):
        raise CandidateRunnerContractError("invalid exact attempt name")
    root.mkdir(parents=True, exist_ok=True)
    attempt = root / attempt_name
    try:
        attempt.mkdir()
    except FileExistsError as error:
        raise CandidateRunnerContractError(
            f"append-only attempt exists: {attempt}"
        ) from error
    return attempt


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
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
        raise CandidateRunnerContractError(
            f"append-only artifact exists: {path}"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def preserved_attempt_tree_sha256(
    *,
    repository_root: Path,
    candidate_root: Path,
    model_id: str,
    native_trajectory_id: str,
    selection_seed: int,
) -> str:
    repository_root = repository_root.resolve()
    candidate_root = candidate_root.resolve()
    preserved_roots = (
        candidate_root / "_workers" / model_id / "ownership.lock",
        candidate_root / "_workers" / model_id / "attempt_001",
        candidate_root
        / "_trajectories"
        / model_id
        / native_trajectory_id
        / f"seed_{selection_seed}"
        / "attempt_001",
    )
    files: set[Path] = set()
    for root in preserved_roots:
        if root.is_file():
            files.add(root)
        elif root.is_dir():
            files.update(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise CandidateRunnerContractError(
            f"no attempt_001 preservation files for {model_id}"
        )
    digest = hashlib.sha256()
    for path in sorted(
        files,
        key=lambda item: item.relative_to(repository_root).as_posix(),
    ):
        relative = path.relative_to(repository_root).as_posix()
        record = f"{sha256_file(path)}  {relative}\n"
        digest.update(record.encode())
    return digest.hexdigest()


def claim_model_worker(
    *,
    candidate_root: Path,
    model_id: str,
    selection_seed: int,
    trajectory_ids: Sequence[str],
    candidate_ids: Sequence[str],
    provenance: Mapping[str, Any],
) -> Path:
    if model_id not in MODEL_IDS:
        raise CandidateRunnerContractError(
            f"unknown v2.6 candidate worker model: {model_id}"
        )
    if len(tuple(trajectory_ids)) != 3 or len(tuple(candidate_ids)) != 4:
        raise CandidateRunnerContractError(
            "model worker must own exactly three trajectories and four "
            "candidates"
        )
    worker_root = (
        candidate_root.resolve() / "_workers" / model_id
    )
    worker_root.mkdir(parents=True, exist_ok=True)
    ownership = {
        "schema_version": WORKER_SCHEMA,
        "model_id": model_id,
        "selection_seed": int(selection_seed),
        "trajectory_ids": list(trajectory_ids),
        "candidate_ids": list(candidate_ids),
        "provenance": dict(provenance),
    }
    ownership_path = worker_root / "ownership.lock"
    try:
        exclusive_json(ownership_path, ownership)
    except CandidateRunnerContractError as error:
        try:
            existing = json.loads(
                ownership_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            existing = None
        qualifier = (
            "same provenance"
            if existing == ownership
            else "different or unreadable provenance"
        )
        raise CandidateRunnerContractError(
            f"model worker already owned ({qualifier}): {model_id}"
        ) from error
    attempt = worker_root / "attempt_001"
    try:
        attempt.mkdir()
    except FileExistsError as error:
        raise CandidateRunnerContractError(
            f"model worker attempt already exists: {attempt}"
        ) from error
    manifest_path = attempt / "worker_manifest.json"
    exclusive_json(
        manifest_path,
        {
            **ownership,
            "status": "CLAIMED",
            "ownership_lock_path": str(ownership_path),
            "ownership_lock_sha256": sha256_file(ownership_path),
        },
    )
    exclusive_json(
        attempt / "RUNNING.json",
        {
            "schema_version": WORKER_SCHEMA,
            "status": "RUNNING",
            "model_id": model_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "worker_manifest_sha256": sha256_file(manifest_path),
        },
    )
    return attempt


def claim_model_continuation(
    *,
    repository_root: Path,
    candidate_root: Path,
    plan: CandidateContinuationPlan,
    authorization_sha256: str,
    prior_ownership_lock_sha256: str,
    attempt_001_preservation_sha256: str,
    provenance: Mapping[str, Any],
) -> Path:
    repository_root = repository_root.resolve()
    candidate_root = candidate_root.resolve()
    native_ids = {
        candidate.shared_trajectory_id
        for candidate in plan.evaluation_only_candidates
    }
    if len(native_ids) != 1:
        raise CandidateRunnerContractError(
            "continuation claim has no unique native trajectory"
        )
    worker_root = candidate_root / "_workers" / plan.model_id
    ownership_path = worker_root / "ownership.lock"
    prior_terminal = (
        worker_root / "attempt_001" / "WORKER_FAILED.json"
    )
    if (
        not ownership_path.is_file()
        or sha256_file(ownership_path) != prior_ownership_lock_sha256
        or not prior_terminal.is_file()
    ):
        raise CandidateRunnerContractError(
            "continuation prior ownership/terminal contract mismatch"
        )
    current_preservation = preserved_attempt_tree_sha256(
        repository_root=repository_root,
        candidate_root=candidate_root,
        model_id=plan.model_id,
        native_trajectory_id=next(iter(native_ids)),
        selection_seed=plan.selection_seed,
    )
    if current_preservation != attempt_001_preservation_sha256:
        raise CandidateRunnerContractError(
            "continuation attempt_001 preservation hash mismatch"
        )
    attempt = worker_root / "attempt_002"
    try:
        attempt.mkdir()
    except FileExistsError as error:
        raise CandidateRunnerContractError(
            f"continuation worker attempt_002 already exists: "
            f"{plan.model_id}"
        ) from error
    manifest_path = attempt / "worker_manifest.json"
    manifest = {
        "schema_version": CONTINUATION_WORKER_SCHEMA,
        "status": "CLAIMED",
        "model_id": plan.model_id,
        "selection_seed": plan.selection_seed,
        "worker_attempt": "attempt_002",
        "authorization_sha256": authorization_sha256,
        "prior_ownership_lock_path": _relative(
            repository_root,
            ownership_path,
        ),
        "prior_ownership_lock_sha256": prior_ownership_lock_sha256,
        "prior_worker_terminal_path": _relative(
            repository_root,
            prior_terminal,
        ),
        "prior_worker_terminal_sha256": sha256_file(prior_terminal),
        "attempt_001_preservation_sha256": current_preservation,
        "evaluation_only_candidate_ids": [
            candidate.candidate_id
            for candidate in plan.evaluation_only_candidates
        ],
        "training_trajectory_ids": [
            trajectory.shared_trajectory_id
            for trajectory in plan.training_trajectories
        ],
        "native_trajectory_retrained": False,
        "provenance": dict(provenance),
    }
    exclusive_json(manifest_path, manifest)
    exclusive_json(
        attempt / "RUNNING.json",
        {
            "schema_version": CONTINUATION_WORKER_SCHEMA,
            "status": "RUNNING",
            "model_id": plan.model_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "worker_manifest_sha256": sha256_file(manifest_path),
        },
    )
    return attempt


def _read_json_mapping(path: Path, *, role: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateRunnerContractError(
            f"cannot read {role}: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise CandidateRunnerContractError(
            f"{role} is not a JSON mapping: {path}"
        )
    return value


def _authorized_existing_path(
    *,
    repository_root: Path,
    relative_path: Any,
    role: str,
) -> Path:
    value = Path(str(relative_path))
    if value.is_absolute():
        raise CandidateRunnerContractError(
            f"{role} must be repository-relative"
        )
    path = validate_candidate_io_path(
        repository_root=repository_root,
        path=repository_root / value,
        role=role,
        access="read",
    )
    if not path.is_file():
        raise CandidateRunnerContractError(f"missing {role}: {path}")
    return path


def _model_scope(
    plan: CandidateTrajectoryPlan,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    model_ids = {trajectory.model_id for trajectory in plan.trajectories}
    if len(model_ids) != 1:
        raise CandidateRunnerContractError(
            "model worker plan must contain exactly one model"
        )
    model_id = next(iter(model_ids))
    trajectory_ids = tuple(
        trajectory.shared_trajectory_id
        for trajectory in plan.trajectories
    )
    candidate_ids = tuple(
        candidate.candidate_id
        for trajectory in plan.trajectories
        for candidate in trajectory.candidates
    )
    if len(trajectory_ids) != 3 or len(candidate_ids) != 4:
        raise CandidateRunnerContractError(
            "model worker plan resource accounting mismatch"
        )
    return model_id, trajectory_ids, candidate_ids


def finalize_model_worker(
    *,
    candidate_root: Path,
    worker_attempt: Path,
    plan: CandidateTrajectoryPlan,
    provenance: Mapping[str, Any],
    trajectory_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    model_id, trajectory_ids, candidate_ids = _model_scope(plan)
    failures = [
        dict(result)
        for result in trajectory_results
        if result.get("status") != "COMPLETE"
    ]
    if len(trajectory_results) != 3 or failures:
        value = {
            "schema_version": WORKER_SCHEMA,
            "status": "FAILED",
            "model_id": model_id,
            "failure_class": (
                "trajectory_failure"
                if failures
                else "incomplete_worker"
            ),
            "completed_trajectory_count": sum(
                result.get("status") == "COMPLETE"
                for result in trajectory_results
            ),
            "expected_trajectory_count": 3,
            "trajectory_results": [
                dict(result) for result in trajectory_results
            ],
            "provenance": dict(provenance),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        exclusive_json(worker_attempt / "WORKER_FAILED.json", value)
        return value

    candidate_records: dict[str, Mapping[str, str]] = {}
    for candidate_id in candidate_ids:
        attempt = (
            candidate_root
            / model_id
            / candidate_id
            / f"seed_{plan.selection_seed}"
            / "attempt_001"
        )
        result_path = attempt / "candidate_result.json"
        complete_path = attempt / "COMPLETE.json"
        if not result_path.is_file() or not complete_path.is_file():
            value = {
                "schema_version": WORKER_SCHEMA,
                "status": "FAILED",
                "model_id": model_id,
                "failure_class": "missing_candidate_artifact",
                "missing_candidate_id": candidate_id,
                "provenance": dict(provenance),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            exclusive_json(worker_attempt / "WORKER_FAILED.json", value)
            return value
        candidate_records[candidate_id] = {
            "candidate_result_path": str(
                result_path.resolve().relative_to(
                    candidate_root.resolve()
                )
            ),
            "candidate_result_sha256": sha256_file(result_path),
            "complete_path": str(
                complete_path.resolve().relative_to(
                    candidate_root.resolve()
                )
            ),
            "complete_sha256": sha256_file(complete_path),
        }
    value = {
        "schema_version": WORKER_SCHEMA,
        "status": "COMPLETE",
        "model_id": model_id,
        "selection_seed": plan.selection_seed,
        "trajectory_ids": list(trajectory_ids),
        "candidate_ids": list(candidate_ids),
        "trajectory_count": 3,
        "candidate_count": 4,
        "provenance": dict(provenance),
        "worker_manifest_sha256": sha256_file(
            worker_attempt / "worker_manifest.json"
        ),
        "trajectory_results": [
            dict(result) for result in trajectory_results
        ],
        "candidate_artifacts": candidate_records,
        "selection_aggregate_created": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    exclusive_json(worker_attempt / "WORKER_COMPLETE.json", value)
    return value


def finalize_model_continuation(
    *,
    candidate_root: Path,
    worker_attempt: Path,
    plan: CandidateContinuationPlan,
    provenance: Mapping[str, Any],
    operation_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    failures = [
        dict(result)
        for result in operation_results
        if result.get("status") != "COMPLETE"
    ]
    if len(operation_results) != 3 or failures:
        value = {
            "schema_version": CONTINUATION_WORKER_SCHEMA,
            "status": "FAILED",
            "model_id": plan.model_id,
            "failure_class": (
                "continuation_operation_failure"
                if failures
                else "incomplete_continuation"
            ),
            "completed_operation_count": sum(
                result.get("status") == "COMPLETE"
                for result in operation_results
            ),
            "expected_operation_count": 3,
            "operation_results": [
                dict(result) for result in operation_results
            ],
            "native_trajectory_retrained": False,
            "provenance": dict(provenance),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        exclusive_json(worker_attempt / "WORKER_FAILED.json", value)
        return value
    original_trajectory_ids = {
        candidate.shared_trajectory_id
        for candidate in plan.evaluation_only_candidates
    }
    original_trajectory_ids.update(
        trajectory.shared_trajectory_id
        for trajectory in plan.training_trajectories
    )
    candidate_ids = [
        candidate.candidate_id
        for candidate in plan.evaluation_only_candidates
    ]
    candidate_ids.extend(
        candidate.candidate_id
        for trajectory in plan.training_trajectories
        for candidate in trajectory.candidates
    )
    records: dict[str, Mapping[str, str]] = {}
    for candidate_id in candidate_ids:
        attempt = (
            candidate_root
            / plan.model_id
            / candidate_id
            / f"seed_{plan.selection_seed}"
            / "attempt_001"
        )
        result_path = attempt / "candidate_result.json"
        complete_path = attempt / "COMPLETE.json"
        if not result_path.is_file() or not complete_path.is_file():
            value = {
                "schema_version": CONTINUATION_WORKER_SCHEMA,
                "status": "FAILED",
                "model_id": plan.model_id,
                "failure_class": "missing_candidate_artifact",
                "missing_candidate_id": candidate_id,
                "native_trajectory_retrained": False,
                "provenance": dict(provenance),
                "finished_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
            exclusive_json(worker_attempt / "WORKER_FAILED.json", value)
            return value
        records[candidate_id] = {
            "candidate_result_path": str(
                result_path.resolve().relative_to(
                    candidate_root.resolve()
                )
            ),
            "candidate_result_sha256": sha256_file(result_path),
            "complete_path": str(
                complete_path.resolve().relative_to(
                    candidate_root.resolve()
                )
            ),
            "complete_sha256": sha256_file(complete_path),
        }
    value = {
        "schema_version": CONTINUATION_WORKER_SCHEMA,
        "status": "COMPLETE",
        "model_id": plan.model_id,
        "selection_seed": plan.selection_seed,
        "worker_attempt": "attempt_002",
        "trajectory_ids": sorted(original_trajectory_ids),
        "candidate_ids": candidate_ids,
        "trajectory_count": 3,
        "trained_trajectory_count": 2,
        "reused_native_trajectory_count": 1,
        "candidate_count": 4,
        "native_trajectory_retrained": False,
        "provenance": dict(provenance),
        "worker_manifest_sha256": sha256_file(
            worker_attempt / "worker_manifest.json"
        ),
        "operation_results": [
            dict(result) for result in operation_results
        ],
        "candidate_artifacts": records,
        "selection_aggregate_created": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    exclusive_json(worker_attempt / "WORKER_COMPLETE.json", value)
    return value


def validate_candidate_aggregate_readiness(
    *,
    candidate_root: Path,
    config: Mapping[str, Any],
    source_commit: str,
    relevant_source_sha256: str,
    selection_config_sha256: str,
    development_manifest_sha256: str,
    selection_plan_sha256: str,
    approved_worker_terminals: Mapping[
        str,
        Mapping[str, Any],
    ]
    | None = None,
    unavailable_model_ids: Sequence[str] = (),
) -> Mapping[str, Any]:
    candidate_root = candidate_root.resolve()
    unavailable = tuple(unavailable_model_ids)
    if (
        len(set(unavailable)) != len(unavailable)
        or not set(unavailable) <= set(SECONDARY_MODEL_IDS)
    ):
        raise CandidateRunnerContractError(
            "only preregistered secondary models may be unavailable"
        )
    required_model_ids = tuple(
        model_id
        for model_id in MODEL_IDS
        if model_id not in set(unavailable)
    )
    common_provenance = {
        "selection_config_sha256": selection_config_sha256,
        "development_manifest_sha256": development_manifest_sha256,
        "selection_plan_sha256": selection_plan_sha256,
    }
    if (
        approved_worker_terminals is not None
        and set(approved_worker_terminals) != set(required_model_ids)
    ):
        raise CandidateRunnerContractError(
            "aggregate authorization worker scope mismatch"
        )
    workers: dict[str, Mapping[str, Any]] = {}
    for model_id in required_model_ids:
        plan = build_model_trajectory_plan(config, model_id=model_id)
        _, trajectory_ids, candidate_ids = _model_scope(plan)
        approved = (
            approved_worker_terminals.get(model_id)
            if approved_worker_terminals is not None
            else None
        )
        if approved is not None:
            attempt_name = str(approved.get("attempt", ""))
            terminal_schema = str(approved.get("schema_version", ""))
            expected_source_commit = str(
                approved.get("source_commit", "")
            )
            expected_source_hash = str(
                approved.get("relevant_source_sha256", "")
            )
            if (
                attempt_name not in {"attempt_001", "attempt_002"}
                or terminal_schema
                not in {WORKER_SCHEMA, CONTINUATION_WORKER_SCHEMA}
                or len(expected_source_commit) != 40
                or len(expected_source_hash) != 64
            ):
                raise CandidateRunnerContractError(
                    f"approved worker terminal is invalid: {model_id}"
                )
        else:
            attempt_name = "attempt_001"
            terminal_schema = WORKER_SCHEMA
            expected_source_commit = source_commit
            expected_source_hash = relevant_source_sha256
        terminal_path = (
            candidate_root
            / "_workers"
            / model_id
            / attempt_name
            / "WORKER_COMPLETE.json"
        )
        if not terminal_path.is_file():
            raise CandidateRunnerContractError(
                "all four model workers must be COMPLETE before aggregate: "
                f"missing {model_id}"
            )
        terminal = _read_json_mapping(
            terminal_path,
            role="model worker completion",
        )
        expected = {
            "schema_version": terminal_schema,
            "status": "COMPLETE",
            "model_id": model_id,
            "selection_seed": plan.selection_seed,
            "trajectory_ids": list(trajectory_ids),
            "candidate_ids": list(candidate_ids),
            "trajectory_count": 3,
            "candidate_count": 4,
            "selection_aggregate_created": False,
        }
        mismatches = {
            key: {"expected": value, "actual": terminal.get(key)}
            for key, value in expected.items()
            if terminal.get(key) != value
        }
        if mismatches:
            raise CandidateRunnerContractError(
                f"model worker terminal mismatch for {model_id}: "
                f"{mismatches}"
            )
        if (
            terminal_schema == CONTINUATION_WORKER_SCHEMA
            and terminal.get("native_trajectory_retrained") is not False
        ):
            raise CandidateRunnerContractError(
                f"continuation retrained native trajectory: {model_id}"
            )
        if (
            approved is not None
            and sha256_file(terminal_path)
            != approved.get("terminal_sha256")
        ):
            raise CandidateRunnerContractError(
                f"approved worker terminal hash mismatch: {model_id}"
            )
        expected_provenance = {
            **common_provenance,
            "source_commit": expected_source_commit,
            "relevant_source_sha256": expected_source_hash,
        }
        terminal_provenance = terminal.get("provenance")
        if (
            not isinstance(terminal_provenance, Mapping)
            or any(
                terminal_provenance.get(key) != value
                for key, value in expected_provenance.items()
            )
        ):
            raise CandidateRunnerContractError(
                f"model worker provenance mismatch: {model_id}"
            )
        records = terminal.get("candidate_artifacts")
        if (
            not isinstance(records, Mapping)
            or set(records) != set(candidate_ids)
        ):
            raise CandidateRunnerContractError(
                f"model worker candidate index mismatch: {model_id}"
            )
        for candidate_id, record in records.items():
            if not isinstance(record, Mapping):
                raise CandidateRunnerContractError(
                    f"candidate index is invalid: {model_id}/{candidate_id}"
                )
            for path_key, hash_key in (
                ("candidate_result_path", "candidate_result_sha256"),
                ("complete_path", "complete_sha256"),
            ):
                path = (
                    candidate_root / str(record.get(path_key, ""))
                ).resolve()
                if (
                    candidate_root not in path.parents
                    or not path.is_file()
                    or sha256_file(path) != record.get(hash_key)
                ):
                    raise CandidateRunnerContractError(
                        "candidate aggregate artifact mismatch: "
                        f"{model_id}/{candidate_id}/{path_key}"
                    )
        workers[model_id] = {
            "terminal_path": str(terminal_path),
            "terminal_sha256": sha256_file(terminal_path),
            "attempt": attempt_name,
            "schema_version": terminal_schema,
            "source_commit": expected_source_commit,
            "relevant_source_sha256": expected_source_hash,
            "candidate_count": 4,
            "trajectory_count": 3,
        }
    attempts = {worker["attempt"] for worker in workers.values()}
    source_states = {
        (
            worker["source_commit"],
            worker["relevant_source_sha256"],
        )
        for worker in workers.values()
    }
    return {
        "schema_version": WORKER_SCHEMA,
        "status": "READY_FOR_VALIDATION_AGGREGATE",
        "worker_count": len(required_model_ids),
        "trajectory_count": 3 * len(required_model_ids),
        "candidate_count": 4 * len(required_model_ids),
        "workers": workers,
        "unavailable_model_ids": list(unavailable),
        "mixed_worker_attempts": len(attempts) > 1,
        "mixed_source_provenance": len(source_states) > 1,
        "test_split_read": False,
    }


def _bounded_child(
    target: Callable[..., None],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    result_queue,
) -> None:
    try:
        target(*args, **kwargs)
    except (KeyboardInterrupt, SystemExit) as error:
        result_queue.put(
            {
                "kind": "interrupted",
                "exception": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    except BaseException as error:
        if isinstance(
            error,
            (
                CandidateAdapterContractError,
                InvalidCandidateError,
            ),
        ):
            failure_class = "invalid_candidate"
        elif isinstance(error, CandidateRunnerContractError):
            failure_class = "contract"
        else:
            failure_class = "infrastructure_or_code"
        result_queue.put(
            {
                "kind": "exception",
                "failure_class": failure_class,
                "exception": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    else:
        result_queue.put({"kind": "complete"})


def run_bounded_process(
    *,
    target: Callable[..., None],
    attempt_path: Path,
    max_wall_seconds: float,
    termination_grace_seconds: float,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    on_target_complete: Callable[[Mapping[str, Any]], None] | None = None,
    target_complete_cleanup_grace_seconds: float | None = None,
) -> Mapping[str, Any]:
    if max_wall_seconds <= 0 or termination_grace_seconds <= 0:
        raise ValueError("wall cap and termination grace must be positive")
    if (on_target_complete is None) != (
        target_complete_cleanup_grace_seconds is None
    ):
        raise ValueError(
            "target-complete callback and cleanup grace must be paired"
        )
    if (
        target_complete_cleanup_grace_seconds is not None
        and target_complete_cleanup_grace_seconds <= 0
    ):
        raise ValueError(
            "target-complete cleanup grace must be positive"
        )
    attempt_path = attempt_path.resolve()
    context = multiprocessing.get_context("fork")
    result_queue = context.Queue()
    process = context.Process(
        target=_bounded_child,
        args=(target, tuple(args), dict(kwargs or {}), result_queue),
    )
    started = time.monotonic()
    process.start()
    if on_target_complete is not None:
        deadline = started + max_wall_seconds
        child = None
        while child is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                child = result_queue.get(
                    timeout=min(0.1, remaining)
                )
            except queue.Empty:
                if not process.is_alive():
                    try:
                        child = result_queue.get_nowait()
                    except queue.Empty:
                        break
        if child is None:
            if process.is_alive():
                process.terminate()
                process.join(termination_grace_seconds)
                if process.is_alive():
                    process.kill()
                    process.join(termination_grace_seconds)
                failure_class = "wall_cap"
                exception = None
            else:
                process.join()
                failure_class = "infrastructure_or_code"
                exception = (
                    f"child exited {process.exitcode} without terminal "
                    "metadata"
                )
            value = {
                "schema_version": RUNNER_SCHEMA,
                "status": "FAILED",
                "failure_class": failure_class,
                "actual_elapsed_seconds": (
                    time.monotonic() - started
                ),
                "child_pid": process.pid,
                "exception": exception,
                "traceback": "",
                "last_checkpoint": None,
            }
            exclusive_json(attempt_path / "FAILED.json", value)
            return value
        if child["kind"] != "complete":
            process.join(termination_grace_seconds)
            if process.is_alive():
                process.terminate()
                process.join(termination_grace_seconds)
            if process.is_alive():
                process.kill()
                process.join(termination_grace_seconds)
            failure_class = (
                "interrupted"
                if child["kind"] == "interrupted"
                else child.get(
                    "failure_class",
                    "infrastructure_or_code",
                )
            )
            value = {
                "schema_version": RUNNER_SCHEMA,
                "status": "FAILED",
                "failure_class": failure_class,
                "actual_elapsed_seconds": (
                    time.monotonic() - started
                ),
                "child_pid": process.pid,
                "exception": child.get("exception"),
                "traceback": child.get("traceback"),
                "last_checkpoint": None,
            }
            exclusive_json(attempt_path / "FAILED.json", value)
            return value
        target_elapsed = time.monotonic() - started
        provisional = {
            "schema_version": RUNNER_SCHEMA,
            "status": "COMPLETE",
            "failure_class": None,
            "actual_elapsed_seconds": target_elapsed,
            "child_pid": process.pid,
            "target_completed": True,
        }
        try:
            on_target_complete(provisional)
        except BaseException as error:
            if process.is_alive():
                process.terminate()
                process.join(termination_grace_seconds)
            if process.is_alive():
                process.kill()
                process.join(termination_grace_seconds)
            value = {
                "schema_version": RUNNER_SCHEMA,
                "status": "FAILED",
                "failure_class": "terminal_callback",
                "actual_elapsed_seconds": (
                    time.monotonic() - started
                ),
                "child_pid": process.pid,
                "exception": repr(error),
                "traceback": traceback.format_exc(),
                "last_checkpoint": None,
            }
            exclusive_json(attempt_path / "FAILED.json", value)
            return value
        process.join(target_complete_cleanup_grace_seconds)
        cleanup_forced = process.is_alive()
        if cleanup_forced:
            process.terminate()
            process.join(termination_grace_seconds)
            if process.is_alive():
                process.kill()
                process.join(termination_grace_seconds)
        value = {
            **provisional,
            "cleanup_forced": cleanup_forced,
            "cleanup_exitcode": process.exitcode,
            "cleanup_elapsed_seconds": (
                time.monotonic() - started - target_elapsed
            ),
        }
        exclusive_json(attempt_path / "COMPLETE.json", value)
        return value
    process.join(max_wall_seconds)
    elapsed = time.monotonic() - started
    if process.is_alive():
        process.terminate()
        process.join(termination_grace_seconds)
        if process.is_alive():
            process.kill()
            process.join(termination_grace_seconds)
        value = {
            "schema_version": RUNNER_SCHEMA,
            "status": "FAILED",
            "failure_class": "wall_cap",
            "actual_elapsed_seconds": time.monotonic() - started,
            "child_pid": process.pid,
            "last_checkpoint": None,
        }
        exclusive_json(attempt_path / "FAILED.json", value)
        return value
    try:
        child = result_queue.get_nowait()
    except queue.Empty:
        child = {
            "kind": "exception",
            "exception": (
                f"child exited {process.exitcode} without terminal metadata"
            ),
            "traceback": "",
        }
    elapsed = time.monotonic() - started
    if child["kind"] == "complete" and process.exitcode == 0:
        value = {
            "schema_version": RUNNER_SCHEMA,
            "status": "COMPLETE",
            "failure_class": None,
            "actual_elapsed_seconds": elapsed,
            "child_pid": process.pid,
        }
        exclusive_json(attempt_path / "COMPLETE.json", value)
        return value
    failure_class = (
        "interrupted"
        if child["kind"] == "interrupted"
        else child.get("failure_class", "infrastructure_or_code")
    )
    value = {
        "schema_version": RUNNER_SCHEMA,
        "status": "FAILED",
        "failure_class": failure_class,
        "actual_elapsed_seconds": elapsed,
        "child_pid": process.pid,
        "exception": child.get("exception"),
        "traceback": child.get("traceback"),
        "last_checkpoint": None,
    }
    exclusive_json(attempt_path / "FAILED.json", value)
    return value


def build_candidate_result_manifest(
    *,
    model_id: str,
    candidate_id: str,
    shared_trajectory_id: str,
    selection_seed: int,
    requested_updates: int,
    actual_updates: int,
    sampled_checkpoint_step: int,
    source_commit: str,
    relevant_source_sha256: str,
    selection_config_sha256: str,
    development_manifest_sha256: str,
    train_file_sha256: str,
    train_content_sha256: str,
    validation_file_sha256: str,
    validation_content_sha256: str,
    selection_plan_sha256: str,
    trajectory_manifest_path: str,
    trajectory_manifest_sha256: str,
    trajectory_complete_path: str,
    trajectory_complete_sha256: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    validation_sample_path: str,
    validation_sample_sha256: str,
    candidate_definition_sha256: str,
    actual_wall_seconds: float,
    training_source_commit: str | None = None,
    training_relevant_source_sha256: str | None = None,
    continuation_authorization_sha256: str | None = None,
) -> Mapping[str, Any]:
    completed = actual_updates == requested_updates
    if (
        not completed
        or requested_updates != 20_000
        or sampled_checkpoint_step not in {10_000, 20_000}
    ):
        raise CandidateRunnerContractError(
            "incomplete trajectory cannot produce a candidate result"
        )
    value = {
        "schema_version": CANDIDATE_RESULT_SCHEMA,
        "status": "COMPLETE",
        "model_id": model_id,
        "candidate_id": candidate_id,
        "selection_seed": int(selection_seed),
        "shared_trajectory_id": shared_trajectory_id,
        "selection_config_sha256": selection_config_sha256,
        "candidate_definition_sha256": candidate_definition_sha256,
        "source_commit": source_commit,
        "relevant_source_sha256": relevant_source_sha256,
        "development_manifest_sha256": development_manifest_sha256,
        "train_file_sha256": train_file_sha256,
        "train_content_sha256": train_content_sha256,
        "validation_file_sha256": validation_file_sha256,
        "validation_content_sha256": validation_content_sha256,
        "selection_plan_sha256": selection_plan_sha256,
        "requested_updates": int(requested_updates),
        "actual_updates": int(actual_updates),
        "trajectory_completed_requested_updates": completed,
        "sampled_checkpoint_step": int(sampled_checkpoint_step),
        "wall_cap_reached": False,
        "actual_wall_seconds": float(actual_wall_seconds),
        "trajectory_manifest_path": trajectory_manifest_path,
        "trajectory_manifest_sha256": trajectory_manifest_sha256,
        "trajectory_complete_path": trajectory_complete_path,
        "trajectory_complete_sha256": trajectory_complete_sha256,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "validation_sample_path": validation_sample_path,
        "validation_sample_sha256": validation_sample_sha256,
    }
    if training_source_commit is not None:
        value.update(
            {
                "evaluation_source_commit": source_commit,
                "evaluation_relevant_source_sha256": (
                    relevant_source_sha256
                ),
                "training_source_commit": training_source_commit,
                "training_relevant_source_sha256": (
                    training_relevant_source_sha256
                ),
                "continuation_authorization_sha256": (
                    continuation_authorization_sha256
                ),
            }
        )
    return value


RELEVANT_SOURCE_PATHS = (
    "eval/validation_selection_v2_6.py",
    "experiments/candidate_runner_v2_6.py",
    "generators/candidate_adapters_v2_6.py",
    "generators/candidate_model_backends_v2_6.py",
    "models/candidate_components_v2_6.py",
    "scripts/aggregate_candidate_selection_v2_6.py",
    "scripts/run_candidate_training_v2_6.py",
    "scripts/select_validation_candidates_v2_6.py",
)


def relevant_source_sha256(repository_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in RELEVANT_SOURCE_PATHS:
        path = repository_root / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def current_git_head(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise CandidateRunnerContractError("cannot resolve source commit")
    return value


def _exclusive_torch_save(path: Path, value: Any) -> None:
    try:
        with path.open("xb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise CandidateRunnerContractError(
            f"append-only checkpoint exists: {path}"
        ) from error


def _exclusive_npz(path: Path, batch) -> None:
    try:
        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                **{
                    field: getattr(batch, field)
                    for field in batch.__dataclass_fields__
                },
            )
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise CandidateRunnerContractError(
            f"append-only sample exists: {path}"
        ) from error


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    with os.fdopen(descriptor, "ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _relative(repository_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repository_root.resolve()))


def _actual_backend_updates(backend: Any) -> tuple[int, bool, float]:
    if hasattr(backend, "budget_allocation"):
        actual = backend.budget_allocation["actual"]
        updates = sum(
            int(value["actual_steps"]) for value in actual.values()
        )
        wall_reached = any(
            bool(value["wall_cap_reached"]) for value in actual.values()
        )
        elapsed = sum(
            float(value["actual_wall_seconds"])
            for value in actual.values()
        )
        return updates, wall_reached, elapsed
    budget = backend.actual_training_budget
    return (
        int(budget["actual_steps"]),
        bool(budget["wall_cap_reached"]),
        float(budget["actual_wall_seconds"]),
    )


def _save_balanced_class_checkpoint(
    *,
    checkpoints: Path,
    backend: Any,
    event: Mapping[str, Any],
    candidate_steps: set[int],
    train_only_zscore: TrainOnlyZScore | None,
    shared_trajectory_id: str,
    selection_plan_sha256: str,
) -> None:
    label = int(event["class_label"])
    class_step = int(event["class_step"])
    relevant_class_steps = {
        total_step // 2 for total_step in candidate_steps
    }
    if class_step not in relevant_class_steps:
        return
    class_path = (
        checkpoints
        / f"class_{label}_step_{class_step:05d}.pt"
    )
    _exclusive_torch_save(class_path, backend)
    for total_step in sorted(candidate_steps):
        per_class = total_step // 2
        class_zero = (
            checkpoints / f"class_0_step_{per_class:05d}.pt"
        )
        class_one = (
            checkpoints / f"class_1_step_{per_class:05d}.pt"
        )
        combined_path = checkpoints / f"step_{total_step:05d}.pt"
        if (
            class_zero.is_file()
            and class_one.is_file()
            and not combined_path.exists()
        ):
            zero_backend = torch.load(
                class_zero,
                map_location="cpu",
                weights_only=False,
            )
            combined = torch.load(
                class_one,
                map_location="cpu",
                weights_only=False,
            )
            combined.models[0] = zero_backend.models[0]
            combined.training_data_hashes_by_class[0] = (
                zero_backend.training_data_hashes_by_class[0]
            )
            if (
                hasattr(zero_backend, "actual_training_budget")
                and 0 in zero_backend.actual_training_budget
            ):
                combined.actual_training_budget[0] = (
                    zero_backend.actual_training_budget[0]
                )
            _exclusive_torch_save(
                combined_path,
                build_checkpoint_bundle(
                    backend=combined,
                    train_only_zscore=train_only_zscore,
                    shared_trajectory_id=shared_trajectory_id,
                    sampled_checkpoint_step=total_step,
                    selection_plan_sha256=selection_plan_sha256,
                ),
            )


def build_checkpoint_bundle(
    *,
    backend: Any,
    train_only_zscore: TrainOnlyZScore | None,
    shared_trajectory_id: str,
    sampled_checkpoint_step: int,
    selection_plan_sha256: str,
) -> Mapping[str, Any]:
    return {
        "schema_version": "benchmark-v2.6-candidate-checkpoint-v1",
        "backend": backend,
        "shared_trajectory_id": shared_trajectory_id,
        "sampled_checkpoint_step": int(sampled_checkpoint_step),
        "selection_plan_sha256": selection_plan_sha256,
        "train_only_zscore": train_only_zscore,
        "train_only_zscore_state": (
            train_only_zscore.checkpoint_state()
            if train_only_zscore is not None
            else None
        ),
    }


def load_candidate_checkpoint_bundle(
    checkpoint_path: Path,
    *,
    model_id: str,
    device: str,
) -> Mapping[str, Any]:
    if model_id in {
        "ctgan_separate_class",
        "tvae_separate_class",
    }:
        import ctgan.synthesizers.base as ctgan_base

        original_set_device = ctgan_base._set_device
        ctgan_base._set_device = lambda enable_gpu: torch.device("cpu")
        try:
            bundle = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        finally:
            ctgan_base._set_device = original_set_device
        if not isinstance(bundle, Mapping):
            raise CandidateRunnerContractError(
                "candidate checkpoint bundle is not a mapping"
            )
        backend = bundle.get("backend")
        models = getattr(backend, "models", None)
        if not isinstance(models, Mapping) or set(models) != {0, 1}:
            raise CandidateRunnerContractError(
                "tabular candidate checkpoint has no class-pair models"
            )
        for synthesizer in models.values():
            synthesizer.set_device(device)
        return bundle
    return torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )


def restore_candidate_checkpoint_for_sampling(
    checkpoint_path: Path,
    *,
    model_id: str,
    device: str,
    train: SequenceBatch,
) -> Mapping[str, Any]:
    bundle = load_candidate_checkpoint_bundle(
        checkpoint_path,
        model_id=model_id,
        device=device,
    )
    if model_id == "ctgan_separate_class":
        backend = bundle.get("backend")
        prepare = getattr(backend, "prepare_sampling_from_train", None)
        if not callable(prepare):
            raise CandidateRunnerContractError(
                "CTGAN checkpoint lacks sampling-state restore support"
            )
        sampling_train = train
        transform = bundle.get("train_only_zscore")
        if transform is not None:
            if not isinstance(transform, TrainOnlyZScore):
                raise CandidateRunnerContractError(
                    "CTGAN checkpoint z-score state is invalid"
                )
            sampling_train = transform.transform_train(train)
        prepare(sampling_train)
    return bundle


def _execute_trajectory_child(payload: Mapping[str, Any]) -> None:
    repository_root = Path(payload["repository_root"]).resolve()
    attempt = Path(payload["attempt_path"]).resolve()
    candidate_root = Path(payload["candidate_root"]).resolve()
    config_path = Path(payload["config_path"]).resolve()
    development_path = Path(payload["development_manifest_path"]).resolve()
    device = str(payload["device"])
    if not device.startswith("cuda"):
        raise CandidateRunnerContractError(
            "learned candidate trajectories require an explicit CUDA device"
        )
    if not torch.cuda.is_available():
        raise CandidateRunnerContractError(
            "authorized candidate CUDA device is unavailable"
        )
    torch.cuda.set_device(torch.device(device))
    config = read_yaml_mapping(config_path)
    plan = build_trajectory_plan(config)
    trajectory = next(
        item
        for item in plan.trajectories
        if item.model_id == payload["model_id"]
        and item.shared_trajectory_id == payload["shared_trajectory_id"]
    )
    context = load_train_only_candidate_context(
        repository_root=repository_root,
        development_manifest_path=development_path,
    )
    source_hash = relevant_source_sha256(repository_root)
    if source_hash != payload["relevant_source_sha256"]:
        raise CandidateRunnerContractError(
            "candidate source changed after authorization"
        )
    config_hash = sha256_file(config_path)
    if config_hash != payload["selection_config_sha256"]:
        raise CandidateRunnerContractError(
            "candidate config changed after authorization"
        )
    checkpoints = attempt / "checkpoints"
    checkpoints.mkdir()
    progress_path = attempt / "progress.jsonl"
    trajectory_manifest = {
        "schema_version": TRAJECTORY_SCHEMA,
        "status": "RUNNING",
        "source_commit": payload["source_commit"],
        "relevant_source_sha256": source_hash,
        "selection_config_sha256": config_hash,
        "development_manifest_sha256": context.manifest_sha256,
        "train_file_sha256": context.train_file_sha256,
        "train_content_sha256": context.train_content_sha256,
        "validation_file_sha256": context.validation_file_sha256,
        "validation_content_sha256": context.validation_content_sha256,
        "selection_plan_sha256": context.plan.plan_hash,
        "model_id": trajectory.model_id,
        "shared_trajectory_id": trajectory.shared_trajectory_id,
        "selection_seed": plan.selection_seed,
        "candidate_ids": [
            item.candidate_id for item in trajectory.candidates
        ],
        "requested_updates": trajectory.requested_updates,
        "max_wall_seconds": trajectory.max_wall_seconds,
        "device": device,
        "test_split_read": False,
        "validation_labels_or_lengths_used_for_fit": False,
    }
    manifest_path = attempt / "manifest.json"
    exclusive_json(manifest_path, trajectory_manifest)
    trajectory_manifest_hash = sha256_file(manifest_path)
    fitting_spec = max(
        trajectory.candidates,
        key=lambda item: item.sampled_checkpoint_step,
    )
    adapter = build_candidate_adapter(fitting_spec)
    adapter.bind_train_only_sampling_plan(context.plan)
    candidate_steps = {
        item.sampled_checkpoint_step for item in trajectory.candidates
    }

    def progress(event):
        _append_jsonl(progress_path, dict(event))

    def checkpoint(event, backend):
        if trajectory.model_id in {
            "ctgan_separate_class",
            "tvae_separate_class",
        }:
            _save_balanced_class_checkpoint(
                checkpoints=checkpoints,
                backend=backend,
                event=event,
                candidate_steps=candidate_steps,
                train_only_zscore=adapter._zscore,
                shared_trajectory_id=trajectory.shared_trajectory_id,
                selection_plan_sha256=context.plan.plan_hash,
            )
            return
        step = int(event["step"])
        if step in candidate_steps:
            _exclusive_torch_save(
                checkpoints / f"step_{step:05d}.pt",
                build_checkpoint_bundle(
                    backend=backend,
                    train_only_zscore=adapter._zscore,
                    shared_trajectory_id=trajectory.shared_trajectory_id,
                    sampled_checkpoint_step=step,
                    selection_plan_sha256=context.plan.plan_hash,
                ),
            )

    base_config = dict(
        config["models"][trajectory.model_id]["adapter_base_config"]
    )
    base_config.pop("adapter", None)
    base_config["device"] = device
    if trajectory.model_id == "cof_seqgen":
        base_config["tau"] = context.tau.tolist()
    base_config["progress_callback"] = progress
    base_config["checkpoint_callback"] = checkpoint
    adapter.fit_train_only(
        context.train,
        base_config=base_config,
        seed=plan.selection_seed,
    )
    backend = adapter._backend
    if backend is None:
        raise CandidateRunnerContractError("candidate backend disappeared")
    actual_updates, wall_reached, training_seconds = (
        _actual_backend_updates(backend)
    )
    if wall_reached or actual_updates != trajectory.requested_updates:
        raise InvalidCandidateError(
            "trajectory did not finish requested updates within its cap"
        )
    checkpoint_records = {}
    for candidate in trajectory.candidates:
        checkpoint_path = (
            checkpoints
            / f"step_{candidate.sampled_checkpoint_step:05d}.pt"
        )
        if not checkpoint_path.is_file():
            raise CandidateRunnerContractError(
                f"missing candidate checkpoint: {checkpoint_path}"
            )
        checkpoint_records[candidate.candidate_id] = {
            "path": _relative(repository_root, checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "sampled_checkpoint_step": candidate.sampled_checkpoint_step,
        }
    trajectory_complete_path = attempt / "TRAJECTORY_COMPLETE.json"
    exclusive_json(
        trajectory_complete_path,
        {
            "schema_version": TRAJECTORY_SCHEMA,
            "status": "COMPLETE",
            "shared_trajectory_id": trajectory.shared_trajectory_id,
            "requested_updates": trajectory.requested_updates,
            "actual_updates": actual_updates,
            "completed_hard_cap_contract": True,
            "wall_cap_reached": False,
            "actual_training_seconds": training_seconds,
            "checkpoints": checkpoint_records,
        },
    )
    trajectory_complete_hash = sha256_file(trajectory_complete_path)
    for candidate in trajectory.candidates:
        checkpoint_path = (
            checkpoints
            / f"step_{candidate.sampled_checkpoint_step:05d}.pt"
        )
        if not checkpoint_path.is_file():
            raise CandidateRunnerContractError(
                f"missing candidate checkpoint: {checkpoint_path}"
            )
        checkpoint_bundle = restore_candidate_checkpoint_for_sampling(
            checkpoint_path,
            model_id=trajectory.model_id,
            device=device,
            train=context.train,
        )
        if (
            not isinstance(checkpoint_bundle, Mapping)
            or checkpoint_bundle.get("schema_version")
            != "benchmark-v2.6-candidate-checkpoint-v1"
            or checkpoint_bundle.get("shared_trajectory_id")
            != trajectory.shared_trajectory_id
            or int(
                checkpoint_bundle.get("sampled_checkpoint_step", -1)
            )
            != candidate.sampled_checkpoint_step
            or checkpoint_bundle.get("selection_plan_sha256")
            != context.plan.plan_hash
        ):
            raise CandidateRunnerContractError(
                "candidate checkpoint bundle provenance mismatch"
            )
        candidate_adapter = build_candidate_adapter(candidate)
        candidate_adapter.bind_train_only_sampling_plan(context.plan)
        candidate_adapter.attach_trained_backend_for_sampling(
            checkpoint_bundle["backend"],
            train_only_zscore=checkpoint_bundle[
                "train_only_zscore"
            ],
        )
        candidate_attempt = claim_next_attempt(
            candidate_root
            / trajectory.model_id
            / candidate.candidate_id
            / f"seed_{plan.selection_seed}"
        )
        sample_path = candidate_attempt / "validation_sample.npz"
        sample = candidate_adapter.sample_validation(
            context.plan,
            seed=plan.selection_seed,
        )
        _exclusive_npz(sample_path, sample)
        checkpoint_hash = sha256_file(checkpoint_path)
        result = build_candidate_result_manifest(
            model_id=trajectory.model_id,
            candidate_id=candidate.candidate_id,
            shared_trajectory_id=trajectory.shared_trajectory_id,
            selection_seed=plan.selection_seed,
            requested_updates=trajectory.requested_updates,
            actual_updates=actual_updates,
            sampled_checkpoint_step=candidate.sampled_checkpoint_step,
            source_commit=payload["source_commit"],
            relevant_source_sha256=source_hash,
            selection_config_sha256=config_hash,
            development_manifest_sha256=context.manifest_sha256,
            train_file_sha256=context.train_file_sha256,
            train_content_sha256=context.train_content_sha256,
            validation_file_sha256=context.validation_file_sha256,
            validation_content_sha256=context.validation_content_sha256,
            selection_plan_sha256=context.plan.plan_hash,
            trajectory_manifest_path=_relative(
                repository_root,
                manifest_path,
            ),
            trajectory_manifest_sha256=trajectory_manifest_hash,
            trajectory_complete_path=_relative(
                repository_root,
                trajectory_complete_path,
            ),
            trajectory_complete_sha256=trajectory_complete_hash,
            checkpoint_path=_relative(
                repository_root,
                checkpoint_path,
            ),
            checkpoint_sha256=checkpoint_hash,
            validation_sample_path=_relative(
                repository_root,
                sample_path,
            ),
            validation_sample_sha256=sha256_file(sample_path),
            candidate_definition_sha256=candidate.definition_sha256,
            actual_wall_seconds=training_seconds,
        )
        exclusive_json(
            candidate_attempt / "candidate_result.json",
            result,
        )
        exclusive_json(
            candidate_attempt / "COMPLETE.json",
            {
                "schema_version": CANDIDATE_RESULT_SCHEMA,
                "status": "COMPLETE",
                "candidate_result_sha256": sha256_file(
                    candidate_attempt / "candidate_result.json"
                ),
            },
        )


def _execute_native_evaluation_child(payload: Mapping[str, Any]) -> None:
    repository_root = Path(payload["repository_root"]).resolve()
    attempt = Path(payload["attempt_path"]).resolve()
    candidate_root = Path(payload["candidate_root"]).resolve()
    config_path = Path(payload["config_path"]).resolve()
    development_path = Path(payload["development_manifest_path"]).resolve()
    model_id = str(payload["model_id"])
    device = str(payload["device"])
    if model_id not in {
        "ctgan_separate_class",
        "tvae_separate_class",
    }:
        raise CandidateRunnerContractError(
            "native evaluation continuation is limited to CTGAN/TVAE"
        )
    if not device.startswith("cuda"):
        raise CandidateRunnerContractError(
            "learned candidate evaluation requires an explicit CUDA device"
        )
    if not torch.cuda.is_available():
        raise CandidateRunnerContractError(
            "authorized candidate CUDA device is unavailable"
        )
    torch.cuda.set_device(torch.device(device))
    config = read_yaml_mapping(config_path)
    plan = build_model_continuation_plan(config, model_id=model_id)
    context = load_train_only_candidate_context(
        repository_root=repository_root,
        development_manifest_path=development_path,
    )
    source_hash = relevant_source_sha256(repository_root)
    config_hash = sha256_file(config_path)
    if source_hash != payload["relevant_source_sha256"]:
        raise CandidateRunnerContractError(
            "candidate source changed after continuation authorization"
        )
    if config_hash != payload["selection_config_sha256"]:
        raise CandidateRunnerContractError(
            "candidate config changed after continuation authorization"
        )
    scope = payload.get("continuation_scope")
    if not isinstance(scope, Mapping):
        raise CandidateRunnerContractError(
            "native continuation scope is missing"
        )
    manifest_path = _authorized_existing_path(
        repository_root=repository_root,
        relative_path=scope["native_trajectory_manifest_path"],
        role="native trajectory manifest",
    )
    complete_path = _authorized_existing_path(
        repository_root=repository_root,
        relative_path=scope["native_trajectory_complete_path"],
        role="native trajectory completion",
    )
    if (
        sha256_file(manifest_path)
        != scope["native_trajectory_manifest_sha256"]
        or sha256_file(complete_path)
        != scope["native_trajectory_complete_sha256"]
    ):
        raise CandidateRunnerContractError(
            "native continuation trajectory artifact hash mismatch"
        )
    trajectory_complete = _read_json_mapping(
        complete_path,
        role="native trajectory completion",
    )
    if (
        trajectory_complete.get("status") != "COMPLETE"
        or int(trajectory_complete.get("actual_updates", -1)) != 20_000
        or trajectory_complete.get("completed_hard_cap_contract") is not True
        or trajectory_complete.get("wall_cap_reached") is not False
    ):
        raise CandidateRunnerContractError(
            "native continuation trajectory is not complete"
        )
    exclusive_json(
        attempt / "evaluation_manifest.json",
        {
            "schema_version": (
                "benchmark-v2.6-native-evaluation-continuation-v1"
            ),
            "status": "RUNNING",
            "model_id": model_id,
            "source_commit": payload["source_commit"],
            "relevant_source_sha256": source_hash,
            "native_training_source_commit": scope[
                "native_training_source_commit"
            ],
            "native_training_relevant_source_sha256": scope[
                "native_training_relevant_source_sha256"
            ],
            "selection_config_sha256": config_hash,
            "development_manifest_sha256": context.manifest_sha256,
            "selection_plan_sha256": context.plan.plan_hash,
            "authorization_sha256": payload["authorization_sha256"],
            "native_trajectory_retrained": False,
            "candidate_ids": [
                candidate.candidate_id
                for candidate in plan.evaluation_only_candidates
            ],
            "test_split_read": False,
            "device": device,
        },
    )
    checkpoints = scope.get("native_checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise CandidateRunnerContractError(
            "native continuation checkpoint index is missing"
        )
    for candidate in plan.evaluation_only_candidates:
        record = checkpoints.get(candidate.candidate_id)
        if not isinstance(record, Mapping):
            raise CandidateRunnerContractError(
                f"native continuation checkpoint missing: "
                f"{candidate.candidate_id}"
            )
        checkpoint_path = _authorized_existing_path(
            repository_root=repository_root,
            relative_path=record["path"],
            role=f"native checkpoint {candidate.candidate_id}",
        )
        if sha256_file(checkpoint_path) != record.get("sha256"):
            raise CandidateRunnerContractError(
                f"native continuation checkpoint hash mismatch: "
                f"{candidate.candidate_id}"
            )
        checkpoint_bundle = restore_candidate_checkpoint_for_sampling(
            checkpoint_path,
            model_id=model_id,
            device=device,
            train=context.train,
        )
        if (
            checkpoint_bundle.get("schema_version")
            != "benchmark-v2.6-candidate-checkpoint-v1"
            or checkpoint_bundle.get("shared_trajectory_id")
            != candidate.shared_trajectory_id
            or int(
                checkpoint_bundle.get("sampled_checkpoint_step", -1)
            )
            != candidate.sampled_checkpoint_step
            or checkpoint_bundle.get("selection_plan_sha256")
            != context.plan.plan_hash
        ):
            raise CandidateRunnerContractError(
                "native continuation checkpoint provenance mismatch"
            )
        adapter = build_candidate_adapter(candidate)
        adapter.bind_train_only_sampling_plan(context.plan)
        adapter.attach_trained_backend_for_sampling(
            checkpoint_bundle["backend"],
            train_only_zscore=checkpoint_bundle["train_only_zscore"],
        )
        candidate_attempt = claim_exact_attempt(
            candidate_root
            / model_id
            / candidate.candidate_id
            / f"seed_{plan.selection_seed}",
            attempt_name="attempt_001",
        )
        sample_path = candidate_attempt / "validation_sample.npz"
        sample = adapter.sample_validation(
            context.plan,
            seed=plan.selection_seed,
        )
        _exclusive_npz(sample_path, sample)
        result = build_candidate_result_manifest(
            model_id=model_id,
            candidate_id=candidate.candidate_id,
            shared_trajectory_id=candidate.shared_trajectory_id,
            selection_seed=plan.selection_seed,
            requested_updates=20_000,
            actual_updates=20_000,
            sampled_checkpoint_step=candidate.sampled_checkpoint_step,
            source_commit=payload["source_commit"],
            relevant_source_sha256=source_hash,
            selection_config_sha256=config_hash,
            development_manifest_sha256=context.manifest_sha256,
            train_file_sha256=context.train_file_sha256,
            train_content_sha256=context.train_content_sha256,
            validation_file_sha256=context.validation_file_sha256,
            validation_content_sha256=context.validation_content_sha256,
            selection_plan_sha256=context.plan.plan_hash,
            trajectory_manifest_path=_relative(
                repository_root,
                manifest_path,
            ),
            trajectory_manifest_sha256=sha256_file(manifest_path),
            trajectory_complete_path=_relative(
                repository_root,
                complete_path,
            ),
            trajectory_complete_sha256=sha256_file(complete_path),
            checkpoint_path=_relative(
                repository_root,
                checkpoint_path,
            ),
            checkpoint_sha256=sha256_file(checkpoint_path),
            validation_sample_path=_relative(
                repository_root,
                sample_path,
            ),
            validation_sample_sha256=sha256_file(sample_path),
            candidate_definition_sha256=candidate.definition_sha256,
            actual_wall_seconds=float(
                trajectory_complete["actual_training_seconds"]
            ),
            training_source_commit=scope[
                "native_training_source_commit"
            ],
            training_relevant_source_sha256=scope[
                "native_training_relevant_source_sha256"
            ],
            continuation_authorization_sha256=payload[
                "authorization_sha256"
            ],
        )
        exclusive_json(
            candidate_attempt / "candidate_result.json",
            result,
        )
        exclusive_json(
            candidate_attempt / "COMPLETE.json",
            {
                "schema_version": CANDIDATE_RESULT_SCHEMA,
                "status": "COMPLETE",
                "candidate_result_sha256": sha256_file(
                    candidate_attempt / "candidate_result.json"
                ),
            },
        )


def validate_candidate_authorization(
    *,
    repository_root: Path,
    authorization_path: Path,
    source_commit: str,
    source_hash: str,
    selection_config_sha256: str,
    development_manifest_sha256: str,
    model_id: str,
) -> Mapping[str, Any]:
    path = validate_candidate_io_path(
        repository_root=repository_root,
        path=authorization_path,
        role="candidate authorization",
        access="read",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateRunnerContractError(
            "cannot read candidate authorization"
        ) from error
    expected = {
        "schema_version": "benchmark-v2.6-candidate-authorization-v1",
        "candidate_training_authorized": True,
        "source_commit": source_commit,
        "relevant_source_sha256": source_hash,
        "selection_config_sha256": selection_config_sha256,
        "development_manifest_sha256": development_manifest_sha256,
        "test_access_authorized": False,
        "fresh_test_authorized": False,
        "five_seed_full_experiment_authorized": False,
        "worker_scope_mode": "one_model_per_worker",
        "worker_model_ids": list(MODEL_IDS),
        "aggregate_by_worker_authorized": False,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches or not str(value.get("approval_text", "")).strip():
        raise CandidateRunnerContractError(
            f"candidate authorization mismatch: {mismatches}"
        )
    if model_id not in value["worker_model_ids"]:
        raise CandidateRunnerContractError(
            f"candidate authorization excludes model worker: {model_id}"
        )
    return value


def validate_candidate_continuation_authorization(
    *,
    repository_root: Path,
    candidate_root: Path,
    authorization_path: Path,
    source_commit: str,
    source_hash: str,
    selection_config_sha256: str,
    development_manifest_sha256: str,
    selection_plan_sha256: str,
    model_id: str,
    plan: CandidateContinuationPlan,
) -> Mapping[str, Any]:
    repository_root = repository_root.resolve()
    candidate_root = candidate_root.resolve()
    path = validate_candidate_io_path(
        repository_root=repository_root,
        path=authorization_path,
        role="candidate continuation authorization",
        access="read",
    )
    value = _read_json_mapping(
        path,
        role="candidate continuation authorization",
    )
    expected = {
        "schema_version": (
            "benchmark-v2.6-candidate-continuation-authorization-v1"
        ),
        "candidate_continuation_authorized": True,
        "source_commit": source_commit,
        "relevant_source_sha256": source_hash,
        "selection_config_sha256": selection_config_sha256,
        "development_manifest_sha256": development_manifest_sha256,
        "selection_plan_sha256": selection_plan_sha256,
        "test_access_authorized": False,
        "fresh_test_authorized": False,
        "five_seed_full_experiment_authorized": False,
        "validation_selection_execution_authorized": False,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    worker_model_ids = value.get("worker_model_ids")
    if worker_model_ids != [
        "ctgan_separate_class",
        "tvae_separate_class",
    ]:
        mismatches["worker_model_ids"] = {
            "expected": [
                "ctgan_separate_class",
                "tvae_separate_class",
            ],
            "actual": worker_model_ids,
        }
    if mismatches or not str(value.get("approval_text", "")).strip():
        raise CandidateRunnerContractError(
            f"candidate continuation authorization mismatch: {mismatches}"
        )
    scope_by_model = value.get("continuation_scope")
    if not isinstance(scope_by_model, Mapping):
        raise CandidateRunnerContractError(
            "candidate continuation scope is missing"
        )
    scope = scope_by_model.get(model_id)
    if not isinstance(scope, Mapping):
        raise CandidateRunnerContractError(
            f"candidate continuation excludes model worker: {model_id}"
        )
    native_ids = {
        candidate.shared_trajectory_id
        for candidate in plan.evaluation_only_candidates
    }
    if len(native_ids) != 1:
        raise CandidateRunnerContractError(
            f"continuation native scope is invalid: {model_id}"
        )
    native_trajectory_id = next(iter(native_ids))
    evaluation_ids = [
        candidate.candidate_id
        for candidate in plan.evaluation_only_candidates
    ]
    training_ids = [
        candidate.candidate_id
        for trajectory in plan.training_trajectories
        for candidate in trajectory.candidates
    ]
    scope_expected = {
        "worker_attempt": "attempt_002",
        "evaluation_only_candidate_ids": evaluation_ids,
        "training_only_candidate_ids": training_ids,
        "native_trajectory_retraining_authorized": False,
    }
    scope_mismatches = {
        key: {"expected": expected_value, "actual": scope.get(key)}
        for key, expected_value in scope_expected.items()
        if scope.get(key) != expected_value
    }
    if scope_mismatches:
        raise CandidateRunnerContractError(
            f"candidate continuation scope mismatch: {scope_mismatches}"
        )
    ownership_path = (
        candidate_root / "_workers" / model_id / "ownership.lock"
    )
    prior_terminal_path = _authorized_existing_path(
        repository_root=repository_root,
        relative_path=scope.get("prior_worker_terminal_path"),
        role="prior model worker terminal",
    )
    expected_prior_terminal = (
        candidate_root
        / "_workers"
        / model_id
        / "attempt_001"
        / "WORKER_FAILED.json"
    ).resolve()
    if prior_terminal_path != expected_prior_terminal:
        raise CandidateRunnerContractError(
            "prior model worker terminal path mismatch"
        )
    fixed_hashes = (
        (
            ownership_path,
            scope.get("prior_ownership_lock_sha256"),
            "ownership lock",
        ),
        (
            prior_terminal_path,
            scope.get("prior_worker_terminal_sha256"),
            "prior worker terminal",
        ),
    )
    for artifact_path, expected_hash, role in fixed_hashes:
        if (
            not artifact_path.is_file()
            or sha256_file(artifact_path) != expected_hash
        ):
            raise CandidateRunnerContractError(
                f"candidate continuation {role} hash mismatch"
            )
    preservation_hash = preserved_attempt_tree_sha256(
        repository_root=repository_root,
        candidate_root=candidate_root,
        model_id=model_id,
        native_trajectory_id=native_trajectory_id,
        selection_seed=plan.selection_seed,
    )
    manifest_path = _authorized_existing_path(
        repository_root=repository_root,
        relative_path=scope.get("native_trajectory_manifest_path"),
        role="native trajectory manifest",
    )
    complete_path = _authorized_existing_path(
        repository_root=repository_root,
        relative_path=scope.get("native_trajectory_complete_path"),
        role="native trajectory completion",
    )
    native_attempt = (
        candidate_root
        / "_trajectories"
        / model_id
        / native_trajectory_id
        / f"seed_{plan.selection_seed}"
        / "attempt_001"
    ).resolve()
    if (
        manifest_path != native_attempt / "manifest.json"
        or complete_path != native_attempt / "TRAJECTORY_COMPLETE.json"
    ):
        raise CandidateRunnerContractError(
            "candidate continuation native trajectory path mismatch"
        )
    for artifact_path, expected_hash, role in (
        (
            manifest_path,
            scope.get("native_trajectory_manifest_sha256"),
            "native trajectory manifest",
        ),
        (
            complete_path,
            scope.get("native_trajectory_complete_sha256"),
            "native trajectory completion",
        ),
    ):
        if sha256_file(artifact_path) != expected_hash:
            raise CandidateRunnerContractError(
                f"candidate continuation {role} hash mismatch"
            )
    native_manifest = _read_json_mapping(
        manifest_path,
        role="native trajectory manifest",
    )
    for key, expected_value in (
        (
            "source_commit",
            scope.get("native_training_source_commit"),
        ),
        (
            "relevant_source_sha256",
            scope.get("native_training_relevant_source_sha256"),
        ),
        ("selection_config_sha256", selection_config_sha256),
        ("development_manifest_sha256", development_manifest_sha256),
        ("selection_plan_sha256", selection_plan_sha256),
        ("model_id", model_id),
        ("shared_trajectory_id", native_trajectory_id),
    ):
        if native_manifest.get(key) != expected_value:
            raise CandidateRunnerContractError(
                f"native trajectory provenance mismatch: {key}"
            )
    checkpoints = scope.get("native_checkpoints")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(
        evaluation_ids
    ):
        raise CandidateRunnerContractError(
            "candidate continuation checkpoint index mismatch"
        )
    for candidate in plan.evaluation_only_candidates:
        record = checkpoints.get(candidate.candidate_id)
        if not isinstance(record, Mapping):
            raise CandidateRunnerContractError(
                f"candidate continuation checkpoint missing: "
                f"{candidate.candidate_id}"
            )
        checkpoint_path = _authorized_existing_path(
            repository_root=repository_root,
            relative_path=record.get("path"),
            role=f"native checkpoint {candidate.candidate_id}",
        )
        expected_checkpoint_path = (
            native_attempt
            / "checkpoints"
            / f"step_{candidate.sampled_checkpoint_step:05d}.pt"
        )
        if (
            checkpoint_path != expected_checkpoint_path
            or sha256_file(checkpoint_path) != record.get("sha256")
        ):
            raise CandidateRunnerContractError(
                f"candidate continuation checkpoint hash/path mismatch: "
                f"{candidate.candidate_id}"
            )
    if preservation_hash != scope.get(
        "attempt_001_preservation_sha256"
    ):
        raise CandidateRunnerContractError(
            "candidate continuation attempt_001 preservation hash mismatch"
        )
    return value


def validate_candidate_aggregate_authorization(
    *,
    repository_root: Path,
    authorization_path: Path,
    source_commit: str,
    source_hash: str,
    selection_config_sha256: str,
    development_manifest_sha256: str,
) -> Mapping[str, Any]:
    path = validate_candidate_io_path(
        repository_root=repository_root,
        path=authorization_path,
        role="candidate aggregate authorization",
        access="read",
    )
    value = _read_json_mapping(
        path,
        role="candidate aggregate authorization",
    )
    expected = {
        "schema_version": "benchmark-v2.6-candidate-authorization-v1",
        "source_commit": source_commit,
        "relevant_source_sha256": source_hash,
        "selection_config_sha256": selection_config_sha256,
        "development_manifest_sha256": development_manifest_sha256,
        "validation_selection_execution_authorized": True,
        "aggregate_only_authorized": True,
        "aggregate_by_worker_authorized": False,
        "test_access_authorized": False,
        "fresh_test_authorized": False,
        "five_seed_full_experiment_authorized": False,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches or not str(value.get("approval_text", "")).strip():
        raise CandidateRunnerContractError(
            f"candidate aggregate authorization mismatch: {mismatches}"
        )
    evaluated_model_ids = tuple(
        value.get("selection_model_ids", MODEL_IDS)
    )
    unavailable_model_ids = tuple(
        value.get("unavailable_model_ids", ())
    )
    if (
        len(set(evaluated_model_ids)) != len(evaluated_model_ids)
        or len(set(unavailable_model_ids))
        != len(unavailable_model_ids)
        or set(evaluated_model_ids) & set(unavailable_model_ids)
        or set(evaluated_model_ids) | set(unavailable_model_ids)
        != set(MODEL_IDS)
        or not set(unavailable_model_ids) <= set(SECONDARY_MODEL_IDS)
    ):
        raise CandidateRunnerContractError(
            "candidate aggregate authorization model scope mismatch"
        )
    return value


def write_selection_artifact_bundle(
    *,
    output_root: Path,
    authorization_path: Path,
    readiness: Mapping[str, Any],
    report: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    source_commit: str,
    relevant_source_sha256: str,
    worker_count: int,
    trajectory_count: int,
    candidate_count: int,
    unavailable_model_ids: Sequence[str],
    primary_c2_selection_ready: bool,
) -> Mapping[str, Any]:
    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_root.mkdir()
    except FileExistsError as error:
        raise CandidateRunnerContractError(
            f"append-only aggregate attempt exists: {output_root}"
        ) from error
    artifact_values = (
        ("aggregate_readiness.json", readiness),
        ("selection_report.json", report),
        ("selection_manifest.json", selection_manifest),
    )
    for name, value in artifact_values:
        exclusive_json(output_root / name, value)
    checksum_records = [
        {
            "path": name,
            "sha256": sha256_file(output_root / name),
        }
        for name, _ in artifact_values
    ]
    checksum_path = output_root / "checksum_manifest.json"
    exclusive_json(
        checksum_path,
        {
            "schema_version": (
                "benchmark-v2.6-selection-checksum-manifest-v1"
            ),
            "status": "COMPLETE",
            "artifacts": checksum_records,
        },
    )
    indexed_records = [
        *checksum_records,
        {
            "path": checksum_path.name,
            "sha256": sha256_file(checksum_path),
        },
    ]
    index_path = output_root / "artifact_index.json"
    exclusive_json(
        index_path,
        {
            "schema_version": "benchmark-v2.6-selection-artifact-index-v1",
            "status": "COMPLETE",
            "source_commit": source_commit,
            "relevant_source_sha256": relevant_source_sha256,
            "authorization_sha256": sha256_file(authorization_path),
            "artifacts": indexed_records,
        },
    )
    terminal = {
        "schema_version": "benchmark-v2.6-selection-terminal-v1",
        "status": "COMPLETE",
        "worker_count": int(worker_count),
        "trajectory_count": int(trajectory_count),
        "candidate_count": int(candidate_count),
        "unavailable_model_ids": list(unavailable_model_ids),
        "authorization_sha256": sha256_file(authorization_path),
        "selection_report_sha256": sha256_file(
            output_root / "selection_report.json"
        ),
        "selection_manifest_sha256": sha256_file(
            output_root / "selection_manifest.json"
        ),
        "checksum_manifest_sha256": sha256_file(checksum_path),
        "artifact_index_sha256": sha256_file(index_path),
        "primary_c2_selection_ready": bool(
            primary_c2_selection_ready
        ),
        "test_split_read": False,
        "fresh_test_authorized": False,
        "five_seed_full_experiment_authorized": False,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    exclusive_json(output_root / "AGGREGATE_COMPLETE.json", terminal)
    return terminal


def execute_authorized_validation_aggregate(
    *,
    repository_root: Path,
    config_path: Path,
    development_manifest_path: Path,
    candidate_root: Path,
    output_root: Path,
    authorization_path: Path,
    source_commit: str,
    mode: str = "execute",
) -> Mapping[str, Any]:
    if mode not in {"plan", "dry-run", "execute"}:
        raise CandidateRunnerContractError(
            f"unknown candidate aggregate mode: {mode}"
        )
    repository_root = repository_root.resolve()
    candidate_root = validate_candidate_io_path(
        repository_root=repository_root,
        path=candidate_root,
        role="candidate aggregate input root",
        access="read",
    )
    config_path = validate_candidate_io_path(
        repository_root=repository_root,
        path=config_path,
        role="candidate aggregate config",
        access="read",
    )
    development_manifest_path = validate_candidate_io_path(
        repository_root=repository_root,
        path=development_manifest_path,
        role="candidate aggregate development manifest",
        access="read",
    )
    output_root = output_root.resolve()
    selection_root = (
        repository_root / "artifacts/benchmark_v2_6/selection"
    ).resolve()
    if (
        output_root.parent != selection_root
        or not output_root.name.startswith("aggregate_attempt_")
        or output_root == candidate_root
        or candidate_root in output_root.parents
        or set(part.lower() for part in output_root.parts)
        & _FORBIDDEN_PATH_PARTS
    ):
        raise CandidateRunnerContractError(
            "aggregate output is outside the v2.6 selection root"
        )
    config = read_yaml_mapping(config_path)
    context = load_train_only_candidate_context(
        repository_root=repository_root,
        development_manifest_path=development_manifest_path,
    )
    source_hash = relevant_source_sha256(repository_root)
    config_hash = sha256_file(config_path)
    if current_git_head(repository_root) != source_commit:
        raise CandidateRunnerContractError(
            "candidate aggregate source commit is not current HEAD"
        )
    authorization = validate_candidate_aggregate_authorization(
        repository_root=repository_root,
        authorization_path=authorization_path,
        source_commit=source_commit,
        source_hash=source_hash,
        selection_config_sha256=config_hash,
        development_manifest_sha256=context.manifest_sha256,
    )
    evaluated_model_ids = tuple(
        authorization.get("selection_model_ids", MODEL_IDS)
    )
    unavailable_model_ids = tuple(
        authorization.get("unavailable_model_ids", ())
    )
    readiness = validate_candidate_aggregate_readiness(
        candidate_root=candidate_root,
        config=config,
        source_commit=source_commit,
        relevant_source_sha256=source_hash,
        selection_config_sha256=config_hash,
        development_manifest_sha256=context.manifest_sha256,
        selection_plan_sha256=context.plan.plan_hash,
        approved_worker_terminals=(
            authorization.get("approved_worker_terminals")
            if isinstance(
                authorization.get("approved_worker_terminals"),
                Mapping,
            )
            else None
        ),
        unavailable_model_ids=unavailable_model_ids,
    )
    if mode == "plan":
        return {
            "mode": "plan",
            "authorization": authorization,
            "readiness": readiness,
            "report": None,
            "terminal": None,
            "artifacts_written": False,
        }
    report = run_validation_selection(
        repository_root=repository_root,
        config_path=config_path,
        development_manifest_path=development_manifest_path,
        candidate_root=candidate_root,
        evaluated_model_ids=evaluated_model_ids,
        unavailable_model_ids=unavailable_model_ids,
    )
    if mode == "dry-run":
        return {
            "mode": "dry-run",
            "authorization": authorization,
            "readiness": readiness,
            "report": report,
            "terminal": None,
            "artifacts_written": False,
        }
    if report["primary_c2_selection_ready"]:
        selection_manifest = build_selection_freeze(report)
    else:
        selection_manifest = {
            "schema_version": (
                "benchmark-v2.6-primary-selection-failure-v1"
            ),
            "status": "PRIMARY_SELECTION_FAILED_NO_FULL_RUN",
            "selected_candidates": {},
            "model_selection_status": {
                model_id: selection["status"]
                for model_id, selection in report[
                    "model_selections"
                ].items()
            },
            "blocking_primary_models": report[
                "blocking_primary_models"
            ],
            "primary_c2_selection_ready": False,
            "test_split_read": False,
            "requires_separate_user_authorization": True,
        }
    terminal = write_selection_artifact_bundle(
        output_root=output_root,
        authorization_path=authorization_path,
        readiness=readiness,
        report=report,
        selection_manifest=selection_manifest,
        source_commit=source_commit,
        relevant_source_sha256=source_hash,
        worker_count=readiness["worker_count"],
        trajectory_count=readiness["trajectory_count"],
        candidate_count=readiness["candidate_count"],
        unavailable_model_ids=unavailable_model_ids,
        primary_c2_selection_ready=report[
            "primary_c2_selection_ready"
        ],
    )
    return {
        "authorization": authorization,
        "readiness": readiness,
        "report": report,
        "terminal": terminal,
    }


def execute_authorized_candidate_plan(
    *,
    repository_root: Path,
    config_path: Path,
    development_manifest_path: Path,
    candidate_root: Path,
    authorization_path: Path,
    source_commit: str,
    device: str,
    model_id: str,
    termination_grace_seconds: float = 10.0,
) -> Sequence[Mapping[str, Any]]:
    repository_root = repository_root.resolve()
    candidate_root = validate_candidate_io_path(
        repository_root=repository_root,
        path=candidate_root,
        role="candidate artifact root",
        access="write",
    )
    config_path = validate_candidate_io_path(
        repository_root=repository_root,
        path=config_path,
        role="selection config",
        access="read",
    )
    development_manifest_path = validate_candidate_io_path(
        repository_root=repository_root,
        path=development_manifest_path,
        role="development manifest",
        access="read",
    )
    config = read_yaml_mapping(config_path)
    plan = build_model_trajectory_plan(config, model_id=model_id)
    context = load_train_only_candidate_context(
        repository_root=repository_root,
        development_manifest_path=development_manifest_path,
    )
    source_hash = relevant_source_sha256(repository_root)
    config_hash = sha256_file(config_path)
    if current_git_head(repository_root) != source_commit:
        raise CandidateRunnerContractError(
            "candidate authorization source commit is not current HEAD"
        )
    validate_candidate_authorization(
        repository_root=repository_root,
        authorization_path=authorization_path,
        source_commit=source_commit,
        source_hash=source_hash,
        selection_config_sha256=config_hash,
        development_manifest_sha256=context.manifest_sha256,
        model_id=model_id,
    )
    _, trajectory_ids, candidate_ids = _model_scope(plan)
    provenance = {
        "source_commit": source_commit,
        "relevant_source_sha256": source_hash,
        "selection_config_sha256": config_hash,
        "development_manifest_sha256": context.manifest_sha256,
        "selection_plan_sha256": context.plan.plan_hash,
        "authorization_sha256": sha256_file(authorization_path),
    }
    worker_attempt = claim_model_worker(
        candidate_root=candidate_root,
        model_id=model_id,
        selection_seed=plan.selection_seed,
        trajectory_ids=trajectory_ids,
        candidate_ids=candidate_ids,
        provenance=provenance,
    )
    results = []
    for trajectory in plan.trajectories:
        trajectory_root = (
            candidate_root
            / "_trajectories"
            / trajectory.model_id
            / trajectory.shared_trajectory_id
            / f"seed_{plan.selection_seed}"
        )
        attempt = claim_next_attempt(trajectory_root)
        result = run_bounded_process(
            target=_execute_trajectory_child,
            attempt_path=attempt,
            max_wall_seconds=trajectory.max_wall_seconds,
            termination_grace_seconds=termination_grace_seconds,
            args=(
                {
                    "repository_root": str(repository_root),
                    "attempt_path": str(attempt),
                    "candidate_root": str(candidate_root),
                    "config_path": str(config_path),
                    "development_manifest_path": str(
                        development_manifest_path
                    ),
                    "model_id": trajectory.model_id,
                    "shared_trajectory_id": (
                        trajectory.shared_trajectory_id
                    ),
                    "selection_config_sha256": config_hash,
                    "relevant_source_sha256": source_hash,
                    "source_commit": source_commit,
                    "device": device,
                },
            ),
        )
        results.append(result)
        if result["status"] != "COMPLETE":
            break
    terminal = finalize_model_worker(
        candidate_root=candidate_root,
        worker_attempt=worker_attempt,
        plan=plan,
        provenance=provenance,
        trajectory_results=results,
    )
    return tuple([*results, terminal])


def execute_authorized_candidate_continuation(
    *,
    repository_root: Path,
    config_path: Path,
    development_manifest_path: Path,
    candidate_root: Path,
    authorization_path: Path,
    source_commit: str,
    device: str,
    model_id: str,
    termination_grace_seconds: float = 10.0,
) -> Sequence[Mapping[str, Any]]:
    repository_root = repository_root.resolve()
    candidate_root = validate_candidate_io_path(
        repository_root=repository_root,
        path=candidate_root,
        role="candidate continuation artifact root",
        access="write",
    )
    config_path = validate_candidate_io_path(
        repository_root=repository_root,
        path=config_path,
        role="candidate continuation config",
        access="read",
    )
    development_manifest_path = validate_candidate_io_path(
        repository_root=repository_root,
        path=development_manifest_path,
        role="candidate continuation development manifest",
        access="read",
    )
    config = read_yaml_mapping(config_path)
    plan = build_model_continuation_plan(config, model_id=model_id)
    context = load_train_only_candidate_context(
        repository_root=repository_root,
        development_manifest_path=development_manifest_path,
    )
    source_hash = relevant_source_sha256(repository_root)
    config_hash = sha256_file(config_path)
    if current_git_head(repository_root) != source_commit:
        raise CandidateRunnerContractError(
            "candidate continuation source commit is not current HEAD"
        )
    authorization = validate_candidate_continuation_authorization(
        repository_root=repository_root,
        candidate_root=candidate_root,
        authorization_path=authorization_path,
        source_commit=source_commit,
        source_hash=source_hash,
        selection_config_sha256=config_hash,
        development_manifest_sha256=context.manifest_sha256,
        selection_plan_sha256=context.plan.plan_hash,
        model_id=model_id,
        plan=plan,
    )
    scope = authorization["continuation_scope"][model_id]
    authorization_hash = sha256_file(authorization_path)
    provenance = {
        "source_commit": source_commit,
        "relevant_source_sha256": source_hash,
        "selection_config_sha256": config_hash,
        "development_manifest_sha256": context.manifest_sha256,
        "selection_plan_sha256": context.plan.plan_hash,
        "authorization_sha256": authorization_hash,
        "attempt_001_preservation_sha256": scope[
            "attempt_001_preservation_sha256"
        ],
        "native_training_source_commit": scope[
            "native_training_source_commit"
        ],
        "native_training_relevant_source_sha256": scope[
            "native_training_relevant_source_sha256"
        ],
    }
    worker_attempt = claim_model_continuation(
        repository_root=repository_root,
        candidate_root=candidate_root,
        plan=plan,
        authorization_sha256=authorization_hash,
        prior_ownership_lock_sha256=scope[
            "prior_ownership_lock_sha256"
        ],
        attempt_001_preservation_sha256=scope[
            "attempt_001_preservation_sha256"
        ],
        provenance=provenance,
    )

    def evaluate_native(
        candidates: tuple[CandidateAdapterSpec, ...],
    ) -> Mapping[str, Any]:
        native_ids = {
            candidate.shared_trajectory_id for candidate in candidates
        }
        if len(native_ids) != 1:
            raise CandidateRunnerContractError(
                "continuation native evaluation scope mismatch"
            )
        evaluation_attempt = claim_exact_attempt(
            candidate_root
            / "_evaluations"
            / model_id
            / next(iter(native_ids))
            / f"seed_{plan.selection_seed}",
            attempt_name="attempt_002",
        )
        result = run_bounded_process(
            target=_execute_native_evaluation_child,
            attempt_path=evaluation_attempt,
            max_wall_seconds=max(
                candidate.max_wall_seconds for candidate in candidates
            ),
            termination_grace_seconds=termination_grace_seconds,
            args=(
                {
                    "repository_root": str(repository_root),
                    "attempt_path": str(evaluation_attempt),
                    "candidate_root": str(candidate_root),
                    "config_path": str(config_path),
                    "development_manifest_path": str(
                        development_manifest_path
                    ),
                    "model_id": model_id,
                    "selection_config_sha256": config_hash,
                    "relevant_source_sha256": source_hash,
                    "source_commit": source_commit,
                    "device": device,
                    "authorization_sha256": authorization_hash,
                    "continuation_scope": dict(scope),
                },
            ),
        )
        return {
            **dict(result),
            "operation": "evaluation_only",
            "candidate_ids": [
                candidate.candidate_id for candidate in candidates
            ],
            "native_trajectory_retrained": False,
        }

    def train_trajectory(
        trajectory: CandidateTrajectory,
    ) -> Mapping[str, Any]:
        if "native" in trajectory.shared_trajectory_id:
            raise CandidateRunnerContractError(
                "native trajectory retraining is forbidden in continuation"
            )
        trajectory_attempt = claim_exact_attempt(
            candidate_root
            / "_trajectories"
            / trajectory.model_id
            / trajectory.shared_trajectory_id
            / f"seed_{plan.selection_seed}",
            attempt_name="attempt_002",
        )
        result = run_bounded_process(
            target=_execute_trajectory_child,
            attempt_path=trajectory_attempt,
            max_wall_seconds=trajectory.max_wall_seconds,
            termination_grace_seconds=termination_grace_seconds,
            args=(
                {
                    "repository_root": str(repository_root),
                    "attempt_path": str(trajectory_attempt),
                    "candidate_root": str(candidate_root),
                    "config_path": str(config_path),
                    "development_manifest_path": str(
                        development_manifest_path
                    ),
                    "model_id": trajectory.model_id,
                    "shared_trajectory_id": (
                        trajectory.shared_trajectory_id
                    ),
                    "selection_config_sha256": config_hash,
                    "relevant_source_sha256": source_hash,
                    "source_commit": source_commit,
                    "device": device,
                },
            ),
        )
        return {
            **dict(result),
            "operation": "training_only",
            "shared_trajectory_id": trajectory.shared_trajectory_id,
            "candidate_ids": [
                candidate.candidate_id
                for candidate in trajectory.candidates
            ],
        }

    results = dispatch_continuation_plan(
        plan,
        evaluate_native=evaluate_native,
        train_trajectory=train_trajectory,
    )
    terminal = finalize_model_continuation(
        candidate_root=candidate_root,
        worker_attempt=worker_attempt,
        plan=plan,
        provenance=provenance,
        operation_results=results,
    )
    return tuple([*results, terminal])
