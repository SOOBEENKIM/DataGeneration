"""Append-only future execution runner contracts for CoF-SeqGen v3."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import multiprocessing
import os
from pathlib import Path
import queue
import subprocess
import time
import traceback
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from eval.cof_seqgen_v3_contract import (
    FROZEN_AMOUNT_CONTRACT,
    V3ContractError,
    bind_train_only_conditioning_contract,
    build_train_joint_support_state,
    canonical_sha256,
    sha256_file,
)
from eval.evaluation_only_v2_7 import evaluate_validation_guards
from eval.validation_selection_v2_6 import load_development_context
from experiments.cof_seqgen_v3_preparation import (
    build_v3_plan,
    dry_run_v3,
    v3_relevant_source_sha256,
)
from experiments.candidate_runner_v2_6 import (
    load_train_only_candidate_context,
)
from generators.cof_seqgen_v3_candidate import (
    CoFSeqGenV3CandidateAdapter,
    V3FittedCandidate,
)
from generators.sampling_plan import SamplingPlan
from models.cof_seqgen_v3 import (
    CoFSeqDenoiserV3,
    CoFSeqGenV3,
    DirectJointDiscretePath,
    FactorizedJointDiscretePath,
    JointStateCodec,
)


EXECUTION_COUNTS_ZERO = {
    "gpu_queries": 0,
    "cuda_calls": 0,
    "model_fit_calls": 0,
    "optimizer_updates": 0,
    "checkpoint_writes": 0,
    "model_sample_calls": 0,
    "validation_execution_calls": 0,
    "data_generation_calls": 0,
    "selection_calls": 0,
    "test_split_reads": 0,
    "fresh_test_calls": 0,
    "tstr_calls": 0,
    "privacy_calls": 0,
    "five_seed_full_run_calls": 0,
}
EXPECTED_CANDIDATES = (
    ("cof_v3_c01_direct_joint", "direct_joint"),
    ("cof_v3_c02_factorized_joint", "factorized_joint"),
)
DEFAULT_CHILD_STARTUP_TIMEOUT_SECONDS = 300.0
DEFAULT_CHILD_PROGRESS_TIMEOUT_SECONDS = 900.0
EXECUTION_RELEVANT_SOURCE_PATHS = (
    "configs/benchmark_v2/cof_seqgen_v3_source_preparation.yaml",
    "configs/benchmark_v2/cof_seqgen_v3_validation_runner.yaml",
    "models/cof_seqgen_v3.py",
    "eval/cof_seqgen_v3_contract.py",
    "generators/cof_seqgen_v3_candidate.py",
    "experiments/cof_seqgen_v3_execution_runner.py",
    "scripts/run_cof_seqgen_v3_validation.py",
)


@dataclass(frozen=True)
class V3ExecutionOperation:
    candidate_id: str
    architecture: str
    training_split: str
    validation_use: str
    seed: int
    requested_updates: int
    max_wall_seconds: float


@dataclass(frozen=True)
class V3ExecutionPlan:
    repository_root: Path
    runner_config_path: Path
    runner_config_sha256: str
    source_plan: Any
    operations: tuple[V3ExecutionOperation, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class RunnerChildSpec:
    """Spawn-safe description of one runner-owned child operation."""

    target: Any
    payload: Mapping[str, Any]


class V3AttemptStore:
    """Exclusive candidate owner and append-only attempt writer."""

    REQUIRED_ARTIFACTS = (
        "manifest.json",
        "runtime.json",
        "checkpoint.pt",
        "checkpoint_provenance.json",
        "joint_support_state.json",
        "amount_fit_state.json",
        "conditioning_binding.json",
        "sample.npz",
        "metrics.json",
        "evaluation.json",
    )
    TERMINALS = ("COMPLETE", "INVALID", "FAILED")

    def __init__(
        self,
        *,
        runtime_root: Path,
        operation: V3ExecutionOperation,
        ownership_path: Path,
        expected_attempt_name: str | None = None,
    ) -> None:
        self.runtime_root = runtime_root
        self.operation = operation
        self.ownership_path = ownership_path
        self.expected_attempt_name = expected_attempt_name
        self.attempt_path: Path | None = None

    @staticmethod
    def _exclusive_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            raise

    @classmethod
    def claim(
        cls,
        *,
        runtime_root: Path,
        operation: V3ExecutionOperation,
        ownership_id: str,
        authorization_sha256: str,
        provenance_sha256: str,
        continuation: Mapping[str, Any] | None = None,
    ) -> "V3AttemptStore":
        if (
            not ownership_id
            or len(authorization_sha256) != 64
            or len(provenance_sha256) != 64
        ):
            raise V3ContractError("invalid candidate ownership claim")
        runtime_root = runtime_root.resolve()
        worker_root = (
            runtime_root
            / "workers"
            / operation.candidate_id
        )
        ownership = worker_root / "ownership.lock"
        expected_attempt_name: str | None = None
        if ownership.exists():
            if not isinstance(continuation, Mapping):
                raise V3ContractError(
                    "candidate ownership exists; hash-bound continuation "
                    "authorization is required"
                )
            required_keys = {
                "schema_version",
                "previous_attempt",
                "previous_attempt_tree_sha256",
                "previous_terminal_status",
                "previous_terminal_sha256",
                "previous_authorization_sha256",
                "next_attempt",
            }
            if (
                set(continuation) != required_keys
                or continuation.get("schema_version")
                != "cof-seqgen-v3-continuation-v1"
                or continuation.get("previous_attempt") != "attempt_001"
                or continuation.get("previous_terminal_status") != "FAILED"
                or continuation.get("next_attempt") != "attempt_002"
            ):
                raise V3ContractError("continuation authorization is invalid")
            prior_attempt = (
                runtime_root
                / "candidates"
                / operation.candidate_id
                / f"seed_{operation.seed}"
                / "attempt_001"
            )
            prior_terminal = prior_attempt / "FAILED.json"
            prior_manifest = _read_json_mapping(
                prior_attempt / "manifest.json"
            )
            if (
                not prior_terminal.is_file()
                or read_only_tree_inventory(prior_attempt)["tree_sha256"]
                != continuation["previous_attempt_tree_sha256"]
                or sha256_file(prior_terminal)
                != continuation["previous_terminal_sha256"]
                or prior_manifest.get("authorization_sha256")
                != continuation["previous_authorization_sha256"]
                or (prior_attempt.parent / "attempt_002").exists()
            ):
                raise V3ContractError(
                    "continuation preservation provenance mismatch"
                )
            expected_attempt_name = "attempt_002"
            ownership = worker_root / "ownership_attempt_002.json"
        elif continuation is not None:
            raise V3ContractError(
                "continuation authorization has no preserved ownership"
            )
        value = {
            "schema_version": "cof-seqgen-v3-worker-ownership-v1",
            "candidate_id": operation.candidate_id,
            "architecture": operation.architecture,
            "ownership_id": ownership_id,
            "authorization_sha256": authorization_sha256,
            "provenance_sha256": provenance_sha256,
        }
        if continuation is not None:
            value["continuation"] = dict(continuation)
        try:
            cls._exclusive_bytes(
                ownership,
                (
                    json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
        except FileExistsError as error:
            raise V3ContractError(
                f"candidate ownership continuation already exists: "
                f"{operation.candidate_id}"
            ) from error
        return cls(
            runtime_root=runtime_root,
            operation=operation,
            ownership_path=ownership,
            expected_attempt_name=expected_attempt_name,
        )

    def allocate_attempt(
        self,
        *,
        manifest: Mapping[str, Any],
    ) -> Path:
        if self.attempt_path is not None:
            raise V3ContractError("store already owns an active attempt")
        if (
            manifest.get("candidate_id") != self.operation.candidate_id
            or manifest.get("architecture") != self.operation.architecture
        ):
            raise V3ContractError("candidate attempt manifest mismatch")
        base = (
            self.runtime_root
            / "candidates"
            / self.operation.candidate_id
            / f"seed_{self.operation.seed}"
        )
        base.mkdir(parents=True, exist_ok=True)
        numbers = [
            int(path.name.removeprefix("attempt_"))
            for path in base.glob("attempt_[0-9][0-9][0-9]")
            if path.is_dir()
        ]
        number = max(numbers, default=0) + 1
        attempt = base / f"attempt_{number:03d}"
        if (
            self.expected_attempt_name is not None
            and attempt.name != self.expected_attempt_name
        ):
            raise V3ContractError("continuation attempt allocation mismatch")
        attempt.mkdir(parents=False, exist_ok=False)
        self.attempt_path = attempt
        self.write_json("manifest.json", manifest)
        return attempt

    @classmethod
    def attach_existing(
        cls,
        *,
        runtime_root: Path,
        operation: V3ExecutionOperation,
        ownership_path: Path,
        attempt_path: Path,
    ) -> "V3AttemptStore":
        """Attach a spawned child to its parent's already-created attempt."""

        runtime_root = runtime_root.resolve()
        ownership_path = ownership_path.resolve()
        attempt_path = attempt_path.resolve()
        worker_root = (
            runtime_root / "workers" / operation.candidate_id
        ).resolve()
        candidate_root = (
            runtime_root
            / "candidates"
            / operation.candidate_id
            / f"seed_{operation.seed}"
        ).resolve()
        if (
            not ownership_path.is_file()
            or worker_root not in ownership_path.parents
            or not attempt_path.is_dir()
            or candidate_root not in attempt_path.parents
        ):
            raise V3ContractError("spawn child artifact ownership mismatch")
        manifest = _read_json_mapping(attempt_path / "manifest.json")
        if (
            manifest.get("candidate_id") != operation.candidate_id
            or manifest.get("architecture") != operation.architecture
            or any(
                (attempt_path / f"{status}.json").exists()
                for status in cls.TERMINALS
            )
        ):
            raise V3ContractError("spawn child attempt is not writable")
        store = cls(
            runtime_root=runtime_root,
            operation=operation,
            ownership_path=ownership_path,
        )
        store.attempt_path = attempt_path
        return store

    def _target(self, relative: str) -> Path:
        if self.attempt_path is None:
            raise V3ContractError("candidate attempt is not allocated")
        target = (self.attempt_path / relative).resolve()
        try:
            target.relative_to(self.attempt_path.resolve())
        except ValueError as error:
            raise V3ContractError("artifact path escapes attempt") from error
        return target

    def write_json(self, relative: str, value: Mapping[str, Any]) -> Path:
        target = self._target(relative)
        payload = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self._exclusive_bytes(target, payload)
        return target

    def write_bytes(self, relative: str, value: bytes) -> Path:
        target = self._target(relative)
        self._exclusive_bytes(target, value)
        return target

    def append_jsonl(self, relative: str, value: Mapping[str, Any]) -> Path:
        target = self._target(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            with os.fdopen(descriptor, "ab") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            raise
        return target

    def finalize(
        self,
        status: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if status not in self.TERMINALS:
            raise V3ContractError("invalid candidate terminal status")
        attempt = self._target(".").resolve()
        existing = [
            name
            for name in self.TERMINALS
            if (attempt / f"{name}.json").is_file()
        ]
        if existing:
            raise V3ContractError("candidate already has a terminal marker")
        missing = [
            relative
            for relative in self.REQUIRED_ARTIFACTS
            if not (attempt / relative).is_file()
        ]
        if missing:
            raise V3ContractError(
                f"candidate terminal artifacts are missing: {missing}"
            )
        evaluation = json.loads(
            (attempt / "evaluation.json").read_text(encoding="utf-8")
        )
        if status == "COMPLETE" and (
            evaluation.get("status") != "VALID"
            or evaluation.get("all_five_guards_pass") is not True
        ):
            raise V3ContractError(
                "COMPLETE requires a valid all-five-pass evaluation"
            )
        artifact_sha256 = {
            relative: sha256_file(attempt / relative)
            for relative in self.REQUIRED_ARTIFACTS
        }
        terminal = {
            "schema_version": "cof-seqgen-v3-candidate-terminal-v1",
            "status": status,
            "candidate_id": self.operation.candidate_id,
            "architecture": self.operation.architecture,
            "artifact_sha256": artifact_sha256,
            "aggregate_eligible": status == "COMPLETE",
        }
        if metadata:
            reserved = set(terminal) & set(metadata)
            if reserved:
                raise V3ContractError(
                    f"terminal metadata overwrites reserved keys: {reserved}"
                )
            terminal.update(dict(metadata))
        self.write_json(f"{status}.json", terminal)
        return terminal

    def finalize_worker(
        self,
        candidate_terminal: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Write a worker terminal only after verifying candidate terminal."""

        if self.attempt_path is None:
            raise V3ContractError("worker has no allocated candidate attempt")
        candidate_status = str(candidate_terminal.get("status", ""))
        if candidate_status not in self.TERMINALS:
            raise V3ContractError("worker candidate terminal is invalid")
        candidate_path = self.attempt_path / f"{candidate_status}.json"
        if not candidate_path.is_file():
            raise V3ContractError("worker candidate terminal is missing")
        stored = _read_json_mapping(candidate_path)
        if canonical_sha256(stored) != canonical_sha256(candidate_terminal):
            raise V3ContractError("worker candidate terminal changed")
        worker_status = (
            "FAILED" if candidate_status == "FAILED" else "COMPLETE"
        )
        worker_dir = self.ownership_path.parent / self.attempt_path.name
        worker_path = worker_dir / f"WORKER_{worker_status}.json"
        terminal = {
            "schema_version": "cof-seqgen-v3-worker-terminal-v1",
            "status": worker_status,
            "candidate_id": self.operation.candidate_id,
            "architecture": self.operation.architecture,
            "candidate_status": candidate_status,
            "candidate_attempt": self.attempt_path.name,
            "candidate_terminal_path": str(candidate_path),
            "candidate_terminal_sha256": sha256_file(candidate_path),
            "ownership_sha256": sha256_file(self.ownership_path),
            "candidate_terminal_verified_before_worker": True,
        }
        self._exclusive_bytes(
            worker_path,
            (
                json.dumps(
                    terminal,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        return terminal


def _load_runner_config(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise V3ContractError(f"cannot read v3 runner config: {path}") from error
    if not isinstance(value, Mapping):
        raise V3ContractError("v3 runner config root must be a mapping")
    return value


def build_v3_execution_plan(config_path: Path) -> V3ExecutionPlan:
    config_path = config_path.resolve()
    raw = _load_runner_config(config_path)
    root = config_path.parents[2]
    false_flags = (
        "execution_authorized",
        "authorization_creation_authorized",
        "gpu_query_authorized",
        "cuda_authorized",
        "model_fit_authorized",
        "model_sample_authorized",
        "validation_execution_authorized",
        "selection_authorized",
        "test_authorized",
        "tstr_authorized",
        "privacy_authorized",
        "five_seed_full_run_authorized",
    )
    if (
        raw.get("schema_version")
        != "cof-seqgen-v3-validation-runner-v1"
        or raw.get("mode") != "SOURCE_ONLY_IMPLEMENTATION"
        or raw.get("test_split_access") != "FORBIDDEN"
        or raw.get("authorization_required_for_execute") is not True
        or any(raw.get(flag) is not False for flag in false_flags)
    ):
        raise V3ContractError("invalid v3 execution-runner safety boundary")
    source_config = root / str(raw["source_config_path"])
    if sha256_file(source_config) != raw.get("source_config_sha256"):
        raise V3ContractError("v3 source config hash mismatch")
    source_plan = build_v3_plan(source_config)
    if (
        v3_relevant_source_sha256(root)
        != raw.get("source_relevant_sha256")
    ):
        raise V3ContractError("v3 source preparation hash mismatch")
    data = raw.get("data_contract")
    execution = raw.get("execution_contract")
    terminal = raw.get("terminal_contract")
    if (
        not isinstance(data, Mapping)
        or not isinstance(execution, Mapping)
        or not isinstance(terminal, Mapping)
    ):
        raise V3ContractError("v3 execution contract is missing")
    development_manifest = root / str(data["development_manifest_path"])
    sampling_plan_source = root / str(data["sampling_plan_source_path"])
    if (
        sha256_file(development_manifest)
        != data.get("development_manifest_sha256")
        or sha256_file(sampling_plan_source)
        != data.get("sampling_plan_file_sha256")
    ):
        raise V3ContractError("v3 frozen data provenance changed")
    observed = tuple(
        (
            item.get("candidate_id"),
            item.get("architecture"),
        )
        for item in execution.get("candidates", ())
    )
    if (
        observed != EXPECTED_CANDIDATES
        or data.get("fit_split") != "train"
        or data.get("validation_use")
        != "post_generation_five_guard_only"
        or data.get("test_rows_used") != 0
        or execution.get("seed") != 3001
        or execution.get("requested_updates") != 20_000
        or execution.get("max_wall_seconds_per_candidate") != 7_200
        or execution.get("optimizer") != "Adam"
        or execution.get("early_stopping") != "FORBIDDEN"
        or execution.get("sweep") != "FORBIDDEN"
        or execution.get("post_hoc_calibration") != "FORBIDDEN"
        or execution.get("legacy_independent_heads") != "FORBIDDEN"
        or execution.get("aggregate_selection") != "FORBIDDEN"
        or tuple(terminal.get("required_nonterminal", ()))
        != V3AttemptStore.REQUIRED_ARTIFACTS
        or tuple(terminal.get("terminal_markers", ()))
        != tuple(f"{status}.json" for status in V3AttemptStore.TERMINALS)
        or terminal.get("aggregate_eligibility")
        != "all_five_guards_PASS_only"
    ):
        raise V3ContractError("v3 frozen execution plan changed")
    operations = tuple(
        V3ExecutionOperation(
            candidate_id=candidate,
            architecture=architecture,
            training_split=str(data["fit_split"]),
            validation_use=str(data["validation_use"]),
            seed=int(execution["seed"]),
            requested_updates=int(execution["requested_updates"]),
            max_wall_seconds=float(
                execution["max_wall_seconds_per_candidate"]
            ),
        )
        for candidate, architecture in EXPECTED_CANDIDATES
    )
    return V3ExecutionPlan(
        repository_root=root,
        runner_config_path=config_path,
        runner_config_sha256=sha256_file(config_path),
        source_plan=source_plan,
        operations=operations,
        raw=raw,
    )


def v3_execution_plan_report(
    plan: V3ExecutionPlan,
) -> Mapping[str, Any]:
    data = plan.raw["data_contract"]
    return {
        "schema_version": "cof-seqgen-v3-execution-plan-v1",
        "status": "PASS",
        "mode": "SOURCE_ONLY_IMPLEMENTATION",
        "runner_config_sha256": plan.runner_config_sha256,
        "source_config_sha256": plan.raw["source_config_sha256"],
        "source_relevant_sha256": plan.raw["source_relevant_sha256"],
        "sampling_plan_sha256": data["sampling_plan_sha256"],
        "operations": [
            {
                "candidate_id": operation.candidate_id,
                "architecture": operation.architecture,
                "training_split": operation.training_split,
                "validation_use": operation.validation_use,
                "seed": operation.seed,
                "requested_updates": operation.requested_updates,
                "max_wall_seconds": operation.max_wall_seconds,
            }
            for operation in plan.operations
        ],
        "execution_counts": dict(EXECUTION_COUNTS_ZERO),
        "authorization_created": False,
        "runtime_artifact_created": False,
        "execution_authorized": False,
    }


def dry_run_v3_execution(plan: V3ExecutionPlan) -> Mapping[str, Any]:
    """Read-only provenance and import validation for a future execution."""

    source_report = dry_run_v3(plan.source_plan)
    data = plan.raw["data_contract"]
    root = plan.repository_root
    runtime_root = root / str(plan.raw["future_runtime_root"])
    runtime_before = read_only_tree_inventory(runtime_root)
    context = load_train_only_candidate_context(
        repository_root=root,
        development_manifest_path=(
            root / str(data["development_manifest_path"])
        ),
    )
    if (
        context.train_file_sha256 != data["train_file_sha256"]
        or context.train_content_sha256 != data["train_content_sha256"]
        or context.validation_file_sha256
        != data["validation_file_sha256"]
        or context.validation_content_sha256
        != data["validation_content_sha256"]
        or context.plan.plan_hash != data["sampling_plan_sha256"]
        or sha256_file(root / str(data["validation_path"]))
        != data["validation_file_sha256"]
    ):
        raise V3ContractError("v3 train/validation dry-run provenance mismatch")
    runtime_after = read_only_tree_inventory(runtime_root)
    if runtime_after != runtime_before:
        raise V3ContractError("v3 dry-run changed preserved runtime artifacts")
    return {
        **v3_execution_plan_report(plan),
        "schema_version": "cof-seqgen-v3-execution-dry-run-v1",
        "source_preparation_dry_run_status": source_report["status"],
        "frozen_provenance_verified": True,
        "v2_8_candidate_tree_sha256": source_report[
            "v2_8_candidate_tree_sha256"
        ],
        "train_loaded_for_contract_validation": True,
        "validation_file_hash_verified": True,
        "validation_rows_loaded": 0,
        "test_split_reads": 0,
        "runner_relevant_source_sha256": (
            v3_execution_relevant_source_sha256(root)
        ),
        "preserved_runtime_tree_sha256": runtime_before["tree_sha256"],
        "preserved_runtime_file_count": runtime_before["file_count"],
        "runtime_artifact_created": False,
    }


def read_only_tree_inventory(root: Path) -> Mapping[str, Any]:
    """Return a deterministic file/hash inventory without modifying ``root``."""

    root = root.resolve()
    files: dict[str, Mapping[str, Any]] = {}
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            files[path.relative_to(root).as_posix()] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return {
        "tree_sha256": canonical_sha256({"files": files}),
        "file_count": len(files),
        "byte_count": sum(item["bytes"] for item in files.values()),
        "files": files,
    }


def v3_execution_relevant_source_sha256(repository_root: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for relative in EXECUTION_RELEVANT_SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise V3ContractError(f"v3 runner source is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_v3_candidate_adapter(
    *,
    plan: V3ExecutionPlan,
    operation: V3ExecutionOperation,
    sampling_plan: SamplingPlan,
    device: str,
) -> CoFSeqGenV3CandidateAdapter:
    """Construct an inert authorized-only adapter without touching CUDA."""

    if operation not in plan.operations:
        raise V3ContractError("candidate operation is not in the frozen plan")
    data = plan.raw["data_contract"]
    if sampling_plan.plan_hash != data["sampling_plan_sha256"]:
        raise V3ContractError("candidate SamplingPlan hash mismatch")
    source_model = plan.source_plan.definition.raw["model_contract"]
    execution = plan.raw["execution_contract"]
    effective = {
        "learning_rate": execution["learning_rate"],
        "weight_decay": execution["weight_decay"],
        "batch_size": execution["batch_size"],
        "checkpoint_interval_updates": execution[
            "checkpoint_interval_updates"
        ],
        "diffusion_steps": source_model["diffusion_steps"],
        "guidance_scale": source_model["guidance_scale"],
        "discrete_temperature": source_model["discrete_temperature"],
    }
    return CoFSeqGenV3CandidateAdapter(
        architecture=operation.architecture,
        config=effective,
        sampling_plan=sampling_plan,
        train_file_sha256=data["train_file_sha256"],
        train_content_sha256=data["train_content_sha256"],
        device=device,
    )


def validate_v3_execution_authorization(
    *,
    plan: V3ExecutionPlan,
    authorization: Mapping[str, Any],
    source_commit: str,
    runner_relevant_source_sha256: str,
) -> Mapping[str, Any]:
    """Validate, but never create, the later explicit execution approval."""

    data = plan.raw["data_contract"]
    expected_scope = {
        "gpu_query": False,
        "cuda": True,
        "model_fit": True,
        "optimizer_updates": True,
        "checkpoint_write": True,
        "model_sample": True,
        "validation_five_guard_evaluation": True,
        "aggregate_selection": False,
        "test_split_access": False,
        "fresh_test": False,
        "tstr": False,
        "privacy": False,
        "five_seed_full_run": False,
        "sweep": False,
        "early_stopping": False,
    }
    expected_provenance = {
        "development_manifest_sha256": data[
            "development_manifest_sha256"
        ],
        "train_file_sha256": data["train_file_sha256"],
        "train_content_sha256": data["train_content_sha256"],
        "validation_file_sha256": data["validation_file_sha256"],
        "validation_content_sha256": data[
            "validation_content_sha256"
        ],
        "sampling_plan_sha256": data["sampling_plan_sha256"],
    }
    expected_candidates = [
        candidate_id for candidate_id, _ in EXPECTED_CANDIDATES
    ]
    if (
        authorization.get("schema_version")
        != "cof-seqgen-v3-execution-authorization-v1"
        or authorization.get("status") != "AUTHORIZED"
        or authorization.get("source_commit") != source_commit
        or authorization.get("runner_relevant_source_sha256")
        != runner_relevant_source_sha256
        or authorization.get("runner_config_sha256")
        != plan.runner_config_sha256
        or authorization.get("source_config_sha256")
        != plan.raw["source_config_sha256"]
        or authorization.get("approved_candidates")
        != expected_candidates
        or authorization.get("provenance") != expected_provenance
        or authorization.get("scope") != expected_scope
    ):
        raise V3ContractError("v3 execution authorization mismatch")
    return {
        "status": "AUTHORIZED",
        "approved_candidates": expected_candidates,
        "test_split_access": False,
        "aggregate_selection": False,
    }


def _current_head(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V3ContractError(f"cannot read v3 JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise V3ContractError(f"v3 JSON is not an object: {path}")
    return value


def _torch_bytes(value: Mapping[str, Any]) -> bytes:
    buffer = BytesIO()
    torch.save(dict(value), buffer)
    return buffer.getvalue()


def _sample_bytes(sample: Any) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        x_num=sample.x_num,
        dt_bin=sample.dt_bin,
        x_cat=sample.x_cat,
        valid_mask=sample.valid_mask,
        y_entity=sample.y_entity,
        lengths=sample.lengths,
    )
    return buffer.getvalue()


def _authorized_candidate_child(
    *,
    plan: V3ExecutionPlan,
    operation: V3ExecutionOperation,
    train_context: Any,
    store: V3AttemptStore,
    device: str,
    source_commit: str,
    authorization_sha256: str,
    event_queue: Any | None = None,
) -> Mapping[str, Any]:
    """Run inside the runner-owned child after all authorization checks."""

    torch.cuda.set_device(torch.device(device))

    adapter = build_v3_candidate_adapter(
        plan=plan,
        operation=operation,
        sampling_plan=train_context.plan,
        device=device,
    )
    if event_queue is not None:
        event_queue.put(
            {
                "kind": "setup_complete",
                "candidate_id": operation.candidate_id,
                "architecture": operation.architecture,
            }
        )

    def checkpoint_callback(
        step: int,
        state: Mapping[str, Any],
    ) -> None:
        store.write_bytes(
            f"checkpoints/step_{step:06d}.pt",
            _torch_bytes(state),
        )
        store.write_json(
            f"progress/step_{step:06d}.json",
            {
                "step": step,
                "loss": state.get("loss"),
                "elapsed_seconds": state.get("elapsed_seconds"),
                "validation_metric": None,
                "validation_rows_read": 0,
            },
        )
        store.append_jsonl(
            "progress.jsonl",
            {
                "step": step,
                "loss": state.get("loss"),
                "elapsed_seconds": state.get("elapsed_seconds"),
                "validation_metric": None,
                "validation_rows_read": 0,
            },
        )
        if event_queue is not None:
            event_queue.put(
                {
                    "kind": "progress",
                    "step": step,
                    "checkpoint_relative_path": (
                        f"checkpoints/step_{step:06d}.pt"
                    ),
                }
            )

    adapter.checkpoint_callback = checkpoint_callback
    fitted = adapter.fit_train_only(
        train_context.train,
        split="train",
        operation=operation,
    )
    if not isinstance(fitted, V3FittedCandidate):
        raise V3ContractError("v3 adapter returned an invalid checkpoint")
    if event_queue is not None:
        event_queue.put(
            {
                "kind": "progress",
                "phase": "training_complete",
                "step": fitted.actual_updates,
            }
        )
    provenance = build_v3_checkpoint_provenance(
        plan=plan,
        operation=operation,
        source_commit=source_commit,
        train_manifest_sha256=train_context.manifest_sha256,
        joint_support_state=fitted.joint_support_state,
        amount_fit_state=fitted.amount_fit_state,
    )
    checkpoint_payload = {
        "schema_version": "cof-seqgen-v3-checkpoint-v1",
        "candidate_id": operation.candidate_id,
        "architecture": operation.architecture,
        "model_state": fitted.model.state_dict(),
        "optimizer_state": fitted.optimizer.state_dict(),
        "actual_updates": fitted.actual_updates,
        "requested_updates": operation.requested_updates,
        "elapsed_seconds": fitted.elapsed_seconds,
        "loss_history": fitted.loss_history,
        "joint_support_state_sha256": fitted.joint_support_state[
            "state_sha256"
        ],
        "amount_fit_state_sha256": fitted.amount_fit_state[
            "state_sha256"
        ],
        "checkpoint_provenance": provenance,
        "numpy_rng_state": fitted.numpy_rng_state,
        "torch_rng_state": fitted.torch_rng_state,
    }
    checkpoint_path = store.write_bytes(
        "checkpoint.pt",
        _torch_bytes(checkpoint_payload),
    )
    store.write_json(
        "checkpoint_provenance.json",
        {
            **provenance,
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
    )
    store.write_json(
        "amount_fit_state.json",
        fitted.amount_fit_state,
    )
    if event_queue is not None:
        event_queue.put(
            {
                "kind": "progress",
                "phase": "final_checkpoint_complete",
                "step": fitted.actual_updates,
            }
        )
    sample = adapter.sample_validation(
        train_context.plan,
        checkpoint=fitted,
        operation=operation,
    )
    sample_path = store.write_bytes("sample.npz", _sample_bytes(sample))
    if event_queue is not None:
        event_queue.put(
            {
                "kind": "progress",
                "phase": "validation_sample_complete",
                "step": fitted.actual_updates,
            }
        )

    # The validation split is opened only after all fitting and generation.
    context = load_development_context(
        repository_root=plan.repository_root,
        manifest_path=(
            plan.repository_root
            / str(
                plan.raw["data_contract"][
                    "development_manifest_path"
                ]
            )
        ),
    )
    if (
        context.train_content_sha256
        != train_context.train_content_sha256
        or context.plan.plan_hash != train_context.plan.plan_hash
    ):
        raise V3ContractError("validation context changed after sampling")
    guard_result = evaluate_validation_guards(
        train=context.train,
        validation=context.validation,
        sample=sample,
        sampling_plan=context.plan,
        tau=context.tau,
        receiver_categories=context.receiver_categories,
        thresholds=plan.source_plan.definition.raw["frozen_contract"][
            "thresholds"
        ],
    )
    if event_queue is not None:
        event_queue.put(
            {
                "kind": "progress",
                "phase": "validation_evaluation_complete",
                "step": fitted.actual_updates,
            }
        )
    all_pass = guard_result["all_five_guards_pass"] is True
    evaluation = {
        **guard_result,
        "status": "VALID" if all_pass else "INVALID",
        "candidate_id": operation.candidate_id,
        "architecture": operation.architecture,
        "validation_sample_sha256": sha256_file(sample_path),
        "authorization_sha256": authorization_sha256,
        "selection_executed": False,
        "test_split_read": False,
    }
    store.write_json("evaluation.json", evaluation)
    store.write_json(
        "metrics.json",
        {
            "candidate_id": operation.candidate_id,
            "statistics": guard_result["statistics"],
            "thresholds": guard_result["thresholds"],
            "checks": guard_result["checks"],
            "all_five_guards_pass": all_pass,
        },
    )
    store.write_json(
        "runtime.json",
        {
            "status": "COMPLETE" if all_pass else "INVALID",
            "requested_updates": operation.requested_updates,
            "actual_updates": fitted.actual_updates,
            "max_wall_seconds": operation.max_wall_seconds,
            "training_elapsed_seconds": fitted.elapsed_seconds,
            "device": device,
            "peak_gpu_memory_bytes": int(
                torch.cuda.max_memory_allocated(torch.device(device))
            ),
            "selection_executed": False,
            "test_split_read": False,
        },
    )
    return {"terminal_status": "COMPLETE" if all_pass else "INVALID"}


def _authorized_candidate_spawn_target(
    payload: Mapping[str, Any],
    event_queue: Any,
) -> Mapping[str, Any]:
    """Rebuild runtime state in a clean spawned interpreter."""

    config_path = Path(str(payload["runner_config_path"])).resolve()
    plan = build_v3_execution_plan(config_path)
    candidate_id = str(payload["candidate_id"])
    operation = next(
        (
            item
            for item in plan.operations
            if item.candidate_id == candidate_id
        ),
        None,
    )
    if operation is None:
        raise V3ContractError("spawn child candidate is not preregistered")
    source_commit = str(payload["source_commit"])
    authorization_sha256 = str(payload["authorization_sha256"])
    if (
        _current_head(plan.repository_root) != source_commit
        or v3_execution_relevant_source_sha256(plan.repository_root)
        != payload["runner_relevant_source_sha256"]
        or len(authorization_sha256) != 64
    ):
        raise V3ContractError("spawn child source provenance changed")
    store = V3AttemptStore.attach_existing(
        runtime_root=Path(str(payload["runtime_root"])),
        operation=operation,
        ownership_path=Path(str(payload["ownership_path"])),
        attempt_path=Path(str(payload["attempt_path"])),
    )
    manifest = _read_json_mapping(store._target("manifest.json"))
    if (
        manifest.get("source_commit") != source_commit
        or manifest.get("runner_relevant_source_sha256")
        != payload["runner_relevant_source_sha256"]
        or manifest.get("authorization_sha256") != authorization_sha256
    ):
        raise V3ContractError("spawn child attempt provenance changed")
    data = plan.raw["data_contract"]
    train_context = load_train_only_candidate_context(
        repository_root=plan.repository_root,
        development_manifest_path=(
            plan.repository_root
            / str(data["development_manifest_path"])
        ),
    )
    if (
        train_context.train_file_sha256 != data["train_file_sha256"]
        or train_context.train_content_sha256
        != data["train_content_sha256"]
        or train_context.plan.plan_hash != data["sampling_plan_sha256"]
    ):
        raise V3ContractError("spawn child train provenance changed")
    return _authorized_candidate_child(
        plan=plan,
        operation=operation,
        train_context=train_context,
        store=store,
        device=str(payload["device"]),
        source_commit=source_commit,
        authorization_sha256=authorization_sha256,
        event_queue=event_queue,
    )


def run_authorized_v3_candidate(
    *,
    plan: V3ExecutionPlan,
    candidate_id: str,
    device: str,
    authorization_path: Path,
    ownership_id: str,
) -> Mapping[str, Any]:
    """Execute one candidate only after exact hash-bound authorization."""

    operation = next(
        (
            item
            for item in plan.operations
            if item.candidate_id == candidate_id
        ),
        None,
    )
    if operation is None or not device.startswith("cuda:"):
        raise V3ContractError("invalid authorized candidate or CUDA device")
    root = plan.repository_root
    authorization_path = authorization_path.resolve()
    authorization_root = (
        root / "artifacts/benchmark_v3/authorizations"
    ).resolve()
    if authorization_root not in authorization_path.parents:
        raise V3ContractError("authorization path escapes v3 artifact root")
    authorization = _read_json_mapping(authorization_path)
    source_commit = _current_head(root)
    relevant_hash = v3_execution_relevant_source_sha256(root)
    validate_v3_execution_authorization(
        plan=plan,
        authorization=authorization,
        source_commit=source_commit,
        runner_relevant_source_sha256=relevant_hash,
    )
    authorization_sha256 = sha256_file(authorization_path)
    continuations = authorization.get("continuations")
    continuation = (
        continuations.get(operation.candidate_id)
        if isinstance(continuations, Mapping)
        else None
    )
    data = plan.raw["data_contract"]
    train_context = load_train_only_candidate_context(
        repository_root=root,
        development_manifest_path=(
            root / str(data["development_manifest_path"])
        ),
    )
    codec = plan.source_plan.definition.raw["model_contract"]
    support_state = build_train_joint_support_state(
        codec=JointStateCodec(
            codec["gap_bins"],
            codec["receiver_classes"],
        ),
        gap=torch.from_numpy(train_context.train.dt_bin),
        receiver=torch.from_numpy(
            train_context.train.x_cat[..., 0]
        ),
        valid_mask=torch.from_numpy(train_context.train.valid_mask),
        fit_split="train",
        provenance={
            "train_file_sha256": train_context.train_file_sha256,
            "train_content_sha256": train_context.train_content_sha256,
            "sampling_plan_sha256": train_context.plan.plan_hash,
        },
    )
    conditioning = bind_train_only_conditioning_contract(
        plan=train_context.plan,
        expected_plan_sha256=data["sampling_plan_sha256"],
        amount_contract=FROZEN_AMOUNT_CONTRACT,
        fit_split="train",
    )
    manifest_payload = {
        "schema_version": "cof-seqgen-v3-candidate-manifest-v1",
        "candidate_id": operation.candidate_id,
        "architecture": operation.architecture,
        "source_commit": source_commit,
        "runner_relevant_source_sha256": relevant_hash,
        "runner_config_sha256": plan.runner_config_sha256,
        "source_config_sha256": plan.raw["source_config_sha256"],
        "authorization_path": str(
            authorization_path.relative_to(root)
        ),
        "authorization_sha256": authorization_sha256,
        "continuation_sha256": (
            canonical_sha256(continuation)
            if isinstance(continuation, Mapping)
            else None
        ),
        "train_manifest_sha256": train_context.manifest_sha256,
        "train_file_sha256": train_context.train_file_sha256,
        "train_content_sha256": train_context.train_content_sha256,
        "sampling_plan_sha256": train_context.plan.plan_hash,
        "joint_support_state_sha256": support_state["state_sha256"],
        "conditioning_binding_sha256": conditioning["binding_sha256"],
        "amount_contract_sha256": canonical_sha256(
            FROZEN_AMOUNT_CONTRACT
        ),
        "seed": operation.seed,
        "requested_updates": operation.requested_updates,
        "max_wall_seconds": operation.max_wall_seconds,
        "training_split": "train",
        "validation_use": operation.validation_use,
        "validation_rows_used_for_fit": 0,
        "test_rows_used": 0,
        "legacy_independent_heads": "FORBIDDEN",
        "post_hoc_calibration": "FORBIDDEN",
        "selection_executed": False,
    }
    provenance_sha256 = canonical_sha256(manifest_payload)
    store = V3AttemptStore.claim(
        runtime_root=(root / str(plan.raw["future_runtime_root"])),
        operation=operation,
        ownership_id=ownership_id,
        authorization_sha256=authorization_sha256,
        provenance_sha256=provenance_sha256,
        continuation=continuation,
    )
    attempt = store.allocate_attempt(
        manifest={
            **manifest_payload,
            "provenance_sha256": provenance_sha256,
        }
    )
    store.write_json("joint_support_state.json", support_state)
    store.write_json("conditioning_binding.json", conditioning)
    result = run_runner_owned_child(
        child_spec=RunnerChildSpec(
            target=_authorized_candidate_spawn_target,
            payload={
                "runner_config_path": str(plan.runner_config_path),
                "candidate_id": operation.candidate_id,
                "device": device,
                "runtime_root": str(store.runtime_root),
                "ownership_path": str(store.ownership_path),
                "attempt_path": str(attempt),
                "source_commit": source_commit,
                "runner_relevant_source_sha256": relevant_hash,
                "authorization_sha256": authorization_sha256,
            },
        ),
        store=store,
        max_wall_seconds=operation.max_wall_seconds,
        startup_timeout_seconds=DEFAULT_CHILD_STARTUP_TIMEOUT_SECONDS,
        progress_timeout_seconds=DEFAULT_CHILD_PROGRESS_TIMEOUT_SECONDS,
        termination_grace_seconds=float(
            plan.raw["execution_contract"][
                "termination_grace_seconds"
            ]
        ),
    )
    worker_terminal = store.finalize_worker(result)
    return {
        "status": result["status"],
        "candidate_id": operation.candidate_id,
        "attempt_path": str(attempt.relative_to(root)),
        "terminal_sha256": sha256_file(
            attempt / f"{result['status']}.json"
        ),
        "worker_terminal_status": worker_terminal["status"],
        "worker_terminal_sha256": sha256_file(
            store.ownership_path.parent
            / attempt.name
            / f"WORKER_{worker_terminal['status']}.json"
        ),
    }


def execute_candidate_stages(
    *,
    operation: V3ExecutionOperation,
    train_batch: Any,
    validation_batch: Any,
    sampling_plan: Any,
    adapter: Any,
    evaluator: Any,
) -> Mapping[str, Any]:
    """Execute the future train→sample→validation seam.

    Callers must pass an already authorized append-only attempt.  This
    function deliberately receives train and validation as separate named
    inputs so validation can never enter ``fit_train_only``.
    """

    expected = dict(EXPECTED_CANDIDATES)
    if (
        operation.candidate_id not in expected
        or expected[operation.candidate_id] != operation.architecture
        or operation.training_split != "train"
    ):
        raise V3ContractError("v3 candidate fitting is train-only")
    if operation.validation_use != "post_generation_five_guard_only":
        raise V3ContractError("validation use is outside the frozen contract")
    if getattr(adapter, "architecture", None) != operation.architecture:
        raise V3ContractError("candidate adapter architecture mismatch")
    checkpoint = adapter.fit_train_only(
        train_batch,
        split="train",
        operation=operation,
    )
    synthetic = adapter.sample_validation(
        sampling_plan,
        checkpoint=checkpoint,
        operation=operation,
    )
    result = evaluator(
        validation_batch,
        synthetic,
        operation=operation,
    )
    if not isinstance(result, Mapping):
        raise V3ContractError("validation evaluator result must be a mapping")
    return result


def validate_v3_runtime_model(
    *,
    operation: V3ExecutionOperation,
    model: Any,
    enabled_hooks: tuple[str, ...],
) -> Mapping[str, Any]:
    """Verify architecture identity before any optimizer or sample call."""

    denoiser = getattr(model, "denoiser", None)
    if (
        not isinstance(model, CoFSeqGenV3)
        or not isinstance(denoiser, CoFSeqDenoiserV3)
        or hasattr(denoiser, "bin_head")
        or hasattr(denoiser, "cat_heads")
    ):
        raise V3ContractError(
            "legacy independent gap/receiver heads are forbidden"
        )
    if enabled_hooks:
        raise V3ContractError("post-hoc calibration hooks are forbidden")
    expected_path = (
        DirectJointDiscretePath
        if operation.architecture == "direct_joint"
        else FactorizedJointDiscretePath
    )
    if (
        operation.candidate_id not in dict(EXPECTED_CANDIDATES)
        or denoiser.candidate != operation.architecture
        or not isinstance(denoiser.discrete_path, expected_path)
        or model.coherence_lambda != 0.0
        or model.AMOUNT_CONTRACT
        != "train_fitted_centered_empirical_residual_257_v2_8_frozen"
    ):
        raise V3ContractError("v3 runtime architecture path mismatch")
    return {
        "candidate_id": operation.candidate_id,
        "architecture": operation.architecture,
        "legacy_independent_heads_present": False,
        "post_hoc_calibration_hooks": [],
        "coherence_lambda": 0.0,
        "amount_contract": model.AMOUNT_CONTRACT,
    }


def _validated_fit_state(
    state: Mapping[str, Any],
    *,
    role: str,
) -> str:
    if (
        state.get("fit_split") != "train"
        or state.get("validation_rows_used") != 0
        or state.get("test_rows_used") != 0
        or not isinstance(state.get("state_sha256"), str)
    ):
        raise V3ContractError(f"{role} is not train-only")
    payload = {
        key: value
        for key, value in state.items()
        if key != "state_sha256"
    }
    actual = canonical_sha256(payload)
    if actual != state["state_sha256"]:
        raise V3ContractError(f"{role} hash mismatch")
    return actual


def build_v3_checkpoint_provenance(
    *,
    plan: V3ExecutionPlan,
    operation: V3ExecutionOperation,
    source_commit: str,
    train_manifest_sha256: str,
    joint_support_state: Mapping[str, Any],
    amount_fit_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    if (
        len(source_commit) != 40
        or len(train_manifest_sha256) != 64
        or operation not in plan.operations
    ):
        raise V3ContractError("checkpoint identity is invalid")
    support_hash = _validated_fit_state(
        joint_support_state,
        role="joint support state",
    )
    amount_hash = _validated_fit_state(
        amount_fit_state,
        role="amount fit state",
    )
    data = plan.raw["data_contract"]
    support_provenance = joint_support_state.get("provenance")
    if (
        not isinstance(support_provenance, Mapping)
        or support_provenance.get("train_file_sha256")
        != data["train_file_sha256"]
        or support_provenance.get("train_content_sha256")
        != data["train_content_sha256"]
        or support_provenance.get("sampling_plan_sha256")
        != data["sampling_plan_sha256"]
        or amount_fit_state.get("algorithm")
        != "train_fitted_centered_empirical_residual"
        or amount_fit_state.get("quantile_grid_size") != 257
    ):
        raise V3ContractError("checkpoint fit-state provenance mismatch")
    payload = {
        "schema_version": "cof-seqgen-v3-checkpoint-provenance-v1",
        "candidate_id": operation.candidate_id,
        "architecture": operation.architecture,
        "source_commit": source_commit,
        "architecture_source_sha256": plan.raw[
            "source_relevant_sha256"
        ],
        "runner_source_sha256": v3_execution_relevant_source_sha256(
            plan.repository_root
        ),
        "source_config_sha256": plan.raw["source_config_sha256"],
        "runner_config_sha256": plan.runner_config_sha256,
        "train_manifest_sha256": train_manifest_sha256,
        "train_file_sha256": data["train_file_sha256"],
        "train_content_sha256": data["train_content_sha256"],
        "sampling_plan_file_sha256": data[
            "sampling_plan_file_sha256"
        ],
        "sampling_plan_sha256": data["sampling_plan_sha256"],
        "joint_support_state_sha256": support_hash,
        "amount_fit_state_sha256": amount_hash,
        "seed": operation.seed,
        "requested_updates": operation.requested_updates,
        "max_wall_seconds": operation.max_wall_seconds,
    }
    return {
        **payload,
        "provenance_sha256": canonical_sha256(payload),
    }


def validate_v3_checkpoint_provenance(
    *,
    provenance: Mapping[str, Any],
    plan: V3ExecutionPlan,
    operation: V3ExecutionOperation,
    source_commit: str,
    train_manifest_sha256: str,
    joint_support_state: Mapping[str, Any],
    amount_fit_state: Mapping[str, Any],
) -> None:
    rebuilt = build_v3_checkpoint_provenance(
        plan=plan,
        operation=operation,
        source_commit=source_commit,
        train_manifest_sha256=train_manifest_sha256,
        joint_support_state=joint_support_state,
        amount_fit_state=amount_fit_state,
    )
    if canonical_sha256(provenance) != canonical_sha256(rebuilt):
        raise V3ContractError("checkpoint provenance mismatch")


def build_v3_aggregate_input_manifest(
    records: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Validate future aggregate inputs without performing selection.

    This is intentionally a schema seam only.  Candidate execution may emit
    these hash-bound records, but aggregate selection remains forbidden in
    this source-preparation phase.
    """

    expected = {candidate for candidate, _ in EXPECTED_CANDIDATES}
    if set(records) != expected:
        raise V3ContractError("both v3 candidates are required for aggregate")
    expected_checks = {
        "amount_ks",
        "gap_ks",
        "amount_abs_standardized_label_effect",
        "gap_abs_standardized_label_effect",
        "receiver_max_abs_signed_frequency",
    }
    candidate_status: dict[str, str] = {}
    eligible: list[str] = []
    input_sha256: dict[str, str] = {}
    for candidate_id, _architecture in EXPECTED_CANDIDATES:
        record = records[candidate_id]
        terminal = record.get("terminal")
        evaluation = record.get("evaluation")
        if not isinstance(terminal, Mapping) or not isinstance(
            evaluation, Mapping
        ):
            raise V3ContractError("aggregate candidate record is incomplete")
        artifact_hashes = terminal.get("artifact_sha256")
        if (
            terminal.get("status") not in V3AttemptStore.TERMINALS
            or not isinstance(artifact_hashes, Mapping)
            or set(artifact_hashes) != set(V3AttemptStore.REQUIRED_ARTIFACTS)
            or any(
                not isinstance(value, str) or len(value) != 64
                for value in artifact_hashes.values()
            )
        ):
            raise V3ContractError("aggregate terminal provenance is invalid")
        checks = evaluation.get("checks")
        if not isinstance(checks, Mapping) or set(checks) != expected_checks:
            raise V3ContractError("aggregate five-guard evidence is invalid")
        all_pass = (
            terminal.get("status") == "COMPLETE"
            and terminal.get("aggregate_eligible") is True
            and evaluation.get("status") == "VALID"
            and evaluation.get("all_five_guards_pass") is True
            and all(checks[key] == "PASS" for key in expected_checks)
        )
        if bool(terminal.get("aggregate_eligible")) != all_pass:
            raise V3ContractError(
                "aggregate eligibility disagrees with all-five evidence"
            )
        candidate_status[candidate_id] = (
            "ELIGIBLE" if all_pass else "INELIGIBLE"
        )
        if all_pass:
            eligible.append(candidate_id)
        input_sha256[candidate_id] = canonical_sha256(record)
    return {
        "schema_version": "cof-seqgen-v3-aggregate-input-v1",
        "candidate_status": candidate_status,
        "eligible_candidates": eligible,
        "candidate_record_sha256": input_sha256,
        "selection_executed": False,
        "aggregate_authorized": False,
    }


def _spawn_runner_child_entry(
    child_spec: RunnerChildSpec,
    event_queue: Any,
) -> None:
    """Run a spawn-safe child and report lifecycle events to its parent."""

    event_queue.put({"kind": "child_bootstrap"})
    try:
        value = child_spec.target(child_spec.payload, event_queue)
    except (KeyboardInterrupt, SystemExit) as error:
        event_queue.put(
            {
                "kind": "interrupted",
                "exception": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
    except Exception as error:
        event_queue.put(
            {
                "kind": "exception",
                "exception": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
    else:
        event_queue.put({"kind": "complete", "value": value})


def _terminate_runner_owned_process(
    process: Any,
    *,
    termination_grace_seconds: float,
) -> None:
    """Bound cleanup to the child created by this runner."""

    if not process.is_alive():
        process.join(timeout=termination_grace_seconds)
        return
    process.terminate()
    process.join(timeout=termination_grace_seconds)
    if process.is_alive():
        process.kill()
        process.join(timeout=termination_grace_seconds)


def _finalize_supervision_failure(
    *,
    store: V3AttemptStore,
    process: Any,
    started: float,
    failure_class: str,
    setup_signal_received: bool,
    metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    elapsed = time.monotonic() - started
    result = {
        "failure_class": failure_class,
        "actual_elapsed_seconds": elapsed,
        "child_pid": process.pid,
        "runner_owned_child_only": True,
        "setup_signal_received": setup_signal_received,
    }
    if metadata:
        result.update(dict(metadata))
    _write_failure_placeholders(
        store,
        elapsed_seconds=elapsed,
        failure_class=failure_class,
    )
    return store.finalize("FAILED", metadata=result)


def _write_failure_placeholders(
    store: V3AttemptStore,
    *,
    elapsed_seconds: float,
    failure_class: str,
) -> None:
    values: tuple[tuple[str, Mapping[str, Any] | bytes], ...] = (
        (
            "runtime.json",
            {
                "status": "FAILED",
                "elapsed_seconds": elapsed_seconds,
                "failure_class": failure_class,
            },
        ),
        (
            "checkpoint.pt",
            b"COF_V3_NO_CHECKPOINT_DUE_TO_FAILURE\n",
        ),
        (
            "checkpoint_provenance.json",
            {
                "status": "UNAVAILABLE",
                "failure_class": failure_class,
            },
        ),
        (
            "joint_support_state.json",
            {
                "status": "UNAVAILABLE",
                "failure_class": failure_class,
            },
        ),
        (
            "amount_fit_state.json",
            {
                "status": "UNAVAILABLE",
                "failure_class": failure_class,
            },
        ),
        (
            "conditioning_binding.json",
            {
                "status": "UNAVAILABLE",
                "failure_class": failure_class,
            },
        ),
        ("sample.npz", b"COF_V3_NO_SAMPLE_DUE_TO_FAILURE\n"),
        (
            "metrics.json",
            {
                "status": "NOT_COMPUTED",
                "all_five_guards_pass": False,
            },
        ),
        (
            "evaluation.json",
            {
                "status": "NOT_COMPUTED",
                "all_five_guards_pass": False,
                "checks": {},
            },
        ),
    )
    for relative, value in values:
        if store._target(relative).exists():
            continue
        if isinstance(value, bytes):
            store.write_bytes(relative, value)
        else:
            store.write_json(relative, value)


def run_runner_owned_child(
    *,
    child_spec: RunnerChildSpec,
    store: V3AttemptStore,
    max_wall_seconds: float,
    startup_timeout_seconds: float | None = None,
    progress_timeout_seconds: float | None = None,
    termination_grace_seconds: float,
) -> Mapping[str, Any]:
    """Run and supervise only the child process created by this runner."""

    if max_wall_seconds <= 0 or termination_grace_seconds <= 0:
        raise V3ContractError("wall cap and termination grace must be positive")
    if startup_timeout_seconds is None or startup_timeout_seconds <= 0:
        raise V3ContractError(
            "spawn child requires a positive startup timeout"
        )
    if progress_timeout_seconds is not None and progress_timeout_seconds <= 0:
        raise V3ContractError("progress timeout must be positive")
    context = multiprocessing.get_context("spawn")
    event_queue = context.Queue()
    process = context.Process(
        target=_spawn_runner_child_entry,
        args=(child_spec, event_queue),
    )
    started = time.monotonic()
    try:
        process.start()
    except Exception as error:
        event_queue.cancel_join_thread()
        event_queue.close()
        return _finalize_supervision_failure(
            store=store,
            process=process,
            started=started,
            failure_class="child_start_error",
            setup_signal_received=False,
            metadata={
                "exception": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
    setup_received = False
    last_activity = started
    startup_deadline = started + startup_timeout_seconds
    wall_deadline = started + max_wall_seconds
    terminal_event: Mapping[str, Any] | None = None
    while terminal_event is None:
        now = time.monotonic()
        if not setup_received and now >= startup_deadline:
            _terminate_runner_owned_process(
                process,
                termination_grace_seconds=termination_grace_seconds,
            )
            event_queue.cancel_join_thread()
            event_queue.close()
            return _finalize_supervision_failure(
                store=store,
                process=process,
                started=started,
                failure_class="startup_timeout",
                setup_signal_received=False,
            )
        if now >= wall_deadline:
            _terminate_runner_owned_process(
                process,
                termination_grace_seconds=termination_grace_seconds,
            )
            event_queue.cancel_join_thread()
            event_queue.close()
            return _finalize_supervision_failure(
                store=store,
                process=process,
                started=started,
                failure_class="wall_cap",
                setup_signal_received=setup_received,
            )
        if (
            setup_received
            and progress_timeout_seconds is not None
            and now - last_activity >= progress_timeout_seconds
        ):
            _terminate_runner_owned_process(
                process,
                termination_grace_seconds=termination_grace_seconds,
            )
            event_queue.cancel_join_thread()
            event_queue.close()
            return _finalize_supervision_failure(
                store=store,
                process=process,
                started=started,
                failure_class="progress_timeout",
                setup_signal_received=True,
            )
        wait_seconds = min(
            0.05,
            wall_deadline - now,
            startup_deadline - now if not setup_received else 0.05,
        )
        try:
            event = event_queue.get(timeout=max(wait_seconds, 0.001))
        except queue.Empty:
            if not process.is_alive():
                terminal_event = {
                    "kind": "exception",
                    "exception": (
                        f"child exited {process.exitcode} without terminal metadata"
                    ),
                    "traceback": "",
                }
            continue
        kind = event.get("kind")
        if kind == "setup_complete":
            setup_received = True
            last_activity = time.monotonic()
        elif kind == "progress":
            if not setup_received:
                terminal_event = {
                    "kind": "exception",
                    "exception": "progress arrived before setup_complete",
                    "traceback": "",
                }
            else:
                last_activity = time.monotonic()
        elif kind in {"complete", "exception", "interrupted"}:
            terminal_event = event

    process.join(timeout=termination_grace_seconds)
    if process.is_alive():
        _terminate_runner_owned_process(
            process,
            termination_grace_seconds=termination_grace_seconds,
        )
    event_queue.cancel_join_thread()
    event_queue.close()
    elapsed = time.monotonic() - started
    kind = terminal_event.get("kind")
    if kind == "complete" and not setup_received:
        terminal_event = {
            "kind": "exception",
            "exception": "child completed without setup_complete",
            "traceback": "",
        }
        kind = "exception"
    if kind != "complete":
        failure_class = (
            "operator_interrupt"
            if kind == "interrupted"
            else "child_exception"
        )
        return _finalize_supervision_failure(
            store=store,
            process=process,
            started=started,
            failure_class=failure_class,
            setup_signal_received=setup_received,
            metadata={
                "exception": terminal_event.get("exception"),
                "traceback": terminal_event.get("traceback"),
            },
        )
    value = terminal_event.get("value")
    if not isinstance(value, Mapping):
        raise V3ContractError("child result must be terminal metadata")
    terminal_status = str(value.get("terminal_status", ""))
    if terminal_status not in V3AttemptStore.TERMINALS:
        raise V3ContractError("child omitted a valid terminal status")
    return store.finalize(
        terminal_status,
        metadata={
            "actual_elapsed_seconds": elapsed,
            "child_pid": process.pid,
            "runner_owned_child_only": True,
            "setup_signal_received": setup_received,
        },
    )
