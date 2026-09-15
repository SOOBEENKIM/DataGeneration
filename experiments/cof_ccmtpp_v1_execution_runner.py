"""Fail-closed future execution contracts for ``cof_ccmtpp_v1``.

The plan and dry-run paths are read-only.  Execute validates a separately
created authorization before invoking any injected data, model, or device
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import subprocess
import time
import traceback
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from eval.cof_ccmtpp_v1_contract import (
    CCMTPPContractError,
    load_ccmtpp_definition,
    sha256_file,
)
ZERO_EXECUTION_COUNTS = {
    "authorization_creations": 0,
    "data_body_reads": 0,
    "model_imports_or_builds": 0,
    "gpu_queries": 0,
    "cuda_calls": 0,
    "fit_calls": 0,
    "sample_calls": 0,
    "evaluation_calls": 0,
    "internal_test_reads": 0,
    "sparkov_fraud_test_reads": 0,
}
AMOUNT_CONTRACT_NAME = "frozen_non_v3_train_only_encode_inverse_decode_v1"
AUTHORIZATION_SCOPE = {
    "train_body_read": True,
    "validation_body_read_after_fit": True,
    "model_build": True,
    "cuda_model_execution": True,
    "fit": True,
    "validation_sample": True,
    "validation_evaluation": True,
    "append_only_checkpoint_and_artifact_write": True,
    "gpu_inventory_query": False,
    "authorization_creation": False,
    "candidate_sweep": False,
    "hyperparameter_change": False,
    "threshold_change": False,
    "internal_test": False,
    "sparkov_fraud_test": False,
    "tstr": False,
    "privacy": False,
    "full_run": False,
}
EXECUTION_RELEVANT_SOURCE_PATHS = (
    "configs/benchmark_v2/cof_ccmtpp_v1_source_only.yaml",
    "configs/benchmark_v2/cof_ccmtpp_v1_execution_runner.yaml",
    "models/cof_ccmtpp_v1.py",
    "eval/cof_ccmtpp_v1_contract.py",
    "experiments/cof_ccmtpp_v1_execution_runner.py",
    "generators/cof_ccmtpp_v1_execution_backend.py",
    "scripts/run_cof_ccmtpp_v1.py",
)


@dataclass(frozen=True)
class ExecutionJob:
    dataset: str
    candidate_id: str
    seed: int
    attempt: str


@dataclass(frozen=True)
class ExecutionPlan:
    repository_root: Path
    config_path: Path
    config_sha256: str
    source_definition: Any
    candidate_id: str
    dataset_scope: str
    jobs: tuple[ExecutionJob, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionDependencies:
    load_data_body: Callable[..., Any]
    build_model: Callable[..., Any]
    query_device: Callable[..., Any]
    run_job: Callable[..., Any]


@dataclass(frozen=True)
class RunnerChildSpec:
    target: Callable[[Mapping[str, Any], Any], Mapping[str, Any]]
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class TrainOnlyStates:
    binding: Mapping[str, Any]
    receiver_vocabulary: Mapping[str, Any]
    hierarchy: Any
    gap_support: Mapping[str, Any]
    amount_transform: Mapping[str, Any]
    continuous_gap: np.ndarray
    valid_mask: np.ndarray


class CCMTPPAttemptStore:
    """Exclusive ownership and append-only candidate artifact writer."""

    TERMINALS = ("COMPLETE", "INVALID", "FAILED")
    COMPLETE_REQUIRED = (
        "manifest.json",
        "progress.jsonl",
        "receiver_vocabulary_state.json",
        "receiver_hierarchy_state.json",
        "gap_support_state.json",
        "amount_transform_state.json",
        "conditioning_plan.json",
        "checkpoints/final.pt",
        "checkpoint_provenance.json",
        "validation_sample.npz",
        "diagnostics.json",
        "metrics.json",
        "evaluation.json",
        "runtime.json",
        "gate_decision.json",
    )

    def __init__(
        self,
        *,
        runtime_root: Path,
        job: ExecutionJob,
        ownership_path: Path,
    ) -> None:
        self.runtime_root = runtime_root.resolve()
        self.job = job
        self.ownership_path = ownership_path.resolve()
        self.attempt_path: Path | None = None

    @staticmethod
    def _exclusive_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
            )
        except FileExistsError as error:
            raise CCMTPPContractError(f"append-only artifact exists: {path}") from error
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    @classmethod
    def claim(
        cls,
        *,
        runtime_root: Path,
        job: ExecutionJob,
        authorization_sha256: str,
        provenance_sha256: str,
    ) -> "CCMTPPAttemptStore":
        if len(authorization_sha256) != 64 or len(provenance_sha256) != 64:
            raise CCMTPPContractError("artifact ownership hash is invalid")
        runtime_root = runtime_root.resolve()
        ownership = (
            runtime_root / "workers" / job.dataset / job.candidate_id / "ownership.lock"
        )
        value = {
            "schema_version": "cof-ccmtpp-v1-ownership-v1",
            "dataset": job.dataset,
            "candidate_id": job.candidate_id,
            "seed": job.seed,
            "attempt": job.attempt,
            "authorization_sha256": authorization_sha256,
            "provenance_sha256": provenance_sha256,
        }
        try:
            cls._exclusive_bytes(
                ownership,
                (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                ),
            )
        except CCMTPPContractError as error:
            raise CCMTPPContractError("candidate ownership already exists") from error
        return cls(runtime_root=runtime_root, job=job, ownership_path=ownership)

    def allocate_attempt(self, manifest: Mapping[str, Any]) -> Path:
        if self.attempt_path is not None:
            raise CCMTPPContractError("attempt is already allocated")
        if (
            manifest.get("candidate_id") != self.job.candidate_id
            or manifest.get("dataset") != self.job.dataset
            or manifest.get("seed") != self.job.seed
        ):
            raise CCMTPPContractError("attempt manifest identity mismatch")
        attempt = (
            self.runtime_root
            / self.job.dataset
            / self.job.candidate_id
            / f"seed_{self.job.seed}"
            / self.job.attempt
        )
        try:
            attempt.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise CCMTPPContractError("append-only attempt already exists") from error
        self.attempt_path = attempt.resolve()
        self.write_json("manifest.json", manifest)
        return self.attempt_path

    @classmethod
    def attach_existing(
        cls,
        *,
        runtime_root: Path,
        job: ExecutionJob,
        ownership_path: Path,
        attempt_path: Path,
    ) -> "CCMTPPAttemptStore":
        """Attach a spawned child only to the parent's already-owned attempt."""

        store = cls(
            runtime_root=runtime_root,
            job=job,
            ownership_path=ownership_path,
        )
        expected_ownership = (
            store.runtime_root
            / "workers"
            / job.dataset
            / job.candidate_id
            / "ownership.lock"
        ).resolve()
        expected_attempt = (
            store.runtime_root
            / job.dataset
            / job.candidate_id
            / f"seed_{job.seed}"
            / job.attempt
        ).resolve()
        if (
            store.ownership_path != expected_ownership
            or attempt_path.resolve() != expected_attempt
            or not expected_ownership.is_file()
            or not (expected_attempt / "manifest.json").is_file()
        ):
            raise CCMTPPContractError("spawn child attempt ownership mismatch")
        store.attempt_path = expected_attempt
        return store

    def _target(self, relative: str) -> Path:
        if self.attempt_path is None:
            raise CCMTPPContractError("attempt has not been allocated")
        if any((self.attempt_path / f"{name}.json").exists() for name in self.TERMINALS):
            raise CCMTPPContractError("terminal attempt is immutable")
        target = (self.attempt_path / relative).resolve()
        try:
            target.relative_to(self.attempt_path)
        except ValueError as error:
            raise CCMTPPContractError("artifact path escapes owned attempt") from error
        return target

    def write_json(self, relative: str, value: Mapping[str, Any]) -> Path:
        payload = (
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        target = self._target(relative)
        self._exclusive_bytes(target, payload)
        return target

    def write_bytes(self, relative: str, value: bytes) -> Path:
        target = self._target(relative)
        self._exclusive_bytes(target, value)
        return target

    def append_progress(self, value: Mapping[str, Any]) -> Path:
        target = self._target("progress.jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        with os.fdopen(descriptor, "ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return target

    @staticmethod
    def _validate_diagnostics(value: Mapping[str, Any]) -> None:
        receiver = value.get("receiver")
        fidelity = value.get("fidelity")
        required_receiver = {
            "overall_nll", "head_nll", "tail_nll", "unk_nll",
            "repeat_nll", "new_nll", "head_count", "tail_count",
            "unk_count", "repeat_count", "new_count",
        }
        required_fidelity = {
            "full_receiver_tv", "head_receiver_tv", "tail_receiver_tv",
            "unk_rate_error",
        }
        if (
            not isinstance(receiver, Mapping)
            or not required_receiver.issubset(receiver)
            or not isinstance(fidelity, Mapping)
            or not required_fidelity.issubset(fidelity)
        ):
            raise CCMTPPContractError("receiver diagnostic schema is incomplete")

    def _write_index_and_checksums(self) -> tuple[Path, Path]:
        assert self.attempt_path is not None
        files = sorted(
            path.relative_to(self.attempt_path).as_posix()
            for path in self.attempt_path.rglob("*")
            if path.is_file()
            and path.name not in {"artifact_index.json", "checksum_manifest.json"}
            and path.stem not in self.TERMINALS
        )
        index = self.write_json(
            "artifact_index.json",
            {
                "schema_version": "cof-ccmtpp-v1-artifact-index-v1",
                "files": files,
            },
        )
        checksums = {
            relative: sha256_file(self.attempt_path / relative) for relative in files
        }
        checksums["artifact_index.json"] = sha256_file(index)
        checksum = self.write_json(
            "checksum_manifest.json",
            {
                "schema_version": "cof-ccmtpp-v1-checksum-manifest-v1",
                "files": checksums,
            },
        )
        return index, checksum

    def finalize(
        self, status: str, *, metadata: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        if status not in self.TERMINALS or self.attempt_path is None:
            raise CCMTPPContractError("terminal status is invalid")
        if any((self.attempt_path / f"{name}.json").exists() for name in self.TERMINALS):
            raise CCMTPPContractError("terminal marker already exists")
        if status in {"COMPLETE", "INVALID"}:
            missing = [
                relative
                for relative in self.COMPLETE_REQUIRED
                if not (self.attempt_path / relative).is_file()
            ]
            if missing:
                raise CCMTPPContractError(f"terminal artifacts are missing: {missing}")
            diagnostics = _read_json(self.attempt_path / "diagnostics.json")
            self._validate_diagnostics(diagnostics)
            evaluation = _read_json(self.attempt_path / "evaluation.json")
            gate = _read_json(self.attempt_path / "gate_decision.json")
            if (
                (status == "COMPLETE" and evaluation.get("status") != "VALID")
                or (status == "INVALID" and evaluation.get("status") != "INVALID")
                or gate.get("candidate_id") != self.job.candidate_id
                or gate.get("status") not in {"PASS", "FAIL", "NOT_EVALUABLE"}
            ):
                raise CCMTPPContractError("evaluation/gate terminal contract mismatch")
        else:
            for relative, value in (
                ("runtime.json", {"status": "FAILED"}),
                ("diagnostics.json", {"status": "FAILED", "receiver": {}, "fidelity": {}}),
            ):
                if not (self.attempt_path / relative).exists():
                    self.write_json(relative, value)
        index, checksum = self._write_index_and_checksums()
        terminal = {
            "schema_version": "cof-ccmtpp-v1-terminal-v1",
            "status": status,
            "dataset": self.job.dataset,
            "candidate_id": self.job.candidate_id,
            "seed": self.job.seed,
            "attempt": self.job.attempt,
            "artifact_index_sha256": sha256_file(index),
            "checksum_manifest_sha256": sha256_file(checksum),
        }
        if metadata:
            if set(terminal) & set(metadata):
                raise CCMTPPContractError("terminal metadata overwrites identity")
            terminal.update(dict(metadata))
        terminal_path = self.attempt_path / f"{status}.json"
        self._exclusive_bytes(
            terminal_path,
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

    def finalize_worker(
        self, candidate_terminal: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if self.attempt_path is None:
            raise CCMTPPContractError("worker attempt is not allocated")
        status = str(candidate_terminal.get("status", ""))
        if status not in self.TERMINALS:
            raise CCMTPPContractError("candidate terminal is invalid")
        terminal_path = self.attempt_path / f"{status}.json"
        if (
            not terminal_path.is_file()
            or sha256_file(terminal_path)
            != hashlib.sha256(terminal_path.read_bytes()).hexdigest()
        ):
            raise CCMTPPContractError("candidate terminal is not stable")
        worker_status = "FAILED" if status == "FAILED" else "COMPLETE"
        worker_path = (
            self.ownership_path.parent
            / self.job.attempt
            / f"WORKER_{worker_status}.json"
        )
        value = {
            "schema_version": "cof-ccmtpp-v1-worker-terminal-v1",
            "status": worker_status,
            "dataset": self.job.dataset,
            "candidate_id": self.job.candidate_id,
            "attempt": self.job.attempt,
            "candidate_status": status,
            "candidate_terminal_path": str(terminal_path),
            "candidate_terminal_sha256": sha256_file(terminal_path),
            "ownership_sha256": sha256_file(self.ownership_path),
        }
        self._exclusive_bytes(
            worker_path,
            (
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8"),
        )
        return value


def _runner_child_entry(
    target: Callable[[Mapping[str, Any], Any], Mapping[str, Any]],
    payload: Mapping[str, Any],
    event_queue: Any,
) -> None:
    try:
        # A spawned interpreter must import this module before it can invoke the
        # scientific setup target.  Tell the parent when that bootstrap has
        # completed so import latency is not misclassified as a setup deadlock.
        # The bootstrap itself remains bounded by the whole-job wall cap.
        event_queue.put({"kind": "child_started"})
        result = target(payload, event_queue)
        if not isinstance(result, Mapping):
            raise TypeError("runner child result must be a mapping")
        event_queue.put({"kind": "returned", "result": dict(result)})
    except (KeyboardInterrupt, SystemExit) as error:
        event_queue.put(
            {
                "kind": "interrupted",
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    except BaseException as error:
        event_queue.put(
            {
                "kind": "exception",
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )


def _bounded_child_cleanup(
    process: Any,
    *,
    terminate: bool,
    termination_grace_seconds: float,
    kill_grace_seconds: float,
) -> Mapping[str, Any]:
    terminated = False
    killed = False
    if terminate and process.is_alive():
        process.terminate()
        terminated = True
    process.join(timeout=max(termination_grace_seconds, 0.0))
    if process.is_alive():
        process.kill()
        killed = True
        process.join(timeout=max(kill_grace_seconds, 0.0))
    return {
        "child_terminated": terminated,
        "child_killed": killed,
        "child_exitcode": process.exitcode,
        "child_alive_after_cleanup": process.is_alive(),
    }


def run_runner_owned_child(
    *,
    child_spec: RunnerChildSpec,
    store: CCMTPPAttemptStore,
    max_wall_seconds: float,
    startup_timeout_seconds: float,
    progress_timeout_seconds: float,
    queue_poll_seconds: float,
    termination_grace_seconds: float,
    kill_grace_seconds: float,
) -> Mapping[str, Any]:
    """Queue-first spawn monitor with bounded cleanup and parent finalization."""

    if store.attempt_path is None or min(
        max_wall_seconds,
        startup_timeout_seconds,
        progress_timeout_seconds,
        queue_poll_seconds,
    ) <= 0:
        raise CCMTPPContractError("child watchdog configuration is invalid")
    context = multiprocessing.get_context("spawn")
    event_queue = context.Queue()
    process = context.Process(
        target=_runner_child_entry,
        args=(child_spec.target, dict(child_spec.payload), event_queue),
    )
    started = time.monotonic()
    process.start()
    child_started = False
    setup_started: float | None = None
    setup_complete = False
    last_progress: float | None = None
    result: Mapping[str, Any] | None = None
    failure_class: str | None = None
    failure_event: Mapping[str, Any] = {}
    try:
        while True:
            now = time.monotonic()
            if now - started >= max_wall_seconds:
                failure_class = "wall_cap"
                break
            if (
                child_started
                and not setup_complete
                and setup_started is not None
                and now - setup_started >= startup_timeout_seconds
            ):
                failure_class = "startup_watchdog"
                break
            if (
                setup_complete
                and last_progress is not None
                and now - last_progress >= progress_timeout_seconds
            ):
                failure_class = "progress_watchdog"
                break
            try:
                event = event_queue.get(timeout=queue_poll_seconds)
            except queue.Empty:
                if not process.is_alive():
                    try:
                        event = event_queue.get_nowait()
                    except queue.Empty:
                        failure_class = "unexpected_child_exit"
                        break
                    else:
                        pass
                else:
                    continue
            if not isinstance(event, Mapping):
                failure_class = "invalid_child_event"
                break
            kind = event.get("kind")
            if kind == "child_started":
                if child_started or setup_complete:
                    failure_class = "duplicate_child_started"
                    break
                child_started = True
                setup_started = time.monotonic()
            elif kind == "setup_complete":
                if not child_started:
                    failure_class = "setup_before_child_started"
                    break
                setup_complete = True
                last_progress = time.monotonic()
            elif kind == "progress":
                if not setup_complete:
                    failure_class = "progress_before_setup"
                    break
                last_progress = time.monotonic()
            elif kind == "returned":
                result = event.get("result")
                break
            elif kind == "exception":
                failure_class = "child_exception"
                failure_event = dict(event)
                break
            elif kind == "interrupted":
                failure_class = "child_interrupted"
                failure_event = dict(event)
                break
            else:
                failure_class = "invalid_child_event"
                failure_event = dict(event)
                break
    except KeyboardInterrupt:
        cleanup = _bounded_child_cleanup(
            process,
            terminate=True,
            termination_grace_seconds=termination_grace_seconds,
            kill_grace_seconds=kill_grace_seconds,
        )
        store.append_progress(
            {"status": "INTERRUPTED", "elapsed_seconds": time.monotonic() - started}
        )
        terminal = store.finalize(
            "FAILED",
            metadata={"failure_class": "parent_interrupt", **cleanup},
        )
        store.finalize_worker(terminal)
        raise
    if failure_class is not None:
        cleanup = _bounded_child_cleanup(
            process,
            terminate=True,
            termination_grace_seconds=termination_grace_seconds,
            kill_grace_seconds=kill_grace_seconds,
        )
        store.append_progress(
            {
                "status": "FAILED",
                "failure_class": failure_class,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        terminal = store.finalize(
            "FAILED",
            metadata={
                "failure_class": failure_class,
                "failure_event": dict(failure_event),
                "elapsed_seconds": time.monotonic() - started,
                **cleanup,
            },
        )
        store.finalize_worker(terminal)
        return terminal
    cleanup = _bounded_child_cleanup(
        process,
        terminate=False,
        termination_grace_seconds=termination_grace_seconds,
        kill_grace_seconds=kill_grace_seconds,
    )
    if cleanup["child_alive_after_cleanup"] or cleanup["child_exitcode"] not in {0, None}:
        terminal = store.finalize(
            "FAILED",
            metadata={"failure_class": "child_cleanup", **cleanup},
        )
        store.finalize_worker(terminal)
        return terminal
    if not isinstance(result, Mapping) or result.get("terminal_status") not in {
        "COMPLETE", "INVALID"
    }:
        terminal = store.finalize(
            "FAILED", metadata={"failure_class": "invalid_child_result", **cleanup}
        )
        store.finalize_worker(terminal)
        return terminal
    terminal = store.finalize(str(result["terminal_status"]))
    store.finalize_worker(terminal)
    return terminal


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_source_commit(repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CCMTPPContractError("cannot resolve source commit") from error
    value = completed.stdout.strip()
    if len(value) != 40:
        raise CCMTPPContractError("source commit is invalid")
    return value


def execution_relevant_source_sha256(repository_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in EXECUTION_RELEVANT_SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise CCMTPPContractError(f"runner source is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CCMTPPContractError(f"cannot read runner config: {path}") from error
    if not isinstance(value, Mapping):
        raise CCMTPPContractError("runner config root must be a mapping")
    return value


def _validate_runner_config(path: Path, raw: Mapping[str, Any]) -> Any:
    if (
        raw.get("schema_version") != "cof-ccmtpp-v1-execution-runner-v1"
        or raw.get("mode") != "SOURCE_ONLY_RUNNER_PREPARATION"
        or raw.get("family") != "cof_ccmtpp_v1"
        or raw.get("candidate_chain") != ["C1", "C2", "C3", "C4"]
        or raw.get("initial_candidate") != "C1"
        or raw.get("seed") != 4001
        or raw.get("attempt") != "attempt_001"
        or raw.get("worker_scope") != {
            "datasets": ["amlsim", "sparkov"],
            "jobs_per_process": 1,
            "authorization_datasets": 1,
            "visible_devices_per_process": 1,
        }
        or set(raw.get("datasets", {})) != {"amlsim", "sparkov"}
        or any(value is not False for value in raw.get("source_only", {}).values())
        or raw.get("execution") != {
            "explicit_device": "cuda:0",
            "gpu_inventory_query": "FORBIDDEN",
            "validation_sample_batch_size": 128,
            "progress_interval_updates": 100,
            "checkpoint_serialization": "cpu_first",
            "validation_use": "post_fit_generation_and_evaluation_only",
            "internal_test": "FORBIDDEN",
            "sparkov_fraud_test": "FORBIDDEN",
        }
    ):
        raise CCMTPPContractError("runner source-only contract changed")
    root = path.resolve().parents[2]
    source_config = root / str(raw["source_config_path"])
    if sha256_file(source_config) != raw.get("source_config_sha256"):
        raise CCMTPPContractError("source-preparation config hash mismatch")
    return load_ccmtpp_definition(source_config)


def build_execution_plan(
    config_path: Path, *, candidate_id: str, dataset: str
) -> ExecutionPlan:
    config_path = config_path.resolve()
    raw = _load_yaml(config_path)
    source_definition = _validate_runner_config(config_path, raw)
    if candidate_id not in raw["candidate_chain"]:
        raise CCMTPPContractError("candidate is outside C1-C4")
    if dataset not in raw["datasets"]:
        raise CCMTPPContractError("dataset is outside amlsim/sparkov")
    jobs = (
        ExecutionJob(
            dataset=dataset,
            candidate_id=candidate_id,
            seed=int(raw["seed"]),
            attempt=str(raw["attempt"]),
        ),
    )
    return ExecutionPlan(
        repository_root=config_path.parents[2],
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        source_definition=source_definition,
        candidate_id=candidate_id,
        dataset_scope=dataset,
        jobs=jobs,
        raw=raw,
    )


def _validate_dataset_worker_scope(plan: ExecutionPlan) -> None:
    allowed = tuple(plan.raw.get("worker_scope", {}).get("datasets", ()))
    if (
        plan.dataset_scope not in allowed
        or len(plan.jobs) != 1
        or plan.jobs[0].dataset != plan.dataset_scope
        or plan.jobs[0].candidate_id != plan.candidate_id
    ):
        raise CCMTPPContractError(
            "one dataset and one candidate cell are required per worker process"
        )


def execution_plan_report(plan: ExecutionPlan) -> Mapping[str, Any]:
    _validate_dataset_worker_scope(plan)
    return {
        "schema_version": "cof-ccmtpp-v1-execution-plan-v1",
        "status": "PASS",
        "candidate_id": plan.candidate_id,
        "dataset_scope": plan.dataset_scope,
        "initial_candidate": plan.raw["initial_candidate"],
        "jobs": [
            {
                "dataset": job.dataset,
                "candidate_id": job.candidate_id,
                "seed": job.seed,
                "attempt": job.attempt,
            }
            for job in plan.jobs
        ],
        "direct_execution_allowed": False,
        "source_commit": current_source_commit(plan.repository_root),
        "runner_relevant_source_sha256": execution_relevant_source_sha256(
            plan.repository_root
        ),
        "runner_config_sha256": plan.config_sha256,
        "source_config_sha256": plan.source_definition.config_sha256,
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),
        "authorization_created": False,
        "runtime_artifact_created": False,
    }


def dry_run_execution(plan: ExecutionPlan) -> Mapping[str, Any]:
    """Validate source/provenance bindings without opening split data bodies."""

    if sha256_file(plan.config_path) != plan.config_sha256:
        raise CCMTPPContractError("runner config changed after planning")
    datasets = []
    for job in plan.jobs:
        record = plan.raw["datasets"][job.dataset]
        # Existence and type checks do not open the array body. Content hashes
        # remain immutable authorization bindings and are rehashed only after
        # a valid execute authorization has been accepted.
        for role in ("train", "validation"):
            path = plan.repository_root / str(record[f"{role}_path"])
            if not path.is_file():
                raise CCMTPPContractError(f"{job.dataset} {role} split is missing")
        datasets.append(
            {
                "dataset": job.dataset,
                "train_sha256_binding": record["train_sha256"],
                "validation_sha256_binding": record["validation_sha256"],
                "train_transform_sha256": record["transform_sha256"],
                "sampling_plan_sha256": record["sampling_plan_sha256"],
                "threshold_sha256": record["threshold_sha256"],
                "split_body_hash_recomputed": False,
            }
        )
    report = dict(execution_plan_report(plan))
    parent = {"C2": "C1", "C3": "C2", "C4": "C3"}.get(plan.candidate_id)
    report.update(
        {
            "schema_version": "cof-ccmtpp-v1-execution-dry-run-v1",
            "candidate_gate": (
                "SEPARATE_AUTHORIZATION_REQUIRED"
                if plan.candidate_id == "C1"
                else f"LOCKED_PENDING_{parent}_COMPLETE_AND_PASS"
            ),
            "dataset_provenance": datasets,
            "data_body_files_opened": [],
            "model_modules_imported": [],
            "gpu_or_cuda_probes": [],
            "test_paths_resolved": [],
        }
    )
    return report


def _job_authorization_record(plan: ExecutionPlan, job: ExecutionJob) -> Mapping[str, Any]:
    dataset = plan.raw["datasets"][job.dataset]
    return {
        "dataset": job.dataset,
        "candidate_id": job.candidate_id,
        "seed": job.seed,
        "attempt": job.attempt,
        "bundle_root": dataset["bundle_root"],
        "bundle_tree_sha256": dataset["bundle_tree_sha256"],
        "train_path": dataset["train_path"],
        "train_sha256": dataset["train_sha256"],
        "validation_path": dataset["validation_path"],
        "validation_sha256": dataset["validation_sha256"],
        "transform_path": dataset["transform_path"],
        "transform_sha256": dataset["transform_sha256"],
        "provenance_path": dataset["provenance_path"],
        "provenance_sha256": dataset["provenance_sha256"],
        "summary_path": dataset["summary_path"],
        "summary_sha256": dataset["summary_sha256"],
        "sampling_plan_sha256": dataset["sampling_plan_sha256"],
        "threshold_sha256": dataset["threshold_sha256"],
    }


def _c0_parent_evidence(plan: ExecutionPlan) -> Mapping[str, Any]:
    evidence = {}
    for job in plan.jobs:
        record = plan.raw["datasets"][job.dataset]
        evidence[job.dataset] = {
            "kind": "frozen_non_v3_hash_reference",
            "candidate_id": "C0",
            "attempt_path": record["c0_attempt"],
            "manifest_sha256": record["c0_manifest_sha256"],
            "evaluation_sha256": record["c0_evaluation_sha256"],
            "sample_sha256": record["c0_sample_sha256"],
            "complete_sha256": record["c0_complete_sha256"],
            "sampling_plan_sha256": record["sampling_plan_sha256"],
            "threshold_sha256": record["threshold_sha256"],
        }
    return evidence


def build_expected_authorization_claim(
    plan: ExecutionPlan,
    *,
    approval: Mapping[str, Any] | None = None,
    parent_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an in-memory expected claim; this never writes authorization."""

    _validate_dataset_worker_scope(plan)

    parent_id = {
        "C1": "C0", "C2": "C1", "C3": "C2", "C4": "C3"
    }[plan.candidate_id]
    if parent_evidence is None:
        parent_evidence = (
            _c0_parent_evidence(plan) if plan.candidate_id == "C1" else {}
        )
    claim: dict[str, Any] = {
        "schema_version": "cof-ccmtpp-v1-execution-authorization-v1",
        "authorized_action": "EXTERNAL_VALIDATION_CANDIDATE",
        "source_commit": current_source_commit(plan.repository_root),
        "runner_relevant_source_sha256": execution_relevant_source_sha256(
            plan.repository_root
        ),
        "runner_config_sha256": plan.config_sha256,
        "source_config_sha256": plan.source_definition.config_sha256,
        "runtime_root": str(plan.raw["runtime_root"]),
        "authorization_history_root": str(plan.raw["authorization_history_root"]),
        "candidate_id": plan.candidate_id,
        "dataset_scope": plan.dataset_scope,
        "parent_candidate_id": parent_id,
        "datasets": [job.dataset for job in plan.jobs],
        "attempts": {job.dataset: job.attempt for job in plan.jobs},
        "seed": int(plan.raw["seed"]),
        "jobs": [_job_authorization_record(plan, job) for job in plan.jobs],
        "parent_evidence": dict(parent_evidence),
        "scope": dict(AUTHORIZATION_SCOPE),
        "approval": dict(approval or {
            "approved": False,
            "text": "SEPARATE_EXPLICIT_USER_APPROVAL_REQUIRED",
        }),
    }
    claim["authorization_sha256"] = canonical_sha256(claim)
    return claim


def validate_authorization_document(
    plan: ExecutionPlan, authorization: Mapping[str, Any]
) -> Mapping[str, Any]:
    approval = authorization.get("approval")
    if (
        not isinstance(approval, Mapping)
        or approval.get("approved") is not True
        or not isinstance(approval.get("text"), str)
        or plan.candidate_id not in approval["text"]
    ):
        raise CCMTPPContractError("authorization lacks explicit candidate approval")
    parent = authorization.get("parent_evidence")
    if not isinstance(parent, Mapping):
        raise CCMTPPContractError("authorization parent evidence is invalid")
    if plan.candidate_id != "C1" and set(parent) != {
        job.dataset for job in plan.jobs
    }:
        raise CCMTPPContractError("parent COMPLETE and gate PASS evidence is required")
    expected = build_expected_authorization_claim(
        plan,
        approval=approval,
        parent_evidence=parent,
    )
    if authorization.get("scope") != AUTHORIZATION_SCOPE:
        raise CCMTPPContractError("authorization scope changed")
    if dict(authorization) != expected:
        raise CCMTPPContractError("authorization hash-bound identity mismatch")
    if plan.candidate_id == "C1":
        _validate_c0_parent_evidence(plan, parent)
    else:
        validate_parent_gate_evidence(plan, parent)
    return {
        "status": "PASS",
        "candidate_id": plan.candidate_id,
        "dataset_scope": plan.dataset_scope,
        "datasets": expected["datasets"],
        "attempts": expected["attempts"],
        "authorization_sha256": expected["authorization_sha256"],
        "parent_candidate_id": expected["parent_candidate_id"],
    }


def validate_authorization_path(
    plan: ExecutionPlan, authorization_path: Path
) -> Mapping[str, Any]:
    """Read one append-only authorization and validate it before all execution access."""

    configured_root = Path(str(plan.raw["authorization_history_root"]))
    if not configured_root.is_absolute():
        configured_root = plan.repository_root / configured_root
    history_root = configured_root.resolve()
    try:
        resolved = authorization_path.resolve(strict=True)
        resolved.relative_to(history_root)
    except (OSError, ValueError) as error:
        raise CCMTPPContractError(
            "authorization must be inside append-only history"
        ) from error
    if resolved.suffix != ".json" or not resolved.is_file():
        raise CCMTPPContractError("authorization history entry is not JSON")
    try:
        payload = resolved.read_bytes()
        authorization = json.loads(payload.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CCMTPPContractError("authorization history entry is invalid") from error
    if not isinstance(authorization, Mapping):
        raise CCMTPPContractError("authorization history entry is not an object")
    validated = dict(validate_authorization_document(plan, authorization))
    for job in plan.jobs:
        record = plan.raw["datasets"][job.dataset]
        for path_key, hash_key in (
            ("train_path", "train_sha256"),
            ("validation_path", "validation_sha256"),
            ("transform_path", "transform_sha256"),
            ("provenance_path", "provenance_sha256"),
            ("summary_path", "summary_sha256"),
        ):
            path = plan.repository_root / str(record[path_key])
            if sha256_file(path) != record[hash_key]:
                raise CCMTPPContractError(
                    f"authorized {job.dataset} input hash changed: {path_key}"
                )
    validated.update(
        {
            "authorization_path": str(resolved),
            "authorization_file_sha256": hashlib.sha256(payload).hexdigest(),
            "document": dict(authorization),
        }
    )
    return validated


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CCMTPPContractError(f"cannot read parent artifact: {path}") from error
    if not isinstance(value, Mapping):
        raise CCMTPPContractError(f"parent artifact is not an object: {path}")
    return value


def _validate_c0_parent_evidence(
    plan: ExecutionPlan, evidence: Mapping[str, Any]
) -> None:
    expected = _c0_parent_evidence(plan)
    if evidence != expected:
        raise CCMTPPContractError("C0 parent authorization evidence changed")
    for dataset, record in expected.items():
        attempt = plan.repository_root / str(record["attempt_path"])
        checks = {
            "manifest.json": record["manifest_sha256"],
            "evaluation.json": record["evaluation_sha256"],
            "sample.npz": record["sample_sha256"],
            "COMPLETE.json": record["complete_sha256"],
            "validation_sampling_plan.npz": record["sampling_plan_sha256"],
            "train_bootstrap_thresholds.json": record["threshold_sha256"],
        }
        for filename, expected_hash in checks.items():
            if sha256_file(attempt / filename) != expected_hash:
                raise CCMTPPContractError(
                    f"C0 parent artifact changed for {dataset}: {filename}"
                )


def validate_parent_gate_evidence(
    plan: ExecutionPlan, evidence: Mapping[str, Any]
) -> Mapping[str, Any]:
    parent_id = {
        "C2": "C1", "C3": "C2", "C4": "C3"
    }.get(plan.candidate_id)
    expected_datasets = {job.dataset for job in plan.jobs}
    if parent_id is None or set(evidence) != expected_datasets:
        raise CCMTPPContractError("parent evidence does not match sequential gate")
    runtime_root = (plan.repository_root / str(plan.raw["runtime_root"])).resolve()
    source_commit = current_source_commit(plan.repository_root)
    source_hash = execution_relevant_source_sha256(plan.repository_root)
    for job in plan.jobs:
        record = evidence.get(job.dataset)
        if not isinstance(record, Mapping) or set(record) != {
            "attempt_path",
            "complete_sha256",
            "checksum_manifest_sha256",
            "gate_decision_sha256",
            "checkpoint_provenance_sha256",
        }:
            raise CCMTPPContractError("parent evidence record is incomplete")
        attempt = Path(str(record["attempt_path"])).resolve()
        expected_attempt = (
            runtime_root
            / job.dataset
            / parent_id
            / f"seed_{job.seed}"
            / "attempt_001"
        ).resolve()
        if attempt != expected_attempt:
            raise CCMTPPContractError("parent attempt path is not the immediate parent")
        complete_path = attempt / "COMPLETE.json"
        checksum_path = attempt / "checksum_manifest.json"
        gate_path = attempt / "gate_decision.json"
        provenance_path = attempt / "checkpoint_provenance.json"
        if (
            sha256_file(complete_path) != record["complete_sha256"]
            or sha256_file(checksum_path) != record["checksum_manifest_sha256"]
            or sha256_file(gate_path) != record["gate_decision_sha256"]
            or sha256_file(provenance_path)
            != record["checkpoint_provenance_sha256"]
        ):
            raise CCMTPPContractError("parent artifact hash mismatch")
        checksum = _read_json(checksum_path)
        files = checksum.get("files")
        if not isinstance(files, Mapping) or not files:
            raise CCMTPPContractError("parent checksum manifest is invalid")
        for relative, expected_hash in files.items():
            if sha256_file(attempt / str(relative)) != expected_hash:
                raise CCMTPPContractError("parent checksum verification failed")
        complete = _read_json(complete_path)
        gate = _read_json(gate_path)
        provenance = _read_json(provenance_path)
        manifest = _read_json(attempt / "manifest.json")
        dataset = plan.raw["datasets"][job.dataset]
        expected_provenance = {
            "source_commit": source_commit,
            "runner_relevant_source_sha256": source_hash,
            "source_config_sha256": plan.source_definition.config_sha256,
            "runner_config_sha256": plan.config_sha256,
            "train_sha256": dataset["train_sha256"],
            "train_transform_sha256": dataset["transform_sha256"],
            "threshold_sha256": dataset["threshold_sha256"],
            "sampling_plan_sha256": dataset["sampling_plan_sha256"],
        }
        if (
            complete.get("status") != "COMPLETE"
            or complete.get("candidate_id") != parent_id
            or complete.get("dataset") != job.dataset
            or complete.get("checksum_manifest_sha256")
            != record["checksum_manifest_sha256"]
            or complete.get("artifact_index_sha256")
            != sha256_file(attempt / "artifact_index.json")
            or gate.get("candidate_id") != parent_id
            or gate.get("stop_criterion_id") != parent_id
            or gate.get("status") != "PASS"
            or any(
                provenance.get(key) != value
                for key, value in expected_provenance.items()
            )
            or manifest.get("candidate_id") != parent_id
            or manifest.get("dataset") != job.dataset
            or any(
                manifest.get("provenance", {}).get(key) != value
                for key, value in expected_provenance.items()
            )
        ):
            raise CCMTPPContractError(
                "parent COMPLETE/gate/provenance contract mismatch"
            )
    return {
        "status": "PASS",
        "parent_candidate_id": parent_id,
        "datasets": [job.dataset for job in plan.jobs],
    }


def validate_runtime_candidate(
    *,
    plan: ExecutionPlan,
    job: ExecutionJob,
    model: Any,
    train_state_binding: Mapping[str, Any],
    training_hyperparameters: Mapping[str, Any],
    enabled_hooks: tuple[str, ...],
) -> Mapping[str, Any]:
    from models.cof_ccmtpp_v1 import (
        CoFCCMTPPV1,
        FlatCopyReceiverDecoder,
        FlatReceiverDecoder,
        HierarchicalCopyReceiverDecoder,
    )

    if job not in plan.jobs or not isinstance(model, CoFCCMTPPV1):
        raise CCMTPPContractError("candidate factor identity mismatch")
    if model.candidate != plan.candidate_id:
        raise CCMTPPContractError("candidate factor isolation mismatch")
    if enabled_hooks:
        raise CCMTPPContractError("post-hoc or unregistered hook is forbidden")
    source_model = plan.source_definition.raw["model"]
    expected_training = {
        key: source_model[key]
        for key in (
            "optimizer", "learning_rate", "weight_decay", "batch_size",
            "requested_updates", "max_wall_seconds",
            "checkpoint_interval_updates", "early_stopping", "update_sweep",
        )
    }
    if training_hyperparameters != expected_training:
        raise CCMTPPContractError("runtime training hyperparameter changed")
    if any((
        model.d_model != source_model["d_model"],
        model.n_heads != source_model["n_heads"],
        model.n_layers != source_model["n_layers"],
        model.max_length != source_model["max_length"],
        model.gap_components != source_model["gap_mixture_components"],
        model.dropout != source_model["dropout"],
        model.amount_contract != AMOUNT_CONTRACT_NAME,
        model.structure_loss is not None,
    )):
        raise CCMTPPContractError("runtime model hyperparameter changed")
    expected_type = {
        "C1": FlatReceiverDecoder,
        "C2": FlatCopyReceiverDecoder,
        "C3": HierarchicalCopyReceiverDecoder,
        "C4": HierarchicalCopyReceiverDecoder,
    }[plan.candidate_id]
    if (
        not isinstance(model.receiver_decoder, expected_type)
        or model.y_balanced_likelihood != (plan.candidate_id == "C4")
        or (plan.candidate_id in {"C1", "C2"} and model.hierarchy is not None)
        or (plan.candidate_id in {"C3", "C4"} and model.hierarchy is None)
    ):
        raise CCMTPPContractError("candidate factor isolation mismatch")
    dataset = plan.raw["datasets"][job.dataset]
    required_zero = (
        "validation_rows_used", "internal_test_rows_used", "fraud_test_rows_used"
    )
    if (
        train_state_binding.get("fit_split") != "train"
        or any(train_state_binding.get(key) != 0 for key in required_zero)
    ):
        raise CCMTPPContractError("runtime state fitting is not train-only")
    required_hashes = (
        "receiver_vocabulary_sha256", "receiver_hierarchy_sha256",
        "gap_support_sha256",
    )
    if any(
        not isinstance(train_state_binding.get(key), str)
        or len(train_state_binding[key]) != 64
        for key in required_hashes
    ):
        raise CCMTPPContractError("train-only state hash is invalid")
    if any((
        train_state_binding.get("train_sha256") != dataset["train_sha256"],
        train_state_binding.get("amount_transform_sha256")
        != dataset["transform_sha256"],
        train_state_binding.get("sampling_plan_sha256")
        != dataset["sampling_plan_sha256"],
        train_state_binding.get("threshold_sha256") != dataset["threshold_sha256"],
        float(train_state_binding.get("gap_support_max", -1)) != model.gap_max,
        model.hierarchy is not None
        and train_state_binding.get("receiver_hierarchy_sha256")
        != model.hierarchy.state_sha256,
    )):
        raise CCMTPPContractError("train-only state provenance mismatch")
    return {
        "candidate_id": plan.candidate_id,
        "dataset": job.dataset,
        "factor_isolation": "PASS",
        "train_only_state_binding": "PASS",
        "hyperparameter_contract": "PASS",
    }


def resolve_dataset_access(
    *,
    plan: ExecutionPlan,
    dataset: str,
    split: str,
    purpose: str,
    fit_complete: bool,
) -> Path:
    _validate_dataset_worker_scope(plan)
    if dataset != plan.dataset_scope:
        raise CCMTPPContractError("dataset access is outside worker dataset scope")
    normalized = str(split).lower()
    if any(token in normalized for token in (
        "internal_test", "internal-test", "fraudtest", "fraud_test",
        "fresh_test", "fresh-test",
    )):
        raise CCMTPPContractError(
            "test/fraudTest is forbidden before path resolution"
        )
    if dataset not in plan.raw["datasets"] or normalized not in {"train", "validation"}:
        raise CCMTPPContractError("dataset split is outside external validation")
    if normalized == "train" and purpose not in {"fit", "train_state"}:
        raise CCMTPPContractError("train body purpose is invalid")
    if normalized == "validation" and (
        not fit_complete or purpose not in {"sampling_plan", "evaluation"}
    ):
        raise CCMTPPContractError("validation is allowed only after fit")
    record = plan.raw["datasets"][dataset]
    relative = str(record[f"{normalized}_path"])
    if any(token in relative.lower() for token in (
        "internal_test", "fraudtest", "fraud_test", "fresh_test"
    )):
        raise CCMTPPContractError("forbidden path token before path resolution")
    path = (plan.repository_root / relative).resolve()
    if sha256_file(path) != record[f"{normalized}_sha256"]:
        raise CCMTPPContractError(f"authorized {normalized} hash changed")
    return path


def fit_train_only_states(
    *,
    plan: ExecutionPlan,
    job: ExecutionJob,
    batch: Any,
    transform_state: Mapping[str, Any],
    transform_file_sha256: str,
    split: str,
) -> TrainOnlyStates:
    from models.cof_ccmtpp_v1 import fit_train_receiver_hierarchy

    if split != "train":
        raise CCMTPPContractError("vocabulary/hierarchy/gap/amount fitting is train-only")
    if job not in plan.jobs:
        raise CCMTPPContractError("train-state job is outside the plan")
    dataset = plan.raw["datasets"][job.dataset]
    if (
        transform_file_sha256 != dataset["transform_sha256"]
        or transform_state.get("fit_role") != "train"
        or transform_state.get("pad_code") != 0
        or transform_state.get("unk_code") != 1
    ):
        raise CCMTPPContractError("amount/vocabulary transform is not frozen train-only")
    try:
        x_num = np.asarray(batch.x_num)
        dt_bin = np.asarray(batch.dt_bin)
        x_cat = np.asarray(batch.x_cat)
        valid = np.asarray(batch.valid_mask)
        y = np.asarray(batch.y_entity)
        lengths = np.asarray(batch.lengths)
    except AttributeError as error:
        raise CCMTPPContractError("train batch contract is incomplete") from error
    if (
        x_num.ndim != 3
        or x_num.shape[-1] != 1
        or dt_bin.shape != x_num.shape[:2]
        or x_cat.shape != (*x_num.shape[:2], 1)
        or valid.shape != x_num.shape[:2]
        or valid.dtype != np.bool_
        or y.shape != (x_num.shape[0],)
        or lengths.shape != (x_num.shape[0],)
        or not np.array_equal(
            valid,
            np.arange(x_num.shape[1])[None, :] < lengths[:, None],
        )
    ):
        raise CCMTPPContractError("train sequence/mask contract is invalid")
    vocabulary = transform_state.get("receiver_vocabulary")
    if not isinstance(vocabulary, list):
        raise CCMTPPContractError("receiver vocabulary state is missing")
    codes = sorted(int(record["code"]) for record in vocabulary)
    if codes != list(range(2, len(codes) + 2)):
        raise CCMTPPContractError("train receiver vocabulary is not contiguous")
    receiver_classes = len(codes) + 2
    receiver = x_cat[..., 0].astype(np.int64, copy=False)
    if np.any(receiver[valid] < 2) or np.any(receiver[valid] >= receiver_classes):
        raise CCMTPPContractError("train receiver is outside train vocabulary")
    tau = np.asarray(transform_state.get("gap_tau"), dtype=np.float64)
    if (
        tau.ndim != 1
        or len(tau) < 1
        or not np.isfinite(tau).all()
        or np.any(tau < 0)
        or np.any(dt_bin[valid] < 0)
        or np.any(dt_bin[valid] >= len(tau))
    ):
        raise CCMTPPContractError("train gap support state is invalid")
    continuous_gap = np.zeros(dt_bin.shape, dtype=np.float64)
    continuous_gap[valid] = tau[dt_bin[valid]]
    common = plan.source_definition.raw["common_contract"]
    hierarchy_rule = common["receiver_hierarchy_rule"]
    hierarchy = fit_train_receiver_hierarchy(
        receiver=np_to_torch_long(receiver),
        valid_mask=np_to_torch_bool(valid),
        receiver_classes=receiver_classes,
        fit_split="train",
        provenance={
            "train_sha256": dataset["train_sha256"],
            "transform_sha256": dataset["transform_sha256"],
        },
        head_min_count=int(hierarchy_rule["head_min_count"]),
        max_head_categories=int(hierarchy_rule["max_head_categories"]),
        tail_cluster_count=int(hierarchy_rule["tail_cluster_count"]),
    )
    vocabulary_payload = {
        "schema_version": "cof-ccmtpp-v1-receiver-vocabulary-v1",
        "fit_split": "train",
        "pad_code": 0,
        "unk_code": 1,
        "receiver_classes": receiver_classes,
        "codes": codes,
        "train_sha256": dataset["train_sha256"],
        "transform_file_sha256": transform_file_sha256,
        "transform_state_sha256": transform_state.get("state_sha256"),
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
    }
    vocabulary_state = {
        **vocabulary_payload,
        "state_sha256": canonical_sha256(vocabulary_payload),
    }
    gap_payload = {
        "schema_version": "cof-ccmtpp-v1-gap-support-v1",
        "fit_split": "train",
        "representation": "gap_tau[dt_bin]",
        "tau": tau.tolist(),
        "max_gap": float(continuous_gap[valid].max()),
        "train_sha256": dataset["train_sha256"],
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
    }
    gap_state = {**gap_payload, "state_sha256": canonical_sha256(gap_payload)}
    amount_payload = {
        "schema_version": "cof-ccmtpp-v1-amount-transform-binding-v1",
        "fit_split": "train",
        "contract": AMOUNT_CONTRACT_NAME,
        "amount_log_mean": float(transform_state["amount_log_mean"]),
        "amount_log_std": float(transform_state["amount_log_std"]),
        "transform_file_sha256": transform_file_sha256,
        "transform_state_sha256": transform_state.get("state_sha256"),
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
    }
    if (
        not np.isfinite(amount_payload["amount_log_mean"])
        or not np.isfinite(amount_payload["amount_log_std"])
        or amount_payload["amount_log_std"] <= 0
    ):
        raise CCMTPPContractError("amount transform state is invalid")
    amount_state = {**amount_payload, "state_sha256": canonical_sha256(amount_payload)}
    binding = {
        "fit_split": "train",
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
        "train_sha256": dataset["train_sha256"],
        "receiver_vocabulary_sha256": vocabulary_state["state_sha256"],
        "receiver_hierarchy_sha256": hierarchy.state_sha256,
        "gap_support_sha256": gap_state["state_sha256"],
        "gap_support_max": gap_state["max_gap"],
        "amount_transform_sha256": transform_file_sha256,
        "sampling_plan_sha256": dataset["sampling_plan_sha256"],
        "threshold_sha256": dataset["threshold_sha256"],
    }
    return TrainOnlyStates(
        binding=binding,
        receiver_vocabulary=vocabulary_state,
        hierarchy=hierarchy,
        gap_support=gap_state,
        amount_transform=amount_state,
        continuous_gap=continuous_gap,
        valid_mask=valid.copy(),
    )


def np_to_torch_long(value: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(value)).long()


def np_to_torch_bool(value: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(value)).bool()


def build_checkpoint_provenance(
    *,
    plan: ExecutionPlan,
    job: ExecutionJob,
    authorization_sha256: str,
    train_state_binding: Mapping[str, Any],
    actual_updates: int,
    checkpoint_sha256: str,
) -> Mapping[str, Any]:
    """Bind a checkpoint to every datum needed by the sequential unlock gate."""

    if (
        job not in plan.jobs
        or len(authorization_sha256) != 64
        or len(checkpoint_sha256) != 64
        or int(actual_updates) < 0
    ):
        raise CCMTPPContractError("checkpoint provenance identity is invalid")
    dataset = plan.raw["datasets"][job.dataset]
    required_binding = {
        "fit_split": "train",
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
        "train_sha256": dataset["train_sha256"],
        "amount_transform_sha256": dataset["transform_sha256"],
        "sampling_plan_sha256": dataset["sampling_plan_sha256"],
        "threshold_sha256": dataset["threshold_sha256"],
    }
    if any(train_state_binding.get(key) != value for key, value in required_binding.items()):
        raise CCMTPPContractError("checkpoint provenance is not train-only")
    for key in (
        "receiver_vocabulary_sha256",
        "receiver_hierarchy_sha256",
        "gap_support_sha256",
    ):
        if not isinstance(train_state_binding.get(key), str) or len(
            str(train_state_binding[key])
        ) != 64:
            raise CCMTPPContractError("checkpoint fit-state hash is invalid")
    payload = {
        "schema_version": "cof-ccmtpp-v1-checkpoint-provenance-v1",
        "dataset": job.dataset,
        "candidate_id": job.candidate_id,
        "seed": job.seed,
        "source_commit": current_source_commit(plan.repository_root),
        "runner_relevant_source_sha256": execution_relevant_source_sha256(
            plan.repository_root
        ),
        "source_config_sha256": plan.source_definition.config_sha256,
        "runner_config_sha256": plan.config_sha256,
        "authorization_sha256": authorization_sha256,
        "train_sha256": dataset["train_sha256"],
        "train_transform_sha256": dataset["transform_sha256"],
        "receiver_vocabulary_sha256": train_state_binding[
            "receiver_vocabulary_sha256"
        ],
        "receiver_hierarchy_sha256": train_state_binding[
            "receiver_hierarchy_sha256"
        ],
        "gap_support_sha256": train_state_binding["gap_support_sha256"],
        "threshold_sha256": dataset["threshold_sha256"],
        "sampling_plan_sha256": dataset["sampling_plan_sha256"],
        "amount_contract": AMOUNT_CONTRACT_NAME,
        "actual_updates": int(actual_updates),
        "checkpoint_sha256": checkpoint_sha256,
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
    }
    return {**payload, "provenance_sha256": canonical_sha256(payload)}


def validate_checkpoint_provenance(
    *,
    provenance: Mapping[str, Any],
    plan: ExecutionPlan,
    job: ExecutionJob,
    authorization_sha256: str,
    train_state_binding: Mapping[str, Any],
    actual_updates: int,
    checkpoint_sha256: str,
) -> None:
    expected = build_checkpoint_provenance(
        plan=plan,
        job=job,
        authorization_sha256=authorization_sha256,
        train_state_binding=train_state_binding,
        actual_updates=actual_updates,
        checkpoint_sha256=checkpoint_sha256,
    )
    if dict(provenance) != expected:
        raise CCMTPPContractError("checkpoint provenance mismatch")


def _authorized_dependency_child(
    payload: Mapping[str, Any], event_queue: Any
) -> Mapping[str, Any]:
    dependencies = payload["dependencies"]
    plan = payload["plan"]
    job = payload["job"]
    authorization = payload["authorization"]
    data = dependencies.load_data_body(
        plan=plan,
        job=job,
        authorization=authorization,
    )
    model = dependencies.build_model(plan=plan, job=job, data=data)
    device = dependencies.query_device(
        plan=plan,
        job=job,
        authorization=authorization,
    )
    event_queue.put({"kind": "setup_complete", "dataset": job.dataset})
    return dependencies.run_job(
        plan=plan,
        job=job,
        authorization=authorization,
        data=data,
        model=model,
        device=device,
        attempt_path=Path(str(payload["attempt_path"])),
        event_queue=event_queue,
    )


def execute_authorized(
    *,
    plan: ExecutionPlan,
    authorization_path: Path | None,
    dependencies: ExecutionDependencies | Callable[[], ExecutionDependencies],
) -> Mapping[str, Any]:
    if authorization_path is None:
        raise CCMTPPContractError(
            "execute requires a valid append-only authorization before access"
        )
    # This is deliberately the first operation. No dependency callback, split
    # body, model import, CUDA call, or runtime directory is touched before it.
    authorization_record = validate_authorization_path(plan, authorization_path)
    authorization = authorization_record["document"]
    active_dependencies = dependencies() if callable(dependencies) else dependencies
    if not isinstance(active_dependencies, ExecutionDependencies):
        raise CCMTPPContractError("execution dependency factory is invalid")
    runtime_root = Path(str(plan.raw["runtime_root"]))
    if not runtime_root.is_absolute():
        runtime_root = plan.repository_root / runtime_root
    watchdog = plan.raw["watchdog"]
    results = []
    for job in plan.jobs:
        job_record = next(
            (
                record
                for record in authorization["jobs"]
                if record["dataset"] == job.dataset
            ),
            None,
        )
        if job_record is None:
            raise CCMTPPContractError("authorized job record is missing")
        provenance_payload = {
            "authorization_sha256": authorization["authorization_sha256"],
            "job": job_record,
            "source_commit": authorization["source_commit"],
            "runner_relevant_source_sha256": authorization[
                "runner_relevant_source_sha256"
            ],
            "runner_config_sha256": authorization["runner_config_sha256"],
            "source_config_sha256": authorization["source_config_sha256"],
        }
        parent_unlock_provenance = {
            "source_commit": authorization["source_commit"],
            "runner_relevant_source_sha256": authorization[
                "runner_relevant_source_sha256"
            ],
            "source_config_sha256": authorization["source_config_sha256"],
            "runner_config_sha256": authorization["runner_config_sha256"],
            "train_sha256": job_record["train_sha256"],
            "train_transform_sha256": job_record["transform_sha256"],
            "threshold_sha256": job_record["threshold_sha256"],
            "sampling_plan_sha256": job_record["sampling_plan_sha256"],
        }
        provenance_sha256 = canonical_sha256(provenance_payload)
        store = CCMTPPAttemptStore.claim(
            runtime_root=runtime_root,
            job=job,
            authorization_sha256=authorization["authorization_sha256"],
            provenance_sha256=provenance_sha256,
        )
        manifest = {
            "schema_version": "cof-ccmtpp-v1-attempt-manifest-v1",
            "dataset": job.dataset,
            "candidate_id": job.candidate_id,
            "seed": job.seed,
            "attempt": job.attempt,
            "authorization_sha256": authorization["authorization_sha256"],
            "authorization_file_sha256": authorization_record[
                "authorization_file_sha256"
            ],
            "provenance_sha256": provenance_sha256,
            "provenance": parent_unlock_provenance,
            "execution_provenance": provenance_payload,
            "forbidden_access": {
                "entity_id_input": True,
                "internal_test": True,
                "sparkov_fraud_test": True,
                "threshold_change": True,
                "candidate_sweep": True,
            },
        }
        attempt = store.allocate_attempt(manifest)
        terminal = run_runner_owned_child(
            child_spec=RunnerChildSpec(
                target=_authorized_dependency_child,
                payload={
                    "dependencies": active_dependencies,
                    "plan": plan,
                    "job": job,
                    "authorization": authorization,
                    "attempt_path": str(attempt),
                },
            ),
            store=store,
            max_wall_seconds=float(plan.source_definition.raw["model"]["max_wall_seconds"]),
            startup_timeout_seconds=float(watchdog["startup_timeout_seconds"]),
            progress_timeout_seconds=float(watchdog["progress_timeout_seconds"]),
            queue_poll_seconds=float(watchdog["queue_poll_seconds"]),
            termination_grace_seconds=float(watchdog["termination_grace_seconds"]),
            kill_grace_seconds=float(watchdog["kill_grace_seconds"]),
        )
        results.append(
            {
                "dataset": job.dataset,
                "candidate_id": job.candidate_id,
                "status": terminal["status"],
                "attempt_path": str(attempt),
                "terminal_sha256": sha256_file(
                    attempt / f"{terminal['status']}.json"
                ),
            }
        )
    status = "COMPLETE" if all(
        record["status"] in {"COMPLETE", "INVALID"} for record in results
    ) else "FAILED"
    return {
        "schema_version": "cof-ccmtpp-v1-execution-result-v1",
        "status": status,
        "candidate_id": plan.candidate_id,
        "authorization_sha256": authorization["authorization_sha256"],
        "jobs": results,
    }
