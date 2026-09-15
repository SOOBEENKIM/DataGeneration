"""Fail-closed validation runner contracts for CoF-HCMTTPP-v2 H1.

Plan and dry-run are deliberately metadata-only.  The execution backend is
lazy-imported only after an independently created, dataset-scoped
authorization has passed every provenance check.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import subprocess
import time
import traceback
from typing import Any, Callable, Mapping

import yaml


class H1RunnerContractError(RuntimeError):
    """A fail-closed H1 execution-contract violation."""


ZERO_EXECUTION_COUNTS = {
    "authorization_creations": 0,
    "data_body_reads": 0,
    "model_imports_or_builds": 0,
    "gpu_queries": 0,
    "cuda_calls": 0,
    "fit_calls": 0,
    "sample_calls": 0,
    "evaluation_calls": 0,
    "runtime_writes": 0,
    "internal_test_reads": 0,
    "sparkov_fraud_test_reads": 0,
}

EXECUTION_RELEVANT_SOURCE_PATHS = (
    "configs/benchmark_v2/cof_hcmttpp_v2_source_only.yaml",
    "configs/benchmark_v2/cof_hcmttpp_v2_execution_runner.yaml",
    "models/cof_hcmttpp_v2.py",
    "experiments/cof_hcmttpp_v2_execution_runner.py",
    "generators/cof_hcmttpp_v2_execution_backend.py",
    "scripts/run_cof_hcmttpp_v2.py",
)
READINESS_BOUNDARY_SOURCE_PATHS = (
    "experiments/cof_hcmttpp_v2_execution_readiness.py",
    "experiments/cof_hcmttpp_v2_execution_runner.py",
    "generators/cof_hcmttpp_v2_execution_backend.py",
)

CORRECTIVE_ATTEMPT = "attempt_003"
READINESS_ATTEMPT = "attempt_003"
READINESS_OWNERSHIP_SCHEMA = "cof-hcmttpp-v2-h1-readiness-ownership-v1"

AUTHORIZATION_SCOPE = {
    "dataset_cells": 1,
    "candidate": "H1",
    "seed": 4001,
    "attempt": CORRECTIVE_ATTEMPT,
    "execution_class": (
        "FIRST_VALID_H1_EVALUATION_AFTER_TWO_DETERMINISTIC_IMPLEMENTATION_FAILURES"
    ),
    "train_body_read": True,
    "validation_body_read_after_final_checkpoint": True,
    "model_build": True,
    "cuda_model_execution": True,
    "fit": True,
    "validation_sample": True,
    "validation_evaluation": True,
    "append_only_runtime_write": True,
    "gpu_inventory_query": False,
    "authorization_creation": False,
    "internal_test": False,
    "sparkov_fraud_test": False,
    "c1_v1_retry": False,
    "c2_c3_c4": False,
    "h2": False,
    "pointer": False,
    "hierarchy": False,
    "y_balanced_likelihood": False,
    "coherence_loss": False,
    "retry": False,
    "sweep": False,
    "early_stopping": False,
}


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
    dataset_scope: str
    jobs: tuple[ExecutionJob, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionDependencies:
    load_train_body: Callable[..., Any]
    build_model: Callable[..., Any]
    select_device: Callable[..., Any]
    run_job: Callable[..., Any]


@dataclass(frozen=True)
class RunnerChildSpec:
    target: Callable[[Mapping[str, Any], Any], Mapping[str, Any]]
    payload: Mapping[str, Any]


class H1AttemptStore:
    """Append-only, dataset-owned H1 artifact writer with terminal-last commit."""

    TERMINALS = ("COMPLETE", "INVALID", "FAILED")
    COMPLETE_REQUIRED = (
        "manifest.json",
        "frozen_parent_references.json",
        "progress.jsonl",
        "gap_hurdle_state.json",
        "positive_gap_spline_state.json",
        "positive_gap_tail_state.json",
        "amount_transform_reference.json",
        "receiver_vocabulary_reference.json",
        "conditioning_plan.json",
        "checkpoints/final.pt",
        "checkpoints/latest",
        "checkpoint_provenance.json",
        "validation_sample.npz",
        "diagnostics.json",
        "metrics.json",
        "evaluation.json",
        "runtime.json",
        "gate_decision.json",
    )
    REQUIRED_GAP_DIAGNOSTICS = {
        "zero_rate_by_class",
        "zero_rate_absolute_error_by_class",
        "overall_gap_ks_by_class",
        "positive_only_gap_ks_by_class",
        "positive_quantile_differences_by_class",
        "central_tail_counts_by_class",
        "tail_mass_error_by_class",
        "conditional_tail_gate_by_class_and_history_stratum",
        "conditional_tail_scale_by_class_and_history_stratum",
        "signed_scale_diagnostics",
        "positive_cdf_total_mass_error",
        "threshold_cdf_continuity_error",
        "finite_nll_count_by_route",
        "finite_density_count",
        "sampled_min_max",
        "lower_boundary_rate",
        "nonfinite_sample_count",
        "exact_upper_clip_count",
        "frozen_bin_mass",
        "raw_vs_mapped_ks_difference",
    }
    REQUIRED_RECEIVER_DIAGNOSTICS = {
        "classwise_receiver_tv",
        "full_receiver_tv",
        "head_receiver_tv",
        "tail_receiver_tv",
        "unk_rate_error",
        "head",
        "tail",
        "unk",
        "repeat",
        "new",
    }

    def __init__(self, *, runtime_root: Path, job: ExecutionJob, ownership_path: Path):
        self.runtime_root = runtime_root.resolve()
        self.job = job
        self.ownership_path = ownership_path.resolve()
        self.attempt_path: Path | None = None

    @staticmethod
    def _exclusive_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as error:
            raise H1RunnerContractError(f"append-only artifact exists: {path}") from error
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
    ) -> "H1AttemptStore":
        if len(authorization_sha256) != 64 or len(provenance_sha256) != 64:
            raise H1RunnerContractError("ownership hash is invalid")
        if (
            job.candidate_id != "H1"
            or job.seed != 4001
            or job.attempt != CORRECTIVE_ATTEMPT
        ):
            raise H1RunnerContractError("only exact H1 attempt_003 is claimable")
        runtime_root = runtime_root.resolve()
        ownership = (
            runtime_root
            / "workers"
            / job.dataset
            / "H1"
            / f"ownership_{CORRECTIVE_ATTEMPT}.lock"
        )
        value = {
            "schema_version": "cof-hcmttpp-v2-h1-ownership-v1",
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
        except H1RunnerContractError as error:
            raise H1RunnerContractError("dataset H1 ownership already exists") from error
        return cls(runtime_root=runtime_root, job=job, ownership_path=ownership)

    @staticmethod
    def _readiness_ownership_id(value: Mapping[str, Any]) -> str:
        identity = {
            key: value[key]
            for key in (
                "schema_version",
                "scope",
                "dataset",
                "candidate_id",
                "seed",
                "attempt",
                "authorization_sha256",
                "provenance_sha256",
            )
        }
        return hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def claim_readiness_attempt_003(
        cls,
        *,
        scratch_root: Path,
        job: ExecutionJob,
        authorization_sha256: str,
        provenance_sha256: str,
    ) -> "H1AttemptStore":
        """Claim the exact source-only attempt_003 readiness fixture scope.

        This deliberately distinct API cannot authorize production execution:
        it accepts only H1/4001/attempt_003, writes under its caller-supplied
        scratch root, and marks the ownership record as CPU-synthetic
        readiness-only.
        """

        if len(authorization_sha256) != 64 or len(provenance_sha256) != 64:
            raise H1RunnerContractError("readiness ownership hash is invalid")
        if (
            job.dataset not in {"amlsim", "sparkov"}
            or job.candidate_id != "H1"
            or job.seed != 4001
            or job.attempt != READINESS_ATTEMPT
        ):
            raise H1RunnerContractError(
                "only dataset-scoped H1 attempt_003 readiness is claimable"
            )
        scratch_root = scratch_root.resolve()
        ownership = (
            scratch_root
            / "workers"
            / job.dataset
            / "H1"
            / f"ownership_{READINESS_ATTEMPT}.lock"
        )
        value: dict[str, Any] = {
            "schema_version": READINESS_OWNERSHIP_SCHEMA,
            "scope": "SOURCE_ONLY_CPU_SYNTHETIC_EXECUTION_READINESS",
            "dataset": job.dataset,
            "candidate_id": job.candidate_id,
            "seed": job.seed,
            "attempt": job.attempt,
            "authorization_sha256": authorization_sha256,
            "provenance_sha256": provenance_sha256,
        }
        value["ownership_id"] = cls._readiness_ownership_id(value)
        try:
            cls._exclusive_bytes(
                ownership,
                (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                ),
            )
        except H1RunnerContractError as error:
            raise H1RunnerContractError(
                "dataset H1 readiness ownership already exists"
            ) from error
        return cls(runtime_root=scratch_root, job=job, ownership_path=ownership)

    def allocate_readiness_attempt_003(self, manifest: Mapping[str, Any]) -> Path:
        """Allocate the exact scratch-only readiness attempt append-only."""

        if self.attempt_path is not None:
            raise H1RunnerContractError("readiness attempt is already allocated")
        ownership = json.loads(self.ownership_path.read_text(encoding="utf-8"))
        if (
            self.job.attempt != READINESS_ATTEMPT
            or ownership.get("schema_version") != READINESS_OWNERSHIP_SCHEMA
            or manifest.get("scope")
            != "SOURCE_ONLY_CPU_SYNTHETIC_EXECUTION_READINESS"
            or manifest.get("dataset") != self.job.dataset
            or manifest.get("candidate_id") != self.job.candidate_id
            or manifest.get("seed") != self.job.seed
            or manifest.get("attempt") != self.job.attempt
            or manifest.get("authorization_sha256")
            != ownership.get("authorization_sha256")
            or manifest.get("ownership_id") != ownership.get("ownership_id")
        ):
            raise H1RunnerContractError("readiness attempt manifest identity mismatch")
        attempt = (
            self.runtime_root
            / self.job.dataset
            / "H1"
            / "seed_4001"
            / READINESS_ATTEMPT
        )
        try:
            attempt.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise H1RunnerContractError(
                "append-only readiness attempt already exists"
            ) from error
        self.attempt_path = attempt.resolve()
        self.write_json("manifest.json", manifest)
        return self.attempt_path

    def allocate_attempt(self, manifest: Mapping[str, Any]) -> Path:
        if self.attempt_path is not None:
            raise H1RunnerContractError("attempt is already allocated")
        if (
            manifest.get("dataset") != self.job.dataset
            or manifest.get("candidate_id") != "H1"
            or manifest.get("seed") != 4001
        ):
            raise H1RunnerContractError("attempt manifest identity mismatch")
        if self.job.attempt != CORRECTIVE_ATTEMPT:
            raise H1RunnerContractError("only exact H1 attempt_003 is allocatable")
        attempt = (
            self.runtime_root
            / self.job.dataset
            / "H1"
            / "seed_4001"
            / CORRECTIVE_ATTEMPT
        )
        try:
            attempt.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise H1RunnerContractError("append-only attempt already exists") from error
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
        readiness_only: bool = False,
        expected_ownership_id: str | None = None,
        expected_authorization_sha256: str | None = None,
    ) -> "H1AttemptStore":
        store = cls(runtime_root=runtime_root, job=job, ownership_path=ownership_path)
        if job.attempt == READINESS_ATTEMPT and readiness_only:
            if (
                expected_ownership_id is None
                or expected_authorization_sha256 is None
            ):
                raise H1RunnerContractError(
                    "attempt_003 is restricted to explicit source-only readiness"
                )
            expected_ownership = (
                store.runtime_root
                / "workers"
                / job.dataset
                / "H1"
                / f"ownership_{READINESS_ATTEMPT}.lock"
            ).resolve()
            expected_attempt = (
                store.runtime_root
                / job.dataset
                / "H1"
                / "seed_4001"
                / READINESS_ATTEMPT
            ).resolve()
            expected_schema = READINESS_OWNERSHIP_SCHEMA
        elif job.attempt == CORRECTIVE_ATTEMPT and not readiness_only:
            if expected_ownership_id is not None or expected_authorization_sha256 is not None:
                raise H1RunnerContractError(
                    "execution ownership does not accept readiness identity"
                )
            expected_ownership = (
                store.runtime_root
                / "workers"
                / job.dataset
                / "H1"
                / f"ownership_{CORRECTIVE_ATTEMPT}.lock"
            ).resolve()
            expected_attempt = (
                store.runtime_root
                / job.dataset
                / "H1"
                / "seed_4001"
                / CORRECTIVE_ATTEMPT
            ).resolve()
            expected_schema = "cof-hcmttpp-v2-h1-ownership-v1"
        else:
            raise H1RunnerContractError("spawn child attempt scope is not allowed")
        if (
            store.ownership_path != expected_ownership
            or attempt_path.resolve() != expected_attempt
            or not expected_ownership.is_file()
            or not (expected_attempt / "manifest.json").is_file()
        ):
            raise H1RunnerContractError("spawn child attempt ownership mismatch")
        try:
            ownership = json.loads(expected_ownership.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise H1RunnerContractError(
                "spawn child ownership record is invalid"
            ) from error
        if (
            not isinstance(ownership, Mapping)
            or ownership.get("schema_version") != expected_schema
            or ownership.get("dataset") != job.dataset
            or ownership.get("candidate_id") != job.candidate_id
            or ownership.get("seed") != job.seed
            or ownership.get("attempt") != job.attempt
            or len(str(ownership.get("authorization_sha256", ""))) != 64
            or len(str(ownership.get("provenance_sha256", ""))) != 64
        ):
            raise H1RunnerContractError("spawn child ownership record identity mismatch")
        if readiness_only:
            if (
                ownership.get("scope")
                != "SOURCE_ONLY_CPU_SYNTHETIC_EXECUTION_READINESS"
                or ownership.get("ownership_id") != expected_ownership_id
                or ownership.get("authorization_sha256")
                != expected_authorization_sha256
                or ownership.get("ownership_id")
                != cls._readiness_ownership_id(ownership)
            ):
                raise H1RunnerContractError(
                    "spawn child readiness ownership identity mismatch"
                )
        store.attempt_path = expected_attempt
        return store

    def _target(self, relative: str) -> Path:
        if self.attempt_path is None:
            raise H1RunnerContractError("attempt has not been allocated")
        if any((self.attempt_path / f"{name}.json").exists() for name in self.TERMINALS):
            raise H1RunnerContractError("terminal attempt is immutable")
        target = (self.attempt_path / relative).resolve()
        try:
            target.relative_to(self.attempt_path)
        except ValueError as error:
            raise H1RunnerContractError("artifact path escapes owned attempt") from error
        return target

    def write_json(self, relative: str, value: Mapping[str, Any]) -> Path:
        payload = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
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
            {"schema_version": "cof-hcmttpp-v2-h1-index-v1", "files": files},
        )
        checksums = {name: sha256_file(self.attempt_path / name) for name in files}
        checksums["artifact_index.json"] = sha256_file(index)
        checksum = self.write_json(
            "checksum_manifest.json",
            {"schema_version": "cof-hcmttpp-v2-h1-checksum-v1", "files": checksums},
        )
        return index, checksum

    @classmethod
    def _validate_diagnostics(cls, diagnostics: Mapping[str, Any]) -> None:
        gap = diagnostics.get("gap")
        receiver = diagnostics.get("receiver")
        forbidden = diagnostics.get("forbidden_access")
        if (
            not isinstance(gap, Mapping)
            or not cls.REQUIRED_GAP_DIAGNOSTICS.issubset(gap)
            or not isinstance(receiver, Mapping)
            or not cls.REQUIRED_RECEIVER_DIAGNOSTICS.issubset(receiver)
            or not isinstance(diagnostics.get("amount"), Mapping)
            or not isinstance(diagnostics.get("hard_validity"), Mapping)
            or not isinstance(forbidden, Mapping)
            or forbidden.get("internal_test") != 0
            or forbidden.get("sparkov_fraud_test") != 0
        ):
            raise H1RunnerContractError("H1 diagnostic schema is incomplete")

    def finalize(
        self, status: str, *, metadata: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        if status not in self.TERMINALS or self.attempt_path is None:
            raise H1RunnerContractError("terminal status is invalid")
        if any((self.attempt_path / f"{name}.json").exists() for name in self.TERMINALS):
            raise H1RunnerContractError("terminal attempt is immutable")
        if status in {"COMPLETE", "INVALID"}:
            missing = [
                name for name in self.COMPLETE_REQUIRED if not (self.attempt_path / name).is_file()
            ]
            if missing:
                raise H1RunnerContractError(f"terminal artifacts are missing: {missing}")
            evaluation = json.loads((self.attempt_path / "evaluation.json").read_text())
            gate = json.loads((self.attempt_path / "gate_decision.json").read_text())
            diagnostics = json.loads(
                (self.attempt_path / "diagnostics.json").read_text()
            )
            self._validate_diagnostics(diagnostics)
            expected_evaluation = "VALID" if status == "COMPLETE" else "INVALID"
            if evaluation.get("status") != expected_evaluation or gate.get("status") not in {
                "PASS",
                "FAIL",
            }:
                raise H1RunnerContractError("evaluation/gate terminal contract mismatch")
        else:
            if not (self.attempt_path / "runtime.json").exists():
                self.write_json("runtime.json", {"status": "FAILED"})
            if not (self.attempt_path / "diagnostics.json").exists():
                self.write_json("diagnostics.json", {"status": "FAILED"})
        index, checksum = self._write_index_and_checksums()
        terminal = {
            "schema_version": "cof-hcmttpp-v2-h1-terminal-v1",
            "status": status,
            "dataset": self.job.dataset,
            "candidate_id": "H1",
            "seed": 4001,
            "attempt": self.job.attempt,
            "artifact_index_sha256": sha256_file(index),
            "checksum_manifest_sha256": sha256_file(checksum),
        }
        if metadata:
            if set(terminal) & set(metadata):
                raise H1RunnerContractError("terminal metadata overwrites identity")
            terminal.update(dict(metadata))
        self._exclusive_bytes(
            self.attempt_path / f"{status}.json",
            (json.dumps(terminal, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )
        return terminal

    def finalize_worker(self, terminal: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.attempt_path is None or terminal.get("status") not in self.TERMINALS:
            raise H1RunnerContractError("candidate terminal is invalid")
        candidate = self.attempt_path / f"{terminal['status']}.json"
        if not candidate.is_file():
            raise H1RunnerContractError("candidate terminal is missing")
        status = "FAILED" if terminal["status"] == "FAILED" else "COMPLETE"
        value = {
            "schema_version": "cof-hcmttpp-v2-h1-worker-terminal-v1",
            "status": status,
            "dataset": self.job.dataset,
            "candidate_terminal_sha256": sha256_file(candidate),
            "ownership_sha256": sha256_file(self.ownership_path),
        }
        path = (
            self.ownership_path.parent
            / self.job.attempt
            / f"WORKER_{status}.json"
        )
        self._exclusive_bytes(
            path,
            (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )
        return value


def evaluate_h1_gate(
    observed: Mapping[str, Any], references: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Apply the preregistered unrounded H1 gate without calibration."""

    checks: dict[str, bool] = {}

    def value(mapping: Mapping[str, Any], metric: str, label: str | None = None) -> float:
        raw = mapping.get(metric)
        if label is not None:
            raw = raw.get(label) if isinstance(raw, Mapping) else None
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return math.nan
        return number

    for label in ("y0", "y1"):
        current = value(observed, "gap_ks", label)
        limit = value(references, "c0_gap_ks", label)
        checks[f"{label}_overall_gap_ks_le_c0"] = (
            math.isfinite(current) and math.isfinite(limit) and current <= limit
        )
    current_positive = value(observed, "positive_gap_ks", "y0")
    c1_positive = value(references, "c1_positive_gap_ks", "y0")
    checks["y0_positive_gap_ks_lt_c1"] = (
        math.isfinite(current_positive)
        and math.isfinite(c1_positive)
        and current_positive < c1_positive
    )
    for metric, reference_metric in (
        ("coherence", "c1_coherence"),
        ("receiver_tv", "c1_receiver_tv"),
    ):
        for label in ("y0", "y1"):
            current = value(observed, metric, label)
            limit = value(references, reference_metric, label)
            checks[f"{label}_{metric}_le_c1"] = (
                math.isfinite(current) and math.isfinite(limit) and current <= limit
            )
    full = value(observed, "full_receiver_tv")
    full_limit = value(references, "c1_full_receiver_tv")
    checks["full_receiver_tv_le_c1"] = (
        math.isfinite(full) and math.isfinite(full_limit) and full <= full_limit
    )
    checks["hard_validity_pass"] = observed.get("hard_validity") is True
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema_version": "cof-hcmttpp-v2-h1-gate-v1",
        "status": status,
        "checks": checks,
        "family_state": (
            "H1_GATE_PASS"
            if status == "PASS"
            else "STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL"
        ),
    }


def _runner_child_entry(
    target: Callable[[Mapping[str, Any], Any], Mapping[str, Any]],
    payload: Mapping[str, Any],
    event_queue: Any,
) -> None:
    try:
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
    store: H1AttemptStore,
    max_wall_seconds: float,
    startup_timeout_seconds: float,
    progress_timeout_seconds: float,
    queue_poll_seconds: float,
    termination_grace_seconds: float,
    kill_grace_seconds: float,
) -> Mapping[str, Any]:
    """Spawn and monitor queue-first; never use an unbounded join."""

    if store.attempt_path is None or min(
        max_wall_seconds,
        startup_timeout_seconds,
        progress_timeout_seconds,
        queue_poll_seconds,
    ) <= 0:
        raise H1RunnerContractError("child watchdog configuration is invalid")
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
                if process.is_alive():
                    continue
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    failure_class = "unexpected_child_exit"
                    break
            if not isinstance(event, Mapping):
                failure_class = "invalid_child_event"
                break
            kind = event.get("kind")
            if kind == "child_started":
                if child_started:
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
        store.append_progress({"status": "INTERRUPTED"})
        terminal = store.finalize(
            "FAILED", metadata={"failure_class": "parent_interrupt", **cleanup}
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
        store.append_progress({"status": "FAILED", "failure_class": failure_class})
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
            "FAILED", metadata={"failure_class": "child_cleanup", **cleanup}
        )
        store.finalize_worker(terminal)
        return terminal
    if not isinstance(result, Mapping) or result.get("terminal_status") not in {
        "COMPLETE",
        "INVALID",
    }:
        terminal = store.finalize(
            "FAILED", metadata={"failure_class": "invalid_child_result", **cleanup}
        )
        store.finalize_worker(terminal)
        return terminal
    terminal = store.finalize(str(result["terminal_status"]))
    store.finalize_worker(terminal)
    return terminal


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_only_tree_record(root: Path) -> Mapping[str, Any]:
    """Hash an immutable artifact tree without changing filesystem state."""

    root = root.resolve()
    if not root.is_dir():
        raise H1RunnerContractError(f"preserved artifact tree is missing: {root}")
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    records: dict[str, Mapping[str, Any]] = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        records[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "tree_sha256": canonical_sha256({"files": records}),
        "file_count": len(files),
        "byte_count": sum(record["bytes"] for record in records.values()),
    }


def verify_implementation_correction_evidence(
    plan: ExecutionPlan,
) -> Mapping[str, Any]:
    """Verify both failed attempts, both fixes, and the CPU readiness gate."""

    correction = plan.raw["implementation_corrections"]
    report_records: dict[str, Any] = {}
    for name in ("rqs_inverse", "spawn_ownership"):
        record = correction[name]
        report = plan.repository_root / str(record["report_path"])
        report_hash = sha256_file(report)
        if report_hash != record["report_sha256"]:
            raise H1RunnerContractError(
                f"authorized {name} correction report changed"
            )
        report_records[name] = {
            "path": record["report_path"],
            "sha256": report_hash,
            "status": "PASS",
        }

    readiness = correction["execution_readiness"]
    readiness_document = plan.repository_root / str(readiness["document_path"])
    readiness_document_hash = sha256_file(readiness_document)
    if (
        readiness.get("status") != "PASS"
        or readiness.get("passed_source_commit")
        != "f5125fbde533a965b0e3c71b1b3b8950c7fffc85"
        or readiness.get("passed_relevant_execution_source_sha256")
        != "14a848a33ef4883bb55c41cb45e9c67c6e3764b7e5bb47d3819b625117567afb"
        or readiness.get("passed_boundary_sha256")
        != "77a58fd66a3352adb71c3dce5f56c764414f09afc9ef6ecb1636fa4e9ece5b9f"
        or readiness_document_hash != readiness.get("document_sha256")
    ):
        raise H1RunnerContractError("attempt_003 execution readiness evidence changed")

    failed_records: dict[str, Any] = {}
    for attempt, datasets in correction["failed_attempts"].items():
        failed_records[attempt] = {}
        for dataset, prior in datasets.items():
            attempt_root = plan.repository_root / str(prior["path"])
            actual_tree = dict(read_only_tree_record(attempt_root))
            if actual_tree["tree_sha256"] != prior["tree_sha256"]:
                raise H1RunnerContractError(
                    f"authorized prior {dataset} {attempt} tree changed"
                )
            failed_hash = sha256_file(attempt_root / "FAILED.json")
            if failed_hash != prior["failed_sha256"]:
                raise H1RunnerContractError(
                    f"authorized prior {dataset} {attempt} FAILED marker changed"
                )
            failed_records[attempt][dataset] = {
                "path": prior["path"],
                **actual_tree,
                "failed_sha256": failed_hash,
                "preservation_status": "PASS",
            }
    return {
        "status": "PASS",
        "execution_class": correction["execution_class"],
        "authorized_attempt": correction["authorized_attempt"],
        "reports": report_records,
        "execution_readiness": {
            "document_path": readiness["document_path"],
            "document_sha256": readiness_document_hash,
            "passed_boundary_sha256": readiness["passed_boundary_sha256"],
            "status": "PASS",
        },
        "failed_attempts": failed_records,
        "preservation_status": "PASS",
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_source_commit(repository_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise H1RunnerContractError("cannot resolve source commit") from error
    value = result.stdout.strip()
    if len(value) != 40:
        raise H1RunnerContractError("source commit is invalid")
    return value


def _base_commit_is_ancestor(repository_root: Path, base: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, "HEAD"],
            cwd=repository_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def execution_relevant_source_sha256(repository_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in EXECUTION_RELEVANT_SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise H1RunnerContractError(f"runner source is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def execution_readiness_boundary_sha256(repository_root: Path) -> str:
    """Hash the exact parent/spawn/backend CPU-readiness source boundary."""

    digest = hashlib.sha256()
    for relative in READINESS_BOUNDARY_SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise H1RunnerContractError(f"readiness source is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise H1RunnerContractError(f"cannot read runner config: {path}") from error
    if not isinstance(value, Mapping):
        raise H1RunnerContractError("runner config root must be a mapping")
    return value


def _validate_runner_config(path: Path, raw: Mapping[str, Any]) -> None:
    root = path.resolve().parents[2]
    required_source_only = raw.get("source_only", {})
    factor = raw.get("factor_isolation", {})
    execution = raw.get("execution", {})
    correction = raw.get("implementation_corrections", {})
    expected_prior = {
        "attempt_001": {
            "amlsim": {
                "path": "artifacts/cof_hcmttpp_v2/external_validation/amlsim/H1/seed_4001/attempt_001",
                "tree_sha256": "07bca9894bf6b619a4b4d6359773b2da2ac40a7112bebe70096712360671b389",
                "failed_sha256": "5a65749584bb9f4a584bd75ed74eb7d10e0547c66c7ffd3c3f9da00431712042",
            },
            "sparkov": {
                "path": "artifacts/cof_hcmttpp_v2/external_validation/sparkov/H1/seed_4001/attempt_001",
                "tree_sha256": "309fbd56bc6c522b818e842b8c6082df5a0114b7f840537a0361164c744faf28",
                "failed_sha256": "94af493b9dd3f3ddb2847f3cc7d6375ef744d713caf0ad83276d386ba53ed8a3",
            },
        },
        "attempt_002": {
            "amlsim": {
                "path": "artifacts/cof_hcmttpp_v2/external_validation/amlsim/H1/seed_4001/attempt_002",
                "tree_sha256": "91a4670b6b8bebd47f289f9124d100d4d0f52e704ad02dac73ec23c70a767095",
                "failed_sha256": "eca6161ad16e2b3de3010d35a318fa4a7e05d0138fe8175c9b293ad704a8f5e9",
            },
            "sparkov": {
                "path": "artifacts/cof_hcmttpp_v2/external_validation/sparkov/H1/seed_4001/attempt_002",
                "tree_sha256": "b4a900aa1e88fde24299d613fbdf90b67ba648493d904beaf9734ff348fce6a7",
                "failed_sha256": "c98002596dc43e22866555d280df88eefa18d20a0e19f3bd35deb172d8663ff6",
            },
        },
    }
    if (
        raw.get("schema_version") != "cof-hcmttpp-v2-h1-execution-runner-v3"
        or raw.get("mode") != "FIRST_VALID_ATTEMPT_003_PREPARATION"
        or raw.get("family") != "cof_hcmttpp_v2"
        or raw.get("candidate_id") != "H1"
        or raw.get("seed") != 4001
        or raw.get("attempt") != CORRECTIVE_ATTEMPT
        or raw.get("datasets_allowed") != ["amlsim", "sparkov"]
        or raw.get("jobs_per_process") != 1
        or set(raw.get("datasets", {})) != {"amlsim", "sparkov"}
        or any(value is not False for value in required_source_only.values())
        or factor.get("changed_factors") != ["gap_decoder"]
        or factor.get("gap_decoder")
        != "exact_zero_hurdle_16_bin_rqs_conditional_tail"
        or factor.get("receiver_path") != "flat_no_copy"
        or factor.get("pointer") != "FORBIDDEN"
        or factor.get("hierarchy") != "FORBIDDEN"
        or factor.get("y_balanced_likelihood") is not False
        or factor.get("coherence_loss") != "FORBIDDEN"
        or factor.get("c1_v1_retry") != "FORBIDDEN"
        or factor.get("c2_c3_c4") != "FORBIDDEN"
        or factor.get("h2") != "FORBIDDEN"
        or execution.get("explicit_device") != "cuda:0"
        or execution.get("gpu_inventory_query") != "FORBIDDEN"
        or execution.get("validation_sample_batch_size") != 128
        or execution.get("checkpoint_serialization") != "cpu_first"
        or execution.get("internal_test") != "FORBIDDEN"
        or execution.get("sparkov_fraud_test") != "FORBIDDEN"
        or correction.get("execution_class")
        != "FIRST_VALID_H1_EVALUATION_AFTER_TWO_DETERMINISTIC_IMPLEMENTATION_FAILURES"
        or correction.get("authorized_attempt") != CORRECTIVE_ATTEMPT
        or correction.get("provenance_statement")
        != "attempt_003_is_the_first_valid_H1_evaluation_after_two_deterministic_implementation_failures_not_architecture_or_tuning_retry"
        or correction.get("rqs_inverse", {}).get("failed_attempt")
        != "attempt_001"
        or correction.get("rqs_inverse", {}).get("corrected_source_commit")
        != "c809c069b5562e764bfc6e43f4f69e28b34a3404"
        or correction.get("rqs_inverse", {}).get("report_path")
        != "docs/benchmark_v2/cof_hcmttpp_v2_h1_rqs_inverse_correction.md"
        or correction.get("rqs_inverse", {}).get("report_sha256")
        != "e78df91540d25d53cf8153125caea676c2bdaf70cd93b4b6328489a947b040bf"
        or correction.get("spawn_ownership", {}).get("failed_attempt")
        != "attempt_002"
        or correction.get("spawn_ownership", {}).get("corrected_source_commit")
        != "9e8f737f2db3fbbf16adb3187475b8b08b10b06e"
        or correction.get("spawn_ownership", {}).get("report_path")
        != "docs/benchmark_v2/cof_hcmttpp_v2_h1_attempt_002_spawn_ownership_correction.md"
        or correction.get("spawn_ownership", {}).get("report_sha256")
        != "e3cef48e6b76f8dc05079307b98c142c8ffff90b91bdf89237532740f70ec3d3"
        or correction.get("generic_retry") is not False
        or correction.get("arbitrary_attempt") is not False
        or correction.get("attempt_sweep") is not False
        or correction.get("failed_attempts") != expected_prior
    ):
        raise H1RunnerContractError("runner source-only contract changed")
    source_config = root / str(raw.get("source_config_path"))
    model_path = root / str(raw.get("model_path"))
    if sha256_file(source_config) != raw.get("source_config_sha256"):
        raise H1RunnerContractError("source config hash mismatch")
    if sha256_file(model_path) != raw.get("model_sha256"):
        raise H1RunnerContractError("H1 model source hash mismatch")
    base = str(raw.get("source_preparation_commit", ""))
    if len(base) != 40 or not _base_commit_is_ancestor(root, base):
        raise H1RunnerContractError("source preparation commit is not an ancestor")


def build_execution_plan(
    config_path: Path, *, dataset: str, candidate_id: str
) -> ExecutionPlan:
    config_path = config_path.resolve()
    raw = _load_yaml(config_path)
    _validate_runner_config(config_path, raw)
    if candidate_id != "H1":
        raise H1RunnerContractError("only H1 is allowed")
    if dataset not in raw["datasets_allowed"]:
        raise H1RunnerContractError("dataset must be amlsim or sparkov")
    jobs = (
        ExecutionJob(
            dataset=dataset,
            candidate_id="H1",
            seed=int(raw["seed"]),
            attempt=str(raw["attempt"]),
        ),
    )
    return ExecutionPlan(
        repository_root=config_path.parents[2],
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        dataset_scope=dataset,
        jobs=jobs,
        raw=raw,
    )


def _validate_one_cell(plan: ExecutionPlan) -> ExecutionJob:
    if len(plan.jobs) != 1:
        raise H1RunnerContractError("one dataset H1 cell is required")
    job = plan.jobs[0]
    if (
        job.dataset != plan.dataset_scope
        or job.candidate_id != "H1"
        or job.seed != 4001
        or job.attempt != CORRECTIVE_ATTEMPT
    ):
        raise H1RunnerContractError("one dataset H1 cell contract changed")
    return job


def execution_plan_report(plan: ExecutionPlan) -> Mapping[str, Any]:
    job = _validate_one_cell(plan)
    return {
        "schema_version": "cof-hcmttpp-v2-h1-execution-plan-v1",
        "status": "PASS",
        "dataset_scope": plan.dataset_scope,
        "job_count": 1,
        "jobs": [
            {
                "dataset": job.dataset,
                "candidate_id": job.candidate_id,
                "seed": job.seed,
                "attempt": job.attempt,
            }
        ],
        "source_commit": current_source_commit(plan.repository_root),
        "runner_relevant_source_sha256": execution_relevant_source_sha256(
            plan.repository_root
        ),
        "runner_config_sha256": plan.config_sha256,
        "current_readiness_boundary_sha256": (
            execution_readiness_boundary_sha256(plan.repository_root)
        ),
        "direct_execution_allowed": False,
        "authorization_created": False,
        "runtime_artifacts_created": False,
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),
    }


def dry_run_execution(plan: ExecutionPlan) -> Mapping[str, Any]:
    """Check path existence and immutable bindings without opening NPZ bodies."""

    job = _validate_one_cell(plan)
    if sha256_file(plan.config_path) != plan.config_sha256:
        raise H1RunnerContractError("runner config changed after planning")
    record = plan.raw["datasets"][job.dataset]
    checked = []
    for role in (
        "train",
        "validation",
        "transform",
        "provenance",
        "summary",
        "sampling_plan",
        "threshold",
    ):
        path = plan.repository_root / str(record[f"{role}_path"])
        if not path.is_file():
            raise H1RunnerContractError(f"{job.dataset} {role} input is missing")
        checked.append(
            {
                "role": role,
                "path": str(path.relative_to(plan.repository_root)),
                "sha256_binding": record[f"{role}_sha256"],
                "body_opened": False,
            }
        )
    materialization_complete = (
        plan.repository_root
        / str(record["materialization_artifact_root"])
        / "COMPLETE.json"
    )
    if not materialization_complete.is_file():
        raise H1RunnerContractError(
            f"{job.dataset} materialization COMPLETE is missing"
        )
    report = dict(execution_plan_report(plan))
    report.update(
        {
            "schema_version": "cof-hcmttpp-v2-h1-execution-dry-run-v1",
            "authorization_gate": "SEPARATE_DATASET_SCOPED_AUTHORIZATION_REQUIRED",
            "candidate_gate": "H1_ONLY",
            "input_bindings": checked,
            "data_body_files_opened": [],
            "model_modules_imported": [],
            "gpu_or_cuda_probes": [],
            "runtime_paths_written": [],
            "test_paths_resolved": [],
            "implementation_correction_evidence": (
                verify_implementation_correction_evidence(plan)
            ),
        }
    )
    return report


def _job_authorization_record(plan: ExecutionPlan, job: ExecutionJob) -> Mapping[str, Any]:
    record = plan.raw["datasets"][job.dataset]
    roles = (
        "train",
        "validation",
        "transform",
        "provenance",
        "summary",
        "sampling_plan",
        "threshold",
    )
    return {
        "dataset": job.dataset,
        "candidate_id": job.candidate_id,
        "seed": job.seed,
        "attempt": job.attempt,
        "bundle_root": record["bundle_root"],
        "bundle_tree_sha256": record["bundle_tree_sha256"],
        "materialization_artifact_root": record["materialization_artifact_root"],
        "materialization_complete_sha256": record[
            "materialization_complete_sha256"
        ],
        **{
            key: record[key]
            for role in roles
            for key in (f"{role}_path", f"{role}_sha256")
        },
        "c0": dict(record["c0"]),
        "c1": dict(record["c1"]),
        "gate_references": dict(record["gate_references"]),
    }


def build_expected_authorization_claim(
    plan: ExecutionPlan, *, approval: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the exact in-memory claim; never create an authorization file."""

    job = _validate_one_cell(plan)
    claim: dict[str, Any] = {
        "schema_version": "cof-hcmttpp-v2-h1-execution-authorization-v3",
        "authorized_action": "EXTERNAL_VALIDATION_H1_FIRST_VALID_ATTEMPT_003",
        "source_commit": current_source_commit(plan.repository_root),
        "runner_relevant_source_sha256": execution_relevant_source_sha256(
            plan.repository_root
        ),
        "runner_config_sha256": plan.config_sha256,
        "current_readiness_boundary_sha256": (
            execution_readiness_boundary_sha256(plan.repository_root)
        ),
        "source_config_sha256": plan.raw["source_config_sha256"],
        "model_sha256": plan.raw["model_sha256"],
        "runtime_root": plan.raw["runtime_root"],
        "authorization_history_root": plan.raw["authorization_history_root"],
        "dataset_scope": job.dataset,
        "candidate_id": "H1",
        "seed": 4001,
        "attempt": CORRECTIVE_ATTEMPT,
        "jobs": [_job_authorization_record(plan, job)],
        "implementation_corrections": dict(
            plan.raw["implementation_corrections"]
        ),
        "factor_isolation": dict(plan.raw["factor_isolation"]),
        "training": dict(plan.raw["training"]),
        "scope": dict(AUTHORIZATION_SCOPE),
        "approval": dict(
            approval
            or {
                "approved": False,
                "text": "SEPARATE_EXPLICIT_DATASET_SCOPED_H1_APPROVAL_REQUIRED",
            }
        ),
    }
    claim["authorization_sha256"] = canonical_sha256(claim)
    return claim


def validate_authorization_document(
    plan: ExecutionPlan, authorization: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Validate identity before touching any data/model/device dependency."""

    job = _validate_one_cell(plan)
    if authorization.get("dataset_scope") != job.dataset:
        raise H1RunnerContractError("dataset-scoped authorization mismatch")
    approval = authorization.get("approval")
    if (
        not isinstance(approval, Mapping)
        or approval.get("approved") is not True
        or not isinstance(approval.get("text"), str)
        or "H1" not in approval["text"]
    ):
        raise H1RunnerContractError("authorization lacks explicit H1 approval")
    expected = build_expected_authorization_claim(plan, approval=approval)
    if dict(authorization) != expected:
        raise H1RunnerContractError("authorization hash-bound identity mismatch")
    return {
        "status": "PASS",
        "dataset_scope": job.dataset,
        "candidate_id": "H1",
        "seed": 4001,
        "attempt": CORRECTIVE_ATTEMPT,
        "authorization_sha256": expected["authorization_sha256"],
        "document": dict(authorization),
    }


def validate_authorization_path(
    plan: ExecutionPlan, authorization_path: Path
) -> Mapping[str, Any]:
    configured = Path(str(plan.raw["authorization_history_root"]))
    history_root = (
        configured if configured.is_absolute() else plan.repository_root / configured
    ).resolve()
    try:
        resolved = authorization_path.resolve(strict=True)
        resolved.relative_to(history_root)
    except (OSError, ValueError) as error:
        raise H1RunnerContractError(
            "authorization must be inside append-only history"
        ) from error
    try:
        payload = resolved.read_bytes()
        document = json.loads(payload.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise H1RunnerContractError("authorization document is invalid") from error
    if not isinstance(document, Mapping):
        raise H1RunnerContractError("authorization document must be an object")

    # Identity validation intentionally precedes all frozen input hashing.
    result = dict(validate_authorization_document(plan, document))
    job = _validate_one_cell(plan)
    record = plan.raw["datasets"][job.dataset]
    for role in (
        "train",
        "validation",
        "transform",
        "provenance",
        "summary",
        "sampling_plan",
        "threshold",
    ):
        path = plan.repository_root / str(record[f"{role}_path"])
        if sha256_file(path) != record[f"{role}_sha256"]:
            raise H1RunnerContractError(f"authorized {role} input hash changed")
    materialization_complete = (
        plan.repository_root
        / str(record["materialization_artifact_root"])
        / "COMPLETE.json"
    )
    if sha256_file(materialization_complete) != record[
        "materialization_complete_sha256"
    ]:
        raise H1RunnerContractError("authorized materialization COMPLETE changed")
    result["implementation_correction_evidence"] = (
        verify_implementation_correction_evidence(plan)
    )
    for parent, filenames in (
        (
            "c0",
            {
                "manifest_sha256": "manifest.json",
                "sample_sha256": "sample.npz",
                "evaluation_sha256": "evaluation.json",
                "complete_sha256": "COMPLETE.json",
            },
        ),
        (
            "c1",
            {
                "manifest_sha256": "manifest.json",
                "sample_sha256": "validation_sample.npz",
                "evaluation_sha256": "evaluation.json",
                "gate_sha256": "gate_decision.json",
                "complete_sha256": "COMPLETE.json",
            },
        ),
    ):
        evidence = record[parent]
        attempt = plan.repository_root / str(evidence["attempt_path"])
        for hash_key, filename in filenames.items():
            if sha256_file(attempt / filename) != evidence[hash_key]:
                raise H1RunnerContractError(
                    f"authorized frozen {parent} evidence hash changed: {filename}"
                )
    result.update(
        {
            "authorization_path": str(resolved),
            "authorization_file_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    return result


def resolve_dataset_access(
    *,
    plan: ExecutionPlan,
    split: str,
    purpose: str,
    final_checkpoint_complete: bool,
) -> Path:
    """Resolve only authorized train/validation paths; test is pre-resolution blocked."""

    job = _validate_one_cell(plan)
    normalized = str(split).lower().replace("-", "_")
    forbidden_tokens = (
        "test",
        "fraudtest",
        "fraud_test",
        "internal_test",
        "fresh_test",
    )
    if any(token in normalized for token in forbidden_tokens):
        raise H1RunnerContractError("forbidden test path before resolution")
    if normalized == "train":
        if purpose != "tail_state_fit_and_training":
            raise H1RunnerContractError("train purpose is invalid")
    elif normalized == "validation":
        if not final_checkpoint_complete:
            raise H1RunnerContractError("validation requires final checkpoint")
        if purpose != "fixed_plan_sampling_and_evaluation":
            raise H1RunnerContractError("validation purpose is invalid")
    else:
        raise H1RunnerContractError("forbidden split before resolution")
    relative = str(plan.raw["datasets"][job.dataset][f"{normalized}_path"])
    if any(token in relative.lower() for token in forbidden_tokens):
        raise H1RunnerContractError("forbidden test path before resolution")
    return (plan.repository_root / relative).resolve()


def build_checkpoint_provenance(
    *,
    plan: ExecutionPlan,
    job: ExecutionJob,
    authorization_sha256: str,
    train_state_binding: Mapping[str, Any],
    actual_updates: int,
    checkpoint_sha256: str,
) -> Mapping[str, Any]:
    if job != _validate_one_cell(plan):
        raise H1RunnerContractError("checkpoint job is outside H1 plan")
    if (
        train_state_binding.get("fit_split") != "train"
        or any(
            int(train_state_binding.get(key, -1)) != 0
            for key in (
                "validation_rows_used",
                "internal_test_rows_used",
                "fraud_test_rows_used",
            )
        )
        or actual_updates != int(plan.raw["training"]["requested_updates"])
        or any(len(value) != 64 for value in (authorization_sha256, checkpoint_sha256))
    ):
        raise H1RunnerContractError("checkpoint provenance input is invalid")
    payload = {
        "schema_version": "cof-hcmttpp-v2-h1-checkpoint-provenance-v1",
        "source_commit": current_source_commit(plan.repository_root),
        "runner_relevant_source_sha256": execution_relevant_source_sha256(
            plan.repository_root
        ),
        "runner_config_sha256": plan.config_sha256,
        "source_config_sha256": plan.raw["source_config_sha256"],
        "model_sha256": plan.raw["model_sha256"],
        "dataset": job.dataset,
        "candidate_id": "H1",
        "seed": 4001,
        "attempt": job.attempt,
        "authorization_sha256": authorization_sha256,
        **dict(train_state_binding),
        "actual_updates": actual_updates,
        "checkpoint_sha256": checkpoint_sha256,
    }
    return {**payload, "provenance_sha256": canonical_sha256(payload)}


def validate_checkpoint_provenance(
    provenance: Mapping[str, Any],
    *,
    plan: ExecutionPlan,
    job: ExecutionJob,
    authorization_sha256: str,
    train_state_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        expected = build_checkpoint_provenance(
            plan=plan,
            job=job,
            authorization_sha256=authorization_sha256,
            train_state_binding=train_state_binding,
            actual_updates=int(plan.raw["training"]["requested_updates"]),
            checkpoint_sha256=str(provenance["checkpoint_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise H1RunnerContractError("checkpoint provenance is incomplete") from error
    if dict(provenance) != expected:
        raise H1RunnerContractError("checkpoint provenance mismatch")
    return {"status": "PASS", "provenance_sha256": expected["provenance_sha256"]}


def execute_authorized(
    *,
    plan: ExecutionPlan,
    authorization_path: Path | None,
    dependencies: ExecutionDependencies | Callable[[], ExecutionDependencies],
) -> Mapping[str, Any]:
    """Execute only after the exact dataset authorization has validated."""

    if authorization_path is None:
        raise H1RunnerContractError("execute requires authorization before access")
    authorization_record = validate_authorization_path(plan, authorization_path)
    authorization = authorization_record["document"]
    active = dependencies() if callable(dependencies) else dependencies
    if not isinstance(active, ExecutionDependencies):
        raise H1RunnerContractError("execution dependency factory is invalid")
    job = _validate_one_cell(plan)
    runtime_root = Path(str(plan.raw["runtime_root"]))
    if not runtime_root.is_absolute():
        runtime_root = plan.repository_root / runtime_root
    provenance = {
        "authorization_sha256": authorization["authorization_sha256"],
        "source_commit": authorization["source_commit"],
        "runner_relevant_source_sha256": authorization[
            "runner_relevant_source_sha256"
        ],
        "runner_config_sha256": authorization["runner_config_sha256"],
        "source_config_sha256": authorization["source_config_sha256"],
        "model_sha256": authorization["model_sha256"],
        "job": authorization["jobs"][0],
    }
    provenance_sha256 = canonical_sha256(provenance)
    store = H1AttemptStore.claim(
        runtime_root=runtime_root,
        job=job,
        authorization_sha256=authorization["authorization_sha256"],
        provenance_sha256=provenance_sha256,
    )
    attempt = store.allocate_attempt(
        {
            "schema_version": "cof-hcmttpp-v2-h1-manifest-v1",
            "dataset": job.dataset,
            "candidate_id": "H1",
            "seed": 4001,
            "attempt": job.attempt,
            "authorization_file_sha256": authorization_record[
                "authorization_file_sha256"
            ],
            "authorization_sha256": authorization["authorization_sha256"],
            "provenance_sha256": provenance_sha256,
            "source_commit": authorization["source_commit"],
            "runner_relevant_source_sha256": authorization[
                "runner_relevant_source_sha256"
            ],
            "runner_config_sha256": plan.config_sha256,
            "source_config_sha256": plan.raw["source_config_sha256"],
            "model_sha256": plan.raw["model_sha256"],
            "input_binding": authorization["jobs"][0],
            "factor_isolation": plan.raw["factor_isolation"],
            "training": plan.raw["training"],
            "forbidden_access": {
                "internal_test": True,
                "sparkov_fraud_test": True,
            },
        }
    )
    watchdog = plan.raw["watchdog"]
    child = RunnerChildSpec(
        target=_authorized_dependency_child,
        payload={
            "dependencies": active,
            "plan": plan,
            "job": job,
            "authorization": authorization,
            "runtime_root": str(runtime_root),
            "ownership_path": str(store.ownership_path),
            "attempt_path": str(attempt),
        },
    )
    terminal = run_runner_owned_child(
        child_spec=child,
        store=store,
        max_wall_seconds=float(plan.raw["training"]["max_wall_seconds"]),
        startup_timeout_seconds=float(watchdog["startup_timeout_seconds"]),
        progress_timeout_seconds=float(watchdog["progress_timeout_seconds"]),
        queue_poll_seconds=float(watchdog["queue_poll_seconds"]),
        termination_grace_seconds=float(watchdog["termination_grace_seconds"]),
        kill_grace_seconds=float(watchdog["kill_grace_seconds"]),
    )
    return {
        "status": "TERMINAL",
        "dataset": job.dataset,
        "candidate_id": "H1",
        "attempt_path": str(attempt),
        "terminal": terminal,
    }


def _authorized_dependency_child(
    payload: Mapping[str, Any], event_queue: Any
) -> Mapping[str, Any]:
    dependencies = payload["dependencies"]
    plan = payload["plan"]
    job = payload["job"]
    authorization = payload["authorization"]
    data = dependencies.load_train_body(
        plan=plan, job=job, authorization=authorization
    )
    model = dependencies.build_model(plan=plan, job=job, data=data)
    device = dependencies.select_device(
        plan=plan, job=job, authorization=authorization
    )
    event_queue.put({"kind": "setup_complete", "dataset": job.dataset})
    return dependencies.run_job(
        plan=plan,
        job=job,
        authorization=authorization,
        data=data,
        model=model,
        device=device,
        ownership_path=Path(str(payload["ownership_path"])),
        attempt_path=Path(str(payload["attempt_path"])),
        event_queue=event_queue,
    )
