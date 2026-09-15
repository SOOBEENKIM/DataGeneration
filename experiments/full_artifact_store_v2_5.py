from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


SCHEMA_VERSION = "benchmark-v2.5-full-attempt"
MANIFEST_REQUIRED = {
    "schema_version",
    "git_commit",
    "config_hash",
    "code_hash",
    "scenario",
    "kappa",
    "generator",
    "seed",
    "sampling_plan_hash",
    "data_hashes",
    "gpu",
    "cuda_version",
    "pytorch_version",
    "requested_training_budget",
    "actual_training_budget",
    "baseline_definition_version",
    "evaluation_version",
}
DATA_HASH_KEYS = {"train", "validation", "test"}
TERMINAL_FILES = (
    "CANCELLED.json",
    "COMPLETE.json",
    "FAILED.json",
    "INVALID.json",
    "INTERRUPTED.json",
    "UNAVAILABLE.json",
)
COMPLETION_REQUIRED = {
    "sample.npz",
    "checkpoints/final.pt",
    "metrics.json",
    "runtime.json",
    "evaluation.json",
}


class ArtifactContractError(RuntimeError):
    pass


class ArtifactCorruptionError(ArtifactContractError):
    pass


class RunAlreadyComplete(ArtifactContractError):
    pass


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _atomic_replace_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = MANIFEST_REQUIRED - manifest.keys()
    if missing:
        raise ArtifactContractError(
            f"manifest is missing required fields: {sorted(missing)}"
        )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ArtifactContractError("unexpected v2.5 manifest schema")
    if manifest["scenario"] != "joint_semimarkov_v2b":
        raise ArtifactContractError("v2.5 permits only joint_semimarkov_v2b")
    if float(manifest["kappa"]) != 1.0:
        raise ArtifactContractError("v2.5 permits only kappa=1.0")
    if int(manifest["seed"]) not in range(1, 6):
        raise ArtifactContractError("v2.5 seed must be one of 1..5")
    data_hashes = manifest["data_hashes"]
    if not isinstance(data_hashes, Mapping):
        raise ArtifactContractError("data_hashes must be a mapping")
    if set(data_hashes) != DATA_HASH_KEYS:
        raise ArtifactContractError(
            "train/validation/test hashes must all be present and no others"
        )
    for key, value in data_hashes.items():
        if not isinstance(value, str) or len(value) != 64:
            raise ArtifactContractError(f"invalid {key} data hash")
    for key in ("git_commit", "config_hash", "code_hash"):
        value = manifest[key]
        expected = 40 if key == "git_commit" else 64
        if not isinstance(value, str) or len(value) != expected:
            raise ArtifactContractError(f"invalid {key}")
    plan_hash = manifest["sampling_plan_hash"]
    if not isinstance(plan_hash, str) or len(plan_hash) != 64:
        raise ArtifactContractError("invalid shared SamplingPlan hash")


def run_directory(
    artifact_root: str | Path,
    *,
    scenario: str,
    kappa: float,
    generator: str,
    seed: int,
) -> Path:
    root = Path(artifact_root)
    if root.name != "benchmark_v2_5":
        raise ArtifactContractError(
            "v2.5 runtime artifacts require a benchmark_v2_5 root"
        )
    return (
        root
        / "full"
        / scenario
        / f"kappa_{kappa:.2f}"
        / generator
        / f"seed_{seed}"
    )


@dataclass(frozen=True)
class AttemptSelection:
    path: Path
    attempt: int
    resume: bool
    latest_checkpoint: Path | None


class FullAttemptStore:
    """Immutable attempt allocator and crash-safe artifact writer for v2.5."""

    def __init__(self, attempt_path: Path, manifest: Mapping[str, Any]):
        self.path = attempt_path
        self.manifest = dict(manifest)

    @classmethod
    def select(
        cls,
        artifact_root: str | Path,
        manifest: Mapping[str, Any],
        *,
        minimum_attempt: int = 1,
    ) -> tuple["FullAttemptStore", AttemptSelection]:
        validate_manifest(manifest)
        if minimum_attempt < 1:
            raise ValueError("minimum attempt must be positive")
        base = run_directory(
            artifact_root,
            scenario=str(manifest["scenario"]),
            kappa=float(manifest["kappa"]),
            generator=str(manifest["generator"]),
            seed=int(manifest["seed"]),
        )
        base.mkdir(parents=True, exist_ok=True)
        attempts = sorted(
            path
            for path in base.glob("attempt_[0-9][0-9][0-9]")
            if path.is_dir()
        )
        resume_checkpoint_for_new_attempt: Path | None = None
        if attempts:
            latest = attempts[-1]
            number = int(latest.name[len("attempt_") :])
            latest_manifest_path = latest / "manifest.json"
            if not latest_manifest_path.is_file():
                raise ArtifactCorruptionError(
                    f"attempt has no immutable manifest: {latest}"
                )
            try:
                existing = json.loads(latest_manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ArtifactCorruptionError(
                    f"cannot read attempt manifest: {latest}"
                ) from error
            terminal = [name for name in TERMINAL_FILES if (latest / name).exists()]
            if len(terminal) > 1:
                raise ArtifactCorruptionError(
                    f"attempt has conflicting terminal markers: {latest}"
                )
            same_manifest = canonical_json_bytes(existing) == canonical_json_bytes(
                manifest
            )
            if same_manifest and terminal == ["COMPLETE.json"]:
                raise RunAlreadyComplete(str(latest))
            if same_manifest and terminal == ["UNAVAILABLE.json"]:
                raise ArtifactContractError(
                    f"mandatory baseline remains unavailable: {latest}"
                )
            if same_manifest and terminal == ["INVALID.json"]:
                raise ArtifactContractError(
                    f"invalid seed cannot be retried under the same manifest: {latest}"
                )
            if same_manifest and terminal == ["CANCELLED.json"]:
                raise ArtifactContractError(
                    f"cancelled seed is already terminal: {latest}"
                )
            if same_manifest and terminal == ["FAILED.json"]:
                failure = json.loads((latest / "FAILED.json").read_text())
                if failure.get("mandatory_stop"):
                    raise ArtifactContractError(
                        "three consecutive identical infrastructure failures"
                    )
            if same_manifest and terminal == ["INTERRUPTED.json"]:
                pointer = latest / "checkpoints" / "latest"
                if pointer.exists():
                    try:
                        value = json.loads(pointer.read_text())
                        checkpoint = latest / "checkpoints" / value["path"]
                    except (OSError, KeyError, json.JSONDecodeError) as error:
                        raise ArtifactCorruptionError(
                            f"invalid latest checkpoint pointer: {pointer}"
                        ) from error
                    if not checkpoint.is_file():
                        raise ArtifactCorruptionError(
                            f"latest checkpoint is missing: {checkpoint}"
                        )
                    if sha256_file(checkpoint) != value.get("sha256"):
                        raise ArtifactCorruptionError(
                            f"latest checkpoint hash mismatch: {checkpoint}"
                        )
                    resume_checkpoint_for_new_attempt = checkpoint
            if (
                same_manifest
                and not terminal
                and number >= minimum_attempt
            ):
                pointer = latest / "checkpoints" / "latest"
                checkpoint = None
                if pointer.exists():
                    try:
                        value = json.loads(pointer.read_text())
                        checkpoint = latest / "checkpoints" / value["path"]
                    except (OSError, KeyError, json.JSONDecodeError) as error:
                        raise ArtifactCorruptionError(
                            f"invalid latest checkpoint pointer: {pointer}"
                        ) from error
                    if not checkpoint.is_file():
                        raise ArtifactCorruptionError(
                            f"latest checkpoint is missing: {checkpoint}"
                        )
                    if sha256_file(checkpoint) != value.get("sha256"):
                        raise ArtifactCorruptionError(
                            f"latest checkpoint hash mismatch: {checkpoint}"
                        )
                selection = AttemptSelection(
                    latest,
                    number,
                    True,
                    checkpoint,
                )
                return cls(latest, manifest), selection
            next_number = max(number + 1, minimum_attempt)
        else:
            next_number = minimum_attempt
        attempt = base / f"attempt_{next_number:03d}"
        attempt.mkdir(parents=False, exist_ok=False)
        (attempt / "checkpoints").mkdir()
        _exclusive_bytes(
            attempt / "manifest.json",
            canonical_json_bytes(manifest),
        )
        _exclusive_bytes(
            attempt / "RUNNING.json",
            canonical_json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "attempt": next_number,
                    "status": "RUNNING",
                }
            ),
        )
        _exclusive_bytes(attempt / "stdout.log", b"")
        _exclusive_bytes(attempt / "stderr.log", b"")
        selection = AttemptSelection(attempt, next_number, False, None)
        if resume_checkpoint_for_new_attempt is not None:
            selection = AttemptSelection(
                attempt,
                next_number,
                True,
                resume_checkpoint_for_new_attempt,
            )
        return cls(attempt, manifest), selection

    def _ensure_nonterminal(self) -> None:
        terminals = [name for name in TERMINAL_FILES if (self.path / name).exists()]
        if terminals:
            raise ArtifactContractError(
                f"attempt is already terminal: {terminals[0]}"
            )

    def append_log(self, stream: str, text: str) -> None:
        self._ensure_nonterminal()
        if stream not in ("stdout", "stderr"):
            raise ValueError("stream must be stdout or stderr")
        path = self.path / f"{stream}.log"
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
        try:
            payload = text.encode()
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def append_progress(self, record: Mapping[str, Any]) -> None:
        self._ensure_nonterminal()
        required = {
            "step",
            "loss",
            "validation_metric",
            "elapsed_seconds",
            "peak_gpu_memory_bytes",
        }
        missing = required - record.keys()
        if missing:
            raise ArtifactContractError(
                f"progress record missing {sorted(missing)}"
            )
        payload = canonical_json_bytes(record)
        path = self.path / "progress.jsonl"
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def write_partial_metrics(self, value: Mapping[str, Any]) -> Path:
        self._ensure_nonterminal()
        return _atomic_replace_bytes(
            self.path / "partial_metrics.json",
            canonical_json_bytes(value),
        )

    def write_heartbeat(self, value: Mapping[str, Any]) -> Path:
        self._ensure_nonterminal()
        if value.get("status") != "RUNNING":
            raise ArtifactContractError("heartbeat status must be RUNNING")
        return _atomic_replace_bytes(
            self.path / "heartbeat.json",
            canonical_json_bytes(value),
        )

    def append_resource(self, record: Mapping[str, Any]) -> None:
        self._ensure_nonterminal()
        required = {
            "elapsed_seconds",
            "cpu_rss_bytes",
            "gpu_memory_bytes",
            "gpu_utilization_percent",
        }
        missing = required - record.keys()
        if missing:
            raise ArtifactContractError(
                f"resource record missing {sorted(missing)}"
            )
        path = self.path / "resource.jsonl"
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        try:
            os.write(descriptor, canonical_json_bytes(record))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def append_artifact_index_entry(
        self,
        relative_path: str,
        *,
        role: str,
    ) -> dict[str, Any]:
        self._ensure_nonterminal()
        base = self.path.resolve()
        target = (self.path / relative_path).resolve()
        if target == base or base not in target.parents:
            raise ArtifactContractError("artifact path escapes attempt")
        if not target.is_file():
            raise ArtifactContractError(
                f"indexed artifact does not exist: {target}"
            )
        relative = target.relative_to(base).as_posix()
        entry = {
            "path": relative,
            "role": role,
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
        index = self.path / "artifact_index.jsonl"
        descriptor = os.open(
            index,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        try:
            os.write(descriptor, canonical_json_bytes(entry))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return entry

    def write_checkpoint(
        self,
        *,
        step: int,
        writer: Callable[[Path], None],
        suffix: str = ".pt",
    ) -> Path:
        self._ensure_nonterminal()
        if step < 1:
            raise ValueError("checkpoint step must be positive")
        target = self.path / "checkpoints" / f"step_{step:08d}{suffix}"
        if target.exists():
            raise ArtifactContractError(
                f"checkpoint would be overwritten: {target}"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            dir=target.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            writer(temporary)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise ArtifactContractError("checkpoint writer produced no data")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        pointer = {
            "path": target.name,
            "sha256": sha256_file(target),
            "step": step,
        }
        _atomic_replace_bytes(
            self.path / "checkpoints" / "latest",
            canonical_json_bytes(pointer),
        )
        return target

    def write_immutable_bytes(self, relative_path: str, payload: bytes) -> Path:
        self._ensure_nonterminal()
        base = self.path.resolve()
        target = (self.path / relative_path).resolve()
        if target == base or base not in target.parents:
            raise ArtifactContractError("artifact path escapes attempt")
        return _exclusive_bytes(target, payload)

    def write_immutable_file(
        self,
        relative_path: str,
        writer: Callable[[Path], None],
    ) -> Path:
        self._ensure_nonterminal()
        base = self.path.resolve()
        target = (self.path / relative_path).resolve()
        if target == base or base not in target.parents:
            raise ArtifactContractError("artifact path escapes attempt")
        if target.exists():
            raise ArtifactContractError(
                f"artifact would be overwritten: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            dir=target.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            writer(temporary)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise ArtifactContractError("artifact writer produced no data")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def write_immutable_json(
        self,
        relative_path: str,
        value: Mapping[str, Any],
    ) -> Path:
        return self.write_immutable_bytes(
            relative_path,
            canonical_json_bytes(value),
        )

    def complete(self, value: Mapping[str, Any]) -> Path:
        self._ensure_nonterminal()
        missing = [
            relative
            for relative in sorted(COMPLETION_REQUIRED)
            if not (self.path / relative).is_file()
        ]
        if missing:
            raise ArtifactContractError(
                f"cannot complete attempt; missing {missing}"
            )
        contracts = value.get("hard_guards")
        if not isinstance(contracts, Mapping) or any(
            result != "PASS" for result in contracts.values()
        ):
            raise ArtifactContractError(
                "COMPLETE requires every hard guard to be PASS"
            )
        hashes = {
            relative: sha256_file(self.path / relative)
            for relative in sorted(COMPLETION_REQUIRED)
        }
        return _exclusive_bytes(
            self.path / "COMPLETE.json",
            canonical_json_bytes(
                {
                    **value,
                    "schema_version": SCHEMA_VERSION,
                    "status": "COMPLETE",
                    "artifact_hashes": hashes,
                }
            ),
        )

    def fail(
        self,
        *,
        exception: str,
        last_checkpoint: str | None,
        peak_gpu_memory_bytes: int,
        failure_class: str = "code",
        failure_fingerprint: str | None = None,
        actual_elapsed_seconds: float | None = None,
        runner_owned_child_terminated: bool = False,
    ) -> Path:
        self._ensure_nonterminal()
        if failure_class not in {
            "infrastructure",
            "oom",
            "code",
            "contract",
            "wall_cap",
        }:
            raise ValueError("unknown failure class")
        fingerprint = failure_fingerprint or hashlib.sha256(
            exception.encode()
        ).hexdigest()
        consecutive = 1
        if failure_class == "infrastructure":
            base = self.path.parent
            previous = sorted(
                path
                for path in base.glob("attempt_[0-9][0-9][0-9]")
                if path.is_dir() and path != self.path
            )
            for attempt in reversed(previous):
                marker = attempt / "FAILED.json"
                if not marker.is_file():
                    break
                try:
                    value = json.loads(marker.read_text())
                except (OSError, json.JSONDecodeError):
                    break
                if (
                    value.get("failure_class") != "infrastructure"
                    or value.get("failure_fingerprint") != fingerprint
                ):
                    break
                consecutive += 1
        return _exclusive_bytes(
            self.path / "FAILED.json",
            canonical_json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "FAILED",
                    "exception": exception,
                    "last_checkpoint": last_checkpoint,
                    "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
                    "failure_class": failure_class,
                    "failure_fingerprint": fingerprint,
                    "actual_elapsed_seconds": actual_elapsed_seconds,
                    "runner_owned_child_terminated": (
                        runner_owned_child_terminated
                    ),
                    "consecutive_identical_infrastructure_failures": (
                        consecutive if failure_class == "infrastructure" else 0
                    ),
                    "mandatory_stop": bool(
                        failure_class == "infrastructure"
                        and consecutive >= 3
                    ),
                    "stdout_path": "stdout.log",
                    "stderr_path": "stderr.log",
                }
            ),
        )

    def unavailable(self, *, reason: str) -> Path:
        self._ensure_nonterminal()
        return _exclusive_bytes(
            self.path / "UNAVAILABLE.json",
            canonical_json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "UNAVAILABLE",
                    "reason": reason,
                    "full_experiment_must_stop": False,
                    "generator_remaining_seeds_must_cancel": True,
                    "stdout_path": "stdout.log",
                    "stderr_path": "stderr.log",
                }
            ),
        )

    def invalid(
        self,
        *,
        reason: str,
        hard_guards: Mapping[str, str],
    ) -> Path:
        self._ensure_nonterminal()
        if not hard_guards or all(
            value == "PASS" for value in hard_guards.values()
        ):
            raise ArtifactContractError(
                "INVALID requires at least one failed hard guard"
            )
        return _exclusive_bytes(
            self.path / "INVALID.json",
            canonical_json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "INVALID",
                    "reason": reason,
                    "hard_guards": dict(hard_guards),
                    "full_experiment_seed_valid": False,
                    "stdout_path": "stdout.log",
                    "stderr_path": "stderr.log",
                }
            ),
        )

    def interrupted(
        self,
        *,
        interruption: str,
        last_checkpoint: str | None,
        actual_elapsed_seconds: float,
    ) -> Path:
        self._ensure_nonterminal()
        if interruption not in {"keyboard", "system_exit", "sigterm"}:
            raise ValueError("unknown operator interruption")
        return _exclusive_bytes(
            self.path / "INTERRUPTED.json",
            canonical_json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "INTERRUPTED",
                    "interruption": interruption,
                    "last_checkpoint": last_checkpoint,
                    "actual_elapsed_seconds": actual_elapsed_seconds,
                    "operator_initiated": True,
                    "stdout_path": "stdout.log",
                    "stderr_path": "stderr.log",
                }
            ),
        )

    def cancelled(self, *, reason: str) -> Path:
        self._ensure_nonterminal()
        return _exclusive_bytes(
            self.path / "CANCELLED.json",
            canonical_json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "CANCELLED",
                    "reason": reason,
                    "explicitly_cancelled": True,
                    "stdout_path": "stdout.log",
                    "stderr_path": "stderr.log",
                }
            ),
        )
