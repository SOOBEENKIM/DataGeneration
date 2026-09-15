from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml


CONFIG_PATH = Path("configs/benchmark_v2/full_v2_5.yaml")
ARTIFACT_ROOT = Path("artifacts/benchmark_v2_5")
DATA_MANIFEST_PATH = Path(
    "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
    "kappa_1.00/data_manifest.json"
)
AUTHORIZATION_PATH = Path(
    "artifacts/benchmark_v2_5/full/authorization/authorization.json"
)
SCENARIO = "joint_semimarkov_v2b"
KAPPA = 1.0
SEEDS = (1, 2, 3, 4, 5)
CPU_GENERATORS = (
    "empirical_iid",
    "block_2",
    "block_4",
    "block_8",
    "full_sequence_reference",
    "independent_markov",
    "joint_markov",
    "plug_in_hmm",
    "plug_in_hsmm",
)
GPU_GENERATORS = (
    "ctgan_separate_class",
    "tvae_separate_class",
    "neural_sequence",
    "cof_seqgen",
)
GENERATOR_ORDER = CPU_GENERATORS + GPU_GENERATORS
V2_4_INDEX_PATH = Path("artifacts/benchmark_v2_4/artifact_index.json")
V2_4_GATE_REPORT_PATH = Path(
    "artifacts/benchmark_v2_4/gates/gate_report.json"
)
CAPACITY_INDEX_PATH = Path(
    "artifacts/benchmark_v2_5/capacity_preflight/artifact_index.json"
)
CAPACITY_CORRECTION_PATH = Path(
    "artifacts/benchmark_v2_5/capacity_preflight/"
    "capacity_preflight_correction_001.md"
)
PRIMARY_COMPARATORS = (
    "ctgan_separate_class",
    "tvae_separate_class",
    "empirical_iid",
)
SECONDARY_COMPARATORS = (
    "neural_sequence",
    "independent_markov",
    "joint_markov",
    "plug_in_hmm",
    "plug_in_hsmm",
)
C2_NOT_SUPPORTED = (
    "C2 was not supported in this preregistered experiment."
)
C2_NOT_EVALUABLE = (
    "C2 was not evaluable in this preregistered experiment."
)
PRIMARY_C2_GENERATORS = {
    "empirical_iid",
    "ctgan_separate_class",
    "tvae_separate_class",
    "cof_seqgen",
}
RUN_TERMINAL_STATUSES = {
    "COMPLETE",
    "FAILED",
    "INVALID",
    "UNAVAILABLE",
    "CANCELLED",
}
GPU_MAX_CONCURRENCY = 4
GPU_ZERO_IDLE_WAIT_BUDGET_SECONDS = 1800.0
GPU_POLL_INTERVAL_SECONDS = 30.0


class AuthorizationError(RuntimeError):
    """Full execution authorization is absent or differs from frozen inputs."""


class RunnerInterrupted(RuntimeError):
    """An operator interrupt from an owned child propagated to the parent."""

    def __init__(self, interruption: str, marker: str) -> None:
        super().__init__(f"runner interrupted by {interruption}")
        self.interruption = interruption
        self.marker = marker


class _OwnedSigterm(BaseException):
    pass


class _WatchdogSigterm(BaseException):
    pass


def _owned_process_entry(
    result_queue: Any,
    wall_cap_requested: Any,
    worker: Callable[..., Mapping[str, Any]],
    worker_kwargs: Mapping[str, Any],
    attempt_path: str,
    manifest: Mapping[str, Any],
) -> None:
    from experiments.full_artifact_store_v2_5 import FullAttemptStore

    started = time.monotonic()

    def handle_sigterm(unused_signum: int, unused_frame: Any) -> None:
        if wall_cap_requested.is_set():
            raise _WatchdogSigterm()
        raise _OwnedSigterm()

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        result_queue.put(
            {"kind": "result", "value": dict(worker(**worker_kwargs))}
        )
    except _WatchdogSigterm:
        return
    except (KeyboardInterrupt, SystemExit, _OwnedSigterm) as error:
        if isinstance(error, KeyboardInterrupt):
            interruption = "keyboard"
        elif isinstance(error, SystemExit):
            interruption = "system_exit"
        else:
            interruption = "sigterm"
        store = FullAttemptStore(Path(attempt_path), manifest)
        marker = store.interrupted(
            interruption=interruption,
            last_checkpoint=_last_checkpoint_name(store.path),
            actual_elapsed_seconds=time.monotonic() - started,
        )
        result_queue.put(
            {
                "kind": "interrupted",
                "interruption": interruption,
                "marker": str(marker),
            }
        )
    except Exception as error:
        result_queue.put(
            {
                "kind": "exception",
                "exception": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )


def _last_checkpoint_name(attempt_path: Path) -> str | None:
    pointer = attempt_path / "checkpoints" / "latest"
    if not pointer.is_file():
        return None
    try:
        return str(json.loads(pointer.read_text())["path"])
    except (KeyError, OSError, json.JSONDecodeError):
        return None


def _last_recorded_gpu_memory(attempt_path: Path) -> int:
    resource_path = attempt_path / "resource.jsonl"
    if not resource_path.is_file():
        return 0
    try:
        lines = resource_path.read_text().splitlines()
        return (
            int(json.loads(lines[-1])["gpu_memory_bytes"])
            if lines
            else 0
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return 0


def run_owned_attempt_process(
    *,
    worker: Callable[..., Mapping[str, Any]],
    worker_kwargs: Mapping[str, Any],
    store: Any,
    max_wall_seconds: float,
    termination_grace_seconds: float = 5.0,
    process_start_method: str = "spawn",
    operator_stop_event: threading.Event | None = None,
) -> Mapping[str, Any]:
    """Run one cell-seed in a runner-owned process under a hard deadline."""
    if max_wall_seconds <= 0 or termination_grace_seconds <= 0:
        raise ValueError("watchdog timing values must be positive")
    import multiprocessing

    context = multiprocessing.get_context(process_start_method)
    result_queue = context.Queue()
    wall_cap_requested = context.Event()
    process = context.Process(
        target=_owned_process_entry,
        args=(
            result_queue,
            wall_cap_requested,
            worker,
            dict(worker_kwargs),
            str(store.path),
            dict(store.manifest),
        ),
    )
    started = time.monotonic()
    deadline = started + max_wall_seconds
    process.start()
    try:
        while process.is_alive():
            if (
                operator_stop_event is not None
                and operator_stop_event.is_set()
            ):
                process.terminate()
                process.join(timeout=termination_grace_seconds)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=termination_grace_seconds)
                marker_path = store.path / "INTERRUPTED.json"
                if not marker_path.is_file():
                    marker_path = store.interrupted(
                        interruption="sigterm",
                        last_checkpoint=_last_checkpoint_name(store.path),
                        actual_elapsed_seconds=time.monotonic() - started,
                    )
                raise RunnerInterrupted("sigterm", str(marker_path))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            process.join(timeout=min(remaining, 0.2))
    except (KeyboardInterrupt, SystemExit) as error:
        process.terminate()
        process.join(timeout=termination_grace_seconds)
        if process.is_alive():
            process.kill()
            process.join(timeout=termination_grace_seconds)
        interruption = (
            "keyboard"
            if isinstance(error, KeyboardInterrupt)
            else "system_exit"
        )
        marker_path = store.path / "INTERRUPTED.json"
        if not marker_path.is_file():
            marker_path = store.interrupted(
                interruption=interruption,
                last_checkpoint=_last_checkpoint_name(store.path),
                actual_elapsed_seconds=time.monotonic() - started,
            )
        raise RunnerInterrupted(interruption, str(marker_path)) from error
    if process.is_alive():
        wall_cap_requested.set()
        process.terminate()
        process.join(timeout=termination_grace_seconds)
        if process.is_alive():
            process.kill()
            process.join(timeout=termination_grace_seconds)
        elapsed = time.monotonic() - started
        if process.is_alive():
            raise RuntimeError("runner-owned child survived SIGKILL")
        marker = store.fail(
            exception=(
                "runner hard wall-clock deadline reached at "
                f"{max_wall_seconds} seconds"
            ),
            last_checkpoint=_last_checkpoint_name(store.path),
            peak_gpu_memory_bytes=_last_recorded_gpu_memory(store.path),
            failure_class="wall_cap",
            actual_elapsed_seconds=elapsed,
            runner_owned_child_terminated=True,
        )
        marker_value = json.loads(marker.read_text())
        return {
            "status": "FAILED",
            "failure_class": "wall_cap",
            "failure_fingerprint": marker_value["failure_fingerprint"],
            "actual_elapsed_seconds": elapsed,
            "marker": str(marker),
        }
    elapsed = time.monotonic() - started
    try:
        envelope = result_queue.get(timeout=termination_grace_seconds)
    except Exception:
        envelope = {
            "kind": "exception",
            "exception": f"worker exited with code {process.exitcode}",
            "traceback": "",
        }
    if envelope["kind"] == "result":
        return {
            **envelope["value"],
            "actual_elapsed_seconds": elapsed,
        }
    if envelope["kind"] == "interrupted":
        raise RunnerInterrupted(
            envelope["interruption"],
            envelope["marker"],
        )
    marker = store.fail(
        exception=envelope["exception"],
        last_checkpoint=_last_checkpoint_name(store.path),
        peak_gpu_memory_bytes=0,
        failure_class="code",
        actual_elapsed_seconds=elapsed,
    )
    marker_value = json.loads(marker.read_text())
    return {
        "status": "FAILED",
        "failure_class": "code",
        "failure_fingerprint": marker_value["failure_fingerprint"],
        "actual_elapsed_seconds": elapsed,
        "marker": str(marker),
        "exception": envelope["exception"],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repository_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repository_root,
        text=True,
    ).strip()


@dataclass(frozen=True)
class JobPlan:
    generator: str
    seed: int
    device_class: str
    scenario: str
    kappa: float
    requested_steps: int
    max_wall_seconds: float
    checkpoint_interval_steps: int
    sampling_plan_hash: str
    attempt_path: str
    phase: str
    dependencies: tuple[str, ...]
    execution_order: int
    expected_artifacts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GPUAssignment:
    job: JobPlan
    gpu: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenInputs:
    train: Any
    validation: Any
    test: Any
    sampling_plan: Any
    tau: np.ndarray
    row_guard_thresholds: Any
    manifest: Mapping[str, Any]


class _StoreStream:
    def __init__(self, store: Any, stream: str) -> None:
        self.store = store
        self.stream = stream

    def write(self, text: str) -> int:
        if text:
            self.store.append_log(self.stream, text)
        return len(text)

    def flush(self) -> None:
        return None


class _ResourceMonitor:
    def __init__(
        self,
        store: Any,
        *,
        gpu_uuid: str | None,
        interval_seconds: float = 60.0,
    ) -> None:
        self.store = store
        self.gpu_uuid = gpu_uuid
        self.interval_seconds = interval_seconds
        self.started = time.monotonic()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _cpu_rss_bytes() -> int:
        try:
            import resource

            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value * 1024 if sys.platform != "darwin" else value
        except (ImportError, OSError, ValueError):
            return 0

    def _gpu_values(self) -> tuple[int, int]:
        if self.gpu_uuid is None:
            return 0, 0
        try:
            rows, _ = query_gpu_inventory()
        except (OSError, RuntimeError, subprocess.SubprocessError):
            return 0, 0
        for row in rows:
            if row["uuid"] == self.gpu_uuid:
                return (
                    int(row["memory_used_mib"]) * 1024 * 1024,
                    int(row["utilization_percent"]),
                )
        return 0, 0

    def _record(self) -> None:
        elapsed = time.monotonic() - self.started
        gpu_memory, gpu_utilization = self._gpu_values()
        self.store.write_heartbeat(
            {
                "status": "RUNNING",
                "elapsed_seconds": elapsed,
            }
        )
        self.store.append_resource(
            {
                "elapsed_seconds": elapsed,
                "cpu_rss_bytes": self._cpu_rss_bytes(),
                "gpu_memory_bytes": gpu_memory,
                "gpu_utilization_percent": gpu_utilization,
            }
        )

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            try:
                self._record()
            except Exception:
                return

    def __enter__(self) -> "_ResourceMonitor":
        self._record()
        self.thread.start()
        return self

    def __exit__(self, *unused: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
        try:
            self._record()
        except Exception:
            pass


class FullRunStateStore:
    """Append-only run events plus atomic current-state markers."""

    def __init__(self, root: Path, *, run_manifest_hash: str) -> None:
        if len(run_manifest_hash) != 64:
            raise ValueError("run manifest hash must be SHA-256")
        self.root = root
        self.run_manifest_hash = run_manifest_hash
        self.root.mkdir(parents=True, exist_ok=True)

    def _event(
        self,
        *,
        generator: str | None,
        seed: int | None,
        attempt: int | None,
        status: str,
        failure_class: str | None,
        manifest_hash: str | None,
        **extra: Any,
    ) -> Mapping[str, Any]:
        event = {
            "generator": generator,
            "seed": seed,
            "attempt": attempt,
            "status": status,
            "failure_class": failure_class,
            "timestamp": _utc_timestamp(),
            "manifest_hash": manifest_hash or self.run_manifest_hash,
            **extra,
        }
        descriptor = os.open(
            self.root / "run_state.jsonl",
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        try:
            payload = (
                json.dumps(
                    event,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode()
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return event

    def start(self) -> None:
        value = {
            "status": "RUNNING",
            "timestamp": _utc_timestamp(),
            "run_manifest_hash": self.run_manifest_hash,
        }
        _atomic_json_replace(self.root / "RUNNING.json", value)
        self._event(
            generator=None,
            seed=None,
            attempt=None,
            status="RUNNING",
            failure_class=None,
            manifest_hash=self.run_manifest_hash,
        )

    def record(self, job: JobPlan, result: Mapping[str, Any]) -> None:
        self._event(
            generator=job.generator,
            seed=job.seed,
            attempt=(
                int(result["attempt"])
                if result.get("attempt") is not None
                else None
            ),
            status=str(result["status"]),
            failure_class=result.get("failure_class"),
            manifest_hash=result.get("manifest_hash"),
            failure_fingerprint=result.get("failure_fingerprint"),
        )

    def stop(
        self,
        *,
        reason: str,
        mandatory: bool,
        interrupted: bool = False,
    ) -> None:
        status = "INTERRUPTED" if interrupted else "STOPPED"
        value = {
            "status": status,
            "timestamp": _utc_timestamp(),
            "reason": reason,
            "mandatory": mandatory,
            "run_manifest_hash": self.run_manifest_hash,
        }
        _atomic_json_replace(self.root / "STOPPED.json", value)
        self._event(
            generator=None,
            seed=None,
            attempt=None,
            status=status,
            failure_class=(
                "operator_interrupt" if interrupted else "global_stop"
            ),
            manifest_hash=self.run_manifest_hash,
            reason=reason,
            mandatory=mandatory,
        )

    def complete(self, *, final_marker_hash: str) -> None:
        value = {
            "status": "COMPLETE",
            "timestamp": _utc_timestamp(),
            "run_manifest_hash": self.run_manifest_hash,
            "final_marker_hash": final_marker_hash,
        }
        _atomic_json_replace(self.root / "COMPLETE.json", value)
        self._event(
            generator=None,
            seed=None,
            attempt=None,
            status="COMPLETE",
            failure_class=None,
            manifest_hash=self.run_manifest_hash,
            final_marker_hash=final_marker_hash,
        )


class RunPolicyController:
    def __init__(self, run_state: FullRunStateStore) -> None:
        self.run_state = run_state
        self.disabled_secondary: set[str] = set()
        self.results: list[Mapping[str, Any]] = []
        self.c2_evaluable = True
        self.c2_reasons: list[str] = []
        self.last_infrastructure_fingerprint: str | None = None
        self.consecutive_infrastructure = 0
        self.mandatory_stopped = False

    def should_cancel(self, job: JobPlan) -> bool:
        return job.generator in self.disabled_secondary

    def observe(self, job: JobPlan, result: Mapping[str, Any]) -> None:
        self.run_state.record(job, result)
        self.results.append({"job": job.to_dict(), **result})
        status = result.get("status")
        failure_class = result.get("failure_class")
        if failure_class == "infrastructure":
            fingerprint = str(result.get("failure_fingerprint"))
            if fingerprint == self.last_infrastructure_fingerprint:
                self.consecutive_infrastructure += 1
            else:
                self.last_infrastructure_fingerprint = fingerprint
                self.consecutive_infrastructure = 1
        else:
            self.last_infrastructure_fingerprint = None
            self.consecutive_infrastructure = 0
        if self.consecutive_infrastructure >= 3 and not self.mandatory_stopped:
            self.run_state.stop(
                reason=(
                    "three consecutive identical infrastructure failures: "
                    f"{self.last_infrastructure_fingerprint}"
                ),
                mandatory=True,
            )
            self.mandatory_stopped = True
            self.c2_evaluable = False
        if (
            result.get("mandatory_stop") is True
            and not self.mandatory_stopped
        ):
            self.run_state.stop(
                reason=(
                    "attempt marker requires mandatory stop after repeated "
                    "infrastructure failure"
                ),
                mandatory=True,
            )
            self.mandatory_stopped = True
            self.c2_evaluable = False
        failed = status in {"FAILED", "INVALID", "UNAVAILABLE"}
        if failed and job.generator in PRIMARY_C2_GENERATORS:
            self.c2_evaluable = False
            self.c2_reasons.append(
                f"{job.generator} seed {job.seed} status={status}"
            )
        elif failed:
            self.disabled_secondary.add(job.generator)

    def summary(self, *, expected_jobs: int) -> Mapping[str, Any]:
        return {
            "results": self.results,
            "mandatory_stopped": self.mandatory_stopped,
            "c2_evaluable": self.c2_evaluable,
            "c2_not_evaluable_reasons": self.c2_reasons,
            "all_jobs_terminal": (
                len(self.results) == expected_jobs
                and all(
                    result.get("status") in RUN_TERMINAL_STATUSES
                    for result in self.results
                )
            ),
        }


def replay_continuation_results(
    *,
    jobs: Sequence[JobPlan],
    continuation_plan: Mapping[str, Any],
    policy: RunPolicyController,
) -> None:
    """Replay preserved terminal outcomes into the new run-level policy."""
    reused_by_job = {
        (item["job"]["generator"], int(item["job"]["seed"])): item
        for item in continuation_plan["reused"]
    }
    for job in jobs:
        reused = reused_by_job.get((job.generator, job.seed))
        if reused is not None:
            policy.observe(job, reused)


def execute_job_schedule(
    jobs: Sequence[JobPlan],
    *,
    execute: Callable[[JobPlan], Mapping[str, Any]],
    cancel: Callable[[JobPlan, str], Mapping[str, Any]],
    run_state: FullRunStateStore,
) -> Mapping[str, Any]:
    """Apply the frozen run-level failure policy to an ordered job plan."""
    run_state.start()
    policy = RunPolicyController(run_state)
    for job in jobs:
        if policy.mandatory_stopped:
            break
        if policy.should_cancel(job):
            result = dict(
                cancel(
                    job,
                    "remaining seeds cancelled after generator terminal failure",
                )
            )
        else:
            try:
                result = dict(execute(job))
            except RunnerInterrupted as error:
                run_state.stop(
                    reason=str(error),
                    mandatory=True,
                    interrupted=True,
                )
                raise
        policy.observe(job, result)
    return policy.summary(expected_jobs=len(jobs))


def build_job_plan(
    raw: Mapping[str, Any],
    *,
    sampling_plan_hash: str,
) -> list[JobPlan]:
    if not isinstance(sampling_plan_hash, str) or len(sampling_plan_hash) != 64:
        raise ValueError("SamplingPlan hash must be a full SHA-256")
    configured = tuple(raw.get("baselines", ()))
    if configured != GENERATOR_ORDER:
        raise ValueError("full runner requires the exact 13 frozen generators")
    scope = raw.get("scope", {})
    if (
        scope.get("scenarios") != [SCENARIO]
        or scope.get("kappas") != [KAPPA]
        or tuple(scope.get("model_seeds", ())) != SEEDS
    ):
        raise ValueError("full runner scope differs from the frozen protocol")
    template = str(raw["paths"]["attempt_template"])
    jobs: list[JobPlan] = []
    order = 0
    for generator in GENERATOR_ORDER:
        model = raw["baselines"][generator]
        device = str(model["device"])
        if generator in CPU_GENERATORS:
            requested_steps = 0
            maximum = float(model.get("max_wall_seconds", 7200))
            interval = 0
            phase = "A"
            dependencies = ("frozen_data",)
        else:
            requested_steps = int(
                model.get(
                    "requested_steps_total",
                    model.get("requested_steps"),
                )
            )
            maximum = float(
                model.get(
                    "max_wall_seconds_total",
                    model.get("max_wall_seconds"),
                )
            )
            interval = int(model["checkpoint_interval_steps"])
            phase = "B"
            dependencies = ("phase_a_complete",)
        for seed in SEEDS:
            order += 1
            attempt = template.format(
                generator=generator,
                seed=seed,
                attempt=1,
            )
            jobs.append(
                JobPlan(
                    generator=generator,
                    seed=seed,
                    device_class=device,
                    scenario=SCENARIO,
                    kappa=KAPPA,
                    requested_steps=requested_steps,
                    max_wall_seconds=maximum,
                    checkpoint_interval_steps=interval,
                    sampling_plan_hash=sampling_plan_hash,
                    attempt_path=attempt,
                    phase=phase,
                    dependencies=dependencies,
                    execution_order=order,
                    expected_artifacts=(
                        "manifest.json",
                        "RUNNING.json",
                        "heartbeat.json",
                        "stdout.log",
                        "stderr.log",
                        "progress.jsonl",
                        "partial_metrics.json",
                        "resource.jsonl",
                        "checkpoints/step_*.pt",
                        "checkpoints/latest",
                        "checkpoints/final.pt",
                        "sample.npz",
                        "metrics.json",
                        "runtime.json",
                        "evaluation.json",
                        "artifact_index.jsonl",
                        "COMPLETE.json|FAILED.json|INVALID.json|UNAVAILABLE.json",
                    ),
                )
            )
    return jobs


def select_idle_gpus(
    gpu_rows: list[Mapping[str, Any]],
    *,
    process_uuids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in sorted(
        gpu_rows,
        key=lambda value: int(value["physical_index"]),
    ):
        reasons = []
        if str(row["uuid"]) in process_uuids:
            reasons.append("compute process present")
        if int(row["memory_used_mib"]) > 1024:
            reasons.append("memory used exceeds 1024 MiB")
        if int(row["utilization_percent"]) > 5:
            reasons.append("utilization exceeds 5 percent")
        record = dict(row)
        if reasons:
            record["exclusion_reason"] = "; ".join(reasons)
            excluded.append(record)
        else:
            record["selection_reason"] = "all frozen idle criteria satisfied"
            selected.append(record)
    return selected, excluded


def query_gpu_inventory() -> tuple[list[dict[str, Any]], set[str]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError("unexpected nvidia-smi GPU row")
        rows.append(
            {
                "physical_index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_used_mib": int(fields[3]),
                "utilization_percent": int(fields[4]),
            }
        )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if processes.returncode != 0:
        raise RuntimeError(
            "cannot verify GPU compute-process inventory with nvidia-smi"
        )
    process_uuids = {
        value.strip()
        for value in processes.stdout.splitlines()
        if value.strip() and "No running" not in value
    }
    return rows, process_uuids


def assert_full_gpu_runtime_access(
    *,
    require_idle: bool = True,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    listing = subprocess.run(
        ["nvidia-smi", "-L"],
        check=False,
        capture_output=True,
        text=True,
    )
    if listing.returncode != 0 or "GPU " not in listing.stdout:
        raise RuntimeError("nvidia-smi -L is not usable in the full-run shell")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("PyTorch CUDA is not usable in the full-run shell")
    rows, process_uuids = query_gpu_inventory()
    selected, excluded = select_idle_gpus(
        rows,
        process_uuids=process_uuids,
    )
    if require_idle and not selected:
        raise RuntimeError("no idle GPU satisfies the frozen selector")
    return selected, excluded


def build_gpu_waves(
    jobs: list[JobPlan],
    gpus: list[Mapping[str, Any]],
) -> list[list[GPUAssignment]]:
    if not gpus:
        raise RuntimeError("no idle GPU satisfies the frozen selector")
    if any(job.device_class != "gpu" for job in jobs):
        raise ValueError("GPU scheduler received a non-GPU job")
    waves: list[list[GPUAssignment]] = []
    for start in range(0, len(jobs), len(gpus)):
        wave_jobs = jobs[start : start + len(gpus)]
        waves.append(
            [
                GPUAssignment(job=job, gpu=dict(gpus[index]))
                for index, job in enumerate(wave_jobs)
            ]
        )
    return waves


def _attempt_number_tree_digest(
    full_root: Path,
    *,
    attempt: int,
    terminal_markers_only: bool = False,
) -> tuple[str, int, int]:
    attempt_part = f"attempt_{attempt:03d}"
    marker_names = {
        "CANCELLED.json",
        "COMPLETE.json",
        "FAILED.json",
        "INVALID.json",
        "UNAVAILABLE.json",
    }
    paths = sorted(
        path
        for path in full_root.rglob("*")
        if path.is_file()
        and attempt_part in path.parts
        and (not terminal_markers_only or path.name in marker_names)
    )
    lines = (
        f"{sha256_file(path)}  {path.relative_to(full_root).as_posix()}\n"
        for path in paths
    )
    return (
        hashlib.sha256("".join(lines).encode()).hexdigest(),
        len(paths),
        sum(path.stat().st_size for path in paths),
    )


def _validate_continuation_manifest(
    manifest: Mapping[str, Any],
    *,
    job: JobPlan,
    continuation: Mapping[str, Any],
) -> None:
    expected = {
        "git_commit": continuation["previous_source_commit"],
        "code_hash": continuation["previous_relevant_code_sha256"],
        "config_hash": continuation["config_sha256"],
        "scenario": job.scenario,
        "kappa": job.kappa,
        "generator": job.generator,
        "seed": job.seed,
        "sampling_plan_hash": continuation["sampling_plan_sha256"],
        "data_hashes": continuation["data_hashes"],
        "baseline_definition_version": "benchmark-v2.5",
        "evaluation_version": "benchmark-v2.5-evaluation-v1",
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise AuthorizationError(
            "attempt_002 continuation provenance mismatch: "
            f"{job.generator} seed {job.seed}: {mismatches}"
        )


def plan_scheduler_continuation(
    *,
    artifact_root: Path,
    jobs: Sequence[JobPlan],
    continuation: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Read-only classification of preserved attempt_002 and pending jobs."""
    terminal_attempt = int(continuation.get("terminal_attempt", -1))
    new_attempt = int(continuation.get("new_attempt", -1))
    if terminal_attempt != 2 or new_attempt != 3:
        raise AuthorizationError(
            "scheduler continuation is restricted to attempt_002 -> "
            "attempt_003"
        )
    full_root = Path(artifact_root) / "full"
    tree_digest, tree_files, tree_bytes = _attempt_number_tree_digest(
        full_root,
        attempt=terminal_attempt,
    )
    marker_digest, marker_files, marker_bytes = (
        _attempt_number_tree_digest(
            full_root,
            attempt=terminal_attempt,
            terminal_markers_only=True,
        )
    )
    if tree_digest != continuation.get("attempt_tree_sha256"):
        raise AuthorizationError("attempt_002 preservation tree hash mismatch")
    if marker_digest != continuation.get("terminal_marker_tree_sha256"):
        raise AuthorizationError(
            "attempt_002 terminal marker tree hash mismatch"
        )
    marker_names = {
        "CANCELLED.json",
        "COMPLETE.json",
        "FAILED.json",
        "INVALID.json",
        "UNAVAILABLE.json",
    }
    fallback_classes = {
        "COMPLETE": None,
        "FAILED": "code",
        "INVALID": "hard_guard",
        "UNAVAILABLE": "unavailable",
        "CANCELLED": "dependency_cancelled",
    }
    reused: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for job in jobs:
        attempt_path = (
            full_root
            / job.scenario
            / f"kappa_{job.kappa:.2f}"
            / job.generator
            / f"seed_{job.seed}"
            / f"attempt_{terminal_attempt:03d}"
        )
        terminals = (
            sorted(
                path
                for path in attempt_path.iterdir()
                if path.is_file() and path.name in marker_names
            )
            if attempt_path.is_dir()
            else []
        )
        if len(terminals) > 1:
            raise AuthorizationError(
                f"conflicting attempt_002 terminal markers: {attempt_path}"
            )
        if not terminals:
            if attempt_path.exists():
                manifest_path = attempt_path / "manifest.json"
                if not manifest_path.is_file():
                    raise AuthorizationError(
                        "partial attempt_002 has no immutable manifest: "
                        f"{attempt_path}"
                    )
                _validate_continuation_manifest(
                    json.loads(manifest_path.read_text()),
                    job=job,
                    continuation=continuation,
                )
            pending.append(
                {
                    "job": job.to_dict(),
                    "target_attempt": new_attempt,
                    "preserved_partial_attempt": (
                        str(attempt_path) if attempt_path.exists() else None
                    ),
                }
            )
            continue
        manifest_path = attempt_path / "manifest.json"
        if not manifest_path.is_file():
            raise AuthorizationError(
                f"terminal attempt_002 has no immutable manifest: {attempt_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text())
            marker = json.loads(terminals[0].read_text())
        except json.JSONDecodeError as error:
            raise AuthorizationError(
                f"invalid preserved attempt_002 JSON: {attempt_path}"
            ) from error
        _validate_continuation_manifest(
            manifest,
            job=job,
            continuation=continuation,
        )
        status = str(marker.get("status"))
        if terminals[0].name != f"{status}.json" or status not in fallback_classes:
            raise AuthorizationError(
                f"attempt_002 terminal status mismatch: {terminals[0]}"
            )
        reused.append(
            {
                "status": status,
                "job": job.to_dict(),
                "attempt_path": str(attempt_path),
                "attempt": terminal_attempt,
                "manifest_hash": _manifest_hash(manifest),
                "failure_class": marker.get(
                    "failure_class",
                    fallback_classes[status],
                ),
                "failure_fingerprint": marker.get("failure_fingerprint"),
                "mandatory_stop": marker.get("mandatory_stop", False),
                "reused": True,
                "continuation_source": "attempt_002",
            }
        )
    expected_terminal = int(
        continuation.get("expected_terminal_jobs", -1)
    )
    expected_pending = int(continuation.get("expected_pending_jobs", -1))
    if len(reused) != expected_terminal or len(pending) != expected_pending:
        raise AuthorizationError(
            "continuation job counts differ from authorization: "
            f"reused={len(reused)}, pending={len(pending)}"
        )
    if len(reused) + len(pending) != len(jobs):
        raise AuthorizationError("continuation does not cover all planned jobs")
    expected_pending_keys = {
        ("ctgan_separate_class", 4),
        ("ctgan_separate_class", 5),
        *(("tvae_separate_class", seed) for seed in SEEDS),
        *(("neural_sequence", seed) for seed in SEEDS),
        *(("cof_seqgen", seed) for seed in SEEDS),
    }
    actual_pending_keys = {
        (item["job"]["generator"], int(item["job"]["seed"]))
        for item in pending
    }
    if actual_pending_keys != expected_pending_keys:
        raise AuthorizationError(
            "continuation pending set differs from the preserved 17 jobs"
        )
    status_counts = {
        status: sum(item["status"] == status for item in reused)
        for status in fallback_classes
    }
    if status_counts != {
        "COMPLETE": 40,
        "FAILED": 0,
        "INVALID": 4,
        "UNAVAILABLE": 0,
        "CANCELLED": 4,
    }:
        raise AuthorizationError(
            "attempt_002 terminal status counts differ from preservation"
        )
    return {
        "status": "CONTINUATION_READY",
        "terminal_attempt": terminal_attempt,
        "new_attempt": new_attempt,
        "reused": reused,
        "pending": pending,
        "attempt_tree_sha256": tree_digest,
        "attempt_tree_files": tree_files,
        "attempt_tree_bytes": tree_bytes,
        "terminal_marker_tree_sha256": marker_digest,
        "terminal_marker_files": marker_files,
        "terminal_marker_bytes": marker_bytes,
        "terminal_counts": status_counts,
    }


def run_live_gpu_schedule(
    jobs: Sequence[JobPlan],
    *,
    inventory_provider: Callable[
        [], tuple[list[dict[str, Any]], set[str]]
    ],
    execute_wave: Callable[
        [Sequence[GPUAssignment]], Sequence[Mapping[str, Any]]
    ],
    cancel: Callable[[JobPlan, str], Mapping[str, Any]],
    run_state: FullRunStateStore,
    max_concurrency: int,
    wait_budget_seconds: float,
    poll_interval_seconds: float,
    policy: RunPolicyController | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any]:
    """Schedule each GPU wave from the current live idle inventory."""
    if max_concurrency < 1:
        raise ValueError("GPU maximum concurrency must be positive")
    if wait_budget_seconds <= 0 or poll_interval_seconds <= 0:
        raise ValueError("GPU wait budget and poll interval must be positive")
    controller = policy or RunPolicyController(run_state)
    cursor = 0
    wave_count = 0
    wait_count = 0
    wait_path: Path | None = None
    wait_started: float | None = None
    selections = []
    while cursor < len(jobs) and not controller.mandatory_stopped:
        rows, process_uuids = inventory_provider()
        selected, excluded = select_idle_gpus(
            rows,
            process_uuids=process_uuids,
        )
        selection_path = _next_exclusive_path(
            run_state.root,
            prefix="gpu_selection",
            suffix=".json",
        )
        _write_json_exclusive(
            selection_path,
            {
                "timestamp": _utc_timestamp(),
                "selected": [dict(value) for value in selected],
                "excluded": [dict(value) for value in excluded],
                "configured_max_concurrency": max_concurrency,
                "effective_concurrency": min(
                    max_concurrency,
                    len(selected),
                ),
            },
        )
        if not selected:
            if wait_path is None:
                wait_count += 1
                wait_path = _next_exclusive_path(
                    run_state.root,
                    prefix="gpu_wait_attempt",
                    suffix="",
                )
                wait_path.mkdir()
                wait_started = monotonic()
                _write_json_exclusive(
                    wait_path / "WAITING_FOR_GPU.json",
                    {
                        "status": "WAITING_FOR_GPU",
                        "timestamp": _utc_timestamp(),
                        "wait_budget_seconds": wait_budget_seconds,
                        "poll_interval_seconds": poll_interval_seconds,
                        "external_processes_terminated": False,
                    },
                )
            assert wait_started is not None
            elapsed = monotonic() - wait_started
            heartbeat = {
                "status": "WAITING_FOR_GPU",
                "timestamp": _utc_timestamp(),
                "elapsed_wait_seconds": elapsed,
                "selected": [],
                "excluded": [dict(value) for value in excluded],
            }
            heartbeat_path = wait_path / "heartbeat.jsonl"
            descriptor = os.open(
                heartbeat_path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                0o644,
            )
            try:
                os.write(
                    descriptor,
                    (
                        json.dumps(
                            heartbeat,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    ).encode(),
                )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if elapsed >= wait_budget_seconds:
                reason = (
                    "zero idle GPU wait budget exhausted; "
                    "no external process was terminated"
                )
                _write_json_exclusive(
                    wait_path / "WAIT_BUDGET_EXHAUSTED.json",
                    {
                        "status": "WAIT_BUDGET_EXHAUSTED",
                        "timestamp": _utc_timestamp(),
                        "elapsed_wait_seconds": elapsed,
                        "reason": reason,
                    },
                )
                run_state.stop(reason=reason, mandatory=True)
                controller.mandatory_stopped = True
                break
            sleep(poll_interval_seconds)
            continue
        if wait_path is not None:
            assert wait_started is not None
            _write_json_exclusive(
                wait_path / "RESUMED.json",
                {
                    "status": "RESUMED",
                    "timestamp": _utc_timestamp(),
                    "elapsed_wait_seconds": monotonic() - wait_started,
                    "selected": [dict(value) for value in selected],
                },
            )
            wait_path = None
            wait_started = None
        available = selected[:max_concurrency]
        assignments = []
        while cursor < len(jobs) and len(assignments) < len(available):
            job = jobs[cursor]
            cursor += 1
            if controller.should_cancel(job):
                controller.observe(
                    job,
                    cancel(
                        job,
                        (
                            "remaining seeds cancelled after generator "
                            "terminal failure"
                        ),
                    ),
                )
            else:
                assignments.append(
                    GPUAssignment(
                        job=job,
                        gpu=available[len(assignments)],
                    )
                )
        selections.append(
            {
                "selected": [dict(value) for value in available],
                "excluded": [dict(value) for value in excluded],
            }
        )
        if not assignments:
            continue
        results = list(execute_wave(assignments))
        if len(results) != len(assignments):
            raise RuntimeError("GPU wave returned the wrong result count")
        wave_count += 1
        for assignment, result in zip(assignments, results):
            controller.observe(assignment.job, result)
    return {
        "status": (
            "STOPPED" if controller.mandatory_stopped else "COMPLETE"
        ),
        "wave_count": wave_count,
        "wait_count": wait_count,
        "job_count": len(jobs),
        "selection_events": selections,
        "policy": controller.summary(expected_jobs=len(controller.results)),
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _atomic_json_replace(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_dry_run(
    *,
    repository_root: Path,
    config_path: Path,
    work_root: Path,
) -> Mapping[str, Any]:
    """Exercise orchestration contracts with metadata-only fixtures.

    This path deliberately does not import the DGP, generator registry, model
    adapters, or CUDA runtime.
    """
    if work_root.exists():
        raise FileExistsError(
            f"dry-run work root already exists: {work_root}"
        )
    work_root.mkdir(parents=True)
    resolved_config = (
        config_path
        if config_path.is_absolute()
        else repository_root / config_path
    )
    raw = yaml.safe_load(resolved_config.read_text())
    config_hash = sha256_file(resolved_config)
    plan_hash = "1" * 64
    data_hashes = {
        "train": "2" * 64,
        "validation": "3" * 64,
        "test": "4" * 64,
    }
    data_manifest = {
        "schema_version": "benchmark-v2.5-frozen-data-manifest",
        "status": "COMPLETE",
        "config_sha256": config_hash,
        "scenario": SCENARIO,
        "kappa": KAPPA,
        "data_hashes": data_hashes,
        "sampling_plan": {"sha256": plan_hash},
        "fixture_only": True,
    }
    data_path = work_root / "fixture_data_manifest.json"
    _write_json_exclusive(data_path, data_manifest)
    approval_text = "benchmark-v2.5 dry-run fixture authorization"
    authorization = {
        "schema_version": "benchmark-v2.5-full-authorization-v1",
        "source_commit": _git_output(repository_root, "rev-parse", "HEAD"),
        "config_sha256": config_hash,
        "v2_4": {
            "gate_tag": "benchmark-v2.4-gate-pass",
            "gate_commit": _git_output(
                repository_root,
                "rev-parse",
                "benchmark-v2.4-gate-pass^{commit}",
            ),
            "artifact_index_sha256": sha256_file(
                repository_root / V2_4_INDEX_PATH
            ),
            "gate_report_sha256": sha256_file(
                repository_root / V2_4_GATE_REPORT_PATH
            ),
        },
        "capacity_preflight": {
            "artifact_index_sha256": sha256_file(
                repository_root / CAPACITY_INDEX_PATH
            ),
            "correction_note_sha256": sha256_file(
                repository_root / CAPACITY_CORRECTION_PATH
            ),
        },
        "frozen_data": {
            "manifest_sha256": sha256_file(data_path),
            "data_hashes": data_hashes,
            "sampling_plan_sha256": plan_hash,
        },
        "scope": {
            "scenario": SCENARIO,
            "kappa": KAPPA,
            "seeds": list(SEEDS),
            "generators": list(GENERATOR_ORDER),
            "analysis": ["primary", "secondary"],
        },
        "approval_text_sha256": hashlib.sha256(
            approval_text.encode("utf-8")
        ).hexdigest(),
        "approval_text": approval_text,
        "fixture_only": True,
    }
    auth_path = work_root / "fixture_authorization.json"
    _write_json_exclusive(auth_path, authorization)
    validate_full_authorization(
        auth_path,
        repository_root=repository_root,
        config_path=config_path,
        data_manifest_path=data_path,
    )
    jobs = build_job_plan(raw, sampling_plan_hash=plan_hash)

    from experiments.full_artifact_store_v2_5 import FullAttemptStore

    artifact_root = work_root / "benchmark_v2_5"
    first_job = jobs[0]
    attempt_manifest = {
        "schema_version": "benchmark-v2.5-full-attempt",
        "git_commit": authorization["source_commit"],
        "config_hash": config_hash,
        "code_hash": "6" * 64,
        "scenario": first_job.scenario,
        "kappa": first_job.kappa,
        "generator": first_job.generator,
        "seed": first_job.seed,
        "sampling_plan_hash": plan_hash,
        "data_hashes": data_hashes,
        "gpu": {"id": None, "name": None},
        "cuda_version": None,
        "pytorch_version": "dry-run-no-import",
        "requested_training_budget": {
            "steps": first_job.requested_steps,
            "max_wall_seconds": first_job.max_wall_seconds,
        },
        "actual_training_budget": {"steps": 0, "wall_seconds": 0.0},
        "baseline_definition_version": "benchmark-v2.5",
        "evaluation_version": "benchmark-v2.5-evaluation-v1",
    }
    store, selected = FullAttemptStore.select(
        artifact_root,
        attempt_manifest,
    )
    store.write_heartbeat({"status": "RUNNING", "step": 0})
    store.append_progress(
        {
            "step": 1,
            "loss": 0.0,
            "validation_metric": None,
            "elapsed_seconds": 0.0,
            "peak_gpu_memory_bytes": 0,
        }
    )
    checkpoint = store.write_checkpoint(
        step=1,
        writer=lambda path: path.write_bytes(b"fixture checkpoint"),
    )
    _, resumed = FullAttemptStore.select(
        artifact_root,
        attempt_manifest,
    )
    changed_manifest = {
        **attempt_manifest,
        "config_hash": "7" * 64,
    }
    changed_store, changed = FullAttemptStore.select(
        artifact_root,
        changed_manifest,
    )
    changed_store.invalid(
        reason="fixture hard guard failure",
        hard_guards={"fixture_guard": "FAIL"},
    )
    fake_gpus = [
        {"physical_index": index, "uuid": f"DRY-GPU-{index}"}
        for index in (1, 2, 3)
    ]
    gpu_jobs = [job for job in jobs if job.device_class == "gpu"]
    waves = build_gpu_waves(gpu_jobs, fake_gpus)
    return {
        "schema_version": "benchmark-v2.5-full-runner-dry-run-v1",
        "status": "PASS",
        "job_count": len(jobs),
        "authorization_validated": True,
        "exact_resume_verified": bool(
            selected.attempt == 1
            and resumed.resume
            and resumed.latest_checkpoint == checkpoint
        ),
        "mismatch_new_attempt_verified": bool(
            changed.attempt == 2 and not changed.resume
        ),
        "invalid_routing_verified": (
            changed_store.path / "INVALID.json"
        ).is_file(),
        "scheduler_job_count": sum(len(wave) for wave in waves),
        "scheduler_wave_count": len(waves),
        "calls": {"dgp": 0, "gpu": 0, "fit": 0, "sample": 0},
        "work_root": str(work_root),
    }


def _resolved(repository_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def _verify_execution_source_clean(repository_root: Path) -> None:
    scoped = (
        "benchmarks",
        "eval",
        "experiments",
        "generators",
        "models",
        "scripts",
        "configs/benchmark_v2/full_v2_5.yaml",
    )
    status = _git_output(
        repository_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *scoped,
    )
    if status:
        raise AuthorizationError(
            "execution source/config worktree differs from authorized HEAD"
        )


def validate_scheduler_continuation_changed_paths(
    changed_paths: Sequence[str],
) -> tuple[str, ...]:
    """Permit only the runner/artifact seam plus tests and protocol docs."""
    normalized = tuple(sorted(set(str(path) for path in changed_paths)))
    allowed_source = {
        "scripts/run_full_experiment_v2_5.py",
        "experiments/full_artifact_store_v2_5.py",
    }
    invalid = [
        path
        for path in normalized
        if path not in allowed_source
        and not path.startswith("tests/")
        and not path.startswith("docs/benchmark_v2/")
    ]
    if invalid:
        raise AuthorizationError(
            "continuation source change is not scheduler-only: "
            f"{invalid}"
        )
    if "scripts/run_full_experiment_v2_5.py" not in normalized:
        raise AuthorizationError(
            "scheduler-only continuation is missing the runner correction"
        )
    return normalized


def validate_authorization_static(
    authorization_path: Path,
    *,
    repository_root: Path,
    config_path: Path,
) -> Mapping[str, Any]:
    resolved_authorization = _resolved(repository_root, authorization_path)
    if not resolved_authorization.is_file():
        raise AuthorizationError(
            "missing external authorization manifest: "
            f"{resolved_authorization}"
        )
    try:
        authorization = json.loads(resolved_authorization.read_text())
    except json.JSONDecodeError as error:
        raise AuthorizationError(
            "authorization input is not valid JSON"
        ) from error
    if (
        authorization.get("schema_version")
        != "benchmark-v2.5-full-authorization-v1"
    ):
        raise AuthorizationError("authorization schema mismatch")
    resolved_config = _resolved(repository_root, config_path)
    if not resolved_config.is_file():
        raise AuthorizationError(f"missing frozen config: {resolved_config}")
    config_hash = sha256_file(resolved_config)
    if authorization.get("config_sha256") != config_hash:
        raise AuthorizationError("authorization config hash mismatch")
    source_commit = _git_output(repository_root, "rev-parse", "HEAD")
    if authorization.get("source_commit") != source_commit:
        raise AuthorizationError("authorization source commit mismatch")
    continuation = authorization.get("continuation")
    if continuation is not None:
        if not isinstance(continuation, Mapping):
            raise AuthorizationError("continuation authorization is invalid")
        required_continuation = {
            "previous_source_commit",
            "previous_relevant_code_sha256",
            "current_relevant_code_sha256",
            "changed_paths",
            "terminal_attempt",
            "new_attempt",
            "expected_terminal_jobs",
            "expected_pending_jobs",
            "attempt_tree_sha256",
            "terminal_marker_tree_sha256",
            "config_sha256",
            "data_hashes",
            "sampling_plan_sha256",
        }
        missing = required_continuation - continuation.keys()
        if missing:
            raise AuthorizationError(
                "continuation authorization fields are missing: "
                f"{sorted(missing)}"
            )
        previous_commit = str(continuation["previous_source_commit"])
        if len(previous_commit) != 40:
            raise AuthorizationError(
                "previous continuation source commit is invalid"
            )
        actual_changed = tuple(
            sorted(
                line
                for line in _git_output(
                    repository_root,
                    "diff",
                    "--name-only",
                    f"{previous_commit}..{source_commit}",
                ).splitlines()
                if line
            )
        )
        declared_changed = tuple(
            sorted(str(path) for path in continuation["changed_paths"])
        )
        if actual_changed != declared_changed:
            raise AuthorizationError(
                "continuation changed-path provenance mismatch"
            )
        validate_scheduler_continuation_changed_paths(actual_changed)
        from experiments.provenance_v2_5 import hash_code

        current_code_hash = hash_code(repository_root)
        if (
            continuation["current_relevant_code_sha256"]
            != current_code_hash
            or authorization.get("relevant_code_sha256")
            != current_code_hash
        ):
            raise AuthorizationError(
                "continuation current relevant code hash mismatch"
            )
        if continuation["config_sha256"] != config_hash:
            raise AuthorizationError(
                "continuation config hash mismatch"
            )
        if (
            int(continuation["terminal_attempt"]) != 2
            or int(continuation["new_attempt"]) != 3
            or int(continuation["expected_terminal_jobs"]) != 48
            or int(continuation["expected_pending_jobs"]) != 17
        ):
            raise AuthorizationError(
                "continuation attempt/job-count contract mismatch"
            )
    raw = yaml.safe_load(resolved_config.read_text())
    if raw.get("execution", {}).get("full_experiment_authorized") is not False:
        raise AuthorizationError(
            "frozen config authorization flag must remain false"
        )
    expected_scope = {
        "scenario": SCENARIO,
        "kappa": KAPPA,
        "seeds": list(SEEDS),
        "generators": list(GENERATOR_ORDER),
        "analysis": ["primary", "secondary"],
    }
    if authorization.get("scope") != expected_scope:
        raise AuthorizationError("authorization scope mismatch")
    approval_hash = authorization.get("approval_text_sha256")
    if not isinstance(approval_hash, str) or len(approval_hash) != 64:
        raise AuthorizationError("authorization approval text hash is invalid")
    approval_text = authorization.get("approval_text")
    approval_path = authorization.get("approval_text_path")
    if isinstance(approval_text, str):
        approval_bytes = approval_text.encode("utf-8")
    elif isinstance(approval_path, str):
        candidate = Path(approval_path)
        if not candidate.is_absolute():
            candidate = resolved_authorization.parent / candidate
        if not candidate.is_file():
            raise AuthorizationError("authorization approval text is missing")
        approval_bytes = candidate.read_bytes()
    else:
        raise AuthorizationError(
            "authorization approval text hash has no verifiable source"
        )
    if hashlib.sha256(approval_bytes).hexdigest() != approval_hash:
        raise AuthorizationError("authorization approval text hash mismatch")

    v2_4 = authorization.get("v2_4", {})
    if v2_4.get("gate_tag") != "benchmark-v2.4-gate-pass":
        raise AuthorizationError("v2.4 gate tag mismatch")
    gate_commit = _git_output(
        repository_root,
        "rev-parse",
        "benchmark-v2.4-gate-pass^{commit}",
    )
    if v2_4.get("gate_commit") != gate_commit:
        raise AuthorizationError("v2.4 gate commit mismatch")
    index_path = repository_root / V2_4_INDEX_PATH
    gate_path = repository_root / V2_4_GATE_REPORT_PATH
    if v2_4.get("artifact_index_sha256") != sha256_file(index_path):
        raise AuthorizationError("v2.4 artifact index hash mismatch")
    if v2_4.get("gate_report_sha256") != sha256_file(gate_path):
        raise AuthorizationError("v2.4 gate report hash mismatch")

    capacity = authorization.get("capacity_preflight", {})
    if capacity.get("artifact_index_sha256") != sha256_file(
        repository_root / CAPACITY_INDEX_PATH
    ):
        raise AuthorizationError("capacity artifact index hash mismatch")
    if capacity.get("correction_note_sha256") != sha256_file(
        repository_root / CAPACITY_CORRECTION_PATH
    ):
        raise AuthorizationError("capacity correction note hash mismatch")
    return authorization


def validate_full_authorization(
    authorization_path: Path,
    *,
    repository_root: Path,
    config_path: Path,
    data_manifest_path: Path,
) -> Mapping[str, Any]:
    authorization = validate_authorization_static(
        authorization_path,
        repository_root=repository_root,
        config_path=config_path,
    )
    resolved_data_manifest = _resolved(
        repository_root,
        data_manifest_path,
    )
    if not resolved_data_manifest.is_file():
        raise AuthorizationError(
            f"missing frozen data manifest: {resolved_data_manifest}"
        )
    try:
        data_manifest = json.loads(resolved_data_manifest.read_text())
    except json.JSONDecodeError as error:
        raise AuthorizationError(
            "frozen data manifest is not valid JSON"
        ) from error
    resolved_config = _resolved(repository_root, config_path)
    config_hash = sha256_file(resolved_config)

    frozen = authorization.get("frozen_data", {})
    if frozen.get("manifest_sha256") != sha256_file(
        resolved_data_manifest
    ):
        raise AuthorizationError("frozen data manifest hash mismatch")
    if data_manifest.get("status") != "COMPLETE":
        raise AuthorizationError("frozen data manifest is not complete")
    if data_manifest.get("config_sha256") != config_hash:
        raise AuthorizationError("frozen data config hash mismatch")
    if (
        data_manifest.get("scenario") != SCENARIO
        or float(data_manifest.get("kappa", -1)) != KAPPA
    ):
        raise AuthorizationError("frozen data scope mismatch")
    if frozen.get("data_hashes") != data_manifest.get("data_hashes"):
        raise AuthorizationError("frozen split data hashes mismatch")
    plan_hash = data_manifest.get("sampling_plan", {}).get("sha256")
    if frozen.get("sampling_plan_sha256") != plan_hash:
        raise AuthorizationError("shared SamplingPlan hash mismatch")
    if not isinstance(plan_hash, str) or len(plan_hash) != 64:
        raise AuthorizationError("shared SamplingPlan hash is invalid")
    continuation = authorization.get("continuation")
    if isinstance(continuation, Mapping):
        if continuation.get("data_hashes") != data_manifest.get(
            "data_hashes"
        ):
            raise AuthorizationError(
                "continuation frozen data hashes mismatch"
            )
        if continuation.get("sampling_plan_sha256") != plan_hash:
            raise AuthorizationError(
                "continuation SamplingPlan hash mismatch"
            )
    return authorization


def apply_final_decision_contract(
    analysis: Mapping[str, Any],
    *,
    c2_evaluable: bool = True,
) -> dict[str, Any]:
    output = dict(analysis)
    if tuple(output.get("primary_family", ())) != PRIMARY_COMPARATORS:
        raise ValueError("C2 primary family differs from the frozen protocol")
    if tuple(output.get("secondary_family", ())) != SECONDARY_COMPARATORS:
        raise ValueError("secondary Holm family differs from the frozen protocol")
    if not c2_evaluable:
        output["c2_status"] = "NOT_EVALUABLE"
        output["c2_supported"] = False
        output["c2_decision"] = C2_NOT_EVALUABLE
    elif output.get("c2_supported") is True:
        output["c2_status"] = "SUPPORTED"
        output["c2_decision"] = (
            "C2 supported in this preregistered experiment"
        )
    else:
        output["c2_status"] = "NOT_SUPPORTED"
        output["c2_decision"] = C2_NOT_SUPPORTED
    return output


def _load_sequence_batch(path: Path) -> Any:
    from benchmarks.types import SequenceBatch

    with np.load(path, allow_pickle=False) as archive:
        values = {
            field: archive[field]
            for field in SequenceBatch.__dataclass_fields__
        }
    return SequenceBatch(**values)


def _load_sampling_plan(path: Path) -> Any:
    from generators.sampling_plan import SamplingPlan

    with np.load(path, allow_pickle=False) as archive:
        plan_hash = str(np.asarray(archive["plan_hash"]).item())
        return SamplingPlan(
            y_entity=archive["y_entity"],
            lengths=archive["lengths"],
            valid_mask=archive["valid_mask"],
            plan_hash=plan_hash,
        )


def terminalize_evaluation_attempt(
    *,
    store: Any,
    evaluation: Mapping[str, Any],
    completion_payload: Mapping[str, Any],
) -> tuple[str, Path]:
    """Route an evaluated seed using hard validity guards only."""
    if evaluation.get("status") != "VALID":
        marker = store.invalid(
            reason=str(evaluation.get("invalid_reason")),
            hard_guards=evaluation["hard_guards"],
        )
        return "INVALID", marker
    marker = store.complete(completion_payload)
    return "COMPLETE", marker


def _manifest_path(
    repository_root: Path,
    value: str | None,
    *,
    fallback: Path,
) -> Path:
    if value is None:
        return fallback
    candidate = Path(value)
    return candidate if candidate.is_absolute() else repository_root / candidate


def load_and_validate_frozen_inputs(
    *,
    repository_root: Path,
    config_path: Path,
    data_manifest_path: Path,
) -> FrozenInputs:
    from eval.model_guards_v2_5 import RowGuardThresholds
    from experiments.frozen_metadata_v2_5 import decode_frozen_metadata
    from experiments.provenance_v2_5 import hash_batch

    manifest_path = _resolved(repository_root, data_manifest_path)
    manifest = json.loads(manifest_path.read_text())
    config = _resolved(repository_root, config_path)
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("frozen data manifest is not COMPLETE")
    if manifest.get("config_sha256") != sha256_file(config):
        raise RuntimeError("frozen data config hash mismatch")
    if (
        manifest.get("scenario") != SCENARIO
        or float(manifest.get("kappa", -1)) != KAPPA
    ):
        raise RuntimeError("frozen data scope mismatch")
    data_directory = manifest_path.parent
    split_paths = {
        split: data_directory / f"{split}.npz"
        for split in ("train", "validation", "test")
    }
    for split, path in split_paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing frozen {split} split: {path}")
        expected_file_hash = manifest.get("file_sha256", {}).get(split)
        if expected_file_hash != sha256_file(path):
            raise RuntimeError(f"frozen {split} file hash mismatch")
    train = _load_sequence_batch(split_paths["train"])
    validation = _load_sequence_batch(split_paths["validation"])
    test = _load_sequence_batch(split_paths["test"])
    batches = {"train": train, "validation": validation, "test": test}
    for split, batch in batches.items():
        if hash_batch(batch) != manifest["data_hashes"][split]:
            raise RuntimeError(f"frozen {split} content hash mismatch")

    plan_value = manifest.get("sampling_plan", {})
    plan_path = _manifest_path(
        repository_root,
        plan_value.get("path"),
        fallback=data_directory / "shared_sampling_plan.npz",
    )
    if sha256_file(plan_path) != plan_value.get("file_sha256"):
        raise RuntimeError("SamplingPlan file hash mismatch")
    plan = _load_sampling_plan(plan_path)
    if plan.plan_hash != plan_value.get("sha256"):
        raise RuntimeError("SamplingPlan content hash mismatch")

    metadata_path = data_directory / "meta.json"
    if sha256_file(metadata_path) != manifest.get("metadata_sha256"):
        raise RuntimeError("frozen metadata hash mismatch")
    metadata = decode_frozen_metadata(
        json.loads(metadata_path.read_text())
    )
    tau = np.asarray(metadata["tau"], dtype=float)

    threshold_path = _manifest_path(
        repository_root,
        manifest.get("row_guard_thresholds_path"),
        fallback=(
            repository_root
            / "artifacts/benchmark_v2_5/prerun_calibration/"
            "row_guard_thresholds.json"
        ),
    )
    if sha256_file(threshold_path) != manifest.get(
        "row_guard_thresholds_sha256"
    ):
        raise RuntimeError("row-guard threshold artifact hash mismatch")
    threshold_artifact = json.loads(threshold_path.read_text())
    if (
        threshold_artifact.get("status") != "COMPLETE"
        or threshold_artifact.get("learned_results_used") is not False
    ):
        raise RuntimeError("row-guard calibration contract mismatch")
    thresholds = RowGuardThresholds(**threshold_artifact["thresholds"])
    return FrozenInputs(
        train=train,
        validation=validation,
        test=test,
        sampling_plan=plan,
        tau=tau,
        row_guard_thresholds=thresholds,
        manifest=manifest,
    )


def _ensure_frozen_data(
    *,
    authorization: Mapping[str, Any],
    repository_root: Path,
    config_path: Path,
    data_manifest_path: Path,
    data_root: Path,
    artifact_root: Path,
) -> None:
    manifest_path = _resolved(repository_root, data_manifest_path)
    if manifest_path.is_file():
        return
    frozen = authorization.get("frozen_data", {})
    if frozen.get("allow_prepare_if_absent") is not True:
        raise AuthorizationError(
            "frozen data is absent and authorization does not permit "
            "exclusive preparation"
        )
    resolved_data_root = _resolved(repository_root, data_root)
    resolved_artifact_root = _resolved(repository_root, artifact_root)
    destination = manifest_path.parent
    calibration = resolved_artifact_root / "prerun_calibration"
    if destination.exists() or calibration.exists():
        raise RuntimeError(
            "partial frozen data/calibration exists; refusing overwrite"
        )
    from scripts.prepare_full_data_v2_5 import prepare_data

    resolved_config = _resolved(repository_root, config_path)
    raw = yaml.safe_load(resolved_config.read_text())
    prepare_data(
        raw,
        config_path=resolved_config,
        data_root=resolved_data_root,
        artifact_root=resolved_artifact_root,
    )
    if not manifest_path.is_file():
        raise RuntimeError("exclusive frozen-data preparation produced no manifest")


def _adapter_config(
    raw: Mapping[str, Any],
    *,
    job: JobPlan,
    inputs: FrozenInputs,
    callbacks: Any | None,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    value = dict(raw["baselines"][job.generator])
    optimizer = value.get("optimizer")
    if isinstance(optimizer, Mapping):
        if job.generator == "ctgan_separate_class":
            generator = optimizer["generator"]
            discriminator = optimizer["discriminator"]
            value.update(
                {
                    "generator_lr": generator["lr"],
                    "generator_decay": generator["weight_decay"],
                    "discriminator_lr": discriminator["lr"],
                    "discriminator_decay": discriminator["weight_decay"],
                }
            )
        else:
            value["lr"] = optimizer["lr"]
            value["weight_decay"] = optimizer.get("weight_decay", 0.0)
    if job.device_class == "gpu":
        value["device"] = "cuda:0"
        value["cuda"] = True
        value["validation_batch"] = inputs.validation
        value["checkpoint_interval_steps"] = job.checkpoint_interval_steps
        if callbacks is not None:
            value["progress_callback"] = callbacks.progress
            value["checkpoint_callback"] = callbacks.checkpoint
    else:
        value["device"] = "cpu"
        value["max_wall_seconds"] = job.max_wall_seconds
    if job.generator == "cof_seqgen":
        value["tau"] = inputs.tau
        value["window_width"] = raw["data"]["window_width"]
        value["temperature"] = raw["data"]["soft_g_temperature"]
    if resume_checkpoint is not None:
        value["resume_checkpoint"] = str(resume_checkpoint)
    return value


def validate_real_adapter_integrations(
    raw: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Import and map all frozen adapters without fit/sample/CUDA/DGP."""
    from generators.full_registry_v2_5 import baseline_registry

    class Callbacks:
        @staticmethod
        def progress(event: Mapping[str, Any]) -> None:
            raise AssertionError("preflight callback must not execute")

        @staticmethod
        def checkpoint(event: Mapping[str, Any], adapter: Any) -> None:
            raise AssertionError("preflight callback must not execute")

    registry = baseline_registry()
    jobs = build_job_plan(raw, sampling_plan_hash="0" * 64)
    job_by_generator = {
        job.generator: job
        for job in jobs
        if job.seed == SEEDS[0]
    }
    inputs = FrozenInputs(
        train=None,
        validation=None,
        test=None,
        sampling_plan=None,
        tau=np.asarray([0.5, 1.5], dtype=float),
        row_guard_thresholds=None,
        manifest={},
    )
    reports = []
    for generator, spec in registry.items():
        adapter = spec.factory()
        fit_parameters = inspect.signature(adapter.fit).parameters
        sample_parameters = inspect.signature(adapter.sample).parameters
        fit_valid = all(
            name in fit_parameters for name in ("train", "config", "seed")
        )
        sample_valid = all(
            name in sample_parameters for name in ("plan", "seed")
        )
        mapped = _adapter_config(
            raw,
            job=job_by_generator[generator],
            inputs=inputs,
            callbacks=Callbacks(),
            resume_checkpoint=Path("/nonexistent/preflight-checkpoint.pt"),
        )
        learned_contract = (
            spec.training_device != "gpu"
            or (
                callable(getattr(adapter, "save_training_checkpoint", None))
                and callable(mapped.get("progress_callback"))
                and callable(mapped.get("checkpoint_callback"))
                and mapped.get("resume_checkpoint")
                == "/nonexistent/preflight-checkpoint.pt"
            )
        )
        if not fit_valid or not sample_valid or not learned_contract:
            raise RuntimeError(
                f"real adapter integration contract failed: {generator}"
            )
        reports.append(
            {
                "generator": generator,
                "adapter": (
                    f"{type(adapter).__module__}.{type(adapter).__qualname__}"
                ),
                "device_class": spec.training_device,
                "fit_signature_valid": fit_valid,
                "sample_signature_valid": sample_valid,
                "config_mapping_valid": True,
                "callback_resume_contract_valid": learned_contract,
            }
        )
    if tuple(item["generator"] for item in reports) != GENERATOR_ORDER:
        raise RuntimeError("real registry order differs from frozen config")
    return {
        "status": "PASS",
        "adapters": reports,
        "calls": {"fit": 0, "sample": 0, "dgp": 0, "cuda": 0},
    }


def _synthetic_writer(sample: Any):
    def writer(path: Path) -> None:
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                **{
                    field: getattr(sample, field)
                    for field in sample.__dataclass_fields__
                },
            )

    return writer


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _actual_budget(adapter: Any, elapsed_seconds: float) -> Mapping[str, Any]:
    value = getattr(adapter, "actual_training_budget", {})
    if isinstance(value, Mapping) and value:
        return _json_safe(value)
    return {
        "actual_steps": 0,
        "actual_wall_seconds": elapsed_seconds,
        "wall_cap_reached": False,
    }


def _budget_cap_reached(value: Mapping[str, Any]) -> bool:
    if value.get("wall_cap_reached") is True:
        return True
    actual = value.get("actual")
    if isinstance(actual, Mapping):
        return any(
            isinstance(item, Mapping) and item.get("wall_cap_reached") is True
            for item in actual.values()
        )
    return any(
        isinstance(item, Mapping) and item.get("wall_cap_reached") is True
        for item in value.values()
    )


def _failure_class(error: BaseException) -> str:
    message = str(error).lower()
    if "out of memory" in message or "cuda oom" in message:
        return "oom"
    if isinstance(error, (ConnectionError, subprocess.SubprocessError)):
        return "infrastructure"
    if isinstance(error, (ValueError, TimeoutError)):
        return "contract"
    return "code"


def _index_paths(store: Any, paths: Iterable[tuple[str, str]]) -> None:
    for relative, role in paths:
        store.append_artifact_index_entry(relative, role=role)


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _prepare_full_attempt(
    *,
    repository_root: Path,
    config_path: Path,
    data_manifest_path: Path,
    artifact_root: Path,
    job: JobPlan,
    gpu: Mapping[str, Any] | None,
    minimum_attempt: int = 1,
) -> Mapping[str, Any]:
    from experiments.full_artifact_store_v2_5 import (
        FullAttemptStore,
        RunAlreadyComplete,
        TERMINAL_FILES,
        canonical_json_bytes,
        run_directory,
    )
    from experiments.provenance_v2_5 import build_manifest

    import torch

    inputs = load_and_validate_frozen_inputs(
        repository_root=repository_root,
        config_path=config_path,
        data_manifest_path=data_manifest_path,
    )
    resolved_config = _resolved(repository_root, config_path)
    raw = yaml.safe_load(resolved_config.read_text())
    gpu_record = (
        {
            "id": int(gpu["physical_index"]),
            "uuid": str(gpu["uuid"]),
            "name": str(gpu["name"]),
        }
        if gpu is not None
        else {"id": None, "uuid": None, "name": None}
    )
    requested = {
        "steps": job.requested_steps,
        "max_wall_seconds": job.max_wall_seconds,
        "checkpoint_interval_steps": job.checkpoint_interval_steps,
    }
    manifest = build_manifest(
        repository_root=repository_root,
        config_path=resolved_config,
        scenario=job.scenario,
        kappa=job.kappa,
        generator=job.generator,
        seed=job.seed,
        train=inputs.train,
        validation=inputs.validation,
        test=inputs.test,
        sampling_plan=inputs.sampling_plan,
        gpu=gpu_record,
        cuda_version=torch.version.cuda,
        pytorch_version=torch.__version__,
        requested_training_budget=requested,
    )
    resolved_artifact_root = _resolved(repository_root, artifact_root)
    manifest_digest = _manifest_hash(manifest)
    run_root = run_directory(
        resolved_artifact_root,
        scenario=job.scenario,
        kappa=job.kappa,
        generator=job.generator,
        seed=job.seed,
    )
    attempts = sorted(run_root.glob("attempt_[0-9][0-9][0-9]"))
    if attempts:
        latest = attempts[-1]
        manifest_path = latest / "manifest.json"
        try:
            existing_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"cannot validate existing attempt manifest: {latest}"
            ) from error
        terminals = [
            name for name in TERMINAL_FILES if (latest / name).is_file()
        ]
        if len(terminals) > 1:
            raise RuntimeError(
                f"existing attempt has conflicting terminal markers: {latest}"
            )
        same_manifest = canonical_json_bytes(
            existing_manifest
        ) == canonical_json_bytes(manifest)
        if (
            same_manifest
            and len(terminals) == 1
            and terminals[0] != "INTERRUPTED.json"
        ):
            marker_path = latest / terminals[0]
            marker = json.loads(marker_path.read_text())
            status = str(marker["status"])
            fallback_classes = {
                "COMPLETE": None,
                "FAILED": "code",
                "INVALID": "hard_guard",
                "UNAVAILABLE": "unavailable",
                "CANCELLED": "dependency_cancelled",
            }
            if status not in fallback_classes:
                raise RuntimeError(
                    f"unknown existing terminal status in {marker_path}"
                )
            return {
                "status": status,
                "job": job.to_dict(),
                "attempt_path": str(latest),
                "attempt": int(latest.name.rsplit("_", 1)[-1]),
                "manifest_hash": manifest_digest,
                "failure_class": marker.get(
                    "failure_class",
                    fallback_classes[status],
                ),
                "failure_fingerprint": marker.get("failure_fingerprint"),
                "mandatory_stop": marker.get("mandatory_stop", False),
                "reused": True,
            }
    try:
        store, selection = FullAttemptStore.select(
            resolved_artifact_root,
            manifest,
            minimum_attempt=minimum_attempt,
        )
    except RunAlreadyComplete as complete:
        return {
            "status": "COMPLETE",
            "job": job.to_dict(),
            "attempt_path": str(complete),
            "attempt": int(
                Path(str(complete)).name.rsplit("_", 1)[-1]
            ),
            "manifest_hash": manifest_digest,
            "failure_class": None,
            "reused": True,
        }
    return {
        "status": "READY",
        "store": store,
        "selection": selection,
        "manifest": manifest,
        "manifest_hash": manifest_digest,
    }


def _execute_full_job_child(
    *,
    repository_root: Path,
    config_path: Path,
    data_manifest_path: Path,
    job: JobPlan,
    gpu: Mapping[str, Any] | None,
    attempt_path: str,
    manifest: Mapping[str, Any],
    resume_checkpoint: str | None,
) -> Mapping[str, Any]:
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu["physical_index"])
    from eval.full_evaluation_v2_5 import (
        evaluate_full_seed,
        fit_full_evaluation_reference,
    )
    from experiments.full_artifact_store_v2_5 import FullAttemptStore
    from experiments.training_callbacks_v2_5 import AttemptTrainingCallbacks

    import torch

    store = FullAttemptStore(Path(attempt_path), manifest)
    inputs = load_and_validate_frozen_inputs(
        repository_root=repository_root,
        config_path=config_path,
        data_manifest_path=data_manifest_path,
    )
    raw = yaml.safe_load(
        _resolved(repository_root, config_path).read_text()
    )
    gpu_record = dict(manifest["gpu"])
    requested = dict(manifest["requested_training_budget"])
    callbacks = (
        AttemptTrainingCallbacks(
            store,
            checkpoint_interval_steps=job.checkpoint_interval_steps,
        )
        if job.device_class == "gpu"
        else None
    )
    started = time.monotonic()
    monitor = _ResourceMonitor(
        store,
        gpu_uuid=(str(gpu["uuid"]) if gpu is not None else None),
    )
    stdout = _StoreStream(store, "stdout")
    stderr = _StoreStream(store, "stderr")
    try:
        try:
            from generators.full_registry_v2_5 import generator_for_v2_5

            adapter = generator_for_v2_5(job.generator)
        except ImportError as error:
            marker = store.unavailable(
                reason=f"{type(error).__name__}: {error}",
            )
            return {
                "status": "UNAVAILABLE",
                "job": job.to_dict(),
                "attempt_path": str(store.path),
                "failure_class": "unavailable",
                "marker": str(marker),
            }
        adapter_config = _adapter_config(
            raw,
            job=job,
            inputs=inputs,
            callbacks=callbacks,
            resume_checkpoint=(
                Path(resume_checkpoint)
                if resume_checkpoint is not None
                else None
            ),
        )
        with monitor, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
            stderr
        ):
            print(
                f"START generator={job.generator} seed={job.seed} "
                f"resume={resume_checkpoint is not None}"
            )
            adapter.fit(inputs.train, config=adapter_config, seed=job.seed)
            elapsed_fit = time.monotonic() - started
            budget = _actual_budget(adapter, elapsed_fit)
            if callbacks is None:
                store.append_progress(
                    {
                        "step": 0,
                        "loss": None,
                        "validation_metric": None,
                        "elapsed_seconds": elapsed_fit,
                        "peak_gpu_memory_bytes": 0,
                    }
                )
                store.write_partial_metrics(
                    {
                        "status": "FITTED",
                        "step": 0,
                        "elapsed_seconds": elapsed_fit,
                    }
                )
            sample = adapter.sample(
                inputs.sampling_plan,
                seed=job.seed + 1_000_000,
            )
            evaluation_reference = fit_full_evaluation_reference(
                inputs.train,
                inputs.test,
                tau=inputs.tau,
                window_width=float(raw["data"]["window_width"]),
                minimum_bin_count=int(
                    raw["evaluation"]["minimum_bin_count"]
                ),
                row_guard_thresholds=inputs.row_guard_thresholds,
                v2_4_reference_contract_verified=True,
            )
            evaluation = evaluate_full_seed(
                sample,
                train=inputs.train,
                real_test=inputs.test,
                plan=inputs.sampling_plan,
                tau=inputs.tau,
                window_width=float(raw["data"]["window_width"]),
                reference=evaluation_reference,
            )
            elapsed_total = time.monotonic() - started
            peak = (
                int(torch.cuda.max_memory_allocated())
                if job.device_class == "gpu" and torch.cuda.is_available()
                else 0
            )
            runtime = {
                "generator": job.generator,
                "seed": job.seed,
                "fit_seconds": elapsed_fit,
                "total_seconds": elapsed_total,
                "peak_gpu_memory_bytes": peak,
                "gpu": gpu_record,
                "requested_training_budget": requested,
                "actual_training_budget": budget,
            }
            metrics = {
                "generator": job.generator,
                "seed": job.seed,
                "association_recovery_error": evaluation.get(
                    "association_recovery_error"
                ),
                "hard_guards": evaluation["hard_guards"],
                "support_diagnostics": evaluation.get(
                    "diagnostics",
                    {},
                ).get("support_diagnostics", {}),
                "sampling_plan_hash": inputs.sampling_plan.plan_hash,
            }
            store.write_immutable_file("sample.npz", _synthetic_writer(sample))
            assert callbacks is not None or hasattr(adapter, "fit")
            if callbacks is not None:
                callbacks.save_final_checkpoint(adapter)
            else:
                store.write_immutable_file(
                    "checkpoints/final.pt",
                    lambda path: torch.save(adapter, path),
                )
            store.write_immutable_json("metrics.json", _json_safe(metrics))
            store.write_immutable_json("runtime.json", _json_safe(runtime))
            store.write_immutable_json(
                "evaluation.json",
                _json_safe(evaluation),
            )
            _index_paths(
                store,
                (
                    ("sample.npz", "final_sample"),
                    ("checkpoints/final.pt", "final_checkpoint"),
                    ("metrics.json", "raw_metrics"),
                    ("runtime.json", "runtime"),
                    ("evaluation.json", "evaluation"),
                    ("stdout.log", "stdout_log"),
                    ("stderr.log", "stderr_log"),
                    ("progress.jsonl", "progress"),
                    ("partial_metrics.json", "partial_metrics"),
                    ("resource.jsonl", "resource"),
                ),
            )
            terminal_status, marker = terminalize_evaluation_attempt(
                store=store,
                evaluation=evaluation,
                completion_payload={
                    "association_recovery_error": evaluation[
                        "association_recovery_error"
                    ],
                    "hard_guards": evaluation["hard_guards"],
                    "sampling_plan_hash": inputs.sampling_plan.plan_hash,
                    "actual_training_budget": budget,
                },
            )
            if terminal_status == "INVALID":
                return {
                    "status": "INVALID",
                    "job": job.to_dict(),
                    "attempt_path": str(store.path),
                    "failure_class": "hard_guard",
                    "marker": str(marker),
                }
            return {
                "status": "COMPLETE",
                "job": job.to_dict(),
                "attempt_path": str(store.path),
                "failure_class": None,
                "marker": str(marker),
            }
    except Exception as error:
        store.append_log("stderr", traceback.format_exc())
        peak = (
            int(torch.cuda.max_memory_allocated())
            if job.device_class == "gpu" and torch.cuda.is_available()
            else 0
        )
        pointer = store.path / "checkpoints" / "latest"
        last_checkpoint = None
        if pointer.is_file():
            try:
                last_checkpoint = json.loads(pointer.read_text())["path"]
            except (KeyError, json.JSONDecodeError):
                last_checkpoint = None
        marker = store.fail(
            exception=f"{type(error).__name__}: {error}",
            last_checkpoint=last_checkpoint,
            peak_gpu_memory_bytes=peak,
            failure_class=_failure_class(error),
            actual_elapsed_seconds=time.monotonic() - started,
        )
        marker_value = json.loads(marker.read_text())
        return {
            "status": "FAILED",
            "job": job.to_dict(),
            "attempt_path": str(store.path),
            "marker": str(marker),
            "failure_class": marker_value["failure_class"],
            "failure_fingerprint": marker_value["failure_fingerprint"],
            "mandatory_stop": marker_value.get("mandatory_stop", False),
            "exception": f"{type(error).__name__}: {error}",
        }


def execute_full_job(
    *,
    repository_root: Path,
    config_path: Path,
    data_manifest_path: Path,
    artifact_root: Path,
    job: JobPlan,
    gpu: Mapping[str, Any] | None,
    operator_stop_event: threading.Event | None = None,
    minimum_attempt: int = 1,
) -> Mapping[str, Any]:
    prepared = _prepare_full_attempt(
        repository_root=repository_root,
        config_path=config_path,
        data_manifest_path=data_manifest_path,
        artifact_root=artifact_root,
        job=job,
        gpu=gpu,
        minimum_attempt=minimum_attempt,
    )
    if prepared["status"] != "READY":
        return prepared
    store = prepared["store"]
    selection = prepared["selection"]
    result = run_owned_attempt_process(
        worker=_execute_full_job_child,
        worker_kwargs={
            "repository_root": repository_root,
            "config_path": config_path,
            "data_manifest_path": data_manifest_path,
            "job": job,
            "gpu": gpu,
            "attempt_path": str(store.path),
            "manifest": prepared["manifest"],
            "resume_checkpoint": (
                str(selection.latest_checkpoint)
                if selection.latest_checkpoint is not None
                else None
            ),
        },
        store=store,
        max_wall_seconds=job.max_wall_seconds,
        termination_grace_seconds=5.0,
        process_start_method="spawn",
        operator_stop_event=operator_stop_event,
    )
    return {
        "job": job.to_dict(),
        "attempt_path": str(store.path),
        "attempt": selection.attempt,
        "manifest_hash": prepared["manifest_hash"],
        **result,
    }


def cancel_full_job(
    *,
    repository_root: Path,
    config_path: Path,
    data_manifest_path: Path,
    artifact_root: Path,
    job: JobPlan,
    reason: str,
    minimum_attempt: int = 1,
) -> Mapping[str, Any]:
    prepared = _prepare_full_attempt(
        repository_root=repository_root,
        config_path=config_path,
        data_manifest_path=data_manifest_path,
        artifact_root=artifact_root,
        job=job,
        gpu=None,
        minimum_attempt=minimum_attempt,
    )
    if prepared["status"] != "READY":
        return prepared
    store = prepared["store"]
    selection = prepared["selection"]
    marker = store.cancelled(reason=reason)
    return {
        "status": "CANCELLED",
        "failure_class": "dependency_cancelled",
        "failure_fingerprint": hashlib.sha256(reason.encode()).hexdigest(),
        "attempt": selection.attempt,
        "attempt_path": str(store.path),
        "manifest_hash": prepared["manifest_hash"],
        "marker": str(marker),
    }


def _run_gpu_wave(
    assignments: Sequence[GPUAssignment],
    *,
    repository_root: Path,
    config_path: Path,
    data_manifest_path: Path,
    artifact_root: Path,
    minimum_attempt: int = 1,
) -> list[Mapping[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    futures = {}
    results_by_order: dict[int, Mapping[str, Any]] = {}
    operator_stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        for assignment in assignments:
            future = executor.submit(
                execute_full_job,
                repository_root=repository_root,
                config_path=config_path,
                data_manifest_path=data_manifest_path,
                artifact_root=artifact_root,
                job=assignment.job,
                gpu=assignment.gpu,
                operator_stop_event=operator_stop_event,
                minimum_attempt=minimum_attempt,
            )
            futures[future] = assignment.job.execution_order
        try:
            for future in as_completed(futures):
                order = futures[future]
                results_by_order[order] = future.result()
        except RunnerInterrupted:
            operator_stop_event.set()
            for pending in futures:
                if pending.done():
                    continue
                try:
                    pending.result()
                except RunnerInterrupted:
                    pass
            raise
    return [
        results_by_order[assignment.job.execution_order]
        for assignment in assignments
    ]


def _all_success(results: Iterable[Mapping[str, Any]]) -> bool:
    return all(result.get("status") == "COMPLETE" for result in results)


def _latest_attempt(
    artifact_root: Path,
    job: JobPlan,
) -> Path | None:
    base = (
        artifact_root
        / "full"
        / job.scenario
        / f"kappa_{job.kappa:.2f}"
        / job.generator
        / f"seed_{job.seed}"
    )
    attempts = sorted(base.glob("attempt_[0-9][0-9][0-9]"))
    return attempts[-1] if attempts else None


def _collect_raw_records(
    artifact_root: Path,
    jobs: Sequence[JobPlan],
) -> tuple[
    dict[str, dict[int, dict[str, Any]]],
    list[dict[str, Any]],
]:
    records: dict[str, dict[int, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for job in jobs:
        attempt = _latest_attempt(artifact_root, job)
        if attempt is None:
            raise RuntimeError(
                f"cannot aggregate before all 65 jobs are terminal: {job}"
            )
        terminals = [
            name
            for name in (
                "CANCELLED.json",
                "COMPLETE.json",
                "FAILED.json",
                "INVALID.json",
                "UNAVAILABLE.json",
            )
            if (attempt / name).is_file()
        ]
        if len(terminals) != 1:
            raise RuntimeError(f"attempt is not singly terminal: {attempt}")
        marker = json.loads((attempt / terminals[0]).read_text())
        manifest = json.loads((attempt / "manifest.json").read_text())
        record: dict[str, Any] = {
            "status": marker["status"],
            "sampling_plan_hash": manifest["sampling_plan_hash"],
            "hard_guards": marker.get("hard_guards", {}),
            "association_recovery_error": marker.get(
                "association_recovery_error"
            ),
            "support_diagnostics": {},
        }
        runtime: Mapping[str, Any] = {}
        evaluation: Mapping[str, Any] = {}
        evaluation_path = attempt / "evaluation.json"
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text())
            diagnostics = evaluation.get("diagnostics", {})
            support_diagnostics = diagnostics.get(
                "support_diagnostics",
                {},
            )
            if (
                not support_diagnostics
                and isinstance(evaluation.get("confirmatory_support"), Mapping)
            ):
                support_diagnostics = {
                    "8_bin": evaluation["confirmatory_support"]
                }
            record["support_diagnostics"] = support_diagnostics
        if terminals[0] == "COMPLETE.json":
            metrics = json.loads((attempt / "metrics.json").read_text())
            runtime = json.loads((attempt / "runtime.json").read_text())
            record.update(
                {
                    "hard_guards": metrics["hard_guards"],
                    "association_recovery_error": metrics[
                        "association_recovery_error"
                    ],
                    "support_diagnostics": metrics.get(
                        "support_diagnostics",
                        record["support_diagnostics"],
                    ),
                }
            )
        support_4 = record["support_diagnostics"].get("4_bin", {})
        support_8 = record["support_diagnostics"].get("8_bin", {})
        records.setdefault(job.generator, {})[job.seed] = record
        rows.append(
            {
                "generator": job.generator,
                "seed": job.seed,
                "status": marker["status"],
                "association_recovery_error": record[
                    "association_recovery_error"
                ],
                "sampling_plan_hash": manifest["sampling_plan_hash"],
                "attempt_path": str(attempt),
                "total_seconds": runtime.get("total_seconds"),
                "fit_seconds": runtime.get("fit_seconds"),
                "peak_gpu_memory_bytes": runtime.get(
                    "peak_gpu_memory_bytes"
                ),
                "support_4_all_bins_valid": support_4.get(
                    "all_bins_valid"
                ),
                "support_4_dropped_bin_macro_gap": support_4.get(
                    "dropped_bin_macro_gap"
                ),
                "support_4_occupancy_penalized_gap": support_4.get(
                    "occupancy_penalized_gap"
                ),
                "support_4_invalid_bin_count": support_4.get(
                    "invalid_bin_count"
                ),
                "support_8_all_bins_valid": support_8.get(
                    "all_bins_valid"
                ),
                "support_8_dropped_bin_macro_gap": support_8.get(
                    "dropped_bin_macro_gap"
                ),
                "support_8_occupancy_penalized_gap": support_8.get(
                    "occupancy_penalized_gap"
                ),
                "support_8_invalid_bin_count": support_8.get(
                    "invalid_bin_count"
                ),
            }
        )
    return records, rows


def _support_diagnostic_rows(
    raw_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for raw_row in raw_rows:
        for bin_count in (4, 8):
            prefix = f"support_{bin_count}_"
            rows.append(
                {
                    "generator": raw_row["generator"],
                    "seed": raw_row["seed"],
                    "status": raw_row["status"],
                    "bin_count": bin_count,
                    "all_bins_valid": raw_row.get(
                        f"{prefix}all_bins_valid"
                    ),
                    "dropped_bin_macro_gap": raw_row.get(
                        f"{prefix}dropped_bin_macro_gap"
                    ),
                    "occupancy_penalized_gap": raw_row.get(
                        f"{prefix}occupancy_penalized_gap"
                    ),
                    "invalid_bin_count": raw_row.get(
                        f"{prefix}invalid_bin_count"
                    ),
                }
            )
    return rows


def render_support_diagnostics_markdown(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "## Support diagnostics (non-decision)",
        "",
        (
            "| Generator | Seed | Bins | All bins valid | "
            "Dropped-bin score | Occupancy-penalized score | Invalid bins |"
        ),
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {generator} | {seed} | {bin_count}-bin | "
            "{all_bins_valid} | {dropped_bin_macro_gap} | "
            "{occupancy_penalized_gap} | {invalid_bin_count} |".format(
                **row
            )
        )
    lines.extend(
        (
            "",
            (
                "These 4-bin and 8-bin values are diagnostics only and "
                "cannot change the primary decision."
            ),
            "",
        )
    )
    return "\n".join(lines)


def _write_csv_exclusive(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _csv_bytes(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fieldnames})
    return buffer.getvalue().encode("utf-8")


def _next_exclusive_path(
    directory: Path,
    *,
    prefix: str,
    suffix: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for number in range(1, 10_000):
        candidate = directory / f"{prefix}_{number:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot allocate append-only {prefix} path")


def _prior_finalization_audit(full_root: Path) -> Mapping[str, Any]:
    attempts = []
    for path in sorted(full_root.glob("finalization_attempt_[0-9][0-9][0-9]")):
        if not path.is_dir():
            continue
        files = [
            {
                "path": item.relative_to(path).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
            for item in sorted(path.rglob("*"))
            if item.is_file()
        ]
        attempts.append(
            {
                "attempt": path.name,
                "status": (
                    "COMPLETE"
                    if (path / "FINALIZATION_COMPLETE.json").is_file()
                    else "PARTIAL"
                ),
                "files": files,
            }
        )
    return {
        "schema_version": "benchmark-v2.5-prior-finalization-audit-v1",
        "attempts": attempts,
    }


def _validate_staged_finalization(
    full_root: Path,
    marker: Mapping[str, Any],
) -> Path:
    attempt_name = marker.get("finalization_attempt")
    if (
        not isinstance(attempt_name, str)
        or not attempt_name.startswith("finalization_attempt_")
        or "/" in attempt_name
    ):
        raise RuntimeError("finalization marker has an invalid attempt path")
    attempt_path = full_root / attempt_name
    index_path = attempt_path / "artifact_index.json"
    checksum_path = attempt_path / "checksum_manifest_report.json"
    completion_path = attempt_path / "FINALIZATION_COMPLETE.json"
    required = (index_path, checksum_path, completion_path)
    if any(not path.is_file() for path in required):
        raise RuntimeError("published finalization is incomplete")
    index_hash = sha256_file(index_path)
    checksum_hash = sha256_file(checksum_path)
    if (
        marker.get("artifact_index_sha256") != index_hash
        or marker.get("checksum_sha256") != checksum_hash
    ):
        raise RuntimeError("published finalization marker is corrupt")
    checksum = json.loads(checksum_path.read_text())
    completion = json.loads(completion_path.read_text())
    if (
        checksum.get("artifact_index_sha256") != index_hash
        or completion.get("artifact_index_sha256") != index_hash
        or completion.get("checksum_sha256") != checksum_hash
        or completion.get("status") != "FINALIZATION_COMPLETE"
    ):
        raise RuntimeError("staged finalization checksum contract is corrupt")
    index = json.loads(index_path.read_text())
    indexed_artifacts = index.get("artifacts", [])
    if (
        index.get("artifact_count") != len(indexed_artifacts)
        or checksum.get("indexed_artifact_count") != len(indexed_artifacts)
    ):
        raise RuntimeError("staged finalization artifact count is corrupt")
    base = attempt_path.resolve()
    for artifact in indexed_artifacts:
        path = (attempt_path / artifact["path"]).resolve()
        if (
            (path == base or base not in path.parents)
            or not path.is_file()
            or path.stat().st_size != int(artifact["bytes"])
            or sha256_file(path) != artifact["sha256"]
        ):
            raise RuntimeError(
                f"staged final artifact is corrupt: {artifact['path']}"
            )
    return attempt_path


def publish_finalization_artifacts(
    *,
    full_root: Path,
    artifacts: Mapping[str, bytes],
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Publish final outputs through an append-only staging attempt."""
    if not artifacts:
        raise ValueError("finalization requires artifacts")
    existing_marker = full_root / "FINAL_COMPLETE.json"
    if existing_marker.is_file():
        marker = json.loads(existing_marker.read_text())
        _validate_staged_finalization(full_root, marker)
        return marker
    full_root.mkdir(parents=True, exist_ok=True)
    attempt_path = _next_exclusive_path(
        full_root,
        prefix="finalization_attempt",
        suffix="",
    )
    attempt_path.mkdir()
    audit = _prior_finalization_audit(full_root)
    # The just-created empty attempt is not prior state.
    audit = {
        **audit,
        "attempts": [
            value
            for value in audit["attempts"]
            if value["attempt"] != attempt_path.name
        ],
    }
    audit_path = attempt_path / "prior_partial_finalizations.json"
    _write_json_exclusive(audit_path, audit)
    entries = [
        {
            "path": audit_path.name,
            "bytes": audit_path.stat().st_size,
            "sha256": sha256_file(audit_path),
        }
    ]
    for relative, payload in sorted(artifacts.items()):
        target = (attempt_path / relative).resolve()
        base = attempt_path.resolve()
        if target == base or base not in target.parents:
            raise ValueError("finalization artifact escapes staging attempt")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256_file(target)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise RuntimeError(f"finalization hash mismatch: {relative}")
        entries.append(
            {
                "path": target.relative_to(attempt_path).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": digest,
            }
        )
    index_path = attempt_path / "artifact_index.json"
    _write_json_exclusive(
        index_path,
        {
            "schema_version": "benchmark-v2.5-final-artifact-index-v2",
            "artifact_count": len(entries),
            "artifacts": sorted(entries, key=lambda value: value["path"]),
        },
    )
    for entry in entries:
        target = attempt_path / entry["path"]
        if (
            target.stat().st_size != entry["bytes"]
            or sha256_file(target) != entry["sha256"]
        ):
            raise RuntimeError(
                f"staged finalization verification failed: {entry['path']}"
            )
    checksum_path = attempt_path / "checksum_manifest_report.json"
    _write_json_exclusive(
        checksum_path,
        {
            "schema_version": "benchmark-v2.5-checksum-manifest-v2",
            "artifact_index_sha256": sha256_file(index_path),
            "indexed_artifact_count": len(entries),
            "provenance": dict(provenance),
        },
    )
    completion_path = attempt_path / "FINALIZATION_COMPLETE.json"
    _write_json_exclusive(
        completion_path,
        {
            "status": "FINALIZATION_COMPLETE",
            "completed_at": _utc_timestamp(),
            "artifact_index_sha256": sha256_file(index_path),
            "checksum_sha256": sha256_file(checksum_path),
        },
    )
    number = int(attempt_path.name.rsplit("_", 1)[1])
    marker = {
        "schema_version": "benchmark-v2.5-final-complete-v1",
        "status": "FINAL_COMPLETE",
        "finalization_attempt": attempt_path.name,
        "attempt": number,
        "completed_at": _utc_timestamp(),
        "artifact_index_sha256": sha256_file(index_path),
        "checksum_sha256": sha256_file(checksum_path),
    }
    _atomic_json_replace(existing_marker, marker)
    return marker


def _validate_existing_final_outputs(artifact_root: Path) -> Path | None:
    full_root = artifact_root / "full"
    marker_path = full_root / "FINAL_COMPLETE.json"
    if not marker_path.is_file():
        return None
    marker = json.loads(marker_path.read_text())
    _validate_staged_finalization(full_root, marker)
    return marker_path


def _finalize_full_results(
    *,
    artifact_root: Path,
    jobs: Sequence[JobPlan],
    raw: Mapping[str, Any],
    c2_evaluable: bool,
    run_manifest_hash: str,
    continuation_provenance: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    from eval.full_statistics_v2_5 import analyze_full_experiment

    records, raw_rows = _collect_raw_records(artifact_root, jobs)
    c2_required = ("cof_seqgen",) + PRIMARY_COMPARATORS
    c2_valid = all(
        generator in records
        and set(records[generator]) == set(SEEDS)
        and all(
            records[generator][seed]["status"] == "COMPLETE"
            for seed in SEEDS
        )
        for generator in c2_required
    )
    analysis = analyze_full_experiment(
        records,
        bootstrap_resamples=int(raw["scope"]["bootstrap_resamples"]),
        bootstrap_seed=int(raw["scope"]["bootstrap_seed"]),
    )
    analysis = apply_final_decision_contract(
        analysis,
        c2_evaluable=(c2_evaluable and c2_valid),
    )
    if not c2_valid:
        retained_pairwise = {
            generator: comparison
            for generator, comparison in analysis.get("pairwise", {}).items()
            if generator not in PRIMARY_COMPARATORS
        }
        analysis = {
            **analysis,
            "status": "INVALID",
            "reason": (
                "C2 aggregate not computed because CoF/CTGAN/TVAE/"
                "empirical_iid did not all have five valid seeds"
            ),
            "pairwise": retained_pairwise,
            "c2_checks": {},
            "c2_supported": False,
            "c2_status": "NOT_EVALUABLE",
            "c2_decision": C2_NOT_EVALUABLE,
            "c2_aggregate_computed": False,
        }
    else:
        analysis["c2_aggregate_computed"] = True

    raw_fields = (
        "generator",
        "seed",
        "status",
        "association_recovery_error",
        "sampling_plan_hash",
        "attempt_path",
    )
    runtime_fields = (
        "generator",
        "seed",
        "status",
        "fit_seconds",
        "total_seconds",
        "peak_gpu_memory_bytes",
    )
    aggregate_rows = [
        {
            "generator": generator,
            **summary,
        }
        for generator, summary in analysis.get("summaries", {}).items()
    ]
    effect_rows = [
        {
            "generator": generator,
            **comparison,
        }
        for generator, comparison in analysis.get("pairwise", {}).items()
    ]
    aggregate_fields = (
        "generator",
        "status",
        "reason",
        "mean_association_recovery_error",
        "std_association_recovery_error",
        "seeds",
        "errors",
    )
    effect_fields = (
        "generator",
        "status",
        "reason",
        "multiplicity_family",
        "mean_effect",
        "hedges_g",
        "unpaired_bootstrap_95_ci",
        "welch_t_two_sided",
        "welch_p_two_sided",
        "holm_adjusted_p",
        "reference_only",
    )
    support_rows = _support_diagnostic_rows(raw_rows)
    support_fields = (
        "generator",
        "seed",
        "status",
        "bin_count",
        "all_bins_valid",
        "dropped_bin_macro_gap",
        "occupancy_penalized_gap",
        "invalid_bin_count",
    )
    report = "\n".join(
        (
            "# Benchmark v2.5 full experiment report",
            "",
            f"- Status: {analysis['status']}",
            f"- Primary endpoint: {analysis['primary_endpoint']}",
            f"- C2 supported: {analysis['c2_supported']}",
            f"- Conclusion: {analysis['c2_decision']}",
            "- Partial seed results were not used for decisions.",
            "",
        )
    ) + "\n" + render_support_diagnostics_markdown(support_rows)
    artifacts = {
        "raw_results.csv": _csv_bytes(raw_rows, raw_fields),
        "per_seed_runtime.csv": _csv_bytes(raw_rows, runtime_fields),
        "support_diagnostics.csv": _csv_bytes(
            support_rows,
            support_fields,
        ),
        "aggregate_statistics.csv": _csv_bytes(
            aggregate_rows,
            aggregate_fields,
        ),
        "effect_sizes_and_holm.csv": _csv_bytes(
            effect_rows,
            effect_fields,
        ),
        "full_experiment_report.md": report.encode("utf-8"),
        "analysis.json": (
            json.dumps(
                _json_safe(analysis),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    }
    final_marker = publish_finalization_artifacts(
        full_root=artifact_root / "full",
        artifacts=artifacts,
        provenance={
            "run_manifest_hash": run_manifest_hash,
            "shared_sampling_plan_sha256": analysis.get(
                "shared_sampling_plan_hash"
            ),
            "continuation": (
                dict(continuation_provenance)
                if continuation_provenance is not None
                else None
            ),
        },
    )
    return {"analysis": analysis, "final_marker": final_marker}


def run_full_experiment(
    *,
    repository_root: Path,
    config_path: Path,
    data_manifest_path: Path,
    authorization_path: Path,
    artifact_root: Path,
    data_root: Path,
) -> Mapping[str, Any]:
    resolved_artifact_root = _resolved(repository_root, artifact_root)
    resolved_authorization = _resolved(repository_root, authorization_path)
    authorization_hash = (
        sha256_file(resolved_authorization)
        if resolved_authorization.is_file()
        else hashlib.sha256(b"missing-authorization").hexdigest()
    )
    run_manifest_hash = _manifest_hash(
        {
            "authorization_sha256": authorization_hash,
            "source_commit": _git_output(
                repository_root,
                "rev-parse",
                "HEAD",
            ),
            "config_path": str(config_path),
            "data_manifest_path": str(data_manifest_path),
        }
    )
    raw_authorization: Mapping[str, Any] = {}
    if resolved_authorization.is_file():
        try:
            candidate = json.loads(resolved_authorization.read_text())
            if isinstance(candidate, Mapping):
                raw_authorization = candidate
        except json.JSONDecodeError:
            pass
    continuation_requested = isinstance(
        raw_authorization.get("continuation"),
        Mapping,
    )
    run_state_root = resolved_artifact_root / "full"
    if continuation_requested:
        run_state_root = run_state_root / "continuation_attempt_003"
    run_state = FullRunStateStore(
        run_state_root,
        run_manifest_hash=run_manifest_hash,
    )
    run_state.start()
    try:
        authorization = validate_authorization_static(
            authorization_path,
            repository_root=repository_root,
            config_path=config_path,
        )
        _verify_execution_source_clean(repository_root)
        _ensure_frozen_data(
            authorization=authorization,
            repository_root=repository_root,
            config_path=config_path,
            data_manifest_path=data_manifest_path,
            data_root=data_root,
            artifact_root=artifact_root,
        )
        validate_full_authorization(
            authorization_path,
            repository_root=repository_root,
            config_path=config_path,
            data_manifest_path=data_manifest_path,
        )
        inputs = load_and_validate_frozen_inputs(
            repository_root=repository_root,
            config_path=config_path,
            data_manifest_path=data_manifest_path,
        )
        resolved_config = _resolved(repository_root, config_path)
        raw = yaml.safe_load(resolved_config.read_text())
        jobs = build_job_plan(
            raw,
            sampling_plan_hash=inputs.sampling_plan.plan_hash,
        )
        continuation = authorization.get("continuation")
        continuation_plan = (
            plan_scheduler_continuation(
                artifact_root=resolved_artifact_root,
                jobs=jobs,
                continuation=continuation,
            )
            if isinstance(continuation, Mapping)
            else None
        )
        validate_real_adapter_integrations(raw)
        initial_selected, initial_excluded = (
            assert_full_gpu_runtime_access(require_idle=False)
        )
    except Exception as error:
        run_state.stop(
            reason=f"global preflight contract failed: {error}",
            mandatory=True,
        )
        raise
    final_marker_path = _validate_existing_final_outputs(
        resolved_artifact_root
    )
    if final_marker_path is not None:
        run_state.complete(
            final_marker_hash=sha256_file(final_marker_path)
        )
        return {
            "status": "COMPLETE_EXISTING",
            "final_complete": str(final_marker_path),
            "sha256": sha256_file(final_marker_path),
            "jobs_started": 0,
        }
    policy = RunPolicyController(run_state)
    minimum_attempt = (
        int(continuation_plan["new_attempt"])
        if continuation_plan is not None
        else 1
    )
    pending_keys = (
        {
            (item["job"]["generator"], int(item["job"]["seed"]))
            for item in continuation_plan["pending"]
        }
        if continuation_plan is not None
        else {(job.generator, job.seed) for job in jobs}
    )
    if continuation_plan is not None:
        replay_continuation_results(
            jobs=jobs,
            continuation_plan=continuation_plan,
            policy=policy,
        )

    def execute_cpu(job: JobPlan) -> Mapping[str, Any]:
        return execute_full_job(
            repository_root=repository_root,
            config_path=config_path,
            data_manifest_path=data_manifest_path,
            artifact_root=artifact_root,
            job=job,
            gpu=None,
            minimum_attempt=minimum_attempt,
        )

    def cancel_job(job: JobPlan) -> Mapping[str, Any]:
        return cancel_full_job(
            repository_root=repository_root,
            config_path=config_path,
            data_manifest_path=data_manifest_path,
            artifact_root=artifact_root,
            job=job,
            reason=(
                "remaining seeds cancelled after generator terminal failure"
            ),
            minimum_attempt=minimum_attempt,
        )

    for job in [
        value
        for value in jobs
        if value.device_class == "cpu"
        and (value.generator, value.seed) in pending_keys
    ]:
        if policy.mandatory_stopped:
            break
        try:
            result = (
                cancel_job(job)
                if policy.should_cancel(job)
                else execute_cpu(job)
            )
        except RunnerInterrupted as error:
            run_state.stop(
                reason=str(error),
                mandatory=True,
                interrupted=True,
            )
            raise
        except Exception as error:
            run_state.stop(
                reason=f"cell manifest/global contract failed: {error}",
                mandatory=True,
            )
            raise
        policy.observe(job, result)
    if policy.mandatory_stopped:
        return {
            "status": "STOPPED",
            **policy.summary(expected_jobs=len(jobs)),
        }

    preflight_selection_path = _next_exclusive_path(
        resolved_artifact_root / "full",
        prefix="gpu_preflight_selection",
        suffix=".json",
    )
    _write_json_exclusive(
        preflight_selection_path,
        {
            "timestamp": _utc_timestamp(),
            "selected": initial_selected,
            "excluded": initial_excluded,
            "idle_gpu_required_for_preflight": False,
        },
    )
    gpu_jobs = [
        value
        for value in jobs
        if value.device_class == "gpu"
        and (value.generator, value.seed) in pending_keys
    ]
    try:
        gpu_schedule = run_live_gpu_schedule(
            gpu_jobs,
            inventory_provider=query_gpu_inventory,
            execute_wave=lambda assignments: _run_gpu_wave(
                assignments,
                repository_root=repository_root,
                config_path=config_path,
                data_manifest_path=data_manifest_path,
                artifact_root=artifact_root,
                minimum_attempt=minimum_attempt,
            ),
            cancel=lambda job, reason: cancel_full_job(
                repository_root=repository_root,
                config_path=config_path,
                data_manifest_path=data_manifest_path,
                artifact_root=artifact_root,
                job=job,
                reason=reason,
                minimum_attempt=minimum_attempt,
            ),
            run_state=run_state,
            max_concurrency=GPU_MAX_CONCURRENCY,
            wait_budget_seconds=GPU_ZERO_IDLE_WAIT_BUDGET_SECONDS,
            poll_interval_seconds=GPU_POLL_INTERVAL_SECONDS,
            policy=policy,
        )
    except RunnerInterrupted as error:
        run_state.stop(
            reason=str(error),
            mandatory=True,
            interrupted=True,
        )
        raise
    except Exception as error:
        run_state.stop(
            reason=f"GPU cell manifest/global contract failed: {error}",
            mandatory=True,
        )
        raise
    if gpu_schedule["status"] == "STOPPED":
        return {
            "status": "STOPPED",
            **policy.summary(expected_jobs=len(jobs)),
            "gpu_schedule": gpu_schedule,
        }
    schedule = policy.summary(expected_jobs=len(jobs))
    if policy.mandatory_stopped or not schedule["all_jobs_terminal"]:
        return {"status": "STOPPED", **schedule}
    try:
        finalized = _finalize_full_results(
            artifact_root=resolved_artifact_root,
            jobs=jobs,
            raw=raw,
            c2_evaluable=policy.c2_evaluable,
            run_manifest_hash=run_manifest_hash,
            continuation_provenance=(
                {
                    "preserved_attempt": 2,
                    "new_attempt": 3,
                    "reused_terminal_jobs": len(
                        continuation_plan["reused"]
                    ),
                    "executed_pending_jobs": len(
                        continuation_plan["pending"]
                    ),
                    "attempt_002_tree_sha256": continuation_plan[
                        "attempt_tree_sha256"
                    ],
                    "attempt_002_terminal_marker_tree_sha256": (
                        continuation_plan[
                            "terminal_marker_tree_sha256"
                        ]
                    ),
                }
                if continuation_plan is not None
                else None
            ),
        )
    except Exception as error:
        run_state.stop(
            reason=f"finalization artifact contract failed: {error}",
            mandatory=True,
        )
        raise
    final_path = resolved_artifact_root / "full" / "FINAL_COMPLETE.json"
    run_state.complete(final_marker_hash=sha256_file(final_path))
    return {
        "status": "COMPLETE",
        "job_count": len(jobs),
        **schedule,
        **finalized,
    }


def run_plan(
    *,
    repository_root: Path,
    config_path: Path,
    data_manifest_path: Path,
) -> Mapping[str, Any]:
    resolved_config = _resolved(repository_root, config_path)
    resolved_data = _resolved(repository_root, data_manifest_path)
    _verify_execution_source_clean(repository_root)
    raw = yaml.safe_load(resolved_config.read_text())
    inputs = load_and_validate_frozen_inputs(
        repository_root=repository_root,
        config_path=config_path,
        data_manifest_path=data_manifest_path,
    )
    plan_hash = inputs.sampling_plan.plan_hash
    jobs = build_job_plan(raw, sampling_plan_hash=plan_hash)
    adapter_preflight = validate_real_adapter_integrations(raw)
    return {
        "schema_version": "benchmark-v2.5-full-plan-v1",
        "source_commit": _git_output(repository_root, "rev-parse", "HEAD"),
        "config_sha256": sha256_file(resolved_config),
        "data_manifest_sha256": sha256_file(resolved_data),
        "sampling_plan_sha256": plan_hash,
        "job_count": len(jobs),
        "jobs": [job.to_dict() for job in jobs],
        "frozen_data_deep_validation": True,
        "data_hashes": dict(inputs.manifest["data_hashes"]),
        "adapter_preflight": adapter_preflight,
        "execution_performed": False,
        "gpu_queried": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dedicated frozen-protocol v2.5 full orchestrator.",
    )
    parser.add_argument("--mode", choices=("plan", "dry-run", "full"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--data-manifest", type=Path, default=DATA_MANIFEST_PATH)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION_PATH)
    parser.add_argument("--work-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(".").resolve()
    if args.mode == "dry-run":
        if args.work_root is None:
            raise ValueError("dry-run requires an explicit --work-root")
        result = run_dry_run(
            repository_root=repository_root,
            config_path=args.config,
            work_root=args.work_root,
        )
        print(json.dumps(result, indent=2))
        return
    if args.mode == "full":
        result = run_full_experiment(
            repository_root=repository_root,
            config_path=args.config,
            data_manifest_path=args.data_manifest,
            authorization_path=args.authorization,
            artifact_root=ARTIFACT_ROOT,
            data_root=Path("data/benchmark_v2_5"),
        )
        print(json.dumps(_json_safe(result), indent=2))
        return
    result = run_plan(
        repository_root=repository_root,
        config_path=args.config,
        data_manifest_path=args.data_manifest,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
