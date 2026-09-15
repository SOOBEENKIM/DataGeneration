from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_benchmark
from experiments.provenance_v2_5 import hash_code
from generators.cof_seqgen_adapter import CoFSeqGenAdapter
from generators.sampling_plan import SamplingPlan
from scripts.validate_full_experiment_v2_5_preparation import validate_config


DEFAULT_CONFIG = Path("configs/benchmark_v2/full_v2_5.yaml")
DEFAULT_REPORT = Path(
    "artifacts/benchmark_v2_5/capacity_preflight/"
    "capacity_preflight_report.md"
)
FORBIDDEN_DURABLE_OUTPUTS = {
    "checkpoint",
    "sample",
    "association_recovery",
    "row_guard",
    "fidelity",
    "tvd",
    "coherence",
    "quality_metric",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def _nvidia_gpu(index: int) -> Mapping[str, Any]:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=index,uuid,name,driver_version,"
            "memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    values = [part.strip() for part in query.split(",")]
    if len(values) != 7:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {query}")
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    process_rows = [
        line.strip()
        for line in processes.splitlines()
        if line.strip() and "No running processes" not in line
    ]
    process_count = sum(
        row.split(",", 1)[0].strip() == values[1]
        for row in process_rows
    )
    return {
        "physical_index": int(values[0]),
        "uuid": values[1],
        "name": values[2],
        "driver_version": values[3],
        "memory_total_mib": int(values[4]),
        "memory_used_mib": int(values[5]),
        "utilization_percent": int(values[6]),
        "compute_process_count_before_probe": int(process_count),
    }


def _higher_p95(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot calculate p95 of an empty timing list")
    return float(np.quantile(values, 0.95, method="higher"))


def projected_seconds(
    *,
    p95_update_seconds: float,
    p95_sampling_seconds_per_entity: float,
    requested_updates: int,
    entity_count: int,
    fixed_overhead_seconds: float,
    safety_multiplier: float,
) -> float:
    return float(
        (
            p95_update_seconds * requested_updates
            + p95_sampling_seconds_per_entity * entity_count
            + fixed_overhead_seconds
        )
        * safety_multiplier
    )


def _lpt_wall_hours(cof_hours: float, gpu_count: int) -> float:
    jobs = [0.60] * 10 + [0.20] * 5 + [cof_hours] * 5
    loads = [0.0] * gpu_count
    for duration in sorted(jobs, reverse=True):
        target = min(range(gpu_count), key=loads.__getitem__)
        loads[target] += duration
    return max(loads)


def _report_text(result: Mapping[str, Any]) -> str:
    gpu = result["gpu"]
    lines = [
        "# CoF v2.5 timing-only capacity preflight",
        "",
        f"Status: **{result['status']}**.",
        "",
        "This is a timing/runtime artifact only. It is not a model-performance "
        "or sample-quality result and does not authorize the full experiment.",
        "",
        "## Frozen provenance",
        "",
        f"- source commit: `{result['source_commit']}`",
        f"- config SHA-256: `{result['config_hash']}`",
        f"- code SHA-256: `{result['code_hash']}`",
        f"- shared SamplingPlan SHA-256: `{result.get('sampling_plan_hash', 'UNAVAILABLE')}`",
        "- scenario/κ: `joint_semimarkov_v2b`, `1.0`",
        "- full runs: 0; five-seed runs: 0; sweeps: 0",
        "",
        "## Selected GPU",
        "",
        f"- physical index: `{gpu.get('physical_index')}`",
        f"- UUID: `{gpu.get('uuid')}`",
        f"- model: `{gpu.get('name')}`",
        f"- driver: `{gpu.get('driver_version')}`",
        f"- CUDA runtime reported by PyTorch: `{result.get('cuda_version')}`",
        f"- PyTorch: `{result.get('pytorch_version')}`",
        f"- pre-probe memory: `{gpu.get('memory_used_mib')}` / "
        f"`{gpu.get('memory_total_mib')}` MiB",
        f"- pre-probe utilization: `{gpu.get('utilization_percent')}`%",
        f"- pre-probe compute processes: `{gpu.get('compute_process_count_before_probe')}`",
        f"- selection reason: {result.get('selection_reason')}",
        "",
    ]
    if result["status"] == "UNAVAILABLE":
        lines.extend(
            [
                "## Decision",
                "",
                f"UNAVAILABLE: {result['reason']}",
                "",
                "No timing probe or full experiment was run.",
                "",
            ]
        )
        return "\n".join(lines)
    if result["status"] == "FAILED":
        lines.extend(
            [
                "## Decision",
                "",
                f"FAILED: {result['reason']}",
                "",
                "The cap and architecture remain unchanged. No full experiment "
                "is authorized.",
                "",
            ]
        )
        return "\n".join(lines)
    lines.extend(
        [
            "## Timing measurements",
            "",
            f"- updates: 500 total = 20 excluded warm-up + 480 measured",
            f"- mean train update: `{result['mean_train_update_seconds']:.6f}` s",
            f"- p95 train update: `{result['p95_train_update_seconds']:.6f}` s",
            "- sampling: 1 excluded warm-up chunk + 5 measured chunks",
            "- sampling chunk size: `256` entities",
            f"- p95 sampling time/entity: "
            f"`{result['p95_sampling_seconds_per_entity']:.8f}` s",
            f"- peak allocated GPU memory: "
            f"`{result['peak_gpu_memory_mib']:.2f}` MiB",
            f"- minimum required memory headroom: "
            f"`{result['minimum_memory_headroom_fraction']:.0%}`",
            "- CUDA OOM count: `0`; NaN count: `0`; repeated error count: `0`",
            "",
            "No durable checkpoint, generated sample, or quality metric was "
            "created. Generated tensors were discarded in memory.",
            "",
            "## Fixed projection",
            "",
            "```text",
            "(",
            "  p95_train_update_seconds * 20000",
            "  + p95_sampling_seconds_per_entity * 7989",
            "  + 600",
            ") * 1.15",
            "```",
            "",
            f"- projected seconds/CoF seed: `{result['projected_seconds']:.2f}`",
            f"- projected GPU-hours/CoF seed: `{result['cof_gpu_hours_per_seed']:.4f}`",
            f"- two-hour decision: **{result['decision']}**",
            "",
            "## Scheduling projection",
            "",
            "| GPUs | Expected learned-model wall time |",
            "|---:|---:|",
        ]
    )
    for gpu_count in (1, 2, 3):
        lines.append(
            f"| {gpu_count} | "
            f"{result['expected_wall_hours'][gpu_count]:.3f} h |"
        )
    lines.extend(
        [
            "",
            "The scheduling table retains the preregistered non-CoF planning "
            "inputs (CTGAN 0.60, TVAE 0.60, neural 0.20 GPU-hours per seed) "
            "and replaces only the superseded CoF estimate with this timing "
            "projection.",
            "",
            "## Stop boundary",
            "",
            "No association-recovery, fidelity, TVD, coherence, row-guard, or "
            "sample-quality evaluation was run. The full model run, five-seed "
            "run, sweep, and FULL_EXPERIMENT remain unauthorized and unstarted.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(_report_text(result))


def run_preflight(
    *,
    repository_root: Path,
    config_path: Path,
    report_path: Path,
    expected_commit: str,
    physical_gpu_index: int,
) -> Mapping[str, Any]:
    if report_path != DEFAULT_REPORT:
        raise ValueError("capacity report path is frozen")
    head = _git_head(repository_root)
    if head != expected_commit:
        raise RuntimeError(
            f"source commit mismatch: expected {expected_commit}, got {head}"
        )
    resolved_config = repository_root / config_path
    raw = yaml.safe_load(resolved_config.read_text())
    validate_config(raw)
    config_hash = _sha256(resolved_config)
    code_hash = hash_code(repository_root)
    preflight = raw["capacity_preflight"]
    gpu: Mapping[str, Any] = {
        "physical_index": physical_gpu_index,
        "uuid": None,
        "name": None,
        "driver_version": None,
        "memory_total_mib": None,
        "memory_used_mib": None,
        "utilization_percent": None,
        "compute_process_count_before_probe": None,
    }
    base = {
        "source_commit": head,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "gpu": gpu,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "selection_reason": "pending idle-GPU verification",
    }
    try:
        try:
            gpu = _nvidia_gpu(physical_gpu_index)
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            result = {
                **base,
                "status": "UNAVAILABLE",
                "reason": f"nvidia-smi unavailable: {error}",
                "selection_reason": "GPU availability could not be verified",
            }
            _write_report(repository_root / report_path, result)
            return result
        base["gpu"] = gpu
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        unavailable_reasons = []
        if visible != str(physical_gpu_index):
            unavailable_reasons.append(
                "CUDA_VISIBLE_DEVICES must expose exactly the selected physical GPU"
            )
        if gpu["compute_process_count_before_probe"]:
            unavailable_reasons.append("selected GPU has a compute process")
        if gpu["memory_used_mib"] > int(
            preflight["idle_max_memory_used_mib"]
        ):
            unavailable_reasons.append("selected GPU exceeds idle memory limit")
        if gpu["utilization_percent"] > int(
            preflight["idle_max_utilization_percent"]
        ):
            unavailable_reasons.append(
                "selected GPU exceeds idle utilization limit"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            unavailable_reasons.append(
                "PyTorch CUDA is unavailable or more than one GPU is visible"
            )
        if unavailable_reasons:
            result = {
                **base,
                "status": "UNAVAILABLE",
                "reason": "; ".join(unavailable_reasons),
                "selection_reason": "idle-GPU contract was not satisfied",
            }
            _write_report(repository_root / report_path, result)
            return result
        base["selection_reason"] = (
            "no compute process and preregistered idle memory/utilization "
            "limits satisfied; only this GPU was exposed"
        )

        benchmark_config = BenchmarkConfig.from_mapping(
            raw,
            "joint_semimarkov_v2b",
            1.0,
        )
        bundle = generate_benchmark(
            benchmark_config,
            int(raw["data"]["dataset_seed"]),
        )
        shared_plan = SamplingPlan.from_train_policy(
            bundle.train,
            entity_count=int(raw["sampling_plan"]["entity_count"]),
            seed=int(raw["sampling_plan"]["seed"]),
        )
        base["sampling_plan_hash"] = shared_plan.plan_hash

        model_config = dict(raw["baselines"]["cof_seqgen"])
        optimizer = model_config.pop("optimizer")
        model_config.update(
            {
                "device": "cuda:0",
                "lr": float(optimizer["lr"]),
                "weight_decay": float(optimizer["weight_decay"]),
                "requested_steps": int(preflight["max_updates"]),
                "checkpoint_interval_steps": int(
                    preflight["max_updates"]
                ),
                "tau": bundle.metadata["tau"],
                "window_width": float(raw["data"]["window_width"]),
                "temperature": float(raw["data"]["soft_g_temperature"]),
            }
        )
        update_times: list[float] = []
        model_config["update_timing_callback"] = (
            lambda event: update_times.append(
                float(event["update_seconds"])
            )
        )
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        adapter = CoFSeqGenAdapter()
        adapter.fit(bundle.train, config=model_config, seed=1)
        warmup_updates = int(preflight["warmup_updates_excluded"])
        measured_updates = int(preflight["measured_updates"])
        if len(update_times) != warmup_updates + measured_updates:
            raise RuntimeError(
                f"expected 500 timed update events, got {len(update_times)}"
            )
        measured_train = update_times[warmup_updates:]

        chunk_size = int(preflight["sampling_chunk_size"])
        measured_chunks = int(preflight["sampling_measured_chunks"])
        warmup_chunks = int(preflight["sampling_warmup_chunks_excluded"])
        if model_config["sampling_chunk_size"] != chunk_size:
            raise RuntimeError("probe and full sampling chunk size differ")
        chunk_times: list[float] = []
        total_chunks = warmup_chunks + measured_chunks
        for chunk_index in range(total_chunks):
            start = chunk_index * chunk_size
            stop = start + chunk_size
            chunk = SamplingPlan(
                shared_plan.y_entity[start:stop],
                shared_plan.lengths[start:stop],
                shared_plan.valid_mask[start:stop],
                shared_plan.plan_hash,
            )
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            generated = adapter.sample(
                chunk,
                seed=1_000_000 + chunk_index,
            )
            torch.cuda.synchronize(0)
            elapsed = time.perf_counter() - started
            if not np.isfinite(generated.x_num).all():
                raise RuntimeError("non-finite generated tensor in timing path")
            if chunk_index >= warmup_chunks:
                chunk_times.append(elapsed)
            del generated
        p95_update = _higher_p95(measured_train)
        p95_sampling_per_entity = (
            _higher_p95(chunk_times) / chunk_size
        )
        peak_bytes = int(torch.cuda.max_memory_allocated(0))
        total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
        minimum_headroom = float(
            preflight["minimum_gpu_memory_headroom_fraction"]
        )
        memory_safe = peak_bytes <= total_bytes * (1 - minimum_headroom)
        projection = projected_seconds(
            p95_update_seconds=p95_update,
            p95_sampling_seconds_per_entity=p95_sampling_per_entity,
            requested_updates=int(preflight["projection_requested_updates"]),
            entity_count=int(preflight["projection_entity_count"]),
            fixed_overhead_seconds=float(
                preflight["fixed_overhead_seconds"]
            ),
            safety_multiplier=float(
                preflight["projection_safety_multiplier"]
            ),
        )
        meaningful = (
            projection <= float(preflight["maximum_projected_seconds"])
            and memory_safe
        )
        cof_hours = projection / 3600
        result = {
            **base,
            "status": "COMPLETE",
            "mean_train_update_seconds": float(np.mean(measured_train)),
            "p95_train_update_seconds": p95_update,
            "p95_sampling_seconds_per_entity": p95_sampling_per_entity,
            "peak_gpu_memory_mib": peak_bytes / (1024**2),
            "minimum_memory_headroom_fraction": minimum_headroom,
            "projected_seconds": projection,
            "cof_gpu_hours_per_seed": cof_hours,
            "decision": (
                "PASS — meaningful configuration within the 2-hour cap"
                if meaningful
                else "FAIL — not authorized for a full run"
            ),
            "expected_wall_hours": {
                count: _lpt_wall_hours(cof_hours, count)
                for count in (1, 2, 3)
            },
        }
        _write_report(repository_root / report_path, result)
        return result
    except Exception as error:
        report = repository_root / report_path
        if not report.exists():
            result = {
                **base,
                "status": "FAILED",
                "reason": f"{type(error).__name__}: {error}",
            }
            _write_report(report, result)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered CoF v2.5 timing-only preflight.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_preflight(
        repository_root=Path(".").resolve(),
        config_path=args.config,
        report_path=args.report,
        expected_commit=args.expected_commit,
        physical_gpu_index=args.physical_gpu_index,
    )
    print(
        f"capacity preflight status={result['status']} "
        f"report={args.report}"
    )


if __name__ == "__main__":
    main()
