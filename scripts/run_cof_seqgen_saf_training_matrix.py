#!/usr/bin/env python3
"""Run a reproducible SAF seed/candidate matrix with one process per GPU."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
TRAINING_SCRIPT = REPOSITORY_ROOT / "scripts" / "run_cof_seqgen_saf_training.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    matrix = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected_implementation = matrix.get("model_implementation_version")
    output_root = (REPOSITORY_ROOT / matrix["output_root"]).resolve()
    config_root = output_root / "resolved_configs"
    log_root = output_root / "logs"
    config_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    gpu_ids = [int(value) for value in matrix["gpu_ids"]]
    candidates = list(matrix["candidates"])
    seeds = [int(value) for value in matrix["seeds"]]
    base = dict(matrix["training"])
    jobs = []
    for seed in seeds:
        for candidate in candidates:
            job_id = f"seed_{seed}_{candidate}"
            jobs.append((job_id, seed, candidate))

    buckets = [[] for _ in gpu_ids]
    for index, job in enumerate(jobs):
        buckets[index % len(gpu_ids)].append(job)

    def worker(gpu_id: int, assigned: list[tuple[str, int, str]]) -> list[Dict[str, Any]]:
        records = []
        for job_id, seed, candidate in assigned:
            output_dir = output_root / f"seed_{seed}" / candidate
            resolved = {
                "dataset_dir": matrix["dataset_dir"],
                "output_dir": str(output_dir.relative_to(REPOSITORY_ROOT)),
                "candidate_id": candidate,
                "seed": seed,
                "device": f"cuda:{gpu_id}",
                **base,
            }
            config_path = config_root / f"{job_id}.yaml"
            config_path.write_text(
                yaml.safe_dump(resolved, sort_keys=False),
                encoding="utf-8",
            )
            process = subprocess.run(
                [str(PYTHON), str(TRAINING_SCRIPT), "--config", str(config_path)],
                cwd=REPOSITORY_ROOT,
                env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
                capture_output=True,
                text=True,
            )
            log_path = log_root / f"{job_id}.log"
            log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(f"{job_id} failed; see {log_path}")
            report_path = output_dir / "training_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                expected_implementation is not None
                and report.get("model_implementation_version")
                != expected_implementation
            ):
                raise RuntimeError(f"{job_id} implementation version mismatch")
            records.append(
                {
                    "job_id": job_id,
                    "gpu_id": gpu_id,
                    "candidate_id": candidate,
                    "seed": seed,
                    "model_implementation_version": report.get(
                        "model_implementation_version"
                    ),
                    "best_epoch": report["best_epoch"],
                    "best_validation_loss": report["best_validation_loss"],
                    "checkpoint_sha256": report["checkpoint_sha256"],
                    "resolved_config_sha256": _sha256(config_path),
                    "test_accessed": report["test_accessed"],
                }
            )
        return records

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [
            executor.submit(worker, gpu_id, assigned)
            for gpu_id, assigned in zip(gpu_ids, buckets)
        ]
        records = [record for future in futures for record in future.result()]
    records.sort(key=lambda item: (item["seed"], item["candidate_id"]))
    manifest = {
        "schema_version": "cof-seqgen-saf-training-matrix-v1",
        "model_implementation_version": expected_implementation,
        "source_config": str(args.config.resolve()),
        "dataset_dir": matrix["dataset_dir"],
        "gpu_ids": gpu_ids,
        "candidates": candidates,
        "seeds": seeds,
        "records": records,
        "all_test_accessed_false": all(not record["test_accessed"] for record in records),
    }
    manifest_path = output_root / "training_matrix_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "jobs": len(records),
                "manifest_sha256": _sha256(manifest_path),
                "all_test_accessed_false": manifest["all_test_accessed_false"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
