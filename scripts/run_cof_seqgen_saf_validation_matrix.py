#!/usr/bin/env python3
"""Generate and validate a completed SAF seed matrix across available GPUs."""

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
VALIDATION_SCRIPT = REPOSITORY_ROOT / "scripts" / "validate_cof_seqgen_saf.py"


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
    runs = [(int(seed), Path(root)) for seed, root in matrix["checkpoint_roots"].items()]
    buckets = [[] for _ in gpu_ids]
    for index, run in enumerate(sorted(runs)):
        buckets[index % len(gpu_ids)].append(run)

    def worker(gpu_id: int, assigned: list[tuple[int, Path]]) -> list[Dict[str, Any]]:
        records = []
        for seed, checkpoint_root in assigned:
            output_dir = output_root / f"seed_{seed}"
            checkpoints = {
                candidate: str(
                    checkpoint_root / candidate / "checkpoint_best.pt"
                )
                for candidate in candidates
            }
            resolved = {
                "dataset_dir": matrix["dataset_dir"],
                "output_dir": str(output_dir.relative_to(REPOSITORY_ROOT)),
                "checkpoints": checkpoints,
                "device": f"cuda:{gpu_id}",
                "generation_seed": int(matrix["generation_seed"]),
                "sample_batch_size": int(matrix["sample_batch_size"]),
                "generation_entities": matrix.get("generation_entities"),
                "metric_audit_path": matrix["metric_audit_path"],
            }
            config_path = config_root / f"seed_{seed}.yaml"
            config_path.write_text(
                yaml.safe_dump(resolved, sort_keys=False),
                encoding="utf-8",
            )
            process = subprocess.run(
                [str(PYTHON), str(VALIDATION_SCRIPT), "--config", str(config_path)],
                cwd=REPOSITORY_ROOT,
                env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
                capture_output=True,
                text=True,
            )
            log_path = log_root / f"seed_{seed}.log"
            log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(f"validation seed {seed} failed; see {log_path}")
            report_path = output_dir / "validation_comparison.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                expected_implementation is not None
                and report.get("model_implementation_version")
                != expected_implementation
            ):
                raise RuntimeError(f"validation seed {seed} implementation mismatch")
            records.append(
                {
                    "seed": seed,
                    "gpu_id": gpu_id,
                    "model_implementation_version": report.get(
                        "model_implementation_version"
                    ),
                    "report_path": str(report_path),
                    "report_sha256": _sha256(report_path),
                    "shared_generation_plan_sha256": report[
                        "shared_generation_plan_sha256"
                    ],
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
    records.sort(key=lambda item: item["seed"])
    manifest = {
        "schema_version": "cof-seqgen-saf-validation-matrix-v1",
        "model_implementation_version": expected_implementation,
        "source_config": str(args.config.resolve()),
        "records": records,
        "all_test_accessed_false": all(not record["test_accessed"] for record in records),
    }
    manifest_path = output_root / "validation_matrix_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "seeds": len(records),
                "manifest_sha256": _sha256(manifest_path),
                "all_test_accessed_false": manifest["all_test_accessed_false"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
