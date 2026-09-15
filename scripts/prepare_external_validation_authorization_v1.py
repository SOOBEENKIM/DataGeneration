"""Prepare the append-only external validation v1 continuation authorization.

Plan and dry-run are manifest-only.  They do not load an NPZ, read a raw CSV,
import a model, query a GPU, or create an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

from scripts.run_external_validation_v1 import (
    ExternalValidationError,
    build_external_validation_batch_authorization,
    validate_external_launch_schedule,
    validate_external_validation_batch_authorization,
    write_external_validation_authorization,
)


_LAUNCH_DATASETS = frozenset({"amlsim", "sparkov"})
_LAUNCH_MODELS = frozenset(
    {
        "empirical_iid",
        "ctgan_separate_class",
        "tvae_separate_class",
        "cof_seqgen_frozen_non_v3",
    }
)


def build_external_validation_launch_argv(
    *,
    python_executable: str,
    authorization_path: Path,
    dataset: str,
    model: str,
    device: str,
) -> list[str]:
    """Build one exact execution argv without running it or reading data.

    Flags and values are deliberately separate list elements so shell-plan
    rendering cannot merge ``--model`` with the model identifier.
    """
    if dataset not in _LAUNCH_DATASETS:
        raise ExternalValidationError("launch dataset is not preregistered")
    if model not in _LAUNCH_MODELS:
        raise ExternalValidationError("launch model is not preregistered")
    if device not in {"cpu", "cuda:0"}:
        raise ExternalValidationError("launch device must be cpu or cuda:0")
    return [
        python_executable,
        "-m",
        "scripts.run_external_validation_v1",
        "--repo-root",
        ".",
        "--config",
        "configs/benchmark_v2/external_validation_v1.yaml",
        "--mode",
        "execute",
        "--authorization",
        str(authorization_path),
        "--dataset",
        dataset,
        "--model",
        model,
        "--device",
        device,
    ]


def build_external_validation_wave_launch_plan(
    *,
    python_executable: str,
    authorization_path: Path,
    config_path: Path,
) -> Mapping[str, Any]:
    """Build the exact parallel wave argv plan without executing any job."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    schedule = validate_external_launch_schedule(config)
    env_by_slot = {
        "physical_gpu_a": "EXTV1_GPU_A",
        "physical_gpu_b": "EXTV1_GPU_B",
        "physical_gpu_c": "EXTV1_GPU_C",
    }
    waves = []
    for wave in schedule["waves"]:
        jobs = []
        for job in wave["jobs"]:
            jobs.append(
                {
                    **job,
                    "cuda_visible_devices_env": env_by_slot.get(
                        job["device_slot"]
                    ),
                    "argv": build_external_validation_launch_argv(
                        python_executable=python_executable,
                        authorization_path=authorization_path,
                        dataset=job["dataset"],
                        model=job["model"],
                        device=job["runner_device"],
                    ),
                }
            )
        waves.append({**wave, "jobs": jobs})
    return {
        "schema_version": schedule["schema_version"],
        "scheduling_only": True,
        "waves": waves,
        "execution_counts": {
            "gpu_inventory_queries": 0,
            "cuda_calls": 0,
            "model_fit_calls": 0,
            "model_sample_calls": 0,
            "validation_metric_calls": 0,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_authorization_preparation_plan(
    *,
    repo_root: Path,
    config_path: Path,
    approval_text: str,
    mode: str,
) -> Mapping[str, Any]:
    if mode not in {"plan", "dry-run"}:
        raise ExternalValidationError("authorization preparation mode must be plan or dry-run")
    authorization = build_external_validation_batch_authorization(
        repo_root=repo_root,
        config_path=config_path,
        approval_text=approval_text,
    )
    validated = validate_external_validation_batch_authorization(
        repo_root=repo_root,
        config_path=config_path,
        authorization=authorization,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    launch_plan = build_external_validation_wave_launch_plan(
        python_executable=sys.executable,
        authorization_path=(
            repo_root
            / config["continuation"]["authorization_root"]
            / "authorization_<source-head>_attempt_001.json"
        ),
        config_path=config_path,
    )
    return {
        "schema_version": "external-validation-authorization-preparation-v1",
        "mode": mode,
        "status": "PASS",
        "job_count": validated["job_count"],
        "authorization_sha256_if_written": validated["authorization_sha256"],
        "authorization_root": str(
            repo_root / config["continuation"]["authorization_root"]
        ),
        "launch_schedule": launch_plan,
        "jobs": [
            {
                "dataset": job["dataset"],
                "model": job["model"],
                "seed": job["seed"],
                "attempt": job["attempt"],
                "device_class": job["device_class"],
                "wave_id": job["wave_id"],
                "device_slot": job["device_slot"],
                "runner_device": job["runner_device"],
                "cuda_visible_devices_count": job[
                    "cuda_visible_devices_count"
                ],
                "data_transformer_n_jobs": job[
                    "data_transformer_n_jobs"
                ],
                "heavy_transform_exclusion_group": job[
                    "heavy_transform_exclusion_group"
                ],
                "requested_steps": job["requested_steps"],
                "training_hard_cap_seconds": job[
                    "training_hard_cap_seconds"
                ],
                "job_hard_cap_seconds": job["job_hard_cap_seconds"],
                "expected_gpu_hours_upper": job[
                    "expected_gpu_hours_upper"
                ],
            }
            for job in authorization["jobs"]
        ],
        "access_counts": {
            "frozen_npz_body_reads": 0,
            "raw_csv_reads": 0,
            "model_imports": 0,
            "gpu_inventory_queries": 0,
        },
        "execution_counts": {
            "model_fit_calls": 0,
            "model_sample_calls": 0,
            "validation_metric_calls": 0,
        },
        "memory_safety": authorization["memory_safety"],
        "completed_result_reuse": authorization["completed_result_reuse"],
        "authorization_created": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "dry-run", "create"), required=True)
    parser.add_argument("--approval-text", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config.resolve()
    if args.mode in {"plan", "dry-run"}:
        if args.output is not None:
            raise ExternalValidationError("plan/dry-run cannot create authorization output")
        result = build_authorization_preparation_plan(
            repo_root=repo_root,
            config_path=config_path,
            approval_text=args.approval_text,
            mode=args.mode,
        )
    else:
        if args.output is None:
            raise ExternalValidationError("create requires an append-only output path")
        authorization = build_external_validation_batch_authorization(
            repo_root=repo_root,
            config_path=config_path,
            approval_text=args.approval_text,
        )
        validated = validate_external_validation_batch_authorization(
            repo_root=repo_root,
            config_path=config_path,
            authorization=authorization,
        )
        output = args.output.resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected_root = (
            repo_root / config["continuation"]["authorization_root"]
        ).resolve()
        if output.parent != expected_root:
            raise ExternalValidationError(
                "continuation authorization output is outside its append-only root"
            )
        write_external_validation_authorization(output, authorization)
        observed = _sha256_file(output)
        if observed != validated["authorization_sha256"]:
            raise ExternalValidationError("written authorization hash mismatch")
        result = {
            "status": "CREATED",
            "path": str(output),
            "authorization_sha256": observed,
            "job_count": 8,
            "execution_counts": {
                "model_fit_calls": 0,
                "model_sample_calls": 0,
                "validation_metric_calls": 0,
                "gpu_inventory_queries": 0,
            },
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
