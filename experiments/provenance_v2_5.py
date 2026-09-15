from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np

from benchmarks.types import SequenceBatch
from generators.sampling_plan import SamplingPlan
from .full_artifact_store_v2_5 import SCHEMA_VERSION, validate_manifest


DEFAULT_CODE_PATHS = (
    Path("benchmarks"),
    Path("eval"),
    Path("experiments"),
    Path("generators"),
    Path("models"),
    Path("scripts"),
)


def hash_batch(batch: SequenceBatch) -> str:
    digest = hashlib.sha256()
    for field in (
        "x_num",
        "dt_bin",
        "x_cat",
        "valid_mask",
        "y_entity",
        "lengths",
        "entity_ids",
    ):
        value = np.asarray(getattr(batch, field))
        if value.dtype.kind in ("O", "U"):
            digest.update(
                "\0".join(str(item) for item in value.tolist()).encode()
            )
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(field.encode())
    return digest.hexdigest()


def hash_code(
    repository_root: Path,
    paths: Iterable[Path] = DEFAULT_CODE_PATHS,
) -> str:
    files: list[Path] = []
    for relative in paths:
        path = repository_root / relative
        if path.is_dir():
            files.extend(path.rglob("*.py"))
        elif path.is_file():
            files.append(path)
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(repository_root)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()


def build_manifest(
    *,
    repository_root: Path,
    config_path: Path,
    scenario: str,
    kappa: float,
    generator: str,
    seed: int,
    train: SequenceBatch,
    validation: SequenceBatch,
    test: SequenceBatch,
    sampling_plan: SamplingPlan,
    gpu: Mapping[str, Any],
    cuda_version: str | None,
    pytorch_version: str,
    requested_training_budget: Mapping[str, Any],
) -> Mapping[str, Any]:
    config_bytes = (repository_root / config_path).read_bytes()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(repository_root),
        "config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "code_hash": hash_code(repository_root),
        "scenario": scenario,
        "kappa": float(kappa),
        "generator": generator,
        "seed": int(seed),
        "sampling_plan_hash": sampling_plan.plan_hash,
        "data_hashes": {
            "train": hash_batch(train),
            "validation": hash_batch(validation),
            "test": hash_batch(test),
        },
        "gpu": dict(gpu),
        "cuda_version": cuda_version,
        "pytorch_version": pytorch_version,
        "requested_training_budget": dict(requested_training_budget),
        "actual_training_budget": {
            "steps": 0,
            "wall_seconds": 0.0,
        },
        "baseline_definition_version": "benchmark-v2.5",
        "evaluation_version": "benchmark-v2.5-evaluation-v1",
    }
    validate_manifest(manifest)
    return manifest
