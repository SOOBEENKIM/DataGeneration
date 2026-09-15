from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import yaml

from benchmarks.temporal_coupling_v2 import (
    BenchmarkConfig,
    generate_benchmark,
    generate_fixed_binning_split,
)
from benchmarks.types import SequenceBatch
from eval.model_guards_v2_5 import calibrate_row_guard_thresholds
from experiments.frozen_metadata_v2_5 import encode_frozen_metadata
from experiments.provenance_v2_5 import hash_batch
from generators.sampling_plan import SamplingPlan
from scripts.validate_full_experiment_v2_5_preparation import validate_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _exclusive_batch(path: Path, batch: SequenceBatch) -> None:
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            **{
                field: getattr(batch, field)
                for field in SequenceBatch.__dataclass_fields__
            },
        )
        # numpy adds .npz when the supplied filename lacks that suffix.
        written = (
            temporary
            if temporary.exists() and temporary.stat().st_size
            else Path(f"{temporary}.npz")
        )
        if not written.is_file():
            raise RuntimeError("npz writer produced no file")
        os.replace(written, path)
    finally:
        temporary.unlink(missing_ok=True)
        Path(f"{temporary}.npz").unlink(missing_ok=True)


def _exclusive_sampling_plan(path: Path, plan: SamplingPlan) -> None:
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        plan.save(str(temporary))
        written = (
            temporary
            if temporary.exists() and temporary.stat().st_size
            else Path(f"{temporary}.npz")
        )
        if not written.is_file():
            raise RuntimeError("SamplingPlan writer produced no file")
        os.replace(written, path)
    finally:
        temporary.unlink(missing_ok=True)
        Path(f"{temporary}.npz").unlink(missing_ok=True)


def prepare_data(
    raw: Mapping[str, Any],
    *,
    config_path: Path,
    data_root: Path,
    artifact_root: Path,
) -> Mapping[str, Any]:
    validate_config(raw)
    if data_root.name != "benchmark_v2_5":
        raise ValueError("data root must end in benchmark_v2_5")
    if artifact_root.name != "benchmark_v2_5":
        raise ValueError("artifact root must end in benchmark_v2_5")
    scenario, kappa = "joint_semimarkov_v2b", 1.0
    destination = (
        data_root / "frozen" / scenario / f"kappa_{kappa:.2f}"
    )
    calibration = artifact_root / "prerun_calibration"
    if destination.exists() or calibration.exists():
        raise FileExistsError(
            "v2.5 data/calibration output already exists; never overwrite it"
        )
    destination.mkdir(parents=True)
    calibration.mkdir(parents=True)
    config = BenchmarkConfig.from_mapping(raw, scenario, kappa)
    bundle = generate_benchmark(config, int(raw["data"]["dataset_seed"]))
    fixed = replace(
        config,
        bin_edges=np.asarray(bundle.metadata["bin_edges"]),
    )
    validation, validation_latent = generate_fixed_binning_split(
        fixed,
        n_entities=int(raw["data"]["n_validation"]),
        seed=int(raw["data"]["validation_seed"]),
        split_id=int(raw["data"]["validation_split_id"]),
        prefix="validation",
    )
    paths = {
        "train": destination / "train.npz",
        "validation": destination / "validation.npz",
        "test": destination / "test.npz",
    }
    _exclusive_batch(paths["train"], bundle.train)
    _exclusive_batch(paths["validation"], validation)
    _exclusive_batch(paths["test"], bundle.test)
    plan_config = raw["sampling_plan"]
    sampling_plan = SamplingPlan.from_train_policy(
        bundle.train,
        entity_count=int(plan_config["entity_count"]),
        seed=int(plan_config["seed"]),
    )
    sampling_plan_path = destination / "shared_sampling_plan.npz"
    _exclusive_sampling_plan(sampling_plan_path, sampling_plan)
    tau = np.asarray(bundle.metadata["tau"])
    guard = raw["evaluation"]["row_marginal_guard_calibration"]
    thresholds = calibrate_row_guard_thresholds(
        bundle.train,
        tau=tau,
        entity_count=int(guard["entity_count_per_replicate"]),
        trials=int(guard["trials"]),
        seed=int(guard["seed"]),
        receiver_practical_margin=float(
            guard["receiver"]["practical_margin"]
        ),
    )
    threshold_path = calibration / "row_guard_thresholds.json"
    _exclusive_json(
        threshold_path,
        {
            "schema_version": "benchmark-v2.5-row-guard-calibration",
            "status": "COMPLETE",
            "learned_results_used": False,
            "thresholds": thresholds.to_dict(),
            "train_hash": hash_batch(bundle.train),
            "config_sha256": _sha256(config_path),
        },
    )
    metadata_path = destination / "meta.json"
    _exclusive_json(
        metadata_path,
        encode_frozen_metadata({
            "schema_version": "benchmark-v2.5-frozen-data",
            "scenario": scenario,
            "kappa": kappa,
            "dataset_seed": int(raw["data"]["dataset_seed"]),
            "validation_seed": int(raw["data"]["validation_seed"]),
            "validation_split_id": int(
                raw["data"]["validation_split_id"]
            ),
            "bin_edges": bundle.metadata["bin_edges"],
            "tau": bundle.metadata["tau"],
            "binning_fit_split": "train",
            "validation_seed_provenance": validation_latent["metadata"],
            "test_usage": "evaluation_only",
        }),
    )
    manifest = {
        "schema_version": "benchmark-v2.5-frozen-data-manifest",
        "status": "COMPLETE",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "scenario": scenario,
        "kappa": kappa,
        "data_hashes": {
            "train": hash_batch(bundle.train),
            "validation": hash_batch(validation),
            "test": hash_batch(bundle.test),
        },
        "sampling_plan": {
            "path": str(sampling_plan_path),
            "sha256": sampling_plan.plan_hash,
            "file_sha256": _sha256(sampling_plan_path),
            "fit_split": "train",
            "seed": int(plan_config["seed"]),
            "shared_by_all_generators_and_model_seeds": True,
        },
        "file_sha256": {
            key: _sha256(path) for key, path in paths.items()
        },
        "metadata_sha256": _sha256(metadata_path),
        "row_guard_thresholds_path": str(threshold_path),
        "row_guard_thresholds_sha256": _sha256(threshold_path),
        "learned_model_run_started": False,
    }
    _exclusive_json(destination / "data_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare frozen CPU-only v2.5 data and guard references.",
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark_v2/full_v2_5.yaml",
    )
    parser.add_argument(
        "--data-root",
        default="data/benchmark_v2_5",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/benchmark_v2_5",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        required=True,
        help="Required acknowledgement; this script never trains a model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    raw = yaml.safe_load(config_path.read_text())
    result = prepare_data(
        raw,
        config_path=config_path,
        data_root=Path(args.data_root),
        artifact_root=Path(args.artifact_root),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
