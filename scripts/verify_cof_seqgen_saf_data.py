"""Verify acquired sources and every persisted CoFSeqGen-SAF canonical file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


class SAFDataVerificationError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_file(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file():
        raise SAFDataVerificationError(f"missing file: {path}")
    if path.stat().st_size != int(expected["bytes"]):
        raise SAFDataVerificationError(f"byte-size mismatch: {path}")
    if sha256_file(path) != expected["sha256"]:
        raise SAFDataVerificationError(f"SHA-256 mismatch: {path}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify(repository_root: Path) -> dict[str, Any]:
    acquisition_config_path = (
        repository_root / "configs/benchmark_v2/cof_seqgen_saf_acquisition.yaml"
    )
    dataset_config_path = (
        repository_root / "configs/benchmark_v2/cof_seqgen_saf_datasets.yaml"
    )
    acquisition_config = yaml.safe_load(
        acquisition_config_path.read_text(encoding="utf-8")
    )
    dataset_config = yaml.safe_load(dataset_config_path.read_text(encoding="utf-8"))
    runtime_root = repository_root / dataset_config["runtime_root"]
    acquisition_summary_path = (
        runtime_root / "manifests/acquisition/acquisition_summary.json"
    )
    materialization_summary_path = runtime_root / "manifests/materialization_summary.json"
    acquisition = json.loads(acquisition_summary_path.read_text(encoding="utf-8"))
    materialization = json.loads(
        materialization_summary_path.read_text(encoding="utf-8")
    )
    if acquisition["config_sha256"] != sha256_file(acquisition_config_path):
        raise SAFDataVerificationError("acquisition config hash is stale")
    if materialization["config_sha256"] != sha256_file(dataset_config_path):
        raise SAFDataVerificationError("dataset materialization config hash is stale")

    acquisition_checks: dict[str, Any] = {}
    raw_root = repository_root / acquisition_config["raw_root"]
    for dataset, manifest in sorted(acquisition["datasets"].items()):
        status = manifest["status"]
        if status == "BLOCKED":
            acquisition_checks[dataset] = {"status": status, "verified_files": 0}
            continue
        files = manifest.get("files", [])
        for item in files:
            if status == "REGISTERED_LOCAL_GENERATOR":
                path = repository_root / item["name"]
            else:
                path = raw_root / dataset / item["name"]
            _assert_file(path, item)
            if status == "ACQUIRED" and path.stat().st_mode & 0o222:
                raise SAFDataVerificationError(f"raw source is writable: {path}")
        acquisition_checks[dataset] = {
            "status": status,
            "verified_files": len(files),
        }

    canonical_checks: dict[str, Any] = {}
    canonical_root = runtime_root / "canonical"
    for dataset in materialization["datasets_materialized"]:
        dataset_root = canonical_root / dataset
        manifest_path = dataset_root / "canonical_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "canonical_manifest.json" in manifest["files"]:
            raise SAFDataVerificationError(f"self-referential manifest: {dataset}")
        for relative, expected in manifest["files"].items():
            _assert_file(dataset_root / relative, expected)
        report = json.loads(
            (dataset_root / "dataset_report.json").read_text(encoding="utf-8")
        )
        failed_gates = sorted(key for key, value in report["gates"].items() if not value)
        if failed_gates:
            raise SAFDataVerificationError(
                f"{dataset} has failed gates: {failed_gates}"
            )
        canonical_checks[dataset] = {
            "canonical_manifest_sha256": sha256_file(manifest_path),
            "verified_files": len(manifest["files"]),
            "entity_count": report["sequence_report"]["entity_count"],
            "event_count": report["sequence_report"]["event_count"],
            "split_assignment_sha256": report["sequence_report"][
                "split_assignment_sha256"
            ],
            "failed_gates": failed_gates,
        }
    if not materialization["all_materialized_gates_pass"]:
        raise SAFDataVerificationError("global materialization gate is false")
    result = {
        "schema_version": "cof-seqgen-saf-data-verification-v1",
        "all_checks_pass": True,
        "acquisition_summary_sha256": sha256_file(acquisition_summary_path),
        "materialization_summary_sha256": sha256_file(materialization_summary_path),
        "acquisition": acquisition_checks,
        "canonical": canonical_checks,
    }
    _write_json(runtime_root / "manifests/data_verification_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = verify(args.repo_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
