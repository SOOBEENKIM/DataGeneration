"""Plan/dry-run validator for the source-only external sequence protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml


class ExternalPlanError(RuntimeError):
    """Raised when frozen external protocol provenance does not match."""


ZERO_EXECUTION_COUNTS = {
    "gpu_inventory_queries": 0,
    "cuda_calls": 0,
    "model_fit_calls": 0,
    "model_sample_calls": 0,
    "transform_fit_calls": 0,
    "window_materialization_calls": 0,
    "external_test_execution_calls": 0,
    "authorization_creations": 0,
    "launch_command_creations": 0,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha256(repo_root: Path, commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _load_config(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ExternalPlanError("external protocol config root must be a mapping")
    return value


def _validate_frozen_baseline(repo_root: Path, config: Mapping[str, Any]) -> None:
    frozen = config["frozen_baseline"]
    expected_commit = "99a445f6dc893a8c2240d950de4f92877cc07f8a"
    expected_config = "81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3"
    if frozen["source_commit"] != expected_commit:
        raise ExternalPlanError("frozen non-v3 CoF source commit mismatch")
    if frozen["config_sha256"] != expected_config:
        raise ExternalPlanError("frozen non-v3 CoF config hash mismatch")
    for kind in ("model", "adapter", "config"):
        observed = _git_blob_sha256(
            repo_root,
            expected_commit,
            frozen[f"{kind}_path"],
        )
        if observed != frozen[f"{kind}_sha256"]:
            raise ExternalPlanError(f"frozen baseline {kind} hash mismatch")
    if _sha256_file(repo_root / frozen["config_path"]) != expected_config:
        raise ExternalPlanError("current frozen config bytes changed")
    if frozen["architecture_changes_allowed"] is not False:
        raise ExternalPlanError("external baseline architecture must remain frozen")


def _validate_protocol(config: Mapping[str, Any]) -> None:
    window = config["window"]
    if (
        window["length"] != 32
        or window["stride"] != 32
        or window["minimum_tail"] != 16
        or window["shorter_tail"] != "drop"
        or window["padding"] != "right_zero_with_prefix_valid_mask"
        or window["label"] != "any_fraud_transaction_in_window"
    ):
        raise ExternalPlanError("external 32-event window contract mismatch")
    split = config["split"]
    if split["entity_fractions"] != {
        "train": 0.70,
        "validation": 0.15,
        "internal_test": 0.15,
    }:
        raise ExternalPlanError("external split fractions changed")
    if (
        split["method"]
        != "group_stratified_by_retained_window_any_fraud_per_entity"
        or split["seed"] != 20260801
        or split["leakage_policy"] != "fail_closed_entity_window_transaction"
    ):
        raise ExternalPlanError("external group-stratified split contract changed")
    if config["transforms"]["fit_split"] != "train":
        raise ExternalPlanError("external transforms must fit train only")
    transforms = config["transforms"]
    if (
        transforms["amount"]["operator"] != "log1p_then_population_zscore"
        or transforms["gap"]["bins"] != 16
        or transforms["receiver"]["pad_code"] != 0
        or transforms["receiver"]["unk_code"] != 1
    ):
        raise ExternalPlanError("external train-only transform contract changed")
    amlsim = config["datasets"]["amlsim"]
    if (
        amlsim["entity"] != "SENDER_ACCOUNT_ID"
        or amlsim["sort"] != ["TIMESTAMP", "TX_ID"]
        or amlsim["receiver"] != "RECEIVER_ACCOUNT_ID"
        or amlsim["fraud_label"] != "IS_FRAUD"
    ):
        raise ExternalPlanError("AMLSim external mapping changed")
    sparkov = config["datasets"]["sparkov"]
    if (
        sparkov["entity"] != "cc_num"
        or sparkov["sort"] != ["trans_date_trans_time", "source_row_order"]
        or sparkov["receiver"] != "merchant"
        or sparkov["fraud_label"] != "is_fraud"
    ):
        raise ExternalPlanError("Sparkov external mapping changed")
    if (
        sparkov["public_test"]["calibration_allowed"] is not False
        or sparkov["public_test"]["selection_allowed"] is not False
        or sparkov["public_test"]["role"] != "locked_temporal_robustness_only"
    ):
        raise ExternalPlanError("Sparkov public test must remain locked")
    if amlsim["gap_unit"] != "simulation_steps":
        raise ExternalPlanError("AMLSim gap unit must remain simulation steps")
    if sparkov["gap_unit"] != "seconds":
        raise ExternalPlanError("Sparkov gap unit must remain seconds")
    thresholds = config["external_thresholds"]
    if thresholds["controlled_benchmark_thresholds_copied"] is not False:
        raise ExternalPlanError("controlled benchmark thresholds cannot be copied")
    if thresholds["execution_enabled"] is not False:
        raise ExternalPlanError("external threshold bootstrap is not authorized")


def _verify_preservation(repo_root: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    verified: dict[str, Any] = {}
    for name, record in config["preservation"].items():
        observed = _sha256_file(repo_root / record["path"])
        if observed != record["sha256"]:
            raise ExternalPlanError(f"preserved artifact hash mismatch: {record['path']}")
        verified[name] = {"path": record["path"], "sha256": observed}
    return verified


def _verify_raw_source(
    source: Mapping[str, Any],
    required_columns: Sequence[str],
) -> Mapping[str, Any]:
    path = Path(source["path"])
    if not path.is_file():
        raise ExternalPlanError(f"raw source missing: {path}")
    observed_hash = _sha256_file(path)
    if observed_hash != source["sha256"]:
        raise ExternalPlanError(f"raw source hash mismatch: {path}")
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(required_columns) - set(columns))
    if missing:
        raise ExternalPlanError(f"raw schema is missing protocol columns: {missing}")
    return {
        "path": str(path),
        "sha256": observed_hash,
        "columns": columns,
        "required_columns": list(required_columns),
        "schema_contract": "PASS",
        "access": "read_only_hash_and_header",
    }


def build_external_protocol_plan(
    *,
    repo_root: Path,
    config_path: Path,
    mode: str,
) -> Mapping[str, Any]:
    if mode not in {"plan", "dry-run"}:
        raise ExternalPlanError("mode must be plan or dry-run")
    config = _load_config(config_path)
    _validate_frozen_baseline(repo_root, config)
    _validate_protocol(config)
    preservation = _verify_preservation(repo_root, config)
    sources: dict[str, Any] = {}
    raw_read_counts = {"raw_hash_reads": 0, "raw_header_reads": 0}
    if mode == "dry-run":
        amlsim = config["datasets"]["amlsim"]
        sparkov = config["datasets"]["sparkov"]
        amlsim_required = [
            amlsim[key]
            for key in ("entity", "timestamp", "transaction_id", "receiver", "amount", "fraud_label")
        ]
        sparkov_required = [
            sparkov[key]
            for key in ("entity", "timestamp", "transaction_id", "receiver", "amount", "fraud_label")
        ]
        for name, source, required in (
            ("amlsim_development", amlsim["development_source"], amlsim_required),
            ("sparkov_development", sparkov["development_source"], sparkov_required),
            ("sparkov_public_test_locked", sparkov["public_test"], sparkov_required),
        ):
            sources[name] = _verify_raw_source(source, required)
            raw_read_counts["raw_hash_reads"] += 1
            raw_read_counts["raw_header_reads"] += 1
    return {
        "schema_version": "external-sequence-protocol-plan-v1",
        "mode": mode,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "frozen_baseline": dict(config["frozen_baseline"]),
        "datasets": {
            "amlsim": {
                "entity": config["datasets"]["amlsim"]["entity"],
                "sort": config["datasets"]["amlsim"]["sort"],
                "receiver": config["datasets"]["amlsim"]["receiver"],
            },
            "sparkov": {
                "entity": config["datasets"]["sparkov"]["entity"],
                "sort": config["datasets"]["sparkov"]["sort"],
                "receiver": config["datasets"]["sparkov"]["receiver"],
                "public_test_role": config["datasets"]["sparkov"]["public_test"]["role"],
            },
        },
        "window": {
            key: config["window"][key]
            for key in ("length", "minimum_tail", "stride", "padding")
        },
        "split": dict(config["split"]),
        "transforms": dict(config["transforms"]),
        "external_thresholds": dict(config["external_thresholds"]),
        "preservation": preservation,
        "verified_raw_sources": sources,
        "raw_read_counts": raw_read_counts,
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),
        "runtime_artifacts_created": False,
        "execution_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "dry-run"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_external_protocol_plan(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        mode=args.mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
