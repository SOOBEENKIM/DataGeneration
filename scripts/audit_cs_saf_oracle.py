"""Run the frozen CS-SAF train-only oracle audit; no model fitting or GPU use."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.cs_saf_oracle import SemiMarkovCopyOracle, decide_oracle, summarize_context
from benchmarks.temporal_coupling_v2 import BenchmarkConfig
from data.cof_seqgen_saf_tensorizer import CategoryCodec, load_canonical_dataset
from models.cof_seqgen_saf import fit_train_only_gap_support


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    # Hash only already predicate-filtered train rows, never sealed event files.
    return hashlib.sha256(frame.to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def load_train_cell(path: Path):
    return load_canonical_dataset(path, allowed_splits=("train",))


def audit_cell(path: Path, config: dict, raw_generator: dict, kappa: int) -> tuple[dict, dict]:
    dataset = load_train_cell(path)
    report_path = path / "dataset_report.json"
    manifest_path = path / "canonical_manifest.json"
    recorded = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    acquisition_path = path.parent.parent / "manifests/acquisition/controlled_coupling_dgp.json"
    acquisition = json.loads(acquisition_path.read_text())
    if sha256(acquisition_path) != manifest["source_acquisition_manifest_sha256"]:
        raise ValueError("DGP acquisition provenance mismatch")
    for source in acquisition["files"]:
        if sha256(ROOT / source["name"]) != source["sha256"]:
            raise ValueError("production DGP source changed since materialization")
    if sha256(report_path) != manifest["files"]["dataset_report.json"]["sha256"]:
        raise ValueError("canonical metadata checksum mismatch")
    if recorded["audit"]["kappa"] != kappa or recorded["audit"]["scenario"] != config["scenario"]:
        raise ValueError("canonical DGP cell does not match the frozen audit")
    if not all(recorded["gates"].values()):
        raise ValueError("source canonical gates failed")
    static = dataset.static_context.sort_values("entity_id").reset_index(drop=True)
    events = dataset.events.sort_values(["entity_id", "event_index"]).reset_index(drop=True)
    labels = static.entity_label.to_numpy(dtype=np.int64)
    if set(labels) != {0, 1}:
        raise ValueError("both observed contexts are required")
    codec = CategoryCodec.fit(static.entity_label)
    encoded = codec.encode(labels)
    if len(np.unique(encoded)) != 2 or np.any(encoded < 3):
        raise ValueError("static context codec failed")
    support = fit_train_only_gap_support(events.gap, max_positive_states=config["max_positive_gap_states"])
    if support.zero_is_explicit:
        raise ValueError("exponential DGP unexpectedly has an exact zero atom")
    if not set(support.representatives).issubset(set(events.gap.dropna())):
        raise ValueError("representatives are not observed train atoms")
    production = BenchmarkConfig.from_mapping(raw_generator, config["scenario"], kappa)
    oracle = SemiMarkovCopyOracle(production, np.asarray(support.upper_bounds))
    row_entity = static.set_index("entity_id").index.get_indexer(events.entity_id)
    if np.any(row_entity < 0):
        raise ValueError("unknown event entity")
    positions = events.event_index.to_numpy(dtype=int)
    lengths = np.bincount(row_entity, minlength=len(static))
    gaps = np.full((len(static), int(lengths.max())), np.nan)
    marks = np.zeros(gaps.shape, dtype=int)
    gaps[row_entity, positions] = events.gap.to_numpy()
    marks[row_entity, positions] = events.receiver_or_mark.str.removeprefix("receiver-").astype(int)
    summaries = {}
    max_no_gap_range = 0.0
    for label in (0, 1):
        selected = np.flatnonzero(labels == label)
        fragments = []
        active = kappa == 1 and label == 1
        for start in range(0, len(selected), config["entity_batch_size"]):
            ids = selected[start:start+config["entity_batch_size"]]
            pred = oracle.filter_batch(gaps[ids], marks[ids], lengths[ids], active=active)
            no_gap_curve = oracle.copy_curve(pred["prior_burst"], active=False)
            max_no_gap_range = max(max_no_gap_range, float(np.ptp(no_gap_curve, axis=1).max()))
            pred["entity_index"] += start
            fragments.append(pred)
        predictions = {key: np.concatenate([p[key] for p in fragments]) for key in fragments[0]}
        summaries[str(label)] = summarize_context(predictions, len(selected), production.n_receiver_categories)
    provenance = {
        "dataset_id": dataset.schema.dataset_id,
        "canonical_manifest_sha256": sha256(manifest_path),
        "metadata_integrity_checked": True,
        "event_file_full_hash_recomputed": False,
        "full_hash_note": "sealed rows not read; filtered train rows hashed instead",
        "split_assignment_sha256": dataset.source_split_assignment_sha256,
        "train_static_sha256": frame_hash(static),
        "train_events_sha256": frame_hash(events),
        "train_entity_count": len(static),
        "train_event_count": len(events),
        "observed_context_prevalence": float(labels.mean()),
        "static_codec_codes": sorted(np.unique(encoded).tolist()),
        "gap_support": support.to_dict(),
        "support_representatives_not_observed_count": 0,
        "no_gap_oracle_max_range": max_no_gap_range,
        "loaded_content_splits": ["train"],
        "oracle_latent_files_loaded": False,
    }
    return summaries, provenance


def run_audit(config: dict, canonical_root: Path, source_commit: str) -> dict:
    if config["allowed_content_splits"] != ["train"] or config["held_out_test_access"]:
        raise ValueError("this audit allows train content only")
    generator_path = ROOT / config["generator_config"]
    raw_generator = yaml.safe_load(generator_path.read_text())
    cells, provenance = {}, {}
    for cell in config["cells"]:
        kappa = int(cell["kappa"])
        if str(kappa) in cells:
            raise ValueError("duplicate kappa cell")
        print(f"Auditing train only: {cell['dataset_id']}", flush=True)
        cells[str(kappa)], provenance[str(kappa)] = audit_cell(
            canonical_root / cell["dataset_id"], config, raw_generator, kappa)
    decision = decide_oracle(cells, config["criteria"])
    decision["checks"]["paired_entity_splits"] = len({p["split_assignment_sha256"] for p in provenance.values()}) == 1
    decision["checks"]["paired_train_static"] = len({p["train_static_sha256"] for p in provenance.values()}) == 1
    decision["checks"]["no_gap_oracle_invariant"] = all(
        p["no_gap_oracle_max_range"] <= config["criteria"]["no_gap_oracle_max_range"]
        for p in provenance.values())
    decision["decision"] = "PASS" if all(decision["checks"].values()) else "FAIL"
    source_paths = ["benchmarks/cs_saf_oracle.py", "benchmarks/temporal_coupling_v2.py",
                    "benchmarks/semi_markov.py", "scripts/audit_cs_saf_oracle.py",
                    "models/cof_seqgen_saf.py", "data/cof_seqgen_saf_tensorizer.py",
                    "data/cof_seqgen_saf_contract.py", config["generator_config"],
                    "configs/benchmark_v2/cs_saf_architecture_v1.yaml",
                    "configs/benchmark_v2/cs_saf_oracle_audit_v1.yaml"]
    return {
        "schema_version": "cs-saf-oracle-audit-result-v1",
        "scope": config["scope"], "source_commit": source_commit,
        "source_sha256": {name: sha256(ROOT/name) for name in source_paths},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "criteria": config["criteria"], **decision, "cells": cells,
        "provenance": provenance,
        "prevalence_reweighting": [
            {"pi": pi, "aggregate_copy_response_range":
             pi*cells["1"]["1"]["mean_copy_probability_range"] + (1-pi)*cells["1"]["0"]["mean_copy_probability_range"]}
            for pi in config["prevalence_reweighting"]],
        "prevalence_reweighting_role": config["prevalence_reweighting_role"],
        "estimand": config["estimand"],
        "causal_do_gap_effect_claimed": False,
        "model_training_performed": False,
        "generated_model_samples_evaluated": False,
        "prevalence_grid_materialized": False,
        "loaded_content_splits": ["train"],
        "validation_accessed": False, "test_accessed": False,
        "next_stage": "IMPLEMENT_MODEL_AND_PREVALENCE_DATA_CONTRACT" if decision["decision"] == "PASS" else "STOP_REPAIR_IDENTIFICATION_DESIGN_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # A new immutable output directory is required even for a failed run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True)
        if status.strip():
            raise RuntimeError("commit the reviewed source/config before the audit")
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        config = yaml.safe_load((ROOT / "configs/benchmark_v2/cs_saf_oracle_audit_v1.yaml").read_text())
        result = run_audit(config, args.canonical_root, source_commit)
        payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n"
        (args.output_dir / "audit.json").write_text(payload)
        terminal = {"status": "COMPLETE", "scientific_decision": result["decision"],
                    "report_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                    "source_commit": source_commit, "test_accessed": False}
        (args.output_dir / "COMPLETE.json").write_text(json.dumps(terminal, indent=2)+"\n")
        print(json.dumps(terminal), flush=True)
        return 0 if result["decision"] == "PASS" else 2
    except Exception as exc:
        (args.output_dir / "FAILED.json").write_text(json.dumps(
            {"status": "FAILED", "error_type": type(exc).__name__, "message": str(exc)}, indent=2)+"\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
