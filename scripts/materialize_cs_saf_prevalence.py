"""Materialize the frozen four-prevalence by two-kappa development grid."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.cs_saf_prevalence import prevalence_view
from data.cof_seqgen_saf_tensorizer import load_canonical_dataset
from data.cof_seqgen_saf_contract import CanonicalEntitySequenceDataset, CanonicalSchema
from scripts.audit_cs_saf_oracle import frame_hash, sha256


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def materialize(canonical_root, output_root):
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise RuntimeError("commit source/config before materialization")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    output_root.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((ROOT/"configs/benchmark_v2/cs_saf_pilot_v1.yaml").read_text())
    dgp = yaml.safe_load((ROOT/"configs/benchmark_v2/full_v2_5.yaml").read_text())
    copies = dgp["coupling"]["joint_semimarkov"]["receiver_repeat_probability"]
    summary = {"schema_version": "cs-saf-prevalence-materialization-v1", "source_commit": commit,
               "config_sha256": sha256(ROOT/"configs/benchmark_v2/cs_saf_pilot_v1.yaml"),
               "content_splits": ["train", "validation"], "test_accessed": False,
               "test_content_materialized": False, "cells": {}}
    for kappa in (0, 1):
        source = canonical_root/f"controlled_coupling_joint_semimarkov_v2b_kappa_{kappa}_00"
        dataset = load_canonical_dataset(source, allowed_splits=("train", "validation"))
        full_splits = pd.read_parquet(source/"entity_splits.parquet")
        ids = dataset.static_context.entity_id.tolist()
        oracle = pd.read_parquet(source/"oracle_latents.parquet", filters=[("entity_id", "in", ids)],
                                 columns=["entity_id", "event_id", "event_index", "gap_state"])
        schema_base = json.loads((source/"schema.json").read_text())
        original_static = dataset.static_context.sort_values("entity_id").reset_index(drop=True)
        original_events = dataset.events.sort_values(["entity_id", "event_index"]).reset_index(drop=True)
        invariant_columns = [c for c in original_events if c != "receiver_or_mark"]
        baseline_invariant_hash = frame_hash(original_events[invariant_columns])
        for pi in cfg["prevalences"]:
            name = f"pi_{pi:.2f}_kappa_{kappa}"
            destination = output_root/name
            destination.mkdir()
            static, events, changes = prevalence_view(
                dataset.static_context, dataset.events, oracle, prevalence=pi, kappa=kappa,
                generation_seed=cfg["data_generation_seed"], mark_seed=cfg["new_mark_stream_seed"],
                category_count=dgp["data"]["n_receiver_categories"],
                q_low=copies["low"], q_high=copies["high"])
            schema_raw = dict(schema_base, dataset_id=f"controlled_coupling_cs_saf_{name}")
            schema = CanonicalSchema(schema_raw["dataset_id"], schema_raw["time_representation"],
                                      schema_raw["timestamp_unit"], static_context_columns=("entity_label",))
            canonical = CanonicalEntitySequenceDataset(schema, static, events, dataset.entity_splits,
                          required_splits=("train", "validation"),
                          source_split_assignment_sha256=dataset.source_split_assignment_sha256)
            schema_raw["schema_sha256"] = schema.schema_sha256
            unchanged = frame_hash(events[invariant_columns]) == baseline_invariant_hash
            exact_base = pi != .05 or (frame_hash(static) == frame_hash(original_static)
                                       and frame_hash(events) == frame_hash(original_events))
            null_marks = kappa != 0 or events.receiver_or_mark.equals(original_events.receiver_or_mark)
            if not (unchanged and exact_base and null_marks):
                raise RuntimeError("paired materialization invariants failed")
            static.to_parquet(destination/"static_context.parquet", index=False)
            events.to_parquet(destination/"events.parquet", index=False)
            full_splits.to_parquet(destination/"entity_splits.parquet", index=False)
            write_json(destination/"schema.json", schema_raw)
            groups = static.merge(dataset.entity_splits, on="entity_id").groupby(["split", "entity_label"]).size()
            report = {"dataset_id": schema.dataset_id, "pi": pi, "kappa": kappa, **changes,
                      "source_commit": commit, "source_canonical_manifest_sha256": sha256(source/"canonical_manifest.json"),
                      "split_assignment_sha256": canonical.split_assignment_sha256,
                      "context_counts": {f"{s}_label_{y}": int(n) for (s, y), n in groups.items()},
                      "event_count": len(events), "nonmark_content_preserved": unchanged,
                      "pi_005_exact_original_development_content": exact_base if pi == .05 else None,
                      "null_marks_preserved": null_marks if kappa == 0 else None,
                      "oracle_used_only_in_dgp_materializer": True,
                      "oracle_fields_written_to_model_data": False,
                      "content_splits": ["train", "validation"], "test_accessed": False,
                      "files": {p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file()}}
            write_json(destination/"cs_saf_manifest.json", report)
            summary["cells"][name] = report
            print(f"Materialized {name}: {report['context_counts']}", flush=True)
    if len({r["split_assignment_sha256"] for r in summary["cells"].values()}) != 1:
        raise RuntimeError("split membership changed across the grid")
    write_json(output_root/"COMPLETE.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("choose a new output root; prior artifacts are immutable")
    try:
        materialize(args.canonical_root, args.output_root)
    except Exception as exc:
        if args.output_root.is_dir() and not (args.output_root/"COMPLETE.json").exists():
            write_json(args.output_root/"FAILED.json", {"error": type(exc).__name__, "message": str(exc)})
        raise
