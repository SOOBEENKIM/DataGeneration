"""Run immutable train-only diagnostics on the 12 completed CS-SAF v1 fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from data.cof_seqgen_saf_tensorizer import SAFTensorizer, SAFTensorizerState, load_canonical_dataset
from experiments.cs_saf_pilot import frozen_source, dense_split, state_digest
from experiments.cof_seqgen_saf_training import _seed_everything
from experiments.cs_saf_forensics import analyze_model
from models.cs_saf import CSSAF
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    commit = frozen_source()
    config_path = ROOT/"configs/benchmark_v2/cs_saf_forensics_v1.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    evidence = json.loads((ROOT/"docs/cs_saf/pilot_v1_result.json").read_text())
    if sha256(args.pilot_root/"COMPLETE.json") != evidence["terminal_sha256"]["pilot"]:
        raise ValueError("pilot terminal identity changed")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    _seed_everything(20260930)
    result = {"schema_version": cfg["schema_version"], "source_commit": commit,
              "config_sha256": sha256(config_path), "checkpoint_source_commit": evidence["execution_sources"]["cpu_and_pilot"],
              "pilot_decision_unchanged": "FAIL", "loaded_content_splits": ["train"],
              "validation_content_accessed": False, "test_accessed": False,
              "new_model_fits": 0, "optimizer_steps": 0,
              "torch_version": torch.__version__, "device": args.device, "jobs": {}}
    device = torch.device(args.device)
    try:
        for pi in cfg["prevalences"]:
            for kappa in cfg["kappas"]:
                name = f"pi_{pi:.2f}_kappa_{kappa}"
                data_path = args.data_root/name
                expected_manifest = evidence["preparation"]["cells"][name]["data_manifest_sha256"]
                if sha256(data_path/"cs_saf_manifest.json") != expected_manifest:
                    raise ValueError("data manifest changed")
                dataset = load_canonical_dataset(data_path, allowed_splits=("train",))
                tensorizer_state = None
                for candidate in cfg["candidates"]:
                    job = name+"/"+candidate
                    expected = evidence["trained_jobs"][job]
                    checkpoint_path = args.pilot_root/job/"checkpoint_best.pt"
                    if sha256(checkpoint_path) != expected["checkpoint_sha256"]:
                        raise ValueError("checkpoint bytes changed")
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    if checkpoint["source_commit"] != result["checkpoint_source_commit"]:
                        raise ValueError("unexpected checkpoint source")
                    if checkpoint["data_manifest_sha256"] != expected_manifest:
                        raise ValueError("checkpoint/data mismatch")
                    if tensorizer_state is None:
                        tensorizer_state = checkpoint["tensorizer_state"]
                        state = SAFTensorizerState.from_dict(tensorizer_state)
                        data = dense_split(dataset, SAFTensorizer(state), "train")
                        digest = hashlib.sha256()
                        for key, value in sorted(data.items()):
                            digest.update(key.encode())
                            digest.update(json.dumps(value).encode() if isinstance(value, list)
                                          else value.contiguous().numpy().tobytes())
                        train_tensor_sha256 = digest.hexdigest()
                    elif tensorizer_state != checkpoint["tensorizer_state"]:
                        raise ValueError("paired tensorizers differ")
                    model = CSSAF(candidate, state.gap_support, **state.model_config_kwargs()).to(device)
                    model.load_state_dict(checkpoint["model_state"])
                    before = state_digest(model)
                    if before != expected["best_state_sha256"]:
                        raise ValueError("checkpoint tensor identity mismatch")
                    metrics, arrays = analyze_model(model, data, cfg, device)
                    if before != state_digest(model) or sha256(checkpoint_path) != expected["checkpoint_sha256"]:
                        raise RuntimeError("forensics mutated a checkpoint")
                    artifact = args.output/name/candidate
                    artifact.mkdir(parents=True, exist_ok=False)
                    np.savez_compressed(artifact/"train_entity_and_gradient_arrays.npz", **arrays)
                    metrics.update(checkpoint_sha256=expected["checkpoint_sha256"], model_state_sha256=before,
                        checkpoint_epoch=checkpoint["epoch"], data_manifest_sha256=expected_manifest,
                        arrays_sha256=sha256(artifact/"train_entity_and_gradient_arrays.npz"),
                        filtered_train_tensor_sha256=train_tensor_sha256,
                        full_event_file_hash_recomputed=False,
                        file_hash_note="only train rows decoded; full container hash remains the prior verified provenance",
                        checkpoint_unchanged=True)
                    write_json(artifact/"audit.json", metrics)
                    result["jobs"][job] = metrics
                    write_json(args.output/"progress.json", result)
                    print(job, {y: round(s["mean_copy_range"], 6) for y, s in metrics["groups"].items()},
                          "gradient_cosine", round(metrics["route_gradient_cosine_between_labels"], 6), flush=True)
        result["status"] = "COMPLETE"
        write_json(args.output/"audit.json", result)
        write_json(args.output/"COMPLETE.json", {"status": "COMPLETE", "source_commit": commit,
                   "audit_sha256": sha256(args.output/"audit.json"), "model_fits": 0,
                   "optimizer_steps": 0, "validation_content_accessed": False, "test_accessed": False})
    except Exception as exc:
        write_json(args.output/"FAILED.json", {"source_commit": commit, "error": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
