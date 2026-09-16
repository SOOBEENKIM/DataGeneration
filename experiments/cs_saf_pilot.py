"""Controlled CS-SAF preparation, checkpointed training and predictive audit."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import yaml

from data.cof_seqgen_saf_tensorizer import SAFTensorizer, SAFTensorizerState, load_canonical_dataset
from experiments.cof_seqgen_saf_training import _seed_everything, _atomic_torch_save
from models.cs_saf import CSSAF, VERSION
from benchmarks.cs_saf_oracle import SemiMarkovCopyOracle, summarize_context, decide_oracle
from benchmarks.temporal_coupling_v2 import BenchmarkConfig
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json

ROOT = Path(__file__).resolve().parents[1]


def frozen_source():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("commit reviewed source/config before execution")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def dense_split(dataset, tensorizer, split):
    static = dataset.static_context.set_index("entity_id").loc[list(dataset.entity_ids_for_split(split))].reset_index()
    events = dataset.events[dataset.events.entity_id.isin(static.entity_id)].sort_values(["entity_id", "event_index"])
    row = pd.Index(static.entity_id).get_indexer(events.entity_id)
    pos = events.event_index.to_numpy(dtype=int)
    lengths = np.bincount(row, minlength=len(static))
    if lengths.max() > 32 or lengths.min() < 2:
        raise ValueError("controlled length contract changed")
    shape = (len(static), 32)
    gap = np.full(shape, np.nan, dtype=np.float32)
    mark = np.zeros(shape, dtype=np.int64)
    value = np.zeros(shape, dtype=np.float32)
    gap[row, pos] = events.gap.to_numpy(dtype=np.float32)
    mark[row, pos] = tensorizer.state.receiver_codec.encode(events.receiver_or_mark)
    value[row, pos] = tensorizer.state.event_numeric_codecs[0][1].encode(events.amount_or_numeric_value)
    codes = tensorizer.state.static_categorical_codecs[0][1].encode(static.entity_label)
    valid = np.arange(32)[None, :] < lengths[:, None]
    if np.any(mark[valid] < 3) or set(codes) != {3, 4}:
        raise ValueError("controlled categorical codec contract failed")
    return {"gap": torch.from_numpy(gap), "receiver": torch.from_numpy(mark),
            "numeric_value": torch.from_numpy(value), "valid_mask": torch.from_numpy(valid),
            "codes": torch.from_numpy(codes), "lengths": torch.from_numpy(lengths),
            "entity_ids": static.entity_id.tolist()}


def prepare_data(data_root, cache_root):
    commit = frozen_source()
    cache_root.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((ROOT/"configs/benchmark_v2/cs_saf_pilot_v1.yaml").read_text())
    audit_cfg = yaml.safe_load((ROOT/"configs/benchmark_v2/cs_saf_oracle_audit_v1.yaml").read_text())
    raw_dgp = yaml.safe_load((ROOT/"configs/benchmark_v2/full_v2_5.yaml").read_text())
    index = {"source_commit": commit, "test_accessed": False, "cells": {}, "oracle_gates": {}}
    for pi in cfg["prevalences"]:
        oracle_cells = {}
        for kappa in (0, 1):
            name = f"pi_{pi:.2f}_kappa_{kappa}"
            path = data_root/name
            manifest = json.loads((path/"cs_saf_manifest.json").read_text())
            for filename, expected in manifest["files"].items():
                if sha256(path/filename) != expected:
                    raise ValueError(f"materialized data checksum mismatch: {name}/{filename}")
            dataset = load_canonical_dataset(path, allowed_splits=("train", "validation"))
            tensorizer = SAFTensorizer.fit(dataset, max_positive_gap_states=31)
            payload = {"train": dense_split(dataset, tensorizer, "train"),
                       "validation": dense_split(dataset, tensorizer, "validation"),
                       "tensorizer_state": tensorizer.state.to_dict(), "data_manifest_sha256": sha256(path/"cs_saf_manifest.json"),
                       "source_commit": commit, "test_accessed": False}
            torch.save(payload, cache_root/f"{name}.pt")
            production = BenchmarkConfig.from_mapping(raw_dgp, "joint_semimarkov_v2b", kappa)
            oracle = SemiMarkovCopyOracle(production, np.asarray(tensorizer.state.gap_support.upper_bounds))
            summaries = {}
            train = payload["train"]
            for label in (0, 1):
                ids = torch.where(train["codes"] == label+3)[0]
                fragments = []
                for start in range(0, len(ids), 512):
                    ix = ids[start:start+512]
                    pred = oracle.filter_batch(train["gap"][ix].numpy(), train["receiver"][ix].numpy()-3,
                          train["lengths"][ix].numpy(), active=bool(kappa*label))
                    pred["entity_index"] += start
                    fragments.append(pred)
                pred = {key: np.concatenate([p[key] for p in fragments]) for key in fragments[0]}
                summaries[str(label)] = summarize_context(pred, len(ids), production.n_receiver_categories)
            oracle_cells[str(kappa)] = summaries
            index["cells"][name] = {"cache_sha256": sha256(cache_root/f"{name}.pt"),
                "data_manifest_sha256": payload["data_manifest_sha256"],
                "train_entities": len(train["lengths"]), "validation_entities": len(payload["validation"]["lengths"]),
                "oracle": summaries}
            print(f"Prepared {name}", flush=True)
        gate = decide_oracle(oracle_cells, audit_cfg["criteria"])
        index["oracle_gates"][f"{pi:.2f}"] = gate
        if gate["decision"] != "PASS":
            write_json(cache_root/"FAILED.json", index)
            raise RuntimeError(f"new-grid train-only oracle gate failed at pi={pi}")
    write_json(cache_root/"COMPLETE.json", index)
    return index


def batch(data, ids, device):
    return {**{k: data[k][ids].to(device) for k in ("gap", "receiver", "numeric_value", "valid_mask")},
            "static_categorical": (data["codes"][ids].to(device),)}


def subset(data, per_label):
    ids = torch.cat([torch.where(data["codes"] == c)[0][:per_label] for c in (3, 4)])
    return {k: ([v[int(i)] for i in ids] if isinstance(v, list) else v[ids]) for k, v in data.items()}


def make_model(payload, candidate, device):
    state = SAFTensorizerState.from_dict(payload["tensorizer_state"])
    return CSSAF(candidate, state.gap_support, **state.model_config_kwargs()).to(device)


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate(model, data, size, device):
    model.eval()
    totals = {f"{n}_{v}": 0.0 for n in ("gap", "mark", "value") for v in ("sum", "count")}
    for start in range(0, len(data["lengths"]), size):
        ids = torch.arange(start, min(start+size, len(data["lengths"])))
        terms = model.loss_terms(**batch(data, ids, device))
        for key in totals:
            totals[key] += float(terms[key])
    metrics = {f"{n}_nll": totals[f"{n}_sum"]/totals[f"{n}_count"] for n in ("gap", "mark", "value")}
    metrics["base_nll"] = sum(metrics.values())
    return metrics


def train_job(cache_path, output, candidate, device, *, cpu_gate=False):
    source_commit = frozen_source()
    output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((ROOT/"configs/benchmark_v2/cs_saf_pilot_v1.yaml").read_text())
    payload = torch.load(cache_path, map_location="cpu")
    cache_index = json.loads((cache_path.parent/"COMPLETE.json").read_text())
    if sha256(cache_path) != cache_index["cells"][cache_path.stem]["cache_sha256"]:
        raise ValueError("prepared cache checksum mismatch")
    train, validation = payload["train"], payload["validation"]
    options = dict(cfg["training"])
    if cpu_gate:
        train = subset(train, cfg["cpu_gate"]["train_entities_per_label"])
        validation = subset(validation, cfg["cpu_gate"]["validation_entities_per_label"])
        options.update({k: cfg["cpu_gate"][k] for k in ("epochs", "batch_size", "learning_rate", "patience")})
    torch.set_num_threads(1)
    _seed_everything(cfg["model_seed"])
    model = make_model(payload, candidate, device)
    initial_digest = state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    counts = {c: int((train["lengths"][train["codes"] == c]-1).sum()) for c in (3, 4)}
    generator = torch.Generator().manual_seed(cfg["model_seed"])
    history, best, best_epoch, stale = [], float("inf"), -1, 0
    started = time.monotonic()
    for epoch in range(options["epochs"]):
        model.train()
        order = torch.randperm(len(train["lengths"]), generator=generator)
        total, total_base, total_auxiliary = 0.0, 0.0, 0.0
        for start in range(0, len(order), options["batch_size"]):
            ids = order[start:start+options["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            terms = model.loss_terms(**batch(train, ids, device))
            loss, base, auxiliary = model.objective(terms, train_entities=len(order), transition_counts_by_code=counts)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(ids)
            total_base += float(base.detach())*len(ids)
            total_auxiliary += float(auxiliary.detach())*len(ids)
        metrics = evaluate(model, validation, options["batch_size"], device)
        if not np.isfinite(metrics["base_nll"]):
            raise FloatingPointError("nonfinite validation likelihood")
        record = {"epoch": epoch, "train_objective": total/len(order), "train_base_nll": total_base/len(order),
                  "train_balanced_repeat_nll": total_auxiliary/len(order), "validation": metrics}
        history.append(record)
        with (output/"progress.jsonl").open("a") as handle:
            handle.write(json.dumps(record)+"\n")
        improved = metrics["base_nll"] < best-1e-8
        stale = 0 if improved else stale+1
        if improved:
            best, best_epoch = metrics["base_nll"], epoch
            _atomic_torch_save({"version": VERSION, "candidate": candidate, "model_state": model.state_dict(),
                               "tensorizer_state": payload["tensorizer_state"], "epoch": epoch,
                               "source_commit": source_commit, "data_manifest_sha256": payload["data_manifest_sha256"],
                               "test_accessed": False}, output/"checkpoint_best.pt")
        print(f"{cache_path.stem} {candidate} epoch={epoch} validation={metrics['base_nll']:.6f}", flush=True)
        if stale >= options["patience"]:
            break
    checkpoint = torch.load(output/"checkpoint_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    report = {"version": VERSION, "source_commit": source_commit, "candidate": candidate,
              "data_cell": cache_path.stem, "data_manifest_sha256": payload["data_manifest_sha256"],
              "cache_sha256": sha256(cache_path), "config_sha256": sha256(ROOT/"configs/benchmark_v2/cs_saf_pilot_v1.yaml"),
              "cpu_gate": cpu_gate, "seed": cfg["model_seed"], "options": options,
              "parameters": sum(p.numel() for p in model.parameters()), "initial_state_sha256": initial_digest,
              "best_state_sha256": state_digest(model), "best_epoch": best_epoch, "best_validation_base_nll": best,
              "train_entities": len(train["lengths"]), "validation_entities": len(validation["lengths"]),
              "transition_counts_by_code": counts, "history": history, "seconds": time.monotonic()-started,
              "checkpoint_sha256": sha256(output/"checkpoint_best.pt"),
              "loaded_content_splits": ["train", "validation"], "test_accessed": False,
              "device": str(device), "torch": torch.__version__}
    write_json(output/"training_report.json", report)
    return model, dict(payload, train=train, validation=validation), report


@torch.no_grad()
def audit_model(model, payload, cfg, device):
    model.eval()
    data = payload["validation"]
    sums = {label: {"copy": [], "repeat": []} for label in (0, 1)}
    control_max = 0.0
    for start in range(0, len(data["lengths"]), 256):
        ids = torch.arange(start, min(start+256, len(data["lengths"])))
        inputs = batch(data, ids, device)
        hidden = model.encoder(**{k: v for k, v in inputs.items() if k != "receiver"}, receiver=inputs["receiver"])
        contexts = model.context(hidden, inputs["static_categorical"])
        mask = inputs["valid_mask"].clone(); mask[:, 0] = False
        previous = inputs["receiver"].roll(1, dims=1)
        flat_context, flat_previous = contexts[mask], previous[mask]
        copy_parts, repeat_parts = [], []
        for j in range(0, len(flat_context), cfg["audit_chunk_histories"]):
            c, p = flat_context[j:j+cfg["audit_chunk_histories"]], flat_previous[j:j+cfg["audit_chunk_histories"]]
            copy, repeat = model.response_curves(c, p)
            zero_copy, zero_repeat = model.response_curves(c, p, zero_gap=True)
            control_max = max(control_max, float((zero_copy.max(1).values-zero_copy.min(1).values).max()),
                              float((zero_repeat.max(1).values-zero_repeat.min(1).values).max()))
            copy_parts.append(copy.max(1).values-copy.min(1).values)
            repeat_parts.append(repeat.max(1).values-repeat.min(1).values)
        for metric, parts in (("copy", copy_parts), ("repeat", repeat_parts)):
            ranges = torch.zeros_like(inputs["gap"])
            ranges[mask] = torch.cat(parts)
            entity_mean = ranges.sum(1)/mask.sum(1)
            for label in (0, 1):
                selected = entity_mean[inputs["static_categorical"][0] == label+3]
                sums[label][metric].extend(selected.cpu().tolist())
    response = {str(label): {"entities": len(values["copy"]),
                **{f"mean_{m}_range": float(np.mean(x)) for m, x in values.items()}}
                for label, values in sums.items()}
    rng = np.random.default_rng(cfg["sampling_seed"])
    plan = rng.integers(len(payload["train"]["lengths"]), size=cfg["generation_entities"])
    torch.manual_seed(cfg["sampling_seed"])
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["sampling_seed"])
    violations, invalid_marks, finite_values, generated = 0, 0, True, 0
    for start in range(0, len(plan), cfg["generation_batch_size"]):
        ids = torch.tensor(plan[start:start+cfg["generation_batch_size"]])
        sample = model.sample_fixed_lengths(payload["train"]["lengths"][ids].tolist(),
                              static_categorical=(payload["train"]["codes"][ids],), device=device)
        mask = sample["valid_mask"].clone(); mask[:, 0] = False
        reps = torch.tensor(model.support.representatives, device=device, dtype=sample["gap"].dtype)
        violations += int((~torch.isin(sample["gap"][mask], reps)).sum())
        invalid_marks += int((sample["receiver"][sample["valid_mask"]] < 3).sum())
        finite_values &= bool(torch.isfinite(sample["numeric_value"][sample["valid_mask"]]).all())
        if not torch.isnan(sample["gap"][:, 0]).all():
            raise ValueError("first generated gap must be missing")
        generated += int(mask.sum())
    return {"responses": response, "zero_gap_control_max_range": control_max,
            "generated_gap_count": generated, "gap_support_violations": violations,
            "invalid_reserved_marks": invalid_marks, "finite_generated_values": finite_values,
            "sampling_plan_sha256": hashlib.sha256(plan.tobytes()).hexdigest(),
            "test_accessed": False, "validation_accessed": True}


def decide_pilot(audits, gate):
    checks = {}
    for metric in ("copy", "repeat"):
        active = audits[1]["responses"]["1"][f"mean_{metric}_range"]
        null = max(audits[k]["responses"][str(y)][f"mean_{metric}_range"]
                   for k, y in ((0, 0), (0, 1), (1, 0)))
        checks[f"{metric}_active_material"] = active >= gate["active_mean_range_min"]
        checks[f"{metric}_null_small"] = null <= gate["noncausal_mean_range_max"]
        checks[f"{metric}_selective"] = active >= gate["selectivity_ratio_min"]*null
    checks["matched_zero_gap_control_invariant"] = all(a["zero_gap_control_max_range"] <= gate["zero_gap_control_max_range"] for a in audits.values())
    checks["support_preserved"] = all(a["gap_support_violations"] == 0 for a in audits.values())
    checks["valid_mark_and_value_outputs"] = all(a["invalid_reserved_marks"] == 0 and a["finite_generated_values"] for a in audits.values())
    return {"decision": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
