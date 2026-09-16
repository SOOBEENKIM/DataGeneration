"""Frozen three-objective diagnostic; historical v1/v2 execution is untouched."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from experiments.cof_seqgen_saf_training import _seed_everything, _atomic_torch_save
from experiments.cs_saf_pilot import (
    ROOT, batch, subset, evaluate, state_digest, frozen_source,
    load_pilot_config, verify_v2_cache_index, verify_v2_initialization,
)
from models.cs_saf_v2 import CSSAFv2
from scripts.audit_cs_saf_oracle import sha256
from scripts.materialize_cs_saf_prevalence import write_json

CONTRACT_SHA = "1de81bc0f0af1fe0ac829a1f24fe82a58c88348212e66f74b25e230a1ead0cdd"
CANDIDATES = ("CS2-U1", "CS2-A1", "CS2-B1")
COLUMNS = ("copy_range", "repeat_range", "central_copy_range", "central_repeat_range",
           "observed_repeat_BCE", "zero_gap_repeat_BCE", "route_minus_zero_repeat_BCE")


def load_contract():
    path = ROOT/"configs/benchmark_v2/cs_saf_loss_control_v1.yaml"
    if sha256(path) != CONTRACT_SHA:
        raise ValueError("loss-control preregistration changed")
    contract = yaml.safe_load(path.read_text())
    cfg, v2_path = load_pilot_config("v2")
    for p, key in [(v2_path, "inherited_v2_config_sha256"),
                   (ROOT/"configs/benchmark_v2/cs_saf_pilot_v1.yaml", "inherited_pilot_config_sha256"),
                   (ROOT/"docs/cs_saf/v2_pilot_v1_result.json", "prior_result_sha256")]:
        if sha256(p) != contract[key]:
            raise ValueError(f"registered input changed: {p}")
    return contract, cfg


class LossControlModel(CSSAFv2):
    """A has U's exact forward/initialization and one new scalar loss term."""
    def __init__(self, candidate, support, **kwargs):
        if candidate not in CANDIDATES:
            raise ValueError("unknown loss-control candidate")
        super().__init__("CS2-U1" if candidate == "CS2-A1" else candidate, support, **kwargs)
        self.diagnostic_candidate = candidate

    def objective(self, terms, *, train_entities, transition_counts_by_code):
        loss, base, balanced = super().objective(terms, train_entities=train_entities,
                                                transition_counts_by_code=transition_counts_by_code)
        if self.diagnostic_candidate != "CS2-A1":
            return loss, base, balanced
        auxiliary = terms["repeat_per_entity"].mean() * (train_entities/sum(transition_counts_by_code.values()))
        return base+auxiliary, base, auxiliary

    def architecture_contract(self):
        result = super().architecture_contract()
        result.update(candidate=self.diagnostic_candidate, loss_diagnostic="three_objective_v1",
                      auxiliary_reduction={"CS2-U1": "none", "CS2-A1": "global_train_transitions",
                                           "CS2-B1": "equal_context_train_transitions"}[self.diagnostic_candidate])
        return result


def make_model(payload, candidate, device):
    state = SAFTensorizerState.from_dict(payload["tensorizer_state"])
    return LossControlModel(candidate, state.gap_support, **state.model_config_kwargs()).to(device)


def load_cache(path):
    verify_v2_cache_index(path.parent)
    index = json.loads((path.parent/"COMPLETE.json").read_text())
    if path.stem not in ("pi_0.05_kappa_0", "pi_0.05_kappa_1"):
        raise ValueError("only registered prevalence/cells may be loaded")
    if sha256(path) != index["cells"][path.stem]["cache_sha256"]:
        raise ValueError("prepared cache changed")
    if index["oracle_gates"]["0.05"]["decision"] != "PASS":
        raise ValueError("train-only oracle prerequisite failed")
    return torch.load(path, map_location="cpu")


def train(cache, output, candidate, device, *, cpu=False):
    commit = frozen_source()
    contract, cfg = load_contract()
    if candidate not in contract["candidates"]:
        raise ValueError("unregistered objective")
    payload = load_cache(cache)
    output.mkdir(parents=True, exist_ok=False)
    train_data, validation = payload["train"], payload["validation"]
    options = dict(cfg["training"])
    if cpu:
        train_data = subset(train_data, cfg["cpu_gate"]["train_entities_per_label"])
        validation = subset(validation, cfg["cpu_gate"]["validation_entities_per_label"])
        options.update({k: cfg["cpu_gate"][k] for k in ("epochs", "batch_size", "learning_rate", "patience")})
    torch.set_num_threads(1)
    _seed_everything(cfg["model_seed"])
    model = make_model(payload, candidate, device)
    initial = state_digest(model)
    initialization = verify_v2_initialization(model, payload, cfg["model_seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    n = len(train_data["lengths"])
    counts = {c: int((train_data["lengths"][train_data["codes"] == c]-1).sum()) for c in (3, 4)}
    generator = torch.Generator().manual_seed(cfg["model_seed"])
    order_hash = hashlib.sha256()
    history, best, best_epoch, stale = [], float("inf"), -1, 0
    started = time.monotonic()
    for epoch in range(options["epochs"]):
        model.train()
        order = torch.randperm(n, generator=generator)
        order_hash.update(order.numpy().tobytes())
        total, total_base, total_aux = 0., 0., 0.
        for start in range(0, n, options["batch_size"]):
            ids = order[start:start+options["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            terms = model.loss_terms(**batch(train_data, ids, device))
            loss, base, aux = model.objective(terms, train_entities=n, transition_counts_by_code=counts)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(ids)
            total_base += float(base.detach())*len(ids)
            total_aux += float(aux.detach())*len(ids)
        metrics = evaluate(model, validation, options["batch_size"], device)
        if not np.isfinite(metrics["base_nll"]):
            raise FloatingPointError("nonfinite validation loss")
        record = {"epoch": epoch, "train_objective": total/n, "train_base_nll": total_base/n,
                  "train_auxiliary_diagnostic": total_aux/n,
                  "auxiliary_applied": candidate != "CS2-U1", "validation": metrics,
                  "entity_order_prefix_sha256": order_hash.hexdigest()}
        history.append(record)
        with (output/"progress.jsonl").open("a") as handle:
            handle.write(json.dumps(record)+"\n")
        improved = metrics["base_nll"] < best-1e-8
        stale = 0 if improved else stale+1
        checkpoint = {"version": model.architecture_contract()["implementation_version"], "candidate": candidate,
                      "model_state": model.state_dict(), "tensorizer_state": payload["tensorizer_state"],
                      "epoch": epoch, "source_commit": commit, "config_sha256": CONTRACT_SHA,
                      "data_manifest_sha256": payload["data_manifest_sha256"], "test_accessed": False}
        if improved:
            best, best_epoch = metrics["base_nll"], epoch
            _atomic_torch_save(checkpoint, output/"checkpoint_best.pt")
        if epoch == 9:
            _atomic_torch_save(checkpoint, output/"checkpoint_epoch_9.pt")
        print(f"{cache.stem} {candidate} epoch={epoch} val={metrics['base_nll']:.6f}", flush=True)
        if stale >= options["patience"]:
            break
    checkpoint = torch.load(output/"checkpoint_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    report = {"source_commit": commit, "config_sha256": CONTRACT_SHA, "candidate": candidate,
              "version": model.architecture_contract()["implementation_version"],
              "data_cell": cache.stem, "cache_sha256": sha256(cache),
              "data_manifest_sha256": payload["data_manifest_sha256"],
              "cpu_gate": cpu, "seed": cfg["model_seed"], "options": options,
              "parameters": sum(p.numel() for p in model.parameters()), "initial_state_sha256": initial,
              "initialization_contract": initialization, "architecture": model.architecture_contract(),
              "best_state_sha256": state_digest(model), "best_epoch": best_epoch,
              "best_validation_base_nll": best, "history": history, "epochs_completed": len(history),
              "train_entities": n, "validation_entities": len(validation["lengths"]),
              "transition_counts_by_code": counts, "train_events": int(train_data["lengths"].sum()),
              "seconds": time.monotonic()-started, "checkpoint_sha256": sha256(output/"checkpoint_best.pt"),
              "fixed_epoch_checkpoint_sha256": sha256(output/"checkpoint_epoch_9.pt") if (output/"checkpoint_epoch_9.pt").exists() else None,
              "test_accessed": False, "loaded_content_splits": ["train", "validation"],
              "device": str(device), "torch": torch.__version__}
    write_json(output/"training_report.json", report)
    return model, dict(payload, train=train_data, validation=validation), report


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "entity_standard_error": float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None,
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9))}


def central_intervals(model, train_data, quantiles=(.05, .95)):
    result = {}
    for label in (0, 1):
        selected = train_data["codes"] == label+3
        mask = train_data["valid_mask"][selected].clone(); mask[:, 0] = False
        bins = model._support_code(train_data["gap"][selected][mask]).cpu().numpy()-3
        counts = np.bincount(bins, minlength=len(model.support.representatives))
        cumulative = np.cumsum(counts)/counts.sum()
        lo, hi = [int(np.searchsorted(cumulative, q)) for q in quantiles]
        if hi <= lo:
            raise ValueError("central interval needs at least two bins")
        result[str(label)] = {"inclusive_bins": [lo, hi], "train_bin_counts": counts.tolist()}
    return result


@torch.no_grad()
def diagnose(model, data, intervals, device, batch_size=256):
    model.eval()
    groups, arrays = {}, {}
    for label in (0, 1):
        selected = torch.where(data["codes"] == label+3)[0]
        lo, hi = intervals[str(label)]["inclusive_bins"]
        blocks, entity_ids = [], []
        for start in range(0, len(selected), batch_size):
            ids = selected[start:start+batch_size]
            inputs = batch(data, ids, device)
            context = model.context(model.encoder(**inputs), inputs["static_categorical"])
            mask = inputs["valid_mask"].clone(); mask[:, 0] = False
            rows, _ = torch.where(mask)
            previous = inputs["receiver"].roll(1, dims=1)[mask]
            codes = inputs["static_categorical"][0][:, None].expand_as(mask)[mask]
            q, p = model.response_curves(context[mask], previous, static_codes=codes)
            _, zero = model.response_curves(context[mask], previous, static_codes=codes, zero_gap=True)
            bins = model._support_code(inputs["gap"][mask])-3
            factual = p.gather(1, bins[:, None])[:, 0].double().clamp(1e-12, 1-1e-12)
            zero = zero[:, 0].double().clamp(1e-12, 1-1e-12)
            observed = inputs["receiver"][mask] == previous
            bce = -torch.where(observed, factual.log(), torch.log1p(-factual))
            zero_bce = -torch.where(observed, zero.log(), torch.log1p(-zero))
            metrics = torch.stack((q.max(1).values-q.min(1).values, p.max(1).values-p.min(1).values,
                q[:, lo:hi+1].max(1).values-q[:, lo:hi+1].min(1).values,
                p[:, lo:hi+1].max(1).values-p[:, lo:hi+1].min(1).values,
                bce, zero_bce, bce-zero_bce), dim=1).double()
            entity = torch.zeros((len(ids), len(COLUMNS)), dtype=torch.float64, device=device)
            entity.index_add_(0, rows, metrics/mask.sum(1)[rows, None])
            blocks.append(entity.cpu().numpy())
            entity_ids.extend(data["entity_ids"][int(i)] for i in ids)
        values = np.concatenate(blocks)
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite diagnostic")
        groups[str(label)] = {"entities": len(values),
            "transitions": int((data["lengths"][selected]-1).sum()),
            "metrics": {name: summarize(values[:, j]) for j, name in enumerate(COLUMNS)},
            "fraction_copy_range_above_005": float((values[:, 0] > .05).mean()),
            "central_interval": intervals[str(label)]}
        arrays[f"label_{label}_metrics"] = values
        arrays[f"label_{label}_entity_ids"] = np.asarray(entity_ids, dtype=str)
    return {"groups": groups, "columns": list(COLUMNS), "test_accessed": False}, arrays


def save_diagnostics(model, payload, output, device):
    contract, _ = load_contract()
    intervals = central_intervals(model, payload["train"], contract["diagnostics"]["central_interval_quantiles"])
    result, arrays = {}, {}
    best_digest = state_digest(model)
    for name, filename in [("best", "checkpoint_best.pt"), ("epoch_9", "checkpoint_epoch_9.pt")]:
        path = output/filename
        if not path.exists():
            result[name] = {"status": "MISSING_NO_REPLACEMENT"}
            continue
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        result[name] = {"epoch": checkpoint["epoch"], "state_sha256": state_digest(model), "splits": {}}
        for split in ("train", "validation"):
            summary, raw = diagnose(model, payload[split], intervals, device)
            result[name]["splits"][split] = summary
            arrays.update({f"{name}_{split}_{key}": value for key, value in raw.items()})
    model.load_state_dict(torch.load(output/"checkpoint_best.pt", map_location=device)["model_state"])
    if state_digest(model) != best_digest:
        raise RuntimeError("diagnostics changed best checkpoint")
    np.savez_compressed(output/"entity_diagnostics.npz", **arrays)
    record = {"checkpoints": result, "entity_arrays_sha256": sha256(output/"entity_diagnostics.npz"),
              "test_accessed": False, "no_optimizer_steps": True}
    write_json(output/"null_diagnostics.json", record)
    return record


def paired_contrasts(job_folders):
    """Require identical entity identities and order; never silently intersect."""
    result = {}
    for left, right in (("CS2-A1", "CS2-U1"), ("CS2-B1", "CS2-A1")):
        with np.load(job_folders[left]/"entity_diagnostics.npz", allow_pickle=False) as a, \
             np.load(job_folders[right]/"entity_diagnostics.npz", allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                # Missing fixed epoch is explicitly reported, best remains available.
                common = set(a.files) & set(b.files)
            else:
                common = set(a.files)
            contrasts = {}
            for key in sorted(k for k in common if k.endswith("_metrics")):
                id_key = key.replace("_metrics", "_entity_ids")
                if not np.array_equal(a[id_key], b[id_key]):
                    raise ValueError("paired diagnostic entities do not align")
                delta = a[key]-b[key]
                contrasts[key] = {col: summarize(delta[:, j]) for j, col in enumerate(COLUMNS)}
            result[f"{left}_minus_{right}"] = {"metrics": contrasts,
                "unavailable_arrays": sorted(set(a.files)^set(b.files))}
    return result
