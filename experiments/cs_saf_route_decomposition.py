"""Read-only, train-reference logit decomposition of frozen CS-SAF v2 states."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from experiments.cs_saf_pilot import batch
from experiments.cs_saf_loss_control import summarize

ARMS = ("F", "M", "R", "Z")
PAIRS = (("M", "F"), ("M", "Z"), ("R", "F"), ("F", "Z"))
COLUMNS = tuple(f"{a}_{m}" for a in ARMS for m in
                ("repeat_BCE", "mark_NLL", "copy_range", "repeat_range")) + (
    "mean_logit", "absolute_mean_logit", "residual_RMS") + tuple(
    f"{a}_minus_{b}_repeat_BCE" for a, b in PAIRS)


def reference_measure(model, train):
    """Only explicit train tensors may be supplied; no validation-dependent fit."""
    weights, metadata = [], {}
    for label in (0, 1):
        selected = train["codes"] == label+3
        mask = train["valid_mask"][selected].clone(); mask[:, 0] = False
        codes = model._support_code(train["gap"][selected][mask]).cpu().numpy()-3
        counts = np.bincount(codes, minlength=len(model.support.representatives))
        if counts.sum() == 0:
            raise ValueError("both train contexts need transitions")
        p = counts/counts.sum()
        weights.append(p)
        metadata[str(label)] = {"counts": counts.tolist(), "probabilities": p.tolist(),
                                "transitions": int(counts.sum()), "fit_split": "train"}
    return torch.tensor(np.stack(weights), dtype=torch.float64), metadata


def decompose_logits(model, context, codes, weights):
    slots = model.slots(codes, context.shape[:-1])
    if context.ndim != 2 or weights.shape != (2, len(model.support.representatives)):
        raise ValueError("invalid flattened context/reference measure")
    weights = weights.to(device=context.device, dtype=torch.float64)
    if not torch.isfinite(weights).all() or (weights < 0).any() or not torch.allclose(weights.sum(1), torch.ones(2, device=context.device, dtype=torch.float64), atol=1e-12, rtol=0):
        raise ValueError("reference weights must be normalized probabilities")
    gaps = torch.tensor(model.support.representatives, device=context.device, dtype=context.dtype)
    embedded = model.gap_route(model._support_code(gaps))
    route = torch.empty((len(context), len(gaps)), device=context.device, dtype=context.dtype)
    for slot in (0, 1):
        selected = slots == slot
        if selected.any():
            u = torch.tanh(F.linear(context[selected], model.route_context_weight[slot], model.route_context_bias[slot]))*model.route_interaction_weight[slot]
            v = torch.tanh(F.linear(embedded, model.route_gap_weight[slot]))
            route[selected] = u@v.T/math.sqrt(16)
    route = route.double()
    base = model.copy_base(context).double()
    pi = weights[slots]
    mean = (route*pi).sum(1, keepdim=True)
    residual = route-mean
    return {"F": base+mean+residual, "M": base+mean, "R": base+residual, "Z": base}, {
        "route": route, "base": base, "mean": mean, "residual": residual, "weights": pi}


def arm_predictions(logits, fresh_logp, previous, actual, bins):
    """Stable observable likelihoods; no latent copy coin labels."""
    log_prev = fresh_logp.gather(1, previous[:, None]).double()
    log_actual = fresh_logp.gather(1, actual[:, None]).double()[:, 0]
    observed = actual == previous
    results = {}
    for name, ell in logits.items():
        log_q, log_not_q = F.logsigmoid(ell), F.logsigmoid(-ell)
        log_repeat = torch.logaddexp(log_q, log_not_q+log_prev)
        log_nonrepeat = log_not_q+torch.log1p(-log_prev.exp())
        selected_bins = bins if ell.shape[1] > 1 else torch.zeros_like(bins)
        lr = log_repeat.gather(1, selected_bins[:, None])[:, 0]
        lnr = log_nonrepeat.gather(1, selected_bins[:, None])[:, 0]
        factual_not_q = log_not_q.gather(1, selected_bins[:, None])[:, 0]
        q, p = log_q.exp(), log_repeat.exp()
        results[name] = {"copy": q, "repeat": p,
            "repeat_BCE": -torch.where(observed, lr, lnr),
            "mark_NLL": -torch.where(observed, lr, factual_not_q+log_actual),
            "copy_range": q.max(1).values-q.min(1).values,
            "repeat_range": p.max(1).values-p.min(1).values}
    return results


def audit_payload(model, payload):
    state = payload["tensorizer_state"]
    if state["fit_split"] != "train":
        raise ValueError("tensorizer was not train-fitted")
    if set(payload["train"]["entity_ids"]) & set(payload["validation"]["entity_ids"]):
        raise ValueError("entity split leakage")
    reps = torch.tensor(model.support.representatives, dtype=torch.float32)
    if not torch.equal(model._support_code(reps), torch.arange(len(reps))+3):
        raise ValueError("saved support representatives do not round-trip")
    result = {"train_fitted_tensorizer": True, "entity_disjoint": True,
              "representatives_round_trip": True, "support_bins": len(reps), "splits": {}}
    for split in ("train", "validation"):
        d = payload[split];mask = d["valid_mask"].clone();mask[:, 0] = False
        if not torch.isnan(d["gap"][:, 0]).all() or not torch.isfinite(d["gap"][mask]).all():
            raise ValueError("first/nonfirst gap contract")
        if not torch.equal(d["valid_mask"], torch.arange(d["gap"].shape[1])[None, :] < d["lengths"][:, None]):
            raise ValueError("padding/length mismatch")
        if not ((d["lengths"] >= 2) & (d["lengths"] <= 32)).all() or set(d["codes"].tolist()) != {3,4}:
            raise ValueError("length/context contract")
        if not ((d["receiver"][d["valid_mask"]] >= 3) & (d["receiver"][d["valid_mask"]] < model.config.receiver_vocab_size)).all():
            raise ValueError("reserved/unknown mark")
        if len(set(d["entity_ids"])) != len(d["entity_ids"]) or not torch.isfinite(d["numeric_value"][d["valid_mask"]]).all():
            raise ValueError("identity/value contract")
        result["splits"][split] = {"entities": len(d["lengths"]), "transitions": int(mask.sum()),
                                   "rare_entities": int((d["codes"] == 4).sum())}
    return result


@torch.no_grad()
def analyze(model, data, weights, device, *, batch_size=256):
    # Match the parent audit's group-wise GRU batch shape: CUDA arithmetic can
    # differ with batch shape even when all checkpoint tensors are unchanged.
    model.eval()
    checks = {"reconstruction_max_absolute_error": 0., "weighted_residual_mean_max_absolute_error": 0.,
              "original_copy_and_repeat_probability_match_max_absolute_error": 0.,
              "pairwise_repeat_BCE_and_mark_NLL_delta_match_max_absolute_error": 0.,
              "M_Z_probability_range_max_absolute": 0.}
    groups, arrays = {}, {}
    for label in (0, 1):
        ids_all = torch.where(data["codes"] == label+3)[0]
        blocks, names = [], []
        for start in range(0, len(ids_all), batch_size):
            ids = ids_all[start:start+batch_size]
            x = batch(data, ids, device)
            context = model.context(model.encoder(**x), x["static_categorical"])
            mask = x["valid_mask"].clone();mask[:, 0] = False
            rows, _ = torch.where(mask)
            previous = x["receiver"].roll(1, dims=1)[mask]
            codes = x["static_categorical"][0][:, None].expand_as(mask)[mask]
            logits, parts = decompose_logits(model, context[mask], codes, weights)
            fresh = model.new_mark_head(context[mask]);fresh[:, :3] = -torch.inf
            predicted = arm_predictions(logits, fresh.log_softmax(-1), previous,
                                        x["receiver"][mask], model._support_code(x["gap"][mask])-3)
            original_q, original_p = model.response_curves(context[mask], previous, static_codes=codes)
            local = {
                "reconstruction_max_absolute_error": (logits["F"]-parts["base"]-parts["route"]).abs().max(),
                "weighted_residual_mean_max_absolute_error": (parts["residual"]*parts["weights"]).sum(1).abs().max(),
                "original_copy_and_repeat_probability_match_max_absolute_error": torch.maximum(
                    (original_q-predicted["F"]["copy"]).abs().max(), (original_p-predicted["F"]["repeat"]).abs().max()),
                "pairwise_repeat_BCE_and_mark_NLL_delta_match_max_absolute_error": torch.stack([
                    ((predicted[a]["repeat_BCE"]-predicted[b]["repeat_BCE"])
                     -(predicted[a]["mark_NLL"]-predicted[b]["mark_NLL"])).abs().max() for a,b in PAIRS]).max(),
                "M_Z_probability_range_max_absolute": torch.stack([predicted[a][m].abs().max()
                    for a in ("M", "Z") for m in ("copy_range", "repeat_range")]).max()}
            for key,value in local.items():checks[key] = max(checks[key], float(value))
            metrics = [predicted[a][m] for a in ARMS for m in ("repeat_BCE", "mark_NLL", "copy_range", "repeat_range")]
            metrics += [parts["mean"][:,0], parts["mean"][:,0].abs(),
                        (parts["residual"].square()*parts["weights"]).sum(1).sqrt()]
            metrics += [predicted[a]["repeat_BCE"]-predicted[b]["repeat_BCE"] for a,b in PAIRS]
            metrics = torch.stack(metrics, dim=1)
            entity = torch.zeros((len(ids), len(COLUMNS)), dtype=torch.float64, device=device)
            entity.index_add_(0, rows, metrics/mask.sum(1)[rows, None])
            blocks.append(entity.cpu().numpy());names.extend(data["entity_ids"][int(i)] for i in ids)
        values = np.concatenate(blocks)
        if not np.isfinite(values).all():raise FloatingPointError("nonfinite decomposition")
        groups[str(label)] = {"entities": len(values), "metrics": {name:summarize(values[:,j]) for j,name in enumerate(COLUMNS)}}
        arrays[f"label_{label}_metrics"] = values
        arrays[f"label_{label}_entity_ids"] = np.asarray(names,dtype=str)
    return {"groups":groups,"columns":list(COLUMNS),"mechanical_errors":checks,
            "entity_batch_size":batch_size,"test_accessed":False},arrays
