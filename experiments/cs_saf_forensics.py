"""Train-only fixed-checkpoint response and local-gradient diagnostics."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

ROUTE_NAMES = ("interaction_weight", "context_projection.weight",
               "context_projection.bias", "gap_projection.weight")


def route_parameters(model):
    named = dict(model.named_parameters())
    return tuple(named[name] for name in ROUTE_NAMES)


def flat_gradient(value, parameters, *, retain_graph=False):
    return torch.cat([x.reshape(-1) for x in torch.autograd.grad(
        value, parameters, retain_graph=retain_graph)]).detach().cpu().double().numpy()


def cosine(a, b):
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a, b)/denominator) if denominator > 0 else None


def local_direction(response_gradient, loss_gradient):
    return {"unscaled_descent_derivative": float(-np.dot(response_gradient, loss_gradient)),
            "negative_cosine": None if cosine(response_gradient, loss_gradient) is None
                else -cosine(response_gradient, loss_gradient)}


def fixed_feature_curves(model, context, previous):
    """Keep encoders fixed; differentiate only the direct rank-32 route."""
    context = context.detach()
    gaps = torch.tensor(model.support.representatives, device=context.device, dtype=context.dtype)
    embedding = model.gap_route(model._support_code(gaps)).detach()
    base = model.copy_base(context).detach()
    u = torch.tanh(model.context_projection(context))*model.interaction_weight
    v = torch.tanh(model.gap_projection(embedding))
    interaction = u@v.T/math.sqrt(32)
    logits = base+interaction
    fresh = model.new_mark_head(context).detach()
    fresh[:, :3] = -torch.inf
    previous_fresh = fresh.softmax(-1).gather(1, previous[:, None])
    copy = logits.sigmoid()
    repeat = copy+(1-copy)*previous_fresh
    return logits, copy, repeat, previous_fresh[:, 0], base[:, 0], interaction


def repeat_nll(logits, previous_fresh, observed_repeat):
    log_repeat = torch.logaddexp(F.logsigmoid(logits),
                                F.logsigmoid(-logits)+previous_fresh.log())
    log_nonrepeat = F.logsigmoid(-logits)+torch.log1p(-previous_fresh.clamp(max=1-1e-7))
    return -torch.where(observed_repeat, log_repeat, log_nonrepeat)


def analyze_model(model, data, cfg, device):
    """Return compact summaries and arrays; no optimizer or parameter writes."""
    model.eval()
    parameters = route_parameters(model)
    dimensions = sum(p.numel() for p in parameters)
    groups, arrays, loss_gradients, range_gradients = {}, {}, {}, {}
    total_events = int(data["lengths"].sum())
    bins_count = len(model.support.representatives)
    for label in (0, 1):
        ids_all = torch.where(data["codes"] == label+3)[0]
        n = len(ids_all)
        transitions = int((data["lengths"][ids_all]-1).sum())
        all_gaps = data["gap"][ids_all]
        all_mask = data["valid_mask"][ids_all].clone(); all_mask[:, 0] = False
        # Use the trained model's float32 boundary convention, including ties.
        observed_bins = (model._support_code(all_gaps[all_mask])-3).numpy()
        counts = np.bincount(observed_bins, minlength=bins_count)
        cumulative = np.cumsum(counts)/counts.sum()
        lo, hi = [int(np.searchsorted(cumulative, q)) for q in cfg["central_observed_gap_mass"]]
        if hi <= lo:
            raise ValueError("central interval requires at least two occupied bins")
        g_loss, g_range = np.zeros(dimensions), np.zeros(dimensions)
        entity_rows, ids_text = [], []
        curve_sum = np.zeros(bins_count)
        extrema_min, extrema_max = np.zeros(bins_count), np.zeros(bins_count)
        bands = [{"first": a, "last": b, "transitions": 0, "copy_sum": 0., "repeat_sum": 0.}
                 for a, b in cfg["position_bands"]]
        relation_error = 0.
        for start in range(0, n, cfg["entity_batch_size"]):
            ids = ids_all[start:start+cfg["entity_batch_size"]]
            inputs = {k: data[k][ids].to(device) for k in ("gap", "receiver", "numeric_value", "valid_mask")}
            inputs["static_categorical"] = (data["codes"][ids].to(device),)
            with torch.no_grad():
                context = model.context(model.encoder(**inputs), inputs["static_categorical"])
            mask = inputs["valid_mask"].clone(); mask[:, 0] = False
            rows, positions = torch.where(mask)
            lengths = mask.sum(1)
            weight = 1/lengths[rows].to(context.dtype)
            previous = inputs["receiver"].roll(1, dims=1)[mask]
            logits, q, p, fresh, base, interaction = fixed_feature_curves(model, context[mask], previous)
            qr = q.max(1).values-q.min(1).values
            pr = p.max(1).values-p.min(1).values
            bins = model._support_code(inputs["gap"][mask])-3
            actual_logits = logits.gather(1, bins[:, None])[:, 0]
            observed = inputs["receiver"][mask] == previous
            bce = repeat_nll(actual_logits, fresh, observed)
            g_loss += flat_gradient(bce.sum()/transitions, parameters, retain_graph=True)
            g_range += flat_gradient((qr*weight).sum()/n, parameters)
            with torch.no_grad():
                central = q[:, lo:hi+1].max(1).values-q[:, lo:hi+1].min(1).values
                logit_range = interaction.max(1).values-interaction.min(1).values
                actual_repeat = p.gather(1, bins[:, None])[:, 0]
                metrics = torch.stack((qr, pr, central, logit_range, base.sigmoid(), fresh,
                    q[:, 0]-q[:, -1], actual_repeat, observed.float(), bce), dim=1)
                entity = torch.zeros((len(ids), metrics.shape[1]), device=device)
                entity.index_add_(0, rows, metrics*weight[:, None])
                entity_rows.append(entity.cpu().numpy())
                ids_text.extend(data["entity_ids"][int(i)] for i in ids)
                curve_sum += (q*weight[:, None]).sum(0).cpu().double().numpy()
                extrema_min += np.bincount(q.argmin(1).cpu().numpy(), weights=weight.cpu().numpy(), minlength=bins_count)
                extrema_max += np.bincount(q.argmax(1).cpu().numpy(), weights=weight.cpu().numpy(), minlength=bins_count)
                relation_error = max(relation_error, float((pr-(1-fresh)*qr).abs().max()))
                for band in bands:
                    selected = (positions >= band["first"]) & (positions <= band["last"])
                    band["transitions"] += int(selected.sum())
                    band["copy_sum"] += float(qr[selected].sum())
                    band["repeat_sum"] += float(pr[selected].sum())
        entity = np.concatenate(entity_rows).astype(np.float64)
        columns = ("copy_range", "repeat_range", "central_copy_range", "interaction_logit_range",
                   "base_copy_probability", "previous_fresh_probability", "short_minus_long_copy",
                   "predicted_repeat", "observed_repeat", "repeat_nll")
        summary = {"entities": n, "transitions": transitions,
                   **{"mean_"+name: float(entity[:, j].mean()) for j, name in enumerate(columns)},
                   "copy_range_entity_quantiles": {str(q): float(np.quantile(entity[:, 0], q)) for q in (.1,.5,.9,.95,1.)},
                   "fraction_entities_copy_range_above_005": float((entity[:, 0] > .05).mean()),
                   "central_bin_interval_inclusive": [lo, hi], "observed_bin_counts": counts.tolist(),
                   "mean_copy_curve": (curve_sum/n).tolist(),
                   "argmin_entity_weighted_frequencies": (extrema_min/n).tolist(),
                   "argmax_entity_weighted_frequencies": (extrema_max/n).tolist(),
                   "repeat_range_identity_max_absolute_error": relation_error,
                   "conditional_repeat_gradient_norm": float(np.linalg.norm(g_loss)),
                   "response_gradient_norm": float(np.linalg.norm(g_range)),
                   "global_base_route_coefficient": transitions/total_events,
                   "global_B1_route_coefficient": transitions/total_events+.5}
        summary["position_bands_transition_means"] = [
            {"first": b["first"], "last": b["last"], "transitions": b["transitions"],
             "copy_range": b["copy_sum"]/b["transitions"], "repeat_range": b["repeat_sum"]/b["transitions"]}
            for b in bands if b["transitions"]]
        groups[str(label)] = summary
        arrays.update({f"label_{label}_entity_metrics": entity,
                       f"label_{label}_entity_ids": np.array(ids_text, dtype=str),
                       f"label_{label}_conditional_loss_gradient": g_loss,
                       f"label_{label}_range_gradient": g_range})
        loss_gradients[label], range_gradients[label] = g_loss, g_range
    directions = {}
    for target in (0, 1):
        for source in (0, 1):
            value = local_direction(range_gradients[target], loss_gradients[source])
            coefficient = groups[str(source)]["global_B1_route_coefficient" if model.balanced else "global_base_route_coefficient"]
            value["current_candidate_global_objective_derivative"] = value["unscaled_descent_derivative"]*coefficient
            directions[f"loss_label_{source}_to_response_label_{target}"] = value
    return {"groups": groups, "route_gradient_cosine_between_labels": cosine(loss_gradients[0], loss_gradients[1]),
            "local_route_only_directions": directions, "route_parameter_names": list(ROUTE_NAMES),
            "route_parameter_shapes": [list(p.shape) for p in parameters],
            "entity_metric_columns": list(columns), "train_events": total_events,
            "train_transitions": sum(g["transitions"] for g in groups.values()),
            "new_training": False, "loaded_content_splits": ["train"],
            "validation_content_accessed": False, "test_accessed": False}, arrays
