"""Head-only retraining of the frozen native mostlyai-engine SequentialModel with a fixed label-first order.

Only the regressor layers and predictors of the output columns are optimised; embedders, column embedders,
history LSTM, context compressor, attention, SLEN/SIDX/SDEC heads and the label head keep the native weights.
Registered 2026-11-08 in docs/ibm_output_heads_v1/PROTOCOL.md.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import (get_cardinalities, get_sequence_length_stats, get_ctx_sequence_length, get_argn_name,
                                     SLEN_SUB_COLUMN_PREFIX, SIDX_SUB_COLUMN_PREFIX, SDEC_SUB_COLUMN_PREFIX)
from mostlyai.engine._tabular.argn import SequentialModel
from mostlyai.engine._tabular.common import load_model_weights
from mostlyai.engine._tabular.generation import _resolve_gen_column_order
from mostlyai.engine._tabular.training import BatchCollator

LABEL = "event_is_fraud"
OUTPUT_COLUMNS = ["gap", "receiver_or_mark", "amount_or_numeric_value", "category", "use_chip"]


def argn_column(stats, column):
    s = stats["columns"][column]
    return get_argn_name(s["argn_processor"], s["argn_table"], s["argn_column"])


def label_first_order(stats, cardinalities, label=LABEL):
    order = list(_resolve_gen_column_order(column_stats=stats["columns"], cardinalities=cardinalities, rebalancing=None,
                                           imputation=None, sample_seed=None, fairness=None))
    lab = argn_column(stats, label); order.remove(lab)
    fixed = [c for c in order if c.startswith((SLEN_SUB_COLUMN_PREFIX, SIDX_SUB_COLUMN_PREFIX, SDEC_SUB_COLUMN_PREFIX))]
    return fixed + [lab] + [c for c in order if c not in fixed]


def build_model(ws: Workspace, device, column_order):
    ts = ws.tgt_stats.read(); cs = ws.ctx_stats.read(); cfg = ws.model_tabular_configs.read()
    card = get_cardinalities(ts); lengths = get_sequence_length_stats(ts)
    model = SequentialModel(tgt_cardinalities=card, tgt_seq_len_median=lengths["median"], tgt_seq_len_max=lengths["max"],
                            ctx_cardinalities=get_cardinalities(cs), ctxseq_len_median=get_ctx_sequence_length(cs, key="median"),
                            model_size=cfg["model_units"], column_order=column_order, device=torch.device(device))
    load_model_weights(model=model, path=ws.model_tabular_weights_path, device=torch.device(device))
    model.to(torch.device(device))
    return model, ts, card


def head_parameter_names(model, stats, columns=OUTPUT_COLUMNS):
    """State-dict keys of the trainable heads: regressors + predictors of every sub-column of `columns`."""
    prefixes = tuple(argn_column(stats, c) + "__" for c in columns)
    names = []
    for key, _ in model.named_parameters():
        # engine names: regressors.regressors.regressor@tgt:t4/c4__cat.0.weight / predictors.predictors.tgt:t4/c4__cat.weight
        for module in ("regressors.regressors.regressor@", "predictors.predictors."):
            if key.startswith(module) and key[len(module):].startswith(prefixes):
                names.append(key)
    assert any(k.startswith("regressors.") for k in names), "regressor layers not matched"
    return names


def _records(path):
    return pd.read_parquet(path).to_dict("records")


def weighted_losses(model, data, label_key, fraud_code, output_keys, fraud_weight):
    """Per-sample weighted CE over the output sub-columns; padding masked as in the engine."""
    output, _ = model(data, mode="trn")
    slen_cols = [k for k in data if k.startswith(SLEN_SUB_COLUMN_PREFIX)]
    mask = torch.zeros_like(data[slen_cols[0]], dtype=torch.bool)
    for c in slen_cols:
        mask |= data[c] != 0
    mask = mask.squeeze(-1)
    fraud = data[label_key].squeeze(-1).eq(fraud_code)
    w = torch.where(fraud, torch.full_like(mask, fraud_weight, dtype=torch.float32), torch.ones_like(mask, dtype=torch.float32)) * mask
    crit = nn.CrossEntropyLoss(reduction="none")
    total = 0.0
    for col in output_keys:
        ce = crit(output[col].transpose(1, 2), data[col].squeeze(2))
        total = total + torch.sum(ce * w, dim=1) / torch.clamp(torch.sum(w, dim=1), min=1e-6)
    return total, fraud & mask


def train_heads(ws: Workspace, out_dir: Path, mode: str, seed: int, device="cuda", max_epochs=30, patience=5,
                batch_size=8, window=100, lr=1e-3, log=print, val_window=None, columns=OUTPUT_COLUMNS):
    """Amendment 1 (2026-11-08): validation uses FULL sequences (val_window=None) so every fraud event of the
    100 internal-validation customers enters each evaluation; windowed validation saw 0-28 fraud events per epoch
    and made the balanced criterion unusable for early stopping. patience 3 -> 5."""
    assert mode in {"natural", "balanced"}
    torch.manual_seed(seed); np.random.seed(seed)
    ts = ws.tgt_stats.read(); card = get_cardinalities(ts)
    order = label_first_order(ts, card)
    model, ts, card = build_model(ws, device, order)
    label_key = argn_column(ts, LABEL) + "__cat"; fraud_code = int(ts["columns"][LABEL]["codes"]["1"])
    output_keys = [k for k in card if k.startswith(tuple(argn_column(ts, c) + "__" for c in columns))]
    names = set(head_parameter_names(model, ts, columns))
    for n, p in model.named_parameters():
        p.requires_grad_(n in names)
    params = [p for n, p in model.named_parameters() if n in names]
    trn = _records(list(ws.encoded_data_trn.fetch_all())[0]); val = _records(list(ws.encoded_data_val.fetch_all())[0])
    # class weight from the trn partition (events)
    lab = np.concatenate([np.asarray(r[label_key]) for r in trn]); n_fraud = int((lab == fraud_code).sum()); n_norm = int(len(lab) - n_fraud)
    fraud_weight = float(n_norm / max(n_fraud, 1)) if mode == "balanced" else 1.0
    collate = BatchCollator(True, window, torch.device(device)); val_collate = BatchCollator(True, val_window, torch.device(device))
    trn_loader = DataLoader(trn, shuffle=True, batch_size=batch_size, collate_fn=collate)
    val_loader = DataLoader(val, shuffle=False, batch_size=batch_size // 2, collate_fn=val_collate)
    opt = torch.optim.Adam(params, lr=lr)
    best, best_state, bad, history = float("inf"), None, 0, []
    out_dir.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train(); trn_losses = []
        for batch in trn_loader:
            losses, _ = weighted_losses(model, batch, label_key, fraud_code, output_keys, fraud_weight)
            loss = losses.mean(); opt.zero_grad(); loss.backward(); opt.step(); trn_losses.append(loss.item())
        model.eval(); val_losses = []; fraud_events = 0
        with torch.no_grad():
            for batch in val_loader:
                losses, fr = weighted_losses(model, batch, label_key, fraud_code, output_keys, fraud_weight)
                val_losses.append(losses.mean().item()); fraud_events += int(fr.sum())
        tl, vl = float(np.mean(trn_losses)), float(np.mean(val_losses))
        history.append(dict(epoch=epoch, trn_loss=tl, val_loss=vl, val_fraud_events=fraud_events, seconds=round(time.time() - t0, 1)))
        log(f"{mode} seed={seed} epoch={epoch} trn={tl:.4f} val={vl:.4f} fraud_events_val={fraud_events}")
        if vl < best - 1e-5:
            best, bad = vl, 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}
        else:
            bad += 1
            if bad >= patience:
                break
    torch.save(best_state, out_dir / "heads.pt")
    pd.DataFrame(history).to_csv(out_dir / "learning.csv", index=False)
    meta = dict(mode=mode, seed=seed, fraud_weight=fraud_weight, trn_events=int(len(lab)), trn_fraud_events=n_fraud, epochs=len(history),
                best_val_loss=best, best_epoch=int(min(history, key=lambda h: h["val_loss"])["epoch"]), trainable_parameters=int(sum(p.numel() for p in params)),
                total_parameters=int(sum(p.numel() for p in model.parameters())), column_order=order, output_sub_columns=output_keys,
                head_keys=sorted(names), retrained_columns=list(columns), window=window, val_window=val_window, batch_size=batch_size, lr=lr, patience=patience, max_epochs=max_epochs, seconds=round(time.time() - t0, 1))
    (out_dir / "TRAIN.json").write_text(json.dumps(meta, indent=1))
    return meta


@torch.no_grad()
def evaluate_heads(ws: Workspace, head_state=None, device="cuda", batch_size=4, columns=OUTPUT_COLUMNS, label=LABEL):
    """Per-column validation CE (full sequences, label-first order) for native heads or a retrained head state,
    split by label of the row. Used for the epoch-0 baseline and per-column attribution (amendment 2)."""
    ts = ws.tgt_stats.read(); card = get_cardinalities(ts); order = label_first_order(ts, card)
    model, ts, card = build_model(ws, device, order)
    if head_state is not None:
        model.load_state_dict({k: v.to(model.device) for k, v in head_state.items()}, strict=False)
    model.eval()
    label_key = argn_column(ts, label) + "__cat"; fraud_code = int(ts["columns"][label]["codes"]["1"])
    keys = {c: [k for k in card if k.startswith(argn_column(ts, c) + "__")] for c in columns}
    val = _records(list(ws.encoded_data_val.fetch_all())[0])
    loader = DataLoader(val, shuffle=False, batch_size=batch_size, collate_fn=BatchCollator(True, None, torch.device(device)))
    crit = nn.CrossEntropyLoss(reduction="none"); sums = {(c, s): 0.0 for c in columns for s in ("normal", "fraud")}; counts = {"normal": 0, "fraud": 0}
    for data in loader:
        output, _ = model(data, mode="trn")
        slen_cols = [k for k in data if k.startswith(SLEN_SUB_COLUMN_PREFIX)]
        mask = torch.zeros_like(data[slen_cols[0]], dtype=torch.bool)
        for c in slen_cols:
            mask |= data[c] != 0
        mask = mask.squeeze(-1); fraud = data[label_key].squeeze(-1).eq(fraud_code) & mask; normal = mask & ~fraud
        for c in columns:
            ce = sum(crit(output[k].transpose(1, 2), data[k].squeeze(2)) for k in keys[c])
            sums[(c, "fraud")] += float(ce[fraud].sum()); sums[(c, "normal")] += float(ce[normal].sum())
        counts["fraud"] += int(fraud.sum()); counts["normal"] += int(normal.sum())
    rows = [dict(column=c, split=s, events=counts[s], mean_ce=sums[(c, s)] / max(counts[s], 1)) for c in columns for s in ("normal", "fraud")]
    return pd.DataFrame(rows)
