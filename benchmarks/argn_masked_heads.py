"""ibm_masked_heads_v1 (2026-11-08): head retraining with input substitution so the heads carry the fraud
signal on the label path. For fraud rows selected with probability p, the INPUT tokens of the non-label
columns are replaced by those of a random normal row of the same sequence; the loss targets stay original.
Everything else follows benchmarks.argn_output_heads (frozen backbone, label-first order, balanced weight).
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import get_cardinalities, SLEN_SUB_COLUMN_PREFIX
from mostlyai.engine._tabular.training import BatchCollator
from benchmarks.argn_output_heads import (LABEL, OUTPUT_COLUMNS, argn_column, label_first_order, build_model,
                                          head_parameter_names, _records)

RETRAINED = ("gap", "amount_or_numeric_value", "category", "use_chip")


def substitute_inputs(data, label_key, fraud_code, keys, p, generator):
    """Copy of `data` where, for fraud rows picked with probability p, tokens of `keys` come from a random
    normal row of the same sequence (padding excluded). Returns (inputs, number of substituted rows)."""
    out = dict(data)
    lab = data[label_key].squeeze(-1)
    slen_cols = [k for k in data if k.startswith(SLEN_SUB_COLUMN_PREFIX)]
    mask = torch.zeros_like(lab, dtype=torch.bool)
    for c in slen_cols:
        mask |= data[c].squeeze(-1) != 0
    fraud = lab.eq(fraud_code) & mask
    normal = mask & ~fraud
    if not fraud.any():
        return out, 0
    pick = fraud & (torch.rand(lab.shape, generator=generator, device=lab.device) < p)
    n_normal = normal.sum(1)
    pick &= n_normal.gt(0).unsqueeze(1)
    if not pick.any():
        return out, 0
    order = torch.argsort(normal.float(), dim=1, descending=True, stable=True)  # normal positions first
    r = torch.floor(torch.rand(lab.shape, generator=generator, device=lab.device) * n_normal.clamp(min=1).unsqueeze(1)).long()
    src_t = torch.gather(order, 1, r)
    for k in keys:
        x = data[k]
        src = torch.gather(x, 1, src_t.unsqueeze(-1))
        out[k] = torch.where(pick.unsqueeze(-1), src, x)
    return out, int(pick.sum())


def masked_losses(model, data, label_key, fraud_code, output_keys, fraud_weight, subst_keys, p, generator):
    inputs, n_sub = substitute_inputs(data, label_key, fraud_code, subst_keys, p, generator) if p > 0 else (data, 0)
    output, _ = model(inputs, mode="trn")
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
    return total, n_sub


def train_masked_heads(ws: Workspace, out_dir: Path, p: float, seed: int, device="cuda", max_epochs=100, patience=5,
                       batch_size=8, window=100, lr=1e-3, log=print, columns=RETRAINED):
    """Balanced head retraining with substitution probability p.
    Amendment 1 (2026-11-08): per-column early stopping on the substituted-context (p=1) validation CE of THAT
    column -- each head is an independent module, so each keeps its own best epoch; training stops when every
    column has exhausted `patience`. lr 3e-4 -> 1e-3. Real-context validation CE is logged next to it."""
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    ts = ws.tgt_stats.read(); card = get_cardinalities(ts); order = label_first_order(ts, card)
    model, ts, card = build_model(ws, device, order)
    label_key = argn_column(ts, LABEL) + "__cat"; fraud_code = int(ts["columns"][LABEL]["codes"]["1"])
    output_keys = [k for k in card if k.startswith(tuple(argn_column(ts, c) + "__" for c in columns))]
    subst_keys = [k for k in card if k.startswith(tuple(argn_column(ts, c) + "__" for c in OUTPUT_COLUMNS))]  # merchant included
    names = set(head_parameter_names(model, ts, list(columns)))
    for n, prm in model.named_parameters():
        prm.requires_grad_(n in names)
    params = [prm for n, prm in model.named_parameters() if n in names]
    trn = _records(list(ws.encoded_data_trn.fetch_all())[0]); val = _records(list(ws.encoded_data_val.fetch_all())[0])
    lab = np.concatenate([np.asarray(r[label_key]) for r in trn]); n_fraud = int((lab == fraud_code).sum()); n_norm = int(len(lab) - n_fraud)
    fraud_weight = float(n_norm / max(n_fraud, 1))
    trn_loader = DataLoader(trn, shuffle=True, batch_size=batch_size, collate_fn=BatchCollator(True, window, torch.device(device)))
    val_loader = DataLoader(val, shuffle=False, batch_size=batch_size // 2, collate_fn=BatchCollator(True, None, torch.device(device)))
    opt = torch.optim.Adam(params, lr=lr)
    col_keys = {c: [k for k in output_keys if k.startswith(argn_column(ts, c) + "__")] for c in columns}
    col_params = {c: [n for n in names if any(n.split("@")[-1].startswith(k) or n.endswith(k + ".weight") or n.endswith(k + ".bias") or k in n for k in col_keys[c])] for c in columns}
    assert all(col_params[c] for c in columns) and sum(len(v) for v in col_params.values()) == len(names), {c: len(v) for c, v in col_params.items()}
    best = {c: float("inf") for c in columns}; best_state = {}; best_epoch = {}; bad = {c: 0 for c in columns}; history = []
    out_dir.mkdir(parents=True, exist_ok=True); t0 = time.time()
    crit = nn.CrossEntropyLoss(reduction="none")

    def column_val(p_sub):
        vgen = torch.Generator(device=device); vgen.manual_seed(20261108)
        sums = {c: 0.0 for c in columns}; wsum = 0.0
        for batch in val_loader:
            inputs, _ = substitute_inputs(batch, label_key, fraud_code, subst_keys, p_sub, vgen) if p_sub > 0 else (batch, 0)
            output, _ = model(inputs, mode="trn")
            slen_cols = [k for k in batch if k.startswith(SLEN_SUB_COLUMN_PREFIX)]
            m = torch.zeros_like(batch[slen_cols[0]], dtype=torch.bool)
            for cc in slen_cols:
                m |= batch[cc] != 0
            m = m.squeeze(-1); fr = batch[label_key].squeeze(-1).eq(fraud_code)
            w = torch.where(fr, torch.full_like(m, fraud_weight, dtype=torch.float32), torch.ones_like(m, dtype=torch.float32)) * m
            for c in columns:
                ce = sum(crit(output[k].transpose(1, 2), batch[k].squeeze(2)) for k in col_keys[c])
                sums[c] += float((ce * w).sum())
            wsum += float(w.sum())
        return {c: sums[c] / max(wsum, 1e-6) for c in columns}

    for epoch in range(1, max_epochs + 1):
        model.train(); trn_losses = []; substituted = 0
        for batch in trn_loader:
            losses, n_sub = masked_losses(model, batch, label_key, fraud_code, output_keys, fraud_weight, subst_keys, p, gen)
            loss = losses.mean(); opt.zero_grad(); loss.backward(); opt.step(); trn_losses.append(loss.item()); substituted += n_sub
        model.eval()
        with torch.no_grad():
            v_real = column_val(0.0); v_sub = column_val(1.0)
        tl = float(np.mean(trn_losses))
        row = dict(epoch=epoch, trn_loss=tl, val_loss_real_context=sum(v_real.values()), val_loss_substituted_context=sum(v_sub.values()), substituted_rows=substituted, seconds=round(time.time() - t0, 1))
        row.update({f"val_subst_{c}": v_sub[c] for c in columns}); row.update({f"val_real_{c}": v_real[c] for c in columns}); history.append(row)
        log(f"mask{int(p * 100)} seed={seed} epoch={epoch} trn={tl:.4f} val_real={row['val_loss_real_context']:.4f} val_subst={row['val_loss_substituted_context']:.4f} "
            + " ".join(f"{c.split('_')[0]}={v_sub[c]:.3f}" for c in columns) + f" substituted={substituted}")
        sd = model.state_dict()
        for c in columns:
            if v_sub[c] < best[c] - 1e-5:
                best[c], bad[c], best_epoch[c] = v_sub[c], 0, epoch
                for n in col_params[c]:
                    best_state[n] = sd[n].detach().cpu().clone()
            else:
                bad[c] += 1
        if all(bad[c] >= patience for c in columns):
            break
    torch.save(best_state, out_dir / "heads.pt")
    pd.DataFrame(history).to_csv(out_dir / "learning.csv", index=False)
    meta = dict(mode=f"mask{int(p * 100)}", substitution_probability=p, seed=seed, fraud_weight=fraud_weight, trn_events=int(len(lab)), trn_fraud_events=n_fraud,
                epochs=len(history), best_epoch_by_column=best_epoch, best_val_subst_by_column=best, selection="per-column best substituted-context val CE (amendment 1)",
                trainable_parameters=int(sum(prm.numel() for prm in params)), total_parameters=int(sum(prm.numel() for prm in model.parameters())),
                column_order=order, output_sub_columns=output_keys, substituted_sub_columns=subst_keys, retrained_columns=list(columns), head_keys=sorted(names),
                window=window, val_window=None, batch_size=batch_size, lr=lr, patience=patience, max_epochs=max_epochs, seconds=round(time.time() - t0, 1))
    (out_dir / "TRAIN.json").write_text(json.dumps(meta, indent=1))
    return meta
