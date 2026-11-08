"""ibm_routed_heads_v1 (2026-11-08): label-routed output heads on the frozen native mostlyai-engine SequentialModel.

Normal rows keep the native regressor+predictor of every output column; fraud rows are routed to a separate
copy (`fraud_regressors` / `fraud_predictors`, initialised from the native weights) that is trained on fraud
rows only, without class weights. Routing is by the row's label code: in training the real label, at
generation the label just sampled for that row (label-first order). All native parameters stay frozen.
Registered in docs/ibm_routed_heads_v1/PROTOCOL.md.
"""
import copy, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import get_cardinalities, SLEN_SUB_COLUMN_PREFIX
from mostlyai.engine._tabular.training import BatchCollator
from benchmarks.argn_output_heads import LABEL, OUTPUT_COLUMNS, argn_column, label_first_order, build_model, _records
from benchmarks.argn_masked_heads import substitute_inputs

ROUTED = ("gap", "amount_or_numeric_value", "category", "use_chip")


def attach_routed_heads(model, stats, columns=ROUTED, label=LABEL):
    """Add fraud-only copies of the regressor+predictor of every sub-column of `columns` and install the routing
    hooks. Must be called AFTER the native weights are loaded (the engine's loader is strict and silent)."""
    label_key = argn_column(stats, label) + "__cat"; fraud_code = int(stats["columns"][label]["codes"]["1"])
    sub_cols = [k for k in model.tgt_cardinalities if k.startswith(tuple(argn_column(stats, c) + "__" for c in columns))]
    model.fraud_regressors = nn.ModuleDict({k: copy.deepcopy(model.regressors.get(k)) for k in sub_cols})
    model.fraud_predictors = nn.ModuleDict({k: copy.deepcopy(model.predictors.predictors[k]) for k in sub_cols})
    model.route = None

    def set_route(module, args):  # label embedder pre-hook: (B,T,1) real codes in trn, (B,1) sampled codes in gen
        r = torch.as_tensor(args[0]).eq(fraud_code)
        model.route = r.unsqueeze(-1) if r.dim() == 2 else r

    def route_regressor(module, args, output):
        regressor_in, sub_col = args
        if sub_col not in model.fraud_regressors or model.route is None:
            return output
        x = torch.cat(regressor_in, dim=-1)
        for layer in model.fraud_regressors[sub_col]:
            x = module.dropout(x)
            x = layer(x)
        return torch.where(model.route, nn.ReLU()(x), output)

    def route_predictor(module, args, output):
        x, sub_col = args
        if sub_col not in model.fraud_predictors or model.route is None:
            return output
        return torch.where(model.route, model.fraud_predictors[sub_col](x), output)

    model.embedders.get(label_key).register_forward_pre_hook(set_route)
    model.regressors.register_forward_hook(route_regressor)
    model.predictors.register_forward_hook(route_predictor)
    model.routed_sub_columns = sub_cols
    return sub_cols


def fraud_head_names(model):
    return [n for n, _ in model.named_parameters() if n.startswith(("fraud_regressors.", "fraud_predictors."))]


def routed_model(ws: Workspace, device, head_state=None, columns=ROUTED):
    ts = ws.tgt_stats.read(); card = get_cardinalities(ts); order = label_first_order(ts, card)
    model, ts, card = build_model(ws, device, order)
    attach_routed_heads(model, ts, columns)
    if head_state is not None:
        missing, unexpected = model.load_state_dict({k: v.to(model.device) for k, v in head_state.items()}, strict=False)
        assert not unexpected, unexpected
        assert len(missing) == len(model.state_dict()) - len(head_state), (len(missing), len(head_state))
    return model, ts, card, order


def _valid_mask(data):
    slen_cols = [k for k in data if k.startswith(SLEN_SUB_COLUMN_PREFIX)]
    mask = torch.zeros_like(data[slen_cols[0]], dtype=torch.bool)
    for c in slen_cols:
        mask |= data[c] != 0
    return mask.squeeze(-1)


def train_routed_heads(ws: Workspace, out_dir: Path, substitute: bool, seed: int, device="cuda", max_epochs=100, patience=5,
                       batch_size=8, window=100, lr=1e-3, log=print, columns=ROUTED):
    """Fraud-only, unweighted CE on the routed heads; per-column early stopping on the internal-validation fraud-row
    CE under the arm's own context condition (real or substituted). Both validation conditions are logged."""
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    model, ts, card, order = routed_model(ws, device, None, columns)
    label_key = argn_column(ts, LABEL) + "__cat"; fraud_code = int(ts["columns"][LABEL]["codes"]["1"])
    col_keys = {c: [k for k in card if k.startswith(argn_column(ts, c) + "__")] for c in columns}
    subst_keys = [k for k in card if k.startswith(tuple(argn_column(ts, c) + "__" for c in OUTPUT_COLUMNS))]
    names = set(fraud_head_names(model))
    for n, prm in model.named_parameters():
        prm.requires_grad_(n in names)
    params = [prm for n, prm in model.named_parameters() if n in names]
    col_params = {c: [n for n in names if any(k in n for k in col_keys[c])] for c in columns}
    assert all(col_params[c] for c in columns) and sum(len(v) for v in col_params.values()) == len(names)
    trn = _records(list(ws.encoded_data_trn.fetch_all())[0]); val = _records(list(ws.encoded_data_val.fetch_all())[0])
    lab = np.concatenate([np.asarray(r[label_key]) for r in trn]); n_fraud = int((lab == fraud_code).sum())
    trn_loader = DataLoader(trn, shuffle=True, batch_size=batch_size, collate_fn=BatchCollator(True, window, torch.device(device)))
    val_loader = DataLoader(val, shuffle=False, batch_size=batch_size // 2, collate_fn=BatchCollator(True, None, torch.device(device)))
    opt = torch.optim.Adam(params, lr=lr); crit = nn.CrossEntropyLoss(reduction="none")
    p_train = 1.0 if substitute else 0.0

    def fraud_ce(batch, p_sub, generator):
        inputs, n_sub = substitute_inputs(batch, label_key, fraud_code, subst_keys, p_sub, generator) if p_sub > 0 else (batch, 0)
        output, _ = model(inputs, mode="trn")
        fr = batch[label_key].squeeze(-1).eq(fraud_code) & _valid_mask(batch)
        ce = {c: sum(crit(output[k].transpose(1, 2), batch[k].squeeze(2)) for k in col_keys[c]) for c in columns}
        return ce, fr, n_sub

    def column_val(p_sub):
        vgen = torch.Generator(device=device); vgen.manual_seed(20261108)
        sums = {c: 0.0 for c in columns}; n = 0
        for batch in val_loader:
            ce, fr, _ = fraud_ce(batch, p_sub, vgen)
            for c in columns:
                sums[c] += float(ce[c][fr].sum())
            n += int(fr.sum())
        return {c: sums[c] / max(n, 1) for c in columns}, n

    best = {c: float("inf") for c in columns}; best_state = {}; best_epoch = {}; bad = {c: 0 for c in columns}; history = []
    out_dir.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train(); trn_losses = []; substituted = 0; seen = 0
        for batch in trn_loader:
            ce, fr, n_sub = fraud_ce(batch, p_train, gen)
            if not fr.any():
                continue
            loss = sum(ce[c][fr].mean() for c in columns)
            opt.zero_grad(); loss.backward(); opt.step(); trn_losses.append(loss.item()); substituted += n_sub; seen += int(fr.sum())
        model.eval()
        with torch.no_grad():
            v_real, n_val = column_val(0.0); v_sub, _ = column_val(1.0)
        crit_val = v_sub if substitute else v_real
        tl = float(np.mean(trn_losses)) if trn_losses else float("nan")
        row = dict(epoch=epoch, trn_loss=tl, trn_fraud_rows=seen, substituted_rows=substituted, val_fraud_rows=n_val,
                   val_real_total=sum(v_real.values()), val_subst_total=sum(v_sub.values()), seconds=round(time.time() - t0, 1))
        row.update({f"val_real_{c}": v_real[c] for c in columns}); row.update({f"val_subst_{c}": v_sub[c] for c in columns}); history.append(row)
        log(f"{'routed_subst' if substitute else 'routed'} seed={seed} epoch={epoch} trn={tl:.4f} val_real={row['val_real_total']:.4f} val_subst={row['val_subst_total']:.4f} "
            + " ".join(f"{c.split('_')[0]}={crit_val[c]:.3f}" for c in columns) + f" fraud_rows={seen}")
        sd = model.state_dict()
        for c in columns:
            if crit_val[c] < best[c] - 1e-5:
                best[c], bad[c], best_epoch[c] = crit_val[c], 0, epoch
                for n in col_params[c]:
                    best_state[n] = sd[n].detach().cpu().clone()
            else:
                bad[c] += 1
        if all(bad[c] >= patience for c in columns):
            break
    torch.save(best_state, out_dir / "heads.pt")
    pd.DataFrame(history).to_csv(out_dir / "learning.csv", index=False)
    meta = dict(mode="routed_subst" if substitute else "routed", substitute=substitute, seed=seed, fraud_weight=None, loss="fraud rows only, unweighted",
                trn_events=int(len(lab)), trn_fraud_events=n_fraud, val_fraud_events=n_val, epochs=len(history), best_epoch_by_column=best_epoch,
                best_val_by_column=best, selection="per-column best fraud-row val CE under the arm's own context condition",
                trainable_parameters=int(sum(prm.numel() for prm in params)), total_parameters=int(sum(prm.numel() for prm in model.parameters())),
                column_order=order, routed_sub_columns=model.routed_sub_columns, substituted_sub_columns=subst_keys, routed_columns=list(columns),
                head_keys=sorted(names), window=window, batch_size=batch_size, lr=lr, patience=patience, max_epochs=max_epochs, seconds=round(time.time() - t0, 1))
    (out_dir / "TRAIN.json").write_text(json.dumps(meta, indent=1))
    return meta


@torch.no_grad()
def evaluate_routed(ws: Workspace, head_state=None, device="cuda", batch_size=4, columns=ROUTED):
    """Per-column internal-validation CE by row label for the routed model (native copies when head_state is None)."""
    model, ts, card, _ = routed_model(ws, device, head_state, columns); model.eval()
    label_key = argn_column(ts, LABEL) + "__cat"; fraud_code = int(ts["columns"][LABEL]["codes"]["1"])
    keys = {c: [k for k in card if k.startswith(argn_column(ts, c) + "__")] for c in columns}
    val = _records(list(ws.encoded_data_val.fetch_all())[0])
    loader = DataLoader(val, shuffle=False, batch_size=batch_size, collate_fn=BatchCollator(True, None, torch.device(device)))
    crit = nn.CrossEntropyLoss(reduction="none"); sums = {(c, s): 0.0 for c in columns for s in ("normal", "fraud")}; counts = {"normal": 0, "fraud": 0}
    for data in loader:
        output, _ = model(data, mode="trn"); mask = _valid_mask(data)
        fraud = data[label_key].squeeze(-1).eq(fraud_code) & mask; normal = mask & ~fraud
        for c in columns:
            ce = sum(crit(output[k].transpose(1, 2), data[k].squeeze(2)) for k in keys[c])
            sums[(c, "fraud")] += float(ce[fraud].sum()); sums[(c, "normal")] += float(ce[normal].sum())
        counts["fraud"] += int(fraud.sum()); counts["normal"] += int(normal.sum())
    return pd.DataFrame([dict(column=c, split=s, events=counts[s], mean_ce=sums[(c, s)] / max(counts[s], 1)) for c in columns for s in ("normal", "fraud")])
