"""ibm_state_first_v1 (2026-11-09): "State-First" generation on the frozen native mostlyai-engine SequentialModel.

  p(sequence) = prod_t p(state_t | state history) * p(outputs_t | state_t, history), with separate parameters, state first.

Two generalisations of the registered minimal instance (transition table + label-routed fraud head):
  * learned state process  -- a classifier on label-history features replaces the (prev label, run age, ever) table;
  * K-state routing        -- fraud rows are routed to an ONSET head (previous label normal / first event) or a
                              CONTINUATION head (previous label fraud) instead of one fraud head.
All native parameters stay frozen. Registered in docs/ibm_state_first_v1/PROTOCOL.md.
"""
from contextlib import contextmanager
import copy, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import get_cardinalities, get_argn_name, SLEN_SUB_COLUMN_PREFIX, SIDX_SUB_COLUMN_PREFIX, SDEC_SUB_COLUMN_PREFIX
from mostlyai.engine._tabular.training import BatchCollator
from benchmarks.argn_output_heads import LABEL, OUTPUT_COLUMNS, argn_column, label_first_order, build_model, _records
from benchmarks.argn_masked_heads import substitute_inputs
from benchmarks.state_process_features import MAX_AGE, FEATURES, MEMORY, label_history_features, memory_features, table_prob_fn

ROUTED = ("gap", "amount_or_numeric_value", "category", "use_chip")
STATES = ("onset", "continuation")  # fraud-row states; index 1 and 2 in the route tensor, 0 = normal


# ------------------------------------------------------------------------------------------------ state process
CANDIDATES = {  # amendment 1 (2026-11-09): the default HGB diverged (train accuracy below the base rate, P(fraud | prev fraud) = 0)
    "hgb_default": dict(max_iter=200),
    "hgb_l2_1": dict(learning_rate=0.05, l2_regularization=1.0, min_samples_leaf=50, max_iter=200),
    "hgb_l2_10_d4": dict(learning_rate=0.02, l2_regularization=10.0, min_samples_leaf=100, max_iter=300, max_depth=4),
}


def fit_state_process(train, seed=20261109, label=LABEL, holdout_fraction=0.1):
    """Label-history classifier. Candidate selected by log-loss on a 10% customer hold-out *of the train split* (development
    is never used for selection), then refit on all train customers."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = train.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    y = pd.to_numeric(d[label], errors="coerce").fillna(0).astype(int).eq(1).astype(int)
    X = label_history_features(y, d.entity_id); yv = y.to_numpy()
    ents = np.sort(d.entity_id.unique()); rng = np.random.default_rng(seed); hold = set(rng.choice(ents, size=int(round(len(ents) * holdout_fraction)), replace=False).tolist())
    hm = d.entity_id.isin(hold).to_numpy()
    selection = {}
    for name, kw in CANDIDATES.items():
        c = HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **kw).fit(X[~hm], yv[~hm])
        p = np.clip(c.predict_proba(X[hm])[:, 1], 1e-6, 1 - 1e-6); yh = yv[hm]
        ll = -(yh * np.log(p) + (1 - yh) * np.log(1 - p))
        selection[name] = dict(holdout_logloss=float(ll.mean()), holdout_logloss_fraud_rows=float(ll[yh == 1].mean()), holdout_frauds=int(yh.sum()),
                               p_fraud_given_prev_fraud=float(c.predict_proba(X[hm][X[hm][:, 0] == 1])[:, 1].mean()) if (X[hm][:, 0] == 1).any() else None)
    best = min(selection, key=lambda k: selection[k]["holdout_logloss"])
    clf = HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **CANDIDATES[best]).fit(X, yv)
    return clf, dict(features=FEATURES, rows=int(len(y)), frauds=int(y.sum()), seed=seed, classifier=f"HistGradientBoostingClassifier({CANDIDATES[best]})",
                     selected=best, candidates=CANDIDATES, selection_on_train_holdout=selection, holdout_customers=len(hold))


def label_logloss(frame, p_fn, label=LABEL):
    """Teacher-forced log-loss of a state process on real label sequences. p_fn(X) -> P(fraud)."""
    d = frame.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    y = pd.to_numeric(d[label], errors="coerce").fillna(0).astype(int).eq(1).astype(int).to_numpy()
    X = label_history_features(pd.Series(y), d.entity_id)
    p = np.clip(p_fn(X), 1e-6, 1 - 1e-6)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return dict(logloss=float(ll.mean()), logloss_fraud_rows=float(ll[y == 1].mean()), logloss_normal_rows=float(ll[y == 0].mean()), rows=int(len(y)), frauds=int(y.sum()))




def update_memory(memory, lab):
    """memory (B,6) long, lab (B,) long {0,1} -> new memory after this row."""
    n, prev, age, ever, nf, since = [memory[:, i] for i in range(MEMORY)]
    same = prev.eq(lab) & n.gt(0)
    new = torch.stack([n + 1, lab, torch.where(same, age + 1, torch.ones_like(lab)), torch.maximum(ever, lab), nf + lab,
                       torch.where(lab.eq(1), torch.zeros_like(lab), since + 1)], -1)
    return new


# ------------------------------------------------------------------------------------------------ K-state heads
def attach_state_heads(model, stats, columns=ROUTED, label=LABEL, states=STATES):
    """Fraud rows are routed by state (onset / continuation) to separate copies of regressor+predictor of `columns`.
    Normal rows keep the native heads. Call AFTER the native weights are loaded."""
    label_key = argn_column(stats, label) + "__cat"; fraud_code = int(stats["columns"][label]["codes"]["1"])
    sub_cols = [k for k in model.tgt_cardinalities if k.startswith(tuple(argn_column(stats, c) + "__" for c in columns))]
    model.state_regressors = nn.ModuleDict({s: nn.ModuleDict({k: copy.deepcopy(model.regressors.get(k)) for k in sub_cols}) for s in states})
    model.state_predictors = nn.ModuleDict({s: nn.ModuleDict({k: copy.deepcopy(model.predictors.predictors[k]) for k in sub_cols}) for s in states})
    model.route = None  # (B,T,1) or (B,1) long: 0 normal, 1 onset, 2 continuation

    def set_route(module, args):
        lab = torch.as_tensor(args[0])
        fraud = lab.eq(fraud_code)
        if lab.dim() == 3:  # training: real codes (B,T,1); previous row's label along T
            prev_fraud = torch.roll(fraud, 1, dims=1); prev_fraud[:, 0] = False
        else:               # generation: sampled codes (B,1); previous label from the module memory
            mem = getattr(model, "module_memory", None)
            prev_fraud = (mem[:, 1].eq(1) & mem[:, 0].gt(0)).unsqueeze(-1) if mem is not None else torch.zeros_like(fraud)
        model.route = torch.where(fraud, torch.where(prev_fraud, torch.full_like(lab, 2), torch.ones_like(lab)), torch.zeros_like(lab))

    def route_regressor(module, args, output):
        regressor_in, sub_col = args
        if model.route is None or sub_col not in sub_cols:
            return output
        x_in = torch.cat(regressor_in, dim=-1); out = output
        for i, s in enumerate(states, start=1):
            x = x_in
            for layer in model.state_regressors[s][sub_col]:
                x = module.dropout(x); x = layer(x)
            out = torch.where(model.route.eq(i), nn.ReLU()(x), out)
        return out

    def route_predictor(module, args, output):
        x, sub_col = args
        if model.route is None or sub_col not in sub_cols:
            return output
        out = output
        for i, s in enumerate(states, start=1):
            out = torch.where(model.route.eq(i), model.state_predictors[s][sub_col](x), out)
        return out

    model.embedders.get(label_key).register_forward_pre_hook(set_route)
    model.regressors.register_forward_hook(route_regressor)
    model.predictors.register_forward_hook(route_predictor)
    model.routed_sub_columns = sub_cols; model.states = states
    return sub_cols


def state_head_names(model):
    return [n for n, _ in model.named_parameters() if n.startswith(("state_regressors.", "state_predictors."))]


def state_model(ws: Workspace, device, head_state=None, columns=ROUTED):
    ts = ws.tgt_stats.read(); card = get_cardinalities(ts); order = label_first_order(ts, card)
    model, ts, card = build_model(ws, device, order)
    attach_state_heads(model, ts, columns)
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


def train_state_heads(ws: Workspace, out_dir: Path, seed: int, device="cuda", max_epochs=100, patience=5, batch_size=8, window=100,
                      lr=1e-3, log=print, columns=ROUTED, substitute=True):
    """Fraud rows only, routed by state; unweighted CE; per-(state, column) early stopping on the internal-validation
    fraud-row CE under the substituted-context condition (the registered routed_subst rule)."""
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    model, ts, card, order = state_model(ws, device, None, columns)
    label_key = argn_column(ts, LABEL) + "__cat"; fraud_code = int(ts["columns"][LABEL]["codes"]["1"])
    col_keys = {c: [k for k in card if k.startswith(argn_column(ts, c) + "__")] for c in columns}
    subst_keys = [k for k in card if k.startswith(tuple(argn_column(ts, c) + "__" for c in OUTPUT_COLUMNS))]
    names = set(state_head_names(model))
    for n, prm in model.named_parameters():
        prm.requires_grad_(n in names)
    params = [prm for n, prm in model.named_parameters() if n in names]
    units = [(s, c) for s in STATES for c in columns]
    unit_params = {(s, c): [n for n in names if n.split(".")[1] == s and any(k in n for k in col_keys[c])] for s, c in units}
    assert all(unit_params[u] for u in units) and sum(len(v) for v in unit_params.values()) == len(names)
    trn = _records(list(ws.encoded_data_trn.fetch_all())[0]); val = _records(list(ws.encoded_data_val.fetch_all())[0])
    trn_loader = DataLoader(trn, shuffle=True, batch_size=batch_size, collate_fn=BatchCollator(True, window, torch.device(device)))
    val_loader = DataLoader(val, shuffle=False, batch_size=batch_size // 2, collate_fn=BatchCollator(True, None, torch.device(device)))
    opt = torch.optim.Adam(params, lr=lr); crit = nn.CrossEntropyLoss(reduction="none"); p_train = 1.0 if substitute else 0.0

    def step(batch, p_sub, generator):
        inputs, n_sub = substitute_inputs(batch, label_key, fraud_code, subst_keys, p_sub, generator) if p_sub > 0 else (batch, 0)
        output, _ = model(inputs, mode="trn")  # set_route ran on the (unsubstituted) label input
        route = model.route.squeeze(-1) * _valid_mask(batch).long()
        ce = {c: sum(crit(output[k].transpose(1, 2), batch[k].squeeze(2)) for k in col_keys[c]) for c in columns}
        return ce, route, n_sub

    def unit_val(p_sub):
        vgen = torch.Generator(device=device); vgen.manual_seed(20261108)
        sums = {u: 0.0 for u in units}; cnt = {s: 0 for s in STATES}
        for batch in val_loader:
            ce, route, _ = step(batch, p_sub, vgen)
            for i, s in enumerate(STATES, start=1):
                m = route.eq(i); cnt[s] += int(m.sum())
                for c in columns:
                    sums[(s, c)] += float(ce[c][m].sum())
        return {u: sums[u] / max(cnt[u[0]], 1) for u in units}, cnt

    best = {u: float("inf") for u in units}; best_state = {}; best_epoch = {}; bad = {u: 0 for u in units}; history = []
    out_dir.mkdir(parents=True, exist_ok=True); t0 = time.time(); seen = {s: 0 for s in STATES}
    for epoch in range(1, max_epochs + 1):
        model.train(); losses = []; seen = {s: 0 for s in STATES}
        for batch in trn_loader:
            ce, route, _ = step(batch, p_train, gen)
            if not route.gt(0).any():
                continue
            loss = 0
            for i, s in enumerate(STATES, start=1):
                m = route.eq(i)
                if m.any():
                    loss = loss + sum(ce[c][m].mean() for c in columns); seen[s] += int(m.sum())
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            v_sub, cnt = unit_val(1.0); v_real, _ = unit_val(0.0)
        crit_val = v_sub if substitute else v_real
        row = dict(epoch=epoch, trn_loss=float(np.mean(losses)) if losses else float("nan"), seconds=round(time.time() - t0, 1), **{f"trn_rows_{s}": seen[s] for s in STATES},
                   **{f"val_rows_{s}": cnt[s] for s in STATES}, **{f"val_subst_{s}_{c}": v_sub[(s, c)] for s, c in units}, **{f"val_real_{s}_{c}": v_real[(s, c)] for s, c in units})
        history.append(row)
        log(f"state_heads seed={seed} epoch={epoch} trn={row['trn_loss']:.4f} " + " ".join(f"{s[:4]}/{c.split('_')[0]}={crit_val[(s, c)]:.3f}" for s, c in units) + f" rows={seen}")
        sd = model.state_dict()
        for u in units:
            if crit_val[u] < best[u] - 1e-5:
                best[u], bad[u], best_epoch[u] = crit_val[u], 0, epoch
                for n in unit_params[u]:
                    best_state[n] = sd[n].detach().cpu().clone()
            else:
                bad[u] += 1
        if all(bad[u] >= patience for u in units):
            break
    torch.save(best_state, out_dir / "heads.pt"); pd.DataFrame(history).to_csv(out_dir / "learning.csv", index=False)
    meta = dict(mode="state_routed_subst" if substitute else "state_routed", states=list(STATES), substitute=substitute, seed=seed, loss="fraud rows only, routed by state, unweighted",
                epochs=len(history), best_epoch={f"{s}/{c}": e for (s, c), e in best_epoch.items()}, best_val={f"{s}/{c}": v for (s, c), v in best.items()},
                val_rows_by_state=cnt, trainable_parameters=int(sum(p.numel() for p in params)), column_order=order, routed_sub_columns=model.routed_sub_columns,
                routed_columns=list(columns), window=window, batch_size=batch_size, lr=lr, patience=patience, max_epochs=max_epochs, seconds=round(time.time() - t0, 1))
    (out_dir / "TRAIN.json").write_text(json.dumps(meta, indent=1))
    return meta


# ------------------------------------------------------------------------------------------------ generation
@contextmanager
def state_first_module(stats, p_fn=None, label=LABEL, attach=None, head_state=None):
    """Patch generation for one generate() call: label first; the label predictor output replaced by p_fn(memory_features)
    (learned state process) and per-customer memory (B,6) carried on the recurrent state; `attach(model)` adds heads after
    the native weights are loaded, `head_state` is loaded on top."""
    import mostlyai.engine._tabular.generation as generation
    parent_load = generation.load_model_weights

    def load_with_heads(*args, **kwargs):
        out = parent_load(*args, **kwargs)
        model = kwargs.get("model", args[0] if args else None)
        if attach is not None:
            attach(model)
        if head_state is not None:
            missing, unexpected = model.load_state_dict({k: v.to(model.device) for k, v in head_state.items()}, strict=False)
            assert not unexpected, unexpected
            assert len(missing) == len(model.state_dict()) - len(head_state), (len(missing), len(head_state))
        return out
    col = stats["columns"][label]
    label_col = get_argn_name(col["argn_processor"], col["argn_table"], col["argn_column"]); label_key = label_col + "__cat"
    codes = col["codes"]; c1, c0 = int(codes["1"]), int(codes["0"])
    parent_model = generation.SequentialModel; parent_order = generation._resolve_gen_column_order
    counter = dict(label_steps=0, rows=0)

    def ordered(*args, **kwargs):
        order = list(parent_order(*args, **kwargs))
        if label_col in order:
            order.remove(label_col)
            fixed = [c for c in order if c.startswith((SLEN_SUB_COLUMN_PREFIX, SIDX_SUB_COLUMN_PREFIX, SDEC_SUB_COLUMN_PREFIX))]
            order = fixed + [label_col] + [c for c in order if c not in fixed]
        return order

    class StateFirstModel(parent_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if p_fn is not None:
                self.predictors.register_forward_hook(self._replace_label)

        def _replace_label(self, module, args, output):
            if args[1] != label_key:
                return output
            counter["label_steps"] += 1
            p = torch.as_tensor(p_fn(memory_features(self.module_memory.detach().cpu().numpy())), device=output.device, dtype=torch.float64).clamp(1e-6, 1 - 1e-6)
            logits = torch.full_like(output, -1e9)
            logits[:, -1, c1] = p.log().to(output.dtype); logits[:, -1, c0] = torch.log1p(-p).to(output.dtype)
            return logits

        def forward(self, x, mode, **kwargs):
            if mode != "gen":
                return super().forward(x, mode, **kwargs)
            state = kwargs.get("history_state")
            if state is None:
                memory = torch.zeros((kwargs["batch_size"], MEMORY), dtype=torch.long, device=self.device)
            else:
                memory = state[-1][0]; kwargs["history_state"] = tuple(state[:-1])
            self.module_memory = memory
            outputs, history, recurrent = super().forward(x, mode, **kwargs)
            lab = outputs[label_key].reshape(-1).eq(c1).long(); counter["rows"] += int(lab.numel())
            return outputs, history, (*recurrent, update_memory(memory, lab).unsqueeze(0))

    generation.SequentialModel = StateFirstModel; generation._resolve_gen_column_order = ordered; generation.load_model_weights = load_with_heads
    try:
        yield dict(label_key=label_key, learned_state=p_fn is not None, attached=attach is not None, heads=head_state is not None, counter=counter)
    finally:
        generation.SequentialModel = parent_model; generation._resolve_gen_column_order = parent_order; generation.load_model_weights = parent_load
