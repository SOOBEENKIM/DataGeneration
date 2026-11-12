"""ibm_merchant_memory_v1: customer-history-referencing merchant generation on the frozen native ARGN
(docs/ibm_merchant_memory_v1/PROTOCOL.md).

At the merchant sub-column of every generated row the backbone's predictor output is replaced by
    P(m) = g * novel(m) + (1 - g) * repeat(m)
  g        = P(new merchant | label state, position bin)                (train table)
  repeat   = lam * frequency(past) + (1 - lam) * recency(past)            (customer's own past merchants)
  novel    = normal: backbone probabilities with past merchants masked;
             fraud : pi * shared fraud pool (train) + (1 - pi) * masked backbone
Per-customer merchant counts and last positions ride on the LSTM state tuple (dim 1 = batch), so the engine's
continue_mask filtering keeps them aligned. Backbone weights, the label process and all other columns are untouched.
"""
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch

LABEL = "event_is_fraud"
MERCHANT = "receiver_or_mark"
RARE = "_RARE_"
POS_EDGES = np.array([1, 5, 20, 100, 500])          # position bins: 0 | 1-4 | 5-19 | 20-99 | 100-499 | 500+
N_BINS = len(POS_EDGES) + 1
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SMOOTH = 0.5


def pos_bin(n):
    return np.searchsorted(POS_EDGES, np.asarray(n), side="right")


def _codes(stats, merchant=MERCHANT):
    col = stats["columns"][merchant]
    codes = {str(k): int(v) for k, v in col["codes"].items()}
    return codes, codes[RARE], len(codes)


def _walk(frame, codes, rare_code, kernel_customers=(), label=LABEL, merchant=MERCHANT):
    """Per row: label, position, novel flag (customer strict past, by raw merchant string), merchant code, and — for
    repeat rows of `kernel_customers` only (cost) — the probability of the chosen merchant under the frequency and
    recency kernels."""
    d = frame.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    y = pd.to_numeric(d[label], errors="coerce").fillna(0).astype(int).eq(1).astype(int).to_numpy()
    m = d[merchant].astype(str).to_numpy(); ent = d.entity_id.to_numpy(); code = np.array([codes.get(s, rare_code) for s in m])
    n = len(d); pos = np.zeros(n, int); novel = np.zeros(n, int); p_freq = np.full(n, np.nan); p_rec = np.full(n, np.nan)
    kernel_customers = set(kernel_customers); seen = {}; last = {}; k = 0; prev = None; do_kernel = False
    for i in range(n):
        if ent[i] != prev:
            seen = {}; last = {}; k = 0; prev = ent[i]; do_kernel = prev in kernel_customers
        pos[i] = k
        s = m[i]
        if s in seen:
            if do_kernel:
                p_freq[i] = seen[s] / k
                rec = {t: 1.0 / (k - l) for t, l in last.items()}; p_rec[i] = rec[s] / sum(rec.values())
        else:
            novel[i] = 1
        seen[s] = seen.get(s, 0) + 1; last[s] = k; k += 1
    return pd.DataFrame(dict(entity_id=ent, label=y, pos=pos, bin=pos_bin(pos), novel=novel, code=code, merchant=m, p_freq=p_freq, p_rec=p_rec))


def fit_merchant_memory(train, stats=None, codes=None, rare_code=-1, seed=20261109, holdout_fraction=0.1):
    """stats: ARGN tgt_stats (merchant codes incl. `_RARE_`); or `codes` = {merchant string: index} of another backbone's
    merchant vocabulary (RTF tokens; every train merchant must be in it). Gate, kernel choice and pi do not depend on the
    backbone — only the pool's indexing does."""
    if stats is not None:
        codes, rare_code, V = _codes(stats)
    else:
        V = len(codes)
    rng = np.random.default_rng(seed); ids = np.sort(train.entity_id.unique()); hold = set(rng.choice(ids, size=int(round(len(ids) * holdout_fraction)), replace=False))
    w = _walk(train, codes, rare_code, kernel_customers=hold)
    assert stats is not None or w.code.ge(0).all(), "train merchant outside the backbone vocabulary"
    is_hold = w.entity_id.isin(hold).to_numpy(); fit, ho = w[~is_hold], w[is_hold]
    # gate tables (all train rows; Jeffreys smoothing)
    gate = np.zeros((2, N_BINS)); gate_blind = np.zeros(N_BINS); counts = []
    for b in range(N_BINS):
        sel = w.bin.eq(b)
        gate_blind[b] = (w.novel[sel].sum() + SMOOTH) / (sel.sum() + 2 * SMOOTH)
        for l in (0, 1):
            s2 = sel & w.label.eq(l); gate[l, b] = (w.novel[s2].sum() + SMOOTH) / (s2.sum() + 2 * SMOOTH)
            counts.append(dict(label=l, bin=b, rows=int(s2.sum()), novel=int(w.novel[s2].sum()), p_novel=float(gate[l, b])))
    # repeat kernel: mixture weight chosen on hold-out customers' repeat rows
    rep_ho = ho[ho.novel.eq(0)]
    ll = {lam: float(np.log(np.clip(lam * rep_ho.p_freq + (1 - lam) * rep_ho.p_rec, 1e-12, None)).mean()) for lam in LAMBDAS}
    lam = max(ll, key=ll.get)
    # shared fraud pool: fit customers -> pi on hold-out novel fraud rows; final pool on all train fraud rows
    pool_fit = set(fit.loc[fit.label.eq(1), "merchant"]); ho_nf = ho[ho.label.eq(1) & ho.novel.eq(1)]
    pi = float(ho_nf.merchant.isin(pool_fit).mean()) if len(ho_nf) else np.nan
    fr = w[w.label.eq(1)]; pool = np.bincount(fr.code.to_numpy(), minlength=V).astype(float); pool_merchants = int(fr.merchant.nunique())
    pool = pool / pool.sum()
    return dict(gate=gate.tolist(), gate_blind=gate_blind.tolist(), gate_counts=counts, position_edges=POS_EDGES.tolist(), lam=lam, kernel_holdout_logloss={str(k): -v for k, v in ll.items()},
                holdout_repeat_rows=int(len(rep_ho)), pi=pi, holdout_novel_fraud_rows=int(len(ho_nf)), pool=pool.tolist(), pool_merchants=pool_merchants, pool_rows=int(len(fr)),
                rare_code=rare_code, n_codes=V, train_rows=int(len(w)), train_customers=int(len(ids)), holdout_customers=len(hold), seed=seed,
                train_novel_share=dict(fraud=float(w.novel[w.label.eq(1)].mean()), normal=float(w.novel[w.label.eq(0)].mean())))


@contextmanager
def merchant_module(stats, params, arm="full", label=LABEL, merchant=MERCHANT, calibration=None):
    """Wrap whatever generation.SequentialModel currently is (e.g. the transition/label-first ModuleModel) and replace the
    merchant predictor output. arm: full | gate_only (no fraud pool) | repeat_blind (label-independent gate, no pool)."""
    import mostlyai.engine._tabular.generation as generation
    from mostlyai.engine._common import get_argn_name
    assert arm in ("full", "gate_only", "repeat_blind")
    lcol = stats["columns"][label]; label_key = get_argn_name(lcol["argn_processor"], lcol["argn_table"], lcol["argn_column"]) + "__cat"
    c1 = int(lcol["codes"]["1"])
    mcol = stats["columns"][merchant]; merchant_key = get_argn_name(mcol["argn_processor"], mcol["argn_table"], mcol["argn_column"]) + "__cat"
    codes, rare_code, V = _codes(stats)
    gate = torch.tensor(params["gate"], dtype=torch.float64); gate_blind = torch.tensor(params["gate_blind"], dtype=torch.float64)
    lam = float(params["lam"]); pi = float(params["pi"]) if arm == "full" else 0.0
    # ibm_merchant_marginal_v1 (2026-11-10), optional: calibration = dict(stick=(V,) repeat weights per merchant code, lam=...,
    # cat_of_m=(V,) category index per merchant code, p_cat_novel=(K,) target category shares of normal first visits). None = unchanged.
    cal = calibration or {}
    if cal.get("lam") is not None:
        lam = float(cal["lam"])
    stick = torch.tensor(cal["stick"], dtype=torch.float64) if cal.get("stick") is not None else None
    # ibm_merchant_marginal_v2: additive prior pseudo-count per merchant code (kappa * relative visit rate of the merchant's type)
    prior = torch.tensor(cal["prior"], dtype=torch.float64) if cal.get("prior") is not None else None
    cat_of_m = torch.tensor(cal["cat_of_m"], dtype=torch.long) if cal.get("p_cat_novel") is not None else None
    p_cat = torch.tensor(cal["p_cat_novel"], dtype=torch.float64) if cal.get("p_cat_novel") is not None else None
    pool = torch.tensor(params["pool"], dtype=torch.float64); edges = torch.tensor(params["position_edges"], dtype=torch.float64)
    parent_model = generation.SequentialModel
    counter = dict(merchant_steps=0, rows=0, repeat_mass=0.0, novel_mass=0.0, pool_mass=0.0)

    class MerchantModel(parent_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mm_gate = gate.to(self.device); self.mm_gate_blind = gate_blind.to(self.device); self.mm_pool = pool.to(self.device); self.mm_edges = edges.to(self.device)
            self.mm_label = None
            self.mm_stick = stick.to(self.device) if stick is not None else None
            self.mm_prior = prior.to(self.device) if prior is not None else None
            self.mm_cat_of_m = cat_of_m.to(self.device) if cat_of_m is not None else None; self.mm_p_cat = p_cat.to(self.device) if p_cat is not None else None
            self.embedders.get(label_key).register_forward_pre_hook(self._capture_label)
            self.predictors.register_forward_hook(self._replace_merchant)

        def _capture_label(self, module, args):
            out = args[0]
            if out.dim() == 2:  # generation: sampled codes (B,1)
                self.mm_label = out.reshape(-1).eq(c1).long()

        def _replace_merchant(self, module, args, output):
            if args[1] != merchant_key or self.mm_label is None:
                return output
            counts, last = self.mm_counts, self.mm_last                      # (B,V) float64, (B,V) long (-1 = never)
            n = counts.sum(1)                                                 # events so far (B,)
            B = counts.shape[0]; dev = output.device
            backbone = torch.softmax(output[:, -1, :].double(), -1)           # (B,V)
            has_past = counts.gt(0)
            # gate
            b = torch.bucketize(n, self.mm_edges, right=True)
            g = self.mm_gate_blind[b] if arm == "repeat_blind" else self.mm_gate[self.mm_label, b]
            g = torch.where(n.gt(0), g, torch.ones_like(g))
            # repeat branch
            freq = counts / counts.sum(1, keepdim=True).clamp(min=1)
            if self.mm_prior is not None:  # visited merchants get count + kappa * rho(type): the type acts while counts are small
                fq = counts + self.mm_prior.unsqueeze(0) * has_past; freq = fq / fq.sum(1, keepdim=True).clamp(min=1e-300)
            rec = torch.where(has_past, 1.0 / (n.unsqueeze(1) - last).clamp(min=1).double(), torch.zeros_like(counts))
            rec = rec / rec.sum(1, keepdim=True).clamp(min=1e-12)
            repeat = lam * freq + (1 - lam) * rec
            if self.mm_stick is not None:  # category-level stickiness of the customer's past merchants
                repeat = repeat * self.mm_stick.unsqueeze(0); repeat = repeat / repeat.sum(1, keepdim=True).clamp(min=1e-300)
            if cal.get("rare_no_repeat"):  # ibm_merchant_marginal_v1 amendment 2: the rare bucket is many merchants, never one habitual merchant
                repeat = repeat.clone(); repeat[:, rare_code] = 0.0; repeat = repeat / repeat.sum(1, keepdim=True).clamp(min=1e-300)
            # novel branch: mask the customer's past (the rare bucket stays eligible: it stands for many merchants)
            mask = has_past.clone(); mask[:, rare_code] = False
            nb = backbone.masked_fill(mask, 0.0); nb = nb / nb.sum(1, keepdim=True).clamp(min=1e-12)
            nb_n = nb  # normal rows' novel branch
            if self.mm_p_cat is not None:  # category first: train shares of first-visit categories, backbone within the category
                idx = self.mm_cat_of_m.unsqueeze(0).expand(B, -1)
                mass = torch.zeros((B, self.mm_p_cat.shape[-1]), dtype=nb.dtype, device=nb.device).scatter_add_(1, idx, nb)
                pc = self.mm_p_cat[b] if self.mm_p_cat.dim() == 2 else self.mm_p_cat.unsqueeze(0)  # amendment 1: (position bin, K) table
                target = pc * mass.gt(0); target = target / target.sum(1, keepdim=True).clamp(min=1e-300)
                nb_n = nb * torch.gather(target / mass.clamp(min=1e-300), 1, idx); nb_n = nb_n / nb_n.sum(1, keepdim=True).clamp(min=1e-300)
            novel = torch.where(self.mm_label.eq(1).unsqueeze(1), nb, nb_n)
            if pi > 0:
                pp = self.mm_pool.unsqueeze(0).expand(B, -1).masked_fill(mask, 0.0); pz = pp.sum(1, keepdim=True)
                pp = torch.where(pz.gt(1e-12), pp / pz.clamp(min=1e-12), nb)
                fraud = self.mm_label.eq(1).unsqueeze(1)
                novel = torch.where(fraud, pi * pp + (1 - pi) * nb, nb_n)
                counter["pool_mass"] += float((g * pi)[self.mm_label.eq(1)].sum())
            probs = g.unsqueeze(1) * novel + (1 - g).unsqueeze(1) * repeat
            probs = probs / probs.sum(1, keepdim=True).clamp(min=1e-12)
            counter["merchant_steps"] += 1; counter["rows"] += B; counter["novel_mass"] += float(g.sum()); counter["repeat_mass"] += float((1 - g).sum())
            logits = torch.full_like(output, -1e9); logits[:, -1, :] = probs.clamp(min=1e-12).log().to(output.dtype)
            return logits

        def forward(self, x, mode, **kwargs):
            if mode != "gen":
                return super().forward(x, mode, **kwargs)
            state = kwargs.get("history_state")
            if state is None:
                B = kwargs["batch_size"]
                counts = torch.zeros((B, V), dtype=torch.float64, device=self.device); last = torch.full((B, V), -1, dtype=torch.long, device=self.device)
            else:
                counts, last = state[-2][0], state[-1][0]; kwargs["history_state"] = tuple(state[:-2])
            self.mm_counts, self.mm_last = counts, last; self.mm_label = None
            outputs, history, recurrent = super().forward(x, mode, **kwargs)
            m = outputs[merchant_key].reshape(-1).long(); n = counts.sum(1).long()
            counts = counts.clone(); counts[torch.arange(len(m), device=m.device), m] += 1.0
            last = last.clone(); last[torch.arange(len(m), device=m.device), m] = n
            return outputs, history, (*recurrent, counts.unsqueeze(0), last.unsqueeze(0))

    generation.SequentialModel = MerchantModel
    try:
        yield dict(arm=arm, lam=lam, pi=pi, merchant_key=merchant_key, counter=counter, sticky=stick is not None, category_first=p_cat is not None)
    finally:
        generation.SequentialModel = parent_model
