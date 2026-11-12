"""ibm_merchant_marginal_v1: estimation of the two calibrations of the merchant-memory component
(docs/ibm_merchant_marginal_v1/PROTOCOL.md). Train only.
  R1 stickiness: repeat kernel p(m) ∝ w_{c(m)} * [lam * frequency(m) + (1 - lam) * recency(m)], c(m) = merchant's (train modal)
     category; w by conditional-logit maximum likelihood of the frequency kernel on the fit customers' repeat rows;
     lam re-chosen on the hold-out customers' repeat rows.
  R2 category first: P(category | normal row, first visit to the merchant) for the novel branch.
Merchants are handled in the backbone's code space (the rare bucket is one code with its own category `_RARE_`).
"""
import numpy as np
import pandas as pd
import torch

LABEL, MERCHANT, RARE = "event_is_fraud", "receiver_or_mark", "_RARE_"
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def fit_calibration(train, stats, seed=20261109, holdout_fraction=0.1, sample_rows=400_000, sample_seed=20261110):
    col = stats["columns"][MERCHANT]; codes = {str(k): int(v) for k, v in col["codes"].items()}; rare_code = codes[RARE]; V = len(codes)
    d = train.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    d["m"] = d[MERCHANT].astype(str); d["cat"] = d.category.astype(str)
    y = pd.to_numeric(d[LABEL], errors="coerce").fillna(0).astype(int).to_numpy()
    modal = d.groupby("m").cat.agg(lambda s: s.value_counts().index[0])
    cats = sorted(modal.unique().tolist()) + [RARE]; ci = {c: i for i, c in enumerate(cats)}; K = len(cats)
    cat_of_m = np.full(V, K - 1, dtype=np.int64)
    for m, c in modal.items():
        if m in codes and codes[m] != rare_code:
            cat_of_m[codes[m]] = ci[c]
    code = d.m.map(codes).fillna(rare_code).astype(int).to_numpy(); kcat = cat_of_m[code]; ent = d.entity_id.to_numpy()
    novel_str = (~d.duplicated(["entity_id", "m"])).to_numpy()
    # R2: category shares of normal first visits (string space: a first visit to a rare merchant counts for `_RARE_`)
    sel = novel_str & (y == 0); p_cat = np.bincount(kcat[sel], minlength=K).astype(float); p_cat = p_cat / p_cat.sum()
    # amendment 1 (2026-11-10): the same shares by position bin (the gate's bins), prior = 10 rows of the pooled table
    pos_bin = np.searchsorted(np.array([1, 5, 20, 100, 500]), d.event_index.to_numpy(), side="right"); p_cat_pos = np.zeros((6, K)); pos_rows = []
    for bb in range(6):
        cnt = np.bincount(kcat[sel & (pos_bin == bb)], minlength=K).astype(float); pos_rows.append(int(cnt.sum())); p_cat_pos[bb] = (cnt + 10.0 * p_cat) / (cnt.sum() + 10.0)
    # hold-out split identical to ibm_merchant_memory_v1
    ids = np.sort(d.entity_id.unique()); hold = set(np.random.default_rng(seed).choice(ids, size=int(round(len(ids) * holdout_fraction)), replace=False))
    is_hold = d.entity_id.isin(hold).to_numpy()
    # R1 design: strict-past category counts of repeat rows (code space) for the fit customers
    rng = np.random.default_rng(sample_seed); bounds = np.flatnonzero(np.r_[True, ent[1:] != ent[:-1], True]); X, T = [], []; n_rep = 0
    frac = None
    for a, b in zip(bounds[:-1], bounds[1:]):
        if is_hold[a]:
            continue
        c = code[a:b]; k = kcat[a:b]; rep = pd.Series(c).duplicated().to_numpy(); n_rep += int(rep.sum())
        onehot = np.zeros((b - a, K), dtype=np.float32); onehot[np.arange(b - a), k] = 1.0
        past = np.cumsum(onehot, 0) - onehot
        rows = np.flatnonzero(rep)
        if frac is None:
            frac = min(1.0, sample_rows / max(1.0, 0.92 * (~is_hold).sum()))
        rows = rows[rng.random(len(rows)) < frac]
        X.append(past[rows]); T.append(k[rows])
    X = torch.from_numpy(np.concatenate(X)); T = torch.from_numpy(np.concatenate(T)).long()
    logN = torch.where(X > 0, X.clamp(min=1e-12).log(), torch.full_like(X, -1e9))
    theta = torch.zeros(K, dtype=torch.float32, requires_grad=True); opt = torch.optim.LBFGS([theta], lr=0.5, max_iter=200, history_size=20, line_search_fn="strong_wolfe")

    def nll():
        z = logN + theta.unsqueeze(0)
        return -(z.gather(1, T.unsqueeze(1)).squeeze(1) - torch.logsumexp(z, 1)).mean()

    def closure():
        opt.zero_grad(); loss = nll() + 1e-4 * (theta ** 2).sum(); loss.backward(); return loss
    with torch.no_grad():
        nll0 = float(nll())
    opt.step(closure)
    with torch.no_grad():
        nll1 = float(nll()); th = (theta - theta.mean()).double().numpy()   # weights are identified up to scale
    w_cat = np.exp(th); stick = w_cat[cat_of_m]
    # hold-out log-likelihood of the full kernel, base vs sticky, per lambda (code space, repeat rows)
    ll_base = {l: 0.0 for l in LAMBDAS}; ll_sticky = {l: 0.0 for l in LAMBDAS}; n_ho = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        if not is_hold[a]:
            continue
        seen = {}; cnt = []; last = []; cds = []
        for t in range(a, b):
            c = code[t]; n = t - a
            if c in seen:
                j = seen[c]; cn = np.asarray(cnt, dtype=float); f = cn / n; r = 1.0 / (n - np.asarray(last, dtype=float)); r = r / r.sum(); w = stick[np.asarray(cds)]
                for l in LAMBDAS:
                    kern = l * f + (1 - l) * r
                    ll_base[l] += np.log(max(kern[j], 1e-300)); ll_sticky[l] += np.log(max(w[j] * kern[j] / (w * kern).sum(), 1e-300))
                n_ho += 1; cnt[j] += 1; last[j] = n
            else:
                seen[c] = len(cds); cds.append(c); cnt.append(1); last.append(n)
    ll_base = {l: v / n_ho for l, v in ll_base.items()}; ll_sticky = {l: v / n_ho for l, v in ll_sticky.items()}
    lam_base = max(ll_base, key=ll_base.get); lam_sticky = max(ll_sticky, key=ll_sticky.get)
    order = np.argsort(-w_cat)
    return dict(categories=cats, cat_of_m=cat_of_m.tolist(), w_cat=w_cat.tolist(), p_cat_novel=p_cat.tolist(), p_cat_novel_pos=p_cat_pos.tolist(), novel_rows_by_position_bin=pos_rows, lam_sticky=float(lam_sticky), lam_base=float(lam_base),
                holdout_ll_base={str(k): v for k, v in ll_base.items()}, holdout_ll_sticky={str(k): v for k, v in ll_sticky.items()}, holdout_repeat_rows=int(n_ho),
                holdout_gain=float(ll_sticky[lam_sticky] - ll_base[0.75]), fit_rows=int(len(T)), fit_repeat_rows_total=int(n_rep), fit_nll_before=nll0, fit_nll_after=nll1,
                top_sticky=[(cats[i], round(float(w_cat[i]), 3)) for i in order[:8]], least_sticky=[(cats[i], round(float(w_cat[i]), 3)) for i in order[-8:]],
                rare_first_visit_share=float(p_cat[K - 1]), n_codes=V, rare_code=rare_code, seed=seed, sample_seed=sample_seed)
