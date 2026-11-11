"""ibm_gap_diag_v1 D2 follow-up (added 2026-11-11 after the first D2 output, which put the whole ARGN gap in the NORMAL rows;
recorded as an addition in RESULTS).   ./run python scripts/diag_utility_gap_b.py
(a) cross-synthetic row swap: ARGN fraud rows + RTF normal rows and the reverse (feature-space mix, diagnostic only -- not a generator);
(b) which normal-row structure differs: amount / category / gap structure within customer and within (customer, merchant);
(c) per-feature permutation importance of the normal-row discriminator.
CPU only; train + dev + existing dev generations; never the sealed test.
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import benchmarks.fraud_utility as fu  # noqa: E402
import diag_utility_gap as dg  # noqa: E402

OUT = dg.OUT


def structure(d):
    """Normal-row structure a detector's history features can see."""
    d = d.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    y = pd.to_numeric(d.event_is_fraud, errors="coerce").fillna(0).astype(int); d = d[y.eq(0)].copy()
    d["la"] = np.log1p(pd.to_numeric(d.amount_or_numeric_value, errors="coerce").clip(lower=0).fillna(0)); d["lg"] = np.log1p(pd.to_numeric(d.gap, errors="coerce").clip(lower=0).fillna(0))
    d["m"] = d.receiver_or_mark.astype(str); d["c"] = d.category.astype(str); g = d.groupby("entity_id", sort=False)
    resid = lambda keys: float((d.la - d.groupby(keys).la.transform("mean")).std())
    cm = d.groupby(["entity_id", "m"]); nvis = cm.la.transform("size"); rep = d[nvis.ge(2)]
    prev_same = cm.la.shift(); has = prev_same.notna()
    out = dict(rows=int(len(d)), amount_sd=float(d.la.std()), amount_sd_within_customer=resid(["entity_id"]), amount_sd_within_category=resid(["c"]), amount_sd_within_merchant=resid(["m"]),
               amount_sd_within_customer_category=resid(["entity_id", "c"]), amount_sd_within_customer_merchant=float((rep.la - rep.groupby(["entity_id", "m"]).la.transform("mean")).std()),
               rows_at_repeat_customer_merchant_pairs=float(nvis.ge(2).mean()),
               abs_dlog_amount_vs_previous_visit_same_merchant_median=float((d.la - prev_same)[has].abs().median()), exact_same_amount_as_previous_visit_same_merchant=float(((d.la - prev_same)[has].abs() < 1e-9).mean()),
               amount_lag1_autocorr=float(d.la.corr(g.la.shift())), gap_lag1_autocorr=float(d.lg.corr(g.lg.shift())), gap_sd=float(d.lg.std()), gap_sd_within_customer=float((d.lg - g.lg.transform("mean")).std()),
               same_category_as_previous=float(d.c.eq(g.c.shift()).mean()), same_merchant_as_previous=float(d.m.eq(g.m.shift()).mean()),
               same_merchant_within_5=float(pd.concat([d.m.eq(g.m.shift(k)) for k in range(1, 6)], axis=1).any(axis=1).mean()),
               same_category_within_5=float(pd.concat([d.c.eq(g.c.shift(k)) for k in range(1, 6)], axis=1).any(axis=1).mean()),
               distinct_merchants_per_customer_median=float(g.m.nunique().median()), distinct_categories_per_customer_median=float(g.c.nunique().median()))
    # the same within-(customer, merchant) spread for the customer's most visited merchants (rank 1-3) vs the rest
    rk = d.groupby("entity_id").m.transform(lambda s: s.map(s.value_counts().rank(ascending=False, method="first")))
    top = d[rk.le(3)]; out["share_rows_top3_merchants"] = float(rk.le(3).mean()); out["amount_sd_within_customer_merchant_top3"] = float((top.la - top.groupby(["entity_id", "m"]).la.transform("mean")).std())
    return out


def per_feature_importance(real, synth, cols, seed=0):
    X = pd.concat([real[cols], synth[cols]], ignore_index=True).fillna(0).to_numpy(dtype=float); z = np.r_[np.zeros(len(real)), np.ones(len(synth))].astype(int)
    grp = np.r_["R" + real.entity_id.astype(str).to_numpy(), synth.src.astype(str).to_numpy() + synth.entity_id.astype(str).to_numpy()]; rng = np.random.default_rng(seed)
    single = [c for c in cols if not c.startswith(("cat_", "chan_"))]; drops = {c: [] for c in single}; aucs = []
    for tr, te in GroupKFold(n_splits=5).split(X, z, grp):
        clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200, learning_rate=0.05, l2_regularization=1.0, min_samples_leaf=20).fit(X[tr], z[tr])
        base = roc_auc_score(z[te], clf.predict_proba(X[te])[:, 1]); aucs.append(base)
        for c in single:
            i = cols.index(c); Xp = X[te].copy(); Xp[:, i] = Xp[rng.permutation(len(te)), i]; drops[c].append(base - roc_auc_score(z[te], clf.predict_proba(Xp)[:, 1]))
    return dict(auc=float(np.mean(aucs)), drop=dict(sorted({c: float(np.mean(v)) for c, v in drops.items()}.items(), key=lambda kv: -kv[1])))


def main():
    t0 = time.time(); dev = pd.read_parquet(dg.PREP / "validation.parquet"); fu.configure(dev); dg.TEST = fu.features(dev); train = pd.read_parquet(dg.PREP / "train.parquet")
    raw = {m: [pd.read_parquet(path(fs, gs)) for fs, gs in dg.FITS] for m, path in dg.SOURCES.items()}
    for m in raw:
        for i, r in enumerate(raw[m]):
            dg.F[f"{m}{i}"] = fu.features(r)
    jobs = []
    for i in range(4):
        jobs.append((f"cross|ARGNf_RTFn|{i}|all", [(f"argn_final{i}", 1), (f"rtf_table{i}", 0)], []))
        jobs.append((f"cross|RTFf_ARGNn|{i}|all", [(f"rtf_table{i}", 1), (f"argn_final{i}", 0)], []))
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(dg._fit_eval, jobs))
    res = pd.DataFrame(rows); res["kind"] = res.name.str.split("|").str[1]; res.to_csv(OUT / "UTILITY_GAP_CROSS_FITS.csv", index=False)
    cross = res.groupby("kind")[["history_auprc", "early"]].agg(["mean", "std"]).round(4); print(cross.to_string(), flush=True)
    st = dict(real_train=structure(train), real_dev=structure(dev))
    for m in raw:
        per = [structure(r) for r in raw[m]]; st[m] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    print(pd.DataFrame(st).round(3).to_string(), flush=True)
    cols = dg.cols_all(); ftrain = fu.features(train); rng = np.random.default_rng(0); imp = {}
    rn = ftrain[ftrain.y.eq(0)]; rn = rn.iloc[rng.choice(len(rn), 50000, replace=False)]
    for m in raw:
        s = pd.concat([dg.F[f"{m}{i}"].assign(src=f"{m}{i}_") for i in range(4)], ignore_index=True); sn = s[s.y.eq(0)]; sn = sn.iloc[rng.choice(len(sn), 50000, replace=False)]
        imp[m] = per_feature_importance(rn, sn, cols); print(m, json.dumps({k: round(v, 3) for k, v in imp[m]["drop"].items()}), flush=True)
    (OUT / "UTILITY_GAP_B.json").write_text(json.dumps(dict(cross_swap={k: dict(history_auprc_mean=float(res[res.kind.eq(k)].history_auprc.mean()), history_auprc_sd=float(res[res.kind.eq(k)].history_auprc.std()),
                                                                               per_draw=res[res.kind.eq(k)].history_auprc.round(4).tolist()) for k in res.kind.unique()},
                                                            normal_structure=st, normal_discriminator_per_feature=imp, seconds=round(time.time() - t0, 1), test_read=False), indent=1, default=float) + "\n")
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
