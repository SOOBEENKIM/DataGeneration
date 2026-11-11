"""ibm_normal_rows_v1 amendment 2 (docs/ibm_normal_rows_v1/PROTOCOL.md): CPU post-processing arms aimed at the two diagnosed structures
S1 (short-gap rows stay at the same merchant) and S2 (the category sets the amount scale).
    ./run python scripts/sim_normal_rows_v2.py fit | apply | measure
Fraud rows, labels, lengths, gaps and novel-merchant rows are never changed. No training, no GPU, no new generation, no test file.
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
from sim_normal_rows import base_frame, finish, PREP, OUT, DOCS, BASE, FITS, A, M, C, G, Y, U, MIN_POOL  # noqa: E402

ARMS2 = {"REC2": ("REC2",), "AMT0": ("AMT0",), "AMT2": ("AMT2",), "REC2_AMT0": ("REC2", "AMT0"), "ALL2": ("REC2", "AMT2")}
GBINS = np.array([60, 300, 600, 1800, 3600, 14400, 86400]); GNAMES = ["<=1m", "1-5m", "5-10m", "10-30m", "30-60m", "1-4h", "4-24h", ">24h"]
NB = np.array([1, 4, 19])                                           # previous visits in the category: 1 | 2-4 | 5-19 | 20+
TABLES = OUT / "tables_v2.json"


def gbin(g):
    return np.searchsorted(GBINS, g, side="left")


def cmd_fit():
    t0 = time.time(); d = base_frame(pd.read_parquet(PREP / "train.parquet")); d["seen"] = d.duplicated(["entity_id", "m"]); d["m_prev"] = d.groupby("entity_id", sort=False).m.shift()
    el = d.y.eq(0) & d.prev_y.eq(0) & d.seen & d.g.notna(); b = gbin(d.g[el].to_numpy()); stay = (d.m == d.m_prev)[el].to_numpy(); stay_t = [float(stay[b == k].mean()) for k in range(len(GNAMES))]
    ok = d.y.eq(0) & d.a.gt(0); n = d[ok].copy(); n["la"] = np.log(n.a); mu = n.groupby("c").la.mean(); mu_all = float(n.la.mean()); cnt_c = n.c.value_counts(); big = set(cnt_c[cnt_c.ge(MIN_POOL)].index)
    key = n.c.where(n.c.isin(big), "other"); amt0 = {k: v.round(4).tolist() for k, v in n.la.groupby(key)}; amt0.setdefault("other", n.la.round(4).tolist())
    gp = n.groupby(["entity_id", "c"], sort=False); cnt = gp.cumcount().to_numpy(); prev_sum = (gp.la.cumsum() - n.la).to_numpy(); mu_row = n.c.map(mu).to_numpy(); yv = n.la.to_numpy() - mu_row
    x = np.where(cnt > 0, prev_sum / np.maximum(cnt, 1) - mu_row, 0.0); bucket = np.where(cnt > 0, np.searchsorted(NB, cnt, side="left"), -1); slopes = []; pools = {}; keyv = key.to_numpy()
    for q in range(-1, len(NB) + 1):
        sel = bucket == q; B = 0.0 if q < 0 else float(np.cov(x[sel], yv[sel])[0, 1] / np.var(x[sel], ddof=1)); r = yv[sel] - B * x[sel]
        if q >= 0:
            slopes.append(B)
        kk = keyv[sel]; cq = pd.Series(kk).value_counts(); bigq = set(cq[cq.ge(MIN_POOL)].index); kq = np.where(np.isin(kk, list(bigq)), kk, "other")
        pools[str(q)] = {k: np.round(r[kq == k], 4).tolist() for k in np.unique(kq)}; pools[str(q)].setdefault("other", np.round(r, 4).tolist())
    T = dict(stay=stay_t, mu={k: float(v) for k, v in mu.items()}, mu_all=mu_all, slopes=slopes, pools=pools, amt0=amt0)
    TABLES.write_text(json.dumps(T) + "\n")
    facts = dict(stay_by_gap=dict(zip(GNAMES, stay_t)), eligible_rows=int(el.sum()), slopes=dict(zip(["1", "2-4", "5-19", "20+"], slopes)), rows_by_bucket={str(q): int((bucket == q).sum()) for q in range(-1, len(NB) + 1)},
                 residual_sd_by_bucket={str(q): float(np.std(np.concatenate([np.asarray(v) for k, v in pools[str(q)].items() if k != "other"] or [np.asarray(pools[str(q)]["other"])]))) for q in range(-1, len(NB) + 1)},
                 category_mean_sd_row_weighted=float(np.sqrt(np.average((mu.reindex(cnt_c.index) - mu_all) ** 2, weights=cnt_c))), categories_with_own_pool=len(big), seconds=round(time.time() - t0, 1), test_read=False)
    (DOCS / "FIT_V2.json").write_text(json.dumps(facts, indent=1, default=float) + "\n"); print(json.dumps(facts, indent=1, default=float))


def apply_rec2(d, T, rng):
    seen = d.duplicated(["entity_id", "m"]).to_numpy(); y = d.y.to_numpy(); py = d.prev_y.to_numpy(); g = d.g.to_numpy(); el = (y == 0) & (py == 0) & seen & ~np.isnan(g); b = np.where(el, gbin(np.nan_to_num(g)), 0)
    same0 = (d.m == d.groupby("entity_id", sort=False).m.shift()).to_numpy(); p0 = [float(same0[el & (b == k)].mean()) if (el & (b == k)).any() else 0.0 for k in range(len(GNAMES))]
    s = [float(np.clip((T["stay"][k] - p0[k]) / max(1e-9, 1 - p0[k]), 0, 1)) for k in range(len(GNAMES))]
    m = d.m.to_numpy().copy(); c = d.c.to_numpy().copy(); u = d[U].astype(str).to_numpy().copy(); r = rng.random(len(d)); changed = 0
    for i in np.flatnonzero(el):
        if r[i] < s[b[i]]:
            if m[i] != m[i - 1]:
                changed += 1
            m[i], c[i], u[i] = m[i - 1], c[i - 1], u[i - 1]
    d = d.copy(); d["m"], d["c"], d[U] = m, c, u; return d, dict(eligible=int(el.sum()), changed=changed, p0=dict(zip(GNAMES, p0)), s=dict(zip(GNAMES, s)))


def apply_amt0(d, T, rng):
    ok = (d.y.eq(0) & d.a.gt(0)).to_numpy(); a = d.a.to_numpy().copy(); c = d.c.to_numpy()[ok]; key = np.where(np.isin(c, list(T["amt0"])), c, "other"); out = np.empty(len(c))
    for k in np.unique(key):
        sel = key == k; pool = np.asarray(T["amt0"][k]); out[sel] = pool[rng.integers(0, len(pool), int(sel.sum()))]
    a[ok] = np.maximum(0.01, np.round(np.exp(out), 2)); d = d.copy(); d["a"] = a; return d, dict(rows=int(ok.sum()))


def apply_amt2(d, T, rng):
    ok = (d.y.eq(0) & d.a.gt(0)).to_numpy(); a = d.a.to_numpy().copy(); c = d.c.to_numpy(); e = d.entity_id.to_numpy(); mu, mu_all, slopes = T["mu"], T["mu_all"], T["slopes"]; pools = {q: {k: np.asarray(v) for k, v in t.items()} for q, t in T["pools"].items()}
    u = rng.random(len(d)); state = {}; cur = None; used = dict(first=0, ref=0)
    for i in np.flatnonzero(ok):
        if e[i] != cur:
            cur = e[i]; state = {}
        g = c[i]; s, n = state.get(g, (0.0, 0)); q = -1 if n == 0 else int(np.searchsorted(NB, n, side="left")); m0 = mu.get(g, mu_all); pool = pools[str(q)]; p = pool.get(g, pool["other"])
        la = m0 + (slopes[q] * (s / n - m0) if n else 0.0) + p[int(u[i] * len(p))]; a[i] = max(0.01, round(float(np.exp(la)), 2)); state[g] = (s + float(np.log(a[i])), n + 1); used["ref" if n else "first"] += 1
    d = d.copy(); d["a"] = a; return d, used


def cmd_apply():
    T = json.loads(TABLES.read_text()); log = {}; parts_fn = {"REC2": apply_rec2, "AMT0": apply_amt0, "AMT2": apply_amt2}
    for fs, gs in FITS:
        raw = pd.read_parquet(BASE(fs, gs)); base = base_frame(raw); ref = raw.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True); fr = pd.to_numeric(ref[Y], errors="coerce").eq(1).to_numpy()
        for arm, parts in ARMS2.items():
            rng = np.random.default_rng([20261112, fs, gs, sorted(ARMS2).index(arm)]); d = base.copy(); info = {}
            for p in parts:
                d, info[p] = parts_fn[p](d, T, rng)
            out = finish(d, ref); assert len(out) == len(ref) and out[Y].astype(str).equals(ref[Y].astype(str)), "labels changed"
            for col in (M, C, U):
                assert (out.loc[fr, col].astype(str).to_numpy() == ref.loc[fr, col].astype(str).to_numpy()).all(), f"fraud rows changed: {col}"
            for col in (G, A):
                assert np.allclose(pd.to_numeric(out.loc[fr, col], errors="coerce").to_numpy(dtype=float), pd.to_numeric(ref.loc[fr, col], errors="coerce").to_numpy(dtype=float), equal_nan=True), f"fraud rows changed: {col}"
            assert np.allclose(pd.to_numeric(out[G], errors="coerce").to_numpy(dtype=float), pd.to_numeric(ref[G], errors="coerce").to_numpy(dtype=float), equal_nan=True), "gaps changed"
            folder = OUT / "sim2" / f"{arm}_{fs}"; folder.mkdir(parents=True, exist_ok=True); out.to_parquet(folder / f"generated_validation_{gs}.parquet", index=False); log[f"{arm}_{fs}_{gs}"] = info
            print(arm, fs, gs, json.dumps({k: {a_: b_ for a_, b_ in v.items() if a_ != "p0"} for k, v in info.items()}, default=float)[:420], flush=True)
    (DOCS / "APPLY_V2.json").write_text(json.dumps(log, indent=1, default=float) + "\n")


def extra(frame):
    d = base_frame(frame); nn = d[d.y.eq(0) & d.prev_y.eq(0) & d.g.notna()]; same = (d.m == d.groupby("entity_id", sort=False).m.shift())[nn.index]; on = d[U].astype(str).str.contains("Online")[nn.index]; short = nn.g.lt(600)
    novel = ~d.duplicated(["entity_id", "m"])[nn.index]; n = d[d.y.eq(0)]; la = np.log1p(n.a.clip(lower=0).fillna(0)); cm = la.groupby(n.c).agg(["mean", "size"])
    return dict(same_merchant_gap_lt_10min=float(same[short].mean()), same_merchant_gap_ge_1h=float(same[nn.g.ge(3600)].mean()), online_gap_lt_10min=float(on[short].mean()), novel_gap_lt_10min=float(novel[short].mean()),
                share_gap_lt_10min=float(short.mean()), category_mean_log_amount_sd=float(np.sqrt(np.average((cm["mean"] - la.mean()) ** 2, weights=cm["size"]))))


def cmd_measure():
    from threadpoolctl import threadpool_limits
    threadpool_limits(8)
    import benchmarks.fraud_utility as fu
    import diag_utility_gap as dg
    from diag_utility_gap_b import structure
    from run_ibm_merchant_attr import consistency, merchant_attributes
    t0 = time.time(); dev = pd.read_parquet(PREP / "validation.parquet"); fu.configure(dev); test = fu.features(dev); train = pd.read_parquet(PREP / "train.parquet"); ftrain = fu.features(train); cols = dg.cols_all(); att = merchant_attributes()
    ids = np.sort(train.entity_id.unique()); dets = []
    for i in range(4):                                                # the same real-subset detectors as the diagnosis: their dev top-1% threshold defines "fraud-like"
        pick = np.random.default_rng(dg.SEED + i).choice(ids, size=dg.N_DEV, replace=False); clf = fu.fit(fu.features(train[train.entity_id.isin(pick)]), cols, detector="stable"); dets.append((clf, float(np.quantile(clf.predict_proba(test[cols].to_numpy())[:, 1], 0.99))))
        held = np.random.default_rng(dg.SEED + 100 + i).choice(np.setdiff1d(ids, pick), size=dg.N_DEV, replace=False); h = fu.features(train[train.entity_id.isin(held)]); hn = h[h.y.eq(0)]
        dets[-1] += (float((clf.predict_proba(hn[cols].to_numpy())[:, 1] >= dets[-1][1]).mean()),)
    print("detectors ready", round(time.time() - t0), "s; real fraud-like share", round(float(np.mean([x[2] for x in dets])), 4), flush=True)
    rng = np.random.default_rng(0); rn = ftrain[ftrain.y.eq(0)]; rn = rn.iloc[rng.choice(len(rn), 50000, replace=False)]
    res = dict(real_train=dict(structure=structure(train), extra=extra(train), fraud_amount_vs_med20=float(ftrain[ftrain.y.eq(1)].amount_vs_med20.mean()), fraudlike_normal_share=float(np.mean([x[2] for x in dets]))))
    for arm in ["base", *ARMS2]:
        frames = [pd.read_parquet(BASE(fs, gs) if arm == "base" else OUT / "sim2" / f"{arm}_{fs}" / f"generated_validation_{gs}.parquet") for fs, gs in FITS]; feats = [fu.features(f).assign(src=f"{arm}{i}_") for i, f in enumerate(frames)]
        st = [structure(f) for f in frames]; ex = [extra(f) for f in frames]; s = pd.concat(feats, ignore_index=True); sn = s[s.y.eq(0)]; sn = sn.iloc[rng.choice(len(sn), 50000, replace=False)]; cons = [consistency(f, att) for f in frames]
        fl = [float((dets[i][0].predict_proba(f[f.y.eq(0)][cols].to_numpy())[:, 1] >= dets[i][1]).mean()) for i, f in enumerate(feats)]
        res[arm] = dict(structure={k: float(np.mean([x[k] for x in st])) for k in st[0]}, extra={k: float(np.mean([x[k] for x in ex])) for k in ex[0]}, normal_discriminator=dg.discriminate(rn, sn, cols),
                        fraud_amount_vs_med20=float(s[s.y.eq(1)].amount_vs_med20.mean()), fraudlike_normal_share=float(np.mean(fl)), fraudlike_normal_share_per_generation=fl,
                        normal_category_matches_merchant=float(np.mean([x["normal"]["category_matches_merchant"] for x in cons])), normal_channel_matches_merchant=float(np.mean([x["normal"]["channel_matches_merchant"] for x in cons])))
        print(arm, "fraud-like", round(res[arm]["fraudlike_normal_share"], 4), "disc AUC", round(res[arm]["normal_discriminator"]["auc"], 3), {k: round(v, 3) for k, v in res[arm]["normal_discriminator"]["permutation_auc_drop"].items()},
              "fraud amt", round(res[arm]["fraud_amount_vs_med20"], 3), "cons", round(res[arm]["normal_category_matches_merchant"], 4), round(res[arm]["normal_channel_matches_merchant"], 4), flush=True)
    res["seconds"] = round(time.time() - t0, 1); res["test_read"] = False; (DOCS / "MEASURE_V2.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    keys = ["amount_sd", "amount_sd_within_category", "amount_sd_within_customer_category", "amount_sd_within_customer_merchant", "abs_dlog_amount_vs_previous_visit_same_merchant_median", "same_merchant_as_previous", "same_merchant_within_5",
            "share_rows_top3_merchants", "same_category_as_previous", "distinct_merchants_per_customer_median"]
    pd.set_option("display.width", 250); arms = [k for k, v in res.items() if isinstance(v, dict) and "structure" in v]
    print(pd.concat([pd.DataFrame({k: res[k]["structure"] for k in arms}).loc[keys], pd.DataFrame({k: res[k]["extra"] for k in arms})]).round(3).to_string())


if __name__ == "__main__":
    {"fit": cmd_fit, "apply": cmd_apply, "measure": cmd_measure}[sys.argv[1]]()
