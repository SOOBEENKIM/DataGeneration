"""ibm_normal_rows_v1 (docs/ibm_normal_rows_v1/PROTOCOL.md): CPU post-processing simulation of history-referencing NORMAL rows.
    ./run python scripts/sim_normal_rows.py fit        # train-only tables and facts
    ./run python scripts/sim_normal_rows.py apply      # arms on the adopted model's 4 existing dev generations -> parquet
    ./run python scripts/sim_normal_rows.py measure    # normal-row structure, discriminator, fraud-amount relation
Fraud rows, labels, lengths and novel-merchant rows are never changed. No training, no GPU, no new generation, no test file.
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd

ART = ROOT / "artifacts"; PREP = ART / "ibm_tabformer_v1/prepared"; OUT = ART / "ibm_normal_rows_v1"; DOCS = ROOT / "docs/ibm_normal_rows_v1"
BASE = lambda fs, gs: ART / f"ibm_merchant_marginal_v2/runs/visitrate_{fs}/generated_validation_{gs}.parquet"
FITS = [(fs, gs) for fs in (20261108, 20261109) for gs in (20261011, 20261012)]
ARMS = {"AMT": ("AMT",), "REC": ("REC",), "GAP": ("GAP",), "AMT_REC": ("REC", "AMT"), "ALL": ("REC", "GAP", "AMT")}
A, M, C, G, Y, U = "amount_or_numeric_value", "receiver_or_mark", "category", "gap", "event_is_fraud", "use_chip"
MIN_POOL = 200


def base_frame(frame):
    d = frame.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    d["y"] = pd.to_numeric(d[Y], errors="coerce").fillna(0).astype(int); d["a"] = pd.to_numeric(d[A], errors="coerce").astype(float); d["m"] = d[M].astype(str); d["c"] = d[C].astype(str)
    d["g"] = pd.to_numeric(d[G], errors="coerce").astype(float); d["prev_y"] = d.groupby("entity_id", sort=False).y.shift().fillna(-1).astype(int)
    return d


def amount_roles(d):
    """Among normal rows with a positive amount, in order: first in category / first at merchant (category known) / repeat at merchant."""
    ok = d.y.eq(0) & d.a.gt(0); n = d[ok]
    first_c = n.groupby(["entity_id", "c"], sort=False).cumcount().eq(0); first_m = n.groupby(["entity_id", "m"], sort=False).cumcount().eq(0)
    return ok, n.index[first_c], n.index[first_m & ~first_c], n.index[~first_m]


# ---------------------------------------------------------------- fit (train only)
def cmd_fit():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True); d = base_frame(pd.read_parquet(PREP / "train.parquet"))
    ok, i_c, i_mc, i_m = amount_roles(d); n = d[ok]; la = np.log(n.a)
    anchor_c = la.groupby([n.entity_id, n.c], sort=False).transform("first"); anchor_m = la.groupby([n.entity_id, n.m], sort=False).transform("first")
    pools = dict(first_in_category=pd.DataFrame(dict(c=n.c[i_c], v=n.a[i_c])), new_merchant=pd.DataFrame(dict(c=n.c[i_mc], v=(la - anchor_c)[i_mc])), repeat=pd.DataFrame(dict(c=n.c[i_m], v=(la - anchor_m)[i_m])))
    tables = {}
    for name, p in pools.items():
        cnt = p.c.value_counts(); big = set(cnt[cnt.ge(MIN_POOL)].index); p = p.assign(c=p.c.where(p.c.isin(big), "other")); tables[name] = {k: v.v.round(4).tolist() for k, v in p.groupby("c")}
        if "other" not in tables[name]:
            tables[name]["other"] = p.v.round(4).tolist()
    # recency: among normal rows whose merchant was seen before by the customer and whose previous row is normal, P(same merchant as the previous row | category of the previous row)
    d["seen"] = d.duplicated(["entity_id", "m"]); gp = d.groupby("entity_id", sort=False); d["m_prev"] = gp.m.shift(); d["c_prev"] = gp.c.shift()
    el = d.y.eq(0) & d.prev_y.eq(0) & d.seen; stay = d.m.eq(d.m_prev)[el]; cp = d.c_prev[el]; cnt = cp.value_counts(); big = set(cnt[cnt.ge(MIN_POOL)].index); key = cp.where(cp.isin(big), "other")
    stay_t = stay.groupby(key).mean().to_dict(); stay_t.setdefault("other", float(stay.mean()))
    # gaps: lag-1 rank autocorrelation of normal -> normal gaps within customer; window length; minute resolution
    nn = d.y.eq(0) & d.prev_y.eq(0) & d.g.notna(); x = d[nn]; r = x.groupby("entity_id", sort=False).g.rank(); rs = pd.DataFrame(dict(e=x.entity_id, r=r, rp=r.groupby(x.entity_id, sort=False).shift())).dropna()
    rho = float(rs.groupby("e").apply(lambda q: q.r.corr(q.rp) if len(q) > 5 else np.nan).mean()); phi = float(2 * np.sin(np.pi * rho / 6))
    total = d.groupby("entity_id").g.sum(); minute_share = float((d.g.dropna() % 60 == 0).mean())
    facts = dict(pools={k: {c: len(v) for c, v in t.items()} for k, t in tables.items()}, repeat_delta_quantiles={q: float(pools["repeat"].v.quantile(q)) for q in (.05, .25, .5, .75, .95)}, repeat_delta_exact_zero=float(pools["repeat"].v.abs().lt(1e-9).mean()),
                 new_merchant_delta_quantiles={q: float(pools["new_merchant"].v.quantile(q)) for q in (.05, .25, .5, .75, .95)}, stay_target=stay_t, stay_overall=float(stay.mean()), eligible_rows=int(el.sum()),
                 gap_rank_autocorr=rho, phi=phi, window_seconds_median=float(total.median()), window_days_quantiles={q: float(total.quantile(q) / 86400) for q in (.1, .5, .9)}, gap_multiple_of_60_share=minute_share,
                 round_to_minute=bool(minute_share >= .99), positive_normal_rows=int(ok.sum()), normal_rows=int(d.y.eq(0).sum()), seconds=round(time.time() - t0, 1), test_read=False)
    (OUT / "tables.json").write_text(json.dumps(dict(amount=tables, stay=stay_t, phi=phi, window_seconds=float(total.median()), round_to_minute=facts["round_to_minute"])) + "\n")
    (DOCS / "FIT.json").write_text(json.dumps(facts, indent=1, default=float) + "\n"); print(json.dumps({k: v for k, v in facts.items() if k != "pools"}, indent=1, default=float)); print("pool sizes", {k: sum(v.values()) for k, v in facts["pools"].items()})


# ---------------------------------------------------------------- parts
def apply_rec(d, T, rng):
    """Eligible rows (normal, previous row normal, merchant already seen in the ORIGINAL sequence) copy the previous row's merchant with probability s."""
    seen = d.duplicated(["entity_id", "m"]).to_numpy(); y = d.y.to_numpy(); py = d.prev_y.to_numpy(); el = (y == 0) & (py == 0) & seen
    gp = d.groupby("entity_id", sort=False); same0 = (d.m == gp.m.shift())[el]; cp = gp.c.shift()[el]; key0 = cp.where(cp.isin(set(T["stay"])), "other"); p0 = same0.groupby(key0).mean().to_dict()
    s = {k: float(np.clip((T["stay"][k] - p0.get(k, 0.0)) / max(1e-9, 1 - p0.get(k, 0.0)), 0, 1)) for k in T["stay"]}
    m = d.m.to_numpy().copy(); c = d.c.to_numpy().copy(); u = d[U].astype(str).to_numpy().copy(); r = rng.random(len(d)); stays = 0
    for i in np.flatnonzero(el):                                   # sequential: a stay refers to the (possibly re-drawn) previous row
        if r[i] < s.get(c[i - 1] if c[i - 1] in s else "other", s["other"]):
            if m[i] != m[i - 1]:
                stays += 1
            m[i], c[i], u[i] = m[i - 1], c[i - 1], u[i - 1]
    d = d.copy(); d["m"], d["c"], d[U] = m, c, u; return d, dict(eligible=int(el.sum()), changed=stays, p0=p0, s=s)


def apply_gap(d, T, rng):
    nn = (d.y.eq(0) & d.prev_y.eq(0) & d.g.notna()).to_numpy(); g = d.g.to_numpy().copy(); phi = T["phi"]; e = d.entity_id.to_numpy(); starts = np.flatnonzero(np.r_[True, e[1:] != e[:-1]]); ends = np.r_[starts[1:], len(d)]; scales = []
    for a, b in zip(starts, ends):
        idx = a + np.flatnonzero(nn[a:b])
        if len(idx) > 2:
            z = np.empty(len(idx)); eps = rng.standard_normal(len(idx)); z[0] = eps[0]
            for k in range(1, len(idx)):
                z[k] = phi * z[k - 1] + np.sqrt(1 - phi * phi) * eps[k]
            vals = np.sort(g[idx]); g[idx] = vals[np.argsort(np.argsort(z))]                       # same values, order follows the latent ranks
            fixed = np.nansum(g[a:b]) - g[idx].sum(); room = T["window_seconds"] - fixed
            if room > 0 and g[idx].sum() > 0:
                sc = room / g[idx].sum(); g[idx] = g[idx] * sc; scales.append(sc)
    if T["round_to_minute"]:
        g[nn] = np.round(g[nn] / 60.0) * 60.0
    d = d.copy(); d["g"] = g; return d, dict(scale_quantiles={q: float(np.quantile(scales, q)) for q in (.1, .5, .9)}, rows=int(nn.sum()))


def apply_amt(d, T, rng):
    ok, i_c, i_mc, i_m = amount_roles(d); a = d.a.to_numpy().copy(); tab = T["amount"]

    def draw(name, cats):
        out = np.empty(len(cats)); key = np.where(np.isin(cats, list(tab[name])), cats, "other")
        for k in np.unique(key):
            sel = key == k; pool = np.asarray(tab[name][k]); out[sel] = pool[rng.integers(0, len(pool), int(sel.sum()))]
        return out
    c = d.c.to_numpy(); a[i_c] = draw("first_in_category", c[i_c]); d = d.copy(); d["a"] = a
    n = d[ok]; anchor_c = n.groupby(["entity_id", "c"], sort=False).a.transform("first"); a[i_mc] = anchor_c[i_mc].to_numpy() * np.exp(draw("new_merchant", c[i_mc])); d["a"] = a
    n = d[ok]; anchor_m = n.groupby(["entity_id", "m"], sort=False).a.transform("first"); a[i_m] = anchor_m[i_m].to_numpy() * np.exp(draw("repeat", c[i_m]))
    a[ok.to_numpy()] = np.maximum(0.01, np.round(a[ok.to_numpy()], 2)); d["a"] = a
    return d, dict(first_in_category=int(len(i_c)), new_merchant=int(len(i_mc)), repeat=int(len(i_m)), untouched_normal=int((d.y.eq(0) & ~ok).sum()))


def finish(d, like):
    out = d[["entity_id", "event_index"]].copy()
    out[G] = d.g.to_numpy(); out[M] = d.m.to_numpy(); out[A] = d.a.to_numpy(); out[C] = d.c.to_numpy(); out[U] = d[U].to_numpy(); out[Y] = d[Y].to_numpy()
    return out[list(like.columns)]


def cmd_apply():
    T = json.loads((OUT / "tables.json").read_text()); log = {}
    for fs, gs in FITS:
        raw = pd.read_parquet(BASE(fs, gs)); base = base_frame(raw)
        for arm, parts in ARMS.items():
            rng = np.random.default_rng([20261111, fs, gs, sorted(ARMS).index(arm)]); d = base.copy(); info = {}
            for p in parts:
                d, info[p] = {"REC": apply_rec, "GAP": apply_gap, "AMT": apply_amt}[p](d, T, rng)
            folder = OUT / "sim" / f"{arm}_{fs}"; folder.mkdir(parents=True, exist_ok=True); out = finish(d, raw.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True))
            ref = raw.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True); fr = pd.to_numeric(ref[Y], errors="coerce").eq(1).to_numpy()
            assert len(out) == len(ref) and out[Y].astype(str).equals(ref[Y].astype(str)), "labels changed"
            for col in (M, C, U):
                assert (out.loc[fr, col].astype(str).to_numpy() == ref.loc[fr, col].astype(str).to_numpy()).all(), f"fraud rows changed: {col}"
            for col in (G, A):
                assert np.allclose(pd.to_numeric(out.loc[fr, col], errors="coerce").to_numpy(dtype=float), pd.to_numeric(ref.loc[fr, col], errors="coerce").to_numpy(dtype=float), equal_nan=True), f"fraud rows changed: {col}"
            out.to_parquet(folder / f"generated_validation_{gs}.parquet", index=False); log[f"{arm}_{fs}_{gs}"] = info; print(arm, fs, gs, json.dumps({k: {a: b for a, b in v.items() if a not in ("p0", "s")} for k, v in info.items()}, default=float)[:300], flush=True)
    (DOCS / "APPLY.json").write_text(json.dumps(log, indent=1, default=float) + "\n")


# ---------------------------------------------------------------- measure
def cmd_measure():
    import benchmarks.fraud_utility as fu
    import diag_utility_gap as dg
    from diag_utility_gap_b import structure
    from run_ibm_merchant_attr import consistency, merchant_attributes
    from threadpoolctl import threadpool_limits
    threadpool_limits(8)                                             # a shared machine: HistGradientBoosting with every core stalls under load (first run, recorded in RESULTS)
    t0 = time.time(); dev = pd.read_parquet(PREP / "validation.parquet"); fu.configure(dev); train = pd.read_parquet(PREP / "train.parquet"); ftrain = fu.features(train); cols = dg.cols_all(); att = merchant_attributes()
    rng = np.random.default_rng(0); rn = ftrain[ftrain.y.eq(0)]; rn = rn.iloc[rng.choice(len(rn), 50000, replace=False)]; res = dict(real_train=dict(structure=structure(train), fraud_amount_vs_med20=float(ftrain[ftrain.y.eq(1)].amount_vs_med20.mean()),
                                                                                                                                                     window_days={q: float((pd.to_numeric(train.gap, errors="coerce").groupby(train.entity_id).sum() / 86400).quantile(q)) for q in (.1, .5, .9)}))
    for arm in ["base", *ARMS]:
        frames = [pd.read_parquet(BASE(fs, gs) if arm == "base" else OUT / "sim" / f"{arm}_{fs}" / f"generated_validation_{gs}.parquet") for fs, gs in FITS]
        print(arm, "start", round(time.time() - t0), "s", flush=True); st = [structure(f) for f in frames]; feats = [fu.features(f).assign(src=f"{arm}{i}_") for i, f in enumerate(frames)]; s = pd.concat(feats, ignore_index=True); sn = s[s.y.eq(0)]; sn = sn.iloc[rng.choice(len(sn), 50000, replace=False)]
        cons = [consistency(f, att) for f in frames]; days = pd.concat([pd.to_numeric(f.gap, errors="coerce").groupby(f.entity_id).sum() / 86400 for f in frames])
        res[arm] = dict(structure={k: float(np.mean([x[k] for x in st])) for k in st[0]}, normal_discriminator=dg.discriminate(rn, sn, cols), fraud_amount_vs_med20=float(s[s.y.eq(1)].amount_vs_med20.mean()),
                        normal_category_matches_merchant=float(np.mean([x["normal"]["category_matches_merchant"] for x in cons])), normal_channel_matches_merchant=float(np.mean([x["normal"]["channel_matches_merchant"] for x in cons])),
                        window_days={q: float(days.quantile(q)) for q in (.1, .5, .9)}, gap_multiple_of_60_share=float(np.mean([(pd.to_numeric(f.gap, errors="coerce").dropna() % 60 == 0).mean() for f in frames])))
        print(arm, "disc AUC", round(res[arm]["normal_discriminator"]["auc"], 3), {k: round(v, 3) for k, v in res[arm]["normal_discriminator"]["permutation_auc_drop"].items()}, "fraud amt", round(res[arm]["fraud_amount_vs_med20"], 3), "days", res[arm]["window_days"], flush=True)
    res["seconds"] = round(time.time() - t0, 1); res["test_read"] = False; (DOCS / "MEASURE.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    keys = ["amount_sd_within_customer_merchant", "abs_dlog_amount_vs_previous_visit_same_merchant_median", "exact_same_amount_as_previous_visit_same_merchant", "amount_sd_within_category", "amount_sd_within_customer_category", "amount_sd",
            "same_merchant_as_previous", "same_merchant_within_5", "share_rows_top3_merchants", "same_category_as_previous", "gap_lag1_autocorr", "gap_sd", "distinct_merchants_per_customer_median"]
    pd.set_option("display.width", 250); print(pd.DataFrame({k: v["structure"] for k, v in res.items() if isinstance(v, dict) and "structure" in v}).loc[keys].round(3).to_string())


if __name__ == "__main__":
    {"fit": cmd_fit, "apply": cmd_apply, "measure": cmd_measure}[sys.argv[1]]()
