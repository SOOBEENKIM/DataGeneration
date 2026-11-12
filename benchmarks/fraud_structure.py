"""fraud_structure_v1: structure of fraud in per-customer transaction sequences, in four layers
(A label process, B within-episode outputs, C around-episode precursors, D customer level).

`structure_stats(frame, ctx, ts)` describes one table; `distances(real_stats, gen_stats)` compares two.
The same function on two real tables gives the noise floor. Nothing here feeds generation.
"""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from benchmarks.rare_state_persistence import prepare as _prepare, AMOUNT, MERCHANT, LABEL
from benchmarks.argn_state_evaluation import episode_features

TOP_MERCHANTS = 10
TOP_CATEGORIES = 3
PRE_ROWS = 5
NIGHT_HOURS = (0, 6)


def _w1(a, b, log=False):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if log:
        a = a[a >= 0]; b = b[b >= 0]; a = np.log1p(a); b = np.log1p(b)
    if not len(a) or not len(b):
        return np.nan
    return float(wasserstein_distance(a, b))


def _tv(p, q):
    p = pd.Series(p, dtype=float); q = pd.Series(q, dtype=float)
    return float(p.subtract(q, fill_value=0).abs().sum() / 2)


def prepare(frame):
    d = _prepare(frame)
    d, _ = episode_features(d)
    d = d.reset_index(drop=True)
    g = d.groupby("entity_id", sort=False)
    d["t"] = g.gap.cumsum().fillna(0)  # within-sequence clock (seconds from first event)
    d["length"] = g.event_index.transform("size")
    d["relpos"] = d.event_index / d.length.clip(lower=1)
    # strict-past novelty of merchant / category / channel for this customer
    d["novel_merchant"] = (~d.duplicated(["entity_id", MERCHANT])).astype(int)
    d["novel_category"] = (~d.duplicated(["entity_id", "category"])).astype(int)
    if "use_chip" in d:
        d["novel_channel"] = (~d.duplicated(["entity_id", "use_chip"])).astype(int)
    # strict-past personal medians (expanding, shifted)
    d["past_amount_median"] = g[AMOUNT].transform(lambda x: x.shift().expanding(min_periods=5).median())
    d["past_gap_median"] = g.gap.transform(lambda x: x.shift().expanding(min_periods=5).median())
    # 24h-tolerant episodes over fraud rows
    f = d.fraud.eq(1)
    ff = d[f]
    new = ff.entity_id.ne(ff.entity_id.shift()) | ff.t.diff().gt(86400)
    d["ep_id"] = np.nan
    d.loc[f, "ep_id"] = new.cumsum().astype(float)
    return d


def _episode_table(d):
    f = d[d.fraud.eq(1)].copy()
    if not len(f):
        return pd.DataFrame(columns=["ep_id", "entity_id", "n", "t0", "t1", "first_amount", "last_amount", "amount_cv", "distinct_merchants", "relpos0", "first_index"])
    f["_m"] = f[MERCHANT].astype(str)
    ep = f.groupby("ep_id", sort=False).agg(
        entity_id=("entity_id", "first"), n=("fraud", "size"), t0=("t", "first"), t1=("t", "last"),
        first_amount=(AMOUNT, "first"), last_amount=(AMOUNT, "last"),
        amount_cv=(AMOUNT, lambda x: float(x.std(ddof=0) / x.mean()) if len(x) >= 3 and x.mean() > 0 else np.nan),
        distinct_merchants=("_m", "nunique"), relpos0=("relpos", "first"), first_index=("event_index", "first"),
    ).reset_index()
    return ep


def structure_stats(frame, ctx=None, ctx_numeric=None):
    """frame: events with REQUIRED columns and optionally a `timestamp` column (absolute seconds; hour-of-day
    statistics are computed only when present); ctx: customer context (customer_id + numeric columns) or None."""
    d = prepare(frame)
    ts = d["timestamp"] if "timestamp" in d else None
    f = d[d.fraud.eq(1)]; n = d[d.fraud.eq(0)]
    ep = _episode_table(d)
    s = {"n_events": len(d), "n_customers": int(d.entity_id.nunique()), "n_fraud": int(len(f)),
         "n_fraud_customers": int(f.entity_id.nunique()), "n_episodes24h": int(len(ep))}
    arr = {}
    # ---- A label process -------------------------------------------------------------------------------
    ep = ep.sort_values(["entity_id", "t0"], kind="stable")
    same = ep.entity_id.eq(ep.entity_id.shift())
    arr["A_inter_episode_gap_hours"] = ((ep.t0 - ep.t1.shift()) / 3600.0)[same].to_numpy()
    arr["A_episode_start_relpos"] = ep.relpos0.to_numpy()
    per = d.groupby("entity_id", sort=False).agg(length=("fraud", "size"), frauds=("fraud", lambda x: int(x.eq(1).sum())))
    fc = per.frauds.gt(0)
    s["A_fraud_customer_length_ratio"] = float(per.length[fc].median() / per.length[~fc].median()) if fc.any() and (~fc).any() else np.nan
    # ---- B within-episode outputs ----------------------------------------------------------------------
    long = ep[ep.n.ge(3)]
    arr["B_amount_escalation_log"] = np.log((long.last_amount.clip(lower=.01)) / (long.first_amount.clip(lower=.01))).to_numpy()
    s["B_amount_escalation_median"] = float(np.median(arr["B_amount_escalation_log"])) if len(long) else np.nan
    arr["B_amount_cv"] = long.amount_cv.dropna().to_numpy()
    # merchant repetition inside an episode (rows after the first of the episode)
    if len(f):
        fm = f.assign(_m=f[MERCHANT].astype(str), _c=f.category.astype(str))
        rep_m = fm.duplicated(["ep_id", "_m"]) ; rep_c = fm.duplicated(["ep_id", "_c"])
        later = fm.duplicated(["ep_id"])  # not the first row of its episode
        s["B_merchant_repeat_within_episode"] = float(rep_m[later].mean()) if later.any() else np.nan
        s["B_category_repeat_within_episode"] = float(rep_c[later].mean()) if later.any() else np.nan
    else:
        s["B_merchant_repeat_within_episode"] = s["B_category_repeat_within_episode"] = np.nan
    multi = ep[ep.n.ge(2)]
    arr["B_distinct_merchants_per_row"] = (multi.distinct_merchants / multi.n).to_numpy()
    for col, key in [("novel_merchant", "B_novel_merchant"), ("novel_category", "B_novel_category"), ("novel_channel", "B_novel_channel")]:
        if col in d:
            s[f"{key}_fraud"] = float(f[col].mean()) if len(f) else np.nan
            s[f"{key}_normal"] = float(n[col].mean()) if len(n) else np.nan
    arr["B_within_run_gap_minutes"] = (f.loc[f.run_age.ge(2), "gap"] / 60.0).dropna().to_numpy()
    s["B_within_run_gap_median_minutes"] = float(np.median(arr["B_within_run_gap_minutes"])) if len(arr["B_within_run_gap_minutes"]) else np.nan
    s["B_normal_gap_median_minutes"] = float((n.loc[n.event_index.gt(0), "gap"] / 60.0).median()) if len(n) else np.nan
    if ts is not None:
        hour = ((pd.to_numeric(ts, errors="coerce") % 86400) // 3600)
        hf = hour[f.index].dropna().astype(int).value_counts(normalize=True).reindex(range(24), fill_value=0)
        hn = hour[n.index].dropna().astype(int).value_counts(normalize=True).reindex(range(24), fill_value=0)
        arr["B_hour_fraud_dist"] = hf.to_numpy(); arr["B_hour_normal_dist"] = hn.to_numpy()
        s["B_hour_tv_fraud_vs_normal"] = _tv(hf, hn)
        night = list(range(NIGHT_HOURS[0], NIGHT_HOURS[1]))
        s["B_night_share_fraud"] = float(hf.loc[night].sum()); s["B_night_share_normal"] = float(hn.loc[night].sum())
    # ---- C around the episode --------------------------------------------------------------------------
    onset = d[d.fraud.eq(1) & d.run_age.eq(1) & d.event_index.gt(0)]
    pre_amt, pre_novel, onset_amt, post_gap = [], [], [], []
    idx = d.index.to_numpy()
    for i in onset.index:
        eid = d.at[i, "entity_id"]
        lo = max(0, i - PRE_ROWS); window = d.iloc[lo:i]
        window = window[window.entity_id.eq(eid) & window.fraud.eq(0)]
        pm = d.at[i, "past_amount_median"]
        if len(window) and np.isfinite(pm) and pm > 0:
            pre_amt.append(float(window[AMOUNT].median() / pm))
            pre_novel.append(float(window.novel_merchant.mean()))
        if np.isfinite(pm) and pm > 0:
            onset_amt.append(float(d.at[i, AMOUNT] / pm))
    # first normal row after a fraud run
    run_end = d[d.fraud.eq(1) & d.next_fraud.eq(0)].index + 1
    run_end = run_end[run_end < len(d)]
    after = d.loc[run_end]
    pg = after.past_gap_median
    post_gap = (after.gap / pg.where(pg.gt(0)))[after.fraud.eq(0)].dropna().to_numpy()
    arr["C_preonset_amount_ratio"] = np.asarray(pre_amt); arr["C_onset_amount_ratio"] = np.asarray(onset_amt)
    arr["C_post_episode_gap_ratio"] = post_gap
    s["C_preonset_novel_merchant_share"] = float(np.mean(pre_novel)) if len(pre_novel) else np.nan
    s["C_normal_novel_merchant_share"] = s.get("B_novel_merchant_normal", np.nan)
    s["C_onset_amount_ratio_median"] = float(np.median(onset_amt)) if len(onset_amt) else np.nan
    s["C_preonset_amount_ratio_median"] = float(np.median(pre_amt)) if len(pre_amt) else np.nan
    s["C_post_episode_gap_ratio_median"] = float(np.median(post_gap)) if len(post_gap) else np.nan
    # ---- D customer level ------------------------------------------------------------------------------
    s["D_fraud_customers"] = sorted(int(x) for x in f.entity_id.unique())
    if ctx is not None and ctx_numeric:
        c = ctx.set_index("customer_id").reindex(per.index)
        for col in ctx_numeric:
            x = pd.to_numeric(c[col], errors="coerce")
            a, b = x[fc.to_numpy()].dropna(), x[(~fc).to_numpy()].dropna()
            sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else np.nan
            s[f"D_ctx_smd_{col}"] = float((a.mean() - b.mean()) / sd) if sd and np.isfinite(sd) and sd > 0 else np.nan
    if len(f):
        fm = f[MERCHANT].astype(str).value_counts(normalize=True); nm = n[MERCHANT].astype(str).value_counts(normalize=True)
        s["D_top_merchant_share_fraud"] = float(fm.head(TOP_MERCHANTS).sum()); s["D_top_merchant_share_normal"] = float(nm.head(TOP_MERCHANTS).sum())
        per_m = f.assign(_m=f[MERCHANT].astype(str)).groupby("_m").entity_id.nunique()
        s["D_shared_fraud_merchant_share"] = float(per_m.ge(2).mean())
        per_n = n.assign(_m=n[MERCHANT].astype(str)).groupby("_m").entity_id.nunique()
        s["D_shared_merchant_share_normal"] = float(per_n.ge(2).mean())
        fcat = f.category.astype(str).value_counts(normalize=True); ncat = n.category.astype(str).value_counts(normalize=True)
        s["D_top_category_share_fraud"] = float(fcat.head(TOP_CATEGORIES).sum()); s["D_top_category_share_normal"] = float(ncat.head(TOP_CATEGORIES).sum())
        s["D_top_categories_fraud"] = list(fcat.head(TOP_CATEGORIES).index)
    else:
        for k in ["D_top_merchant_share_fraud", "D_top_merchant_share_normal", "D_shared_fraud_merchant_share", "D_shared_merchant_share_normal", "D_top_category_share_fraud", "D_top_category_share_normal"]:
            s[k] = np.nan
    s["_arrays"] = arr
    return s


ARRAY_DISTANCES = {  # key -> log?
    "A_inter_episode_gap_hours": True, "A_episode_start_relpos": False, "B_amount_escalation_log": False, "B_amount_cv": False,
    "B_distinct_merchants_per_row": False, "B_within_run_gap_minutes": True, "C_preonset_amount_ratio": True,
    "C_onset_amount_ratio": True, "C_post_episode_gap_ratio": True,
}
SCALAR_DISTANCES = [  # absolute differences of rates / ratios
    "A_fraud_customer_length_ratio", "B_merchant_repeat_within_episode", "B_category_repeat_within_episode",
    "B_novel_merchant_fraud", "B_novel_category_fraud", "B_novel_channel_fraud", "C_preonset_novel_merchant_share",
    "D_top_merchant_share_fraud", "D_shared_fraud_merchant_share", "D_top_category_share_fraud", "B_night_share_fraud",
]


def distances(real, gen):
    out = {}
    for k, lg in ARRAY_DISTANCES.items():
        out[f"{k}_w1"] = _w1(real["_arrays"].get(k, []), gen["_arrays"].get(k, []), log=lg)
    for k in SCALAR_DISTANCES:
        if k in real and k in gen:
            out[f"{k}_absdiff"] = abs(float(gen[k]) - float(real[k])) if np.isfinite(real[k]) and np.isfinite(gen[k]) else np.nan
    for k in list(real):
        if k.startswith("D_ctx_smd_") and k in gen:
            out[f"{k}_absdiff"] = abs(float(gen[k]) - float(real[k])) if np.isfinite(real[k]) and np.isfinite(gen[k]) else np.nan
    if "B_hour_fraud_dist" in real["_arrays"] and "B_hour_fraud_dist" in gen["_arrays"]:
        out["B_hour_fraud_tv"] = _tv(real["_arrays"]["B_hour_fraud_dist"], gen["_arrays"]["B_hour_fraud_dist"])
    # same-customer fraud assignment (meaningful only when gen conditions on the same customers)
    a, b = set(real["D_fraud_customers"]), set(gen["D_fraud_customers"])
    out["D_fraud_customer_jaccard"] = float(len(a & b) / len(a | b)) if a | b else np.nan
    return out


def add_anchored_timestamp(frame, real_first_ts):
    """Approximate absolute time of generated events: the real first-event time of the same customer (Series indexed by
    entity_id) plus the cumulative gap. Customers without a real anchor get NaN (excluded from hour statistics)."""
    d = frame.sort_values(["entity_id", "event_index"], kind="stable").copy()
    t = pd.to_numeric(d.gap, errors="coerce").groupby(d.entity_id, sort=False).cumsum().fillna(0)
    t0 = d.entity_id.map(real_first_ts)
    d["timestamp"] = (t0 + t).where(t0.notna())
    return d
