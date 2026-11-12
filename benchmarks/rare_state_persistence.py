"""Stage-1 metrics: standard aggregate fidelity next to rare-state (fraud episode) dynamics.

Every metric compares a generated development table with the real development table.
The same function applied to two real tables gives the real-vs-real noise floor.
Labels are outcomes of the generator; nothing here is fed back into generation.
"""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from benchmarks.argn_state_evaluation import episode_features, hazard_table, AGE_BANDS

AMOUNT = "amount_or_numeric_value"
MERCHANT = "receiver_or_mark"
LABEL = "event_is_fraud"
REQUIRED = ["entity_id", "event_index", "gap", "category", LABEL]


def prepare(frame):
    """Sort, normalise dtypes, add fraud int column; tolerate missing amount/merchant."""
    d = frame.copy()
    for c in [AMOUNT, MERCHANT]:
        if c not in d:
            d[c] = np.nan
    missing = [c for c in REQUIRED if c not in d]
    if missing:
        raise ValueError(f"missing columns {missing}")
    d = d.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    d["gap"] = pd.to_numeric(d.gap, errors="coerce").astype(float)
    d[AMOUNT] = pd.to_numeric(d[AMOUNT], errors="coerce").astype(float)
    label = pd.to_numeric(d[LABEL], errors="coerce")
    d["fraud"] = label.where(label.isin([0, 1]), -1).astype(int)
    d["seen_merchant"] = d.duplicated(["entity_id", MERCHANT]).astype(int)
    return d


def _tv(a, b, cols):
    if not len(a) or not len(b):
        return np.nan
    x = a[cols].astype(str).value_counts(normalize=True)
    y = b[cols].astype(str).value_counts(normalize=True)
    return float(x.subtract(y, fill_value=0).abs().sum() / 2)


def _logw1(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a) & (a >= 0)]; b = b[np.isfinite(b) & (b >= 0)]
    if not len(a) or not len(b):
        return np.nan
    return float(wasserstein_distance(np.log1p(a), np.log1p(b)))


def _quantile_bins(real_values, q=10):
    v = np.asarray(real_values, dtype=float); v = v[np.isfinite(v)]
    return np.unique(np.quantile(v, np.linspace(0, 1, q + 1)[1:-1]))


def standard_metrics(real, gen):
    """Aggregate fidelity a conventional tabular-synthesis report would show."""
    r, g = real, gen
    rows = {}
    rows["S_fraud_rate_real"] = float(r.fraud.eq(1).mean())
    rows["S_fraud_rate_gen"] = float(g.fraud.eq(1).mean())
    rows["S_fraud_rate_absdiff"] = abs(rows["S_fraud_rate_gen"] - rows["S_fraud_rate_real"])
    rows["S_category_tv"] = _tv(r, g, ["category"])
    rows["S_amount_logw1"] = _logw1(r[AMOUNT], g[AMOUNT])
    rows["S_gap_logw1"] = _logw1(r.loc[r.event_index.gt(0), "gap"], g.loc[g.event_index.gt(0), "gap"])
    rl = r.groupby("entity_id").size(); gl = g.groupby("entity_id").size()
    rows["S_length_logw1"] = _logw1(rl, gl)
    rm = r.groupby("entity_id")[MERCHANT].nunique().median(); gm = g.groupby("entity_id")[MERCHANT].nunique().median()
    rows["S_merchants_per_customer_reldiff"] = float(abs(gm - rm) / rm) if rm and np.isfinite(gm) else np.nan
    # Pairwise relations on real-quantile bins (label included as one ordinary column).
    ae = _quantile_bins(r[AMOUNT]); ge = _quantile_bins(r.loc[r.event_index.gt(0), "gap"])
    for d in (r, g):
        d["_abin"] = np.searchsorted(ae, d[AMOUNT].to_numpy(), side="left")
        d["_gbin"] = np.searchsorted(ge, d.gap.fillna(-1).to_numpy(), side="left")
    rows["S_pair_category_amount_tv"] = _tv(r, g, ["category", "_abin"])
    rows["S_pair_gap_amount_tv"] = _tv(r, g, ["_gbin", "_abin"])
    rows["S_pair_category_label_tv"] = _tv(r, g, ["category", "fraud"])
    rows["S_pair_label_amount_tv"] = _tv(r, g, ["fraud", "_abin"])
    return rows


def _runs(d):
    e, runs = episode_features(d)
    known_next = e.next_fraud.isin([0, 1])
    normal = e.fraud.eq(0) & known_next
    fraud = e.fraud.eq(1) & known_next
    per = e.groupby("entity_id", sort=False).fraud.agg(events="size", frauds=lambda x: x.eq(1).sum())
    s = dict(
        continuation_rate=float((fraud & e.next_fraud.eq(1)).sum() / fraud.sum()) if fraud.sum() else np.nan,
        onset_rate=float((normal & e.next_fraud.eq(1)).sum() / normal.sum()) if normal.sum() else np.nan,
        fraud_runs=int(len(runs)),
        mean_run_length=float(runs.length.mean()) if len(runs) else np.nan,
        p90_run_length=float(runs.length.quantile(.9)) if len(runs) else np.nan,
        customers_with_fraud=int(per.frauds.gt(0).sum()),
        runs_per_fraud_customer=float(len(runs) / per.frauds.gt(0).sum()) if per.frauds.gt(0).sum() else np.nan,
        fraud_share_of_fraud_customers=float((per.frauds / per.events)[per.frauds.gt(0)].mean()) if per.frauds.gt(0).sum() else np.nan,
        first_event_fraud_rate=float(e.loc[e.event_index.eq(0), "fraud"].eq(1).mean()),
        age1_termination_rate=float((e.fraud.eq(1) & e.run_age.eq(1) & e.next_fraud.eq(0)).sum()
                                    / (e.fraud.eq(1) & e.run_age.eq(1) & known_next).sum()) if (e.fraud.eq(1) & e.run_age.eq(1) & known_next).sum() else np.nan,
        runs_over_20_share=float(runs.length.gt(20).mean()) if len(runs) else np.nan,
        span_hours=(runs.span_seconds / 3600.0).to_numpy() if len(runs) else np.array([]),
        lengths=runs.length.to_numpy() if len(runs) else np.array([]),
    )
    return e, runs, s


def rare_state_metrics(real, gen):
    """Fraud-episode dynamics: persistence, onset, exit, and within-state outputs."""
    er, rr, sr = _runs(real)
    eg, rg, sg = _runs(gen)
    rows = {}
    for k in ["continuation_rate", "onset_rate", "mean_run_length", "p90_run_length",
              "customers_with_fraud", "runs_per_fraud_customer", "fraud_share_of_fraud_customers", "fraud_runs"]:
        rows[f"R_{k}_real"] = sr[k]; rows[f"R_{k}_gen"] = sg[k]
    rows["R_continuation_absdiff"] = abs(sg["continuation_rate"] - sr["continuation_rate"])
    rows["R_onset_rate_reldiff"] = abs(sg["onset_rate"] - sr["onset_rate"]) / sr["onset_rate"] if sr["onset_rate"] else np.nan
    rows["R_mean_run_length_ratio"] = sg["mean_run_length"] / sr["mean_run_length"] if sr["mean_run_length"] else np.nan
    rows["R_run_length_logw1"] = _logw1(sr["lengths"], sg["lengths"])
    rows["R_run_span_hours_logw1"] = _logw1(sr["span_hours"], sg["span_hours"])
    rows["R_customers_with_fraud_reldiff"] = abs(sg["customers_with_fraud"] - sr["customers_with_fraud"]) / sr["customers_with_fraud"]
    # Added 2026-11-07 after native_argn_episode_probe_v1: boundary failures the hazard table hides.
    for k in ["first_event_fraud_rate", "age1_termination_rate", "runs_over_20_share"]:
        rows[f"R_{k}_real"] = sr[k]; rows[f"R_{k}_gen"] = sg[k]
    rows["R_first_event_fraud_absdiff"] = abs(sg["first_event_fraud_rate"] - sr["first_event_fraud_rate"])
    rows["R_age1_termination_absdiff"] = abs(sg["age1_termination_rate"] - sr["age1_termination_rate"])
    rows["R_runs_per_fraud_customer_absdiff"] = abs(sg["runs_per_fraud_customer"] - sr["runs_per_fraud_customer"])
    fr = er[er.fraud.eq(1)]; fg = eg[eg.fraud.eq(1)]
    nr = er[er.fraud.eq(0)]; ng = eg[eg.fraud.eq(0)]
    rows["R_fraud_gap_logw1"] = _logw1(fr.loc[fr.event_index.gt(0), "gap"], fg.loc[fg.event_index.gt(0), "gap"])
    rows["R_normal_gap_logw1"] = _logw1(nr.loc[nr.event_index.gt(0), "gap"], ng.loc[ng.event_index.gt(0), "gap"])
    rows["R_fraud_amount_logw1"] = _logw1(fr[AMOUNT], fg[AMOUNT])
    rows["R_normal_amount_logw1"] = _logw1(nr[AMOUNT], ng[AMOUNT])
    rows["R_fraud_category_tv"] = _tv(fr, fg, ["category"])
    rows["R_normal_category_tv"] = _tv(nr, ng, ["category"])
    # Gap into the first fraud event of a run relative to the customer's strict-past median gap.
    rows["R_onset_personal_gap_logw1"] = _logw1(_onset_personal(er), _onset_personal(eg))
    # Added 2026-11-08 (ibm_tabformer_v1 registration): episodes tolerant to interleaved normal events,
    # and the channel of fraud events when the table has one (IBM use_chip; NaN for Sparkov).
    e24r, e24g = _episodes_24h(er), _episodes_24h(eg)
    rows["R_episode24h_per_fraud_customer_real"] = e24r; rows["R_episode24h_per_fraud_customer_gen"] = e24g
    rows["R_episode24h_per_fraud_customer_absdiff"] = abs(e24g - e24r) if np.isfinite(e24r) and np.isfinite(e24g) else np.nan
    if "use_chip" in er and "use_chip" in eg:
        onr = fr.use_chip.astype(str).eq("Online Transaction").mean() if len(fr) else np.nan
        ong = fg.use_chip.astype(str).eq("Online Transaction").mean() if len(fg) else np.nan
    else:
        onr = ong = np.nan
    rows["R_fraud_online_share_real"] = float(onr); rows["R_fraud_online_share_gen"] = float(ong)
    rows["R_fraud_online_share_absdiff"] = abs(ong - onr) if np.isfinite(onr) and np.isfinite(ong) else np.nan
    hz = pd.DataFrame(hazard_table(er, "real")).merge(
        pd.DataFrame(hazard_table(eg, "gen")), on=["age_band", "label"], suffixes=("_real", "_gen"))
    return rows, hz


def _episodes_24h(e, tolerance_seconds=86400):
    """Fraud episodes per fraud customer where consecutive fraud events of a customer that are at most
    `tolerance_seconds` apart (by cumulative gap) belong to one episode, whatever lies between them."""
    f = e[e.fraud.eq(1)]
    if not len(f):
        return np.nan
    t = e.gap.fillna(0).groupby(e.entity_id, sort=False).cumsum()[f.index]
    new = f.entity_id.ne(f.entity_id.shift()) | t.diff().gt(tolerance_seconds)
    return float(new.sum() / f.entity_id.nunique())


def _onset_personal(e):
    prior = e.groupby("entity_id", sort=False).gap.transform(lambda x: x.shift().rolling(20, min_periods=5).median())
    onset = e.fraud.eq(1) & e.run_age.eq(1) & e.event_index.gt(0)
    ratio = (e.gap / prior.where(prior.gt(0)))[onset]
    return ratio.dropna().to_numpy()


def all_metrics(real, gen):
    real = prepare(real); gen = prepare(gen)
    rows = standard_metrics(real, gen)
    rare, hazard = rare_state_metrics(real, gen)
    rows.update(rare)
    rows["events_real"] = len(real); rows["events_gen"] = len(gen)
    rows["customers_real"] = real.entity_id.nunique(); rows["customers_gen"] = gen.entity_id.nunique()
    return rows, hazard


DIFF_COLUMNS = [
    "S_fraud_rate_absdiff", "S_category_tv", "S_amount_logw1", "S_gap_logw1", "S_length_logw1",
    "S_merchants_per_customer_reldiff", "S_pair_category_amount_tv", "S_pair_gap_amount_tv",
    "S_pair_category_label_tv", "S_pair_label_amount_tv",
    "R_continuation_absdiff", "R_onset_rate_reldiff", "R_run_length_logw1", "R_run_span_hours_logw1",
    "R_customers_with_fraud_reldiff", "R_fraud_gap_logw1", "R_normal_gap_logw1", "R_fraud_amount_logw1",
    "R_normal_amount_logw1", "R_fraud_category_tv", "R_normal_category_tv", "R_onset_personal_gap_logw1",
    "R_first_event_fraud_absdiff", "R_age1_termination_absdiff", "R_runs_per_fraud_customer_absdiff",
    "R_episode24h_per_fraud_customer_absdiff", "R_fraud_online_share_absdiff",  # 2026-11-08
]
