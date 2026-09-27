"""Boundary-aware observed fraud episodes and customer-cluster diagnostics."""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

AMOUNT = "amount_or_numeric_value"
MERCHANT = "receiver_or_mark"
AGE_BANDS = ["1", "2-5", "6-10", "11-20", "21-50", "51+"]


def episode_features(frame):
    """Input is sorted extended transaction features; add no future model inputs."""
    d = frame.sort_values(["entity_id", "event_index"], kind="stable").copy().reset_index(drop=True)
    assert not d.duplicated(["entity_id", "event_index"]).any()
    d["next_fraud"] = d.groupby("entity_id", sort=False).fraud.shift(-1)
    boundary = d.entity_id.ne(d.entity_id.shift()) | d.fraud.ne(d.fraud.shift())
    d["run_id"] = boundary.cumsum()
    d["run_age"] = d.groupby("run_id", sort=False).cumcount() + 1
    d["age_band"] = np.searchsorted([1, 5, 10, 20, 50], d.run_age, side="left")
    d["left_boundary_run"] = d.groupby("run_id", sort=False).event_index.transform("min").eq(0)
    d["right_boundary_row"] = d["next_fraud"].isna()
    # The gap INTO a run does not contribute to its within-run observed span.
    within = d.gap.where(d.run_age.gt(1), 0).fillna(0)
    d["run_elapsed_seconds"] = within.groupby(d.run_id, sort=False).cumsum()
    f = d[d.fraud.eq(1)]
    runs = f.groupby("run_id", sort=False).agg(
        entity_id=("entity_id", "first"), length=("run_age", "max"),
        span_seconds=("run_elapsed_seconds", "max"),
        left_boundary=("left_boundary_run", "first"),
        right_boundary=("right_boundary_row", "any"),
        last_next_label=("next_fraud", lambda x: x.iloc[-1]),
    ).reset_index(drop=True)
    runs["known_normal_end"] = runs.last_next_label.eq(0)
    runs["completed_known_start"] = ~runs.left_boundary & runs.known_normal_end
    return d, runs


def safe_rate(num, den):
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.full(np.broadcast_shapes(num.shape, den.shape), np.nan), where=den > 0)


def log_w1(a, b, weights=None):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not len(a) or not len(b):
        return np.nan
    return float(wasserstein_distance(np.log1p(a), np.log1p(b), v_weights=weights))


def scalar_summary(d, runs):
    known_next = d.next_fraud.isin([0, 1])
    normal = d.fraud.eq(0) & known_next
    fraud = d.fraud.eq(1) & known_next
    lengths = d.groupby("entity_id", sort=False).size()
    per_customer = d.groupby("entity_id", sort=False).fraud.agg(
        events="size", frauds=lambda x: x.eq(1).sum())
    amount_f = d.loc[d.fraud.eq(1), AMOUNT]
    rows = dict(
        events=len(d), customers=len(lengths), fraud_events=int(d.fraud.eq(1).sum()),
        fraud_rate=float(d.fraud.eq(1).mean()),
        onset_rate=float(safe_rate((normal & d.next_fraud.eq(1)).sum(), normal.sum())),
        continuation_rate=float(safe_rate((fraud & d.next_fraud.eq(1)).sum(), fraud.sum())),
        termination_rate=float(safe_rate((fraud & d.next_fraud.eq(0)).sum(), fraud.sum())),
        onset_denominator=int(normal.sum()), continuation_denominator=int(fraud.sum()),
        fraud_runs=len(runs), left_boundary_runs=int(runs.left_boundary.sum()),
        right_boundary_runs=int(runs.right_boundary.sum()),
        completed_known_start_runs=int(runs.completed_known_start.sum()),
        mean_run_length=float(runs.length.mean()), max_run_length=float(runs.length.max()) if len(runs) else np.nan,
        p90_run_length=float(runs.length.quantile(.9)), mean_run_span_seconds=float(runs.span_seconds.mean()),
        fraud_event_share_after_20=float(d.loc[d.fraud.eq(1), "run_age"].gt(20).mean()),
        customer_length_median=float(lengths.median()), customer_length_p90=float(lengths.quantile(.9)),
        customers_with_fraud=int(per_customer.frauds.gt(0).sum()),
        all_fraud_customers=int(per_customer.frauds.eq(per_customer.events).sum()),
        customer_weighted_fraud_rate=float((per_customer.frauds / per_customer.events).mean()),
        short_customers_le_100=int(lengths.le(100).sum()),
        unique_merchants=int(d[MERCHANT].nunique()),
        median_unique_merchants_per_customer=float(d.groupby("entity_id")[MERCHANT].nunique().median()),
        seen_merchant_rate=float(d.loc[d.event_index.gt(0), "seen_merchant"].mean()),
        fraud_amount_mean=float(amount_f.mean()), fraud_amount_p99=float(amount_f.quantile(.99)),
        amount_p999=float(d[AMOUNT].quantile(.999)), amount_max=float(d[AMOUNT].max()),
    )
    return rows


def hazard_table(d, name):
    f = d[d.fraud.eq(1) & ~d.left_boundary_run]
    rows = []
    for band, label in enumerate(AGE_BANDS):
        s = f[f.age_band.eq(band)]
        eligible = s[s.next_fraud.isin([0, 1])]
        ended = int(eligible.next_fraud.eq(0).sum())
        rows.append(dict(run=name, age_band=band, label=label, observed_events=len(s),
                         eligible_transitions=len(eligible), customers=eligible.entity_id.nunique(),
                         ends=ended, continues=int(eligible.next_fraud.eq(1).sum()),
                         censored_next=int(s.right_boundary_row.sum()),
                         termination_rate=float(safe_rate(ended, len(eligible)))))
    return rows


def age_amount_table(real, generated, name):
    def valid(d):
        ratio = d.amount_history_ratio_raw
        return d[d.fraud.eq(1) & d.event_index.ge(5) & np.isfinite(ratio) & ratio.ge(0)]
    r, s = valid(real), valid(generated)
    weights = np.zeros(len(s))
    coverage = 0.0
    rows = []
    for band, label in enumerate(AGE_BANDS):
        rr, ss = r[r.age_band.eq(band)], s[s.age_band.eq(band)]
        share = len(rr) / len(r) if len(r) else 0
        if len(ss):
            weights[s.age_band.eq(band).to_numpy()] = share / len(ss)
            coverage += share
        rows.append(dict(run=name, age_band=band, label=label, real_events=len(rr),
                         generated_events=len(ss), real_customers=rr.entity_id.nunique(),
                         generated_customers=ss.entity_id.nunique(),
                         generated_amount_mean=float(ss[AMOUNT].mean()),
                         real_ratio_median=float(rr.amount_history_ratio_raw.median()),
                         generated_ratio_median=float(ss.amount_history_ratio_raw.median()),
                         ratio_log_w1=log_w1(rr.amount_history_ratio_raw, ss.amount_history_ratio_raw)))
    result = dict(fraud_ratio_valid_events=len(s), real_fraud_ratio_valid_events=len(r),
                  fraud_ratio_log_w1=log_w1(r.amount_history_ratio_raw, s.amount_history_ratio_raw),
                  age_standardization_coverage=coverage,
                  age_standardized_ratio_log_w1=log_w1(r.amount_history_ratio_raw, s.amount_history_ratio_raw, weights)
                  if len(s) and np.isclose(coverage, 1) else np.nan)
    return rows, result


def cluster_counts(d, ids):
    known = d.next_fraud.isin([0, 1])
    counts = pd.DataFrame({
        "entity_id": d.entity_id, "events": 1, "frauds": d.fraud.eq(1).astype(int),
        "normal_next": (d.fraud.eq(0) & known).astype(int),
        "onsets": (d.fraud.eq(0) & d.next_fraud.eq(1)).astype(int),
        "fraud_next": (d.fraud.eq(1) & known).astype(int),
        "continues": (d.fraud.eq(1) & d.next_fraud.eq(1)).astype(int),
    }).groupby("entity_id", sort=False).sum().reindex(ids, fill_value=0)
    return counts


def bootstrap_rates(d, ids, weights):
    counts = cluster_counts(d, ids)
    totals = weights @ counts.to_numpy(dtype=float)
    c = {name: totals[:, i] for i, name in enumerate(counts.columns)}
    return {"fraud_rate": safe_rate(c["frauds"], c["events"]),
            "onset_rate": safe_rate(c["onsets"], c["normal_next"]),
            "continuation_rate": safe_rate(c["continues"], c["fraud_next"])}


def histogram_by_customer(values, entities, ids, support):
    row = pd.Index(ids).get_indexer(entities)
    col = np.searchsorted(support, np.asarray(values))
    assert (row >= 0).all() and (col < len(support)).all()
    result = np.zeros((len(ids), len(support)), dtype=float)
    np.add.at(result, (row, col), 1)
    return result


def bootstrap_w1_hist(real_counts, synthetic_counts, weights, support):
    r = weights @ real_counts
    s = weights @ synthetic_counts
    rt, st = r.sum(axis=1), s.sum(axis=1)
    r = np.cumsum(r, axis=1) / np.where(rt > 0, rt, np.nan)[:, None]
    s = np.cumsum(s, axis=1) / np.where(st > 0, st, np.nan)[:, None]
    return np.sum(np.abs(r[:, :-1] - s[:, :-1]) * np.diff(np.log1p(support)), axis=1)


def interval(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return dict(ci_low=float(np.quantile(finite, .025)) if len(finite) else np.nan,
                ci_high=float(np.quantile(finite, .975)) if len(finite) else np.nan,
                valid_repeats=len(finite))
