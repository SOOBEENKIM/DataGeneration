"""ibm_gap_diag_v1 D1 follow-up (added 2026-11-11 after the first D1 output; recorded as an addition in RESULTS):
the free run over-generates because time spent in the "ever fraud" state is 2.4x the real one. This measures the real
structure the table misses: re-onset hazard by distance since the last fraud, the span of a customer's fraud activity,
and when (in the calendar window) first frauds happen.   ./run python scripts/diag_label_rate_b.py
CPU only, train and dev, never the sealed test.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd

PREP = ROOT / "artifacts/ibm_tabformer_v1/prepared"; OUT = ROOT / "docs/ibm_gap_diag_v1"
BINS = [0, 5, 20, 50, 100, 200, 500, 10**9]; NAMES = ["1-5", "6-20", "21-50", "51-100", "101-200", "201-500", "501+"]


def profile(d):
    d = d.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True); g = d.entity_id
    y = pd.to_numeric(d.event_is_fraud, errors="coerce").fillna(0).astype(int).eq(1).astype(int)
    i = d.groupby(g, sort=False).cumcount(); n = g.map(g.value_counts()); day = pd.to_numeric(d.gap, errors="coerce").fillna(0).groupby(g).cumsum() / 86400.0
    last_fraud_i = i.where(y.eq(1)).groupby(g).ffill().groupby(g).shift()        # index of the last fraud strictly before this event
    last_fraud_day = day.where(y.eq(1)).groupby(g).ffill().groupby(g).shift()
    prev = y.groupby(g).shift(); sel = prev.eq(0) & last_fraud_i.notna()           # previous event normal, some fraud before
    dist = (i - last_fraud_i)[sel]; days = (day - last_fraud_day)[sel]; yy = y[sel]
    by_events = pd.DataFrame(dict(b=pd.cut(dist, BINS, labels=NAMES), y=yy)).groupby("b", observed=False).y.agg(["size", "sum"])
    by_days = pd.DataFrame(dict(b=pd.cut(days, [-1, 1, 3, 7, 14, 30, 60, 10**6], labels=["<=1d", "1-3d", "3-7d", "7-14d", "14-30d", "30-60d", ">60d"]), y=yy)).groupby("b", observed=False).y.agg(["size", "sum"])
    f = pd.DataFrame(dict(g=g, i=i, n=n, day=day, y=y)); ff = f[f.y.eq(1)]; per = ff.groupby("g").agg(first_i=("i", "min"), last_i=("i", "max"), first_day=("day", "min"), last_day=("day", "max"), k=("y", "size"), n=("n", "first"))
    total_days = f.groupby("g").day.max().reindex(per.index)
    q = lambda s: {p: float(s.quantile(p)) for p in (.1, .25, .5, .75, .9)}
    ever0 = sel.eq(False) & prev.eq(0) & last_fraud_i.isna()                      # previous normal, never fraud before
    rel = (i / n)[ever0]; first_haz = pd.DataFrame(dict(b=pd.cut(rel, np.linspace(0, 1, 11), include_lowest=True).astype(str), y=y[ever0])).groupby("b").y.agg(["size", "sum"])
    month = np.floor(per.first_day / 30.5).clip(upper=11).astype(int).value_counts().sort_index()
    return dict(fraud_customers=int(len(per)), frauds=int(y.sum()),
                reonset_hazard_by_events_since_last_fraud={k: dict(events=int(r["size"]), reonsets=int(r["sum"]), hazard=float(r["sum"] / max(1, r["size"]))) for k, r in by_events.iterrows()},
                reonset_hazard_by_days_since_last_fraud={k: dict(events=int(r["size"]), reonsets=int(r["sum"]), hazard=float(r["sum"] / max(1, r["size"]))) for k, r in by_days.iterrows()},
                first_onset_hazard_by_relative_position={k: dict(events=int(r["size"]), onsets=int(r["sum"]), hazard=float(r["sum"] / max(1, r["size"]))) for k, r in first_haz.iterrows()},
                first_fraud_month_counts={int(k): int(v) for k, v in month.items()},
                fraud_span_events_quantiles=q(per.last_i - per.first_i + 1), fraud_span_days_quantiles=q(per.last_day - per.first_day),
                events_after_last_fraud_quantiles=q(per.n - 1 - per.last_i), days_after_last_fraud_quantiles=q(total_days - per.last_day),
                share_of_fraud_customers_with_100plus_events_after_last_fraud=float(((per.n - 1 - per.last_i) >= 100).mean()),
                frauds_per_fraud_customer_quantiles=q(per.k), first_fraud_day_quantiles=q(per.first_day))


res = dict(train=profile(pd.read_parquet(PREP / "train.parquet")), dev=profile(pd.read_parquet(PREP / "validation.parquet")), test_read=False)
OUT.mkdir(parents=True, exist_ok=True); (OUT / "LABEL_STRUCTURE.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
print(json.dumps(res, indent=1, default=float))
