"""ibm_tabformer_v1 step 0: convert the raw IBM/TabFormer CSV to parquet and profile fraud episodes.

No model, no split decision yet. Reads every row once (the sealed test split does not exist yet;
it is drawn from this profile's customer list in the registered `prepare` step and never read after).
    ./run python scripts/ibm_tabformer_profile.py --raw /abs/path/card_transaction.v1.csv
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/ibm_tabformer_v1"
PARQUET = ROOT / "artifacts/ibm_tabformer_v1/raw_events.parquet"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def convert(raw):
    t0 = time.time()
    d = pd.read_csv(raw, dtype={"Zip": "float64", "Errors?": "string", "Merchant State": "string",
                                "Merchant City": "string"}, low_memory=False)
    d.columns = ["user", "card", "year", "month", "day", "time", "amount", "use_chip", "merchant", "city",
                 "state", "zip", "mcc", "errors", "is_fraud"]
    hh = d.time.str.slice(0, 2).astype(int); mm = d.time.str.slice(3, 5).astype(int)
    ts = pd.to_datetime(dict(year=d.year, month=d.month, day=d.day, hour=hh, minute=mm))
    d["timestamp"] = (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    d["amount"] = d.amount.str.replace("$", "", regex=False).astype(float)
    d["is_fraud"] = d.is_fraud.eq("Yes").astype(int)
    d = d.drop(columns=["time"])
    d = d.sort_values(["user", "timestamp"], kind="stable").reset_index(drop=True)
    d["event_index"] = d.groupby("user", sort=False).cumcount()
    d["gap"] = d.groupby("user", sort=False).timestamp.diff()
    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(PARQUET, index=False)
    return d, time.time() - t0


def profile(d):
    g = d.groupby("user", sort=False)
    per_user = pd.DataFrame(dict(events=g.size(), frauds=g.is_fraud.sum(), cards=g.card.nunique(),
                                 first_year=g.year.min(), last_year=g.year.max(), span_days=(g.timestamp.max() - g.timestamp.min()) / 86400))
    per_user["fraud_rate"] = per_user.frauds / per_user.events
    # episode structure on the full user sequence (same definitions as stage 1)
    f = d.rename(columns={"user": "entity_id", "is_fraud": "event_is_fraud"})[["entity_id", "event_index", "event_is_fraud", "gap", "timestamp"]]
    f["event_is_fraud"] = f.event_is_fraud.astype(int)
    prev = f.groupby("entity_id", sort=False).event_is_fraud.shift()
    cont = f.event_is_fraud[prev.eq(1)].mean(); onset = f.event_is_fraud[prev.eq(0)].mean()
    boundary = f.entity_id.ne(f.entity_id.shift()) | f.event_is_fraud.ne(f.event_is_fraud.shift())
    run = boundary.cumsum(); f["run"] = run
    runs = f[f.event_is_fraud.eq(1)].groupby("run").agg(entity_id=("entity_id", "first"), length=("entity_id", "size"),
                                                         start=("timestamp", "min"), end=("timestamp", "max"), start_index=("event_index", "min"))
    runs["span_hours"] = (runs.end - runs.start) / 3600
    # gaps inside fraud runs vs between fraud run and next normal vs normal
    gap_in_run = f.gap[prev.eq(1) & f.event_is_fraud.eq(1)]; gap_exit = f.gap[prev.eq(1) & f.event_is_fraud.eq(0)]; gap_norm = f.gap[prev.eq(0) & f.event_is_fraud.eq(0)]
    # within-user fraud concentration in calendar time: distinct fraud days per fraud user, and fraud events per fraud day
    fd = d[d.is_fraud.eq(1)].assign(day_id=lambda x: x.timestamp // 86400)
    per_day = fd.groupby(["user", "day_id"]).size()
    fraud_users = per_user[per_user.frauds.gt(0)]
    rl = runs.length
    summary = dict(
        events=int(len(d)), users=int(per_user.shape[0]), cards=int(d.groupby(["user", "card"]).ngroups),
        years=[int(d.year.min()), int(d.year.max())], fraud_events=int(d.is_fraud.sum()), fraud_rate=float(d.is_fraud.mean()),
        events_per_user=dict(mean=float(per_user.events.mean()), median=float(per_user.events.median()), min=int(per_user.events.min()), max=int(per_user.events.max())),
        users_with_fraud=int(len(fraud_users)), users_with_fraud_share=float(len(fraud_users) / len(per_user)),
        frauds_per_fraud_user=dict(mean=float(fraud_users.frauds.mean()), median=float(fraud_users.frauds.median()), max=int(fraud_users.frauds.max())),
        continuation_rate=float(cont), onset_rate=float(onset),
        runs=int(len(runs)), runs_per_fraud_user=float(len(runs) / len(fraud_users)),
        run_length=dict(mean=float(rl.mean()), median=float(rl.median()), p90=float(rl.quantile(.9)), max=int(rl.max()),
                        share_len1=float(rl.eq(1).mean()), share_len_ge5=float(rl.ge(5).mean()), share_len_gt20=float(rl.gt(20).mean())),
        run_span_hours=dict(median=float(runs.span_hours.median()), p90=float(runs.span_hours.quantile(.9)), max=float(runs.span_hours.max())),
        gap_seconds_median=dict(inside_fraud_run=float(gap_in_run.median()), fraud_to_normal=float(gap_exit.median()), normal=float(gap_norm.median())),
        fraud_days_per_fraud_user=dict(mean=float(per_day.groupby(level=0).size().mean()), median=float(per_day.groupby(level=0).size().median())),
        frauds_per_fraud_day=dict(mean=float(per_day.mean()), median=float(per_day.median()), max=int(per_day.max())),
        first_event_fraud_rate=float(f.event_is_fraud[f.event_index.eq(0)].mean()),
    )
    # fraud by calendar year and per-year fraud rate (to choose a time slice without reading per-customer futures)
    by_year = d.groupby("year").agg(events=("is_fraud", "size"), frauds=("is_fraud", "sum"), users=("user", "nunique")).reset_index()
    by_year["fraud_rate"] = by_year.frauds / by_year.events
    # hazard by run age (continuation after k fraud events)
    age = f.groupby("run", sort=False).cumcount() + 1
    nxt = f.groupby("entity_id", sort=False).event_is_fraud.shift(-1)
    h = pd.DataFrame(dict(age=age[f.event_is_fraud.eq(1)], nxt=nxt[f.event_is_fraud.eq(1)])).dropna()
    bands = pd.cut(h.age, [0, 1, 2, 3, 5, 10, 20, 50, 10**9], labels=["1", "2", "3", "4-5", "6-10", "11-20", "21-50", ">50"])
    hazard = h.groupby(bands, observed=True).agg(n=("nxt", "size"), continuation=("nxt", "mean")).reset_index()
    # use_chip / mcc / errors inside vs outside fraud
    cond = {}
    for c in ["use_chip", "errors"]:
        t = d.groupby([c, "is_fraud"], dropna=False).size().unstack(fill_value=0)
        cond[c] = (t.div(t.sum(0), axis=1)).round(4).reset_index().to_dict("records")
    mcc_tv = 0.5 * (d[d.is_fraud.eq(1)].mcc.value_counts(normalize=True).reindex(d.mcc.unique(), fill_value=0)
                    - d[d.is_fraud.eq(0)].mcc.value_counts(normalize=True).reindex(d.mcc.unique(), fill_value=0)).abs().sum()
    summary["mcc_tv_fraud_vs_normal"] = float(mcc_tv); summary["mcc_levels"] = int(d.mcc.nunique()); summary["merchant_levels"] = int(d.merchant.nunique())
    summary["amount_median"] = dict(fraud=float(d.amount[d.is_fraud.eq(1)].median()), normal=float(d.amount[d.is_fraud.eq(0)].median()), negative_share=float(d.amount.lt(0).mean()))
    return summary, per_user, runs, by_year, hazard, cond


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    if a.reuse and PARQUET.exists():
        d = pd.read_parquet(PARQUET); conv = 0.0
    else:
        d, conv = convert(a.raw)
    print("loaded", len(d), "rows", round(time.time() - t0), "s", flush=True)
    summary, per_user, runs, by_year, hazard, cond = profile(d)
    per_user.describe().to_csv(OUT / "PROFILE_per_user_describe.csv")
    per_user.frauds.value_counts().sort_index().rename("users").to_csv(OUT / "PROFILE_frauds_per_user_hist.csv")
    runs.length.value_counts().sort_index().rename("runs").to_csv(OUT / "PROFILE_run_length_hist.csv")
    by_year.to_csv(OUT / "PROFILE_by_year.csv", index=False); hazard.to_csv(OUT / "PROFILE_hazard_by_age.csv", index=False)
    json.dump(dict(summary=summary, conditional=cond, raw_csv=a.raw, raw_sha256=sha(a.raw) if not a.reuse else None,
                   parquet=str(PARQUET), parquet_sha256=sha(PARQUET), convert_seconds=round(conv, 1), seconds=round(time.time() - t0, 1),
                   created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test_split_exists=False),
              open(OUT / "PROFILE.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
