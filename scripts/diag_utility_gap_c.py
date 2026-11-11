"""ibm_gap_diag_v1 D2 follow-up 2 (added 2026-11-11; recorded as an addition in RESULTS): which REAL dev rows does the
synthetic-trained detector get wrong that the real-trained one gets right?   ./run python scripts/diag_utility_gap_c.py
For each of 4 pairs (ARGN final generation i, real train subset i): score real dev, take the top-1% alerts, and profile the
false positives (normal dev rows alerted) of each detector, plus recall by position in the fraud run. CPU only; never the sealed test.
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
import benchmarks.fraud_utility as fu  # noqa: E402
import diag_utility_gap as dg  # noqa: E402


def _score(name):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(4):
        cols = dg.cols_all(); clf = fu.fit(dg.F[name], cols, detector="stable"); return name, clf.predict_proba(dg.TEST[cols].to_numpy())[:, 1]


def profile(t, mask):
    d = t[mask]
    return dict(rows=int(len(d)), online=float(d.online.mean()), novel_merchant=float(1 - d.seen_merchant.mean()), same_merchant_as_previous=float(d.same_prev_m.mean()), same_merchant_within_5=float(d.same5_m.mean()),
                gap_under_10min=float(d.gap_s.lt(600).mean()), gap_under_1h=float(d.gap_s.lt(3600).mean()), log_gap_median=float(d.log_gap.median()), gap_vs_med20_mean=float(d.gap_vs_med20.mean()),
                amount_vs_med20_mean=float(d.amount_vs_med20.mean()), amount_vs_med20_above_1=float(d.amount_vs_med20.gt(1).mean()), log_amount_mean=float(d.log_amount.mean()),
                abs_dlog_amount_vs_prev_visit_median=float(d.dprev.abs().median()), same_category_5=float(d.same_category_5.fillna(0).mean()), customers=int(d.entity_id.nunique()),
                within_20_rows_of_a_fraud=float(d.near_fraud.mean()))


def main():
    t0 = time.time(); dev = pd.read_parquet(dg.PREP / "validation.parquet"); fu.configure(dev); t = fu.features(dev); dg.TEST = t; train = pd.read_parquet(dg.PREP / "train.parquet"); ids = np.sort(train.entity_id.unique())
    g = t.groupby("entity_id", sort=False); m = t.receiver_or_mark.astype(str); t["online"] = t.use_chip.astype(str).str.contains("Online").astype(int); t["gap_s"] = pd.to_numeric(t.gap, errors="coerce")
    t["same_prev_m"] = m.eq(m.groupby(t.entity_id).shift()).astype(int); t["same5_m"] = pd.concat([m.eq(m.groupby(t.entity_id).shift(k)) for k in range(1, 6)], axis=1).any(axis=1).astype(int)
    t["dprev"] = t.log_amount - t.groupby(["entity_id", m]).log_amount.shift()
    near = t.y.groupby(t.entity_id).transform(lambda s: s.rolling(41, center=True, min_periods=1).max()); t["near_fraud"] = near.astype(int)
    prev = g.y.shift().fillna(0); run = (t.y.ne(prev) | t.entity_id.ne(t.entity_id.shift())).cumsum(); t["run_age"] = t.groupby(run).cumcount() + 1
    for i in range(4):
        pick = np.random.default_rng(dg.SEED + i).choice(ids, size=dg.N_DEV, replace=False); dg.F[f"real{i}"] = fu.features(train[train.entity_id.isin(pick)])
        fs, gs = dg.FITS[i]; dg.F[f"argn_final{i}"] = fu.features(pd.read_parquet(dg.SOURCES["argn_final"](fs, gs)))
    with ProcessPoolExecutor(max_workers=8) as ex:
        scores = dict(ex.map(_score, list(dg.F)))
    out = dict(dev=dict(all_normal=profile(t, t.y.eq(0)), all_fraud=profile(t, t.y.eq(1))), pairs=[]); k = int(round(0.01 * len(t)))
    for i in range(4):
        sr, ss = scores[f"real{i}"], scores[f"argn_final{i}"]; ar = sr >= np.sort(sr)[-k]; as_ = ss >= np.sort(ss)[-k]; n = t.y.eq(0).to_numpy(); f = ~n
        rec = lambda a: {f"age{lab}": float(a[f & sel.to_numpy()].mean()) for lab, sel in (("1", t.run_age.eq(1)), ("2", t.run_age.eq(2)), ("3plus", t.run_age.ge(3)))}
        out["pairs"].append(dict(draw=i, alerts=k, real_trained=dict(false_positives=profile(t, ar & n), fraud_recall=float(ar[f].mean()), recall_by_run_age=rec(ar)),
                                 synth_trained=dict(false_positives=profile(t, as_ & n), fraud_recall=float(as_[f].mean()), recall_by_run_age=rec(as_)),
                                 fp_only_synth=profile(t, as_ & ~ar & n), fp_only_real=profile(t, ar & ~as_ & n), fraud_missed_only_by_synth=profile(t, ar & ~as_ & f), fraud_found_only_by_synth=profile(t, as_ & ~ar & f)))
    keys = list(out["dev"]["all_normal"]); avg = lambda path: {q: float(np.mean([path(p)[q] for p in out["pairs"]])) for q in keys}
    summ = pd.DataFrame(dict(dev_normal=out["dev"]["all_normal"], dev_fraud=out["dev"]["all_fraud"], FP_real_trained=avg(lambda p: p["real_trained"]["false_positives"]), FP_synth_trained=avg(lambda p: p["synth_trained"]["false_positives"]),
                             FP_only_synth=avg(lambda p: p["fp_only_synth"]), FP_only_real=avg(lambda p: p["fp_only_real"]), fraud_missed_only_by_synth=avg(lambda p: p["fraud_missed_only_by_synth"])))
    print(summ.round(3).to_string())
    rc = {w: {a: float(np.mean([p[w]["recall_by_run_age"][a] for p in out["pairs"]])) for a in ("age1", "age2", "age3plus")} | dict(all=float(np.mean([p[w]["fraud_recall"] for p in out["pairs"]]))) for w in ("real_trained", "synth_trained")}
    print(json.dumps(rc, indent=1)); out["summary"] = summ.to_dict(); out["recall"] = rc; out["seconds"] = round(time.time() - t0, 1); out["test_read"] = False
    (dg.OUT / "UTILITY_GAP_C.json").write_text(json.dumps(out, indent=1, default=float) + "\n"); print("done", out["seconds"], "s")


if __name__ == "__main__":
    main()
