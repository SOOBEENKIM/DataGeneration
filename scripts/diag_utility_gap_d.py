"""ibm_normal_rows_v1 amendment 1 (docs/ibm_normal_rows_v1/PROTOCOL.md): which NORMAL rows drive the utility gap?
    ./run python scripts/diag_utility_gap_d.py
For 4 pairs (real train subset i, adopted-model generation i): a detector trained on the real subset scores synthetic normal rows
and held-out real train normal rows; a detector trained on the synthetic table scores held-out real normal rows and another
generation's synthetic normal rows. Rows above each detector's dev top-1% threshold are "fraud-like normals"; their share and
composition are compared. CPU only; train + dev + existing dev generations; never a test file; nothing is changed.
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

DOCS = ROOT / "docs/ibm_normal_rows_v1"
CP = set()


def flags(f):
    f = f.copy(); f["online"] = f.use_chip.astype(str).str.contains("Online").astype(int); f["novel"] = 1 - f.seen_merchant; f["rare_named"] = f.receiver_or_mark.astype(str).str.startswith("_RARE_").astype(int)
    f["cp_cat"] = f.category.astype(str).isin(CP).astype(int); f["online_at_cp"] = f.online * f.cp_cat; f["relpos"] = f.event_index / f.groupby("entity_id", sort=False).y.transform("size")
    f["gap_lt_10min"] = pd.to_numeric(f.gap, errors="coerce").astype(float).lt(600).fillna(False).astype(int); f["amount_jump_gt1"] = f.amount_vs_med20.gt(1).astype(int); f["same_cat5"] = f.same_category_5.fillna(0)
    return f


KEYS = ["online", "novel", "rare_named", "cp_cat", "online_at_cp", "relpos", "gap_lt_10min", "gap_vs_med20", "log_gap", "amount_jump_gt1", "amount_vs_med20", "log_amount", "same_cat5", "seen_merchants_5"]


def profile(f):
    out = dict(rows=int(len(f))); out.update({k: float(f[k].mean()) for k in KEYS} if len(f) else {})
    out["top_categories"] = {str(k): round(float(v), 3) for k, v in f.category.astype(str).value_counts(normalize=True).head(6).items()} if len(f) else {}
    return out


def _fit(name):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(4):
        cols = dg.cols_all(); clf = fu.fit(dg.F[name], cols, detector="stable"); sc = {k: clf.predict_proba(dg.F[k][cols].to_numpy())[:, 1] for k in SCORE[name]}
        thr = float(np.quantile(clf.predict_proba(dg.TEST[cols].to_numpy())[:, 1], 0.99))
    return name, thr, sc


SCORE = {}


def main():
    global CP
    from threadpoolctl import threadpool_limits
    threadpool_limits(8); t0 = time.time(); dev = pd.read_parquet(dg.PREP / "validation.parquet"); fu.configure(dev); dg.TEST = fu.features(dev); train = pd.read_parquet(dg.PREP / "train.parquet"); ids = np.sort(train.entity_id.unique())
    tn = train[pd.to_numeric(train.event_is_fraud, errors="coerce").fillna(0).astype(int).eq(0)]; on = tn.use_chip.astype(str).str.contains("Online").groupby(tn.category.astype(str)).mean(); CP = set(on[on.lt(0.01)].index)
    for i in range(4):
        rng = np.random.default_rng(dg.SEED + i); pick = rng.choice(ids, size=dg.N_DEV, replace=False); rest = np.setdiff1d(ids, pick); held = np.random.default_rng(dg.SEED + 100 + i).choice(rest, size=dg.N_DEV, replace=False)
        dg.F[f"real{i}"] = flags(fu.features(train[train.entity_id.isin(pick)])); dg.F[f"held{i}"] = flags(fu.features(train[train.entity_id.isin(held)])); fs, gs = dg.FITS[i]
        dg.F[f"syn{i}"] = flags(fu.features(pd.read_parquet(dg.SOURCES["argn_final"](fs, gs))))
        SCORE[f"real{i}"] = [f"syn{i}", f"held{i}"]; SCORE[f"syn{i}"] = [f"held{i}", f"syn{(i + 1) % 4}"]
    print("features ready", round(time.time() - t0), "s; card-present categories", len(CP), "of", len(on), flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        fitted = list(ex.map(_fit, list(SCORE)))
    res = dict(card_present_categories=len(CP), categories=int(len(on)), pairs=[], test_read=False); agg = {}
    for name, thr, sc in fitted:
        for target, s in sc.items():
            f = dg.F[target]; n = f.y.eq(0).to_numpy(); hit = n & (s >= thr); kind = ("real_detector" if name.startswith("real") else "synth_detector") + "__" + ("synthetic_normals" if target.startswith("syn") else "real_normals")
            r = dict(detector=name, target=target, kind=kind, threshold=thr, normal_rows=int(n.sum()), above=int(hit.sum()), share_above=float(hit.sum() / n.sum()), customers_with_any=int(f.entity_id[hit].nunique()), profile=profile(f[hit]))
            res["pairs"].append(r); agg.setdefault(kind, []).append(r)
    base = {k: [profile(dg.F[f"{k}{i}"][dg.F[f"{k}{i}"].y.eq(0)]) for i in range(4)] for k in ("held", "syn")}; fr = {k: [profile(dg.F[f"{k}{i}"][dg.F[f"{k}{i}"].y.eq(1)]) for i in range(4)] for k in ("held", "syn")}
    mean = lambda rows, k: float(np.mean([r[k] for r in rows if k in r]))
    tab = pd.DataFrame({"real normals (all)": {k: mean(base["held"], k) for k in KEYS}, "synthetic normals (all)": {k: mean(base["syn"], k) for k in KEYS}, "real frauds": {k: mean(fr["held"], k) for k in KEYS}, "synthetic frauds": {k: mean(fr["syn"], k) for k in KEYS}}
                       | {kind: {k: mean([r["profile"] for r in rows], k) for k in KEYS} for kind, rows in agg.items()})
    share = {kind: dict(share_above=float(np.mean([r["share_above"] for r in rows])), rows_above=float(np.mean([r["above"] for r in rows])), customers=float(np.mean([r["customers_with_any"] for r in rows]))) for kind, rows in agg.items()}
    res["summary"] = tab.to_dict(); res["share_above_threshold"] = share; res["seconds"] = round(time.time() - t0, 1); (DOCS / "FRAUDLIKE_NORMALS.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    pd.set_option("display.width", 320); print(json.dumps(share, indent=1)); print(tab.round(3).to_string())
    for kind, rows in agg.items():
        print(kind, "top categories:", [r["profile"]["top_categories"] for r in rows][:2])
    print("done", res["seconds"], "s")


if __name__ == "__main__":
    main()
