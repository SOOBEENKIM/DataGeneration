"""ibm_normal_impl_v1 diagnostic (dev only, no test, no training): why the within-episode fraud amount variability metric (B_amount_cv, a
Wasserstein distance over per-episode std/mean) moves from 2.4x to 5.2x of the floor when the stay module is on. Reads the generated
dev sets of arms B and R, real train and real dev; writes docs/ibm_normal_impl_v1/FRAUD_AMOUNT_CV[_A1].json.
    ./run python scripts/diag_impl_fraud_amount.py [--set a1]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from benchmarks.fraud_structure import prepare, _episode_table, _w1, AMOUNT  # noqa: E402
from run_ibm_normal_impl import FITS, SEEDS, SEEDS_A1, path, DOCS, NATIVE  # noqa: E402

CLIP = 3.0  # per-episode CV above this is an episode whose mean is near zero (refund rows mixed in), not a spread of ordinary amounts


def episodes(frame):
    d = prepare(frame); f = d[d.fraud.eq(1)]; ep = _episode_table(d); long = ep[ep.n.ge(3)]
    neg = f.groupby("ep_id")[AMOUNT].apply(lambda x: bool((x <= 0).any())); long = long.assign(has_nonpositive=long.ep_id.map(neg).fillna(False).to_numpy())
    return long, f


def describe(long, f):
    cv = long.amount_cv.dropna().to_numpy(); a = f[AMOUNT].to_numpy(dtype=float)
    return dict(episodes_ge3=int(len(long)), cv_defined=int(len(cv)), cv_median=float(np.median(cv)) if len(cv) else None, cv_p90=float(np.quantile(cv, .9)) if len(cv) else None, cv_max=float(cv.max()) if len(cv) else None,
                cv_gt_clip=int((cv > CLIP).sum()), episodes_with_nonpositive_amount=float(long.has_nonpositive.mean()) if len(long) else None, fraud_rows=int(len(a)), fraud_amount_nonpositive_share=float((a <= 0).mean()) if len(a) else None,
                fraud_amount_median=float(np.median(a)) if len(a) else None, fraud_amount_p90=float(np.quantile(a, .9)) if len(a) else None), cv


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--set", choices=["v1", "a1"], default="v1"); a = ap.parse_args()
    seeds = {fs: SEEDS[fs] + (SEEDS_A1[fs] if a.set == "a1" else []) for fs in FITS}; prep = NATIVE / "prepared"; res = {}
    dev_desc, dev_cv = describe(*episodes(pd.read_parquet(prep / "validation.parquet"))); tr_desc, tr_cv = describe(*episodes(pd.read_parquet(prep / "train.parquet")))
    res["real_dev"] = dev_desc; res["real_train"] = tr_desc; res["clip"] = CLIP; rows = []
    for arm in ("B", "R"):
        for fs in FITS:
            for gs in seeds[fs]:
                desc, cv = describe(*episodes(pd.read_parquet(path(arm, fs, gs))))
                rows.append(dict(arm=arm, fit=fs, seed=gs, **desc, w1_vs_dev=_w1(cv, dev_cv), w1_vs_dev_clipped=_w1(np.minimum(cv, CLIP), np.minimum(dev_cv, CLIP)), w1_median_only=abs(float(np.median(cv)) - float(np.median(dev_cv))) if len(cv) else None))
    t = pd.DataFrame(rows); res["per_generation"] = rows
    num = ["cv_defined", "cv_median", "cv_p90", "cv_max", "cv_gt_clip", "episodes_with_nonpositive_amount", "fraud_amount_nonpositive_share", "fraud_amount_median", "fraud_amount_p90", "w1_vs_dev", "w1_vs_dev_clipped"]
    res["by_arm_mean"] = {k: {c: float(v[c].mean()) for c in num} for k, v in t.groupby("arm")}; res["by_arm_median"] = {k: {c: float(v[c].median()) for c in num} for k, v in t.groupby("arm")}
    w = t.pivot(index=["fit", "seed"], columns="arm", values=["w1_vs_dev", "w1_vs_dev_clipped", "cv_median", "cv_max", "cv_gt_clip"])
    d = {c: (w[c]["R"] - w[c]["B"]) for c in ("w1_vs_dev", "w1_vs_dev_clipped", "cv_median")}
    res["paired_R_minus_B"] = {c: dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))), positive=int((v > 0).sum()), n=int(len(v))) for c, v in d.items()}
    res["w1_share_from_top3_generations"] = {k: float(v.w1_vs_dev.nlargest(3).sum() / v.w1_vs_dev.sum()) for k, v in t.groupby("arm")}; res["test_read"] = False
    out = DOCS / ("FRAUD_AMOUNT_CV_A1.json" if a.set == "a1" else "FRAUD_AMOUNT_CV.json"); out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    pd.set_option("display.width", 250); print("real dev", dev_desc); print("real train", tr_desc); print(w.round(2).to_string()); print(json.dumps({k: res[k] for k in ("by_arm_mean", "by_arm_median", "paired_R_minus_B", "w1_share_from_top3_generations")}, indent=1))
