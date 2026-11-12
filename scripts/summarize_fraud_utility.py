"""Summarise a fraud_utility run: per-model means and Spearman correlation with stage-1 metrics.
    python scripts/summarize_fraud_utility.py docs/fraud_utility_v1/detector_stable docs/rare_state_persistence_v1
    python scripts/summarize_fraud_utility.py docs/ibm_tabformer_v1/utility_detector_stable docs/ibm_tabformer_v1/stage1
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

UTIL = ["history_auprc", "history_early_detection_top1", "row_auprc", "history_auroc", "history_episode_recall_top1", "row_early_detection_top1"]


def main(util_dir, stage1_dir):
    u = pd.read_csv(Path(util_dir) / "UTILITY_ALL.csv")
    g = u[u.kind.eq("generation")].copy()
    by = u.groupby(["kind", "family", "model_name"], dropna=False)[UTIL + ["train_fraud"]].agg(["mean", "std", "count"])
    by.columns = [f"{a}_{b}" for a, b in by.columns]
    by.sort_values("history_auprc_mean", ascending=False).round(4).to_csv(Path(util_dir) / "UTILITY_BY_MODEL.csv")
    real = u[u.kind.eq("real_subset")][UTIL].agg(["mean", "std", "min", "median", "max"]).round(4)
    real.to_csv(Path(util_dir) / "REAL_CEILING.csv")
    s = pd.read_csv(Path(stage1_dir) / "ALL_GENERATIONS.csv")
    s = s[s.family.ne("reference")].copy()
    s["fit_seed"] = s.fit_seed.astype("float").astype(str); s["generation_seed"] = s.generation_seed.astype("float").astype(str)
    g["fit_seed"] = g.fit_seed.astype("float").astype(str); g["generation_seed"] = g.generation_seed.astype("float").astype(str)
    m = g.merge(s, left_on=["model_name", "fit_seed", "generation_seed"], right_on=["model", "fit_seed", "generation_seed"], how="inner", suffixes=("", "_s1"))
    metrics = [c for c in s.columns if c.startswith(("S_", "R_")) and (c.endswith(("absdiff", "reldiff", "tv", "logw1")))]
    rows = []
    for c in metrics:
        row = dict(metric=c, n=int(m[c].notna().sum()))
        for t in UTIL[:3]:
            ok = m[[c, t]].dropna()
            row[f"rho_{t}"] = spearmanr(ok[c], ok[t]).statistic if len(ok) > 3 and ok[c].nunique() > 1 else np.nan
        rows.append(row)
    pd.DataFrame(rows).round(3).to_csv(Path(util_dir) / "SPEARMAN_UTILITY_VS_METRICS.csv", index=False)
    print(real.to_string()); print(by.sort_values("history_auprc_mean", ascending=False)[["history_auprc_mean", "history_early_detection_top1_mean", "row_auprc_mean", "history_auprc_count"]].round(3).to_string())
    print(pd.DataFrame(rows).round(2).to_string(index=False)); print("merged generations:", len(m), "of", len(g))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
