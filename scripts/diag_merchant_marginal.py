"""Diagnosis before registering ibm_merchant_marginal_v1: where does the category composition of the selected merchants
go wrong — first visits (novel branch), repeats (kernel), or the rare bucket? CPU; train + development + existing
generations; the sealed test is never read.
    ./run python scripts/diag_merchant_marginal.py
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SF = ROOT / "artifacts"; IBM = SF / "ibm_tabformer_v1/prepared"
OUT = ROOT / "docs/ibm_merchant_marginal_v1"
M, RARE = "receiver_or_mark", "_RARE_"
TOP = ["5499", "5541", "4829", "4784", "5812", "5411", "5912", "4900", "5311", "7538"]


def prep(d, modal):
    d = d.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    d["m"] = d[M].astype(str); d["y"] = pd.to_numeric(d.event_is_fraud, errors="coerce").fillna(0).astype(int)
    d["novel"] = ~d.duplicated(["entity_id", "m"]); d["imp"] = d.m.map(modal)
    return d


def tv(a, b):
    return float(0.5 * a.sub(b, fill_value=0).abs().sum())


def stats(d, ref=None):
    n = d[d.y.eq(0)]; k = n[n.imp.notna()]
    o = dict(rows=int(len(n)), novel_share=float(n.novel.mean()), rare_share=float(n.m.eq(RARE).mean()), known=float(n.imp.notna().mean()))
    dist = dict(all=k.imp.value_counts(normalize=True), novel=k[k.novel].imp.value_counts(normalize=True), repeat=k[~k.novel].imp.value_counts(normalize=True))
    # per-customer structure
    g = n.groupby("entity_id"); per = n.groupby(["entity_id", "m"]).size().rename("c").reset_index()
    tot = per.groupby("entity_id").c.sum(); top1 = per.groupby("entity_id").c.max() / tot
    top3 = per.sort_values("c", ascending=False).groupby("entity_id").head(3).groupby("entity_id").c.sum() / tot
    o.update(merchants_per_customer=float(per.groupby("entity_id").size().median()), top1_share=float(top1.median()), top3_share=float(top3.median()))
    # visits per (customer, merchant) by the merchant's category, and distinct merchants per customer by category
    per["imp"] = per.m.map(d.drop_duplicates("m").set_index("m").imp)
    vis = per.groupby("imp").c.mean(); dm = per.groupby("imp").size() / n.entity_id.nunique()
    o["visits_per_customer_merchant"] = {c: float(vis.get(c, np.nan)) for c in TOP}
    o["distinct_merchants_per_customer"] = {c: float(dm.get(c, 0.0)) for c in TOP}
    o["share_all"] = {c: float(dist["all"].get(c, 0.0)) for c in TOP}; o["share_novel"] = {c: float(dist["novel"].get(c, 0.0)) for c in TOP}
    o["share_repeat"] = {c: float(dist["repeat"].get(c, 0.0)) for c in TOP}
    # category of each customer's top-1 merchant
    t1 = per.sort_values("c", ascending=False).drop_duplicates("entity_id"); o["top1_category"] = {c: float(t1.imp.eq(c).mean()) for c in TOP}
    if ref is not None:
        o["tv_all"] = tv(dist["all"], ref["all"]); o["tv_novel"] = tv(dist["novel"], ref["novel"]); o["tv_repeat"] = tv(dist["repeat"], ref["repeat"])
    o["_dist"] = dist
    return o


def mean_dicts(rs):
    out = {}
    for k in rs[0]:
        if k == "_dist":
            continue
        out[k] = {c: float(np.nanmean([r[k][c] for r in rs])) for c in rs[0][k]} if isinstance(rs[0][k], dict) else float(np.mean([r[k] for r in rs]))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet(IBM / "train.parquet"); tr["m"] = tr[M].astype(str); modal = tr.groupby("m").category.agg(lambda s: str(s.astype(str).value_counts().index[0]))
    dev = prep(pd.read_parquet(IBM / "validation.parquet"), modal); ref = stats(dev)["_dist"]
    res = dict(real_dev=mean_dicts([stats(dev, ref)]))
    rng = np.random.default_rng(20261107); trp = prep(tr, modal); ids = np.sort(trp.entity_id.unique())
    res["real_train_240"] = mean_dicts([stats(trp[trp.entity_id.isin(rng.choice(ids, 240, replace=False))], ref) for _ in range(8)])
    arms = {"ARGN_native": "ibm_tabformer_v1/runs/native_{fs}", "ARGN_model": "ibm_routed_heads_v1/runs/routed_subst_table_{fs}", "ARGN_current": "ibm_merchant_attr_v1/runs/attr_all_{fs}", "ARGN_novelcat": "ibm_merchant_marginal_v1/runs/novelcat_{fs}",
            "ARGN_novelcat_pos": "ibm_merchant_marginal_v1/runs/novelcat_pos_{fs}", "ARGN_novelcat_pos_rarefix": "ibm_merchant_marginal_v1/runs/novelcat_pos_rarefix_{fs}",
            "RTF_table": "ibm_realtabformer_v1/runs/rtf_table_{fs}"}
    for a, t in arms.items():
        if not all((SF / (t.format(fs=fs) + f"/generated_validation_{gs}.parquet")).exists() for fs in (20261108, 20261109) for gs in (20261011, 20261012)):
            continue
        res[a] = mean_dicts([stats(prep(pd.read_parquet(SF / (t.format(fs=fs) + f"/generated_validation_{gs}.parquet")), modal), ref) for fs in (20261108, 20261109) for gs in (20261011, 20261012)])
    json.dump(res, open(OUT / ("DIAG_MERCHANT_MARGINAL_after.json" if "ARGN_novelcat" in res else "DIAG_MERCHANT_MARGINAL.json"), "w"), indent=1)
    real = pd.Series(res["real_dev"]["visits_per_customer_merchant"])
    print("visits rank correlation with real dev (top-10 categories):", {k: round(float(pd.Series(v["visits_per_customer_merchant"]).corr(real, method="spearman")), 3) for k, v in res.items()})
    scal = ["novel_share", "rare_share", "merchants_per_customer", "top1_share", "top3_share", "tv_all", "tv_novel", "tv_repeat"]
    print(pd.DataFrame({k: {s: v.get(s) for s in scal} for k, v in res.items()}).round(3).to_string())
    for key in ("share_all", "share_novel", "share_repeat", "visits_per_customer_merchant", "distinct_merchants_per_customer", "top1_category"):
        print(key); print(pd.DataFrame({k: v[key] for k, v in res.items()}).round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
