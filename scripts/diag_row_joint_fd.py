"""ibm_row_joint_diag_v1 addition C: merchant -> category/channel consistency (C1) and FD what-if reconstruction (C2).
Diagnosis only; train + development + existing generations; the sealed test is never read.
    ./run python scripts/diag_row_joint_fd.py
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import benchmarks.fraud_utility as fu  # noqa: E402
from diag_row_joint import _init, score, IBM, SF, FITS, GENS  # noqa: E402

OUT = ROOT / "docs/ibm_row_joint_diag_v1"
M, C_COL, CH_COL, Y = "receiver_or_mark", "category", "use_chip", "event_is_fraud"
ONLINE = "Online Transaction"; RARE = "_RARE_"
CHAN_TABLE = (0.7566248256624826, 0.951530612244898)  # ibm_merchant_memory_v1 amendment 4: P(online | fraud, novel / repeat)


def prep(d):
    d = d.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    for c in (M, C_COL, CH_COL):
        d[c] = d[c].astype(str).astype(object)
    d["y"] = pd.to_numeric(d[Y], errors="coerce").fillna(0).astype(int)
    return d


def merchant_attributes(train):
    g = train.groupby(M)
    return pd.DataFrame(dict(rows=g.size(), modal_cat=g[C_COL].agg(lambda s: s.value_counts().index[0]), online_share=g[CH_COL].apply(lambda s: s.eq(ONLINE).mean()))).assign(
        modal_online=lambda a: a.online_share.ge(0.5))


def consistency(d, att, joint_ref=None):
    d = d.merge(att[["modal_cat", "modal_online"]], left_on=M, right_index=True, how="left"); known = d.modal_cat.notna()
    on = d[CH_COL].eq(ONLINE); out = {}
    for lab, nm in ((0, "normal"), (1, "fraud")):
        s = d.y.eq(lab); k = s & known
        o = dict(rows=int(s.sum()), rare_share=float(d.loc[s, M].eq(RARE).mean()), known_merchant=float(known[s].mean()), online=float(on[s].mean()),
                 channel_matches_merchant=float((on[k] == d.loc[k, "modal_online"].astype(bool)).mean()) if k.any() else None,
                 category_matches_merchant=float(d.loc[k, C_COL].eq(d.loc[k, "modal_cat"]).mean()) if k.any() else None,
                 online_at_swipe_merchant=float((on[k] & ~d.loc[k, "modal_online"].astype(bool)).mean()) if k.any() else None)
        j = (d.loc[s, C_COL] + "|" + d.loc[s, CH_COL]).value_counts(normalize=True)
        if joint_ref is not None:
            o["category_x_channel_tv"] = float(0.5 * j.sub(joint_ref[nm], fill_value=0).abs().sum())
        o["_joint"] = j
        out[nm] = o
    return out


def what_if(d, variant, att, rare_pool):
    d = d.copy(); rng = np.random.default_rng(20261110)
    known = d[M].isin(att.index).to_numpy(); rare = d[M].eq(RARE).to_numpy(); normal = d.y.eq(0).to_numpy(); fraud = ~normal
    if variant in ("fd_normal", "fd_normal+rare", "fd_all"):
        idx = np.flatnonzero(normal & known & ~rare)
        a = att.loc[d.loc[idx, M]]; d.loc[idx, C_COL] = a.modal_cat.to_numpy(); d.loc[idx, CH_COL] = np.where(a.modal_online.to_numpy(), ONLINE, "Swipe Transaction")
    if variant in ("fd_normal+rare", "fd_all"):
        idx = np.flatnonzero(normal & rare); pick = rare_pool.iloc[rng.integers(len(rare_pool), size=len(idx))]
        d.loc[idx, C_COL] = pick[C_COL].to_numpy(); d.loc[idx, CH_COL] = pick[CH_COL].to_numpy()
    if variant == "fd_all":
        idx = np.flatnonzero(fraud & known & ~rare); d.loc[idx, C_COL] = att.loc[d.loc[idx, M], "modal_cat"].to_numpy()
        novel = (~d.duplicated(["entity_id", M])).to_numpy(); fi = np.flatnonzero(fraud)
        p = np.where(novel[fi], CHAN_TABLE[0], CHAN_TABLE[1]); d.loc[fi, CH_COL] = np.where(rng.random(len(fi)) < p, ONLINE, "Swipe Transaction")
    return d


def job(j):
    path, tag, variant, att, rare_pool = j
    d = prep(pd.read_parquet(path)); d = what_if(d, variant, att, rare_pool) if variant != "none" else d
    tr = fu.features(d.drop(columns="y")); out = dict(run=tag, variant=variant)
    for v, cols in (("row", list(fu.ROW)), ("history", list(fu.ROW) + fu.HISTORY), ("category+channel", [c for c in fu.ROW if c.startswith(("cat_", "chan_"))])):
        ap, au = score(tr, cols); out[f"{v}_auprc"] = ap
    return out


def main():
    t0 = time.time(); train = prep(pd.read_parquet(IBM / "train.parquet")); dev = prep(pd.read_parquet(IBM / "validation.parquet"))
    att = merchant_attributes(train)
    from mostlyai.engine._workspace import Workspace
    codes = set(map(str, Workspace(SF / "ibm_tabformer_v1/runs/native_20261108/workspace").tgt_stats.read()["columns"][M]["codes"].keys()))
    rare_pool = train[train.y.eq(0) & ~train[M].isin(codes)][[C_COL, CH_COL]]
    ref = consistency(dev, att); joint_ref = {k: v["_joint"] for k, v in ref.items()}
    res = dict(train_merchants=int(len(att)), argn_merchant_codes=len(codes), rare_normal_train_rows=int(len(rare_pool)),
               train_rare_row_share=float((~train[M].isin(codes)).mean()))
    def clean(c):
        return {lab: {k: v for k, v in o.items() if k != "_joint"} for lab, o in c.items()}
    res["real_dev"] = clean(consistency(dev, att, joint_ref)); res["real_train"] = clean(consistency(train, att, joint_ref))
    arms = {"ARGN_table_kind": "ibm_run_gap_v1/runs/table_kind_{fs}", "ARGN_native": "ibm_tabformer_v1/runs/native_{fs}", "RTF_table": "ibm_realtabformer_v1/runs/rtf_table_{fs}"}
    for a, t in arms.items():
        rs = [clean(consistency(prep(pd.read_parquet(SF / (t.format(fs=fs) + f"/generated_validation_{gs}.parquet"))), att, joint_ref)) for fs in FITS for gs in GENS]
        res[a] = {lab: {k: (float(np.mean([r[lab][k] for r in rs if r[lab][k] is not None])) if any(r[lab][k] is not None for r in rs) else None) for k in rs[0][lab]} for lab in rs[0]}
    jobs = [(SF / f"ibm_run_gap_v1/runs/table_kind_{fs}/generated_validation_{gs}.parquet", f"{fs}_{gs}", v, att, rare_pool)
            for fs in FITS for gs in GENS for v in ("none", "fd_normal", "fd_normal+rare", "fd_all")]
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(str(IBM / "validation.parquet"),)) as ex:
        sw = pd.DataFrame(list(ex.map(job, jobs)))
    sw.to_csv(OUT / "FD_WHAT_IF.csv", index=False)
    res["what_if_mean"] = sw.groupby("variant", sort=False)[["row_auprc", "history_auprc", "category+channel_auprc"]].mean().round(4).to_dict("index")
    res["seconds"] = round(time.time() - t0); json.dump(res, open(OUT / "FD_CONSISTENCY.json", "w"), indent=1)
    for k in ("real_dev", "real_train", "ARGN_native", "ARGN_table_kind", "RTF_table"):
        for lab in ("normal", "fraud"):
            print(k, lab, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in res[k][lab].items()})
    print({k: res[k] for k in ("train_merchants", "argn_merchant_codes", "rare_normal_train_rows", "train_rare_row_share")})
    print(sw.groupby("variant", sort=False)[["row_auprc", "history_auprc", "category+channel_auprc"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
