"""ibm_row_joint_diag_v1 (docs/ibm_row_joint_diag_v1/PROTOCOL.md): where is ARGN's row-only detection gap?
A. TSTR by feature group (row-only detector).  B. oracle column substitution from real train rows of the same label.
Diagnosis only: no fit, no generation, CPU; train + development + existing generations; the sealed test is never read.
    ./run python scripts/diag_row_joint.py
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmarks.fraud_utility as fu  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

SF = ROOT / "artifacts"
IBM = SF / "ibm_tabformer_v1/prepared"
OUT = ROOT / "docs/ibm_row_joint_diag_v1"
FITS, GENS = [20261108, 20261109], [20261011, 20261012]
A_COL, G_COL, C_COL, CH_COL = "amount_or_numeric_value", "gap", "category", "use_chip"
_TEST = None


def sources():
    s = []
    for fs in FITS:
        for gs in GENS:
            s.append(("ARGN_table_kind", f"{fs}_{gs}", SF / f"ibm_run_gap_v1/runs/table_kind_{fs}/generated_validation_{gs}.parquet", None))
            s.append(("RTF_table", f"{fs}_{gs}", SF / f"ibm_realtabformer_v1/runs/rtf_table_{fs}/generated_validation_{gs}.parquet", None))
    for i in range(4):
        s.append(("real_train_subset", str(i), IBM / "train.parquet", 20261108 + i))
    return s


def load(path, draw):
    d = pd.read_parquet(path)
    if draw is not None:
        ids = np.sort(d.entity_id.unique()); d = d[d.entity_id.isin(np.random.default_rng(draw).choice(ids, size=240, replace=False))]
    return d


def groups():
    cat = [c for c in fu.ROW if c.startswith("cat_")]; ch = [c for c in fu.ROW if c.startswith("chan_")]
    g = dict(amount=["log_amount"], gap=["log_gap"], category=cat, channel=ch, merchant_repeat=["seen_merchant"], position=["log_position"])
    out = {k: v for k, v in g.items()}
    out["category+channel"] = cat + ch; out["category+amount"] = cat + ["log_amount"]; out["channel+amount"] = ch + ["log_amount"]
    out["category+channel+amount"] = cat + ch + ["log_amount"]
    for k, v in g.items():
        out[f"all_minus_{k}"] = [c for c in fu.ROW if c not in v]
    out["all_row"] = list(fu.ROW)
    return out


def _init(dev_path):
    global _TEST
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    real = pd.read_parquet(dev_path); fu.configure(real); _TEST = fu.features(real)


def score(tr, cols):
    clf = fu.fit(tr, cols, detector="stable"); s = clf.predict_proba(_TEST[cols].to_numpy())[:, 1]
    return float(average_precision_score(_TEST.y, s)), float(roc_auc_score(_TEST.y, s))


def job_groups(job):
    name, tag, path, draw = job
    tr = fu.features(load(path, draw)); rows = []
    for g, cols in groups().items():
        ap, au = score(tr, cols); rows.append(dict(source=name, run=tag, group=g, n_features=len(cols), auprc=ap, auroc=au))
    return rows


SWAPS = {  # name -> (labels to modify, columns)
    "none": ((), ()),
    "fraud:amount": ((1,), (A_COL,)), "fraud:gap": ((1,), (G_COL,)), "fraud:category": ((1,), (C_COL,)), "fraud:channel": ((1,), (CH_COL,)),
    "fraud:category+channel": ((1,), (C_COL, CH_COL)), "fraud:category+channel+amount": ((1,), (C_COL, CH_COL, A_COL)),
    "fraud:category+channel+amount+gap": ((1,), (C_COL, CH_COL, A_COL, G_COL)),
    "normal:category+channel+amount": ((0,), (C_COL, CH_COL, A_COL)), "normal:category+channel+amount+gap": ((0,), (C_COL, CH_COL, A_COL, G_COL)),
    "both:category+channel+amount+gap": ((0, 1), (C_COL, CH_COL, A_COL, G_COL)),
}


def job_swap(job):
    name, tag, path, swap = job
    d = pd.read_parquet(path).sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    for c in (C_COL, CH_COL):
        d[c] = d[c].astype(str).astype(object)
    d[A_COL] = pd.to_numeric(d[A_COL], errors="coerce").astype(float); d[G_COL] = pd.to_numeric(d[G_COL], errors="coerce").astype(float)
    labels, cols = SWAPS[swap]
    if labels:
        donor = pd.read_parquet(IBM / "train.parquet"); dy = pd.to_numeric(donor.event_is_fraud, errors="coerce").fillna(0).astype(int)
        donor = donor[donor.event_index.gt(0)]; dy = dy[donor.index]  # donors have a gap
        y = pd.to_numeric(d.event_is_fraud, errors="coerce").fillna(0).astype(int); rng = np.random.default_rng(20261110)
        for lab in labels:
            idx = np.flatnonzero((y.to_numpy() == lab) & d.event_index.gt(0).to_numpy()); pool = donor[dy.eq(lab)]
            pick = pool.iloc[rng.integers(len(pool), size=len(idx))]
            for c in cols:
                v = pick[c].to_numpy(); d.loc[idx, c] = v.astype(str) if c in (C_COL, CH_COL) else pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(dtype=float)
    tr = fu.features(d); out = dict(source=name, run=tag, swap=swap)
    for v, cols_ in (("row", list(fu.ROW)), ("history", list(fu.ROW) + fu.HISTORY)):
        ap, au = score(tr, cols_); out[f"{v}_auprc"] = ap; out[f"{v}_auroc"] = au
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); dev = str(IBM / "validation.parquet")
    src = [s for s in sources() if Path(s[2]).exists()]
    swap_jobs = [(n, t, p, s) for (n, t, p, dr) in src if n == "ARGN_table_kind" for s in SWAPS]
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(dev,)) as ex:
        ga = [r for rows in ex.map(job_groups, src) for r in rows]
        pd.DataFrame(ga).to_csv(OUT / "FEATURE_GROUPS.csv", index=False); print("groups done", round(time.time() - t0), "s", flush=True)
        sw = list(ex.map(job_swap, swap_jobs))
    pd.DataFrame(sw).to_csv(OUT / "SUBSTITUTION.csv", index=False)
    a = pd.DataFrame(ga).pivot_table(index="group", columns="source", values="auprc", sort=False); print(a.round(3).to_string())
    b = pd.DataFrame(sw).groupby("swap", sort=False)[["row_auprc", "history_auprc", "row_auroc"]].agg(["mean", "std"]); print(b.round(3).to_string())
    json.dump(dict(seconds=round(time.time() - t0), sources=len(src), swap_jobs=len(swap_jobs), donor="train.parquet rows with event_index > 0, same label, seed 20261110",
                   detector="stable", test_read=False), open(OUT / "RUN.json", "w"), indent=1)


if __name__ == "__main__":
    main()
