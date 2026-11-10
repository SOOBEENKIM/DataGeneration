"""Diagnosis for ibm_merchant_memory_v1 amendment 4 (before registration): why does the fraud rows' online share fall
when the merchant-memory component replaces the merchant token on RTF, and how tightly do channel and category follow
the merchant in the real data? CPU only, train + development + existing generations; the sealed test is never read.
    ./run python scripts/diag_rtf_channel.py
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SF = ROOT / "artifacts"
IBM = SF / "ibm_tabformer_v1/prepared"
OUT = ROOT / "docs/ibm_merchant_memory_v1/DIAG_RTF_CHANNEL.json"
M, CAT, CH, Y = "receiver_or_mark", "category", "use_chip", "event_is_fraud"
ONLINE = "Online Transaction"


def prep(d):
    d = d.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    d[M] = d[M].astype(str); d[CAT] = d[CAT].astype(str); d[CH] = d[CH].astype(str)
    d["y"] = pd.to_numeric(d[Y], errors="coerce").fillna(0).astype(int)
    d["novel"] = ~d.duplicated(["entity_id", M])
    return d


def main():
    train = prep(pd.read_parquet(IBM / "train.parquet")); dev = prep(pd.read_parquet(IBM / "validation.parquet"))
    # merchant attributes in train: modal channel / category, purity, row count
    g = train.groupby(M)
    att = pd.DataFrame(dict(rows=g.size(), online_share=g[CH].apply(lambda s: s.eq(ONLINE).mean()),
                            modal_cat=g[CAT].agg(lambda s: s.value_counts().index[0]), cat_purity=g[CAT].agg(lambda s: s.value_counts(normalize=True).iloc[0])))
    att["modal_online"] = att.online_share.ge(0.5)
    att["ch_purity"] = np.maximum(att.online_share, 1 - att.online_share)
    pool = set(train.loc[train.y.eq(1), M])
    res = dict(train_merchants=int(len(att)), pool_merchants=len(pool),
               merchant_purity_row_weighted=dict(channel=float((att.ch_purity * att.rows).sum() / att.rows.sum()), category=float((att.cat_purity * att.rows).sum() / att.rows.sum())),
               pool=dict(rows_per_merchant_median=float(att.loc[list(pool), "rows"].median()), rows_per_merchant_mean=float(att.loc[list(pool), "rows"].mean()),
                         modal_online_share_of_merchants=float(att.loc[list(pool), "modal_online"].mean()),
                         channel_purity_mean=float(att.loc[list(pool), "ch_purity"].mean()), category_purity_mean=float(att.loc[list(pool), "cat_purity"].mean())))
    fr = train[train.y.eq(1)]
    res["train_fraud"] = dict(online=float(fr[CH].eq(ONLINE).mean()), online_novel=float(fr.loc[fr.novel, CH].eq(ONLINE).mean()), online_repeat=float(fr.loc[~fr.novel, CH].eq(ONLINE).mean()),
                              rows=int(len(fr)), novel_share=float(fr.novel.mean()))

    def table(d, name):
        d = d.merge(att[["modal_online", "modal_cat", "rows"]], left_on=M, right_index=True, how="left")
        d["known"] = d.rows.notna(); d["in_pool"] = d[M].isin(pool)
        d["ch_match"] = d[CH].eq(ONLINE) == d.modal_online.fillna(False).astype(bool); d["cat_match"] = d[CAT].eq(d.modal_cat)
        out = {}
        for lab, nm in ((1, "fraud"), (0, "normal")):
            s = d[d.y.eq(lab)]
            if not len(s):
                continue
            k = s[s.known]
            o = dict(rows=int(len(s)), online=float(s[CH].eq(ONLINE).mean()), known_merchant=float(s.known.mean()), in_pool=float(s.in_pool.mean()),
                     channel_matches_merchant=float(k.ch_match.mean()) if len(k) else None, category_matches_merchant=float(k.cat_match.mean()) if len(k) else None)
            if lab == 1:
                for key, sel in (("novel_pool", s.novel & s.in_pool), ("novel_other", s.novel & ~s.in_pool), ("repeat", ~s.novel)):
                    q = s[sel]; qk = q[q.known]
                    o[key] = dict(rows=int(len(q)), online=float(q[CH].eq(ONLINE).mean()) if len(q) else None,
                                  merchant_modal_online=float(qk.modal_online.astype(bool).mean()) if len(qk) else None,
                                  channel_matches_merchant=float(qk.ch_match.mean()) if len(qk) else None, category_matches_merchant=float(qk.cat_match.mean()) if len(qk) else None)
            out[nm] = o
        return out

    res["real_dev"] = table(dev, "dev")
    gens = {}
    for arm in ["rtf_native", "rtf_table", "rtf_full", "rtf_fraud_only"]:
        rows = []
        for fs in [20261108, 20261109]:
            for gs in [20261011, 20261012]:
                p = SF / f"ibm_realtabformer_v1/runs/{arm}_{fs}/generated_validation_{gs}.parquet"
                if p.exists():
                    rows.append(table(prep(pd.read_parquet(p)), arm))
        if rows:  # mean over generations of every numeric leaf
            def mean(path):
                vals = []
                for r in rows:
                    x = r
                    for k in path:
                        x = x.get(k) if isinstance(x, dict) else None
                    if x is not None:
                        vals.append(x)
                return float(np.mean(vals)) if vals else None
            agg = {}
            for nm in rows[0]:
                agg[nm] = {}
                for k, v in rows[0][nm].items():
                    agg[nm][k] = {kk: mean((nm, k, kk)) for kk in v} if isinstance(v, dict) else mean((nm, k))
            gens[arm] = dict(generations=len(rows), **agg)
    res["generated"] = gens
    OUT.write_text(json.dumps(res, indent=1) + "\n"); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
