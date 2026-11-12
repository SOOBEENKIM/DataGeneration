"""ibm_normal_rows_v1 amendment 4 (docs/ibm_normal_rows_v1/PROTOCOL.md): replicate draws of ALL2 / ALL3 / ALL4 on the same four dev
generations (5 random draws each), and ALL4 = ALL3 + fraud amounts re-referenced to the new personal median.
    ./run python scripts/sim_normal_rows_rep.py
CPU only, thread-limited; train + dev + existing dev generations; never a test file; no training of anything but detectors.
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
import benchmarks.fraud_utility as fu  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sim_normal_rows import base_frame, finish, PREP, OUT, DOCS, BASE, FITS, Y  # noqa: E402
from sim_normal_rows_v2 import apply_rec2, apply_amt2, apply_amt3, TABLES, TABLES3  # noqa: E402

REPS = 5
ARMS = ("ALL2", "ALL3", "ALL4")
G = dict(T=None, base=None, ref=None, test=None, cols=None)


def personal_median(d):
    """For every row: median of the customer's previous NORMAL positive log amounts (through the last normal row before it) and how many there were."""
    n = d[d.y.eq(0) & d.a.gt(0)]; la = np.log(n.a); med = la.groupby(n.entity_id, sort=False).expanding().median().reset_index(level=0, drop=True); cnt = n.groupby("entity_id", sort=False).cumcount() + 1
    s = pd.Series(np.nan, index=d.index); s.loc[n.index] = med.reindex(n.index).to_numpy(); c = pd.Series(np.nan, index=d.index); c.loc[n.index] = cnt.to_numpy()
    return s.groupby(d.entity_id, sort=False).ffill(), c.groupby(d.entity_id, sort=False).ffill().fillna(0)


def fraud_ref(new, old):
    ro, co = personal_median(old); rn, _ = personal_median(new); sel = (old.y.eq(1) & old.a.gt(0) & co.ge(5) & rn.notna() & ro.notna()).to_numpy(); a = new.a.to_numpy().copy()
    a[sel] = np.maximum(0.01, np.round(np.exp(np.log(old.a.to_numpy()[sel]) - ro.to_numpy()[sel] + rn.to_numpy()[sel]), 2)); new = new.copy(); new["a"] = a
    return new, dict(fraud_rows=int(old.y.eq(1).sum()), re_referenced=int(sel.sum()))


def build(arm, i, rep):
    rng = np.random.default_rng([20261114, i, rep, ARMS.index(arm)]); d = G["base"][i].copy(); info = {}
    d, info["REC2"] = apply_rec2(d, G["T"], rng); d, info["AMT"] = (apply_amt2 if arm == "ALL2" else apply_amt3)(d, G["T"], rng)
    if arm == "ALL4":
        d, info["FRAUD"] = fraud_ref(d, G["base"][i])
    return finish(d, G["ref"][i]), info


def job(spec):
    from threadpoolctl import threadpool_limits
    arm, i, rep = spec; t0 = time.time()
    with threadpool_limits(4):
        if arm == "base":
            out, info = G["ref"][i], {}
        else:
            out, info = build(arm, i, rep)
            if rep == 0 and arm == "ALL4":
                fs, gs = FITS[i]; folder = OUT / "sim4" / f"ALL4_{fs}"; folder.mkdir(parents=True, exist_ok=True); out.to_parquet(folder / f"generated_validation_{gs}.parquet", index=False)
        f = fu.features(out); clf = fu.fit(f, G["cols"], detector="stable"); score = clf.predict_proba(G["test"][G["cols"]].to_numpy())[:, 1]; ep = fu.episode_metrics(G["test"], score)
    return dict(arm=arm, generation=i, rep=rep, history_auprc=float(average_precision_score(G["test"].y, score)), early=ep["early_detection_top1"], fraud_amount_vs_med20=float(f[f.y.eq(1)].amount_vs_med20.mean()),
                re_referenced=info.get("FRAUD", {}).get("re_referenced"), seconds=round(time.time() - t0, 1))


def main():
    t0 = time.time(); T = json.loads(TABLES.read_text()); T["map"] = json.loads(TABLES3.read_text()); G["T"] = T
    raws = [pd.read_parquet(BASE(fs, gs)) for fs, gs in FITS]; G["base"] = [base_frame(r) for r in raws]; G["ref"] = [r.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True) for r in raws]
    dev = pd.read_parquet(PREP / "validation.parquet"); fu.configure(dev); G["test"] = fu.features(dev); G["cols"] = list(fu.ROW) + list(fu.HISTORY)
    jobs = [("base", i, 0) for i in range(4)] + [(a, i, r) for a in ARMS for r in range(REPS) for i in range(4)]; rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(job, jobs):
            rows.append(r); print(f"{r['arm']:5s} gen {r['generation']} rep {r['rep']} auprc={r['history_auprc']:.4f} fraud_amt={r['fraud_amount_vs_med20']:.3f} {r['seconds']}s", flush=True)
    res = pd.DataFrame(rows); res.to_csv(DOCS / "REPLICATES.csv", index=False); b = res[res.arm.eq("base")].set_index("generation").history_auprc; summ = {}
    for a in ARMS:
        x = res[res.arm.eq(a)].copy(); x["diff"] = x.history_auprc - x.generation.map(b); per_gen = x.groupby("generation")["diff"].mean(); within = x.groupby("generation")["diff"].std()
        summ[a] = dict(mean_utility=float(x.history_auprc.mean()), mean_diff=float(x["diff"].mean()), diff_by_generation=per_gen.round(4).tolist(), positive_of_20=int((x["diff"] > 0).sum()),
                       sd_across_draws_within_generation=float(within.mean()), se_draws=float(x["diff"].std() / np.sqrt(len(x))), se_generations=float(per_gen.std() / np.sqrt(len(per_gen))),
                       fraud_amount_vs_med20=float(x.fraud_amount_vs_med20.mean()), early=float(x.early.mean()))
    pair = {}
    for a, c in (("ALL2", "ALL3"), ("ALL2", "ALL4"), ("ALL4", "ALL3")):
        da = res[res.arm.eq(a)].set_index(["generation", "rep"]).history_auprc; dc = res[res.arm.eq(c)].set_index(["generation", "rep"]).history_auprc; d = (da - dc); pg = d.groupby("generation").mean()
        pair[f"{a}-{c}"] = dict(mean=float(d.mean()), se_draws=float(d.std() / np.sqrt(len(d))), se_generations=float(pg.std() / np.sqrt(len(pg))), by_generation=pg.round(4).tolist())
    out = dict(base=dict(utility=b.round(4).tolist(), mean=float(b.mean()), fraud_amount_vs_med20=float(res[res.arm.eq("base")].fraud_amount_vs_med20.mean())), arms=summ, pairs=pair, reps=REPS, seconds=round(time.time() - t0, 1), test_read=False)
    (DOCS / "REPLICATES.json").write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
