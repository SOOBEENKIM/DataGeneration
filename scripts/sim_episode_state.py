"""ibm_episode_state_v1 stage 1 (docs/ibm_episode_state_v1/PROTOCOL.md): CPU free-run simulation of the candidate label processes.
    ./run python scripts/sim_episode_state.py
Train-only estimation and selection; dev is used only for the shape check of the selected candidate. Never reads a test file.
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
from benchmarks.argn_episode_state import fit_episode_tables, hazard, row_states, mbin, VARIANTS, BURST_GAP  # noqa: E402
from benchmarks.argn_generation_module import MAX_AGE  # noqa: E402

ART = ROOT / "artifacts"; PREP = ART / "ibm_tabformer_v1/prepared"; OUT = ART / "ibm_episode_state_v1"; DOCS = ROOT / "docs/ibm_episode_state_v1"
FROZEN = ART / "ibm_generation_module_v1/transition_table.json"
REPS = 200
Q = (.25, .5, .75, .9)


def paths(frame):
    """Per customer: length and the calendar bin of the previous event for every row (row 0 -> bin 0)."""
    d, s = row_states(frame); L = d.groupby("entity_id", sort=False).size().to_numpy(); M = np.zeros((len(L), int(L.max())), dtype=np.int8)
    c = np.repeat(np.arange(len(L)), L); i = d.groupby("entity_id", sort=False).cumcount().to_numpy(); M[c, i] = s["month"]
    return d, s, L, M


def real_stats(d, s):
    f = pd.DataFrame(dict(g=d.entity_id.to_numpy(), y=s["y"], prev=s["prev"], dist=s["dist"], ever=s["ever"])); f["i"] = f.groupby("g", sort=False).cumcount(); n = f.groupby("g", sort=False).y.transform("size")
    onset = f.y.eq(1) & f.prev.ne(1); burst = onset & (f.ever.eq(0) | f.dist.gt(BURST_GAP)); per = f.groupby("g", sort=False).agg(k=("y", "sum")); per["bursts"] = burst.groupby(f.g, sort=False).sum(); per["runs"] = onset.groupby(f.g, sort=False).sum()
    ln = f.groupby("g", sort=False).y.size(); fi = f[f.y.eq(1)].groupby("g", sort=False).i.min(); first = fi / ln.reindex(fi.index); fc = per[per.k.gt(0)]
    return summarize(int(f.y.sum()), int(onset.sum()), fc.k.to_numpy(), fc.bursts.to_numpy(), fc.runs.to_numpy(), first.to_numpy(), int(f.g.nunique()), int(len(f)))


def summarize(frauds, runs, k, bursts, runs_per, rel, customers, events, reps=1):
    return dict(frauds=frauds / reps, runs=runs / reps, fraud_customers=len(k) / reps, rate=frauds / events, customers=customers, frauds_per_fraud_customer_quantiles={str(q): float(np.quantile(k, q)) for q in Q},
                frauds_per_fraud_customer_mean=float(k.mean()), frauds_per_fraud_customer_max=int(k.max()), share_above_27=float((k > 27).mean()), bursts_per_fraud_customer=float(bursts.mean()),
                runs_per_fraud_customer=float(runs_per.mean()), first_fraud_relpos_median=float(np.median(rel)), first_fraud_relpos_quantiles={str(q): float(np.quantile(rel, q)) for q in (.1, .25, .5, .75, .9)})


def simulate(prob, L, M, reps, seed):
    """prob(prev, age, ever, dist, k, month) -> P(fraud). Chains sorted by length so the active set shrinks."""
    rng = np.random.default_rng(seed); nc = len(L); cust = np.tile(np.arange(nc), reps); rep = np.repeat(np.arange(reps), nc); order = np.argsort(-L[cust], kind="stable"); cust, rep = cust[order], rep[order]; Lc = L[cust]; n = len(cust)
    prev = np.full(n, -1, dtype=np.int64); age = np.zeros(n, dtype=np.int64); ever = np.zeros(n, dtype=np.int64); dist = np.zeros(n, dtype=np.int64); k = np.zeros(n, dtype=np.int64)
    runs = np.zeros(n, dtype=np.int64); bursts = np.zeros(n, dtype=np.int64); first = np.full(n, -1, dtype=np.int64); kk = n
    for t in range(int(Lc[0])):
        while kk > 0 and Lc[kk - 1] <= t:
            kk -= 1
        s = slice(0, kk); p = prob(prev[s], age[s], ever[s], dist[s], k[s], M[cust[s], t].astype(np.int64)); lab = (rng.random(kk) < p).astype(np.int64)
        onset = (lab == 1) & (prev[s] != 1); runs[s] += onset; bursts[s] += onset & ((ever[s] == 0) | (dist[s] > BURST_GAP)); first[s] = np.where((first[s] < 0) & (lab == 1), t, first[s])
        same = (prev[s] == lab); age[s] = np.where(same, age[s] + 1, 1); prev[s] = lab; ever[s] = np.maximum(ever[s], lab); k[s] += lab; dist[s] = np.where(lab == 1, 0, dist[s]) + 1
    fc = k > 0; per_rep = pd.DataFrame(dict(rep=rep, k=k, fc=fc.astype(int), runs=runs)).groupby("rep").sum()
    out = summarize(int(k.sum()), int(runs.sum()), k[fc], bursts[fc], runs[fc], first[fc] / Lc[fc], nc, int(Lc.sum()) // reps, reps=reps); out["rate"] = float(k.sum() / Lc.sum())
    out["per_rep"] = {name: dict(mean=float(v.mean()), sd=float(v.std()), p2_5=float(v.quantile(.025)), p97_5=float(v.quantile(.975))) for name, v in (("frauds", per_rep.k), ("runs", per_rep.runs), ("fraud_customers", per_rep.fc))}
    return out


def frozen_prob(T):
    def f(prev, age, ever, dist, k, month):
        return T[np.where(prev < 0, 0, prev + 1), np.where(prev < 0, 0, np.minimum(age, MAX_AGE)), ever]
    return f


def check(sim, real):
    c = {}; rel = lambda a, b: abs(a - b) / b
    c["frauds_10pct"] = rel(sim["frauds"], real["frauds"]) <= .10; c["runs_10pct"] = rel(sim["runs"], real["runs"]) <= .10; c["fraud_customers_10pct"] = rel(sim["fraud_customers"], real["fraud_customers"]) <= .10
    c["quantiles_within_3"] = all(abs(sim["frauds_per_fraud_customer_quantiles"][str(q)] - real["frauds_per_fraud_customer_quantiles"][str(q)]) <= 3 for q in Q)
    c["share_above_27_le_5pct"] = sim["share_above_27"] <= .05; c["relpos_median_within_.10"] = abs(sim["first_fraud_relpos_median"] - real["first_fraud_relpos_median"]) <= .10
    c["bursts_within_.15"] = abs(sim["bursts_per_fraud_customer"] - real["bursts_per_fraud_customer"]) <= .15
    return {k: bool(v) for k, v in c.items()}


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(PREP / "train.parquet"); d, s, L, M = paths(train); real = real_stats(d, s); res = dict(reps=REPS, train_real=real, candidates={}, test_read=False)
    T0 = np.asarray(json.loads(FROZEN.read_text())["table"]); res["candidates"]["A0"] = dict(sim=simulate(frozen_prob(T0), L, M, REPS, 10)); res["candidates"]["A0"]["checks"] = check(res["candidates"]["A0"]["sim"], real)
    tables = {}
    for i, v in enumerate(VARIANTS):
        tb = fit_episode_tables(train, v); tables[v] = tb; (OUT / f"episode_tables_{v}.json").write_text(json.dumps(tb, indent=1) + "\n")
        forced = float(hazard(tb, s["prev"], s["age"], s["ever"], s["dist"], s["k"], s["month"]).sum())
        sim = simulate(lambda *a, tb=tb: hazard(tb, *a), L, M, REPS, 11 + i); res["candidates"][v] = dict(forced_expected_frauds=forced, sim=sim, checks=check(sim, real), cells=len(tb["counts"]))
        print(v, "forced", round(forced, 1), json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in sim.items() if not isinstance(x, dict)}), res["candidates"][v]["checks"], flush=True)
    selected = next((v for v in VARIANTS if all(res["candidates"][v]["checks"].values())), None); res["selected"] = selected
    # train 240-customer subsample range for level statistics (what a split of this size looks like)
    rng = np.random.default_rng(20261107); ids = d.entity_id.unique(); f = pd.DataFrame(dict(g=d.entity_id.to_numpy(), y=s["y"])); per = f.groupby("g").y.sum(); sub = []
    for _ in range(200):
        pick = rng.choice(ids, 240, replace=False); kk = per.reindex(pick); sub.append((int(kk.sum()), int((kk > 0).sum())))
    sub = np.asarray(sub); res["train_subsample_240"] = dict(frauds=dict(p2_5=float(np.quantile(sub[:, 0], .025)), median=float(np.median(sub[:, 0])), p97_5=float(np.quantile(sub[:, 0], .975))),
                                                             fraud_customers=dict(p2_5=float(np.quantile(sub[:, 1], .025)), median=float(np.median(sub[:, 1])), p97_5=float(np.quantile(sub[:, 1], .975))))
    dev = pd.read_parquet(PREP / "validation.parquet"); dd, ds, dL, dM = paths(dev); res["dev_real"] = real_stats(dd, ds)
    if selected:
        res["dev_sim_selected"] = simulate(lambda *a: hazard(tables[selected], *a), dL, dM, REPS, 20)
    res["dev_sim_A0"] = simulate(frozen_prob(T0), dL, dM, REPS, 21)
    res["seconds"] = round(time.time() - t0, 1); (DOCS / "SIM.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    cols = ["frauds", "runs", "fraud_customers", "frauds_per_fraud_customer_mean", "frauds_per_fraud_customer_max", "share_above_27", "bursts_per_fraud_customer", "runs_per_fraud_customer", "first_fraud_relpos_median"]
    tab = pd.DataFrame({"train_real": real} | {v: c["sim"] for v, c in res["candidates"].items()}).loc[cols]; print(tab.astype(float).round(3).to_string())
    print("quantiles", {k: (real if k == "train_real" else res["candidates"][k]["sim"])["frauds_per_fraud_customer_quantiles"] for k in ["train_real", "A0", *VARIANTS]})
    print("checks", {v: c["checks"] for v, c in res["candidates"].items()}); print("selected", selected, "subsample240", res["train_subsample_240"])
    show = ["frauds", "runs", "fraud_customers", "frauds_per_fraud_customer_mean", "frauds_per_fraud_customer_max", "bursts_per_fraud_customer", "runs_per_fraud_customer", "first_fraud_relpos_median"]
    print(pd.DataFrame({k: res[k] for k in ["dev_real", "dev_sim_selected", "dev_sim_A0"] if k in res}).loc[show].astype(float).round(3).to_string())
    print("dev quantiles", {k: res[k]["frauds_per_fraud_customer_quantiles"] for k in ["dev_real", "dev_sim_selected", "dev_sim_A0"] if k in res}, "seconds", res["seconds"])


if __name__ == "__main__":
    main()
