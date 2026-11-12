"""ibm_episode_state_v1 amendment 1 (docs/ibm_episode_state_v1/PROTOCOL.md): candidate A5 = A4 + hierarchical shrinkage.
    ./run python scripts/sim_episode_state_a1.py
m is chosen by 5-fold customer cross-validated label log-likelihood on train (independent of the simulation statistics);
then one free-run simulation with the unchanged gate, and the dev shape check. CPU only, never reads a test file.
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
from benchmarks.argn_episode_state import fit_episode_tables, hazard, row_states  # noqa: E402
from sim_episode_state import PREP, OUT, DOCS, REPS, paths, real_stats, simulate, check  # noqa: E402

GRID = [2, 5, 10, 20, 50]


def loglik(tb, frame):
    _, s = row_states(frame); p = np.clip(hazard(tb, s["prev"], s["age"], s["ever"], s["dist"], s["k"], s["month"]), 1e-9, 1 - 1e-9)
    return float((s["y"] * np.log(p) + (1 - s["y"]) * np.log1p(-p)).sum()), int(len(p))


def main():
    t0 = time.time(); train = pd.read_parquet(PREP / "train.parquet"); ids = np.sort(train.entity_id.unique()); fold = np.random.default_rng(20261111).permutation(len(ids)) % 5
    cv = {}
    for name, kw in [("A2", dict(variant="A2")), ("A4", dict(variant="A4"))] + [(f"A5_m{m}", dict(variant="A5", shrink=m)) for m in GRID]:
        tot = 0.0; n = 0
        for f in range(5):
            tr = train[train.entity_id.isin(ids[fold != f])]; te = train[train.entity_id.isin(ids[fold == f])]; ll, k = loglik(fit_episode_tables(tr, **kw), te); tot += ll; n += k
        cv[name] = dict(loglik=tot, nll_per_1000_rows=-1000 * tot / n); print(name, round(cv[name]["nll_per_1000_rows"], 5), flush=True)
    m = max(GRID, key=lambda q: cv[f"A5_m{q}"]["loglik"]); tb = fit_episode_tables(train, "A5", shrink=m); (OUT / "episode_tables_A5.json").write_text(json.dumps(tb, indent=1) + "\n")
    d, s, L, M = paths(train); real = real_stats(d, s); forced = float(hazard(tb, s["prev"], s["age"], s["ever"], s["dist"], s["k"], s["month"]).sum())
    sim = simulate(lambda *a: hazard(tb, *a), L, M, REPS, 31); checks = check(sim, real); passed = all(checks.values())
    dev = pd.read_parquet(PREP / "validation.parquet"); dd, ds, dL, dM = paths(dev); dev_real = real_stats(dd, ds); dev_sim = simulate(lambda *a: hazard(tb, *a), dL, dM, REPS, 32)
    res = dict(cv=cv, selected_m=m, train_real=real, forced_expected_frauds=forced, sim=sim, checks=checks, passed=passed, dev_real=dev_real, dev_sim=dev_sim, seconds=round(time.time() - t0, 1), test_read=False)
    (DOCS / "SIM_AMENDMENT1.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    show = ["frauds", "runs", "fraud_customers", "frauds_per_fraud_customer_mean", "frauds_per_fraud_customer_max", "share_above_27", "bursts_per_fraud_customer", "runs_per_fraud_customer", "first_fraud_relpos_median"]
    print("selected m", m, "forced", round(forced, 1)); print(pd.DataFrame(dict(train_real=real, A5=sim, dev_real=dev_real, dev_A5=dev_sim)).loc[show].astype(float).round(3).to_string())
    print("quantiles", {k: v["frauds_per_fraud_customer_quantiles"] for k, v in dict(train_real=real, A5=sim, dev_real=dev_real, dev_A5=dev_sim).items()})
    print("checks", checks, "PASSED" if passed else "FAILED", "| dev per-rep", dev_sim["per_rep"], "| seconds", res["seconds"])


if __name__ == "__main__":
    main()
