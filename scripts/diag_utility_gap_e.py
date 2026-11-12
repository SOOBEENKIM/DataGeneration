"""ibm_gap_diag_v1 amendment 1 (docs/ibm_gap_diag_v1/PROTOCOL.md): the D2 row swap redone on the 24 independent dev generations of
ibm_normal_impl_v1, arms B (the model before the normal-row part) and RA (the adopted composition).
    OMP_NUM_THREADS=4 ./run python scripts/diag_utility_gap_e.py [--jobs 8]
CPU only; train + dev + existing dev generations; never a test file; nothing is generated or trained except detectors.
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
import benchmarks.fraud_utility as fu  # noqa: E402
import diag_utility_gap as dg  # noqa: E402

OUT = ROOT / "docs/ibm_gap_diag_v1/a1"; RUNS = ROOT / "artifacts/ibm_normal_impl_v1/runs"; ARMS = ("B", "RA"); BATCH = 4
GENS = [(fs, gs) for fs, seeds in {20261108: [*range(20261100, 20261104), *range(20261200, 20261208)], 20261109: [*range(20261110, 20261114), *range(20261300, 20261308)]}.items() for gs in seeds]
KINDS = {"SS": lambda m, i: [(f"{m}{i}", 1), (f"{m}{i}", 0)], "SfRn": lambda m, i: [(f"{m}{i}", 1), (f"real{i}", 0)], "RfSn": lambda m, i: [(f"real{i}", 1), (f"{m}{i}", 0)]}


def contrast(d):
    d = np.asarray(d, float); return dict(n=int(len(d)), mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), positive=int((d > 0).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); a = ap.parse_args(); t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); assert len(GENS) == 24
    dev = pd.read_parquet(dg.PREP / "validation.parquet"); fu.configure(dev); dg.TEST = fu.features(dev); train = pd.read_parquet(dg.PREP / "train.parquet"); ids = np.sort(train.entity_id.unique()); rows = []
    for b in range(0, len(GENS), BATCH):
        dg.F.clear()
        for i in range(b, b + BATCH):
            fs, gs = GENS[i]; pick = np.random.default_rng(dg.SEED + i).choice(ids, size=dg.N_DEV, replace=False); dg.F[f"real{i}"] = fu.features(train[train.entity_id.isin(pick)])
            for m in ARMS:
                dg.F[f"{m}{i}"] = fu.features(pd.read_parquet(RUNS / f"{m}_{fs}" / f"generated_validation_{gs}.parquet"))
        jobs = [(f"real|RR|{i}|all", [(f"real{i}", 1), (f"real{i}", 0)], []) for i in range(b, b + BATCH)]
        jobs += [(f"{m}|{k}|{i}|all", spec(m, i), []) for i in range(b, b + BATCH) for m in ARMS for k, spec in KINDS.items()]
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:      # forked after dg.F / dg.TEST are filled for this batch
            for r in ex.map(dg._fit_eval, jobs):
                src, kind, i, _ = r["name"].split("|"); fs, gs = GENS[int(i)]; r.update(source=src, kind=kind, draw=int(i), fit_seed=fs, generation_seed=gs); rows.append(r)
                print(f"{r['name']:24s} auprc={r['history_auprc']:.4f} fraud={r['train_fraud']} rows={r['train_rows']} {r['seconds']}s", flush=True)
        print("batch", b // BATCH, round(time.time() - t0), "s", flush=True)
    res = pd.DataFrame(rows); res.to_csv(OUT / "ROW_SWAP_FITS.csv", index=False); w = res.pivot(index="draw", columns=["source", "kind"], values="history_auprc").sort_index(); rr = w[("real", "RR")]; out = dict(ceiling=dict(mean=float(rr.mean()), sd=float(rr.std(ddof=1)), n=int(len(rr))))
    for m in ARMS:
        ss, sfrn, rfsn = w[(m, "SS")], w[(m, "SfRn")], w[(m, "RfSn")]; gap = rr - ss
        out[m] = dict(SS=dict(mean=float(ss.mean()), sd=float(ss.std(ddof=1))), SfRn=dict(mean=float(sfrn.mean()), sd=float(sfrn.std(ddof=1))), RfSn=dict(mean=float(rfsn.mean()), sd=float(rfsn.std(ddof=1))),
                      gap_RR_minus_SS=contrast(gap), SfRn_minus_SS=contrast(sfrn - ss), RfSn_minus_SS=contrast(rfsn - ss), RR_minus_SfRn=contrast(rr - sfrn), RR_minus_RfSn=contrast(rr - rfsn),
                      recovered_by_real_normal=float((sfrn - ss).mean() / gap.mean()), recovered_by_real_fraud=float((rfsn - ss).mean() / gap.mean()))
        v = out[m]; main_cause = "normal rows" if v["recovered_by_real_normal"] >= .6 else "fraud rows" if v["recovered_by_real_fraud"] >= .6 else "neither side alone (both / interaction)"; v["main_cause_by_registered_rule"] = main_cause
        print(m, {k: round(v[k]["mean"], 4) for k in ("SS", "SfRn", "RfSn")}, "gap", round(v["gap_RR_minus_SS"]["mean"], 4), "recovered normal", round(v["recovered_by_real_normal"], 3), "fraud", round(v["recovered_by_real_fraud"], 3),
              "RR-SfRn", round(v["RR_minus_SfRn"]["mean"], 4), "+-", round(v["RR_minus_SfRn"]["se"], 4), "->", main_cause, flush=True)
    out["RA_minus_B"] = {k: contrast(w[("RA", k)] - w[("B", k)]) for k in KINDS}; out.update(seconds=round(time.time() - t0, 1), test_read=False, detector="stable", generations_per_arm=24)
    (OUT / "ROW_SWAP.json").write_text(json.dumps(out, indent=1) + "\n"); print("ceiling", out["ceiling"], "RA-B", {k: (round(v["mean"], 4), round(v["se"], 4)) for k, v in out["RA_minus_B"].items()}); print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
