"""fraud_utility_v1 runner: TSTR utility of every registered generation plus real references.
    ./run python scripts/run_fraud_utility.py [--limit N] [--real-draws 20] [--jobs 16]
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from benchmarks.fraud_utility import features, evaluate, configure  # noqa: E402
from run_rare_state_persistence import registry, REAL_DEV, REAL_TRAIN, sha, dataset  # noqa: E402

OUT = ROOT / "docs/fraud_utility_v1"
_TEST = None
_N_DEV = 147
_DETECTOR = "registered"


def _init(test_path, threads):
    global _TEST
    from threadpoolctl import threadpool_limits
    threadpool_limits(threads)  # HistGradientBoosting otherwise uses every core in every worker
    real = pd.read_parquet(test_path); configure(real)
    _TEST = features(real)


def _one(job):
    kind, name, src, extra = job
    t0 = time.time()
    try:
        if kind == "generation":
            train = pd.read_parquet(src)
        elif kind == "real_subset":
            train_all = pd.read_parquet(src); ids = np.sort(train_all.entity_id.unique())
            pick = np.random.default_rng(extra).choice(ids, size=_N_DEV, replace=False)
            train = train_all[train_all.entity_id.isin(pick)]
        else:
            train = pd.read_parquet(src)
        out = evaluate(train, _TEST, detector=_DETECTOR)
    except Exception as e:
        out = dict(error=f"{type(e).__name__}: {e}")
    out.update(kind=kind, model=name, source=str(src), draw=extra, seconds=round(time.time() - t0, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--real-draws", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=16); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--dataset", choices=["sparkov", "ibm"], default="sparkov")
    ap.add_argument("--detector", choices=["registered", "stable"], default="registered")
    args = ap.parse_args(); t0 = time.time()
    global OUT, REAL_DEV, REAL_TRAIN, _N_DEV, _DETECTOR
    ds = dataset(args.dataset); REAL_DEV, REAL_TRAIN, _N_DEV, _DETECTOR = ds["real_dev"], ds["real_train"], ds["n_dev"], args.detector
    if args.dataset != "sparkov":
        OUT = ROOT / f"docs/{ds['out'].parent.name}/utility"
    if args.detector != "registered":
        OUT = OUT.parent / f"{OUT.name}_detector_{args.detector}" if args.dataset != "sparkov" else OUT / f"detector_{args.detector}"
    OUT.mkdir(parents=True, exist_ok=True)
    vocab = configure(pd.read_parquet(REAL_DEV))
    reg = ds["registry"](); reg = reg[[Path(p).exists() for p in reg.path]]
    if args.limit:
        reg = reg.head(args.limit)
    jobs = [("generation", f"{r.family}|{r.model}|{r.fit_seed}|{r.generation_seed}", r.path, None) for r in reg.itertuples()]
    jobs += [("real_subset", f"real_train_subset_{_N_DEV}", str(REAL_TRAIN), args.seed + i) for i in range(args.real_draws)]
    jobs += [("real_self", "real_development_self", str(REAL_DEV), None)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init, initargs=(str(REAL_DEV), args.threads)) as ex:
        for out in ex.map(_one, jobs):
            rows.append(out)
            print(f"{out['model']:70s} draw={out['draw']} row_auprc={out.get('row_auprc', float('nan')):.4f} "
                  f"hist_auprc={out.get('history_auprc', float('nan')):.4f} early={out.get('history_early_detection_top1', float('nan')):.3f} "
                  f"{out['seconds']}s {out.get('error', '')}", flush=True)
    res = pd.DataFrame(rows)
    parts = res.model.str.split("|", expand=True)
    if parts.shape[1] == 4:
        res["family"] = parts[0].where(res.kind.eq("generation"), res.kind)
        res["model_name"] = parts[1].where(res.kind.eq("generation"), res.model)
        res["fit_seed"] = parts[2]; res["generation_seed"] = parts[3]
    res.to_csv(OUT / "UTILITY_ALL.csv", index=False)
    json.dump(dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seconds=round(time.time() - t0, 1),
                   dataset=args.dataset, detector=args.detector, feature_vocabulary=vocab, test=str(REAL_DEV), test_sha256=sha(REAL_DEV), generations=int(res.kind.eq("generation").sum()),
                   real_draws=args.real_draws, errors=int(res.get("error", pd.Series(dtype=object)).notna().sum()),
                   new_fits=0, new_generations=0, gpu=False, test_events_read=False),
              open(OUT / "MANIFEST.json", "w"), indent=1)
    print("done", len(res), "rows", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
