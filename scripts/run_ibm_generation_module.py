"""ibm_generation_module_v1: generation-time rare-state module on the frozen IBM native ARGN fits.
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_generation_module.py table
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_generation_module.py generate --fit 20261108
No new fit. The sealed test split is never read.
"""
import argparse, hashlib, json, logging, os, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for k, v in {"LOKY_MAX_CPU_COUNT": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
import numpy as np
import pandas as pd
import torch
from benchmarks.argn_generation_module import fit_table, generation_module  # noqa: E402

NATIVE = ROOT / "artifacts/ibm_tabformer_v1"
OUT = ROOT / "artifacts/ibm_generation_module_v1"
DOCS = ROOT / "docs/ibm_generation_module_v1"
CORE = ["gap", "receiver_or_mark", "amount_or_numeric_value", "category", "use_chip", "event_is_fraud"]
ARMS = {"order_only": dict(label_first=True, transition=False),
        "transition_only": dict(label_first=False, transition=True),
        "transition_label_first": dict(label_first=True, transition=True)}
GEN_SEEDS = [20261011, 20261012]


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")


def seed(n):
    random.seed(n); np.random.seed(n); torch.manual_seed(n); torch.cuda.manual_seed_all(n); torch.set_num_threads(4)


def table():
    train = pd.read_parquet(NATIVE / "prepared/train.parquet")
    params = fit_table(train)
    OUT.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True)
    write(OUT / "transition_table.json", params)
    pd.DataFrame(params["counts"]).to_csv(DOCS / "TRANSITION_TABLE_COUNTS.csv", index=False)
    write(DOCS / "transition_table.json", dict(table=params["table"], first=params["first"], smoothing=params["smoothing"], max_age=params["max_age"],
                                               train_customers=params["train_customers"], train_events=params["train_events"], train_frauds=params["train_frauds"],
                                               train_sha256=digest(NATIVE / "prepared/train.parquet"), development_or_test_read=False))
    t = np.asarray(params["table"])
    print("first", round(params["first"], 6)); print("P(fraud | prev normal, age 1..5, ever=0/1)", t[1, 1:6, 0].round(5), t[1, 1:6, 1].round(5))
    print("P(fraud | prev fraud, age 1..5, ever=1)", t[2, 1:6, 1].round(4))


def generate(fs, arms):
    from mostlyai.engine import generate as engine_generate
    from mostlyai.engine._workspace import Workspace
    params = json.loads((OUT / "transition_table.json").read_text())
    wsdir = NATIVE / "runs" / f"native_{fs}" / "workspace"; ws = Workspace(wsdir); stats = ws.tgt_stats.read()
    weights = digest(ws.model_tabular_weights_path)
    ctx = pd.read_parquet(NATIVE / "prepared/validation_context.parquet")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for arm in arms:
        cfg = ARMS[arm]; folder = OUT / "runs" / f"{arm}_{fs}"; folder.mkdir(parents=True, exist_ok=True)
        for gs in GEN_SEEDS:
            dest = folder / f"generated_validation_{gs}.parquet"
            if dest.exists():
                continue
            seed(gs); t0 = time.monotonic()
            with generation_module(stats, params if cfg["transition"] else None, label_first=cfg["label_first"]) as info:
                engine_generate(ctx_data=ctx, device=device, workspace_dir=wsdir)
            raw = pd.read_parquet(wsdir / "SyntheticData"); raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
            d = raw.rename(columns={"customer_id": "entity_id"}).copy()
            d["event_index"] = d.groupby("entity_id", sort=False).cumcount(); d.loc[d.event_index.eq(0), "gap"] = np.nan
            d = d[["entity_id", "event_index", *CORE]]; d.to_parquet(dest, index=False)
            assert digest(ws.model_tabular_weights_path) == weights
            lab = pd.to_numeric(d.event_is_fraud, errors="coerce")
            prev = lab.groupby(d.entity_id).shift()
            cont = float(lab[prev.eq(1)].eq(1).mean()) if prev.eq(1).any() else float("nan")
            write(folder / f"generation_validation_{gs}.json", dict(arm=arm, fit_seed=fs, generation_seed=gs, seconds=round(time.monotonic() - t0, 1), events=len(d),
                                                                   customers=int(d.entity_id.nunique()), frauds=int(lab.eq(1).sum()), fraud_customers=int(d.entity_id[lab.eq(1)].nunique()),
                                                                   continuation=cont, weights_sha256=weights, weights_unchanged=True, module=info, sha256=digest(dest),
                                                                   table_sha256=digest(OUT / "transition_table.json")))
            print(f"GENERATED {arm} {fs} {gs}: events={len(d)} frauds={int(lab.eq(1).sum())} fraud_customers={int(d.entity_id[lab.eq(1)].nunique())} cont={cont:.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["table", "generate"]); ap.add_argument("--fit", type=int, default=20261108)
    ap.add_argument("--arms", nargs="*", default=list(ARMS)); a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.command == "table":
        table()
    else:
        generate(a.fit, a.arms)
