"""ibm_merchant_memory_v1: history-referencing merchant generation on the frozen IBM native ARGN + current model
(transition table, label first, routed_subst heads). docs/ibm_merchant_memory_v1/PROTOCOL.md.
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_merchant_memory.py fit
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_merchant_memory.py generate --fit 20261108 --arms repeat_blind gate_only full
Whole-model fits: 0. Estimation uses train only; test is never read.
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

NATIVE = ROOT / "artifacts/ibm_tabformer_v1"
TABLE = ROOT / "artifacts/ibm_generation_module_v1/transition_table.json"
ROUTED_HEADS = ROOT / "artifacts/ibm_routed_heads_v1/heads"
OUT = ROOT / "artifacts/ibm_merchant_memory_v1"
DOCS = ROOT / "docs/ibm_merchant_memory_v1"
PARAMS = OUT / "merchant_memory.json"
CORE = ["gap", "receiver_or_mark", "amount_or_numeric_value", "category", "use_chip", "event_is_fraud"]
ARMS = ["repeat_blind", "gate_only", "full"]
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


def fit(fs):
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_merchant_memory import fit_merchant_memory
    stats = Workspace(NATIVE / "runs" / f"native_{fs}" / "workspace").tgt_stats.read()
    train = pd.read_parquet(NATIVE / "prepared/train.parquet"); t0 = time.time()
    params = fit_merchant_memory(train, stats); params.update(seconds=round(time.time() - t0, 1), stats_fit=fs, train_sha256=digest(NATIVE / "prepared/train.parquet"))
    write(PARAMS, params)
    report = {k: v for k, v in params.items() if k != "pool"}; report["pool_top20"] = sorted(((float(p), i) for i, p in enumerate(params["pool"])), reverse=True)[:20]
    write(DOCS / "MERCHANT_MEMORY.json", report)
    print(json.dumps({k: report[k] for k in ["gate", "gate_blind", "lam", "kernel_holdout_logloss", "pi", "pool_merchants", "train_novel_share", "seconds"]}, indent=1))


def generate(fs, arms):
    from mostlyai.engine import generate as engine_generate
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_generation_module import generation_module
    from benchmarks.argn_routed_heads import attach_routed_heads
    from benchmarks.argn_merchant_memory import merchant_module
    table = json.loads(TABLE.read_text()); params = json.loads(PARAMS.read_text())
    wsdir = NATIVE / "runs" / f"native_{fs}" / "workspace"; ws = Workspace(wsdir); stats = ws.tgt_stats.read(); weights = digest(ws.model_tabular_weights_path)
    assert params["n_codes"] == len(stats["columns"]["receiver_or_mark"]["codes"]), "merchant codes differ between fits"
    ctx = pd.read_parquet(NATIVE / "prepared/validation_context.parquet"); device = "cuda" if torch.cuda.is_available() else "cpu"
    head_path = ROUTED_HEADS / f"routed_subst_{fs}" / "heads.pt"; head_state = torch.load(head_path, map_location="cpu")
    for arm in arms:
        folder = OUT / "runs" / f"{arm}_{fs}"; folder.mkdir(parents=True, exist_ok=True)
        for gs in GEN_SEEDS:
            dest = folder / f"generated_validation_{gs}.parquet"
            if dest.exists():
                continue
            seed(gs); t0 = time.monotonic()
            with generation_module(stats, table, label_first=True, head_state=head_state, attach=lambda m: attach_routed_heads(m, stats)) as base_info:
                with merchant_module(stats, params, arm=arm) as info:
                    engine_generate(ctx_data=ctx, device=device, workspace_dir=wsdir)
            raw = pd.read_parquet(wsdir / "SyntheticData"); raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
            d = raw.rename(columns={"customer_id": "entity_id"}).copy()
            d["event_index"] = d.groupby("entity_id", sort=False).cumcount(); d.loc[d.event_index.eq(0), "gap"] = np.nan
            d = d[["entity_id", "event_index", *CORE]]; d.to_parquet(dest, index=False)
            assert digest(ws.model_tabular_weights_path) == weights
            lab = pd.to_numeric(d.event_is_fraud, errors="coerce"); f = d[lab.eq(1)]; nrm = d[lab.eq(0)]
            novel = ~d.duplicated(["entity_id", "receiver_or_mark"])
            facts = dict(arm=arm, fit_seed=fs, generation_seed=gs, seconds=round(time.monotonic() - t0, 1), events=len(d), customers=int(d.entity_id.nunique()),
                         frauds=int(len(f)), fraud_customers=int(f.entity_id.nunique()), novel_merchant_fraud=float(novel[lab.eq(1)].mean()) if len(f) else None,
                         novel_merchant_normal=float(novel[lab.eq(0)].mean()), merchants_per_customer_median=float(d.groupby("entity_id").receiver_or_mark.nunique().median()),
                         rare_share=float(d.receiver_or_mark.astype(str).eq("_RARE_").mean()), base_weights_sha256=weights, heads_sha256=digest(head_path),
                         params_sha256=digest(PARAMS), base_module=base_info, module=info, sha256=digest(dest))
            write(folder / f"generation_validation_{gs}.json", facts)
            print(f"GENERATED {arm} {fs} {gs}: events={len(d)} frauds={len(f)} novel fraud={facts['novel_merchant_fraud']} normal={facts['novel_merchant_normal']:.3f} "
                  f"merchants/customer={facts['merchants_per_customer_median']} {facts['seconds']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["fit", "generate"]); ap.add_argument("--fit", type=int, default=20261108)
    ap.add_argument("--arms", nargs="*", default=ARMS); a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"fit": lambda: fit(a.fit), "generate": lambda: generate(a.fit, a.arms)}[a.command]()
