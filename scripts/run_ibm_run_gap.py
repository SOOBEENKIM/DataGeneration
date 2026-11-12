"""ibm_run_gap_v1: gap generation indexed by fraud-run position on the current ARGN model (native + transition table,
label first + routed_subst heads + merchant memory `full`). docs/ibm_run_gap_v1/PROTOCOL.md.
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_run_gap.py fit --fit 20261108
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_run_gap.py generate --fit 20261108
Whole-model fits: 0. Tables use train only (the fit's encoded train sequences); the sealed test is never read.
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
MM_PARAMS = ROOT / "artifacts/ibm_merchant_memory_v1/merchant_memory.json"
OUT = ROOT / "artifacts/ibm_run_gap_v1"
DOCS = ROOT / "docs/ibm_run_gap_v1"
CORE = ["gap", "receiver_or_mark", "amount_or_numeric_value", "category", "use_chip", "event_is_fraud"]
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


def params_path(fs):
    return OUT / f"gap_tables_{fs}.json"


def kind_facts(d):
    """Gap (minutes) by row kind in a generated / real frame."""
    d = d.sort_values(["entity_id", "event_index"], kind="stable")
    y = pd.to_numeric(d.event_is_fraud, errors="coerce").fillna(0).astype(int); py = y.groupby(d.entity_id).shift()
    g = pd.to_numeric(d.gap, errors="coerce").astype(float) / 60.0
    out = {}
    for name, sel in (("onset", y.eq(1) & py.eq(0)), ("cont", y.eq(1) & py.eq(1)), ("exit", y.eq(0) & py.eq(1)), ("normal", y.eq(0) & py.eq(0))):
        s = g[sel].dropna().to_numpy()
        out[name] = dict(n=int(len(s)), median=float(np.median(s)) if len(s) else None, le60=float((s <= 60).mean()) if len(s) else None)
    return out


def fit(fs):
    from mostlyai.engine._workspace import Workspace
    from mostlyai.engine._common import get_cardinalities
    from benchmarks.argn_output_heads import argn_column, _records
    from benchmarks.argn_run_gap import fit_gap_tables, LABEL, GAP
    ws = Workspace(NATIVE / "runs" / f"native_{fs}" / "workspace"); stats = ws.tgt_stats.read(); card = get_cardinalities(stats)
    label_key = argn_column(stats, LABEL) + "__cat"; gap_key = argn_column(stats, GAP) + "__bin"; fraud_code = int(stats["columns"][LABEL]["codes"]["1"])
    t0 = time.time()
    recs = _records(list(ws.encoded_data_trn.fetch_all())[0]) + _records(list(ws.encoded_data_val.fetch_all())[0])
    params = fit_gap_tables(recs, label_key, gap_key, fraud_code, card[gap_key])
    params.update(seconds=round(time.time() - t0, 1), fit=fs, gap_key=gap_key, bins=stats["columns"][GAP].get("bins"), codes=stats["columns"][GAP].get("codes"))
    write(params_path(fs), params)
    # readable report: cumulative share of each table below a few gap thresholds is not needed here; store rows and modal mass
    rep = dict(fit=fs, rows=params["rows"], customers=params["customers"], prior_rows=params["prior_rows"], n_codes=params["n_codes"],
               top5={k: sorted(((round(p, 4), i) for i, p in enumerate(params["tables"][k])), reverse=True)[:5] for k in params["tables"]},
               train_real_by_kind=kind_facts(pd.read_parquet(NATIVE / "prepared/train.parquet")))
    write(DOCS / f"GAP_TABLES_{fs}.json", rep); print(json.dumps(rep, indent=1))


def generate(fs, arms):
    from mostlyai.engine import generate as engine_generate
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_generation_module import generation_module
    from benchmarks.argn_routed_heads import attach_routed_heads
    from benchmarks.argn_merchant_memory import merchant_module
    from benchmarks.argn_run_gap import gap_module
    table = json.loads(TABLE.read_text()); mm = json.loads(MM_PARAMS.read_text()); params = json.loads(params_path(fs).read_text())
    wsdir = NATIVE / "runs" / f"native_{fs}" / "workspace"; ws = Workspace(wsdir); stats = ws.tgt_stats.read(); weights = digest(ws.model_tabular_weights_path)
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
                with merchant_module(stats, mm, arm="full") as mm_info:
                    with gap_module(stats, params, arm) as info:
                        engine_generate(ctx_data=ctx, device=device, workspace_dir=wsdir)
            raw = pd.read_parquet(wsdir / "SyntheticData"); raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
            d = raw.rename(columns={"customer_id": "entity_id"}).copy()
            d["event_index"] = d.groupby("entity_id", sort=False).cumcount(); d.loc[d.event_index.eq(0), "gap"] = np.nan
            d = d[["entity_id", "event_index", *CORE]]; d.to_parquet(dest, index=False)
            assert digest(ws.model_tabular_weights_path) == weights
            lab = pd.to_numeric(d.event_is_fraud, errors="coerce")
            facts = dict(arm=arm, fit_seed=fs, generation_seed=gs, seconds=round(time.monotonic() - t0, 1), events=len(d), customers=int(d.entity_id.nunique()),
                         frauds=int(lab.eq(1).sum()), fraud_customers=int(d[lab.eq(1)].entity_id.nunique()), gap_minutes_by_kind=kind_facts(d),
                         base_weights_sha256=weights, heads_sha256=digest(head_path), params_sha256=digest(params_path(fs)), merchant_params_sha256=digest(MM_PARAMS),
                         base_module=base_info, merchant_module=mm_info, module=info, sha256=digest(dest))
            write(folder / f"generation_validation_{gs}.json", facts)
            k = facts["gap_minutes_by_kind"]
            print(f"GENERATED {arm} {fs} {gs}: events={len(d)} frauds={facts['frauds']} onset={k['onset']['median']} cont={k['cont']['median']} exit={k['exit']['median']} "
                  f"normal={k['normal']['median']} interventions={info['counter']} {facts['seconds']}s", flush=True)


if __name__ == "__main__":
    from benchmarks.argn_run_gap import ARMS
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["fit", "generate"]); ap.add_argument("--fit", type=int, default=20261108)
    ap.add_argument("--arms", nargs="*", default=list(ARMS)); a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"fit": lambda: fit(a.fit), "generate": lambda: generate(a.fit, a.arms)}[a.command]()
