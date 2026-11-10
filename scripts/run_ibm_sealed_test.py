"""ibm_sealed_test_v1 (docs/ibm_sealed_test_v1/PROTOCOL.md): generate the frozen ARGN ladder for the sealed test customers.
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_sealed_test.py --fit 20261108
Every table, head and kernel is the frozen train-estimated artefact of the development stages; only the test CONTEXT
(four static customer variables) is read here. One-shot: outputs are never regenerated with other settings.
"""
import argparse, json, logging, os, sys, time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
for k, v in {"LOKY_MAX_CPU_COUNT": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
import numpy as np
import pandas as pd
import torch
from run_ibm_merchant_attr import NATIVE, TABLE, ROUTED_HEADS, MM_PARAMS, CHAN_TABLE, GAP_DIR, CORE, GEN_SEEDS, digest, write, seed, tables_path  # noqa: E402
from run_ibm_merchant_marginal_v2 import V1, DOCS as V2_DOCS, name_rare, type_rates, RARE  # noqa: E402

OUT = ROOT / "artifacts/ibm_sealed_test_v1"
ARMS = ["argn_native", "argn_minimal", "argn_merchant", "argn_gap", "argn_final"]


def generate(fs, arms):
    from mostlyai.engine import generate as engine_generate
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_generation_module import generation_module
    from benchmarks.argn_routed_heads import attach_routed_heads
    from benchmarks.argn_merchant_memory import merchant_module
    from benchmarks.argn_run_gap import gap_module
    from benchmarks.argn_merchant_attr import attr_module
    table = json.loads(TABLE.read_text()); mm = json.loads(MM_PARAMS.read_text()); gap_params = json.loads((GAP_DIR / f"gap_tables_{fs}.json").read_text())
    cal = json.loads((V1 / "calibration.json").read_text()); tables = dict(np.load(tables_path(fs))); state_table = json.loads(CHAN_TABLE.read_text())["table"]
    sel = json.loads((V2_DOCS / "SIM_SELECTED.json").read_text())["selected"]; kappa, lam = float(sel["kappa"]), float(sel["lam"]); assert (kappa, lam) == (8.0, 1.0), "frozen kernel"
    wsdir = NATIVE / "runs" / f"native_{fs}" / "workspace"; ws = Workspace(wsdir); stats = ws.tgt_stats.read(); weights = digest(ws.model_tabular_weights_path)
    codes = stats["columns"]["receiver_or_mark"]["codes"]; rho = type_rates(cal["categories"], set(map(str, codes.keys())))
    calib = dict(cat_of_m=cal["cat_of_m"], p_cat_novel=cal["p_cat_novel_pos"], rare_no_repeat=True, prior=(kappa * rho[np.asarray(cal["cat_of_m"])]).tolist(), lam=lam)
    ctx = pd.read_parquet(NATIVE / "prepared/test_context.parquet"); device = "cuda" if torch.cuda.is_available() else "cpu"
    head_path = ROUTED_HEADS / f"routed_subst_{fs}" / "heads.pt"; head_state = torch.load(head_path, map_location="cpu")
    for arm in arms:
        folder = OUT / "runs" / f"{arm}_{fs}"; folder.mkdir(parents=True, exist_ok=True)
        for gs in GEN_SEEDS:
            dest = folder / f"generated_test_{gs}.parquet"
            if dest.exists():
                continue
            seed(gs); t0 = time.monotonic(); infos = {}
            with ExitStack() as st:
                if arm != "argn_native":
                    infos["base"] = st.enter_context(generation_module(stats, table, label_first=True, head_state=head_state, attach=lambda m: attach_routed_heads(m, stats)))
                if arm in ("argn_merchant", "argn_gap"):
                    infos["merchant"] = st.enter_context(merchant_module(stats, mm, arm="full"))
                if arm == "argn_final":
                    infos["merchant"] = st.enter_context(merchant_module(stats, mm, arm="full", calibration=calib))
                if arm in ("argn_gap", "argn_final"):
                    infos["gap"] = st.enter_context(gap_module(stats, gap_params, "table_kind"))
                if arm == "argn_final":
                    infos["attr"] = st.enter_context(attr_module(stats, tables, state_table, "attr_all"))
                engine_generate(ctx_data=ctx, device=device, workspace_dir=wsdir)
            raw = pd.read_parquet(wsdir / "SyntheticData"); raw.to_parquet(folder / f"native_test_{gs}.parquet", index=False)
            d = raw.rename(columns={"customer_id": "entity_id"}).copy()
            d["event_index"] = d.groupby("entity_id", sort=False).cumcount(); d.loc[d.event_index.eq(0), "gap"] = np.nan
            d = d[["entity_id", "event_index", *CORE]]; named = 0
            if arm == "argn_final":
                d, named = name_rare(d)
            d.to_parquet(dest, index=False); assert digest(ws.model_tabular_weights_path) == weights
            lab = pd.to_numeric(d.event_is_fraud, errors="coerce").fillna(0).astype(int); novel = ~d.duplicated(["entity_id", "receiver_or_mark"])
            facts = dict(arm=arm, fit_seed=fs, generation_seed=gs, split="test", seconds=round(time.monotonic() - t0, 1), events=len(d), customers=int(d.entity_id.nunique()), frauds=int(lab.sum()),
                         fraud_customers=int(d[lab.eq(1)].entity_id.nunique()), novel_merchant_fraud=float(novel[lab.eq(1)].mean()) if lab.any() else None, novel_merchant_normal=float(novel[lab.eq(0)].mean()),
                         merchants_per_customer_median=float(d.groupby("entity_id").receiver_or_mark.nunique().median()), rare_rows_named=named, kernel=dict(kappa=kappa, lam=lam) if arm == "argn_final" else None,
                         base_weights_sha256=weights, heads_sha256=digest(head_path), modules=infos, sha256=digest(dest))
            write(folder / f"generation_test_{gs}.json", facts)
            print(f"GENERATED {arm} {fs} {gs}: events={len(d)} customers={facts['customers']} frauds={facts['frauds']} fraud_customers={facts['fraud_customers']} merchants={facts['merchants_per_customer_median']} {facts['seconds']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fit", type=int, required=True); ap.add_argument("--arms", nargs="*", default=ARMS); a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"); generate(a.fit, a.arms)
