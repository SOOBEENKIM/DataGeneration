"""ibm_merchant_marginal_v1: category-level stickiness and category-first novel choice in the merchant-memory component,
on the current ARGN stack (transition table + routed_subst heads + merchant memory `full` + gap `table_kind` + `attr_all`).
docs/ibm_merchant_marginal_v1/PROTOCOL.md.
    ./run python scripts/run_ibm_merchant_marginal.py fit
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_merchant_marginal.py generate --fit 20261108
Whole-model fits: 0. Estimation uses train only; the sealed test is never read.
"""
import argparse, json, logging, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
for k, v in {"LOKY_MAX_CPU_COUNT": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
import numpy as np
import pandas as pd
import torch
from run_ibm_merchant_attr import NATIVE, TABLE, ROUTED_HEADS, MM_PARAMS, CHAN_TABLE, GAP_DIR, CORE, GEN_SEEDS, digest, write, seed, merchant_attributes, consistency, tables_path  # noqa: E402

OUT = ROOT / "artifacts/ibm_merchant_marginal_v1"
DOCS = ROOT / "docs/ibm_merchant_marginal_v1"
PARAMS = OUT / "calibration.json"
ARMS = ["novelcat", "novelcat_pos"]  # amendment 1 (2026-11-10): sticky not adopted (hold-out gain .011 < .05); position-binned category table added


def fit(fs):
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_merchant_marginal import fit_calibration
    stats = Workspace(NATIVE / "runs" / f"native_{fs}" / "workspace").tgt_stats.read(); t0 = time.time()
    p = fit_calibration(pd.read_parquet(NATIVE / "prepared/train.parquet"), stats); p.update(seconds=round(time.time() - t0, 1), stats_fit=fs, train_sha256=digest(NATIVE / "prepared/train.parquet"))
    p["adopt_sticky"] = bool(p["holdout_gain"] >= 0.05)
    write(PARAMS, p)
    rep = {k: v for k, v in p.items() if k not in ("cat_of_m", "p_cat_novel_pos")}; rep["w_cat"] = dict(zip(p["categories"], [round(x, 4) for x in p["w_cat"]]))
    rep["p_cat_novel"] = dict(zip(p["categories"], [round(x, 4) for x in p["p_cat_novel"]])); rep.pop("categories")
    hi = [p["categories"].index(c) for c in ("5499", "5541", "4829", "4784")]
    rep["high_frequency_type_share_by_position_bin"] = [round(float(sum(row[i] for i in hi)), 4) for row in p["p_cat_novel_pos"]]
    write(DOCS / "CALIBRATION.json", rep)
    print(json.dumps({k: rep[k] for k in ("holdout_ll_base", "holdout_ll_sticky", "lam_base", "lam_sticky", "holdout_gain", "adopt_sticky", "holdout_repeat_rows", "fit_rows", "fit_nll_before",
                                          "fit_nll_after", "top_sticky", "least_sticky", "rare_first_visit_share", "seconds")}, indent=1))


def generate(fs, arms):
    from mostlyai.engine import generate as engine_generate
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_generation_module import generation_module
    from benchmarks.argn_routed_heads import attach_routed_heads
    from benchmarks.argn_merchant_memory import merchant_module
    from benchmarks.argn_run_gap import gap_module
    from benchmarks.argn_merchant_attr import attr_module
    table = json.loads(TABLE.read_text()); mm = json.loads(MM_PARAMS.read_text()); gap_params = json.loads((GAP_DIR / f"gap_tables_{fs}.json").read_text())
    cal = json.loads(PARAMS.read_text()); tables = dict(np.load(tables_path(fs))); state_table = json.loads(CHAN_TABLE.read_text())["table"]; att = merchant_attributes()
    wsdir = NATIVE / "runs" / f"native_{fs}" / "workspace"; ws = Workspace(wsdir); stats = ws.tgt_stats.read(); weights = digest(ws.model_tabular_weights_path)
    assert cal["n_codes"] == len(stats["columns"]["receiver_or_mark"]["codes"]) and cal["rare_code"] == int(stats["columns"]["receiver_or_mark"]["codes"]["_RARE_"])
    stick = (np.asarray(cal["w_cat"])[np.asarray(cal["cat_of_m"])]).tolist()
    calib = dict(sticky=dict(stick=stick, lam=cal["lam_sticky"]), novelcat=dict(cat_of_m=cal["cat_of_m"], p_cat_novel=cal["p_cat_novel"]),
                 novelcat_pos=dict(cat_of_m=cal["cat_of_m"], p_cat_novel=cal["p_cat_novel_pos"]),
                 both=dict(stick=stick, lam=cal["lam_sticky"], cat_of_m=cal["cat_of_m"], p_cat_novel=cal["p_cat_novel"]))
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
                with merchant_module(stats, mm, arm="full", calibration=calib[arm]) as mm_info:
                    with gap_module(stats, gap_params, "table_kind") as gap_info:
                        with attr_module(stats, tables, state_table, "attr_all") as attr_info:
                            engine_generate(ctx_data=ctx, device=device, workspace_dir=wsdir)
            raw = pd.read_parquet(wsdir / "SyntheticData"); raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
            d = raw.rename(columns={"customer_id": "entity_id"}).copy()
            d["event_index"] = d.groupby("entity_id", sort=False).cumcount(); d.loc[d.event_index.eq(0), "gap"] = np.nan
            d = d[["entity_id", "event_index", *CORE]]; d.to_parquet(dest, index=False)
            assert digest(ws.model_tabular_weights_path) == weights
            lab = pd.to_numeric(d.event_is_fraud, errors="coerce"); novel = ~d.duplicated(["entity_id", "receiver_or_mark"]); cons = consistency(d, att)
            per = d.groupby(["entity_id", "receiver_or_mark"]).size(); tot = per.groupby(level=0).sum()
            facts = dict(arm=arm, fit_seed=fs, generation_seed=gs, seconds=round(time.monotonic() - t0, 1), events=len(d), frauds=int(lab.eq(1).sum()), fraud_customers=int(d[lab.eq(1)].entity_id.nunique()),
                         novel_merchant_fraud=float(novel[lab.eq(1)].mean()), novel_merchant_normal=float(novel[lab.eq(0)].mean()),
                         merchants_per_customer_median=float(d.groupby("entity_id").receiver_or_mark.nunique().median()), top1_share_median=float((per.groupby(level=0).max() / tot).median()),
                         rare_share=float(d.receiver_or_mark.astype(str).eq("_RARE_").mean()), consistency=cons, base_weights_sha256=weights, calibration_sha256=digest(PARAMS),
                         base_module=base_info, merchant_module=mm_info, gap_module=gap_info, attr_module=attr_info, sha256=digest(dest))
            write(folder / f"generation_validation_{gs}.json", facts)
            print(f"GENERATED {arm} {fs} {gs}: events={len(d)} frauds={facts['frauds']} novel fraud={facts['novel_merchant_fraud']:.3f} normal={facts['novel_merchant_normal']:.3f} "
                  f"merchants/customer={facts['merchants_per_customer_median']} top1={facts['top1_share_median']:.3f} rare={facts['rare_share']:.3f} normal cat-match={cons['normal']['category_matches_merchant']:.3f} "
                  f"{facts['seconds']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["fit", "generate"]); ap.add_argument("--fit", type=int, default=20261108)
    ap.add_argument("--arms", nargs="*", default=ARMS); a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"fit": lambda: fit(a.fit), "generate": lambda: generate(a.fit, a.arms)}[a.command]()
