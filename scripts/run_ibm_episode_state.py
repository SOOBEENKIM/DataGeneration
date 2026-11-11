"""ibm_episode_state_v1 stage 2 (docs/ibm_episode_state_v1/PROTOCOL.md, amendments 1-2): the adopted stack with the label process
replaced by the selected episode tables (A5, m from SIM_AMENDMENT1.json), generated for the 2013 dev context.
    ./run python scripts/run_ibm_episode_state.py fit                         # CPU: refit A5 on train with the beyond-window entry, check it equals the simulated tables
    ./run python scripts/run_ibm_episode_state.py predict                     # CPU: free-run prediction on the existing dev generations' time paths
    CUDA_VISIBLE_DEVICES=0 ./run python scripts/run_ibm_episode_state.py generate --fit 20261108
No training; backbone weights, heads and every other part are the frozen ones. Never reads a test file.
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
from run_ibm_merchant_attr import NATIVE, TABLE, ROUTED_HEADS, MM_PARAMS, CHAN_TABLE, GAP_DIR, CORE, GEN_SEEDS, digest, write, seed, tables_path, merchant_attributes  # noqa: E402
from run_ibm_merchant_marginal_v2 import V1, DOCS as V2_DOCS, OUT as V2_OUT, name_rare, type_rates  # noqa: E402
from benchmarks.argn_episode_state import fit_episode_tables, hazard, episode_module, N_MONTH  # noqa: E402

OUT = ROOT / "artifacts/ibm_episode_state_v1"; DOCS = ROOT / "docs/ibm_episode_state_v1"
TABLES = OUT / "episode_tables_A5_window.json"


def cmd_fit(_):
    m = json.loads((DOCS / "SIM_AMENDMENT1.json").read_text()); assert m["passed"], "the simulation gate did not pass"
    train = pd.read_parquet(NATIVE / "prepared/train.parquet"); tb = fit_episode_tables(train, "A5", shrink=m["selected_m"]); old = json.loads((OUT / "episode_tables_A5.json").read_text())
    assert all(np.allclose(tb[k], old[k]) for k in ("onset", "cont", "re")), "tables differ from the simulated ones"
    from benchmarks.argn_episode_state import row_states
    _, s = row_states(train); assert int((s["month"] == N_MONTH).sum()) == 0, "real rows beyond the window"
    TABLES.write_text(json.dumps(tb, indent=1) + "\n"); print("tables", TABLES, "m", tb["shrink"], "onset", np.round(tb["onset"], 6).tolist(), "beyond", tb["onset_beyond"], "sha256", digest(TABLES))


def cmd_predict(_):
    from sim_episode_state import paths, simulate
    tb = json.loads(TABLES.read_text()); old = dict(tb); old["onset_beyond"] = tb["onset"][-1]; rows = []
    for fs in (20261108, 20261109):
        for gs in GEN_SEEDS:
            g = pd.read_parquet(V2_OUT / "runs" / f"visitrate_{fs}" / f"generated_validation_{gs}.parquet"); d, s, L, M = paths(g); r = dict(fit=fs, seed=gs, events=int(len(g)), current_frauds=int(s["y"].sum()), beyond_window_row_share=float((s["month"] == N_MONTH).mean()))
            for name, t in (("registered_last_bin", old), ("amendment2_year_average", tb)):
                sim = simulate(lambda *a, t=t: hazard(t, *a), L, M, 100, 50); r[name] = dict(frauds=sim["frauds"], rate=sim["rate"], fraud_customers=sim["fraud_customers"], relpos=sim["first_fraud_relpos_median"], max=sim["frauds_per_fraud_customer_max"])
            rows.append(r); print(json.dumps(r), flush=True)
    write(DOCS / "PREDICTION.json", dict(train_rate=0.0011597510318045923, rows=rows, note="CPU free-run on the adopted model's existing dev generations (generated lengths and gaps); no dev labels used", test_read=False))


def cmd_generate(a):
    from mostlyai.engine import generate as engine_generate
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_generation_module import generation_module
    from benchmarks.argn_routed_heads import attach_routed_heads
    from benchmarks.argn_merchant_memory import merchant_module
    from benchmarks.argn_run_gap import gap_module
    from benchmarks.argn_merchant_attr import attr_module
    fs = a.fit; table = json.loads(TABLE.read_text()); mm = json.loads(MM_PARAMS.read_text()); gap_params = json.loads((GAP_DIR / f"gap_tables_{fs}.json").read_text()); ep = json.loads(TABLES.read_text())
    cal = json.loads((V1 / "calibration.json").read_text()); tables = dict(np.load(tables_path(fs))); state_table = json.loads(CHAN_TABLE.read_text())["table"]
    sel = json.loads((V2_DOCS / "SIM_SELECTED.json").read_text())["selected"]; kappa, lam = float(sel["kappa"]), float(sel["lam"]); assert (kappa, lam) == (8.0, 1.0), "frozen kernel"
    wsdir = NATIVE / "runs" / f"native_{fs}" / "workspace"; ws = Workspace(wsdir); stats = ws.tgt_stats.read(); weights = digest(ws.model_tabular_weights_path)
    codes = stats["columns"]["receiver_or_mark"]["codes"]; rho = type_rates(cal["categories"], set(map(str, codes.keys())))
    calib = dict(cat_of_m=cal["cat_of_m"], p_cat_novel=cal["p_cat_novel_pos"], rare_no_repeat=True, prior=(kappa * rho[np.asarray(cal["cat_of_m"])]).tolist(), lam=lam)
    ctx = pd.read_parquet(NATIVE / "prepared/validation_context.parquet"); device = "cuda" if torch.cuda.is_available() else "cpu"
    head_path = ROUTED_HEADS / f"routed_subst_{fs}" / "heads.pt"; head_state = torch.load(head_path, map_location="cpu")
    folder = OUT / "runs" / f"episode_{fs}"; folder.mkdir(parents=True, exist_ok=True)
    for gs in GEN_SEEDS:
        dest = folder / f"generated_validation_{gs}.parquet"
        if dest.exists():
            continue
        seed(gs); t0 = time.monotonic(); infos = {}
        with ExitStack() as st:
            infos["base"] = st.enter_context(generation_module(stats, table, label_first=True, head_state=head_state, attach=lambda m: attach_routed_heads(m, stats)))
            infos["episode"] = st.enter_context(episode_module(stats, ep))
            infos["merchant"] = st.enter_context(merchant_module(stats, mm, arm="full", calibration=calib))
            infos["gap"] = st.enter_context(gap_module(stats, gap_params, "table_kind"))
            infos["attr"] = st.enter_context(attr_module(stats, tables, state_table, "attr_all"))
            engine_generate(ctx_data=ctx, device=device, workspace_dir=wsdir)
        raw = pd.read_parquet(wsdir / "SyntheticData"); raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
        d = raw.rename(columns={"customer_id": "entity_id"}).copy(); d["event_index"] = d.groupby("entity_id", sort=False).cumcount(); d.loc[d.event_index.eq(0), "gap"] = np.nan
        d = d[["entity_id", "event_index", *CORE]]; d, named = name_rare(d); d.to_parquet(dest, index=False); assert digest(ws.model_tabular_weights_path) == weights
        lab = pd.to_numeric(d.event_is_fraud, errors="coerce").fillna(0).astype(int); per = lab.groupby(d.entity_id).sum(); pos = d.event_index / d.groupby("entity_id").event_index.transform("size")
        days = pd.to_numeric(d.gap, errors="coerce").fillna(0).groupby(d.entity_id).sum() / 86400
        facts = dict(arm="episode", fit_seed=fs, generation_seed=gs, seconds=round(time.monotonic() - t0, 1), events=int(len(d)), customers=int(d.entity_id.nunique()), frauds=int(lab.sum()), fraud_rate=float(lab.mean()),
                     fraud_customers=int((per > 0).sum()), frauds_per_fraud_customer_max=int(per.max()), frauds_per_fraud_customer_quantiles={str(q): float(per[per > 0].quantile(q)) for q in (.25, .5, .75, .9)} if (per > 0).any() else None,
                     fraud_relative_position_mean=float(pos[lab.eq(1)].mean()) if lab.any() else None, total_days_quantiles={str(q): float(days.quantile(q)) for q in (.1, .5, .9)}, rare_rows_named=named,
                     episode_tables_sha256=digest(TABLES), base_weights_sha256=weights, heads_sha256=digest(head_path), modules=infos, sha256=digest(dest))
        write(folder / f"generation_validation_{gs}.json", facts)
        print(f"GENERATED episode {fs} {gs}: events={facts['events']} frauds={facts['frauds']} rate={facts['fraud_rate']:.5f} fraud_customers={facts['fraud_customers']} max={facts['frauds_per_fraud_customer_max']} "
              f"relpos={facts['fraud_relative_position_mean']} expected={infos['episode']['counter']['expected_frauds']:.1f} beyond={infos['episode']['counter']['beyond_window_rows']} {facts['seconds']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["fit", "predict", "generate"]); ap.add_argument("--fit", type=int, default=20261108); a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"); {"fit": cmd_fit, "predict": cmd_predict, "generate": cmd_generate}[a.command](a)
