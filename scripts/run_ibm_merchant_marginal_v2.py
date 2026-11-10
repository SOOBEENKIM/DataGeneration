"""ibm_merchant_marginal_v2 (docs/ibm_merchant_marginal_v2/PROTOCOL.md): the simulation-selected type-rate prior kernel on the
attr_all stack, plus the rare-merchant naming rule.
    ./run python scripts/run_ibm_merchant_marginal_v2.py derive              (naming rule on the v1 novelcat_pos_rarefix outputs; CPU)
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_merchant_marginal_v2.py generate --fit 20261108
Whole-model fits: 0. Train only for every table; the sealed test is never read.
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

V1 = ROOT / "artifacts/ibm_merchant_marginal_v1"
OUT = ROOT / "artifacts/ibm_merchant_marginal_v2"
DOCS = ROOT / "docs/ibm_merchant_marginal_v2"
RARE = "_RARE_"


def name_rare(d):
    """Naming rule: every rare-bucket row denotes a distinct unseen merchant -> unique placeholder id."""
    d = d.copy(); m = d.receiver_or_mark.astype(str); r = m.eq(RARE).to_numpy()
    m = m.to_numpy(dtype=object); m[r] = [f"{RARE}{i}" for i in range(int(r.sum()))]; d["receiver_or_mark"] = pd.Series(m, index=d.index).astype(str)
    return d, int(r.sum())


def type_rates(cats, codes):
    """Relative visit rate by merchant type, exactly as in scripts/sim_merchant_process.py (train normal rows, known merchants)."""
    tr = pd.read_parquet(NATIVE / "prepared/train.parquet"); tr["m"] = tr.receiver_or_mark.astype(str); tr["c"] = tr.category.astype(str)
    trn = tr[pd.to_numeric(tr.event_is_fraud, errors="coerce").fillna(0).astype(int).eq(0)]; modal = trn.groupby("m").c.agg(lambda s: s.value_counts().index[0])
    per = trn[trn.m.isin(codes)].groupby(["entity_id", "m"]).size().rename("v").reset_index(); per["c"] = per.m.map(modal)
    rho = per.groupby("c").v.mean() / per.v.mean(); vec = np.array([float(rho.get(c, 1.0)) for c in cats]); vec[len(cats) - 1] = 1.0
    return vec


def facts_of(d, att):
    lab = pd.to_numeric(d.event_is_fraud, errors="coerce"); novel = ~d.duplicated(["entity_id", "receiver_or_mark"]); per = d.groupby(["entity_id", "receiver_or_mark"]).size(); tot = per.groupby(level=0).sum()
    return dict(events=len(d), frauds=int(lab.eq(1).sum()), fraud_customers=int(d[lab.eq(1)].entity_id.nunique()), novel_merchant_fraud=float(novel[lab.eq(1)].mean()), novel_merchant_normal=float(novel[lab.eq(0)].mean()),
                merchants_per_customer_median=float(d.groupby("entity_id").receiver_or_mark.nunique().median()), top1_share_median=float((per.groupby(level=0).max() / tot).median()),
                rare_share=float(d.receiver_or_mark.astype(str).str.startswith(RARE).mean()), consistency=consistency(d, att))


def derive():
    att = merchant_attributes()
    for fs in (20261108, 20261109):
        folder = OUT / "runs" / f"rarefix_named_{fs}"; folder.mkdir(parents=True, exist_ok=True)
        for gs in GEN_SEEDS:
            src = V1 / "runs" / f"novelcat_pos_rarefix_{fs}" / f"generated_validation_{gs}.parquet"; d, n = name_rare(pd.read_parquet(src)); dest = folder / f"generated_validation_{gs}.parquet"
            d.to_parquet(dest, index=False); f = facts_of(d, att); f.update(arm="rarefix_named", fit_seed=fs, generation_seed=gs, derived_from=str(src), source_sha256=digest(src), rare_rows_named=n, sha256=digest(dest))
            write(folder / f"generation_validation_{gs}.json", f)
            print(f"DERIVED rarefix_named {fs} {gs}: rare rows named={n} novel normal={f['novel_merchant_normal']:.3f} merchants/customer={f['merchants_per_customer_median']}", flush=True)


def generate(fs):
    from mostlyai.engine import generate as engine_generate
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_generation_module import generation_module
    from benchmarks.argn_routed_heads import attach_routed_heads
    from benchmarks.argn_merchant_memory import merchant_module
    from benchmarks.argn_run_gap import gap_module
    from benchmarks.argn_merchant_attr import attr_module
    sel = json.loads((DOCS / "SIM_SELECTED.json").read_text())["selected"]; assert sel and sel["gate_passed"], "the simulation gate did not pass"
    table = json.loads(TABLE.read_text()); mm = json.loads(MM_PARAMS.read_text()); gap_params = json.loads((GAP_DIR / f"gap_tables_{fs}.json").read_text())
    cal = json.loads((V1 / "calibration.json").read_text()); tables = dict(np.load(tables_path(fs))); state_table = json.loads(CHAN_TABLE.read_text())["table"]; att = merchant_attributes()
    wsdir = NATIVE / "runs" / f"native_{fs}" / "workspace"; ws = Workspace(wsdir); stats = ws.tgt_stats.read(); weights = digest(ws.model_tabular_weights_path)
    codes = stats["columns"]["receiver_or_mark"]["codes"]; assert cal["n_codes"] == len(codes) and cal["rare_code"] == int(codes[RARE])
    rho = type_rates(cal["categories"], set(map(str, codes.keys()))); kappa, lam = float(sel.get("kappa", 0.0)), float(sel["lam"]); assert kappa > 0 and not sel.get("beta")
    calib = dict(cat_of_m=cal["cat_of_m"], p_cat_novel=cal["p_cat_novel_pos"], rare_no_repeat=True, prior=(kappa * rho[np.asarray(cal["cat_of_m"])]).tolist(), lam=lam)
    write(DOCS / "KERNEL.json", dict(selected=sel["config"], kappa=kappa, lam=lam, rho=dict(zip(cal["categories"], [round(float(x), 4) for x in rho])), note="repeat weight = (count + kappa * rho(type)); rare bucket excluded from repeats"))
    ctx = pd.read_parquet(NATIVE / "prepared/validation_context.parquet"); device = "cuda" if torch.cuda.is_available() else "cpu"
    head_path = ROUTED_HEADS / f"routed_subst_{fs}" / "heads.pt"; head_state = torch.load(head_path, map_location="cpu")
    folder = OUT / "runs" / f"visitrate_{fs}"; folder.mkdir(parents=True, exist_ok=True)
    for gs in GEN_SEEDS:
        dest = folder / f"generated_validation_{gs}.parquet"
        if dest.exists():
            continue
        seed(gs); t0 = time.monotonic()
        with generation_module(stats, table, label_first=True, head_state=head_state, attach=lambda m: attach_routed_heads(m, stats)) as base_info:
            with merchant_module(stats, mm, arm="full", calibration=calib) as mm_info:
                with gap_module(stats, gap_params, "table_kind") as gap_info:
                    with attr_module(stats, tables, state_table, "attr_all") as attr_info:
                        engine_generate(ctx_data=ctx, device=device, workspace_dir=wsdir)
        raw = pd.read_parquet(wsdir / "SyntheticData"); raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
        d = raw.rename(columns={"customer_id": "entity_id"}).copy()
        d["event_index"] = d.groupby("entity_id", sort=False).cumcount(); d.loc[d.event_index.eq(0), "gap"] = np.nan
        d = d[["entity_id", "event_index", *CORE]]; d, n_named = name_rare(d); d.to_parquet(dest, index=False)
        assert digest(ws.model_tabular_weights_path) == weights
        f = facts_of(d, att); f.update(arm="visitrate", fit_seed=fs, generation_seed=gs, seconds=round(time.monotonic() - t0, 1), rare_rows_named=n_named, kernel=dict(kappa=kappa, lam=lam), base_weights_sha256=weights,
                                       base_module=base_info, merchant_module=mm_info, gap_module=gap_info, attr_module=attr_info, sha256=digest(dest))
        write(folder / f"generation_validation_{gs}.json", f)
        print(f"GENERATED visitrate {fs} {gs}: events={f['events']} frauds={f['frauds']} novel fraud={f['novel_merchant_fraud']:.3f} normal={f['novel_merchant_normal']:.3f} merchants/customer={f['merchants_per_customer_median']} "
              f"top1={f['top1_share_median']:.3f} rare={f['rare_share']:.3f} normal cat-match={f['consistency']['normal']['category_matches_merchant']:.3f} {f['seconds']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["derive", "generate"]); ap.add_argument("--fit", type=int, default=20261108); a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"derive": derive, "generate": lambda: generate(a.fit)}[a.command]()
