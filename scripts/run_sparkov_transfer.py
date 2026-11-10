"""sparkov_transfer_v1 (docs/sparkov_transfer_v1/PROTOCOL.md): the IBM procedure on Sparkov with the current code path.
    ./run python scripts/run_sparkov_transfer.py setup                       (copy the native workspaces, fit the transition table)
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_sparkov_transfer.py heads --fit 20260928
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_sparkov_transfer.py generate --fit 20260928 --arm sf_base
    ./run python scripts/run_sparkov_transfer.py parts --fit 20260928         (gap / attribute / merchant-memory tables, train only)
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_sparkov_transfer.py generate --fit 20260928 --arm sf_gap --parts gap
Whole-model fits: 0 (the native ARGN fits of sparkov_argn_control_v2 are copied, never modified). Development split only.
"""
import argparse, hashlib, json, logging, os, random, shutil, sys, time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for k, v in {"LOKY_MAX_CPU_COUNT": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
import numpy as np
import pandas as pd
import torch

SPK = Path("/home/ssd-990/soobeenkim/DataGeneration/data/sparkov_prepared")
FA = Path("/home/ssd-990/soobeenkim/DataGeneration/workspace/research-argn-fraud-audit/artifacts/sparkov_argn_control_v2")
OUT = ROOT / "artifacts/sparkov_transfer_v1"
DOCS = ROOT / "docs/sparkov_transfer_v1"
TABLE = OUT / "transition_table.json"
FITS = [20260928, 20260929]; GEN_SEEDS = [20261011, 20261012]
CORE = ["gap", "receiver_or_mark", "amount_or_numeric_value", "category", "event_is_fraud"]
ROUTED = ("gap", "amount_or_numeric_value", "category")


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


def wsdir(fs):
    return OUT / "workspaces" / f"native_{fs}"


def setup():
    from benchmarks.argn_generation_module import fit_table
    from mostlyai.engine._workspace import Workspace
    info = {}
    for fs in FITS:
        src = FA / "runs" / f"native_{fs}" / "workspace"; dst = wsdir(fs)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(src, dst)
        sha = digest(Workspace(dst).model_tabular_weights_path); rec = json.loads((FA / "runs" / f"native_{fs}" / "FIT.json").read_text())
        assert sha == rec["weights_sha256"] == digest(Workspace(src).model_tabular_weights_path), "copied weights differ from the recorded fit"
        info[str(fs)] = dict(source=str(src), copy=str(dst), weights_sha256=sha)
    train = pd.read_parquet(SPK / "train.parquet"); p = fit_table(train); write(TABLE, p)
    info["transition_table"] = dict(first=p["first"], train_customers=p["train_customers"], train_events=p["train_events"], train_frauds=p["train_frauds"],
                                    p_fraud_after_normal_age1_never=p["table"][1][1][0], p_fraud_after_fraud_age1=p["table"][2][1][1], p_fraud_after_fraud_age5=p["table"][2][5][1])
    write(DOCS / "SETUP.json", info); print(json.dumps(info, indent=1))


def heads(fs):
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_routed_heads import train_routed_heads
    ws = Workspace(wsdir(fs)); before = digest(ws.model_tabular_weights_path); folder = OUT / "heads" / f"routed_subst_{fs}"
    if (folder / "TRAIN.json").exists():
        print("exists; skipping", folder); return
    meta = train_routed_heads(ws, folder, True, fs, device="cuda" if torch.cuda.is_available() else "cpu", log=lambda s: print(s, flush=True), columns=ROUTED)
    assert digest(ws.model_tabular_weights_path) == before
    meta.update(base_weights_sha256=before, heads_sha256=digest(folder / "heads.pt"), base_weights_unchanged=True); write(folder / "TRAIN.json", meta)
    DOCS.mkdir(parents=True, exist_ok=True); pd.read_csv(folder / "learning.csv").to_csv(DOCS / f"learning_routed_subst_{fs}.csv", index=False)
    write(DOCS / f"train_routed_subst_{fs}.json", {k: v for k, v in meta.items() if k != "head_keys"})
    print(json.dumps({k: meta[k] for k in ["mode", "seed", "epochs", "best_epoch_by_column", "best_val_by_column", "trn_fraud_events", "val_fraud_events", "seconds"]}))


def parts(fs):
    """Train-only tables of the candidate parts for this fit: gap by run position, merchant -> category, merchant memory."""
    from mostlyai.engine._workspace import Workspace
    from mostlyai.engine._common import get_cardinalities
    from benchmarks.argn_output_heads import argn_column, _records
    from benchmarks.argn_run_gap import fit_gap_tables
    from benchmarks.argn_merchant_attr import fit_attr_tables
    from benchmarks.argn_merchant_memory import fit_merchant_memory
    ws = Workspace(wsdir(fs)); stats = ws.tgt_stats.read(); card = get_cardinalities(stats)
    lk = argn_column(stats, "event_is_fraud") + "__cat"; gk = argn_column(stats, "gap") + "__bin"; mk = argn_column(stats, "receiver_or_mark") + "__cat"; ck = argn_column(stats, "category") + "__cat"
    fc = int(stats["columns"]["event_is_fraud"]["codes"]["1"]); rare = int(stats["columns"]["receiver_or_mark"]["codes"]["_RARE_"])
    recs = _records(list(ws.encoded_data_trn.fetch_all())[0]) + _records(list(ws.encoded_data_val.fetch_all())[0])
    gap = fit_gap_tables(recs, lk, gk, fc, card[gk]); write(OUT / f"gap_tables_{fs}.json", gap)
    tables, summary = fit_attr_tables(recs, dict(label=lk, merchant=mk, category=ck, channel=None), fc, rare, card[mk], card[ck], 1); np.savez_compressed(OUT / f"attr_tables_{fs}.npz", **tables)
    mm = fit_merchant_memory(pd.read_parquet(SPK / "train.parquet"), stats); write(OUT / f"merchant_memory_{fs}.json", mm)
    rep = dict(fit=fs, gap_rows=gap["rows"], attr=summary, merchant_memory={k: mm[k] for k in ("gate", "lam", "kernel_holdout_logloss", "pi", "pool_merchants", "train_novel_share", "holdout_repeat_rows")})
    write(DOCS / f"PARTS_{fs}.json", rep); print(json.dumps(rep, indent=1)[:2500])


def kind_gap(d):
    d = d.sort_values(["entity_id", "event_index"], kind="stable"); y = pd.to_numeric(d.event_is_fraud, errors="coerce").fillna(0).astype(int); py = y.groupby(d.entity_id).shift()
    g = pd.to_numeric(d.gap, errors="coerce").astype(float) / 60.0
    return {n: (float(np.median(g[s].dropna())) if s.any() else None) for n, s in (("onset", y.eq(1) & py.eq(0)), ("cont", y.eq(1) & py.eq(1)), ("exit", y.eq(0) & py.eq(1)), ("normal", y.eq(0) & py.eq(0)))}


def generate(fs, arm, part_list):
    from mostlyai.engine import generate as engine_generate
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_generation_module import generation_module
    from benchmarks.argn_routed_heads import attach_routed_heads
    table = json.loads(TABLE.read_text()); ws = Workspace(wsdir(fs)); stats = ws.tgt_stats.read(); weights = digest(ws.model_tabular_weights_path)
    ctx = pd.read_parquet(SPK / "validation_context.parquet"); device = "cuda" if torch.cuda.is_available() else "cpu"
    head_path = OUT / "heads" / f"routed_subst_{fs}" / "heads.pt"; head_state = torch.load(head_path, map_location="cpu")
    train = pd.read_parquet(SPK / "train.parquet"); modal = train.groupby(train.receiver_or_mark.astype(str)).category.agg(lambda s: str(s.astype(str).value_counts().index[0]))
    folder = OUT / "runs" / f"{arm}_{fs}"; folder.mkdir(parents=True, exist_ok=True)
    for gs in GEN_SEEDS:
        dest = folder / f"generated_validation_{gs}.parquet"
        if dest.exists():
            continue
        seed(gs); t0 = time.monotonic(); infos = {}
        with ExitStack() as stack:
            infos["base"] = stack.enter_context(generation_module(stats, table, label_first=True, head_state=head_state, attach=lambda m: attach_routed_heads(m, stats, columns=ROUTED)))
            if "merchant" in part_list:
                from benchmarks.argn_merchant_memory import merchant_module
                infos["merchant"] = stack.enter_context(merchant_module(stats, json.loads((OUT / f"merchant_memory_{fs}.json").read_text()), arm="full"))
            if "gap" in part_list:
                from benchmarks.argn_run_gap import gap_module
                infos["gap"] = stack.enter_context(gap_module(stats, json.loads((OUT / f"gap_tables_{fs}.json").read_text()), "table_kind"))
            if "attr" in part_list or "attr_all" in part_list:  # amendment 1: "attr" = normal rows only, "attr_all" = fraud rows too
                from benchmarks.argn_merchant_attr import attr_module
                infos["attr"] = stack.enter_context(attr_module(stats, dict(np.load(OUT / f"attr_tables_{fs}.npz")), (), "attr_all" if "attr_all" in part_list else "attr_normal"))
            engine_generate(ctx_data=ctx, device=device, workspace_dir=wsdir(fs))
        raw = pd.read_parquet(wsdir(fs) / "SyntheticData"); raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
        d = raw.rename(columns={"customer_id": "entity_id"}).copy()
        d["event_index"] = d.groupby("entity_id", sort=False).cumcount(); d.loc[d.event_index.eq(0), "gap"] = np.nan
        d = d[["entity_id", "event_index", *CORE]]; d.to_parquet(dest, index=False)
        assert digest(ws.model_tabular_weights_path) == weights
        lab = pd.to_numeric(d.event_is_fraud, errors="coerce").fillna(0).astype(int); novel = ~d.duplicated(["entity_id", "receiver_or_mark"]); imp = d.receiver_or_mark.astype(str).map(modal)
        s = d.sort_values(["entity_id", "event_index"]); ys = pd.to_numeric(s.event_is_fraud, errors="coerce").fillna(0).astype(int); prev = ys.groupby(s.entity_id).shift()
        facts = dict(arm=arm, parts=list(part_list), fit_seed=fs, generation_seed=gs, seconds=round(time.monotonic() - t0, 1), events=len(d), customers=int(d.entity_id.nunique()), frauds=int(lab.sum()),
                     fraud_customers=int(d[lab.eq(1)].entity_id.nunique()), continuation=float(ys[prev.eq(1)].mean()) if prev.eq(1).any() else None,
                     novel_merchant_fraud=float(novel[lab.eq(1)].mean()) if lab.any() else None, novel_merchant_normal=float(novel[lab.eq(0)].mean()),
                     merchants_per_customer_median=float(d.groupby("entity_id").receiver_or_mark.nunique().median()),
                     category_matches_merchant_normal=float(d.category.astype(str)[lab.eq(0)].eq(imp[lab.eq(0)]).mean()), category_matches_merchant_fraud=float(d.category.astype(str)[lab.eq(1)].eq(imp[lab.eq(1)]).mean()) if lab.any() else None,
                     gap_minutes_by_kind=kind_gap(d), base_weights_sha256=weights, heads_sha256=digest(head_path), modules=infos, sha256=digest(dest))
        write(folder / f"generation_validation_{gs}.json", facts)
        print(f"GENERATED {arm} {fs} {gs}: events={len(d)} frauds={facts['frauds']} fraud_customers={facts['fraud_customers']} cont={facts['continuation']:.3f} novel f/n={facts['novel_merchant_fraud']:.3f}/{facts['novel_merchant_normal']:.3f} "
              f"merchants={facts['merchants_per_customer_median']} cat-match n/f={facts['category_matches_merchant_normal']:.3f}/{facts['category_matches_merchant_fraud']:.3f} gap={ {k: (round(v) if v is not None else None) for k, v in facts['gap_minutes_by_kind'].items()} } {facts['seconds']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["setup", "heads", "parts", "generate"]); ap.add_argument("--fit", type=int, default=FITS[0])
    ap.add_argument("--arm", default="sf_base"); ap.add_argument("--parts", nargs="*", default=[]); a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"setup": setup, "heads": lambda: heads(a.fit), "parts": lambda: parts(a.fit), "generate": lambda: generate(a.fit, a.arm, a.parts)}[a.command]()
