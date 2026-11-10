"""ibm_merchant_attr_v1: merchant-attribute consistency on the current ARGN model (native + transition table, label first
+ routed_subst heads + merchant memory `full` + gap tables `table_kind`). docs/ibm_merchant_attr_v1/PROTOCOL.md.
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_merchant_attr.py fit --fit 20261108
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_merchant_attr.py generate --fit 20261108
Whole-model fits: 0. Tables use train only; the sealed test is never read.
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
CHAN_TABLE = ROOT / "artifacts/ibm_merchant_memory_v1/channel_table.json"
GAP_DIR = ROOT / "artifacts/ibm_run_gap_v1"
OUT = ROOT / "artifacts/ibm_merchant_attr_v1"
DOCS = ROOT / "docs/ibm_merchant_attr_v1"
CORE = ["gap", "receiver_or_mark", "amount_or_numeric_value", "category", "use_chip", "event_is_fraud"]
GEN_SEEDS = [20261011, 20261012]
ONLINE = "Online Transaction"


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


def tables_path(fs):
    return OUT / f"attr_tables_{fs}.npz"


def merchant_attributes():
    t = pd.read_parquet(NATIVE / "prepared/train.parquet"); t["m"] = t.receiver_or_mark.astype(str); g = t.groupby("m")
    return pd.DataFrame(dict(modal_cat=g.category.agg(lambda s: str(s.astype(str).value_counts().index[0])), modal_online=g.use_chip.apply(lambda s: s.astype(str).eq(ONLINE).mean() >= 0.5)))


def consistency(d, att):
    d = d.copy(); d["m"] = d.receiver_or_mark.astype(str); d = d.merge(att, left_on="m", right_index=True, how="left")
    y = pd.to_numeric(d.event_is_fraud, errors="coerce").fillna(0).astype(int); known = d.modal_cat.notna(); on = d.use_chip.astype(str).eq(ONLINE); out = {}
    for lab, nm in ((0, "normal"), (1, "fraud")):
        k = y.eq(lab) & known
        out[nm] = dict(rows=int(y.eq(lab).sum()), online=float(on[y.eq(lab)].mean()), category_matches_merchant=float(d.loc[k, "category"].astype(str).eq(d.loc[k, "modal_cat"]).mean()),
                       channel_matches_merchant=float((on[k] == d.loc[k, "modal_online"].astype(bool)).mean()), online_at_swipe_merchant=float((on[k] & ~d.loc[k, "modal_online"].astype(bool)).mean()))
    return out


def fit(fs):
    from mostlyai.engine._workspace import Workspace
    from mostlyai.engine._common import get_cardinalities
    from benchmarks.argn_output_heads import argn_column, _records
    from benchmarks.argn_merchant_attr import fit_attr_tables, LABEL, MERCHANT, CATEGORY, CHANNEL, RARE
    ws = Workspace(NATIVE / "runs" / f"native_{fs}" / "workspace"); stats = ws.tgt_stats.read(); card = get_cardinalities(stats)
    keys = dict(label=argn_column(stats, LABEL) + "__cat", merchant=argn_column(stats, MERCHANT) + "__cat", category=argn_column(stats, CATEGORY) + "__cat", channel=argn_column(stats, CHANNEL) + "__cat")
    t0 = time.time(); recs = _records(list(ws.encoded_data_trn.fetch_all())[0]) + _records(list(ws.encoded_data_val.fetch_all())[0])
    tables, summary = fit_attr_tables(recs, keys, int(stats["columns"][LABEL]["codes"]["1"]), int(stats["columns"][MERCHANT]["codes"][RARE]),
                                      card[keys["merchant"]], card[keys["category"]], card[keys["channel"]])
    OUT.mkdir(parents=True, exist_ok=True); np.savez_compressed(tables_path(fs), **tables)
    summary.update(fit=fs, seconds=round(time.time() - t0, 1), keys=keys, channel_codes=stats["columns"][CHANNEL]["codes"], channel_state_table=json.loads(CHAN_TABLE.read_text())["table"],
                   tables_sha256=digest(tables_path(fs)))
    write(DOCS / f"ATTR_TABLES_{fs}.json", summary); print(json.dumps(summary, indent=1))


def generate(fs, arms):
    from mostlyai.engine import generate as engine_generate
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_generation_module import generation_module
    from benchmarks.argn_routed_heads import attach_routed_heads
    from benchmarks.argn_merchant_memory import merchant_module
    from benchmarks.argn_run_gap import gap_module
    from benchmarks.argn_merchant_attr import attr_module
    table = json.loads(TABLE.read_text()); mm = json.loads(MM_PARAMS.read_text()); gap_params = json.loads((GAP_DIR / f"gap_tables_{fs}.json").read_text())
    tables = dict(np.load(tables_path(fs))); state_table = json.loads(CHAN_TABLE.read_text())["table"]; att = merchant_attributes()
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
                    with gap_module(stats, gap_params, "table_kind") as gap_info:
                        with attr_module(stats, tables, state_table, arm) as info:
                            engine_generate(ctx_data=ctx, device=device, workspace_dir=wsdir)
            raw = pd.read_parquet(wsdir / "SyntheticData"); raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
            d = raw.rename(columns={"customer_id": "entity_id"}).copy()
            d["event_index"] = d.groupby("entity_id", sort=False).cumcount(); d.loc[d.event_index.eq(0), "gap"] = np.nan
            d = d[["entity_id", "event_index", *CORE]]; d.to_parquet(dest, index=False)
            assert digest(ws.model_tabular_weights_path) == weights
            lab = pd.to_numeric(d.event_is_fraud, errors="coerce"); cons = consistency(d, att)
            facts = dict(arm=arm, fit_seed=fs, generation_seed=gs, seconds=round(time.monotonic() - t0, 1), events=len(d), customers=int(d.entity_id.nunique()),
                         frauds=int(lab.eq(1).sum()), fraud_customers=int(d[lab.eq(1)].entity_id.nunique()), consistency=cons,
                         base_weights_sha256=weights, heads_sha256=digest(head_path), tables_sha256=digest(tables_path(fs)),
                         base_module=base_info, merchant_module=mm_info, gap_module=gap_info, module=info, sha256=digest(dest))
            write(folder / f"generation_validation_{gs}.json", facts)
            n, f = cons["normal"], cons["fraud"]
            print(f"GENERATED {arm} {fs} {gs}: events={len(d)} frauds={facts['frauds']} normal cat={n['category_matches_merchant']:.3f} chan={n['channel_matches_merchant']:.3f} "
                  f"online@swipe={n['online_at_swipe_merchant']:.4f} | fraud cat={f['category_matches_merchant']:.3f} chan={f['channel_matches_merchant']:.3f} online={f['online']:.3f} "
                  f"{info['counter']} {facts['seconds']}s", flush=True)


if __name__ == "__main__":
    from benchmarks.argn_merchant_attr import ARMS
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["fit", "generate"]); ap.add_argument("--fit", type=int, default=20261108)
    ap.add_argument("--arms", nargs="*", default=list(ARMS)); a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"fit": lambda: fit(a.fit), "generate": lambda: generate(a.fit, a.arms)}[a.command]()
