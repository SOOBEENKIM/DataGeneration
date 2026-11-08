"""ibm_routed_heads_v1: label-routed (fraud-only) output heads on the frozen IBM native ARGN fits, then generate.
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_routed_heads.py train --fit 20261108 [--subst]
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_routed_heads.py generate --fit 20261108
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_routed_heads.py probe --fit 20261108
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_routed_heads.py evaluate --fit 20261108
Whole-model fits: 0. Development/test customers are never used for training or selection.
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
from benchmarks.argn_generation_module import generation_module  # noqa: E402
from benchmarks.argn_routed_heads import train_routed_heads, evaluate_routed, routed_model, attach_routed_heads  # noqa: E402
from benchmarks.argn_output_heads import argn_column, LABEL, _records  # noqa: E402

NATIVE = ROOT / "artifacts/ibm_tabformer_v1"
TABLE = ROOT / "artifacts/ibm_generation_module_v1/transition_table.json"
OUT = ROOT / "artifacts/ibm_routed_heads_v1"
DOCS = ROOT / "docs/ibm_routed_heads_v1"
CORE = ["gap", "receiver_or_mark", "amount_or_numeric_value", "category", "use_chip", "event_is_fraud"]
ARMS = {"routed_table": dict(heads="routed", table=True), "routed_subst_table": dict(heads="routed_subst", table=True)}
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


def train(fs, subst):
    from mostlyai.engine._workspace import Workspace
    ws = Workspace(NATIVE / "runs" / f"native_{fs}" / "workspace"); before = digest(ws.model_tabular_weights_path)
    mode = "routed_subst" if subst else "routed"; folder = OUT / "heads" / f"{mode}_{fs}"
    meta = train_routed_heads(ws, folder, subst, fs, device="cuda" if torch.cuda.is_available() else "cpu", log=lambda s: print(s, flush=True))
    assert digest(ws.model_tabular_weights_path) == before
    meta.update(base_weights_sha256=before, heads_sha256=digest(folder / "heads.pt"), base_weights_unchanged=True)
    write(folder / "TRAIN.json", meta); DOCS.mkdir(parents=True, exist_ok=True)
    pd.read_csv(folder / "learning.csv").to_csv(DOCS / f"learning_{mode}_{fs}.csv", index=False)
    write(DOCS / f"train_{mode}_{fs}.json", {k: v for k, v in meta.items() if k != "head_keys"})
    print(json.dumps({k: meta[k] for k in ["mode", "seed", "epochs", "best_epoch_by_column", "best_val_by_column", "trainable_parameters", "seconds"]}))


def generate(fs, arms):
    from mostlyai.engine import generate as engine_generate
    from mostlyai.engine._workspace import Workspace
    params = json.loads(TABLE.read_text())
    wsdir = NATIVE / "runs" / f"native_{fs}" / "workspace"; ws = Workspace(wsdir); stats = ws.tgt_stats.read()
    weights = digest(ws.model_tabular_weights_path)
    ctx = pd.read_parquet(NATIVE / "prepared/validation_context.parquet")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for arm in arms:
        cfg = ARMS[arm]; head_path = OUT / "heads" / f"{cfg['heads']}_{fs}" / "heads.pt"
        head_state = torch.load(head_path, map_location="cpu")
        folder = OUT / "runs" / f"{arm}_{fs}"; folder.mkdir(parents=True, exist_ok=True)
        for gs in GEN_SEEDS:
            dest = folder / f"generated_validation_{gs}.parquet"
            if dest.exists():
                continue
            seed(gs); t0 = time.monotonic()
            with generation_module(stats, params if cfg["table"] else None, label_first=True, head_state=head_state,
                                   attach=lambda m: attach_routed_heads(m, stats)) as info:
                engine_generate(ctx_data=ctx, device=device, workspace_dir=wsdir)
            raw = pd.read_parquet(wsdir / "SyntheticData"); raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
            d = raw.rename(columns={"customer_id": "entity_id"}).copy()
            d["event_index"] = d.groupby("entity_id", sort=False).cumcount(); d.loc[d.event_index.eq(0), "gap"] = np.nan
            d = d[["entity_id", "event_index", *CORE]]; d.to_parquet(dest, index=False)
            assert digest(ws.model_tabular_weights_path) == weights
            lab = pd.to_numeric(d.event_is_fraud, errors="coerce"); f = d[lab.eq(1)]
            online = float(f.use_chip.astype(str).eq("Online Transaction").mean()) if len(f) else float("nan")
            prev = lab.groupby(d.entity_id).shift(); cont = float(lab[prev.eq(1)].eq(1).mean()) if prev.eq(1).any() else float("nan")
            write(folder / f"generation_validation_{gs}.json", dict(arm=arm, fit_seed=fs, generation_seed=gs, seconds=round(time.monotonic() - t0, 1), events=len(d),
                                                                   customers=int(d.entity_id.nunique()), frauds=int(len(f)), fraud_customers=int(f.entity_id.nunique()), fraud_online_share=online,
                                                                   continuation=cont, base_weights_sha256=weights, heads_sha256=digest(head_path), module=info, sha256=digest(dest)))
            print(f"GENERATED {arm} {fs} {gs}: events={len(d)} frauds={len(f)} fraud_customers={f.entity_id.nunique()} online={online:.3f} cont={cont:.3f}", flush=True)


@torch.no_grad()
def probe(fs, heads):
    """Label-path probe (internal validation): mean P(online) on real fraud rows (routed to the fraud head), on the
    same rows with the label input set to 0 (routed to the native head), on real normal rows, and on real normal
    rows with the label input set to 1 (fraud head on normal context = the generation condition)."""
    from mostlyai.engine._workspace import Workspace
    from mostlyai.engine._tabular.training import BatchCollator
    from torch.utils.data import DataLoader
    ws = Workspace(NATIVE / "runs" / f"native_{fs}" / "workspace")
    rows = []
    for name in ["native"] + heads:
        state = None if name == "native" else torch.load(OUT / "heads" / f"{name}_{fs}" / "heads.pt", map_location="cpu")
        model, ts, card, _ = routed_model(ws, "cuda", state); model.eval()
        label_key = argn_column(ts, LABEL) + "__cat"; code1 = int(ts["columns"][LABEL]["codes"]["1"]); code0 = int(ts["columns"][LABEL]["codes"]["0"])
        uc = [k for k in card if k.startswith(argn_column(ts, "use_chip") + "__")][0]; online = int(ts["columns"]["use_chip"]["codes"]["Online Transaction"])
        recs = _records(list(ws.encoded_data_val.fetch_all())[0])
        loader = DataLoader(recs, shuffle=False, batch_size=4, collate_fn=BatchCollator(True, None, torch.device("cuda")))
        res = dict(fit_seed=fs, heads=name)
        for variant in ["fraud_rows_label_real", "fraud_rows_label_set_0", "normal_rows_label_real", "normal_rows_label_set_1"]:
            ps = []
            for data in loader:
                d = dict(data); fraud = d[label_key].squeeze(-1).eq(code1); normal = d[label_key].squeeze(-1).eq(code0)
                if variant.endswith("set_0"):
                    d[label_key] = torch.where(fraud.unsqueeze(-1), torch.full_like(d[label_key], code0), d[label_key])
                if variant.endswith("set_1"):
                    d[label_key] = torch.where(normal.unsqueeze(-1), torch.full_like(d[label_key], code1), d[label_key])
                out, _ = model(d, mode="trn"); p = out[uc].softmax(-1)[..., online]
                ps.append(p[fraud if variant.startswith("fraud") else normal].cpu())
            res[variant] = round(float(torch.cat(ps).mean()), 4)
        rows.append(res); print(res, flush=True)
    DOCS.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(DOCS / f"LABEL_PATH_PROBE_{fs}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["train", "generate", "probe", "evaluate"]); ap.add_argument("--fit", type=int, default=20261108)
    ap.add_argument("--subst", action="store_true"); ap.add_argument("--arms", nargs="*", default=list(ARMS)); ap.add_argument("--heads", nargs="*", default=["routed", "routed_subst"]); a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.command == "train":
        train(a.fit, a.subst)
    elif a.command == "generate":
        generate(a.fit, a.arms)
    elif a.command == "probe":
        probe(a.fit, a.heads)
    else:
        from mostlyai.engine._workspace import Workspace
        ws = Workspace(NATIVE / "runs" / f"native_{a.fit}" / "workspace"); rows = []
        for name in ["native"] + a.heads:
            state = None if name == "native" else torch.load(OUT / "heads" / f"{name}_{a.fit}" / "heads.pt", map_location="cpu")
            r = evaluate_routed(ws, state); r.insert(0, "heads", name); rows.append(r)
        res = pd.concat(rows); DOCS.mkdir(parents=True, exist_ok=True); res.to_csv(DOCS / f"val_ce_by_column_{a.fit}.csv", index=False)
        print(res.pivot_table(index=["column", "split"], columns="heads", values="mean_ce").round(4).to_string())
