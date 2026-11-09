"""ibm_baselines_v1: standard rare-class remedies on the native IBM ARGN (mostlyai-engine 1.0.4), as registered in
docs/ibm_baselines_v1/PROTOCOL.md. Every arm starts from artifacts/ibm_tabformer_v1/prepared/child and changes one thing:
    rebalance          engine `rebalancing` at generation (label first, fixed marginal), native fits reused
    oversample_x3/x8   fraud-experienced customer sequences duplicated in the trn partition only (after the engine split)
    weight_sqrt/inv    fraud-row loss weight in an isolated source copy of the engine training module (installed engine untouched)
    fixed_label_first  enable_flexible_generation=False with the label as the first target column
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_baselines.py prepare
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_baselines.py fit --arm oversample_x8 --seed 20261108
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_baselines.py generate --arm rebalance --seed 20261108
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_baselines.py probe --arm weight_inv --seed 20261108
The sealed test split is never read. Generated files: runs/<arm>[_table]_<fit>/generated_validation_<gs>.parquet.
"""
from __future__ import annotations
import argparse, difflib, hashlib, importlib.metadata, inspect, json, logging, os, random, shutil, sys, time
from contextlib import contextmanager, nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NATIVE = ROOT / "artifacts/ibm_tabformer_v1"
TABLE = ROOT / "artifacts/ibm_generation_module_v1/transition_table.json"
OUT = ROOT / "artifacts/ibm_baselines_v1"
DOCS = ROOT / "docs/ibm_baselines_v1"
CONFIG = ROOT / "configs/ibm_tabformer_v1.json"
for k, v in {"HF_HOME": str(NATIVE / "cache/hf"), "JOBLIB_TEMP_FOLDER": str(NATIVE / "cache/joblib"), "LOKY_MAX_CPU_COUNT": "4",
             "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
import numpy as np
import pandas as pd
import torch

CFG = json.loads(CONFIG.read_text())
CORE = CFG["core_columns"]; STATIC = CFG["static_columns"]
MERCHANT, AMOUNT, LABEL = "receiver_or_mark", "amount_or_numeric_value", "event_is_fraud"
CATEGORICAL = {MERCHANT, "category", "use_chip", LABEL}
LABEL_FIRST = [LABEL] + [c for c in CORE if c != LABEL]
FIT_ARMS = ["oversample_x3", "oversample_x8", "weight_sqrt", "weight_inv", "fixed_label_first"]
GEN_SEEDS = CFG["generation_seeds"]


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")


def seed(n):
    random.seed(n); np.random.seed(n); torch.manual_seed(n); torch.cuda.manual_seed_all(n); torch.set_num_threads(CFG["cpu_threads"])


def native_frame(raw):
    d = raw.rename(columns={"customer_id": "entity_id"}).copy()
    d["event_index"] = d.groupby("entity_id", sort=False).cumcount()
    d.loc[d.event_index.eq(0), "gap"] = np.nan
    return d[["entity_id", "event_index", *CORE]]


def prepare():
    """Registered constants from the train split only: fraud-row weights, train fraud rate, trn-partition composition."""
    train = pd.read_parquet(NATIVE / "prepared/train.parquet")
    fraud = train[LABEL].eq("1"); n_f, n_n = int(fraud.sum()), int((~fraud).sum())
    trn = pd.read_parquet(NATIVE / "prepared/child/OriginalData/tgt-data/part.000000-trn.parquet")
    val = pd.read_parquet(NATIVE / "prepared/child/OriginalData/tgt-data/part.000000-val.parquet")
    fc = trn.groupby("customer_id")[LABEL].apply(lambda s: s.eq("1").any())
    info = dict(train_events=len(train), train_frauds=n_f, train_fraud_rate=n_f / len(train),
                weight_sqrt=int(round(np.sqrt(n_n / n_f))), weight_inv=int(round(n_n / n_f)),
                rebalance_probabilities={"1": n_f / len(train)},
                engine_trn_partition=dict(customers=int(fc.size), fraud_customers=int(fc.sum()), events=len(trn), frauds=int(trn[LABEL].eq("1").sum())),
                engine_val_partition=dict(customers=int(val.customer_id.nunique()), events=len(val), frauds=int(val[LABEL].eq("1").sum())),
                oversample=dict(), label_first_columns=LABEL_FIRST, train_sha256=digest(NATIVE / "prepared/train.parquet"),
                protocol_sha256=digest(DOCS / "PROTOCOL.md"), script_sha256=digest(__file__), test_read=False)
    for k in [3, 8]:
        nf = int(fc.sum()) * k; info["oversample"][f"x{k}"] = dict(trn_customers=int(fc.size) + int(fc.sum()) * (k - 1), fraud_customers=nf,
                                                                   normal_customers=int(fc.size) - int(fc.sum()),
                                                                   approx_fraud_rate=(int(trn[LABEL].eq("1").sum()) * k) / (len(trn) + (k - 1) * int(trn[trn.customer_id.isin(fc[fc].index)].shape[0])))
    write(DOCS / "PREPARE.json", info); print(json.dumps(info, indent=1, default=str))


def constants():
    return json.loads((DOCS / "PREPARE.json").read_text())


def oversampled_workspace(wsdir, k):
    """Copy OriginalData, duplicate fraud-experienced customers (k-1 extra copies, new ids) in the trn partition only."""
    from mostlyai.engine import analyze, encode
    shutil.copytree(NATIVE / "prepared/child/OriginalData", wsdir / "OriginalData")
    tp = wsdir / "OriginalData/tgt-data/part.000000-trn.parquet"; cp = wsdir / "OriginalData/ctx-data/part.000000-trn.parquet"
    tgt = pd.read_parquet(tp); ctx = pd.read_parquet(cp)
    fraud_ids = np.sort(tgt.loc[tgt[LABEL].eq("1"), "customer_id"].unique())
    tgt_parts, ctx_parts = [tgt], [ctx]
    for copy in range(1, k):
        offset = 1_000_000 * copy
        t = tgt[tgt.customer_id.isin(fraud_ids)].copy(); t["customer_id"] = t.customer_id + offset; tgt_parts.append(t)
        c = ctx[ctx.customer_id.isin(fraud_ids)].copy(); c["customer_id"] = c.customer_id + offset; ctx_parts.append(c)
    tgt2 = pd.concat(tgt_parts, ignore_index=True); ctx2 = pd.concat(ctx_parts, ignore_index=True)
    assert ctx2.customer_id.is_unique and set(tgt2.customer_id) == set(ctx2.customer_id)
    tgt2.to_parquet(tp, index=False); ctx2.to_parquet(cp, index=False)
    seed(CFG["preparation_seed"]); analyze(workspace_dir=wsdir); encode(workspace_dir=wsdir)
    return dict(duplicated_customers=int(len(fraud_ids)), copies=k, trn_customers=int(ctx2.customer_id.nunique()), trn_events=len(tgt2),
                trn_frauds=int(tgt2[LABEL].eq("1").sum()), trn_fraud_rate=float(tgt2[LABEL].eq("1").mean()), val_partition_unchanged=True)


def label_first_workspace(wsdir):
    """Re-split the same train customers with the label as the first target column (fixed order is read from the column order)."""
    from mostlyai.engine import split, analyze, encode
    train = pd.read_parquet(NATIVE / "prepared/train.parquet"); parent = pd.read_parquet(NATIVE / "prepared/train_context.parquet")
    target = train.rename(columns={"entity_id": "customer_id"})[["customer_id", *LABEL_FIRST]].copy(); target.gap = target.gap.fillna(0)
    types = {c: "TABULAR_CATEGORICAL" if c in CATEGORICAL else "TABULAR_NUMERIC_AUTO" for c in LABEL_FIRST}
    ctypes = {c: "TABULAR_CATEGORICAL" if c == "home_state" else "TABULAR_NUMERIC_AUTO" for c in STATIC}
    seed(CFG["preparation_seed"])
    split(tgt_data=target, ctx_data=parent, tgt_context_key="customer_id", ctx_primary_key="customer_id",
          tgt_encoding_types=types, ctx_encoding_types=ctypes, workspace_dir=wsdir)
    analyze(workspace_dir=wsdir); encode(workspace_dir=wsdir)
    from mostlyai.engine._workspace import Workspace
    cols = list(Workspace(wsdir).tgt_stats.read()["columns"]); assert cols == LABEL_FIRST, cols
    # the engine's own split is random: record whether it reproduced the native trn/val customer partition
    native_val = set(pd.read_parquet(NATIVE / "prepared/child/OriginalData/tgt-data/part.000000-val.parquet").customer_id)
    this_val = set(pd.read_parquet(wsdir / "OriginalData/tgt-data/part.000000-val.parquet").customer_id)
    return dict(columns=cols, same_val_partition_as_native=native_val == this_val, val_overlap=len(native_val & this_val))


@contextmanager
def fraud_row_weight(folder, label_key, fraud_code, w):
    """Isolated source copy of mostlyai.engine._tabular.training with the padding mask of every target column scaled by `w`
    on rows whose label sub-column equals the fraud code. Same mechanism as the registered cap-only control; the installed
    engine is untouched and `module.train` is restored afterwards."""
    import mostlyai.engine._tabular.training as module
    source = inspect.getsource(module)
    old = "                # mask out paddings\n                mask = slen_mask\n"
    assert source.count(old) == 1
    new = ("                # mask out paddings; registered fraud-row weight (ibm_baselines_v1)\n"
           "                mask = slen_mask * torch.where(data[FRAUD_LABEL_KEY].squeeze(-1) == FRAUD_CODE, "
           "torch.full_like(slen_mask, FRAUD_WEIGHT), torch.ones_like(slen_mask))\n")
    modified = source.replace(old, new)
    path = folder / "training_fraud_weight.py"; path.write_text(modified)
    (folder / "training_fraud_weight.diff").write_text("".join(difflib.unified_diff(source.splitlines(True), modified.splitlines(True), fromfile="official_training.py", tofile=path.name)))
    write(folder / "source_control.json", dict(official_sha256=hashlib.sha256(source.encode()).hexdigest(), controlled_sha256=digest(path),
                                               installed_source_unchanged=True, label_key=label_key, fraud_code=fraud_code, weight=w))
    namespace = dict(module.__dict__); namespace.update(FRAUD_LABEL_KEY=label_key, FRAUD_CODE=fraud_code, FRAUD_WEIGHT=w)
    exec(compile(modified, str(path), "exec"), namespace)
    original = module.train; module.train = namespace["train"]
    try:
        yield
    finally:
        module.train = original


def label_key_and_code(ws):
    from benchmarks.argn_output_heads import argn_column
    ts = ws.tgt_stats.read(); return argn_column(ts, LABEL) + "__cat", int(ts["columns"][LABEL]["codes"]["1"])


def parameter_count(ws):
    from benchmarks.argn_output_heads import build_model
    model, _, _ = build_model(ws, "cpu", None)
    return int(sum(p.numel() for p in model.parameters()))


def fit(arm, fit_seed):
    from mostlyai.engine import train
    from mostlyai.engine._workspace import Workspace
    assert arm in FIT_ARMS, arm
    const = constants(); folder = OUT / "runs" / f"{arm}_{fit_seed}"
    wsdir = folder / "workspace"; device = CFG["device"] if torch.cuda.is_available() else "cpu"
    if folder.exists() and Workspace(wsdir).model_tabular_weights_path.exists():
        # 2026-11-09: the first launch trained and saved weights but crashed while reading the progress file (CSV, not JSONL);
        # finish the bookkeeping from the saved workspace without retraining (training start time from START.json mtime)
        if not (folder / "FIT.json").exists():
            ws = Workspace(wsdir); start_json = folder / "START.json"
            finish(arm, folder, ws, seconds=ws.model_tabular_weights_path.stat().st_mtime - start_json.stat().st_mtime, device=device, resumed=True)
        sample(arm, fit_seed); return
    folder.mkdir(parents=True, exist_ok=False)
    seed(fit_seed); start = time.monotonic(); data_info = {}
    if arm.startswith("oversample"):
        data_info = oversampled_workspace(wsdir, int(arm.split("_x")[1]))
    elif arm == "fixed_label_first":
        data_info = label_first_workspace(wsdir)
    else:
        shutil.copytree(NATIVE / "prepared/child", wsdir)
    ws = Workspace(wsdir); label_key, code1 = label_key_and_code(ws)
    write(folder / "START.json", dict(arm=arm, fit_seed=fit_seed, engine=importlib.metadata.version("mostlyai-engine"), device=device,
                                      gpu=torch.cuda.get_device_name(0) if device == "cuda" else None, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                      config_sha256=digest(CONFIG), script_sha256=digest(__file__), protocol_sha256=digest(DOCS / "PROTOCOL.md"),
                                      max_training_minutes=CFG["max_training_minutes"], data=data_info, label_key=label_key, fraud_code=code1,
                                      prepare_sha256=digest(DOCS / "PREPARE.json")))
    seed(fit_seed)
    if arm.startswith("weight"):
        w = int(const[arm])
        with fraud_row_weight(folder, label_key, code1, w):
            train(max_training_time=CFG["max_training_minutes"], device=device, workspace_dir=wsdir)
    elif arm == "fixed_label_first":
        train(max_training_time=CFG["max_training_minutes"], device=device, workspace_dir=wsdir, enable_flexible_generation=False)
    else:
        train(max_training_time=CFG["max_training_minutes"], device=device, workspace_dir=wsdir)
    finish(arm, folder, ws, seconds=time.monotonic() - start, device=device, resumed=False)
    sample(arm, fit_seed)


def finish(arm, folder, ws, seconds, device, resumed):
    progress = ws.model_progress_messages_path  # engine writes a CSV (epoch,is_checkpoint,steps,samples,trn_loss,total_time,learn_rate,val_loss)
    msgs = pd.read_csv(progress) if progress.exists() else pd.DataFrame()
    ck = msgs[msgs.get("is_checkpoint", pd.Series(dtype=int)).eq(1)] if len(msgs) else msgs
    write(folder / "FIT.json", dict(seconds=seconds, resumed_bookkeeping=resumed, weights_sha256=digest(ws.model_tabular_weights_path), device=device,
                                    trainable_parameters=parameter_count(ws), epochs_run=float(msgs.epoch.max()) if len(msgs) else None,
                                    best_checkpoint_epoch=float(ck.epoch.iloc[-1]) if len(ck) else None, best_val_loss=float(ck.val_loss.iloc[-1]) if len(ck) else None,
                                    model_configs=ws.model_tabular_configs.read()))
    if progress.exists():
        shutil.copy(progress, folder / "progress_messages.csv")
    print(json.dumps(json.loads((folder / "FIT.json").read_text()), indent=1)[:1500], flush=True)


def sample(arm, fit_seed, table_arms=(False, True)):
    """Native generations and, with the registered transition table (label-first generation order), `<arm>_table`."""
    from mostlyai.engine import generate
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_generation_module import generation_module
    wsdir = OUT / "runs" / f"{arm}_{fit_seed}" / "workspace"; ws = Workspace(wsdir); stats = ws.tgt_stats.read()
    device = CFG["device"] if torch.cuda.is_available() else "cpu"; weights = digest(ws.model_tabular_weights_path)
    ctx = pd.read_parquet(NATIVE / "prepared/validation_context.parquet"); params = json.loads(TABLE.read_text())
    for with_table in table_arms:
        folder = OUT / "runs" / (f"{arm}_table_{fit_seed}" if with_table else f"{arm}_{fit_seed}"); folder.mkdir(parents=True, exist_ok=True)
        for gs in GEN_SEEDS:
            dest = folder / f"generated_validation_{gs}.parquet"
            if dest.exists():
                continue
            seed(gs); start = time.monotonic()
            with generation_module(stats, params, label_first=True) if with_table else nullcontext() as info:
                generate(ctx_data=ctx, device=device, workspace_dir=wsdir)
            raw = pd.read_parquet(wsdir / "SyntheticData"); raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
            d = native_frame(raw); d.to_parquet(dest, index=False)
            assert digest(ws.model_tabular_weights_path) == weights
            write(folder / f"generation_validation_{gs}.json", dict(arm=arm, with_table=with_table, module=info, seconds=time.monotonic() - start, events=len(d),
                                                                   customers=int(d.entity_id.nunique()), frauds=int(pd.to_numeric(d[LABEL], errors="coerce").eq(1).sum()),
                                                                   fraud_customers=int(d.loc[d[LABEL].astype(str).eq("1"), "entity_id"].nunique()),
                                                                   weights_unchanged=True, sha256=digest(dest), table_sha256=digest(TABLE) if with_table else None))
            print(f"GENERATED {folder.name} {gs}: events {len(d)} frauds {int(d[LABEL].astype(str).eq('1').sum())} {time.monotonic() - start:.0f}s", flush=True)


def rebalance(fit_seed):
    """Engine-native generation-time rebalancing of the label marginal on the frozen native fit (no new fit)."""
    from mostlyai.engine import generate
    from mostlyai.engine._workspace import Workspace
    const = constants(); probs = {k: float(v) for k, v in const["rebalance_probabilities"].items()}
    wsdir = NATIVE / "runs" / f"native_{fit_seed}" / "workspace"; ws = Workspace(wsdir); weights = digest(ws.model_tabular_weights_path)
    device = CFG["device"] if torch.cuda.is_available() else "cpu"; ctx = pd.read_parquet(NATIVE / "prepared/validation_context.parquet")
    folder = OUT / "runs" / f"rebalance_{fit_seed}"; folder.mkdir(parents=True, exist_ok=True)
    for gs in GEN_SEEDS:
        dest = folder / f"generated_validation_{gs}.parquet"
        if dest.exists():
            continue
        seed(gs); start = time.monotonic()
        generate(ctx_data=ctx, device=device, workspace_dir=wsdir, rebalancing=dict(column=LABEL, probabilities=probs))
        raw = pd.read_parquet(wsdir / "SyntheticData"); raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
        d = native_frame(raw); d.to_parquet(dest, index=False)
        assert digest(ws.model_tabular_weights_path) == weights
        write(folder / f"generation_validation_{gs}.json", dict(arm="rebalance", rebalancing=dict(column=LABEL, probabilities=probs), seconds=time.monotonic() - start,
                                                               events=len(d), customers=int(d.entity_id.nunique()), frauds=int(d[LABEL].astype(str).eq("1").sum()),
                                                               fraud_customers=int(d.loc[d[LABEL].astype(str).eq("1"), "entity_id"].nunique()), base_weights_sha256=weights,
                                                               weights_unchanged=True, sha256=digest(dest)))
        print(f"GENERATED rebalance_{fit_seed} {gs}: events {len(d)} frauds {int(d[LABEL].astype(str).eq('1').sum())} {time.monotonic() - start:.0f}s", flush=True)


def probe(arm, fit_seed):
    """Label-path probe on the internal validation partition (same four conditions as ibm_routed_heads_v1 §probe) plus the
    age-1 continuation probability P(fraud | previous row fraud) in teacher mode. Weights unchanged."""
    from mostlyai.engine._workspace import Workspace
    from mostlyai.engine._tabular.training import BatchCollator
    from mostlyai.engine._common import get_cardinalities
    from torch.utils.data import DataLoader
    from benchmarks.argn_output_heads import build_model, label_first_order, argn_column, _records
    wsdir = (NATIVE / "runs" / f"native_{fit_seed}" / "workspace") if arm in ("native", "rebalance") else OUT / "runs" / f"{arm}_{fit_seed}" / "workspace"
    ws = Workspace(wsdir); before = digest(ws.model_tabular_weights_path); ts = ws.tgt_stats.read(); card = get_cardinalities(ts)
    order = label_first_order(ts, card); model, _, _ = build_model(ws, "cuda", order); model.eval()
    label_key = argn_column(ts, LABEL) + "__cat"; c1 = int(ts["columns"][LABEL]["codes"]["1"]); c0 = int(ts["columns"][LABEL]["codes"]["0"])
    uc = [k for k in card if k.startswith(argn_column(ts, "use_chip") + "__")][0]; online = int(ts["columns"]["use_chip"]["codes"]["Online Transaction"])
    recs = _records(list(ws.encoded_data_val.fetch_all())[0])
    loader = DataLoader(recs, shuffle=False, batch_size=4, collate_fn=BatchCollator(True, None, torch.device("cuda")))
    res = dict(arm=arm, fit_seed=fit_seed, column_order=[c.split("/")[-1] for c in order])
    with torch.no_grad():
        for variant in ["fraud_rows_label_real", "fraud_rows_label_set_0", "normal_rows_label_real", "normal_rows_label_set_1"]:
            ps = []
            for data in loader:
                d = dict(data); fraud = d[label_key].squeeze(-1).eq(c1); normal = d[label_key].squeeze(-1).eq(c0)
                if variant.endswith("set_0"):
                    d[label_key] = torch.where(fraud.unsqueeze(-1), torch.full_like(d[label_key], c0), d[label_key])
                if variant.endswith("set_1"):
                    d[label_key] = torch.where(normal.unsqueeze(-1), torch.full_like(d[label_key], c1), d[label_key])
                out, _ = model(d, mode="trn"); p = out[uc].softmax(-1)[..., online]
                ps.append(p[fraud if variant.startswith("fraud") else normal].cpu())
            res[variant] = round(float(torch.cat(ps).mean()), 4)
        # label transition in teacher mode: P(fraud at t | real history up to t-1), grouped by the previous row's label
        num = {"prev_normal": [], "prev_fraud": []}
        for data in loader:
            out, _ = model(dict(data), mode="trn"); p = out[label_key].softmax(-1)[..., c1]
            lab = data[label_key].squeeze(-1); valid = lab.ne(0)
            prev = torch.roll(lab, 1, dims=1); prev[:, 0] = -1; ok = valid & prev.ge(0)
            num["prev_normal"].append(p[ok & prev.eq(c0)].cpu()); num["prev_fraud"].append(p[ok & prev.eq(c1)].cpu())
        for k, v in num.items():
            v = torch.cat(v); res[f"p_fraud_{k}"] = round(float(v.mean()), 5); res[f"n_{k}"] = int(v.numel())
    assert digest(ws.model_tabular_weights_path) == before; res["weights_unchanged"] = True
    DOCS.mkdir(parents=True, exist_ok=True); write(DOCS / f"PROBE_{arm}_{fit_seed}.json", res); print(json.dumps(res), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("command", choices=["prepare", "fit", "sample", "generate", "probe"])
    p.add_argument("--arm", default="oversample_x8"); p.add_argument("--seed", type=int, default=CFG["fit_seeds"][0]); a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    assert importlib.metadata.version("mostlyai-engine") == CFG["engine_version"]
    if a.command == "prepare":
        prepare()
    elif a.command == "fit":
        fit(a.arm, a.seed)
    elif a.command == "sample":
        sample(a.arm, a.seed)
    elif a.command == "generate":
        assert a.arm == "rebalance"; rebalance(a.seed)
    else:
        probe(a.arm, a.seed)
