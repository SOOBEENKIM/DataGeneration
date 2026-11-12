"""ibm_tabformer_v1: native ARGN (mostlyai-engine 1.0.4) on the IBM/TabFormer 2013 window.

Mirrors scripts/run_sparkov_argn_control.py (prepare / fit / sample) with the registered IBM choices
in configs/ibm_tabformer_v1.json. The sealed test split is written once by `prepare` and never read.
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_tabformer_argn.py prepare
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_tabformer_argn.py fit --arm parent
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_tabformer_argn.py fit --arm native --seed 20261108
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, logging, os, random, shutil, sys, time
from contextlib import contextmanager, nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/ibm_tabformer_v1"
DOCS = ROOT / "docs/ibm_tabformer_v1"
RAW = OUT / "raw_events.parquet"           # written by scripts/ibm_tabformer_profile.py
CONFIG = ROOT / "configs/ibm_tabformer_v1.json"
for k, v in {"HF_HOME": str(OUT / "cache/hf"), "JOBLIB_TEMP_FOLDER": str(OUT / "cache/joblib"), "LOKY_MAX_CPU_COUNT": "4",
             "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(k, v)
import numpy as np
import pandas as pd
import torch

CFG = json.loads(CONFIG.read_text())
CORE = CFG["core_columns"]; STATIC = CFG["static_columns"]
MERCHANT, AMOUNT, LABEL = "receiver_or_mark", "amount_or_numeric_value", "event_is_fraud"
CATEGORICAL = {MERCHANT, "category", "use_chip", LABEL}


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")


def seed(n):
    random.seed(n); np.random.seed(n); torch.manual_seed(n); torch.cuda.manual_seed_all(n); torch.set_num_threads(CFG["cpu_threads"])


def native_frame(raw):
    d = raw.rename(columns={"customer_id": "entity_id"}).copy()
    d["event_index"] = d.groupby("entity_id", sort=False).cumcount()
    d.loc[d.event_index.eq(0), "gap"] = np.nan
    return d[["entity_id", "event_index", *CORE]]


def prepare():
    from mostlyai.engine import split, analyze, encode
    from mostlyai.engine._workspace import Workspace
    base = OUT / "prepared"; base.mkdir(parents=True, exist_ok=False)
    year = CFG["window_year"]
    raw = pd.read_parquet(RAW, columns=["user", "card", "year", "timestamp", "amount", "use_chip", "merchant", "state", "mcc", "is_fraud"])
    before = raw[raw.year.lt(year)]; win = raw[raw.year.eq(year)].copy()  # later years are never used
    del raw
    # window customers and sequences
    n_win = win.groupby("user").size()
    eligible = np.sort(n_win[n_win.ge(CFG["min_events_in_window"])].index.to_numpy())
    excluded = n_win[n_win.lt(CFG["min_events_in_window"])].rename("events_in_window").reset_index()
    excluded.to_csv(base / "excluded_customers.csv", index=False)
    rng = np.random.default_rng(CFG["split_seed"]); perm = rng.permutation(eligible)
    nt, nd = CFG["train_customers"], CFG["development_customers"]
    roles = pd.DataFrame(dict(user=perm, split=["train"] * nt + ["validation"] * nd + ["test"] * (len(perm) - nt - nd)))
    roles = roles.sort_values("user").reset_index(drop=True); roles["customer_id"] = np.arange(len(roles), dtype=np.int64)
    win = win[win.user.isin(eligible)].sort_values(["user", "timestamp"], kind="stable")
    win["event_index"] = win.groupby("user", sort=False).cumcount()
    win["gap"] = win.groupby("user", sort=False).timestamp.diff()
    ev = pd.DataFrame(dict(user=win.user.to_numpy(), event_index=win.event_index.to_numpy(), gap=win.gap.to_numpy()))
    ev[MERCHANT] = win.merchant.astype(str).to_numpy(); ev[AMOUNT] = win.amount.astype(float).to_numpy()
    ev["category"] = win.mcc.astype(int).astype(str).to_numpy(); ev["use_chip"] = win.use_chip.astype(str).to_numpy()
    ev[LABEL] = win.is_fraud.astype(int).astype(str).to_numpy()
    ev = ev.merge(roles, on="user", validate="many_to_one").sort_values(["customer_id", "event_index"], kind="stable")
    # static context from strictly pre-window history
    b = before[before.user.isin(eligible)]
    g = b.groupby("user")
    ctx = pd.DataFrame(dict(first_year=g.year.min(), cards_before=g.card.nunique(), events_before=g.size(),
                            home_state=g.state.agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else "NONE")))
    ctx = ctx.reindex(eligible)
    ctx["first_year"] = ctx.first_year.fillna(year).astype(int)
    ctx["cards_before"] = ctx.cards_before.fillna(0).astype(int)
    ctx["events_per_year_before"] = (ctx.events_before.fillna(0) / (year - ctx.first_year).clip(lower=1)).astype(float)
    ctx["home_state"] = ctx.home_state.fillna("NONE").astype(str)
    ctx = ctx.drop(columns=["events_before"]).reset_index().rename(columns={"index": "user"})
    ctx = ctx.merge(roles, on="user", validate="one_to_one")
    roles.to_parquet(base / "roles.parquet", index=False)
    for role in ["train", "validation", "test"]:
        ev[ev.split.eq(role)].drop(columns=["user", "split"]).rename(columns={"customer_id": "entity_id"})[["entity_id", "event_index", *CORE]].to_parquet(base / f"{role}.parquet", index=False)
        ctx[ctx.split.eq(role)][["customer_id", *STATIC]].to_parquet(base / f"{role}_context.parquet", index=False)
    del ev, win, before, b
    train = pd.read_parquet(base / "train.parquet"); val = pd.read_parquet(base / "validation.parquet")
    parent = pd.read_parquet(base / "train_context.parquet")
    target = train.rename(columns={"entity_id": "customer_id"})[["customer_id", *CORE]].copy(); target.gap = target.gap.fillna(0)
    types = {c: "TABULAR_CATEGORICAL" if c in CATEGORICAL else "TABULAR_NUMERIC_AUTO" for c in CORE}
    ctypes = {c: "TABULAR_CATEGORICAL" if c == "home_state" else "TABULAR_NUMERIC_AUTO" for c in STATIC}
    seed(CFG["preparation_seed"])
    for name in ["parent", "child"]:
        wsdir = base / name
        if name == "parent":
            split(tgt_data=parent, tgt_primary_key="customer_id", tgt_encoding_types=ctypes, workspace_dir=wsdir)
        else:
            split(tgt_data=target, ctx_data=parent, tgt_context_key="customer_id", ctx_primary_key="customer_id",
                  tgt_encoding_types=types, ctx_encoding_types=ctypes, workspace_dir=wsdir)
        analyze(workspace_dir=wsdir); encode(workspace_dir=wsdir)
    stats = Workspace(base / "child").tgt_stats.read()
    assert set(stats["columns"][LABEL]["codes"]) >= {"0", "1"}
    tf = train[LABEL].eq("1"); vf = val[LABEL].eq("1")
    test_n = int(roles.split.eq("test").sum())
    write(DOCS / "preparation.json", dict(
        config_sha256=digest(CONFIG), protocol_sha256=digest(DOCS / "PROTOCOL.md"), raw_sha256=digest(RAW), window_year=year,
        eligible_customers=int(len(eligible)), excluded_customers=int(len(excluded)), test_customers_sealed=test_n, test_events_read=False,
        train_customers=int(train.entity_id.nunique()), train_events=int(len(train)), train_frauds=int(tf.sum()), train_fraud_customers=int(train[tf].entity_id.nunique()),
        validation_customers=int(val.entity_id.nunique()), validation_events=int(len(val)), validation_frauds=int(vf.sum()), validation_fraud_customers=int(val[vf].entity_id.nunique()),
        events_per_customer=dict(train_mean=float(train.groupby("entity_id").size().mean()), validation_mean=float(val.groupby("entity_id").size().mean())),
        target_columns={c: dict(encoding=stats["columns"][c]["encoding_type"], cardinality=len(stats["columns"][c].get("codes", []))) for c in CORE},
        context_columns=ctypes, inputs={p.name: digest(p) for p in sorted(base.glob("*.parquet"))}))
    print(json.dumps(json.loads((DOCS / "preparation.json").read_text()), indent=1)[:3000])


@contextmanager
def relaxed_cap(folder):
    """Amendment 1 (2026-11-08): the cap-only control of sparkov_argn_control_v2, unchanged. Only the internal
    `max_epochs_cap = ceil(n/50)` assignment is replaced in an isolated source copy; the installed engine is untouched."""
    import difflib, inspect
    import mostlyai.engine._tabular.training as module
    source = inspect.getsource(module)
    old = "max_epochs_cap = math.ceil((trn_cnt + val_cnt) / 50)"
    assert source.count(old) == 1
    modified = source.replace(old, "max_epochs_cap = max_epochs  # registered cap-only control")
    path = folder / "training_cap_control.py"; path.write_text(modified)
    (folder / "training_cap_control.diff").write_text("".join(difflib.unified_diff(source.splitlines(True), modified.splitlines(True), fromfile="official_training.py", tofile=path.name)))
    write(folder / "source_control.json", dict(official_sha256=hashlib.sha256(source.encode()).hexdigest(), controlled_sha256=digest(path), installed_source_unchanged=True))
    namespace = dict(module.__dict__); exec(compile(modified, str(path), "exec"), namespace)
    original = module.train; module.train = namespace["train"]
    try:
        yield
    finally:
        module.train = original


def fit(arm, fit_seed):
    from mostlyai.engine import train, generate
    from mostlyai.engine._workspace import Workspace
    folder = OUT / "runs" / f"{arm}_{fit_seed}"; folder.mkdir(parents=True, exist_ok=False)
    wsdir = folder / "workspace"; shutil.copytree(OUT / "prepared" / ("parent" if arm == "parent" else "child"), wsdir)
    device = CFG["device"] if torch.cuda.is_available() else "cpu"
    seed(fit_seed); start = time.monotonic()
    write(folder / "START.json", dict(arm=arm, fit_seed=fit_seed, engine=importlib.metadata.version("mostlyai-engine"), device=device,
                                      gpu=torch.cuda.get_device_name(0) if device == "cuda" else None, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                      config_sha256=digest(CONFIG), script_sha256=digest(__file__), max_training_minutes=CFG["max_training_minutes"]))
    with relaxed_cap(folder) if arm == "relaxed" else nullcontext():
        train(max_training_time=CFG["max_training_minutes"], device=device, workspace_dir=wsdir)
    ws = Workspace(wsdir)
    write(folder / "FIT.json", dict(seconds=time.monotonic() - start, weights_sha256=digest(ws.model_tabular_weights_path), device=device))
    if arm == "parent":
        seed(CFG["generation_seeds"][0]); generate(sample_size=CFG["development_customers"], device=device, workspace_dir=wsdir)
        d = pd.read_parquet(wsdir / "SyntheticData"); d.customer_id = np.arange(len(d), dtype=np.int64) + 10000
        d.to_parquet(OUT / "prepared/synthetic_context.parquet", index=False)
    else:
        sample(arm, fit_seed)


def sample(arm, fit_seed):
    from mostlyai.engine import generate
    from mostlyai.engine._workspace import Workspace
    folder = OUT / "runs" / f"{arm}_{fit_seed}"; wsdir = folder / "workspace"; ws = Workspace(wsdir)
    device = CFG["device"] if torch.cuda.is_available() else "cpu"
    weights = digest(ws.model_tabular_weights_path)
    for population in ["validation"]:
        ctx = pd.read_parquet(OUT / f"prepared/{population}_context.parquet")
        for gs in CFG["generation_seeds"]:
            dest = folder / f"generated_{population}_{gs}.parquet"
            if dest.exists():
                continue
            seed(gs); start = time.monotonic(); generate(ctx_data=ctx, device=device, workspace_dir=wsdir)
            raw = pd.read_parquet(wsdir / "SyntheticData"); raw.to_parquet(folder / f"native_{population}_{gs}.parquet", index=False)
            d = native_frame(raw); d.to_parquet(dest, index=False)
            assert digest(ws.model_tabular_weights_path) == weights
            write(folder / f"generation_{population}_{gs}.json", dict(seconds=time.monotonic() - start, events=len(d), customers=int(d.entity_id.nunique()),
                                                                     frauds=int(pd.to_numeric(d[LABEL], errors="coerce").eq(1).sum()), weights_unchanged=True, sha256=digest(dest)))
            print(f"GENERATED {arm} {fit_seed} {population} {gs}: {len(d)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("command", choices=["prepare", "fit", "sample"])
    p.add_argument("--arm", choices=["parent", "native", "relaxed"], default="native"); p.add_argument("--seed", type=int, default=CFG["fit_seeds"][0]); a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    assert importlib.metadata.version("mostlyai-engine") == CFG["engine_version"]
    if a.command == "prepare":
        prepare()
    elif a.command == "fit":
        fit(a.arm, a.seed)
    else:
        sample(a.arm, a.seed)
