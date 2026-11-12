"""ibm_state_first_v1: State-First generation on the frozen IBM native ARGN fits (docs/ibm_state_first_v1/PROTOCOL.md).
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_state_first.py fit_state
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_state_first.py check --fit 20261108
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_state_first.py train_heads --fit 20261108
    CUDA_VISIBLE_DEVICES=1 ./run python scripts/run_ibm_state_first.py generate --fit 20261108 --arms learned_r2 table_r3 learned_r3
Whole-model fits: 0. Development labels are used only for the teacher-forced log-loss report; test is never read.
"""
import argparse, hashlib, json, logging, os, pickle, random, sys, time
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
OUT = ROOT / "artifacts/ibm_state_first_v1"
DOCS = ROOT / "docs/ibm_state_first_v1"
STATE_PKL = OUT / "state_process.pkl"
CORE = ["gap", "receiver_or_mark", "amount_or_numeric_value", "category", "use_chip", "event_is_fraud"]
LABEL = "event_is_fraud"
ARMS = {"learned_r2": dict(state="learned", heads="r2"), "table_r3": dict(state="table", heads="r3"), "learned_r3": dict(state="learned", heads="r3")}
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


def load_state_fn(kind):
    from benchmarks.state_process_features import table_prob_fn
    if kind == "table":
        return table_prob_fn(json.loads(TABLE.read_text())["table"]), digest(TABLE)
    clf = pickle.load(open(STATE_PKL, "rb"))
    return (lambda X: clf.predict_proba(X)[:, 1]), digest(STATE_PKL)


def fit_state():
    from benchmarks.argn_state_first import fit_state_process, label_logloss
    from benchmarks.state_process_features import table_prob_fn, FEATURES
    OUT.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(NATIVE / "prepared/train.parquet"); dev = pd.read_parquet(NATIVE / "prepared/validation.parquet")
    t0 = time.time(); clf, meta = fit_state_process(train); secs = time.time() - t0
    pickle.dump(clf, open(STATE_PKL, "wb"))
    learned = lambda X: clf.predict_proba(X)[:, 1]; table = table_prob_fn(json.loads(TABLE.read_text())["table"])
    report = dict(meta, seconds=round(secs, 1), state_process_sha256=digest(STATE_PKL), train_sha256=digest(NATIVE / "prepared/train.parquet"),
                  teacher_forced_logloss=dict(development={"learned": label_logloss(dev, learned), "table": label_logloss(dev, table)},
                                              train={"learned": label_logloss(train, learned), "table": label_logloss(train, table)}),
                  note="development log-loss is reported, not used for selection; test not read")
    # permutation importance on development labels (teacher-forced), cheap and model-agnostic
    from benchmarks.state_process_features import label_history_features
    d = dev.sort_values(["entity_id", "event_index"], kind="stable").reset_index(drop=True)
    y = pd.to_numeric(d[LABEL], errors="coerce").fillna(0).astype(int).eq(1).astype(int).to_numpy(); X = label_history_features(pd.Series(y), d.entity_id)
    base = label_logloss(dev, learned)["logloss"]; rng = np.random.default_rng(20261109); imp = {}
    for j, f in enumerate(FEATURES):
        Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); p = np.clip(clf.predict_proba(Xp)[:, 1], 1e-6, 1 - 1e-6)
        imp[f] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean() - base)
    report["permutation_importance_logloss_increase"] = imp
    write(DOCS / "STATE_PROCESS.json", report); print(json.dumps(report, indent=1))


def check(fs):
    """Identity: untrained state heads reproduce the native outputs exactly on a validation batch (teacher mode)."""
    from mostlyai.engine._workspace import Workspace
    from mostlyai.engine._tabular.training import BatchCollator
    from torch.utils.data import DataLoader
    from benchmarks.argn_state_first import state_model
    from benchmarks.argn_output_heads import build_model, label_first_order, _records
    from mostlyai.engine._common import get_cardinalities
    ws = Workspace(NATIVE / "runs" / f"native_{fs}" / "workspace"); ts = ws.tgt_stats.read(); order = label_first_order(ts, get_cardinalities(ts))
    torch.manual_seed(0); a, _, _ = build_model(ws, "cuda", order); torch.manual_seed(0); b, _, _, _ = state_model(ws, "cuda"); a.eval(); b.eval()
    recs = _records(list(ws.encoded_data_val.fetch_all())[0])[:8]
    data = next(iter(DataLoader(recs, batch_size=8, collate_fn=BatchCollator(True, None, torch.device("cuda")))))
    with torch.no_grad():
        oa, _ = a(dict(data), mode="trn"); ob, _ = b(dict(data), mode="trn")
    diff = max(float((oa[k] - ob[k]).abs().max()) for k in oa); routes = b.route.squeeze(-1)
    res = dict(fit=fs, max_abs_diff=diff, identity=diff == 0.0, route_counts={int(i): int(routes.eq(i).sum()) for i in (0, 1, 2)})
    write(DOCS / f"CHECK_{fs}.json", res); print(res); assert diff == 0.0


def train_heads(fs):
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_state_first import train_state_heads
    ws = Workspace(NATIVE / "runs" / f"native_{fs}" / "workspace"); before = digest(ws.model_tabular_weights_path)
    folder = OUT / "heads" / f"r3_{fs}"
    meta = train_state_heads(ws, folder, fs, device="cuda" if torch.cuda.is_available() else "cpu", log=lambda s: print(s, flush=True))
    assert digest(ws.model_tabular_weights_path) == before
    meta.update(base_weights_sha256=before, heads_sha256=digest(folder / "heads.pt"), base_weights_unchanged=True)
    write(folder / "TRAIN.json", meta); DOCS.mkdir(parents=True, exist_ok=True)
    pd.read_csv(folder / "learning.csv").to_csv(DOCS / f"learning_r3_{fs}.csv", index=False); write(DOCS / f"train_r3_{fs}.json", meta)
    print(json.dumps({k: meta[k] for k in ["epochs", "best_epoch", "best_val", "val_rows_by_state", "trainable_parameters", "seconds"]}))


def generate(fs, arms):
    from mostlyai.engine import generate as engine_generate
    from mostlyai.engine._workspace import Workspace
    from benchmarks.argn_state_first import state_first_module, attach_state_heads
    from benchmarks.argn_routed_heads import attach_routed_heads
    wsdir = NATIVE / "runs" / f"native_{fs}" / "workspace"; ws = Workspace(wsdir); stats = ws.tgt_stats.read(); weights = digest(ws.model_tabular_weights_path)
    ctx = pd.read_parquet(NATIVE / "prepared/validation_context.parquet"); device = "cuda" if torch.cuda.is_available() else "cpu"
    for arm in arms:
        cfg = ARMS[arm]; p_fn, state_sha = load_state_fn(cfg["state"])
        if cfg["heads"] == "r2":
            hp = ROUTED_HEADS / f"routed_subst_{fs}" / "heads.pt"; attach = lambda m: attach_routed_heads(m, stats)
        else:
            hp = OUT / "heads" / f"r3_{fs}" / "heads.pt"; attach = lambda m: attach_state_heads(m, stats)
        head_state = torch.load(hp, map_location="cpu"); folder = OUT / "runs" / f"{arm}_{fs}"; folder.mkdir(parents=True, exist_ok=True)
        for gs in GEN_SEEDS:
            dest = folder / f"generated_validation_{gs}.parquet"
            if dest.exists():
                continue
            seed(gs); t0 = time.monotonic()
            with state_first_module(stats, p_fn=p_fn, attach=attach, head_state=head_state) as info:
                engine_generate(ctx_data=ctx, device=device, workspace_dir=wsdir)
            raw = pd.read_parquet(wsdir / "SyntheticData"); raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
            d = raw.rename(columns={"customer_id": "entity_id"}).copy(); d["event_index"] = d.groupby("entity_id", sort=False).cumcount()
            d.loc[d.event_index.eq(0), "gap"] = np.nan; d = d[["entity_id", "event_index", *CORE]]; d.to_parquet(dest, index=False)
            assert digest(ws.model_tabular_weights_path) == weights
            lab = d[LABEL].astype(str).eq("1")
            write(folder / f"generation_validation_{gs}.json", dict(arm=arm, fit_seed=fs, generation_seed=gs, state=cfg["state"], state_sha256=state_sha, heads=cfg["heads"],
                                                                   heads_sha256=digest(hp), module=info, seconds=time.monotonic() - t0, events=len(d), customers=int(d.entity_id.nunique()),
                                                                   frauds=int(lab.sum()), fraud_customers=int(d.loc[lab, "entity_id"].nunique()), base_weights_unchanged=True, sha256=digest(dest)))
            print(f"GENERATED {arm}_{fs} {gs}: events {len(d)} frauds {int(lab.sum())} fraud_customers {int(d.loc[lab, 'entity_id'].nunique())} {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["fit_state", "check", "train_heads", "generate"]); ap.add_argument("--fit", type=int, default=20261108)
    ap.add_argument("--arms", nargs="*", default=list(ARMS)); a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"fit_state": lambda: fit_state(), "check": lambda: check(a.fit), "train_heads": lambda: train_heads(a.fit), "generate": lambda: generate(a.fit, a.arms)}[a.command]()
