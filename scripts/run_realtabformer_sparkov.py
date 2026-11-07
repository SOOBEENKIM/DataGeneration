"""REaLTabFormer on Sparkov 60-row windows (protocol: docs/realtabformer_sparkov_v1/PROTOCOL.md).

Subcommands: prepare | fit | generate [--seeds 20261011 20261012]
Run on the server with the realtabformer venv python and CUDA_VISIBLE_DEVICES=1.
Each subcommand is idempotent (skips work whose output already exists).
"""
import argparse
import json
import math
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path("/home/ssd-990/soobeenkim/DataGeneration/workspace")
SRC = WS / "research-argn-fraud-audit/artifacts/sparkov_argn_control_v2/prepared"
OUT = WS / "research-argn-state-first/artifacts/realtabformer_sparkov_v1"
PREP, FIT, RUNS, LOGS, REP = (OUT / d for d in ("prepared", "fit", "runs", "logs", "report"))

WINDOW = 60
FIT_SEED = 20261107
GEN_SEEDS = [20261011, 20261012]
BUDGET_S = 8 * 3600
MAX_EPOCHS = 40
CHILD_COLS = ["window_id", "gap", "receiver_or_mark", "amount_or_numeric_value", "category", "event_is_fraud"]
PARENT_STATIC = ["cardholder_gender", "cardholder_state", "birth_year", "city_population"]


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def mkdirs():
    for d in (PREP, FIT, RUNS, LOGS, REP):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- prepare
def build_windows(split):
    ev = pd.read_parquet(SRC / f"{split}.parquet").sort_values(["entity_id", "event_index"]).reset_index(drop=True)
    ctx = pd.read_parquet(SRC / f"{split}_context.parquet").set_index("customer_id")
    ev["pos"] = ev.groupby("entity_id").cumcount()
    ev["window_index"] = ev["pos"] // WINDOW
    keys = ev[["entity_id", "window_index"]].drop_duplicates().reset_index(drop=True)
    keys["window_id"] = np.arange(len(keys))
    ev = ev.merge(keys, on=["entity_id", "window_index"], how="left")
    ev["gap_i"] = ev["gap"].fillna(0).round().astype("int64")
    child = pd.DataFrame({
        "window_id": ev["window_id"].astype("int64"),
        "gap": ev["gap_i"],
        "receiver_or_mark": ev["receiver_or_mark"].astype(str),
        "amount_or_numeric_value": ev["amount_or_numeric_value"].astype(float).round(2),
        "category": ev["category"].astype(str),
        "event_is_fraud": ev["event_is_fraud"].astype(int),
    })[CHILD_COLS]
    n = ev.groupby("window_id").size().rename("window_rows")
    parent = keys.merge(n, left_on="window_id", right_index=True)
    parent = pd.concat([parent.reset_index(drop=True),
                        ctx.loc[parent["entity_id"], PARENT_STATIC].reset_index(drop=True)], axis=1)
    wmap = parent[["window_id", "entity_id", "window_index", "window_rows"]].copy()
    first_ts = ev[ev["event_index"] == ev.groupby("entity_id")["event_index"].transform("min")] \
        .set_index("entity_id")["timestamp"]
    wmap["first_timestamp"] = wmap["entity_id"].map(first_ts)
    parent = parent[["window_id"] + PARENT_STATIC + ["window_index", "window_rows"]]
    return child, parent, wmap, ev


def relational_dataset(child, parent, output_max_length):
    """Replicates REaLTabFormer._fit_relational preprocessing; returns (dataset, drop-count)."""
    from realtabformer import REaLTabFormer
    from realtabformer.data_utils import make_relational_dataset, process_data
    m = REaLTabFormer(model_type="relational", output_max_length=output_max_length,
                      checkpoints_dir=str(FIT / "tmp_measure"), numeric_nparts=2)
    jn = "window_id"
    cidx = child.reset_index(drop=True).groupby(jn).apply(lambda x: x.index.to_list())
    in_out = pd.Series(dict(parent[jn].reset_index(drop=True).items())).map(lambda x: cidx.get(x, []))
    pin, pout = parent.drop(jn, axis=1), child.drop(jn, axis=1)
    m._extract_column_info(pout)
    pout, _ = process_data(pout, numeric_max_len=m.numeric_max_len, numeric_precision=m.numeric_precision,
                           numeric_nparts=m.numeric_nparts)
    vocab = {"decoder": m._generate_vocab(pout)}
    pin, _ = process_data(pin, numeric_max_len=m.numeric_max_len, numeric_precision=m.numeric_precision,
                          numeric_nparts=m.numeric_nparts, col_transform_data=m.parent_col_transform_data)
    vocab["encoder"] = m._generate_vocab(pin)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ds = make_relational_dataset(in_df=pin, out_df=pout, vocab=vocab, in_out_idx=in_out,
                                     output_max_length=output_max_length, return_token_type_ids=False)
    return ds


def cmd_prepare(args):
    mkdirs()
    if (REP / "PREPARE.json").exists():
        print("PREPARE.json exists; skipping")
        return
    info = {}
    maxlen = 0
    for split in ("train", "validation"):
        child, parent, wmap, ev = build_windows(split)
        tag = "train" if split == "train" else "development"
        child.to_parquet(PREP / f"{tag}_child.parquet", index=False)
        parent.to_parquet(PREP / f"{tag}_parent.parquet", index=False)
        wmap.to_parquet(PREP / f"{tag}_window_map.parquet", index=False)
        t0 = time.time()
        ds = relational_dataset(child, parent, None)
        lens = np.array(ds.map(lambda e: dict(L=len(e["labels"])))["L"])
        # relational_max_length in the package = max(len(labels)) + 1
        info[tag] = dict(customers=int(wmap.entity_id.nunique()), windows=int(len(parent)), rows=int(len(child)),
                         fraud_rate=float(child.event_is_fraud.mean()),
                         max_window_rows=int(parent.window_rows.max()),
                         max_label_tokens=int(lens.max()), mean_label_tokens=float(lens.mean()),
                         measure_seconds=round(time.time() - t0, 1))
        maxlen = max(maxlen, int(lens.max()))
        print(tag, info[tag], flush=True)
    info["longest_child_window_tokens"] = maxlen
    info["output_max_length"] = maxlen + 16
    info["window"] = WINDOW
    jdump(info, REP / "PREPARE.json")


# ---------------------------------------------------------------- fit
def make_model(epochs, ckpt, omax):
    from realtabformer import REaLTabFormer
    return REaLTabFormer(model_type="relational", numeric_nparts=2, output_max_length=omax, epochs=epochs,
                         batch_size=8, train_size=0.9, early_stopping_patience=5, random_state=FIT_SEED,
                         checkpoints_dir=str(ckpt), samples_save_dir=str(FIT / "samples"),
                         full_save_dir=str(FIT / "full"))


def run_fit(epochs, ckpt, omax):
    child = pd.read_parquet(PREP / "train_child.parquet")
    parent = pd.read_parquet(PREP / "train_parent.parquet")
    m = make_model(epochs, ckpt, omax)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        trainer = m.fit(df=child, in_df=parent, join_on="window_id", device="cuda")
    dropped = 0
    for x in w:
        s = str(x.message)
        if "has been removed from the training data" in s:
            dropped = int(s.split("A total of ")[1].split(" out of")[0])
    return m, trainer, time.time() - t0, dropped, len(parent)


def save_model(m):
    """Package save() fails on Path attributes other than checkpoints_dir/samples_save_dir."""
    for k, v in list(m.__dict__.items()):
        if isinstance(v, Path) and k not in ("checkpoints_dir", "samples_save_dir"):
            setattr(m, k, v.as_posix())
    m.save(FIT / "model", allow_overwrite=True)


def cmd_recover(args):
    """2026-11-08 amendment: the original fit finished (early stopped) but m.save() crashed on a Path attribute
    (TypeError), losing the in-memory model. Rebuild the identical model object WITHOUT retraining, load the
    best checkpoint weights chosen by the original run, and save. No training conditions changed."""
    import torch
    from safetensors.torch import load_file
    mkdirs()
    omax = json.loads((REP / "PREPARE.json").read_text())["output_max_length"]
    timing = json.loads((REP / "TIMING.json").read_text())
    cap = timing["epoch_cap"]
    ck = FIT / "ckpt_main"
    last = sorted(ck.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))[-1]
    st = json.loads((last / "trainer_state.json").read_text())
    best = Path(st["best_model_checkpoint"])
    child = pd.read_parquet(PREP / "train_child.parquet")
    parent = pd.read_parquet(PREP / "train_parent.parquet")
    m = make_model(cap, ck, omax)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m._fit_relational(child, parent, join_on="window_id", device="cuda")  # builds model/vocab; no training
    dropped = 0
    for x in w:
        if "has been removed from the training data" in str(x.message):
            dropped = int(str(x.message).split("A total of ")[1].split(" out of")[0])
    sd = load_file(str(best / "model.safetensors"))
    missing = m.model.load_state_dict(sd, strict=False)
    print("load_state_dict:", missing, flush=True)
    m.model.eval()
    m.experiment_id = f"id{int((time.time() * 10**10)):024}"
    save_model(m)
    hist = st["log_history"]
    train_l = [h for h in hist if "loss" in h and "eval_loss" not in h]
    eval_l = [h for h in hist if "eval_loss" in h]
    t_start = (REP / "TIMING.json").stat().st_mtime
    t_end = last.stat().st_mtime
    jdump(dict(epoch_cap=cap, epochs_completed=float(st["epoch"]), global_step=st["global_step"],
               stopped_early=bool(st["epoch"] < cap - 1e-6), seconds_approx=t_end - t_start,
               seconds_note="approx: TIMING.json mtime to last checkpoint mtime (includes data prep and evals)",
               parents_dropped=dropped, parents_total=len(parent), output_max_length=omax,
               best_metric=st["best_metric"], best_model_checkpoint=str(best),
               final_train_loss=train_l[-1]["loss"], final_eval_loss=eval_l[-1]["eval_loss"],
               best_eval_loss=min(h["eval_loss"] for h in eval_l),
               experiment_id=m.experiment_id, model_dir=str(FIT / "model" / m.experiment_id),
               recovered_after_save_crash=True, load_state_dict_missing=list(missing.missing_keys),
               log_history=hist),
          REP / "FIT.json")


def cmd_fit(args):
    if args.recover:
        return cmd_recover(args)
    mkdirs()
    if (REP / "FIT.json").exists():
        print("FIT.json exists; skipping")
        return
    omax = json.loads((REP / "PREPARE.json").read_text())["output_max_length"]
    if not (REP / "TIMING.json").exists():
        print("== timing run: epochs=1", flush=True)
        m, tr, secs, dropped, npar = run_fit(1, FIT / "ckpt_timing", omax)
        steps = tr.state.global_step
        del m, tr
        import torch
        torch.cuda.empty_cache()
        cap = int(max(1, min(MAX_EPOCHS, math.floor(BUDGET_S / secs))))
        jdump(dict(epoch_seconds=secs, global_steps_one_epoch=steps, budget_seconds=BUDGET_S,
                   max_epochs=MAX_EPOCHS, epoch_cap=cap, output_max_length=omax,
                   parents_dropped=dropped, parents_total=npar,
                   started_unix=time.time(), expected_finish_unix_upper_bound=time.time() + secs * cap,
                   note="epoch_seconds includes model build/tokenization overhead and periodic eval/checkpoints"),
              REP / "TIMING.json")
    timing = json.loads((REP / "TIMING.json").read_text())
    cap = timing["epoch_cap"]
    print(f"== real fit: epochs cap={cap}", flush=True)
    t0 = time.time()
    m, tr, secs, dropped, npar = run_fit(cap, FIT / "ckpt_main", omax)
    hist = tr.state.log_history
    train_l = [h for h in hist if "loss" in h and "eval_loss" not in h]
    eval_l = [h for h in hist if "eval_loss" in h]
    save_model(m)
    jdump(dict(epoch_cap=cap, epochs_completed=float(tr.state.epoch), global_step=tr.state.global_step,
               stopped_early=bool(tr.state.epoch < cap - 1e-6), seconds=secs,
               parents_dropped=dropped, parents_total=npar, output_max_length=omax,
               best_metric=tr.state.best_metric, best_model_checkpoint=tr.state.best_model_checkpoint,
               final_train_loss=train_l[-1]["loss"] if train_l else None,
               final_eval_loss=eval_l[-1]["eval_loss"] if eval_l else None,
               best_eval_loss=min(h["eval_loss"] for h in eval_l) if eval_l else None,
               experiment_id=m.experiment_id, model_dir=str(FIT / "model" / m.experiment_id),
               log_history=hist),
          REP / "FIT.json")


# ---------------------------------------------------------------- generate
def seed_all(s):
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def cmd_generate(args):
    mkdirs()
    from realtabformer import REaLTabFormer
    fit = json.loads((REP / "FIT.json").read_text())
    model = None
    parent = pd.read_parquet(PREP / "development_parent.parquet")
    wmap = pd.read_parquet(PREP / "development_window_map.parquet")
    real = pd.read_parquet(SRC / "validation.parquet")
    for seed in args.seeds:
        out_path = RUNS / f"generated_development_{seed}.parquet"
        if out_path.exists() and (REP / f"GENERATE_{seed}.json").exists():
            print("exists; skipping", seed)
            continue
        if model is None:
            model = REaLTabFormer.load_from_dir(Path(fit["model_dir"]))
        seed_all(seed)
        t0 = time.time()
        in_df = parent.drop(columns="window_id").copy()
        syn = model.sample(input_unique_ids=parent["window_id"].copy(), input_df=in_df, related_num="window_rows",
                           gen_batch=args.gen_batch, device="cuda")
        secs = time.time() - t0
        syn = syn.copy()
        syn["window_id"] = syn.index.astype("int64")
        counts = syn.groupby("window_id").size()
        want = wmap.set_index("window_id")["window_rows"]
        got = counts.reindex(want.index).fillna(0).astype(int)
        mism = int((got != want).sum())
        syn = syn.merge(wmap[["window_id", "entity_id", "window_index", "first_timestamp"]], on="window_id")
        syn["_ord"] = syn.groupby("window_id").cumcount()
        syn = syn[syn["_ord"] < syn["window_id"].map(want)]  # truncate any surplus (recorded via mismatch)
        syn = syn.sort_values(["entity_id", "window_index", "_ord"], kind="stable").reset_index(drop=True)
        syn["event_index"] = syn.groupby("entity_id").cumcount().astype("int64")
        g = pd.to_numeric(syn["gap"], errors="coerce").fillna(0).clip(lower=0).round().astype(float)
        syn["gap"] = g.where(syn["event_index"] > 0, np.nan)
        syn["timestamp"] = syn["first_timestamp"] + g.where(syn["event_index"] > 0, 0).groupby(syn["entity_id"]).cumsum()
        syn["event_is_fraud"] = pd.to_numeric(syn["event_is_fraud"], errors="coerce").fillna(0).clip(0, 1) \
            .astype(int).astype(str).astype(object)
        syn["receiver_or_mark"] = syn["receiver_or_mark"].astype(str).astype(real["receiver_or_mark"].dtype)
        syn["category"] = syn["category"].astype(str).astype(real["category"].dtype)
        syn["amount_or_numeric_value"] = pd.to_numeric(syn["amount_or_numeric_value"], errors="coerce").astype(float)
        syn["entity_id"] = syn["entity_id"].astype("int64")
        res = syn[list(real.columns)].astype({c: real[c].dtype for c in real.columns if c != "event_is_fraud"})
        res.to_parquet(out_path, index=False)
        jdump(dict(seed=seed, rows=int(len(res)), real_rows=int(len(real)), customers=int(res.entity_id.nunique()),
                   windows=int(len(wmap)), windows_row_count_mismatch=mism,
                   fraud_rate=float((res.event_is_fraud == "1").mean()), seconds=secs,
                   output=str(out_path), conditioned_on_real_window_lengths=True),
              REP / f"GENERATE_{seed}.json")
        print("generated", seed, len(res), mism, secs, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prepare")
    f = sp.add_parser("fit")
    f.add_argument("--recover", action="store_true")
    g = sp.add_parser("generate")
    g.add_argument("--seeds", type=int, nargs="+", default=GEN_SEEDS)
    g.add_argument("--gen_batch", type=int, default=64)
    a = ap.parse_args()
    {"prepare": cmd_prepare, "fit": cmd_fit, "generate": cmd_generate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
