"""REaLTabFormer on IBM 2013 with the two module components (protocol: docs/ibm_realtabformer_v1/PROTOCOL.md).

Run on the server with the realtabformer venv python (NOT ./run):
  environments/realtabformer/bin/python <abs>/scripts/run_realtabformer_ibm.py prepare
  CUDA_VISIBLE_DEVICES=1 ... fit --seed 20261108            (timing epoch, then the real fit; --recover after a save crash)
  ... check --source sparkov                                  (mechanism check on the Sparkov fit before IBM fits exist)
  ... check --source ibm --fit 20261108                       (identity + routing check on the IBM fit)
  ... train_heads --fit 20261108 [--subst]
  ... generate --fit 20261108 --arm rtf_native|rtf_table|rtf_table_routed|rtf_table_routed_subst [--seeds ...]
  ... probe --fit 20261108 ; ... evaluate --fit 20261108
Each subcommand is idempotent (skips outputs that exist). development/test customers are never used for fitting.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from realtabformer.data_utils import SPECIAL_COL_SEP

WS = Path("/home/ssd-990/soobeenkim/DataGeneration/workspace/research-argn-state-first")
sys.path.insert(0, str(WS))
SRC = WS / "artifacts/ibm_tabformer_v1/prepared"
TABLE = WS / "artifacts/ibm_generation_module_v1/transition_table.json"
SPARKOV = WS / "artifacts/realtabformer_sparkov_v1"
OUT = WS / "artifacts/ibm_realtabformer_v1"
PREP, FIT, HEADS, RUNS, LOGS, REP = (OUT / d for d in ("prepared", "fit", "heads", "runs", "logs", "report"))

FIT_SEEDS = [20261108, 20261109]
GEN_SEEDS = [20261011, 20261012]
WINDOW_CANDIDATES = [60, 50, 40]
DECODER_LIMIT = 1024
BUDGET_S = 5 * 3600
MAX_EPOCHS = 40
LABEL = "event_is_fraud"
CHILD_COLS = ["window_id", LABEL, "gap", "receiver_or_mark", "amount_or_numeric_value", "category", "use_chip"]  # label first
PARENT_STATIC = ["first_year", "cards_before", "events_per_year_before", "home_state"]
ARMS = {"rtf_native": dict(table=False, heads=None), "rtf_table": dict(table=True, heads=None),
        "rtf_table_routed": dict(table=True, heads="routed"), "rtf_table_routed_subst": dict(table=True, heads="routed_subst"),
        "rtf_learned": dict(table=False, heads=None, learned=True)}  # ibm_state_first_v1 (2026-11-09): learned state process, no heads
STATE_PKL = WS / "artifacts/ibm_state_first_v1/state_process.pkl"


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


def mkdirs():
    for d in (PREP, FIT, HEADS, RUNS, LOGS, REP):
        d.mkdir(parents=True, exist_ok=True)


def seed_all(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def digest_state(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- prepare
def build_windows(split, window):
    ev = pd.read_parquet(SRC / f"{split}.parquet").sort_values(["entity_id", "event_index"]).reset_index(drop=True)
    ctx = pd.read_parquet(SRC / f"{split}_context.parquet").set_index("customer_id")
    ev["pos"] = ev.groupby("entity_id").cumcount()
    ev["window_index"] = ev["pos"] // window
    keys = ev[["entity_id", "window_index"]].drop_duplicates().reset_index(drop=True)
    keys["window_id"] = np.arange(len(keys))
    ev = ev.merge(keys, on=["entity_id", "window_index"], how="left")
    child = pd.DataFrame({
        "window_id": ev["window_id"].astype("int64"),
        LABEL: ev[LABEL].astype(int),
        "gap": ev["gap"].fillna(0).round().astype("int64"),
        "receiver_or_mark": ev["receiver_or_mark"].astype(str),
        "amount_or_numeric_value": ev["amount_or_numeric_value"].astype(float).round(2),
        "category": ev["category"].astype(str),
        "use_chip": ev["use_chip"].astype(str),
    })[CHILD_COLS]
    n = ev.groupby("window_id").size().rename("window_rows")
    parent = keys.merge(n, left_on="window_id", right_index=True)
    parent = pd.concat([parent.reset_index(drop=True), ctx.loc[parent["entity_id"], PARENT_STATIC].reset_index(drop=True)], axis=1)
    wmap = parent[["window_id", "entity_id", "window_index", "window_rows"]].copy()
    parent = parent[["window_id"] + PARENT_STATIC + ["window_index", "window_rows"]]
    return child, parent, wmap


def relational_dataset(child, parent, output_max_length, model=None):
    """Replicates REaLTabFormer._fit_relational preprocessing. With `model`, uses the fitted vocab/transforms."""
    from realtabformer import REaLTabFormer
    from realtabformer.data_utils import make_relational_dataset, process_data
    jn = "window_id"
    cidx = child.reset_index(drop=True).groupby(jn).apply(lambda x: x.index.to_list())
    in_out = pd.Series(dict(parent[jn].reset_index(drop=True).items())).map(lambda x: cidx.get(x, []))
    pin, pout = parent.drop(jn, axis=1), child.drop(jn, axis=1)
    if model is None:
        m = REaLTabFormer(model_type="relational", output_max_length=output_max_length, checkpoints_dir=str(FIT / "tmp_measure"), numeric_nparts=2)
        m._extract_column_info(pout)
        pout, _ = process_data(pout, numeric_max_len=m.numeric_max_len, numeric_precision=m.numeric_precision, numeric_nparts=m.numeric_nparts)
        vocab = {"decoder": m._generate_vocab(pout)}
        pin, _ = process_data(pin, numeric_max_len=m.numeric_max_len, numeric_precision=m.numeric_precision, numeric_nparts=m.numeric_nparts)
        vocab["encoder"] = m._generate_vocab(pin)
    else:
        m = model
        pout, _ = process_data(pout, numeric_max_len=m.numeric_max_len, numeric_precision=m.numeric_precision, numeric_nparts=m.numeric_nparts, col_transform_data=m.col_transform_data)
        pin, _ = process_data(pin, numeric_max_len=m.numeric_max_len, numeric_precision=m.numeric_precision, numeric_nparts=m.numeric_nparts, col_transform_data=m.in_col_transform_data)
        assert list(pout.columns) == list(m.processed_columns), (list(pout.columns), m.processed_columns)
        vocab = m.vocab
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ds = make_relational_dataset(in_df=pin, out_df=pout, vocab=vocab, in_out_idx=in_out, output_max_length=output_max_length, return_token_type_ids=False)
    return ds


def cmd_prepare(args):
    mkdirs()
    if (REP / "PREPARE.json").exists():
        print("PREPARE.json exists; skipping"); return
    info = dict(candidates=WINDOW_CANDIDATES, decoder_limit=DECODER_LIMIT, measurements={})
    chosen = None
    for window in WINDOW_CANDIDATES:
        child, parent, wmap = build_windows("train", window)
        t0 = time.time()
        ds = relational_dataset(child, parent, None)
        lens = np.array(ds.map(lambda e: dict(L=len(e["labels"])))["L"])
        omax = int(lens.max()) + 16
        info["measurements"][str(window)] = dict(windows=int(len(parent)), max_label_tokens=int(lens.max()), mean_label_tokens=float(lens.mean()),
                                                 tokens_per_row=float((lens.max() - 2) / window), output_max_length=omax, fits_limit=bool(omax <= DECODER_LIMIT),
                                                 measure_seconds=round(time.time() - t0, 1))
        print(window, info["measurements"][str(window)], flush=True)
        if omax <= DECODER_LIMIT:
            chosen = window; break
    assert chosen is not None, "no window candidate fits the decoder limit"
    info["window"] = chosen; info["output_max_length"] = info["measurements"][str(chosen)]["output_max_length"]
    for split, tag in (("train", "train"), ("validation", "development")):
        child, parent, wmap = build_windows(split, chosen)
        child.to_parquet(PREP / f"{tag}_child.parquet", index=False); parent.to_parquet(PREP / f"{tag}_parent.parquet", index=False)
        wmap.to_parquet(PREP / f"{tag}_window_map.parquet", index=False)
        info[tag] = dict(customers=int(wmap.entity_id.nunique()), windows=int(len(parent)), rows=int(len(child)), frauds=int(child[LABEL].sum()),
                         fraud_windows=int(child.groupby("window_id")[LABEL].max().sum()), max_window_index=int(wmap.window_index.max()))
    jdump(info, REP / "PREPARE.json")
    print(json.dumps({k: v for k, v in info.items() if k != "measurements"}, indent=1))


# ---------------------------------------------------------------- fit
def fit_dirs(seed):
    return FIT / f"fit_{seed}"


def make_model(epochs, ckpt, omax, seed, base):
    from realtabformer import REaLTabFormer
    return REaLTabFormer(model_type="relational", numeric_nparts=2, output_max_length=omax, epochs=epochs, batch_size=8, train_size=0.9,
                         early_stopping_patience=5, random_state=seed, checkpoints_dir=str(ckpt), samples_save_dir=str(base / "samples"),
                         full_save_dir=str(base / "full"))


def run_fit(epochs, ckpt, omax, seed, base):
    child = pd.read_parquet(PREP / "train_child.parquet"); parent = pd.read_parquet(PREP / "train_parent.parquet")
    m = make_model(epochs, ckpt, omax, seed, base)
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


def save_model(m, where):
    for k, v in list(m.__dict__.items()):
        if isinstance(v, Path) and k not in ("checkpoints_dir", "samples_save_dir"):
            setattr(m, k, v.as_posix())
    m.save(where, allow_overwrite=True)


def eval_split_signature(trainer_or_ds):
    """Length and sha256 of the first 5 eval examples' labels — recorded at fit time, re-checked when the split is rebuilt."""
    ds = trainer_or_ds
    n = len(ds)
    h = hashlib.sha256()
    for i in range(min(5, n)):
        lab = ds[i]["labels"]; lab = lab.tolist() if hasattr(lab, "tolist") else list(lab)
        h.update(json.dumps([int(t) for t in lab]).encode())
    return dict(eval_windows=n, first5_labels_sha256=h.hexdigest())


def fit_summary(tr_state, m, secs, dropped, npar, cap, omax, extra=None):
    hist = tr_state["log_history"] if isinstance(tr_state, dict) else tr_state.log_history
    train_l = [h for h in hist if "loss" in h and "eval_loss" not in h]; eval_l = [h for h in hist if "eval_loss" in h]
    epoch = tr_state["epoch"] if isinstance(tr_state, dict) else tr_state.epoch
    d = dict(epoch_cap=cap, epochs_completed=float(epoch), global_step=tr_state["global_step"] if isinstance(tr_state, dict) else tr_state.global_step,
             stopped_early=bool(epoch < cap - 1e-6), seconds=secs, parents_dropped=dropped, parents_total=npar, output_max_length=omax,
             best_metric=tr_state["best_metric"] if isinstance(tr_state, dict) else tr_state.best_metric,
             best_model_checkpoint=str(tr_state["best_model_checkpoint"] if isinstance(tr_state, dict) else tr_state.best_model_checkpoint),
             final_train_loss=train_l[-1]["loss"] if train_l else None, final_eval_loss=eval_l[-1]["eval_loss"] if eval_l else None,
             best_eval_loss=min(h["eval_loss"] for h in eval_l) if eval_l else None, experiment_id=m.experiment_id, log_history=hist)
    if extra:
        d.update(extra)
    return d


def cmd_fit(args):
    mkdirs()
    seed = args.seed; base = fit_dirs(seed); base.mkdir(parents=True, exist_ok=True)
    rep = REP / f"FIT_{seed}.json"
    if rep.exists():
        print("FIT exists; skipping", seed); return
    omax = json.loads((REP / "PREPARE.json").read_text())["output_max_length"]
    if args.recover:
        return cmd_recover(args, base, omax, rep)
    timing_p = REP / f"TIMING_{seed}.json"
    if not timing_p.exists():
        print("== timing run: epochs=1", flush=True)
        m, tr, secs, dropped, npar = run_fit(1, base / "ckpt_timing", omax, seed, base)
        steps = tr.state.global_step
        del m, tr
        import torch; torch.cuda.empty_cache()
        cap = int(max(1, min(MAX_EPOCHS, math.floor(BUDGET_S / secs))))
        jdump(dict(epoch_seconds=secs, global_steps_one_epoch=steps, budget_seconds=BUDGET_S, max_epochs=MAX_EPOCHS, epoch_cap=cap,
                   output_max_length=omax, parents_dropped=dropped, parents_total=npar, started_unix=time.time(),
                   note="epoch_seconds includes model build/tokenization overhead and periodic eval/checkpoints"), timing_p)
    cap = json.loads(timing_p.read_text())["epoch_cap"]
    print(f"== real fit: seed={seed} epochs cap={cap}", flush=True)
    m, tr, secs, dropped, npar = run_fit(cap, base / "ckpt_main", omax, seed, base)
    sig = eval_split_signature(tr.eval_dataset) if tr.eval_dataset is not None else None
    save_model(m, base / "model")
    d = fit_summary(tr.state, m, secs, dropped, npar, cap, omax, dict(eval_split=sig))
    d["model_dir"] = str(base / "model" / m.experiment_id)
    jdump(d, rep)
    print("saved", d["model_dir"], flush=True)


def cmd_recover(args, base, omax, rep):
    """Amendment path (as in realtabformer_sparkov_v1): rebuild the model object without retraining, load the best
    checkpoint chosen by the original run, save. No training condition changes."""
    import torch
    from safetensors.torch import load_file
    timing = json.loads((REP / f"TIMING_{args.seed}.json").read_text()); cap = timing["epoch_cap"]
    ck = base / "ckpt_main"
    last = sorted(ck.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))[-1]
    st = json.loads((last / "trainer_state.json").read_text()); best = Path(st["best_model_checkpoint"])
    child = pd.read_parquet(PREP / "train_child.parquet"); parent = pd.read_parquet(PREP / "train_parent.parquet")
    m = make_model(cap, ck, omax, args.seed, base)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m._fit_relational(child, parent, join_on="window_id", device="cuda")
    dropped = 0
    for x in w:
        if "has been removed from the training data" in str(x.message):
            dropped = int(str(x.message).split("A total of ")[1].split(" out of")[0])
    sd = load_file(str(best / "model.safetensors"))
    missing = m.model.load_state_dict(sd, strict=False)
    print("load_state_dict:", missing, flush=True)
    m.model.eval(); m.experiment_id = f"id{int((time.time() * 10**10)):024}"
    save_model(m, base / "model")
    d = fit_summary(st, m, None, dropped, len(parent), cap, omax,
                    dict(recovered_after_save_crash=True, load_state_dict_missing=list(missing.missing_keys), eval_split=None))
    d["model_dir"] = str(base / "model" / m.experiment_id)
    jdump(d, rep)


# ---------------------------------------------------------------- shared loading
def load_fit(seed, device="cuda"):
    from realtabformer import REaLTabFormer
    d = json.loads((REP / f"FIT_{seed}.json").read_text())
    m = REaLTabFormer.load_from_dir(Path(d["model_dir"]))
    m.model.to(device).eval()
    for p in m.model.parameters():
        p.requires_grad_(False)
    return m, d


def load_sparkov(device="cuda"):
    from realtabformer import REaLTabFormer
    d = json.loads((SPARKOV / "report/FIT.json").read_text())
    m = REaLTabFormer.load_from_dir(Path(d["model_dir"]))
    m.model.to(device).eval()
    for p in m.model.parameters():
        p.requires_grad_(False)
    return m, d


def head_dir(seed, name):
    return HEADS / f"{name}_{seed}"


def build_head(m, geom, state_path=None, device="cuda"):
    import torch
    from benchmarks.rtf_ibm_module import FraudHead
    head = FraudHead(m.model.decoder, geom.routed_ids).to(device)
    if state_path is not None:
        sd = torch.load(state_path, map_location=device)
        head.load_state_dict(sd, strict=True)
    head.eval()
    return head


def collate(examples, geom, device):
    """examples: list of dataset rows with input_ids (encoder) and labels (decoder). Returns tensors on device."""
    import torch
    enc = torch.tensor([e["input_ids"] for e in examples], dtype=torch.long, device=device)
    T = max(len(e["labels"]) for e in examples)
    labels = torch.full((len(examples), T), geom.pad, dtype=torch.long, device=device)
    for i, e in enumerate(examples):
        labels[i, : len(e["labels"])] = torch.tensor(e["labels"], dtype=torch.long, device=device)
    dec_in = torch.full_like(labels, geom.pad); dec_in[:, 0] = geom.eos; dec_in[:, 1:] = labels[:, :-1]
    dec_mask = labels.ne(geom.pad).long(); dec_mask[:, 0] = 1
    return enc, labels, dec_in, dec_mask


def fraud_examples(ds, geom):
    """Indices of dataset rows whose labels contain the fraud token."""
    fid = geom.fraud_id
    return [i for i, lab in enumerate(ds["labels"]) if fid in lab]


def rebuild_split(m, seed, omax, fit_report):
    """Recreate the package's train/eval window split (train_test_split(test_size=.1, seed=fit seed)) and verify."""
    child = pd.read_parquet(PREP / "train_child.parquet"); parent = pd.read_parquet(PREP / "train_parent.parquet")
    ds = relational_dataset(child, parent, omax, model=m)
    sp = ds.train_test_split(test_size=1 - 0.9, seed=seed)
    sig = eval_split_signature(sp["test"])
    rec = fit_report.get("eval_split")
    ok = rec is not None and rec == sig
    return sp["train"], sp["test"], dict(rebuilt=sig, recorded=rec, match=ok)


# ---------------------------------------------------------------- check
def cmd_check(args):
    import torch
    from benchmarks.rtf_ibm_module import Geometry, Capture, ModuleProcessor, routed_targets, column_restricted_ce
    from transformers import LogitsProcessorList
    mkdirs()
    device = "cuda"
    if args.source == "sparkov":
        m, rep = load_sparkov(device)
        child = pd.read_parquet(SPARKOV / "prepared/train_child.parquet"); parent = pd.read_parquet(SPARKOV / "prepared/train_parent.parquet")
        omax = rep["output_max_length"]
    else:
        m, rep = load_fit(args.fit, device)
        child = pd.read_parquet(PREP / "train_child.parquet"); parent = pd.read_parquet(PREP / "train_parent.parquet")
        omax = rep["output_max_length"]
    geom = Geometry(m, routed=("gap", "amount_or_numeric_value", "category") if args.source == "sparkov" else ("gap", "amount_or_numeric_value", "category", "use_chip"))
    print("S", geom.S, "label col idx", geom.j_label, "routed sub-cols", geom.routed_j, "K", len(geom.routed_ids), "V", geom.V)
    # a few fraud windows
    fw = child[child[LABEL].eq(1)].window_id.unique()[:16]
    sub_child = child[child.window_id.isin(fw)]; sub_parent = parent[parent.window_id.isin(fw)]
    ds = relational_dataset(sub_child, sub_parent, omax, model=m)
    enc, labels, dec_in, dec_mask = collate([ds[i] for i in range(len(ds))], geom, device)
    cap = Capture(m.model.decoder)
    from benchmarks.rtf_ibm_module import FraudHead
    head_full = FraudHead(m.model.decoder, list(range(geom.V))).to(device).eval()  # full-vocab copy for the identity test
    head = build_head(m, geom, None, device)
    with torch.no_grad():
        out = m.model(input_ids=enc, decoder_input_ids=dec_in, decoder_attention_mask=dec_mask)
        native = out.logits  # (B, T, V)
        x = cap.x
        # (1) identity of the untrained copy at every valid position over the whole vocabulary
        valid = dec_mask.bool()
        fl = head_full(x[valid]); nl = native[valid]
        diff = float((fl - nl).abs().max())
        print("max |fraud head - native| (untrained copy, full vocab):", diff, "positions", int(valid.sum()))
        assert diff < 1e-3, diff
        del head_full
        # (2) routed positions exist iff the label precedes routed columns (IBM); report the count
        mask, cols = routed_targets(geom, labels)
        print("routed positions in", len(ds), "windows:", int(mask.sum()), "fraud rows:", int(labels.eq(geom.fraud_id).sum()))
        if geom.routed_j:
            ce = column_restricted_ce(geom, head(x[mask]), labels[mask], cols[mask])
            print("untrained routed CE mean:", float(ce.mean()))
    # (3) transition processor forcing: p=1 -> all generated rows fraud; p=0 -> none
    from realtabformer.rtf_sampler import RelationalSampler
    sampler = RelationalSampler.sampler_from_model(m, device=device)
    res = {}
    for forced in (1.0, 0.0):
        table = torch.full((3, 22, 2), forced, dtype=torch.float64, device=device)
        proc = ModuleProcessor(geom, table=table, init_state=None, head=None, capture=None)
        kw = sampler._get_min_max_length(5)
        seed_all(1)
        samples = sampler._generate(device=torch.device(device), as_numpy=True, constrain_tokens_gen=True, inputs=enc[:4], do_sample=True,
                                    suppress_tokens=None, forced_decoder_ids=None, logits_processor=LogitsProcessorList([proc]), **kw)
        labs = [[geom.i2t[t] for t in row if geom.i2t[t].startswith(geom.label_col)] for row in samples]
        frac = np.mean([float(l.endswith(f"{geom.label_col}___1")) for row in labs for l in row])
        res[str(forced)] = dict(label_steps=proc.n_label_steps, fraud_fraction=float(frac), rows=len(labs[0]))
        print("forced p", forced, res[str(forced)])
    assert res["1.0"]["fraud_fraction"] == 1.0 and res["0.0"]["fraud_fraction"] == 0.0
    # (4) routed generation with the untrained head runs and only touches fraud rows (count reported)
    if geom.routed_j:
        table = torch.full((3, 22, 2), 0.5, dtype=torch.float64, device=device)
        proc = ModuleProcessor(geom, table=table, init_state=None, head=head, capture=cap)
        kw = sampler._get_min_max_length(5)
        samples = sampler._generate(device=torch.device(device), as_numpy=True, constrain_tokens_gen=True, inputs=enc[:4], do_sample=True,
                                    suppress_tokens=None, forced_decoder_ids=None, logits_processor=LogitsProcessorList([proc]), **kw)
        df = sampler._processes_sample(samples, m.vocab["decoder"], relate_ids=list(range(4)))
        print("routed generation rows:", len(df), "routed-row interventions:", proc.n_routed_rows, "fraud rows:", int(pd.to_numeric(df[LABEL]).sum()))
        assert proc.n_routed_rows >= int(pd.to_numeric(df[LABEL]).sum()) * len(geom.routed_j), (proc.n_routed_rows, int(pd.to_numeric(df[LABEL]).sum()))
    cap.remove()
    jdump(dict(source=args.source, fit=args.fit, S=geom.S, j_label=geom.j_label, routed_j=geom.routed_j, K=len(geom.routed_ids),
               identity_max_diff=diff, forcing=res, fraud_head_parameters=sum(p.numel() for p in head.parameters())), REP / f"CHECK_{args.source}_{args.fit}.json")
    print("CHECK PASSED")


# ---------------------------------------------------------------- train heads
def cmd_train_heads(args):
    import torch
    from benchmarks.rtf_ibm_module import Geometry, Capture, routed_targets, substitute_inputs, column_restricted_ce
    mkdirs()
    name = "routed_subst" if args.subst else "routed"
    hd = head_dir(args.fit, name); hd.mkdir(parents=True, exist_ok=True)
    if (hd / "TRAIN.json").exists():
        print("TRAIN.json exists; skipping", hd); return
    device = "cuda"
    m, rep = load_fit(args.fit, device)
    geom = Geometry(m)
    omax = rep["output_max_length"]
    train_ds, eval_ds, split_info = rebuild_split(m, args.fit, omax, rep)
    print("split:", split_info, flush=True)
    tr_idx = fraud_examples(train_ds, geom); ev_idx = fraud_examples(eval_ds, geom)
    print("fraud windows train/eval:", len(tr_idx), len(ev_idx), flush=True)
    cap = Capture(m.model.decoder)
    head = build_head(m, geom, None, device); head.train()
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    seed_all(args.seed)
    geom.allowed = geom.allowed.to(device); geom.compact = geom.compact.to(device)

    def forward_batch(ds, idx, subst):
        enc, labels, dec_in, dec_mask = collate([ds[i] for i in idx], geom, device)
        if subst:
            dec_in = substitute_inputs(geom, labels, dec_in, rng)
        with torch.no_grad():
            m.model(input_ids=enc, decoder_input_ids=dec_in, decoder_attention_mask=dec_mask)
            x = cap.x.detach()
        mask, cols = routed_targets(geom, labels)
        logits = head(x[mask])
        ce = column_restricted_ce(geom, logits, labels[mask], cols[mask])
        return ce, cols[mask]

    def evaluate(subst):
        head.eval(); tot = {}; n = {}
        with torch.no_grad():
            for s in range(0, len(ev_idx), args.batch):
                ce, cols = forward_batch(eval_ds, ev_idx[s:s + args.batch], subst)
                for j in geom.routed_j:
                    sel = cols.eq(j)
                    tot[j] = tot.get(j, 0.0) + float(ce[sel].sum()); n[j] = n.get(j, 0) + int(sel.sum())
        head.train()
        per = {geom.cols[j]: tot[j] / max(n[j], 1) for j in geom.routed_j}
        return float(np.mean(list(per.values()))), per

    log = []; best = math.inf; best_epoch = -1; bad = 0; t0 = time.time()
    val0_real, per0_real = evaluate(False); val0_sub, per0_sub = evaluate(True)
    log.append(dict(epoch=0, train_ce=None, val_real=val0_real, val_subst=val0_sub, per_real=per0_real, per_subst=per0_sub))
    print(log[-1], flush=True)
    crit = val0_sub if args.subst else val0_real
    best = crit; best_epoch = 0; torch.save(head.state_dict(), hd / "heads.pt")
    for epoch in range(1, args.max_epochs + 1):
        perm = rng.permutation(len(tr_idx)); losses = []
        for s in range(0, len(perm), args.batch):
            idx = [tr_idx[i] for i in perm[s:s + args.batch]]
            ce, _ = forward_batch(train_ds, idx, args.subst)
            loss = ce.mean()
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
        val_real, per_real = evaluate(False); val_sub, per_sub = evaluate(True)
        crit = val_sub if args.subst else val_real
        log.append(dict(epoch=epoch, train_ce=float(np.mean(losses)), val_real=val_real, val_subst=val_sub, per_real=per_real, per_subst=per_sub, seconds=time.time() - t0))
        print({k: v for k, v in log[-1].items() if not k.startswith("per_")}, flush=True)
        if crit < best - 1e-4:
            best = crit; best_epoch = epoch; bad = 0; torch.save(head.state_dict(), hd / "heads.pt")
        else:
            bad += 1
            if bad >= args.patience:
                break
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("per_")} for r in log]).to_csv(hd / "learning.csv", index=False)
    pd.DataFrame([dict(epoch=r["epoch"], **{f"real_{c}": v for c, v in r["per_real"].items()}, **{f"subst_{c}": v for c, v in r["per_subst"].items()}) for r in log]).to_csv(hd / "learning_by_column.csv", index=False)
    sd = torch.load(hd / "heads.pt", map_location="cpu")
    jdump(dict(fit=args.fit, heads=name, substitute=bool(args.subst), epochs_run=log[-1]["epoch"], best_epoch=best_epoch, best_criterion=best,
               criterion="val_subst" if args.subst else "val_real", lr=args.lr, batch=args.batch, patience=args.patience, max_epochs=args.max_epochs,
               fraud_windows_train=len(tr_idx), fraud_windows_eval=len(ev_idx), split=split_info, head_parameters=sum(p.numel() for p in head.parameters()),
               head_sha256=digest_state(sd), seconds=time.time() - t0, routed_sub_columns=[geom.cols[j] for j in geom.routed_j]), hd / "TRAIN.json")
    cap.remove()
    print("done", hd, "best epoch", best_epoch, best, flush=True)


# ---------------------------------------------------------------- generate
def assemble(syn, wmap, real):
    """Decoded child rows (index = window_id) -> event frame in the ibm_tabformer_v1 generated format."""
    syn = syn.copy(); syn["window_id"] = syn.index.astype("int64")
    want = wmap.set_index("window_id")["window_rows"]
    counts = syn.groupby("window_id").size(); got = counts.reindex(want.index).fillna(0).astype(int)
    mism = int((got != want).sum())
    syn = syn.merge(wmap[["window_id", "entity_id", "window_index"]], on="window_id")
    syn["_ord"] = syn.groupby("window_id").cumcount()
    syn = syn[syn["_ord"] < syn["window_id"].map(want)]
    syn = syn.sort_values(["entity_id", "window_index", "_ord"], kind="stable").reset_index(drop=True)
    syn["event_index"] = syn.groupby("entity_id").cumcount().astype("int64")
    g = pd.to_numeric(syn["gap"], errors="coerce").fillna(0).clip(lower=0).round().astype(float)
    syn["gap"] = g.where(syn["event_index"] > 0, np.nan)
    syn[LABEL] = pd.to_numeric(syn[LABEL], errors="coerce").fillna(0).clip(0, 1).astype(int).astype(str).astype(object)
    for c in ("receiver_or_mark", "category", "use_chip"):
        syn[c] = syn[c].astype(str).astype(real[c].dtype)
    syn["amount_or_numeric_value"] = pd.to_numeric(syn["amount_or_numeric_value"], errors="coerce").astype(float)
    syn["entity_id"] = syn["entity_id"].astype("int64")
    res = syn[list(real.columns)].astype({c: real[c].dtype for c in real.columns if c != LABEL})
    return res, mism


def gen_stats(res, real, seconds, extra):
    f = res[LABEL].eq("1"); rf = real[LABEL].eq("1")
    d = dict(events=int(len(res)), real_events=int(len(real)), customers=int(res.entity_id.nunique()), frauds=int(f.sum()), real_frauds=int(rf.sum()),
             fraud_customers=int(res[f].entity_id.nunique()), real_fraud_customers=int(real[rf].entity_id.nunique()),
             fraud_online_share=float(res[f].use_chip.eq("Online Transaction").mean()) if f.any() else None,
             real_fraud_online_share=float(real[rf].use_chip.eq("Online Transaction").mean()), seconds=seconds)
    s = res.sort_values(["entity_id", "event_index"]); y = s[LABEL].eq("1").astype(int); prev = y.groupby(s.entity_id).shift()
    d["continuation"] = float(y[prev.eq(1)].mean()) if prev.eq(1).any() else None
    d.update(extra)
    return d


def encoder_inputs(m, parent_rows, device):
    from realtabformer.data_utils import make_dataset, process_data
    import torch
    in_df, _ = process_data(parent_rows.drop(columns="window_id"), col_transform_data=m.in_col_transform_data)
    ds = make_dataset(in_df, m.vocab["encoder"])
    return torch.tensor([r["input_ids"] for r in ds], dtype=torch.long, device=device)


def cmd_generate(args):
    import torch
    from benchmarks.rtf_ibm_module import Geometry, Capture, ModuleProcessor, final_state, final_state6
    from realtabformer.rtf_sampler import RelationalSampler
    from transformers import LogitsProcessorList
    mkdirs()
    arm = ARMS[args.arm]; device = "cuda"
    m, rep = load_fit(args.fit, device)
    geom = Geometry(m)
    parent = pd.read_parquet(PREP / "development_parent.parquet"); wmap = pd.read_parquet(PREP / "development_window_map.parquet")
    real = pd.read_parquet(SRC / "validation.parquet")
    table = torch.tensor(json.loads(TABLE.read_text())["table"], dtype=torch.float64, device=device) if arm["table"] else None
    learned = bool(arm.get("learned")); state_fn = None
    if learned:
        import pickle
        clf = pickle.load(open(STATE_PKL, "rb")); state_fn = lambda X: clf.predict_proba(X)[:, 1]
    head = cap = None
    if arm["heads"]:
        hd = head_dir(args.fit, arm["heads"])
        head = build_head(m, geom, hd / "heads.pt", device); cap = Capture(m.model.decoder)
        head_sha = json.loads((hd / "TRAIN.json").read_text())["head_sha256"]
    out_dir = RUNS / f"{args.arm}_{args.fit}"; out_dir.mkdir(parents=True, exist_ok=True)
    sampler = RelationalSampler.sampler_from_model(m, device=device)
    for gs in args.seeds:
        dest = out_dir / f"generated_validation_{gs}.parquet"; rep_p = out_dir / f"generation_validation_{gs}.json"
        if dest.exists() and rep_p.exists():
            print("exists; skipping", dest); continue
        seed_all(gs); t0 = time.time()
        if not arm["table"] and head is None and not learned:
            syn = m.sample(input_unique_ids=parent["window_id"].copy(), input_df=parent.drop(columns="window_id").copy(), related_num="window_rows",
                           gen_batch=args.gen_batch, device=device)
            extra = dict(mode="package_sample_window_independent")
        else:
            # rounds over window_index so the transition state carries over within each customer
            wm = wmap.set_index("window_id")
            state = {int(e): torch.zeros(6 if learned else 4, dtype=torch.long) for e in wm.entity_id.unique()}
            pieces = []; label_steps = 0; routed_rows = 0
            for w in sorted(wm.window_index.unique()):
                ids = wm.index[wm.window_index.eq(w)]
                # mixed window lengths share a batch: generate to the batch's longest window and truncate each window
                # to its own window_rows afterwards (singleton batches per length would cost ~1 generate per customer)
                P = parent[parent.window_id.isin(ids)].sort_values("window_rows", ascending=False)
                for s in range(0, len(P), args.gen_batch):
                    Pb = P.iloc[s:s + args.gen_batch]
                    kw = sampler._get_min_max_length(int(Pb.window_rows.max()))
                    enc = encoder_inputs(m, Pb, device)
                    ents = [int(wm.loc[i, "entity_id"]) for i in Pb.window_id]
                    init = torch.stack([state[e] for e in ents]).to(device)
                    proc = ModuleProcessor(geom, table=table, init_state=init, head=head, capture=cap, state_fn=state_fn)
                    samples = sampler._generate(device=torch.device(device), as_numpy=True, constrain_tokens_gen=True, inputs=enc, do_sample=True,
                                                suppress_tokens=None, forced_decoder_ids=None, logits_processor=LogitsProcessorList([proc]), **kw)
                    label_steps += proc.n_label_steps; routed_rows += proc.n_routed_rows
                    df = sampler._processes_sample(samples, m.vocab["decoder"], relate_ids=list(Pb.window_id))
                    df["_ord"] = df.groupby(level=0).cumcount()
                    want = df.index.map(wm["window_rows"])
                    df = df[df["_ord"].to_numpy() < np.asarray(want)].drop(columns="_ord")
                    pieces.append(df)
                    labs = pd.to_numeric(df[LABEL], errors="coerce").fillna(0).astype(int)
                    for wid, e in zip(Pb.window_id, ents):
                        seq = labs[df.index == wid].to_numpy() if (df.index == wid).any() else np.zeros(0, dtype=np.int64)
                        state[e] = (final_state6 if learned else final_state)(seq, state[e].numpy())
                print("window_index", w, "customers", len(P), "rows so far", sum(len(p) for p in pieces), f"{time.time() - t0:.0f}s", flush=True)
            syn = pd.concat(pieces)
            extra = dict(mode="rounds_with_state_carry_over_mixed_length_batches_truncated", label_steps=label_steps, routed_row_interventions=routed_rows,
                         head_sha256=head_sha if head is not None else None, table=str(TABLE) if table is not None else None,
                         learned_state_process=str(STATE_PKL) if learned else None)
        secs = time.time() - t0
        res, mism = assemble(syn, wmap, real)
        res.to_parquet(dest, index=False)
        jdump(gen_stats(res, real, secs, dict(arm=args.arm, fit_seed=args.fit, generation_seed=gs, windows=int(len(wmap)), windows_row_count_mismatch=mism,
                                              conditioned_on_real_window_lengths=True, **extra)), rep_p)
        print("GENERATED", args.arm, args.fit, gs, json.dumps({k: v for k, v in json.loads(rep_p.read_text()).items() if k in ("events", "frauds", "fraud_customers", "fraud_online_share", "continuation", "seconds")}), flush=True)
    if cap is not None:
        cap.remove()


# ---------------------------------------------------------------- probe / evaluate (eval split of the fit: fraud rows)
def cmd_evaluate(args):
    """Teacher-forced per-column CE on the fit's eval windows for native / routed / routed_subst heads, by row label,
    plus the label-path probe: P(online | real normal rows, label input set to 1) through the routed head."""
    import torch
    from benchmarks.rtf_ibm_module import Geometry, Capture, column_restricted_ce
    mkdirs()
    device = "cuda"
    m, rep = load_fit(args.fit, device); geom = Geometry(m)
    geom.allowed = geom.allowed.to(device); geom.compact = geom.compact.to(device)
    _, eval_ds, split_info = rebuild_split(m, args.fit, rep["output_max_length"], rep)
    idx = list(range(len(eval_ds)))
    cap = Capture(m.model.decoder)
    heads = {"native": None}
    for name in ("routed", "routed_subst"):
        p = head_dir(args.fit, name) / "heads.pt"
        if p.exists():
            heads[name] = build_head(m, geom, p, device)
    j_chip = [j for j in geom.routed_j if geom.cols[j].endswith("use_chip")][0]
    online_id = geom.t2i[f"{geom.cols[j_chip]}{SPECIAL_COL_SEP}Online Transaction"]
    rows = []; probe = {k: [] for k in heads}
    with torch.no_grad():
        for s in range(0, len(idx), 8):
            enc, labels, dec_in, dec_mask = collate([eval_ds[i] for i in idx[s:s + 8]], geom, device)
            out = m.model(input_ids=enc, decoder_input_ids=dec_in, decoder_attention_mask=dec_mask); x = cap.x
            flag, k, c = geom.row_labels(labels)
            for j in geom.routed_j:
                pos = c.eq(j) & flag.ge(0)
                tgt = labels[pos]; cols = c[pos]; fl = flag[pos]
                nat = out.logits[pos][:, geom.routed_ids]
                for name, head in heads.items():
                    lg = nat if head is None else head(x[pos])
                    ce = column_restricted_ce(geom, lg, tgt, cols)
                    for lab, nm in ((1, "fraud"), (0, "normal")):
                        sel = fl.eq(lab)
                        if sel.any():
                            rows.append(dict(column=geom.cols[j], split=nm, heads=name, ce_sum=float(ce[sel].sum()), n=int(sel.sum())))
            # label-path probe at the use_chip position: normal rows with the row label input flipped to fraud
            flipped = dec_in.clone()
            for b in range(labels.shape[0]):
                for r in range(0, (labels.shape[1] - 2) // (geom.S + 2) + 1):
                    li = geom.label_index(r, geom.j_label)
                    if li < labels.shape[1] and labels[b, li] == geom.normal_id:
                        flipped[b, li + 1] = geom.fraud_id
            out2 = m.model(input_ids=enc, decoder_input_ids=flipped, decoder_attention_mask=dec_mask); x2 = cap.x
            pos = c.eq(j_chip) & flag.eq(0)
            for name, head in heads.items():
                if head is None:
                    lg = out2.logits[pos][:, geom.routed_ids]
                else:
                    lg = head(x2[pos])
                allowed = geom.allowed[j_chip]
                pr = torch.softmax(lg.masked_fill(~allowed, -math.inf), -1)[:, geom.compact[online_id]]
                probe[name].append(pr.cpu())
            # real fraud rows, real inputs
    cap.remove()
    df = pd.DataFrame(rows).groupby(["column", "split", "heads"]).sum().reset_index(); df["ce"] = df.ce_sum / df.n
    tab = df.pivot_table(index=["column", "split"], columns="heads", values="ce")
    tab.to_csv(REP / f"val_ce_by_column_{args.fit}.csv"); print(tab.round(4).to_string())
    pr = {k: float(torch.cat(v).mean()) for k, v in probe.items() if v}
    jdump(dict(fit=args.fit, split=split_info, normal_rows_label_set_1_p_online=pr, eval_windows=len(idx)), REP / f"LABEL_PATH_PROBE_{args.fit}.json")
    print("P(online | normal rows, label set 1):", pr)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prepare")
    f = sp.add_parser("fit"); f.add_argument("--seed", type=int, required=True); f.add_argument("--recover", action="store_true")
    c = sp.add_parser("check"); c.add_argument("--source", choices=["sparkov", "ibm"], default="ibm"); c.add_argument("--fit", type=int, default=FIT_SEEDS[0])
    t = sp.add_parser("train_heads"); t.add_argument("--fit", type=int, required=True); t.add_argument("--subst", action="store_true")
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--batch", type=int, default=8); t.add_argument("--max_epochs", type=int, default=100)
    t.add_argument("--patience", type=int, default=5); t.add_argument("--seed", type=int, default=20261108)
    g = sp.add_parser("generate"); g.add_argument("--fit", type=int, required=True); g.add_argument("--arm", choices=list(ARMS), required=True)
    g.add_argument("--seeds", type=int, nargs="+", default=GEN_SEEDS); g.add_argument("--gen_batch", type=int, default=128)
    e = sp.add_parser("evaluate"); e.add_argument("--fit", type=int, required=True)
    a = ap.parse_args()
    {"prepare": cmd_prepare, "fit": cmd_fit, "check": cmd_check, "train_heads": cmd_train_heads, "generate": cmd_generate, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
