"""Version-qualified Berka replication; never label author artifacts as new fits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def load_train(repo):
    account = pd.read_csv(repo / "data_train/berka_account_trn.csv", low_memory=False).convert_dtypes()
    transaction = pd.read_csv(repo / "data_train/berka_trans_trn.csv.gz", low_memory=False).convert_dtypes()
    assert len(account) == 2250 and len(transaction) == 526442
    assert account.account_id.is_unique and transaction.trans_id.is_unique
    assert transaction.account_id.isin(account.account_id).all()
    # Engine 1.0.4 requires numpy/Arrow int64 for sequence-length bookkeeping;
    # pandas nullable Int64 keys make groupby.transform('size') return Int64.
    # Identifiers are complete and are never learned/scored as attributes.
    for df, keys in [(account, ["account_id"]), (transaction, ["account_id", "trans_id"])]:
        for key in keys:
            df[key] = df[key].astype("int64")
    # Preserve the supplied row order and date strings. Author output contains _RARE_
    # date values, supporting categorical dates in this compatibility run.
    assert not (pd.to_datetime(transaction.date).groupby(transaction.account_id).diff() < pd.Timedelta(0)).any()
    return account, transaction


def random_adjacent_rows(df, col_by):
    """Sensitivity matching Appendix E's random-pair description, not released QA."""
    df = df.copy()
    df["__IDX"] = df.groupby(col_by).cumcount()
    lengths = df.groupby(col_by).size()
    selected = ((lengths - 1) * np.random.random(len(lengths))).astype(int)
    indexes = selected.rename("__IDX").reset_index()
    first = df.merge(indexes, on=[col_by, "__IDX"])
    indexes["__IDX"] += 1
    second = df.merge(indexes, on=[col_by, "__IDX"])
    return first.drop(columns="__IDX"), second.drop(columns="__IDX")


def verify_qa(out):
    from mostlyai.qa._sampling import sample_two_consecutive_rows
    from mostlyai.qa._accuracy import calculate_accuracy
    fixture = pd.DataFrame({"id": np.repeat(np.arange(100), 10), "position": np.tile(np.arange(10), 100)})
    seed_all(20260927)
    native, _ = sample_two_consecutive_rows(fixture.copy(), "id")
    corrected, successor = random_adjacent_rows(fixture.copy(), "id")
    same = pd.DataFrame({"x": pd.Categorical(["a", "a", "b", "b"])})
    disjoint = pd.DataFrame({"x": pd.Categorical(["c", "c", "d", "d"])})
    identity = calculate_accuracy(same, same)[0]
    separation = calculate_accuracy(same, disjoint)[0]
    assert identity == 1 and separation == 0
    assert native.position.eq(0).all()
    assert corrected.position.nunique() > 1
    assert (successor.position.to_numpy() - corrected.position.to_numpy() == 1).all()
    result = {"qa_version": importlib.metadata.version("mostlyai-qa"),
              "native_selected_positions": sorted(native.position.unique().tolist()),
              "corrected_selected_positions": sorted(corrected.position.unique().tolist()),
              "identical_accuracy": identity, "disjoint_accuracy": separation,
              "source_sha256": hashlib.sha256(Path(__import__('inspect').getfile(sample_two_consecutive_rows)).read_bytes()).hexdigest()}
    save(out / "qa_behavior_check.json", result)
    print(json.dumps(result, indent=2), flush=True)


def score(args):
    from mostlyai.qa import _sampling
    from mostlyai.qa._accuracy import bin_data, calculate_univariates, calculate_bivariates
    from mostlyai.qa.reporting import _calculate_metrics
    assert importlib.metadata.version("mostlyai-qa") == "1.5.3"
    real_account, real_transaction = load_train(args.paper_repo)
    if args.synthetic_run is None:
        synthetic_account = pd.read_parquet(args.paper_repo / "data_synthetic/berka/TabARGN_berka_account")
        synthetic_transaction = pd.read_parquet(args.paper_repo / f"data_synthetic/berka/TabARGN_berka_transaction_run_{args.author_run}")
        provenance = f"author_provided_run_{args.author_run}_not_our_fit"
    else:
        synthetic_account = pd.read_parquet(args.synthetic_run / "parent/SyntheticData")
        synthetic_transaction = pd.read_parquet(args.synthetic_run / "child/SyntheticData")
        provenance = "our_fresh_engine_1.0.4_fit"
    assert synthetic_account.account_id.is_unique
    assert synthetic_transaction.account_id.isin(synthetic_account.account_id).all()
    verify_qa(args.out)
    native_sampler = _sampling.sample_two_consecutive_rows
    results = []
    try:
        for sampler_name, sampler in [("released_first_pair", native_sampler),
                                      ("appendix_random_pair_sensitivity", random_adjacent_rows)]:
            _sampling.sample_two_consecutive_rows = sampler
            seed_all(args.seed)
            frames = []
            # Same ordering as qa.report: synthetic first, training second.
            for account, transaction in [(synthetic_account, synthetic_transaction), (real_account, real_transaction)]:
                frames.append(_sampling.pull_data_for_accuracy(
                    df_tgt=transaction.drop(columns="trans_id"), df_ctx=account,
                    ctx_primary_key="account_id", tgt_context_key="account_id", setup="1:N"))
            syn, trn = frames
            for col in trn:
                if pd.api.types.is_numeric_dtype(trn[col]):
                    syn[col] = pd.to_numeric(syn[col], errors="coerce")
                syn[col] = syn[col].astype(trn[col].dtype)
            trn_bin, bins = bin_data(trn, bins=10)
            syn_bin, _ = bin_data(syn, bins=bins)
            uni = calculate_univariates(trn_bin, syn_bin)
            biv = calculate_bivariates(trn_bin, syn_bin)
            metrics = _calculate_metrics(acc_uni=uni, acc_biv=biv).model_dump(mode="json")
            args.out.mkdir(parents=True, exist_ok=True)
            uni.to_csv(args.out / f"{sampler_name}_univariate.csv", index=False)
            biv.to_csv(args.out / f"{sampler_name}_bivariate.csv", index=False)
            result = {"provenance": provenance, "sampler": sampler_name,
                      "seed": args.seed, "qa_version": importlib.metadata.version("mostlyai-qa"),
                      "date_handling": "categorical_strings_as_supplied",
                      "synthetic_accounts": len(synthetic_account), "synthetic_transactions": len(synthetic_transaction),
                      "training_metric_rows": len(trn), "synthetic_metric_rows": len(syn),
                      "privacy_evaluated": False, "metrics": metrics}
            results.append(result)
            save(args.out / "accuracy_results.json", results)
            print(json.dumps(result, indent=2), flush=True)
    finally:
        _sampling.sample_two_consecutive_rows = native_sampler


def train(args):
    from mostlyai.engine import split, analyze, encode, train as fit, generate
    assert importlib.metadata.version("mostlyai-engine") == "1.0.4"
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {args.out}")
    args.out.mkdir(parents=True)
    seed_all(args.seed)
    account, transaction = load_train(args.paper_repo)
    manifest = {"status": "started", "seed": args.seed, "paper_repo": str(args.paper_repo),
                "engine_version": "1.0.4", "hardware": "CPU, 4 torch threads",
                "training_accounts": len(account), "training_transactions": len(transaction),
                "author_exact_version_known": False, "date_handling": "categorical_strings_as_supplied",
                "max_training_time_minutes_per_table": 300, "other_model_parameters": "engine defaults",
                "dtypes": {"account": account.dtypes.astype(str).to_dict(), "transaction": transaction.dtypes.astype(str).to_dict()},
                "stages": {}}
    save(args.out / "manifest.json", manifest)
    try:
        for name in ["parent", "child"]:
            workspace = args.out / name
            before = time.monotonic()
            if name == "parent":
                split(tgt_data=account, tgt_primary_key="account_id", workspace_dir=workspace)
            else:
                split(tgt_data=transaction, tgt_primary_key="trans_id", tgt_context_key="account_id",
                      ctx_data=account, ctx_primary_key="account_id", workspace_dir=workspace)
            analyze(workspace_dir=workspace)
            encode(workspace_dir=workspace)
            manifest["stages"][name] = {"preprocessing_seconds": time.monotonic() - before}
            manifest["status"] = f"training_{name}"
            save(args.out / "manifest.json", manifest)
            before = time.monotonic()
            fit(max_training_time=300, device="cpu", workspace_dir=workspace)
            manifest["stages"][name]["training_seconds"] = time.monotonic() - before
            manifest["status"] = f"generating_{name}"
            save(args.out / "manifest.json", manifest)
            before = time.monotonic()
            if name == "parent":
                generate(sample_size=len(account), device="cpu", workspace_dir=workspace)
            else:
                generated_account = pd.read_parquet(args.out / "parent/SyntheticData")
                generate(ctx_data=generated_account, device="cpu", workspace_dir=workspace)
            manifest["stages"][name]["generation_seconds"] = time.monotonic() - before
            manifest["stages"][name]["generated_rows"] = len(pd.read_parquet(workspace / "SyntheticData"))
            save(args.out / "manifest.json", manifest)
        manifest["status"] = "generation_complete_not_yet_evaluated"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        save(args.out / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "score"])
    parser.add_argument("--paper-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic-run", type=Path)
    parser.add_argument("--author-run", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(args.out.resolve().parent / "hf-cache"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    (train if args.command == "train" else score)(args)
