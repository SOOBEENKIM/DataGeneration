"""Fresh baseline-only objective control, followed by generation and evaluation."""
from datetime import datetime, timezone
import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch

from run_argn_state_first import ROOT, OUT as PARENT, SOURCE, CFG, seed, digest, write, check_manifest
from benchmarks.argn_state_adapter import engine_adapter
from benchmarks.argn_event_weight_control import event_weighted_adapter
from mostlyai.engine import generate
from mostlyai.engine._workspace import Workspace

OUT = ROOT / "artifacts/argn_event_weight_control_v1"
DOCS = ROOT / "docs/argn_state_first_v1/event_weight_control"
CONFIG = ROOT / "configs/argn_event_weight_control_v1.json"
CONTROL = json.loads(CONFIG.read_text())


def code_hashes():
    paths = [Path(__file__), CONFIG, ROOT / "benchmarks/argn_event_weight_control.py",
             ROOT / "docs/argn_state_first_v1/EVENT_WEIGHT_CONTROL_PROTOCOL.md",
             ROOT / "benchmarks/argn_state_evaluation.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def evaluate(folder, fs):
    from evaluate_sparkov_argn_control import extended, numeric_metrics, additional_relations
    from benchmarks.argn_fraud_audit import summaries
    from benchmarks.argn_state_evaluation import episode_features, scalar_summary, log_w1, age_amount_table, hazard_table
    state = json.loads((SOURCE / "prepared/metric_state.json").read_text())
    raw_real = pd.read_parquet(SOURCE / "prepared/validation.parquet")
    real, real_runs = episode_features(extended(raw_real, state))
    results, numbers, ages, hazards = [], [], [], []
    for gs in CONTROL["generation_seeds"]:
        raw = pd.read_parquet(folder / f"generated_validation_{gs}.parquet")
        assert set(raw.entity_id) == set(real.entity_id)
        d, runs = episode_features(extended(raw, state))
        name = f"B_event_weighted_{fs}_{gs}"
        row = dict(run=name, arm="B_event_weighted", fit_seed=fs, generation_seed=gs,
                   **summaries(raw_real, raw, state), **scalar_summary(d, runs))
        row.update(additional_relations(real, d))
        row["first_event_fraud_rate"] = float(d.loc[d.event_index.eq(0), "fraud"].eq(1).mean())
        row["run_length_log_w1"] = log_w1(real_runs.length, runs.length)
        row["run_span_log_w1"] = log_w1(real_runs.span_seconds, runs.span_seconds)
        rr, ss = real_runs[real_runs.completed_known_start], runs[runs.completed_known_start]
        row["completed_run_length_log_w1"] = log_w1(rr.length, ss.length)
        row["completed_run_span_log_w1"] = log_w1(rr.span_seconds, ss.span_seconds)
        bands, values = age_amount_table(real, d, name)
        row.update(values); ages.extend(bands); hazards.extend(hazard_table(d, name))
        num = numeric_metrics(real, d, name)
        for n in num:
            row[f"{n['label']}_{n['field']}_log_w1"] = n["wasserstein_log1p"]
        numbers.extend(num)
        row["customer_length_log_w1"] = log_w1(real.groupby("entity_id").size(), d.groupby("entity_id").size())
        results.append(row)
    DOCS.mkdir(parents=True, exist_ok=True)
    for name, rows in [("metrics", results), ("numeric", numbers), ("age_amounts", ages), ("hazards", hazards)]:
        pd.DataFrame(rows).to_csv(DOCS / f"{name}_{fs}.csv", index=False)


def run(fs, device):
    check_manifest()
    assert fs in CONTROL["fit_seeds"]
    folder = OUT / "runs" / f"B_event_weighted_{fs}"
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copytree(PARENT / "prepared/workspace", folder / "workspace")
    ws = Workspace(folder / "workspace")
    assert not ws.model_tabular_weights_path.exists()
    preparation = json.loads((PARENT / "prepared/MANIFEST.json").read_text())
    data = preparation["encoded_splits"]["part.000000-trn.parquet"]
    mean_length = data["events"] / data["customers"]
    hashes = code_hashes()
    status = OUT / f"queue_{fs}.json"
    write(folder / "START.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
          config=CONTROL, inherited_hyperparameters=CFG, mean_training_length=mean_length,
          source_hashes=hashes, parent_preparation_sha256=digest(PARENT / "prepared/MANIFEST.json"),
          fit_seed=fs, device=device, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
          torch=torch.__version__, pid=os.getpid(), fresh_initialization=True))
    try:
        write(status, dict(stage="training", seed=fs, pid=os.getpid()))
        seed(fs)
        started = time.monotonic()
        with event_weighted_adapter(ws.tgt_stats.read(), mean_length, folder) as train:
            train(model=CFG["model"], max_training_time=CFG["max_training_minutes"], max_epochs=CFG["max_epochs"],
                  batch_size=CFG["batch_size"], gradient_accumulation_steps=CFG["gradient_accumulation_steps"],
                  max_sequence_window=ws.tgt_stats.read()["seq_len"]["max"], enable_flexible_generation=False,
                  device=device, workspace_dir=folder / "workspace")
        weight_hash = digest(ws.model_tabular_weights_path)
        write(folder / "FIT.json", dict(seconds=time.monotonic()-started, weights_sha256=weight_hash,
              mean_training_length=mean_length, source_hashes=hashes))
        assert hashes == code_hashes()
        print(f"CONTROL_FIT_COMPLETE {fs}", flush=True)
        write(status, dict(stage="generating", seed=fs, pid=os.getpid()))
        context = pd.read_parquet(PARENT / "prepared/validation_context.parquet")
        with engine_adapter(ws.tgt_stats.read(), enabled=False):
            for gs in CONTROL["generation_seeds"]:
                seed(gs); started = time.monotonic()
                generate(ctx_data=context, device=device, workspace_dir=folder / "workspace")
                raw = pd.read_parquet(folder / "workspace/SyntheticData")
                raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
                frame = raw.rename(columns={"customer_id": "entity_id"}).copy()
                frame["event_index"] = frame.groupby("entity_id", sort=False).cumcount()
                frame.loc[frame.event_index.eq(0), "gap"] = np.nan
                dest = folder / f"generated_validation_{gs}.parquet"
                frame.to_parquet(dest, index=False)
                assert digest(ws.model_tabular_weights_path) == weight_hash
                write(folder / f"generation_validation_{gs}.json", dict(seconds=time.monotonic()-started,
                      events=len(frame), customers=frame.entity_id.nunique(), sha256=digest(dest),
                      weights_sha256=weight_hash, native_generated_length=True, labels_generated_jointly=True,
                      no_prevalence_or_run_length_repair=True))
                print(f"CONTROL_GENERATION_COMPLETE {fs} {gs}", flush=True)
        write(status, dict(stage="evaluating", seed=fs, pid=os.getpid()))
        evaluate(folder, fs)
        write(status, dict(stage="complete", seed=fs, pid=os.getpid(), generated_datasets=2,
                           source_hashes=hashes, test_events_read=False))
        print(f"CONTROL_COMPLETE {fs}", flush=True)
    except Exception as error:
        write(status, dict(stage="failed", seed=fs, pid=os.getpid(), error=repr(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)
    run(args.seed, args.device)
