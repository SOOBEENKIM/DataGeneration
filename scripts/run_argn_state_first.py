"""Prepare, train, and freely sample the registered first B versus B+S study."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/argn_state_first_v1"
SOURCE = ROOT.parent / "research-argn-fraud-audit/artifacts/sparkov_argn_control_v2"
CONFIG = ROOT / "configs/argn_state_first_v1.json"
CFG = json.loads(CONFIG.read_text())
os.environ.setdefault("HF_HOME", str(OUT / "cache/huggingface"))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(OUT / "cache/joblib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from importlib.metadata import version
from mostlyai.engine import generate
from mostlyai.engine._workspace import Workspace
from benchmarks.argn_past_state import FEATURES, STATE_COLUMN, PastState
from benchmarks.argn_state_adapter import engine_adapter


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str) + "\n")


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.set_num_threads(CFG["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def source_hashes():
    paths = [CONFIG, Path(__file__), ROOT / "benchmarks/argn_past_state.py",
             ROOT / "benchmarks/argn_state_adapter.py", ROOT / "docs/argn_state_first_v1/PROTOCOL.md"]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def prepare():
    dest = OUT / "prepared"
    dest.mkdir(parents=True, exist_ok=False)
    src = SOURCE / "codec_digit_both"
    shutil.copytree(src, dest / "workspace")
    codec = PastState(Workspace(dest / "workspace").tgt_stats.read())
    manifest = {"config": CFG, "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_hashes": source_hashes(), "source_workspace": str(src),
                "features": FEATURES, "input_files": {}, "encoded_splits": {},
                "test_events_read": False}
    for path in sorted(src.rglob("*")):
        if path.is_file():
            manifest["input_files"][str(path.relative_to(src))] = digest(path)
    for path in sorted((dest / "workspace/OriginalData/encoded-data").glob("*.parquet")):
        frame = pd.read_parquet(path)
        rows = frame.to_dict("records")
        states = []
        for i, row in enumerate(rows):
            states.append(codec.sequence(row))
            if (i + 1) % 100 == 0:
                print(f"PREPARE {path.name} {i+1}/{len(rows)}", flush=True)
        # Arrow stores nested fixed-width feature rows as a list of lists.
        frame[STATE_COLUMN] = [s.tolist() for s in states]
        frame.to_parquet(path, index=False)
        manifest["encoded_splits"][path.name] = {
            "customers": len(frame), "events": sum(len(s) for s in states),
            "max_length": max(len(s) for s in states), "sha256": digest(path),
        }
    ctx = SOURCE / "prepared/validation_context.parquet"
    shutil.copy2(ctx, dest / "validation_context.parquet")
    manifest["development_context_sha256"] = digest(ctx)
    write(dest / "MANIFEST.json", manifest)
    print("PREPARATION_COMPLETE", flush=True)


def check_manifest():
    manifest = json.loads((OUT / "prepared/MANIFEST.json").read_text())
    assert manifest["config"] == CFG
    assert manifest["source_hashes"] == source_hashes(), "registered code changed after preparation"
    for name, info in manifest["encoded_splits"].items():
        assert digest(OUT / "prepared/workspace/OriginalData/encoded-data" / name) == info["sha256"]
    return manifest


def fit(arm, fit_seed, device):
    check_manifest()
    assert arm in CFG["arms"] and fit_seed in CFG["fit_seeds"]
    assert torch.__version__ == CFG["torch_version"]
    assert version("mostlyai-engine") == CFG["engine_version"]
    folder = OUT / "runs" / f"{arm}_{fit_seed}"
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copytree(OUT / "prepared/workspace", folder / "workspace")
    ws = Workspace(folder / "workspace")
    assert not ws.model_tabular_weights_path.exists(), "new fit must start from scratch"
    seed(fit_seed)
    write(folder / "START.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "arm": arm,
        "fit_seed": fit_seed, "config": CFG, "source_hashes": source_hashes(),
        "preparation_sha256": digest(OUT / "prepared/MANIFEST.json"),
        "device": device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "engine": version("mostlyai-engine"),
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "pid": os.getpid(), "fresh_initialization": True,
    })
    start = time.monotonic()
    with engine_adapter(ws.tgt_stats.read(), arm == "B_S", folder) as train:
        train(model=CFG["model"], max_training_time=CFG["max_training_minutes"],
              max_epochs=CFG["max_epochs"], batch_size=CFG["batch_size"],
              gradient_accumulation_steps=CFG["gradient_accumulation_steps"],
              max_sequence_window=ws.tgt_stats.read()["seq_len"]["max"],
              enable_flexible_generation=False, device=device, workspace_dir=folder / "workspace")
    assert ws.model_tabular_weights_path.exists()
    weights = torch.load(ws.model_tabular_weights_path, map_location="cpu", weights_only=True)
    write(folder / "FIT.json", {
        "seconds": time.monotonic() - start, "weights_sha256": digest(ws.model_tabular_weights_path),
        "total_parameters": sum(t.numel() for t in weights.values()),
        "state_parameters": weights["state_projection.weight"].numel(),
        "projection_learned": arm == "B_S", "source_hashes": source_hashes(),
    })
    print(f"FIT_COMPLETE {arm} {fit_seed}", flush=True)


def sample(arm, fit_seed, device):
    check_manifest()
    folder = OUT / "runs" / f"{arm}_{fit_seed}"
    assert (folder / "FIT.json").exists()
    ws = Workspace(folder / "workspace")
    weights_hash = digest(ws.model_tabular_weights_path)
    context = pd.read_parquet(OUT / "prepared/validation_context.parquet")
    with engine_adapter(ws.tgt_stats.read(), arm == "B_S"):
        for gs in CFG["generation_seeds"]:
            dest = folder / f"generated_validation_{gs}.parquet"
            assert not dest.exists()
            seed(gs)
            start = time.monotonic()
            generate(ctx_data=context, device=device, workspace_dir=folder / "workspace")
            raw = pd.read_parquet(folder / "workspace/SyntheticData")
            raw.to_parquet(folder / f"native_validation_{gs}.parquet", index=False)
            frame = raw.rename(columns={"customer_id": "entity_id"}).copy()
            frame["event_index"] = frame.groupby("entity_id", sort=False).cumcount()
            frame.loc[frame.event_index.eq(0), "gap"] = np.nan
            frame.to_parquet(dest, index=False)
            assert digest(ws.model_tabular_weights_path) == weights_hash
            write(folder / f"generation_validation_{gs}.json", {
                "seconds": time.monotonic() - start, "events": len(frame),
                "customers": frame.entity_id.nunique(), "sha256": digest(dest),
                "weights_sha256": weights_hash, "native_generated_length": True,
                "labels_generated_jointly": True, "no_prevalence_or_run_length_repair": True,
            })
            print(f"GENERATION_COMPLETE {arm} {fit_seed} {gs}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "fit", "sample", "queue"])
    parser.add_argument("--arm", choices=CFG["arms"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)
    if args.action == "prepare":
        prepare()
    elif args.action == "fit":
        fit(args.arm, args.seed, args.device)
    elif args.action == "sample":
        sample(args.arm, args.seed, args.device)
    else:
        assert args.arm is not None
        queue_path = OUT / f"queue_{args.arm}.json"
        try:
            for fit_seed in CFG["fit_seeds"]:
                write(queue_path, {"status": "training", "arm": args.arm, "fit_seed": fit_seed, "pid": os.getpid()})
                fit(args.arm, fit_seed, args.device)
            for fit_seed in CFG["fit_seeds"]:
                write(queue_path, {"status": "generating", "arm": args.arm, "fit_seed": fit_seed, "pid": os.getpid()})
                sample(args.arm, fit_seed, args.device)
            write(queue_path, {"status": "complete", "arm": args.arm, "pid": os.getpid()})
        except Exception as error:
            write(queue_path, {"status": "failed", "arm": args.arm, "error": repr(error), "pid": os.getpid()})
            raise


if __name__ == "__main__":
    main()
