"""GPU optimization and native-generation integration check; never a study fit."""
import argparse
import json
import logging
import shutil
import time
from pathlib import Path
import pandas as pd
import torch

from run_argn_state_first import OUT, SOURCE, seed, write, digest
from benchmarks.argn_state_adapter import engine_adapter
from mostlyai.engine import generate
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import SLEN_SUB_COLUMN_PREFIX, encode_slen_sidx_sdec


def run(device):
    parent = OUT / "smoke"
    parent.mkdir(parents=True, exist_ok=False)
    report = {"purpose": "integration only, discarded weights", "device": device,
              "torch": torch.__version__, "arms": {}}
    for arm in ["B", "B_S"]:
        folder = parent / arm
        shutil.copytree(OUT / "prepared/workspace", folder / "workspace")
        ws = Workspace(folder / "workspace")
        ts = ws.tgt_stats.read()
        # Retain two longest full customer records to stress memory/padding.
        for path in ws.encoded_data_trn.fetch_all() + ws.encoded_data_val.fetch_all():
            frame = pd.read_parquet(path)
            key = next(k for k in frame if k.startswith("tgt:"))
            frame = frame.loc[frame[key].map(len).nlargest(2).index]
            frame.to_parquet(path, index=False)
        ts["no_of_training_records"] = ts["no_of_validation_records"] = 2
        ws.tgt_stats.write(ts)
        seed(123)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with engine_adapter(ts, arm == "B_S", folder) as train:
            train(model="MOSTLY_AI/Medium", max_training_time=5, max_epochs=0,
                  batch_size=2, gradient_accumulation_steps=1,
                  enable_flexible_generation=False, device=device, workspace_dir=folder / "workspace")
            assert ws.model_tabular_weights_path.exists()
            raw_weights = torch.load(ws.model_tabular_weights_path, map_location="cpu", weights_only=True)
            state = raw_weights["state_projection.weight"]
            assert all(torch.isfinite(t).all() for t in raw_weights.values())
            # Only for this disposable smoke: force length 8 through length-head
            # logits, avoiding a 3106-step untrained rollout. Study fits never use
            # this override and always retain native generated length weights.
            forced = encode_slen_sidx_sdec(pd.Series([8]), max_seq_len=ts["seq_len"]["max"], prefix=SLEN_SUB_COLUMN_PREFIX)
            for col in forced:
                weight = raw_weights[f"predictors.predictors.{col}.weight"]
                bias = raw_weights[f"predictors.predictors.{col}.bias"]
                weight.zero_(); bias.fill_(-1000); bias[int(forced[col].iloc[0])] = 1000
            torch.save(raw_weights, ws.model_tabular_weights_path)
            context = pd.read_parquet(SOURCE / "prepared/train_context.parquet").head(2)
            generate(ctx_data=context, device=device, workspace_dir=folder / "workspace")
            data = pd.read_parquet(folder / "workspace/SyntheticData")
            assert data.customer_id.nunique() == 2 and len(data) == 16
        report["arms"][arm] = {
            "elapsed_seconds": time.monotonic() - start,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "checkpoint_sha256": digest(ws.model_tabular_weights_path),
            "state_parameters": state.numel(), "generated_events": len(data),
            "all_weights_finite": True,
        }
        print(f"SMOKE_OK {arm}", flush=True)
    write(parent / "RESULT.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.device)
