#!/usr/bin/env python3
"""CLI for checkpointed CoFSeqGen-SAF development training."""

from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import sys

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cof_seqgen_saf_training import (  # noqa: E402
    TrainingConfig,
    train_development_model,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        parser.error("training config must be a mapping")
    report = train_development_model(TrainingConfig(**raw))
    print(
        json.dumps(
            {
                "dataset_id": report["dataset_id"],
                "best_epoch": report["best_epoch"],
                "best_validation_loss": report["best_validation_loss"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "test_accessed": report["test_accessed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
