#!/usr/bin/env python3
"""CLI for sealed-test CoFSeqGen-SAF validation comparisons."""

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

from experiments.cof_seqgen_saf_validation import (  # noqa: E402
    ValidationConfig,
    run_validation_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = run_validation_comparison(ValidationConfig(**raw))
    print(
        json.dumps(
            {
                "dataset_id": report["dataset_id"],
                "generation_entities": report["generation_entities"],
                "report_sha256": report["report_sha256"],
                "test_accessed": report["test_accessed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
