#!/usr/bin/env python3
"""CLI for train-only external baseline fit and validation evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cof_seqgen_saf_external_validation import (  # noqa: E402
    ExternalValidationConfig,
    run_external_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    try:
        report = run_external_validation(ExternalValidationConfig(**raw))
    except Exception as error:
        # A raw-output support violation is a scientific result, not a reason
        # to silently repair generated content.  Persist a machine-readable
        # fail-closed record before preserving the original traceback/exit.
        output_dir = Path(raw["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": "cof-seqgen-saf-external-failure-v1",
            "baseline_id": raw.get("baseline_id"),
            "fit_seed": raw.get("fit_seed"),
            "generation_seed": raw.get("generation_seed"),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "test_accessed": False,
            "claim_status": "FAILED_CLOSED_NO_REPAIR_NO_PERFORMANCE_CLAIM",
        }
        (output_dir / "external_validation_failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        raise
    print(
        json.dumps(
            {
                "baseline_id": report["baseline_id"],
                "dataset_id": report["dataset_id"],
                "fit_seconds": report["fit_seconds"],
                "sample_seconds": report["sample_seconds"],
                "report_sha256": report["report_sha256"],
                "test_accessed": report["test_accessed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
