#!/usr/bin/env python3
"""Execute the preregistered CoFSeqGen-SAF metric audit on train rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.cof_seqgen_saf_metrics import (  # noqa: E402
    audit_metric_validity,
    calibrate_train_only_margins,
)


CANONICAL_ROOT = REPOSITORY_ROOT / "data" / "cof_seqgen_saf" / "canonical"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "artifacts"
    / "cof_seqgen_saf"
    / "metric_validity_audit_train_only.json"
)
REQUIRED_COLUMNS = [
    "entity_id",
    "event_index",
    "gap",
    "receiver_or_mark",
    "amount_or_numeric_value",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _train_events(dataset_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_path = dataset_dir / "entity_splits.parquet"
    event_path = dataset_dir / "events.parquet"
    splits = pd.read_parquet(split_path, columns=["entity_id", "split"])
    train_ids = set(splits.loc[splits["split"] == "train", "entity_id"])
    if not train_ids:
        raise RuntimeError(f"{dataset_dir.name}: empty train split")
    # No statistic is computed before the entity-level train filter. Only the
    # five audit columns are loaded; validation/test rows are immediately
    # discarded and are never passed into a metric function.
    events = pd.read_parquet(event_path, columns=REQUIRED_COLUMNS)
    events = events[events["entity_id"].isin(train_ids)].copy()
    observed_ids = set(events["entity_id"])
    if observed_ids != train_ids:
        raise RuntimeError(f"{dataset_dir.name}: train entity/event mismatch")
    return events, {
        "event_file_sha256": _sha256(event_path),
        "split_file_sha256": _sha256(split_path),
        "train_entity_count": len(train_ids),
        "train_event_count": len(events),
        "nontrain_event_count_used": 0,
    }


def execute(dataset_names: list[str], margin_repeats: int) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for dataset_name in dataset_names:
        dataset_dir = CANONICAL_ROOT / dataset_name
        if not dataset_dir.is_dir():
            raise RuntimeError(f"canonical dataset does not exist: {dataset_name}")
        events, provenance = _train_events(dataset_dir)
        validity = audit_metric_validity(events)
        margins = calibrate_train_only_margins(
            events,
            seeds=tuple(range(margin_repeats)),
        )
        records[dataset_name] = {
            "provenance": provenance,
            "validity": validity,
            "noninferiority_margins_q95": margins,
        }
    return {
        "schema_version": "cof-seqgen-saf-metric-validity-audit-result-v1",
        "scope": "TRAIN_ONLY_NO_MODEL_OUTPUT",
        "source_provenance": {
            "metric_source_sha256": _sha256(
                REPOSITORY_ROOT / "benchmarks" / "cof_seqgen_saf_metrics.py"
            ),
            "audit_runner_sha256": _sha256(Path(__file__).resolve()),
            "metric_config_sha256": _sha256(
                REPOSITORY_ROOT
                / "configs"
                / "benchmark_v2"
                / "cof_seqgen_saf_metric_audit.yaml"
            ),
        },
        "candidate_output_accessed": False,
        "validation_event_used": False,
        "test_event_used": False,
        "margin_repeats": margin_repeats,
        "datasets": records,
        "all_datasets_all_sensitivity_checks_pass": all(
            record["validity"]["all_checks_pass"] for record in records.values()
        ),
        "scientific_interpretation": (
            "Per-dataset sensitivity failure is a power warning. It cannot be "
            "used as model evidence and cannot trigger endpoint replacement."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=sorted(path.name for path in CANONICAL_ROOT.iterdir() if path.is_dir()),
    )
    parser.add_argument("--margin-repeats", type=int, default=31)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.margin_repeats < 5:
        parser.error("--margin-repeats must be at least five")
    result = execute(args.datasets, args.margin_repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "datasets": list(result["datasets"]),
                "all_checks": result["all_datasets_all_sensitivity_checks_pass"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
