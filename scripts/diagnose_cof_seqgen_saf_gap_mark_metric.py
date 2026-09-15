#!/usr/bin/env python3
"""Show whether gap-conditioned absolute-mark TV detects repeat dependence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.cof_seqgen_saf_metrics import evaluate_metric_suite, fit_metric_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["value"]) for item in payload["entities"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    events_path = args.dataset_dir / "events.parquet"
    split_path = args.dataset_dir / "split_ids" / "train.json"
    train_ids = _split_ids(split_path)
    events = pd.read_parquet(events_path)
    train = events.loc[events["entity_id"].astype(str).isin(train_ids)].copy()
    train = train.sort_values(["entity_id", "event_index"], kind="mergesort")
    state = fit_metric_state(train)

    corrupted = train.copy()
    nonfirst = corrupted["gap"].notna()
    bin_ids = np.searchsorted(
        np.asarray(state.gap_bin_edges),
        corrupted.loc[nonfirst, "gap"].to_numpy(float),
        side="right",
    )
    rng = np.random.default_rng(args.seed)
    for bin_id in np.unique(bin_ids):
        indices = corrupted.loc[nonfirst].index[bin_ids == bin_id]
        values = corrupted.loc[indices, "receiver_or_mark"].to_numpy(copy=True)
        corrupted.loc[indices, "receiver_or_mark"] = values[rng.permutation(len(values))]

    metrics = evaluate_metric_suite(train, corrupted, state, include_privacy=False)
    selected = {
        endpoint: float(metrics[endpoint])
        for endpoint in (
            "gap_conditioned_mark_tv",
            "short_gap_repeat_curve_l1",
            "gap_repeat_mi_error",
            "mark_sparse_tv",
        )
    }
    expects_mi_signal = "kappa_1_00" in args.dataset_dir.name
    artifact = {
        "schema_version": "cof-seqgen-saf-gap-mark-metric-diagnosis-v1",
        "dataset_dir": str(args.dataset_dir),
        "scope": "train_only",
        "corruption": (
            "permute current marks within each train-fitted gap bin; this preserves "
            "p(mark_t | gap_bin_t) exactly while destroying mark-transition dependence"
        ),
        "seed": args.seed,
        "metrics": selected,
        "checks": {
            "absolute_gap_conditioned_mark_tv_blind": selected[
                "gap_conditioned_mark_tv"
            ] < 1e-12,
            "repeat_curve_detects_corruption": selected[
                "short_gap_repeat_curve_l1"
            ] > 0.01,
            "repeat_mi_behavior_matches_kappa_cell": (
                selected["gap_repeat_mi_error"] > 1e-5
                if expects_mi_signal
                else selected["gap_repeat_mi_error"] < 1e-5
            ),
        },
        "expects_gap_repeat_mi_signal": expects_mi_signal,
        "test_accessed": False,
        "provenance": {
            "events_sha256": _sha256(events_path),
            "train_split_sha256": _sha256(split_path),
        },
    }
    artifact["all_checks_pass"] = bool(all(artifact["checks"].values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
