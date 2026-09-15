from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm
import yaml

from benchmarks.receiver_diagnostics import (
    entity_category_counts,
    entity_category_frequencies,
)
from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_receiver_only


def simultaneous_interval(
    frequencies: np.ndarray, labels: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = [frequencies[labels == label] for label in (0, 1)]
    difference = groups[1].mean(0) - groups[0].mean(0)
    standard_error = np.sqrt(
        groups[1].var(0, ddof=1) / len(groups[1])
        + groups[0].var(0, ddof=1) / len(groups[0])
    )
    critical = norm.ppf(1 - alpha / (2 * frequencies.shape[1]))
    return difference, difference - critical * standard_error, difference + critical * standard_error


def contaminate(
    categories: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    *,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    result = categories.copy()
    valid = np.arange(categories.shape[1])[None, :] < lengths[:, None]
    changed = valid & (labels[:, None] == 1) & (rng.random(categories.shape) < probability)
    result[changed] = 0
    return result


def gate(
    categories: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    *,
    n_categories: int,
    margin: float,
) -> tuple[bool, float, float]:
    counts = entity_category_counts(categories, lengths, n_categories)
    frequencies = entity_category_frequencies(counts)
    difference, low, high = simultaneous_interval(frequencies, labels)
    passed = bool(np.all(low >= -margin) and np.all(high <= margin))
    return passed, float(np.max(np.abs(difference))), float(np.max(np.maximum(np.abs(low), np.abs(high))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_v2/main_v2_1_candidate.yaml")
    parser.add_argument("--output-root", default="artifacts/benchmark_v2/receiver_diagnosis/v2_1_candidate")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    receiver_gate = raw["receiver_gate"]
    margin = float(receiver_gate["primary"]["category_practical_margin"])
    leakage = float(
        receiver_gate["stress_test"]["fraud_category0_contamination_probability"]
    )
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    for scenario in raw["scenarios"]:
        for kappa in (0.0, 1.0):
            config = BenchmarkConfig.from_mapping(raw, scenario, kappa)
            for replicate in range(args.seeds):
                seed = int(raw["seeds"]["base"]) + replicate
                sample = generate_receiver_only(
                    config,
                    n_entities=int(raw["data"]["n_test"]),
                    seed=seed,
                    split_id=1,
                )
                clean_pass, clean_max, clean_bound = gate(
                    sample["categories"],
                    sample["lengths"],
                    sample["labels"],
                    n_categories=config.n_receiver_categories,
                    margin=margin,
                )
                contaminated = contaminate(
                    sample["categories"],
                    sample["lengths"],
                    sample["labels"],
                    probability=leakage,
                    rng=np.random.default_rng(90_000 + seed),
                )
                leak_pass, leak_max, leak_bound = gate(
                    contaminated,
                    sample["lengths"],
                    sample["labels"],
                    n_categories=config.n_receiver_categories,
                    margin=margin,
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "kappa": kappa,
                        "seed": seed,
                        "n_entities": len(sample["labels"]),
                        "null_gate_pass": clean_pass,
                        "null_max_abs_difference": clean_max,
                        "null_max_simultaneous_bound": clean_bound,
                        "leakage_gate_pass": leak_pass,
                        "leakage_detected": not leak_pass,
                        "leakage_max_abs_difference": leak_max,
                        "leakage_max_simultaneous_bound": leak_bound,
                    }
                )
    summaries = []
    for scenario in raw["scenarios"]:
        for kappa in (0.0, 1.0):
            selected = [
                row for row in rows
                if row["scenario"] == scenario and row["kappa"] == kappa
            ]
            summaries.append(
                {
                    "scenario": scenario,
                    "kappa": kappa,
                    "n_replicates": len(selected),
                    "false_fail_rate": float(np.mean([not row["null_gate_pass"] for row in selected])),
                    "leakage_detection_power": float(np.mean([row["leakage_detected"] for row in selected])),
                    "required_false_fail_max": float(receiver_gate["stress_test"]["maximum_false_fail_rate"]),
                    "required_power_min": float(receiver_gate["stress_test"]["required_detection_power"]),
                }
            )
    candidate_pass = all(
        row["false_fail_rate"] <= row["required_false_fail_max"]
        and row["leakage_detection_power"] >= row["required_power_min"]
        for row in summaries
    )
    for name, values in (("calibration_raw.csv", rows), ("calibration_summary.csv", summaries)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (output / "calibration_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "receiver_gate_v2.1_candidate_calibration",
                "status": "PASS" if candidate_pass else "FAIL",
                "seeds_per_cell": args.seeds,
                "cells": len(summaries),
                "n_entities": int(raw["data"]["n_test"]),
                "interval": "Bonferroni simultaneous normal CI over entity-cluster category frequencies",
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
