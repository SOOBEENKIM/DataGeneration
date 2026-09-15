from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import beta
import yaml

from benchmarks.receiver_diagnostics import entity_category_frequencies
from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_receiver_only
from scripts.calibrate_receiver_gate_v2_1 import simultaneous_interval


def category_counts(
    categories: np.ndarray, lengths: np.ndarray, n_categories: int
) -> np.ndarray:
    valid = np.arange(categories.shape[1])[None, :] < lengths[:, None]
    entities = np.broadcast_to(np.arange(len(lengths))[:, None], categories.shape)
    counts = np.zeros((len(lengths), n_categories), dtype=np.int32)
    np.add.at(counts, (entities[valid], categories[valid]), 1)
    return counts


def gate_counts(
    counts: np.ndarray, labels: np.ndarray, margin: float
) -> tuple[bool, float]:
    frequencies = entity_category_frequencies(counts)
    _, low, high = simultaneous_interval(frequencies, labels)
    bound = float(np.max(np.maximum(np.abs(low), np.abs(high))))
    return bool(np.all(low >= -margin) and np.all(high <= margin)), bound


def contaminated_counts(
    base_counts: np.ndarray,
    categories: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    *,
    target_category: int,
    probability: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    valid = np.arange(categories.shape[1])[None, :] < lengths[:, None]
    changed = (
        valid
        & (labels[:, None] == 1)
        & (categories != target_category)
        & (rng.random(categories.shape) < probability)
    )
    entities = np.broadcast_to(np.arange(len(lengths))[:, None], categories.shape)
    result = base_counts.copy()
    np.add.at(result, (entities[changed], categories[changed]), -1)
    np.add.at(result, (entities[changed], target_category), 1)
    return result


def exact_binomial_interval(successes: int, trials: int) -> tuple[float, float]:
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return low, high


def worker(payload: tuple) -> list[dict]:
    raw, scenario, kappa, seed, categories, magnitudes = payload
    config = BenchmarkConfig.from_mapping(raw, scenario, kappa)
    sample = generate_receiver_only(
        config,
        n_entities=int(raw["data"]["n_test"]),
        seed=seed,
        split_id=1,
    )
    counts = category_counts(
        sample["categories"], sample["lengths"], config.n_receiver_categories
    )
    margin = float(raw["receiver_gate"]["primary"]["category_practical_margin"])
    null_pass, null_bound = gate_counts(counts, sample["labels"], margin)
    rows = []
    for target in categories:
        for magnitude in magnitudes:
            changed = contaminated_counts(
                counts,
                sample["categories"],
                sample["lengths"],
                sample["labels"],
                target_category=target,
                probability=magnitude,
                seed=700_000 + 1000 * seed + 10 * target + int(magnitude * 100),
            )
            leak_pass, leak_bound = gate_counts(changed, sample["labels"], margin)
            rows.append(
                {
                    "scenario": scenario,
                    "kappa": kappa,
                    "seed": seed,
                    "n_entities": len(sample["labels"]),
                    "null_gate_pass": null_pass,
                    "null_max_simultaneous_bound": null_bound,
                    "leakage_category": target,
                    "leakage_magnitude": magnitude,
                    "leakage_detected": not leak_pass,
                    "leakage_max_simultaneous_bound": leak_bound,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_v2/main_v2_2_candidate.yaml")
    parser.add_argument("--output-root", default="artifacts/benchmark_v2/receiver_diagnosis/v2_2_candidate")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    categories = tuple(raw["receiver_gate"]["stress_test"]["leakage_categories"])
    magnitudes = tuple(raw["receiver_gate"]["stress_test"]["leakage_magnitudes"])
    tasks = [
        (
            raw,
            scenario,
            kappa,
            int(raw["seeds"]["base"]) + replicate,
            categories,
            magnitudes,
        )
        for scenario in raw["scenarios"]
        for kappa in (0.0, 1.0)
        for replicate in range(args.seeds)
    ]
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(worker, tasks):
            rows.extend(result)
    with (output / "calibration_raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for scenario in raw["scenarios"]:
        for kappa in (0.0, 1.0):
            cell = [
                row for row in rows
                if row["scenario"] == scenario and row["kappa"] == kappa
            ]
            one_per_seed = {
                row["seed"]: row["null_gate_pass"] for row in cell
            }
            failures = sum(not value for value in one_per_seed.values())
            ff_low, ff_high = exact_binomial_interval(failures, args.seeds)
            for target in categories:
                for magnitude in magnitudes:
                    selected = [
                        row for row in cell
                        if row["leakage_category"] == target
                        and row["leakage_magnitude"] == magnitude
                    ]
                    detections = sum(row["leakage_detected"] for row in selected)
                    power_low, power_high = exact_binomial_interval(
                        detections, args.seeds
                    )
                    summaries.append(
                        {
                            "scenario": scenario,
                            "kappa": kappa,
                            "leakage_category": target,
                            "leakage_magnitude": magnitude,
                            "n_clean_seeds": args.seeds,
                            "false_failures": failures,
                            "false_fail_rate": failures / args.seeds,
                            "false_fail_ci_low": ff_low,
                            "false_fail_ci_high": ff_high,
                            "detections": detections,
                            "detection_power": detections / args.seeds,
                            "detection_power_ci_low": power_low,
                            "detection_power_ci_high": power_high,
                        }
                    )
    with (output / "calibration_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    maximum_false_fail = float(
        raw["receiver_gate"]["stress_test"]["maximum_false_fail_rate"]
    )
    power_requirements = raw["receiver_gate"]["stress_test"]["required_power"]
    passed = all(
        row["false_fail_ci_high"] <= maximum_false_fail
        and row["detection_power_ci_low"]
        >= float(power_requirements[str(row["leakage_magnitude"])])
        for row in summaries
    )
    (output / "calibration_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "receiver_gate_v2.2_candidate_calibration",
                "status": "PASS" if passed else "FAIL",
                "clean_seeds_per_cell": args.seeds,
                "scenario_kappa_cells": 4,
                "leakage_categories": categories,
                "leakage_magnitudes": magnitudes,
                "workers": args.workers,
                "n_entities": int(raw["data"]["n_test"]),
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
