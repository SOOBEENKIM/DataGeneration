from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import norm
import yaml

from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_receiver_only


def worker(payload: tuple) -> tuple[list[dict], list[dict]]:
    raw, scenario, kappa, seed = payload
    config = BenchmarkConfig.from_mapping(raw, scenario, kappa)
    sample = generate_receiver_only(
        config,
        n_entities=int(raw["data"]["n_test"]),
        seed=seed,
        split_id=1,
    )
    state = sample["gap_state"].astype(float)
    valid = sample["valid_mask"]
    labels = sample["labels"]
    positions = []
    for position in range(state.shape[1]):
        chosen = [valid[:, position] & (labels == label) for label in (0, 1)]
        occupancy = [float(state[index, position].mean()) for index in chosen]
        positions.append(
            {
                "scenario": scenario,
                "kappa": kappa,
                "seed": seed,
                "position": position,
                "n_at_risk_y0": int(chosen[0].sum()),
                "n_at_risk_y1": int(chosen[1].sum()),
                "occupancy_y0": occupancy[0],
                "occupancy_y1": occupancy[1],
                "difference": occupancy[1] - occupancy[0],
            }
        )
    segments = []
    relative = np.divide(
        np.arange(state.shape[1])[None, :] + 0.5,
        sample["lengths"][:, None],
    )
    for name, segment_mask in (
        ("early", relative < 1 / 3),
        ("middle", (relative >= 1 / 3) & (relative < 2 / 3)),
        ("late", relative >= 2 / 3),
    ):
        entity_means = np.divide(
            (state * valid * segment_mask).sum(1),
            (valid * segment_mask).sum(1),
        )
        occupancy = [
            float(entity_means[labels == label].mean()) for label in (0, 1)
        ]
        segments.append(
            {
                "scenario": scenario,
                "kappa": kappa,
                "seed": seed,
                "segment": name,
                "n_y0": int(np.sum(labels == 0)),
                "n_y1": int(np.sum(labels == 1)),
                "occupancy_y0": occupancy[0],
                "occupancy_y1": occupancy[1],
                "difference": occupancy[1] - occupancy[0],
            }
        )
    return positions, segments


def summarize(
    rows: list[dict],
    keys: tuple[str, ...],
    *,
    stationary: float,
    margin: float,
    direction_max: float,
) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group, values in sorted(groups.items()):
        differences = np.asarray([row["difference"] for row in values])
        occupancy0 = np.asarray([row["occupancy_y0"] for row in values])
        occupancy1 = np.asarray([row["occupancy_y1"] for row in values])
        positive = float(np.mean(differences > 0))
        direction = max(positive, 1 - positive)
        record = dict(zip(keys, group))
        record.update(
            {
                "replicates": len(values),
                "occupancy_y0_mean": float(occupancy0.mean()),
                "occupancy_y1_mean": float(occupancy1.mean()),
                "difference_mean": float(differences.mean()),
                "difference_q025": float(np.quantile(differences, 0.025)),
                "difference_q975": float(np.quantile(differences, 0.975)),
                "same_direction_fraction": direction,
                "pass": bool(
                    abs(occupancy0.mean() - stationary) <= margin
                    and abs(occupancy1.mean() - stationary) <= margin
                    and abs(differences.mean()) <= margin
                    and direction <= direction_max
                ),
            }
        )
        if "n_at_risk_y0" in values[0]:
            record["minimum_n_at_risk_y0"] = min(
                row["n_at_risk_y0"] for row in values
            )
            record["minimum_n_at_risk_y1"] = min(
                row["n_at_risk_y1"] for row in values
            )
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_v2/main_v2_2_candidate.yaml")
    parser.add_argument("--output-root", default="artifacts/benchmark_v2/positionwise_v2_2")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    rule = raw["evaluation"]["positionwise"]
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    tasks = [
        (raw, scenario, kappa, int(raw["seeds"]["base"]) + replicate)
        for scenario in raw["scenarios"]
        for kappa in (0.0, 1.0)
        for replicate in range(args.seeds)
    ]
    started = time.perf_counter()
    positions, segments = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for position_rows, segment_rows in executor.map(worker, tasks):
            positions.extend(position_rows)
            segments.extend(segment_rows)
    position_summary = summarize(
        positions,
        ("scenario", "kappa", "position"),
        stationary=float(rule["stationary_probability"]),
        margin=float(rule["equivalence_margin"]),
        direction_max=float(rule["monte_carlo_fixed_direction_fraction_max"]),
    )
    segment_summary = summarize(
        segments,
        ("scenario", "kappa", "segment"),
        stationary=float(rule["stationary_probability"]),
        margin=float(rule["segment_equivalence_margin"]),
        direction_max=float(rule["monte_carlo_fixed_direction_fraction_max"]),
    )
    minimum_at_risk = int(rule["minimum_at_risk_per_label"])
    prefix = int(rule["hard_prefix_positions"])
    for row in position_summary:
        row["hard_gate"] = bool(
            row["position"] < prefix
            or (
                row["minimum_n_at_risk_y0"] >= minimum_at_risk
                and row["minimum_n_at_risk_y1"] >= minimum_at_risk
            )
        )
    hard_positions = [row for row in position_summary if row["hard_gate"]]
    passed = all(row["pass"] for row in hard_positions + segment_summary)
    write_csv(output / "positionwise_raw.csv", positions)
    write_csv(output / "positionwise_summary.csv", position_summary)
    write_csv(output / "segment_raw.csv", segments)
    write_csv(output / "segment_summary.csv", segment_summary)
    n_comparisons = prefix
    z = float(norm.ppf(1 - 0.05 / (2 * n_comparisons)))
    conservative_minimum = int(
        np.ceil(
            float(rule["stationary_probability"])
            * (1 - float(rule["stationary_probability"]))
            * (z / float(rule["equivalence_margin"])) ** 2
        )
    )
    (output / "power_analysis.json").write_text(
        json.dumps(
            {
                "familywise_alpha": 0.05,
                "hard_prefix_comparisons": n_comparisons,
                "stationary_probability": float(rule["stationary_probability"]),
                "equivalence_margin": float(rule["equivalence_margin"]),
                "bonferroni_z": z,
                "minimum_positive_entities_analytical": conservative_minimum,
                "preregistered_minimum_at_risk_per_label": minimum_at_risk,
                "selection_uses_observed_position_results": False,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "positionwise_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "positionwise_stationarity_v2.2-candidate",
                "status": "PASS" if passed else "FAIL",
                "seeds_per_cell": args.seeds,
                "hard_position_cells": len(hard_positions),
                "descriptive_position_cells": len(position_summary) - len(hard_positions),
                "segment_cells": len(segment_summary),
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
