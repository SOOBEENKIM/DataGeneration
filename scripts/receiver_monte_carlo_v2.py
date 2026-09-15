from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from benchmarks.receiver_diagnostics import (
    entity_category_counts,
    entity_category_frequencies,
    pooled_and_balanced_tvd,
    positionwise_tvd,
    receiver_run_statistics,
)
from benchmarks.temporal_coupling_v2 import (
    BenchmarkConfig,
    generate_receiver_only,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    output = []
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    metrics = (
        "pooled_tvd", "entity_balanced_tvd", "position_tvd_max",
        "position_tvd_mean", "repeat_rate_difference", "mean_run_difference",
        "max_abs_category_frequency_difference",
    )
    for key, values in groups.items():
        record = dict(zip(group_keys, key))
        record["n_replicates"] = len(values)
        for metric in metrics:
            vector = np.array([value[metric] for value in values], dtype=float)
            record[f"{metric}_mean"] = float(vector.mean())
            record[f"{metric}_median"] = float(np.median(vector))
            record[f"{metric}_q95"] = float(np.quantile(vector, 0.95))
            record[f"{metric}_q99"] = float(np.quantile(vector, 0.99))
        record["raw_tvd_false_fail_rate_at_0.02"] = float(
            np.mean([value["pooled_tvd"] > 0.02 for value in values])
        )
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_v2/main.yaml")
    parser.add_argument("--output-root", default="artifacts/benchmark_v2/receiver_diagnosis")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    raw_config = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    base_n = int(raw_config["data"]["n_test"])
    started = time.perf_counter()
    rows = []
    for scenario in ("markov_persistence_v2a", "joint_semimarkov_v2b"):
        for kappa in (0.0, 1.0):
            base = BenchmarkConfig.from_mapping(raw_config, scenario, kappa)
            for scale in (1, 2, 4):
                n_entities = base_n * scale
                for replicate in range(args.seeds):
                    seed = int(raw_config["seeds"]["base"]) + replicate
                    sample = generate_receiver_only(
                        base, n_entities=n_entities, seed=seed, split_id=1
                    )
                    categories = sample["categories"]
                    lengths, labels = sample["lengths"], sample["labels"]
                    counts = entity_category_counts(
                        categories, lengths, base.n_receiver_categories
                    )
                    frequencies = entity_category_frequencies(counts)
                    pooled, balanced = pooled_and_balanced_tvd(counts, labels)
                    positions = positionwise_tvd(
                        categories, lengths, labels, base.n_receiver_categories
                    )
                    runs = receiver_run_statistics(categories, lengths, labels)
                    repeat0, repeat1 = (
                        runs[0]["entity_mean_repeat_rate"],
                        runs[1]["entity_mean_repeat_rate"],
                    )
                    mean_run0, mean_run1 = (
                        runs[0]["mean_run_length"], runs[1]["mean_run_length"]
                    )
                    frequency_difference = (
                        frequencies[labels == 1].mean(0)
                        - frequencies[labels == 0].mean(0)
                    )
                    mean_repeat = float(
                        np.mean([run["entity_mean_repeat_rate"] for run in runs])
                    )
                    total_rows = int(lengths.sum())
                    design_effect = (1 + mean_repeat) / max(1 - mean_repeat, 1e-6)
                    rows.append(
                        {
                            "scenario": scenario,
                            "kappa": kappa,
                            "n_scale": scale,
                            "n_entities": n_entities,
                            "seed": seed,
                            "n_y0": int(np.sum(labels == 0)),
                            "n_y1": int(np.sum(labels == 1)),
                            "rows_y0": int(lengths[labels == 0].sum()),
                            "rows_y1": int(lengths[labels == 1].sum()),
                            "length_mean_y0": float(lengths[labels == 0].mean()),
                            "length_mean_y1": float(lengths[labels == 1].mean()),
                            "pooled_tvd": pooled,
                            "entity_balanced_tvd": balanced,
                            "position_tvd_max": float(
                                np.nanmax([position["receiver_tvd"] for position in positions])
                            ),
                            "position_tvd_mean": float(
                                np.nanmean([position["receiver_tvd"] for position in positions])
                            ),
                            "repeat_rate_y0": repeat0,
                            "repeat_rate_y1": repeat1,
                            "repeat_rate_difference": repeat1 - repeat0,
                            "mean_run_y0": mean_run0,
                            "mean_run_y1": mean_run1,
                            "mean_run_difference": mean_run1 - mean_run0,
                            "max_abs_category_frequency_difference": float(
                                np.max(np.abs(frequency_difference))
                            ),
                            "max_category_frequency_difference_category": int(
                                np.argmax(np.abs(frequency_difference))
                            ),
                            "total_rows": total_rows,
                            "estimated_design_effect": design_effect,
                            "estimated_effective_rows": total_rows / design_effect,
                            "entity_effective_n": n_entities,
                        }
                    )
    summary = summarize(rows, ("scenario", "kappa", "n_scale", "n_entities"))
    write_csv(output / "monte_carlo_raw.csv", rows)
    write_csv(output / "monte_carlo_summary.csv", summary)
    tvd_rows = [
        {
            key: row[key]
            for key in (
                "scenario", "kappa", "n_scale", "n_entities", "n_replicates",
                "pooled_tvd_mean", "pooled_tvd_median", "pooled_tvd_q95",
                "entity_balanced_tvd_mean", "entity_balanced_tvd_q95",
                "raw_tvd_false_fail_rate_at_0.02",
            )
        }
        for row in summary
    ]
    write_csv(output / "tvd_vs_n.csv", tvd_rows)
    write_csv(
        output / "effective_sample_size.csv",
        [
            {
                key: row[key]
                for key in (
                    "scenario", "kappa", "n_scale", "seed", "n_entities",
                    "total_rows", "estimated_design_effect",
                    "estimated_effective_rows", "entity_effective_n",
                    "repeat_rate_y0", "repeat_rate_y1",
                )
            }
            for row in rows
        ],
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    for scenario in ("markov_persistence_v2a", "joint_semimarkov_v2b"):
        for kappa in (0.0, 1.0):
            selected = [
                row for row in summary
                if row["scenario"] == scenario and row["kappa"] == kappa
            ]
            axis.plot(
                [row["n_entities"] for row in selected],
                [row["pooled_tvd_mean"] for row in selected],
                marker="o", label=f"{scenario}, k={kappa:g}",
            )
    axis.axhline(0.02, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="entities", ylabel="mean pooled-row receiver TVD")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "tvd_vs_n.png", dpi=160)
    plt.close(figure)
    elapsed = time.perf_counter() - started
    (output / "monte_carlo_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "receiver_monte_carlo_v2.0",
                "status": "COMPLETE",
                "replicates": len(rows),
                "seeds_per_cell": args.seeds,
                "elapsed_seconds": elapsed,
                "production_receiver_code_shared": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
