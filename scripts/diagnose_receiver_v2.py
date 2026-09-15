from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.receiver_diagnostics import (
    entity_category_counts,
    entity_category_frequencies,
    permutation_null,
    pooled_and_balanced_tvd,
    positionwise_tvd,
    receiver_run_statistics,
    segmentwise_tvd,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/benchmark_v2")
    parser.add_argument("--output-root", default="artifacts/benchmark_v2/receiver_diagnosis")
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()
    started = time.perf_counter()
    data_root, output = Path(args.data_root), Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    counts_rows, position_rows, segment_rows = [], [], []
    permutation_rows, run_rows = [], []
    for scenario_root in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for kappa_root in sorted(scenario_root.glob("kappa_*")):
            for split in ("train", "test"):
                with np.load(kappa_root / f"{split}.npz", allow_pickle=False) as data:
                    categories = data["x_cat"][..., 0]
                    lengths = data["lengths"]
                    labels = data["y_entity"]
                n_categories = 64
                counts = entity_category_counts(categories, lengths, n_categories)
                frequencies = entity_category_frequencies(counts)
                pooled, balanced = pooled_and_balanced_tvd(counts, labels)
                common = {
                    "scenario": scenario_root.name,
                    "kappa": float(kappa_root.name.split("_", 1)[1]),
                    "split": split,
                }
                for label in (0, 1):
                    selected = labels == label
                    for category in range(n_categories):
                        counts_rows.append(
                            {
                                **common,
                                "label": label,
                                "category": category,
                                "row_count": int(counts[selected, category].sum()),
                                "entity_balanced_mean_frequency": float(
                                    frequencies[selected, category].mean()
                                ),
                                "length_frequency_correlation": float(
                                    np.corrcoef(lengths[selected], frequencies[selected, category])[0, 1]
                                ),
                                "pooled_row_tvd": pooled,
                                "entity_balanced_tvd": balanced,
                                "length_mean": float(lengths[selected].mean()),
                                "length_std": float(lengths[selected].std()),
                            }
                        )
                position_rows.extend(
                    {**common, **row}
                    for row in positionwise_tvd(categories, lengths, labels, n_categories)
                )
                segment_rows.extend(
                    {**common, **row}
                    for row in segmentwise_tvd(categories, lengths, labels, n_categories)
                )
                run_rows.extend(
                    {**common, **row}
                    for row in receiver_run_statistics(categories, lengths, labels)
                )
                permutation_rows.append(
                    {
                        **common,
                        **permutation_null(
                            counts, labels, n_permutations=args.permutations,
                            seed=20260728 + (0 if split == "train" else 1),
                        ),
                    }
                )
    write_csv(output / "receiver_counts.csv", counts_rows)
    write_csv(output / "receiver_tvd_by_position.csv", position_rows)
    write_csv(output / "receiver_tvd_by_segment.csv", segment_rows)
    write_csv(output / "receiver_permutation_null.csv", permutation_rows)
    write_csv(output / "receiver_run_statistics.csv", run_rows)
    elapsed = time.perf_counter() - started
    diagnosis = {
        "schema_version": "receiver_diagnosis_v2.0",
        "elapsed_seconds": elapsed,
        "n_permutations": args.permutations,
        "datasets": len(permutation_rows),
        "status": "COMPLETE",
        "interpretation_warning": (
            "Permutation tails calibrate the finite-sample clustered null; "
            "a large p-value is not equivalence evidence."
        ),
    }
    (output / "receiver_diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    (output / "receiver_diagnosis.md").write_text(
        "# Receiver diagnosis\n\n"
        f"Receiver-only artifact diagnosis completed in {elapsed:.2f} seconds over "
        f"{len(permutation_rows)} scenario/kappa/split datasets with "
        f"{args.permutations} entity-label permutations each.\n\n"
        "This file records raw diagnostics only. H1–H4 conclusions are finalized "
        "after the no-model Monte Carlo in `docs/benchmark_v2/receiver_gate_diagnosis.md`.\n"
    )


if __name__ == "__main__":
    main()
