from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from benchmarks.types import SequenceBatch


def load_batch(path: Path) -> SequenceBatch:
    with np.load(path, allow_pickle=False) as values:
        return SequenceBatch(
            **{key: values[key] for key in SequenceBatch.__dataclass_fields__}
        )


def multiplier_band(
    values: np.ndarray,
    valid: np.ndarray,
    labels: np.ndarray,
    *,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Entity-cluster multiplier band for all positionwise mean differences."""
    means, influence = [], []
    for label in (0, 1):
        selected = labels == label
        observed = np.divide(
            (values[selected] * valid[selected]).sum(0),
            valid[selected].sum(0),
        )
        means.append(observed)
        centered = (values[selected] - observed) * valid[selected]
        influence.append(centered / valid[selected].sum(0))
    observed_difference = means[1] - means[0]
    deviations = np.empty((resamples, values.shape[1]), dtype=float)
    chunk = 100
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        draws = []
        for matrix in influence:
            weights = rng.standard_normal((stop - start, matrix.shape[0]))
            draws.append(weights @ matrix)
        deviations[start:stop] = draws[1] - draws[0]
    critical = float(np.quantile(np.max(np.abs(deviations), axis=1), 0.95))
    return (
        observed_difference,
        observed_difference - critical,
        observed_difference + critical,
        critical,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_v2/main.yaml")
    parser.add_argument("--data-root", default="data/benchmark_v2")
    parser.add_argument("--output-root", default="artifacts/benchmark_v2/gates")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    root, output = Path(args.data_root), Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows, summaries = [], []
    overall = True
    for scenario_index, scenario in enumerate(raw["scenarios"]):
        for kappa in raw["coupling"]["kappas"]:
            folder = root / scenario / f"kappa_{float(kappa):.2f}"
            batch = load_batch(folder / "test.npz")
            with np.load(folder / "test_audit_latent.npz") as latent:
                state = latent["gap_state"].astype(float)
            valid = batch.valid_mask.astype(float)
            difference, low, high, critical = multiplier_band(
                state,
                valid,
                batch.y_entity,
                resamples=args.resamples,
                rng=np.random.default_rng(
                    41_000 + 1000 * scenario_index + int(float(kappa) * 100)
                ),
            )
            occupancy = []
            for label in (0, 1):
                chosen = batch.y_entity == label
                occupancy.append(
                    np.divide(
                        (state[chosen] * valid[chosen]).sum(0),
                        valid[chosen].sum(0),
                    )
                )
            cell_pass = True
            for position in range(state.shape[1]):
                position_pass = (
                    abs(occupancy[0][position] - 0.30) <= 0.03
                    and abs(occupancy[1][position] - 0.30) <= 0.03
                    and abs(difference[position]) <= 0.03
                    and low[position] <= 0 <= high[position]
                )
                cell_pass &= position_pass
                rows.append(
                    {
                        "scenario": scenario,
                        "kappa": float(kappa),
                        "position": position,
                        "occupancy_y0": float(occupancy[0][position]),
                        "occupancy_y1": float(occupancy[1][position]),
                        "occupancy_difference": float(difference[position]),
                        "simultaneous_ci_low": float(low[position]),
                        "simultaneous_ci_high": float(high[position]),
                        "simultaneous_critical_radius": critical,
                        "pass": bool(position_pass),
                    }
                )
            overall &= cell_pass
            summaries.append(
                {
                    "scenario": scenario,
                    "kappa": float(kappa),
                    "n_positions": state.shape[1],
                    "n_failed_positions": int(
                        sum(
                            not row["pass"]
                            for row in rows[-state.shape[1] :]
                        )
                    ),
                    "simultaneous_critical_radius": critical,
                    "pass": bool(cell_pass),
                }
            )
    for name, values in (
        ("positionwise_stationarity_simultaneous.csv", rows),
        ("positionwise_stationarity_summary.csv", summaries),
    ):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (output / "positionwise_stationarity_simultaneous.json").write_text(
        json.dumps(
            {
                "schema_version": "positionwise_entity_cluster_band_v2.0",
                "method": "entity-cluster Gaussian multiplier bootstrap; "
                "95% max-absolute-deviation simultaneous band",
                "resamples": args.resamples,
                "status": "PASS" if overall else "FAIL",
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
