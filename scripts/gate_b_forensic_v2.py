from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from benchmarks.temporal_coupling_v2 import (
    BenchmarkConfig,
    generate_fixed_binning_split,
)
from benchmarks.types import SequenceBatch
from eval.behavior_summaries_v2 import (
    compute_behavior_summaries_v2,
    fit_short_gap_threshold,
)
from eval.coherence_v2 import (
    coherence_bin_diagnostics,
    fit_coherence_reference,
)
from eval.joint_association_v2 import (
    association_statistics,
    bootstrap_delta_interval,
)
from generators.empirical_conditional_block import EmpiricalConditionalBlock
from generators.empirical_conditional_iid import EmpiricalConditionalIID
from generators.sampling_plan import SamplingPlan


def load_batch(path: Path) -> SequenceBatch:
    with np.load(path, allow_pickle=False) as values:
        return SequenceBatch(
            **{key: values[key] for key in SequenceBatch.__dataclass_fields__}
        )


def subset(values: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in values.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def stratified_halves(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left, right = [], []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        middle = len(indices) // 2
        left.extend(indices[:middle])
        right.extend(indices[middle:])
    return np.asarray(left), np.asarray(right)


def distribution_row(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    scenario: str,
    kappa: float,
    seed: int,
    generator: str,
) -> dict:
    finite = values[np.isfinite(values)]
    row = {
        "scenario": scenario,
        "kappa": kappa,
        "seed": seed,
        "generator": generator,
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)),
    }
    for quantile in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0):
        row[f"q{int(quantile * 100):02d}"] = float(np.quantile(finite, quantile))
    for label in (0, 1):
        selected = values[(labels == label) & np.isfinite(values)]
        row[f"mean_y{label}"] = float(selected.mean())
        row[f"std_y{label}"] = float(selected.std(ddof=1))
        row[f"q05_y{label}"] = float(np.quantile(selected, 0.05))
        row[f"q50_y{label}"] = float(np.quantile(selected, 0.50))
        row[f"q95_y{label}"] = float(np.quantile(selected, 0.95))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_v2/main.yaml")
    parser.add_argument("--data-root", default="data/benchmark_v2")
    parser.add_argument("--output-root", default="artifacts/benchmark_v2/gate_b_forensics")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--kappas", type=float, nargs="+")
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    scenario = "joint_semimarkov_v2b"
    bin_rows, distribution_rows, association_rows, score_rows = [], [], [], []
    kappas = raw["coupling"]["kappas"] if args.kappas is None else args.kappas
    for kappa in kappas:
        folder = Path(args.data_root) / scenario / f"kappa_{float(kappa):.2f}"
        train, test = load_batch(folder / "train.npz"), load_batch(folder / "test.npz")
        meta = json.loads((folder / "meta.json").read_text())
        tau, edges = np.asarray(meta["tau"]), np.asarray(meta["bin_edges"])
        cutoff = fit_short_gap_threshold(train, tau=tau)
        real = compute_behavior_summaries_v2(
            test,
            tau=tau,
            short_gap_threshold=cutoff,
            window_width=float(raw["data"]["window_width"]),
        )
        real_delta = float(
            association_statistics(real["joint_alignment"], test.y_entity)[
                "delta_joint"
            ]
        )
        plan = SamplingPlan.from_batch(test)
        iid = EmpiricalConditionalIID()
        iid.fit(train, seed=0)
        blocks = {}
        for length in (1, 2, 4, 8, "full"):
            block = EmpiricalConditionalBlock(length)
            block.fit(train, seed=0)
            blocks[f"block_{length}"] = block
        base = replace(
            BenchmarkConfig.from_mapping(raw, scenario, float(kappa)),
            bin_edges=edges,
        )
        references = {
            bins: fit_coherence_reference(
                real,
                test.y_entity,
                n_bins=bins,
                min_bin_count=int(raw["evaluation"]["minimum_bin_count"]),
            )
            for bins in (4, 8)
        }
        for seed in range(1, args.seeds + 1):
            shuffled = np.random.default_rng(20_000 + seed).permutation(test.y_entity)
            oracle_batch, _ = generate_fixed_binning_split(
                replace(base, randomness_nonce=seed),
                n_entities=len(test.lengths),
                seed=int(raw["seeds"]["base"]) + int(raw["seeds"]["test_offset"]),
                split_id=1,
                prefix=f"forensic-oracle-{seed}",
            )
            iid_batch = iid.sample(plan, seed=seed)
            generated = {
                "c1": (real, shuffled),
                "iid": (
                    compute_behavior_summaries_v2(
                        iid_batch,
                        tau=tau,
                        short_gap_threshold=cutoff,
                        window_width=float(raw["data"]["window_width"]),
                    ),
                    iid_batch.y_entity,
                ),
                "oracle": (
                    compute_behavior_summaries_v2(
                        oracle_batch,
                        tau=tau,
                        short_gap_threshold=cutoff,
                        window_width=float(raw["data"]["window_width"]),
                    ),
                    oracle_batch.y_entity,
                ),
            }
            left, right = stratified_halves(test.y_entity, seed)
            generated["c0"] = (subset(real, right), test.y_entity[right])
            for name, block in blocks.items():
                batch = block.sample(plan, seed=seed)
                generated[name] = (
                    compute_behavior_summaries_v2(
                        batch,
                        tau=tau,
                        short_gap_threshold=cutoff,
                        window_width=float(raw["data"]["window_width"]),
                    ),
                    batch.y_entity,
                )
            real_reference_values = (
                real["joint_alignment"] if seed != 0 else real["joint_alignment"]
            )
            for generator, (summaries, labels) in generated.items():
                joint = summaries["joint_alignment"]
                distribution_rows.append(
                    distribution_row(
                        joint,
                        labels,
                        scenario=scenario,
                        kappa=float(kappa),
                        seed=seed,
                        generator=generator,
                    )
                )
                stats = association_statistics(joint, labels, real_delta=real_delta)
                low, high = bootstrap_delta_interval(
                    joint,
                    labels,
                    resamples=args.bootstrap_resamples,
                    seed=50_000 + seed,
                )
                for channel, values in summaries.items():
                    channel_stats = association_statistics(
                        values, labels,
                        real_delta=(
                            association_statistics(real[channel], test.y_entity)[
                                "delta_joint"
                            ]
                        ),
                    )
                    association_rows.append(
                        {
                            "scenario": scenario,
                            "kappa": float(kappa),
                            "seed": seed,
                            "generator": generator,
                            "channel": channel,
                            **channel_stats,
                            "delta_ci_low": low if channel == "joint_alignment" else None,
                            "delta_ci_high": high if channel == "joint_alignment" else None,
                        }
                    )
                for bins, reference in references.items():
                    diagnostic = coherence_bin_diagnostics(
                        real_reference_values,
                        test.y_entity,
                        joint,
                        labels,
                        reference.thresholds_by_channel["joint_alignment"],
                        reference.min_bin_count,
                    )
                    score_rows.append(
                        {
                            "scenario": scenario,
                            "kappa": float(kappa),
                            "seed": seed,
                            "generator": generator,
                            "n_bins": bins,
                            "dropped_bin_macro_gap": diagnostic["dropped_bin_macro_gap"],
                            "occupancy_penalized_gap": diagnostic["occupancy_penalized_gap"],
                            "worst_score_macro_gap": diagnostic["worst_score_macro_gap"],
                            "invalid_bin_count": diagnostic["invalid_bin_count"],
                            "all_bins_valid": diagnostic["all_bins_valid"],
                            "association_recovery_error": stats["association_recovery_error"],
                        }
                    )
                    for row in diagnostic["rows"]:
                        bin_rows.append(
                            {
                                "scenario": scenario,
                                "kappa": float(kappa),
                                "seed": seed,
                                "generator": generator,
                                "n_bins": bins,
                                **row,
                            }
                        )
            # Sampling-floor C0 is left-reference versus right evaluation.
            left_stats = association_statistics(
                real["joint_alignment"][left], test.y_entity[left]
            )
            right_stats = association_statistics(
                real["joint_alignment"][right], test.y_entity[right]
            )
            association_rows.append(
                {
                    "scenario": scenario,
                    "kappa": float(kappa),
                    "seed": seed,
                    "generator": "c0_sampling_floor",
                    "channel": "joint_alignment",
                    **right_stats,
                    "association_recovery_error": abs(
                        float(left_stats["delta_joint"])
                        - float(right_stats["delta_joint"])
                    ),
                    "recovery_ratio": None,
                    "sign_consistent": None,
                    "delta_ci_low": None,
                    "delta_ci_high": None,
                }
            )
    write_csv(output / "bin_decomposition.csv", bin_rows)
    write_csv(output / "joint_alignment_distributions.csv", distribution_rows)
    write_csv(output / "continuous_association.csv", association_rows)
    write_csv(output / "endpoint_comparison.csv", score_rows)
    summary_rows = []
    keys = sorted(
        {
            (row["kappa"], row["generator"], row["n_bins"])
            for row in score_rows
        }
    )
    for kappa, generator, bins in keys:
        selected = [
            row for row in score_rows
            if row["kappa"] == kappa
            and row["generator"] == generator
            and row["n_bins"] == bins
        ]
        summary_rows.append(
            {
                "kappa": kappa,
                "generator": generator,
                "n_bins": bins,
                "replicates": len(selected),
                **{
                    f"{metric}_mean": float(
                        np.mean([row[metric] for row in selected])
                    )
                    for metric in (
                        "dropped_bin_macro_gap",
                        "occupancy_penalized_gap",
                        "worst_score_macro_gap",
                        "invalid_bin_count",
                        "association_recovery_error",
                    )
                },
            }
        )
    write_csv(output / "endpoint_comparison_summary.csv", summary_rows)
    elapsed = time.perf_counter() - started
    (output / "forensic_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "gate_b_forensic_v2.2-candidate",
                "status": "COMPLETE",
                "scenario": scenario,
                "seeds": args.seeds,
                "bootstrap_resamples": args.bootstrap_resamples,
                "elapsed_seconds": elapsed,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
