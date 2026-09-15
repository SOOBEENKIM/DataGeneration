from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
import yaml

from benchmarks.receiver_diagnostics import tvd
from benchmarks.temporal_coupling_v2 import (
    BenchmarkConfig,
    generate_fixed_binning_split,
)
from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.behavior_summaries_v2 import (
    compute_behavior_summaries_v2,
    fit_short_gap_threshold,
)
from eval.coherence_v2 import (
    coherence_report,
    fit_coherence_reference,
)
from generators.empirical_conditional_block import EmpiricalConditionalBlock
from generators.empirical_conditional_iid import EmpiricalConditionalIID
from generators.sampling_plan import SamplingPlan


def load_batch(path: Path) -> SequenceBatch:
    with np.load(path, allow_pickle=False) as values:
        return SequenceBatch(
            **{key: values[key] for key in SequenceBatch.__dataclass_fields__}
        )


def subset(batch: SequenceBatch, indices: np.ndarray) -> SequenceBatch:
    return SequenceBatch(
        **{
            key: getattr(batch, key)[indices]
            for key in SequenceBatch.__dataclass_fields__
        }
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def structural_gap(report: dict, scenario: str) -> float:
    channels = (
        ("velocity", "gap", "fanout")
        if scenario.endswith("v2a")
        else ("joint_alignment",)
    )
    values = [report[channel]["macro_gap"] for channel in channels]
    return float(np.mean(values)) if all(value is not None for value in values) else float("nan")


def upper_confidence(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.mean() + 2.262 * values.std(ddof=1) / math.sqrt(len(values)))


def stratified_halves(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left, right = [], []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        midpoint = len(indices) // 2
        left.extend(indices[:midpoint])
        right.extend(indices[midpoint:])
    return np.asarray(left), np.asarray(right)


def positionwise_rows(
    batch: SequenceBatch,
    latent: dict[str, np.ndarray],
    *,
    scenario: str,
    kappa: float,
    bootstrap_resamples: int = 2000,
) -> tuple[list[dict], bool]:
    rng = np.random.default_rng(7301 + int(kappa * 100))
    rows, passed = [], True
    state = latent["gap_state"]
    raw_gap = latent["raw_gap"]
    categories = batch.x_cat[..., 0]
    for position in range(batch.x_num.shape[1]):
        valid = batch.lengths > position
        by_label = [np.flatnonzero(valid & (batch.y_entity == label)) for label in (0, 1)]
        occupancy = [float(state[index, position].mean()) for index in by_label]
        differences = np.empty(bootstrap_resamples)
        for iteration in range(bootstrap_resamples):
            means = [
                state[rng.choice(index, len(index), replace=True), position].mean()
                for index in by_label
            ]
            differences[iteration] = means[1] - means[0]
        low, high = np.quantile(differences, [0.025, 0.975])
        gap_ks = ks_2samp(
            raw_gap[by_label[0], position], raw_gap[by_label[1], position]
        ).statistic
        receiver_tvd = tvd(
            np.bincount(categories[by_label[0], position], minlength=64),
            np.bincount(categories[by_label[1], position], minlength=64),
        )
        row = {
            "scenario": scenario,
            "kappa": kappa,
            "position": position,
            "n_y0": len(by_label[0]),
            "n_y1": len(by_label[1]),
            "occupancy_y0": occupancy[0],
            "occupancy_y1": occupancy[1],
            "occupancy_difference": occupancy[1] - occupancy[0],
            "occupancy_difference_ci_low": float(low),
            "occupancy_difference_ci_high": float(high),
            "raw_gap_ks": float(gap_ks),
            "receiver_tvd": receiver_tvd,
        }
        rows.append(row)
        passed &= (
            abs(occupancy[0] - 0.30) <= 0.03
            and abs(occupancy[1] - 0.30) <= 0.03
            and abs(occupancy[1] - occupancy[0]) <= 0.03
            and low <= 0 <= high
        )
    return rows, bool(passed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_v2/main.yaml")
    parser.add_argument("--data-root", default="data/benchmark_v2")
    parser.add_argument("--output-root", default="artifacts/benchmark_v2/gates")
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    data_root, output = Path(args.data_root), Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    iid_rows = {2: [], 4: [], 8: []}
    oracle_rows, c0_rows, robustness_rows = [], [], []
    position_rows, ladder_rows = [], []
    status = {
        "gate_a_positionwise_stationarity": "PASS",
        "gate_b_iid_equivalence_to_c1": "PASS",
        "gate_c_dgp_oracle_near_c0": "PASS",
        "gate_e_bin_robustness": "PASS",
    }
    for scenario in raw["scenarios"]:
        for kappa in raw["coupling"]["kappas"]:
            folder = data_root / scenario / f"kappa_{float(kappa):.2f}"
            train, test = load_batch(folder / "train.npz"), load_batch(folder / "test.npz")
            with np.load(folder / "test_audit_latent.npz") as values:
                latent = {key: values[key] for key in values.files}
            meta = json.loads((folder / "meta.json").read_text())
            tau, edges = np.asarray(meta["tau"]), np.asarray(meta["bin_edges"])
            cutoff = fit_short_gap_threshold(train, tau=tau)
            real_summary = compute_behavior_summaries_v2(
                test, tau=tau, short_gap_threshold=cutoff,
                window_width=float(raw["data"]["window_width"]),
            )
            positions, position_pass = positionwise_rows(
                test, latent, scenario=scenario, kappa=float(kappa)
            )
            position_rows.extend(positions)
            if not position_pass:
                status["gate_a_positionwise_stationarity"] = "FAIL"
            plan = SamplingPlan.from_batch(test)
            iid = EmpiricalConditionalIID()
            iid.fit(train, seed=0)
            base_config = replace(
                BenchmarkConfig.from_mapping(raw, scenario, float(kappa)),
                bin_edges=edges,
            )
            seed_metrics = {bins: {"c1": [], "iid": [], "oracle": [], "c0": []} for bins in (2, 4, 8)}
            for evaluation_seed in range(1, 11):
                synthetic_iid = iid.sample(plan, seed=evaluation_seed)
                iid_summary = compute_behavior_summaries_v2(
                    synthetic_iid, tau=tau, short_gap_threshold=cutoff,
                    window_width=float(raw["data"]["window_width"]),
                )
                oracle, _ = generate_fixed_binning_split(
                    replace(base_config, randomness_nonce=evaluation_seed),
                    n_entities=len(test.lengths),
                    seed=int(raw["seeds"]["base"]) + int(raw["seeds"]["test_offset"]),
                    split_id=1,
                    prefix=f"oracle-{evaluation_seed}",
                )
                if not np.array_equal(oracle.y_entity, test.y_entity) or not np.array_equal(oracle.lengths, test.lengths):
                    raise AssertionError("oracle did not preserve the exact label/length plan")
                oracle_summary = compute_behavior_summaries_v2(
                    oracle, tau=tau, short_gap_threshold=cutoff,
                    window_width=float(raw["data"]["window_width"]),
                )
                shuffled = np.random.default_rng(10_000 + evaluation_seed).permutation(test.y_entity)
                left, right = stratified_halves(test.y_entity, evaluation_seed)
                left_summary = {key: value[left] for key, value in real_summary.items()}
                right_summary = {key: value[right] for key, value in real_summary.items()}
                for bins in (2, 4, 8):
                    reference = fit_coherence_reference(
                        real_summary, test.y_entity, n_bins=bins,
                        min_bin_count=int(raw["evaluation"]["minimum_bin_count"]),
                    )
                    c1_report = coherence_report(
                        real_summary, test.y_entity, real_summary, shuffled, reference
                    )
                    iid_report = coherence_report(
                        real_summary, test.y_entity, iid_summary,
                        synthetic_iid.y_entity, reference,
                    )
                    oracle_report = coherence_report(
                        real_summary, test.y_entity, oracle_summary,
                        oracle.y_entity, reference,
                    )
                    c0_reference = fit_coherence_reference(
                        left_summary, test.y_entity[left], n_bins=bins,
                        min_bin_count=int(raw["evaluation"]["minimum_bin_count"]),
                    )
                    c0_report = coherence_report(
                        left_summary, test.y_entity[left], right_summary,
                        test.y_entity[right], c0_reference,
                    )
                    for name, report in (
                        ("c1", c1_report), ("iid", iid_report),
                        ("oracle", oracle_report), ("c0", c0_report),
                    ):
                        seed_metrics[bins][name].append(structural_gap(report, scenario))
                    robustness_rows.append(
                        {
                            "scenario": scenario, "kappa": kappa,
                            "seed": evaluation_seed, "n_bins": bins,
                            "generator": "iid",
                            "all_channels_valid": all(
                                channel["valid"] for channel in iid_report.values()
                            ),
                            "effective_bins_min": min(reference.n_bins_effective.values()),
                        }
                    )
            for bins in (2, 4, 8):
                values = seed_metrics[bins]
                c1 = np.asarray(values["c1"])
                iid_values = np.asarray(values["iid"])
                oracle_values = np.asarray(values["oracle"])
                c0_values = np.asarray(values["c0"])
                margin = max(0.005, 0.15 * float(c1.mean()))
                iid_improvement = c1 - iid_values
                oracle_excess = oracle_values - c0_values
                iid_upper = upper_confidence(iid_improvement)
                oracle_upper = upper_confidence(oracle_excess)
                row = {
                    "scenario": scenario, "kappa": kappa, "n_bins": bins,
                    "c1_gap_mean": float(c1.mean()),
                    "iid_gap_mean": float(iid_values.mean()),
                    "improvement_mean": float(iid_improvement.mean()),
                    "improvement_ci_upper": iid_upper,
                    "equivalence_margin": margin,
                    "pass_4_8": bins == 2 or iid_upper <= margin,
                }
                iid_rows[bins].append(row)
                oracle_rows.append(
                    {
                        "scenario": scenario, "kappa": kappa, "n_bins": bins,
                        "c0_gap_mean": float(c0_values.mean()),
                        "oracle_gap_mean": float(oracle_values.mean()),
                        "oracle_excess_mean": float(oracle_excess.mean()),
                        "oracle_excess_ci_upper": oracle_upper,
                        "tolerance": margin,
                        "pass_4_8": bins == 2 or oracle_upper <= margin,
                    }
                )
                c0_rows.extend(
                    {
                        "scenario": scenario, "kappa": kappa, "n_bins": bins,
                        "seed": seed + 1, "c0_gap": value,
                    }
                    for seed, value in enumerate(c0_values)
                )
                if bins in (4, 8) and iid_upper > margin:
                    status["gate_b_iid_equivalence_to_c1"] = "FAIL"
                if bins in (4, 8) and oracle_upper > margin:
                    status["gate_c_dgp_oracle_near_c0"] = "FAIL"
            # Context ladder is a CPU diagnostic, separate from learned models.
            for block_length in (1, 2, 4, 8, "full"):
                block = EmpiricalConditionalBlock(block_length)
                block.fit(train, seed=0)
                sample = block.sample(plan, seed=1)
                summary = compute_behavior_summaries_v2(
                    sample, tau=tau, short_gap_threshold=cutoff,
                    window_width=float(raw["data"]["window_width"]),
                )
                for bins in (4, 8):
                    reference = fit_coherence_reference(
                        real_summary, test.y_entity, n_bins=bins,
                        min_bin_count=int(raw["evaluation"]["minimum_bin_count"]),
                    )
                    report = coherence_report(
                        real_summary, test.y_entity, summary,
                        sample.y_entity, reference,
                    )
                    ladder_rows.append(
                        {
                            "scenario": scenario, "kappa": kappa,
                            "block_length": block_length, "n_bins": bins,
                            "structural_gap": structural_gap(report, scenario),
                        }
                    )
    write_csv(output / "positionwise_stationarity.csv", position_rows)
    for bins in (2, 4, 8):
        write_csv(output / f"iid_bootstrap_{bins}bin.csv", iid_rows[bins])
    write_csv(output / "dgp_oracle.csv", oracle_rows)
    write_csv(output / "c0_real_split.csv", c0_rows)
    write_csv(output / "bin_robustness.csv", robustness_rows)
    context_root = output / "context_ladder"
    context_root.mkdir(exist_ok=True)
    write_csv(context_root / "context_length_curve.csv", ladder_rows)
    if not all(row["all_channels_valid"] for row in robustness_rows):
        status["gate_e_bin_robustness"] = "FAIL"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, scenario in zip(axes, raw["scenarios"]):
        selected = [
            row for row in position_rows
            if row["scenario"] == scenario and row["kappa"] == 1.0
        ]
        axis.plot([row["position"] for row in selected], [row["occupancy_y0"] for row in selected], label="y=0")
        axis.plot([row["position"] for row in selected], [row["occupancy_y1"] for row in selected], label="y=1")
        axis.axhline(0.30, color="black", linestyle="--")
        axis.set_title(scenario)
        axis.set_xlabel("position")
    axes[0].set_ylabel("burst occupancy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "positionwise_stationarity.png", dpi=160)
    plt.close(figure)
    (output / "cpu_gate_completion.json").write_text(
        json.dumps(
            {
                "schema_version": "cpu_gate_completion_v2.0",
                "status": "PASS" if all(value == "PASS" for value in status.values()) else "FAIL",
                **status,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
