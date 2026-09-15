from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.behavior_summaries_v2 import (
    compute_behavior_summaries_v2,
    fit_short_gap_threshold,
)
from eval.channel_controls_v2 import channel_only_summaries
from eval.joint_association_v2 import association_statistics


def load_batch(path: Path) -> SequenceBatch:
    with np.load(path, allow_pickle=False) as values:
        return SequenceBatch(
            **{key: values[key] for key in SequenceBatch.__dataclass_fields__}
        )


def alignment_destroyed(
    batch: SequenceBatch, *, seed: int
) -> SyntheticBatch:
    """Permute intact receiver paths within each label/length stratum."""
    rng = np.random.default_rng(seed)
    categories = batch.x_cat.copy()
    for label in (0, 1):
        for length in np.unique(batch.lengths):
            indices = np.flatnonzero(
                (batch.y_entity == label) & (batch.lengths == length)
            )
            if len(indices) < 2:
                continue
            sources = rng.permutation(indices)
            if np.any(sources == indices):
                sources = np.roll(sources, 1)
            categories[indices, : int(length)] = batch.x_cat[
                sources, : int(length)
            ]
    return SyntheticBatch(
        x_num=batch.x_num.copy(),
        dt_bin=batch.dt_bin.copy(),
        x_cat=categories,
        valid_mask=batch.valid_mask.copy(),
        y_entity=batch.y_entity.copy(),
        lengths=batch.lengths.copy(),
    )


def deltas(
    summaries: dict[str, np.ndarray], labels: np.ndarray
) -> dict[str, float]:
    return {
        key: float(association_statistics(values, labels)["delta_joint"])
        for key, values in summaries.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/benchmark_v2_2")
    parser.add_argument("--output-root", default="artifacts/benchmark_v2_3/fanout_semantic_audit")
    parser.add_argument("--permutations", type=int, default=10)
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    preservation = []
    for kappa in (0.0, 1.0):
        folder = (
            Path(args.data_root)
            / "joint_semimarkov_v2b"
            / f"kappa_{kappa:.2f}"
        )
        train, test = load_batch(folder / "train.npz"), load_batch(folder / "test.npz")
        meta = json.loads((folder / "meta.json").read_text())
        tau = np.asarray(meta["tau"])
        cutoff = fit_short_gap_threshold(train, tau=tau)
        original_behavior = compute_behavior_summaries_v2(
            test, tau=tau, short_gap_threshold=cutoff, window_width=7.0
        )
        original_controls = channel_only_summaries(test, tau=tau)
        original_delta = deltas(original_behavior, test.y_entity)
        control_delta = deltas(original_controls, test.y_entity)
        for iteration in range(args.permutations):
            destroyed = alignment_destroyed(test, seed=80_000 + iteration)
            destroyed_behavior = compute_behavior_summaries_v2(
                destroyed,
                tau=tau,
                short_gap_threshold=cutoff,
                window_width=7.0,
            )
            destroyed_controls = channel_only_summaries(destroyed, tau=tau)
            destroyed_delta = deltas(destroyed_behavior, test.y_entity)
            changed_control_delta = deltas(destroyed_controls, test.y_entity)
            rows.append(
                {
                    "kappa": kappa,
                    "permutation": iteration,
                    "original_joint_alignment_delta": original_delta["joint_alignment"],
                    "destroyed_joint_alignment_delta": destroyed_delta["joint_alignment"],
                    "restored_joint_alignment_delta": original_delta["joint_alignment"],
                    "original_fanout_delta": original_delta["fanout"],
                    "destroyed_fanout_delta": destroyed_delta["fanout"],
                    "restored_fanout_delta": original_delta["fanout"],
                }
            )
            for key in original_controls:
                preservation.append(
                    {
                        "kappa": kappa,
                        "permutation": iteration,
                        "control": key,
                        "original_delta": control_delta[key],
                        "destroyed_delta": changed_control_delta[key],
                        "absolute_delta_change": abs(
                            control_delta[key] - changed_control_delta[key]
                        ),
                    }
                )
            # Exact path-multiset preservation checks by label/length.
            for label in (0, 1):
                selected = test.y_entity == label
                original_rows = np.sort(
                    test.x_cat[selected][test.valid_mask[selected], 0]
                )
                changed_rows = np.sort(
                    destroyed.x_cat[selected][destroyed.valid_mask[selected], 0]
                )
                if not np.array_equal(original_rows, changed_rows):
                    raise AssertionError("receiver category frequency changed")
    with (output / "intervention_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "channel_control_preservation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(preservation[0]))
        writer.writeheader()
        writer.writerows(preservation)
    kappa1 = [row for row in rows if row["kappa"] == 1.0]
    joint_reduction = float(
        np.mean(
            [
                abs(row["destroyed_joint_alignment_delta"])
                / abs(row["original_joint_alignment_delta"])
                for row in kappa1
            ]
        )
    )
    fanout_reduction = float(
        np.mean(
            [
                abs(row["destroyed_fanout_delta"])
                / abs(row["original_fanout_delta"])
                for row in kappa1
            ]
        )
    )
    control_max = max(row["absolute_delta_change"] for row in preservation)
    passed = (
        joint_reduction < 0.25
        and fanout_reduction < 0.75
        and control_max < 1e-12
    )
    (output / "fanout_semantic_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "fanout_semantic_audit_v2.3-candidate",
                "status": "PASS" if passed else "FAIL",
                "intervention": "within-label-and-length intact receiver-path permutation",
                "permutations": args.permutations,
                "kappa1_destroyed_over_original_abs_joint_delta": joint_reduction,
                "kappa1_destroyed_over_original_abs_fanout_delta": fanout_reduction,
                "maximum_channel_control_delta_change": control_max,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
