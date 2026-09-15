from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from scripts.calibrate_auroc_gate_v2_3 import exact_interval


METHODS = (
    "A_two_sided_ci_inside",
    "B_validation_oriented_upper",
    "C_repeated_crossfit_upper",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_v2/main_v2_3_candidate.yaml")
    parser.add_argument("--root", default="artifacts/benchmark_v2_3/auroc_calibration")
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.root)
    leakage = list(csv.DictReader((root / "injected_leakage_results.csv").open()))
    clean = list(csv.DictReader((root / "clean_seed_results.csv").open()))
    low_bound, high_bound = map(
        float, raw["auroc_calibration"]["equivalence_interval"]
    )
    subgroup = []
    grouping = ("classifier", "method", "feature", "magnitude")
    keys = sorted({tuple(row[key] for key in grouping) for row in leakage})
    for key in keys:
        selected = [
            row for row in leakage
            if tuple(row[name] for name in grouping) == key
        ]
        method = key[1]
        detections = sum(
            (
                float(row["test_auroc"]) < low_bound
                or float(row["test_auroc"]) > high_bound
            )
            if method == METHODS[0]
            else float(row["test_auroc"]) > high_bound
            for row in selected
        )
        ci_low, ci_high = exact_interval(detections, len(selected))
        subgroup.append(
            {
                **dict(zip(grouping, key)),
                "trials": len(selected),
                "detections": detections,
                "detection_power": detections / len(selected),
                "detection_power_ci_low": ci_low,
                "detection_power_ci_high": ci_high,
            }
        )
    write_csv(root / "leakage_power_by_feature.csv", subgroup)
    receiver_groups = []
    keys = sorted(
        {
            (
                row["classifier"], row["method"], row["direction"],
                row["receiver_category"], row["magnitude"],
            )
            for row in leakage
            if row["feature"] == "receiver_category"
        }
    )
    for classifier, method, direction, category, magnitude in keys:
        selected = [
            row for row in leakage
            if row["classifier"] == classifier
            and row["method"] == method
            and row["direction"] == direction
            and row["receiver_category"] == category
            and row["magnitude"] == magnitude
        ]
        detections = sum(
            (
                float(row["test_auroc"]) < low_bound
                or float(row["test_auroc"]) > high_bound
            )
            if method == METHODS[0]
            else float(row["test_auroc"]) > high_bound
            for row in selected
        )
        ci_low, ci_high = exact_interval(detections, len(selected))
        receiver_groups.append(
            {
                "classifier": classifier,
                "method": method,
                "direction": direction,
                "receiver_category": category,
                "magnitude": magnitude,
                "trials": len(selected),
                "detections": detections,
                "detection_power": detections / len(selected),
                "detection_power_ci_low": ci_low,
                "detection_power_ci_high": ci_high,
            }
        )
    write_csv(root / "receiver_category_direction_power.csv", receiver_groups)
    method_decisions = {}
    seed_count = int(raw["auroc_calibration"]["clean_seeds_per_cell"])
    for method in METHODS:
        failed_seeds = set()
        for row in clean:
            if row["method"] != method:
                continue
            low, high = float(row["ci_low"]), float(row["ci_high"])
            failed = (
                not (low >= low_bound and high <= high_bound)
                if method == METHODS[0] else high > high_bound
            )
            if failed:
                failed_seeds.add(int(row["seed"]))
        ff_low, ff_high = exact_interval(len(failed_seeds), seed_count)
        relevant = [
            row for row in subgroup
            if row["method"] == method
            and float(row["magnitude"]) in (0.02, 0.05)
        ]
        power_pass = all(
            float(row["detection_power_ci_low"])
            >= float(
                raw["auroc_calibration"]["required_power"][
                    str(float(row["magnitude"]))
                ]
            )
            for row in relevant
        )
        method_decisions[method] = {
            "familywise_false_failures": len(failed_seeds),
            "familywise_false_fail_rate": len(failed_seeds) / seed_count,
            "familywise_false_fail_ci_low": ff_low,
            "familywise_false_fail_ci_high": ff_high,
            "clean_pass": ff_high
            <= float(
                raw["auroc_calibration"][
                    "maximum_familywise_false_fail_rate"
                ]
            ),
            "all_classifier_feature_power_pass": power_pass,
        }
    eligible = [
        method for method in METHODS
        if method_decisions[method]["clean_pass"]
        and method_decisions[method]["all_classifier_feature_power_pass"]
    ]
    manifest = json.loads((root / "calibration_manifest.json").read_text())
    manifest.update(
        {
            "status": "PASS" if eligible else "FAIL",
            "selected_method": eligible[0] if eligible else None,
            "strictness_order": list(METHODS),
            "methods": method_decisions,
            "selection_requires_every_classifier_feature_subgroup": True,
            "test_label_used_for_orientation": False,
        }
    )
    (root / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
