from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta, chi2, norm, t
import yaml


METHOD_A = "A_two_sided_ci_inside"
KEYS = ["scenario", "kappa", "seed", "classifier"]
CELL_KEYS = ["scenario", "kappa"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_binomial_interval(
    successes: int,
    trials: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    low = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    )
    high = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )
    return low, high


def order_statistic_rank(sample_size: int, quantile: float) -> int:
    """Return a one-indexed, non-interpolated rank.

    This helper and its historical rank-49/rank-96 regression tests are kept
    for provenance. The superseding v2.4 design uses the 200/200 maximum.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between zero and one")
    return min(sample_size, math.ceil((sample_size + 1) * quantile))


def order_statistic(values: Iterable[float], quantile: float) -> tuple[float, int]:
    ordered = np.sort(np.asarray(list(values), dtype=float))
    if not len(ordered):
        raise ValueError("values must be non-empty")
    rank = order_statistic_rank(len(ordered), quantile)
    return float(ordered[rank - 1]), rank


def maximum_order_statistic(values: Iterable[float]) -> tuple[float, int]:
    ordered = np.sort(np.asarray(list(values), dtype=float))
    if not len(ordered):
        raise ValueError("values must be non-empty")
    return float(ordered[-1]), len(ordered)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    with path.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def clean_pivots(
    clean: pd.DataFrame,
    cells: list[tuple[str, float]],
) -> dict[str, pd.DataFrame]:
    columns = pd.MultiIndex.from_tuples(cells, names=CELL_KEYS)
    result = {}
    for classifier, selected in clean.groupby("classifier", sort=True):
        pivot = selected.pivot(
            index="seed",
            columns=CELL_KEYS,
            values="test_auroc",
        ).reindex(columns=columns)
        if pivot.isna().any().any():
            raise ValueError(f"incomplete clean cell family for {classifier}")
        result[str(classifier)] = pivot.sort_index()
    return result


def integrated_deviations(
    pivots: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, list[str], list[int]]:
    classifiers = sorted(pivots)
    seed_lists = [list(map(int, pivots[name].index)) for name in classifiers]
    if any(seeds != seed_lists[0] for seeds in seed_lists[1:]):
        raise ValueError("classifier clean seed lists differ")
    matrix = np.concatenate(
        [pivots[name].to_numpy(dtype=float) - 0.5 for name in classifiers],
        axis=1,
    )
    return matrix, classifiers, seed_lists[0]


def gaussian_upper_prediction(
    deviations: np.ndarray,
    *,
    target_calibration_seeds: int,
    mean_confidence: float,
    sd_confidence: float,
) -> tuple[float, list[dict], float]:
    """Conservative preregistered proxy for a future 200-seed maximum."""
    if deviations.ndim != 2:
        raise ValueError("deviations must be a seed by dimension matrix")
    n, dimensions = deviations.shape
    if n < 3:
        raise ValueError("at least three source seeds are required")
    if not 0 < mean_confidence < 1 or not 0 < sd_confidence < 1:
        raise ValueError("confidence values must lie strictly between 0 and 1")
    tail_denominator = 2 * dimensions * (target_calibration_seeds + 1)
    z_value = float(norm.ppf(1 - 1 / tail_denominator))
    t_value = float(t.ppf((1 + mean_confidence) / 2, n - 1))
    chi_square_value = float(chi2.ppf(1 - sd_confidence, n - 1))
    rows = []
    for dimension in range(dimensions):
        values = deviations[:, dimension]
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        mean_upper = abs(mean) + t_value * sd / math.sqrt(n)
        sd_upper = sd * math.sqrt((n - 1) / chi_square_value)
        bound = mean_upper + z_value * sd_upper
        rows.append(
            {
                "dimension_index": dimension,
                "source_seeds": n,
                "mean_signed_deviation": mean,
                "sample_sd": sd,
                "absolute_mean_upper": mean_upper,
                "sd_one_sided_upper": sd_upper,
                "gaussian_bonferroni_z": z_value,
                "upper_prediction_bound": bound,
            }
        )
    return max(row["upper_prediction_bound"] for row in rows), rows, z_value


def effect_vector(
    joined: pd.DataFrame,
    *,
    cells: list[tuple[str, float]],
    feature: str,
    classifier: str,
    magnitude: float,
) -> np.ndarray:
    selected = joined[
        (joined["feature"] == feature)
        & (joined["classifier"] == classifier)
        & np.isclose(joined["magnitude"], magnitude)
    ]
    effects = (
        selected.groupby(CELL_KEYS, sort=False)["observed_auc_shift"]
        .mean()
        .reindex(pd.MultiIndex.from_tuples(cells, names=CELL_KEYS))
    )
    if effects.isna().any():
        raise ValueError(
            f"incomplete effect cells for {feature}/{classifier}/{magnitude}"
        )
    return effects.to_numpy(dtype=float)


def modeled_power(
    *,
    clean_noise: np.ndarray,
    effect: np.ndarray,
    current_threshold: float,
    multiplier: int,
) -> float:
    """Power for the integrated maximum under optimistic 1/sqrt(N) scaling."""
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    if clean_noise.ndim != 2 or effect.shape != (clean_noise.shape[1],):
        raise ValueError("effect must contain one value per integrated dimension")
    scale = math.sqrt(multiplier)
    simulated = clean_noise / scale + effect[None, :]
    statistic = np.max(np.abs(simulated), axis=1)
    return float(np.mean(statistic > current_threshold / scale))


def git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reanalyse immutable v2.3 results using the superseding global "
            "eight-dimension AUROC maximum."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark_v2/main_v2_4_candidate.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/benchmark_v2_4/step_a",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()
    output = (repo_root / args.output_root).resolve()
    raw = yaml.safe_load(config_path.read_text())
    clean_path = (repo_root / raw["source"]["clean_results"]).resolve()
    leakage_path = (repo_root / raw["source"]["leakage_results"]).resolve()
    expected_outputs = (
        "clean_global_t_by_seed.csv",
        "threshold_proxy_dimensions.csv",
        "threshold_proxy.json",
        "observed_shift_summary.csv",
        "power_multiplier_search.csv",
        "required_multiplier_summary.csv",
        "step_a_manifest.json",
    )
    output.mkdir(parents=True, exist_ok=True)
    collisions = [name for name in expected_outputs if (output / name).exists()]
    if collisions:
        raise FileExistsError(
            "Step A outputs already exist and will not be overwritten: "
            + ", ".join(collisions)
        )

    clean_all = pd.read_csv(clean_path)
    leakage_all = pd.read_csv(leakage_path)
    clean = clean_all[clean_all["method"] == METHOD_A].copy()
    leakage = leakage_all[
        (leakage_all["method"] == METHOD_A)
        & leakage_all["magnitude"].isin(
            [float(value) for value in raw["step_b"]["leakage_magnitudes"]]
        )
    ].copy()
    if clean.empty or leakage.empty:
        raise ValueError("method-A source rows are incomplete")

    cells = sorted(
        {
            (str(row.scenario), float(row.kappa))
            for row in clean.itertuples(index=False)
        }
    )
    pivots = clean_pivots(clean, cells)
    clean_noise, classifiers, source_seeds = integrated_deviations(pivots)
    expected_dimensions = int(
        raw["step_a"]["power"]["familywise_cells_and_classifiers"]
    )
    if clean_noise.shape != (len(source_seeds), expected_dimensions):
        raise ValueError(
            f"expected integrated shape (*,{expected_dimensions}), "
            f"found {clean_noise.shape}"
        )
    expected_source_seeds = int(raw["step_a"]["immutable_source_clean_seeds"])
    if len(source_seeds) != expected_source_seeds:
        raise ValueError(
            f"expected {expected_source_seeds} source seeds, "
            f"found {len(source_seeds)}"
        )

    global_t = np.max(np.abs(clean_noise), axis=1)
    observed_maximum, observed_rank = maximum_order_statistic(global_t)
    proxy_config = raw["step_a"]["threshold_proxy"]
    target_calibration_seeds = int(proxy_config["target_calibration_seeds"])
    gaussian_bound, dimension_rows, z_value = gaussian_upper_prediction(
        clean_noise,
        target_calibration_seeds=target_calibration_seeds,
        mean_confidence=float(
            proxy_config["gaussian_upper_prediction"][
                "mean_uncertainty_confidence"
            ]
        ),
        sd_confidence=float(
            proxy_config["gaussian_upper_prediction"]["sd_upper_confidence"]
        ),
    )
    dimension_labels = [
        f"{classifier}|{scenario}|{kappa:g}"
        for classifier in classifiers
        for scenario, kappa in cells
    ]
    for row, label in zip(dimension_rows, dimension_labels):
        row["dimension"] = label
    threshold_proxy = max(observed_maximum, gaussian_bound)
    threshold_source = (
        "observed_100_seed_global_T_max"
        if observed_maximum >= gaussian_bound
        else "gaussian_bonferroni_upper_prediction"
    )
    clean_t_rows = [
        {
            "seed": int(seed),
            "T_global_8": float(value),
            "dimensions": expected_dimensions,
        }
        for seed, value in zip(source_seeds, global_t)
    ]
    threshold_payload = {
        "status": "STEP_A_PROXY_NOT_A_CALIBRATED_THRESHOLD",
        "source_clean_seeds": len(source_seeds),
        "target_step_b_calibration_seeds": target_calibration_seeds,
        "target_step_b_order_statistic_rank": int(
            proxy_config["target_order_statistic_rank"]
        ),
        "observed_100_seed_global_T_max": observed_maximum,
        "observed_order_statistic_rank": observed_rank,
        "gaussian_bonferroni_upper_prediction": gaussian_bound,
        "gaussian_bonferroni_z": z_value,
        "selected_current_N_threshold_proxy": threshold_proxy,
        "selected_component": threshold_source,
        "interpolation": "none",
    }

    paired_clean = clean[KEYS + ["test_auroc"]].rename(
        columns={"test_auroc": "clean_test_auroc"}
    )
    joined = leakage.merge(
        paired_clean,
        on=KEYS,
        validate="many_to_one",
    )
    if len(joined) != len(leakage):
        raise ValueError("failed to pair every leakage row with a clean row")
    joined["observed_auc_shift"] = (
        joined["test_auroc"] - joined["clean_test_auroc"]
    )

    shift_rows: list[dict] = []
    for keys, selected in joined.groupby(
        ["feature", "classifier", "magnitude"],
        sort=True,
    ):
        feature, classifier, magnitude = keys
        values = selected["observed_auc_shift"].to_numpy(dtype=float)
        shift_rows.append(
            {
                "feature": str(feature),
                "classifier": str(classifier),
                "magnitude": float(magnitude),
                "rows": len(values),
                "injected_seeds": selected["seed"].nunique(),
                "mean_observed_auc_shift": float(np.mean(values)),
                "median_observed_auc_shift": float(np.median(values)),
                "sd_observed_auc_shift": float(np.std(values, ddof=1)),
                "mean_absolute_observed_auc_shift": float(
                    np.mean(np.abs(values))
                ),
                "minimum_observed_auc_shift": float(np.min(values)),
                "maximum_observed_auc_shift": float(np.max(values)),
                "interpretation": (
                    "v2.3 fixed-operator proxy; population shift held fixed"
                ),
            }
        )

    maximum_multiplier = int(
        raw["step_a"]["power"]["maximum_search_multiplier"]
    )
    targets = {
        float(key): float(value)
        for key, value in raw["step_a"]["power"]["target_power"].items()
    }
    search_rows: list[dict] = []
    required_rows: list[dict] = []
    features = sorted(map(str, joined["feature"].unique()))
    cells_per_classifier = len(cells)
    for feature in features:
        for classifier_index, classifier in enumerate(classifiers):
            for magnitude in sorted(targets):
                target_effect = effect_vector(
                    joined,
                    cells=cells,
                    feature=feature,
                    classifier=classifier,
                    magnitude=magnitude,
                )
                integrated_effect = np.zeros(expected_dimensions, dtype=float)
                start = classifier_index * cells_per_classifier
                stop = start + cells_per_classifier
                integrated_effect[start:stop] = target_effect
                target = targets[magnitude]
                required: int | None = None
                power_at_required: float | None = None
                last_power = 0.0
                for multiplier in range(1, maximum_multiplier + 1):
                    power = modeled_power(
                        clean_noise=clean_noise,
                        effect=integrated_effect,
                        current_threshold=threshold_proxy,
                        multiplier=multiplier,
                    )
                    last_power = power
                    search_rows.append(
                        {
                            "feature": feature,
                            "classifier": classifier,
                            "magnitude": magnitude,
                            "multiplier": multiplier,
                            "modeled_power": power,
                            "target_power": target,
                            "modeled_threshold_proxy": (
                                threshold_proxy / math.sqrt(multiplier)
                            ),
                            "detection_event": "global_T_strictly_gt_proxy",
                            "other_classifier_cells": "clean",
                            "assumption": (
                                "optimistic_inverse_sqrt_all_split_noise"
                            ),
                        }
                    )
                    if power >= target:
                        required = multiplier
                        power_at_required = power
                        break
                required_rows.append(
                    {
                        "feature": feature,
                        "classifier": classifier,
                        "magnitude": magnitude,
                        "target_power": target,
                        "minimum_multiplier": (
                            required
                            if required is not None
                            else f">{maximum_multiplier}"
                        ),
                        "modeled_power_at_minimum": (
                            power_at_required
                            if power_at_required is not None
                            else last_power
                        ),
                        "current_N_global_threshold_proxy": threshold_proxy,
                        "target_classifier_effect_by_cell": json.dumps(
                            {
                                f"{scenario}|{kappa:g}": float(value)
                                for (scenario, kappa), value in zip(
                                    cells, target_effect
                                )
                            },
                            sort_keys=True,
                        ),
                        "integrated_dimensions": expected_dimensions,
                        "interpretation": "optimistic_lower_bound",
                    }
                )

    unresolved = [
        row
        for row in required_rows
        if not isinstance(row["minimum_multiplier"], int)
    ]
    if unresolved:
        largest_required: int | None = None
        selected_multiplier: int | None = None
        decision_status = "REPEATED_CROSSFIT_ANALYSIS_REQUIRED"
    else:
        largest_required = max(
            int(row["minimum_multiplier"]) for row in required_rows
        )
        selected_multiplier = math.ceil(
            largest_required
            * float(raw["step_a"]["power"]["safety_factor"])
        )
        decision_status = (
            "REPEATED_CROSSFIT_ANALYSIS_REQUIRED"
            if largest_required
            > int(
                raw["step_a"]["power"][
                    "unrealistic_multiplier_exclusive_upper_bound"
                ]
            )
            else "GLOBAL_MAX_STATISTIC_SELECTED"
        )

    base = raw["audit_base"]
    audit_counts = (
        None
        if selected_multiplier is None
        else {
            "train_entities": int(base["train_entities"]) * selected_multiplier,
            "orientation_validation_entities": (
                int(base["orientation_validation_entities"])
                * selected_multiplier
            ),
            "test_entities": int(base["test_entities"]) * selected_multiplier,
        }
    )
    calibration_start = int(raw["step_b"]["calibration_clean_seeds"]["start"])
    calibration_count = int(raw["step_b"]["calibration_clean_seeds"]["count"])
    validation_start = int(raw["step_b"]["validation_clean_seeds"]["start"])
    validation_count = int(raw["step_b"]["validation_clean_seeds"]["count"])
    elapsed = time.perf_counter() - started
    manifest = {
        "schema_version": "auroc-gate-global-max-step-a-v2.4",
        "status": decision_status,
        "source_commit_sha": git_head(repo_root),
        "source_method": METHOD_A,
        "config": str(config_path.relative_to(repo_root)),
        "config_sha256": sha256_file(config_path),
        "code": str(Path(__file__).resolve().relative_to(repo_root)),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": {
            str(clean_path.relative_to(repo_root)): sha256_file(clean_path),
            str(leakage_path.relative_to(repo_root)): sha256_file(leakage_path),
        },
        "source_seed_list": source_seeds,
        "step_b_calibration_seed_list": list(
            range(calibration_start, calibration_start + calibration_count)
        ),
        "step_b_validation_seed_list": list(
            range(validation_start, validation_start + validation_count)
        ),
        "cells": [
            {"scenario": scenario, "kappa": kappa}
            for scenario, kappa in cells
        ],
        "classifiers": classifiers,
        "integrated_dimensions": dimension_labels,
        "statistic": "max_abs_deviation_over_4_cells_times_2_classifiers",
        "threshold_proxy": threshold_payload,
        "step_b_threshold": {
            "definition": "maximum_of_200_calibration_global_T_values",
            "order_statistic_rank": 200,
            "interpolation": "none",
            "value": "pending_step_b",
        },
        "power_model": {
            "detection_event": "global_T_strictly_greater_than_threshold",
            "classifier_table_other_classifier_cells": "clean",
            "noise_scaling": "1/sqrt(common audit split multiplier)",
            "effect_scaling": "observed cell-mean shift held fixed",
            "interpretation": (
                "optimistic lower bound; fitting/orientation noise may not "
                "scale exactly at finite N"
            ),
            "clean_noise_patterns": len(source_seeds),
            "maximum_search_multiplier": maximum_multiplier,
        },
        "largest_required_multiplier_before_safety": largest_required,
        "safety_factor": float(raw["step_a"]["power"]["safety_factor"]),
        "selected_audit_multiplier": selected_multiplier,
        "selected_audit_entity_counts": audit_counts,
        "variance_reduction_evaluated": False,
        "descriptive_2pct_features": [],
        "direct_row_marginal_hard_guards": raw[
            "direct_row_marginal_hard_guards"
        ],
        "elapsed_seconds": elapsed,
        "artifacts": list(expected_outputs),
        "superseded_artifact": (
            "artifacts/benchmark_v2_4/pre_amendment_step_a_rank49"
        ),
    }

    write_csv(output / "clean_global_t_by_seed.csv", clean_t_rows)
    write_csv(output / "threshold_proxy_dimensions.csv", dimension_rows)
    write_json(output / "threshold_proxy.json", threshold_payload)
    write_csv(output / "observed_shift_summary.csv", shift_rows)
    write_csv(output / "power_multiplier_search.csv", search_rows)
    write_csv(output / "required_multiplier_summary.csv", required_rows)
    write_json(output / "step_a_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
