from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

from benchmarks.temporal_coupling_v2 import (
    BenchmarkConfig,
    generate_single_row_audit,
)
from benchmarks.validation import stratified_entity_bootstrap_auc


def exact_interval(successes: int, trials: int) -> tuple[float, float]:
    low = (
        0.0
        if successes == 0
        else float(beta.ppf(0.025, successes, trials - successes + 1))
    )
    high = (
        1.0
        if successes == trials
        else float(beta.ppf(0.975, successes + 1, trials - successes))
    )
    return low, high


def model(name: str, raw: dict):
    if name == "logistic":
        transformer = ColumnTransformer(
            [
                ("amount", StandardScaler(), [0]),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    [1, 2],
                ),
            ]
        )
        return make_pipeline(
            transformer,
            LogisticRegression(max_iter=200, class_weight="balanced"),
        )
    return HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=int(raw["auroc_calibration"]["hgb_max_iter"]),
        random_state=42,
        class_weight="balanced",
    )


def inject(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    feature: str,
    target_label: int,
    category: int,
    magnitude: float,
    seed: int,
) -> np.ndarray:
    result = features.copy()
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(labels == target_label)
    count = int(np.ceil(magnitude * len(candidates)))
    selected = rng.choice(candidates, count, replace=False)
    if feature == "amount":
        result[selected, 0] = np.max(features[:, 0]) + 5 * np.std(features[:, 0])
    elif feature == "gap_bin":
        result[selected, 1] = np.max(features[:, 1])
    elif feature == "receiver_category":
        result[selected, 2] = category
    else:
        raise ValueError(feature)
    return result


def evaluate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    classifier: str,
    raw: dict,
    bootstrap: bool,
    seed: int,
) -> list[dict]:
    folds = int(raw["auroc_calibration"]["crossfit_folds"])
    indices = np.arange(len(train_y))
    fold_ids = np.random.default_rng(seed).permutation(indices) % folds
    validation_scores, test_scores = [], []
    for fold in range(folds):
        fitted = model(classifier, raw)
        chosen = fold_ids != fold
        fitted.fit(train_x[chosen], train_y[chosen])
        validation_scores.append(fitted.predict_proba(validation_x)[:, 1])
        test_scores.append(fitted.predict_proba(test_x)[:, 1])
    raw_validation = np.mean(validation_scores, axis=0)
    raw_test = np.mean(test_scores, axis=0)
    sign_b = 1.0 if roc_auc_score(validation_y, raw_validation) >= 0.5 else -1.0
    oriented_b = sign_b * (raw_test - 0.5)
    oriented_parts = []
    for validation_score, test_score in zip(validation_scores, test_scores):
        sign = (
            1.0
            if roc_auc_score(validation_y, validation_score) >= 0.5
            else -1.0
        )
        oriented_parts.append(sign * (test_score - 0.5))
    oriented_c = np.mean(oriented_parts, axis=0)
    methods = {
        "A_two_sided_ci_inside": raw_test,
        "B_validation_oriented_upper": oriented_b,
        "C_repeated_crossfit_upper": oriented_c,
    }
    rows = []
    for method, scores in methods.items():
        point = float(roc_auc_score(test_y, scores))
        if bootstrap:
            _, low, high = stratified_entity_bootstrap_auc(
                scores,
                test_y,
                resamples=int(raw["auroc_calibration"]["bootstrap_resamples"]),
                seed=seed + sum(map(ord, method)),
            )
        else:
            low = high = None
        rows.append(
            {
                "classifier": classifier,
                "method": method,
                "test_auroc": point,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return rows


def worker(payload: tuple) -> tuple[list[dict], list[dict]]:
    raw, data_root, scenario, kappa, replicate = payload
    binning = json.loads(
        (
            Path(data_root) / scenario / "binning.json"
        ).read_text()
    )
    config = BenchmarkConfig.from_mapping(raw, scenario, kappa)
    config = BenchmarkConfig(
        **{**config.__dict__, "bin_edges": np.asarray(binning["edges"])}
    )
    base_seed = int(raw["seeds"]["base"]) + replicate
    audit = raw["auroc_calibration"]
    datasets = {}
    for name, n_entities, split_id in (
        ("train", int(audit["train_entities"]), 0),
        ("validation", int(audit["validation_entities"]), 2),
        ("test", int(audit["test_entities"]), 1),
    ):
        datasets[name] = generate_single_row_audit(
            config,
            n_entities=n_entities,
            seed=base_seed,
            split_id=split_id,
            position_seed=600_000 + 100 * replicate + split_id,
        )
    clean, leakage = [], []
    for classifier in ("logistic", "hist_gradient_boosting"):
        evaluated = evaluate(
            datasets["train"]["features"],
            datasets["train"]["labels"],
            datasets["validation"]["features"],
            datasets["validation"]["labels"],
            datasets["test"]["features"],
            datasets["test"]["labels"],
            classifier=classifier,
            raw=raw,
            bootstrap=True,
            seed=base_seed,
        )
        for row in evaluated:
            clean.append(
                {
                    "scenario": scenario,
                    "kappa": kappa,
                    "seed": base_seed,
                    **row,
                }
            )
    features = tuple(audit["leakage_features"])
    directions = tuple(audit["directions"])
    categories = tuple(audit["receiver_categories"])
    feature = features[replicate % len(features)]
    direction = directions[(replicate // len(features)) % len(directions)]
    target_label = 1 if direction == "positive_label" else 0
    category = categories[
        (replicate // (len(features) * len(directions))) % len(categories)
    ]
    for magnitude in audit["leakage_magnitudes"]:
        changed = {}
        for split_index, name in enumerate(("train", "validation", "test")):
            changed[name] = inject(
                datasets[name]["features"],
                datasets[name]["labels"],
                feature=feature,
                target_label=target_label,
                category=category,
                magnitude=float(magnitude),
                seed=800_000 + 1000 * replicate + 10 * split_index,
            )
        for classifier in ("logistic", "hist_gradient_boosting"):
            evaluated = evaluate(
                changed["train"],
                datasets["train"]["labels"],
                changed["validation"],
                datasets["validation"]["labels"],
                changed["test"],
                datasets["test"]["labels"],
                classifier=classifier,
                raw=raw,
                bootstrap=False,
                seed=base_seed,
            )
            for row in evaluated:
                leakage.append(
                    {
                        "scenario": scenario,
                        "kappa": kappa,
                        "seed": base_seed,
                        "feature": feature,
                        "direction": direction,
                        "receiver_category": category,
                        "magnitude": float(magnitude),
                        **row,
                    }
                )
    return clean, leakage


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_v2/main_v2_3_candidate.yaml")
    parser.add_argument("--data-root", default="data/benchmark_v2_2")
    parser.add_argument("--output-root", default="artifacts/benchmark_v2_3/auroc_calibration")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    tasks = [
        (raw, args.data_root, scenario, kappa, replicate)
        for scenario in raw["scenarios"]
        for kappa in (0.0, 1.0)
        for replicate in range(args.seeds)
    ]
    started = time.perf_counter()
    clean, leakage = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for clean_rows, leakage_rows in executor.map(worker, tasks):
            clean.extend(clean_rows)
            leakage.extend(leakage_rows)
    write_csv(output / "clean_seed_results.csv", clean)
    write_csv(output / "injected_leakage_results.csv", leakage)
    low_bound, high_bound = map(
        float, raw["auroc_calibration"]["equivalence_interval"]
    )
    clean_summary = []
    for scenario in raw["scenarios"]:
        for kappa in (0.0, 1.0):
            for classifier in ("logistic", "hist_gradient_boosting"):
                for method in (
                    "A_two_sided_ci_inside",
                    "B_validation_oriented_upper",
                    "C_repeated_crossfit_upper",
                ):
                    selected = [
                        row for row in clean
                        if row["scenario"] == scenario
                        and row["kappa"] == kappa
                        and row["classifier"] == classifier
                        and row["method"] == method
                    ]
                    failures = sum(
                        not (
                            row["ci_low"] >= low_bound
                            and row["ci_high"] <= high_bound
                        )
                        if method == "A_two_sided_ci_inside"
                        else not (row["ci_high"] <= high_bound)
                        for row in selected
                    )
                    ci_low, ci_high = exact_interval(failures, len(selected))
                    clean_summary.append(
                        {
                            "scenario": scenario,
                            "kappa": kappa,
                            "classifier": classifier,
                            "method": method,
                            "seeds": len(selected),
                            "false_failures": failures,
                            "false_fail_rate": failures / len(selected),
                            "false_fail_ci_low": ci_low,
                            "false_fail_ci_high": ci_high,
                        }
                    )
    write_csv(output / "clean_cell_summary.csv", clean_summary)
    leakage_summary = []
    for classifier in ("logistic", "hist_gradient_boosting"):
        for method in (
            "A_two_sided_ci_inside",
            "B_validation_oriented_upper",
            "C_repeated_crossfit_upper",
        ):
            for magnitude in raw["auroc_calibration"]["leakage_magnitudes"]:
                selected = [
                    row for row in leakage
                    if row["classifier"] == classifier
                    and row["method"] == method
                    and row["magnitude"] == float(magnitude)
                ]
                detections = sum(
                    (
                        row["test_auroc"] < low_bound
                        or row["test_auroc"] > high_bound
                    )
                    if method == "A_two_sided_ci_inside"
                    else row["test_auroc"] > high_bound
                    for row in selected
                )
                ci_low, ci_high = exact_interval(detections, len(selected))
                leakage_summary.append(
                    {
                        "classifier": classifier,
                        "method": method,
                        "magnitude": float(magnitude),
                        "trials": len(selected),
                        "detections": detections,
                        "detection_power": detections / len(selected),
                        "detection_power_ci_low": ci_low,
                        "detection_power_ci_high": ci_high,
                    }
                )
    write_csv(output / "leakage_power_summary.csv", leakage_summary)
    # Familywise clean failure: any scenario/kappa/classifier failure for a seed.
    method_decisions = {}
    for method in (
        "A_two_sided_ci_inside",
        "B_validation_oriented_upper",
        "C_repeated_crossfit_upper",
    ):
        failed_seeds = set()
        for row in clean:
            if row["method"] != method:
                continue
            failed = (
                not (row["ci_low"] >= low_bound and row["ci_high"] <= high_bound)
                if method == "A_two_sided_ci_inside"
                else not (row["ci_high"] <= high_bound)
            )
            if failed:
                failed_seeds.add(row["seed"])
        ff_low, ff_high = exact_interval(len(failed_seeds), args.seeds)
        power_rows = [
            row for row in leakage_summary
            if row["method"] == method
        ]
        power_pass = all(
            row["detection_power_ci_low"]
            >= float(
                raw["auroc_calibration"]["required_power"][
                    str(row["magnitude"])
                ]
            )
            for row in power_rows
            if row["magnitude"] in (0.02, 0.05)
        )
        method_decisions[method] = {
            "familywise_false_failures": len(failed_seeds),
            "familywise_false_fail_rate": len(failed_seeds) / args.seeds,
            "familywise_false_fail_ci_low": ff_low,
            "familywise_false_fail_ci_high": ff_high,
            "clean_pass": ff_high
            <= float(
                raw["auroc_calibration"][
                    "maximum_familywise_false_fail_rate"
                ]
            ),
            "power_pass": power_pass,
        }
    eligible = [
        method
        for method, decision in method_decisions.items()
        if decision["clean_pass"] and decision["power_pass"]
    ]
    # Strictness order is A, then B, then C.
    selected_method = eligible[0] if eligible else None
    (output / "calibration_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "auroc_gate_calibration_v2.3-candidate",
                "status": "PASS" if selected_method else "FAIL",
                "selected_method": selected_method,
                "methods": method_decisions,
                "clean_seeds_per_cell": args.seeds,
                "bootstrap_resamples": int(
                    raw["auroc_calibration"]["bootstrap_resamples"]
                ),
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
