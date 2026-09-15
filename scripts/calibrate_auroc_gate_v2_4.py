from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
import platform
import resource
import subprocess
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
import yaml

from benchmarks.auroc_audit_v2_4 import (
    generate_stationary_row_audit,
    inject_leakage,
    leakage_operator_for_trial,
)
from benchmarks.temporal_coupling_v2 import BenchmarkConfig


CLASSIFIERS = ("hist_gradient_boosting", "logistic")
FEATURES = ("amount", "gap", "receiver_category")
SPLITS = (
    ("train", 0),
    ("orientation_validation", 2),
    ("test", 1),
)


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


def classifier_model(
    name: str,
    *,
    hgb_max_iter: int,
    random_state: int,
):
    if name == "logistic":
        transformer = ColumnTransformer(
            [
                ("continuous", StandardScaler(), [0, 1]),
                (
                    "receiver",
                    OneHotEncoder(handle_unknown="ignore"),
                    [2],
                ),
            ]
        )
        return make_pipeline(
            transformer,
            LogisticRegression(
                max_iter=100,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            ),
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=hgb_max_iter,
            early_stopping=False,
            categorical_features=[False, False, True],
            random_state=random_state,
            class_weight="balanced",
        )
    raise ValueError(f"unknown classifier {name}")


def evaluate_classifier(
    datasets: dict[str, dict],
    *,
    classifier: str,
    hgb_max_iter: int,
    random_state: int,
) -> dict:
    started = time.perf_counter()
    fitted = classifier_model(
        classifier,
        hgb_max_iter=hgb_max_iter,
        random_state=random_state,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fitted.fit(
            datasets["train"]["features"],
            datasets["train"]["labels"],
        )
    validation_scores = fitted.predict_proba(
        datasets["orientation_validation"]["features"]
    )[:, 1]
    validation_auc = float(
        roc_auc_score(
            datasets["orientation_validation"]["labels"],
            validation_scores,
        )
    )
    orientation_sign = 1 if validation_auc >= 0.5 else -1
    test_scores = fitted.predict_proba(datasets["test"]["features"])[:, 1]
    oriented_test_scores = orientation_sign * (test_scores - 0.5)
    test_auc = float(
        roc_auc_score(datasets["test"]["labels"], oriented_test_scores)
    )
    convergence_warnings = sum(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    return {
        "classifier": classifier,
        "orientation_validation_auroc": validation_auc,
        "orientation_sign": orientation_sign,
        "test_auroc": test_auc,
        "test_labels_used_for_orientation": False,
        "fit_predict_seconds": time.perf_counter() - started,
        "convergence_warnings": convergence_warnings,
    }


def task_key(payload: dict) -> str:
    return (
        f"{payload['partition']}__m{payload['multiplier']}__"
        f"s{payload['audit_seed']}__{payload['scenario']}__"
        f"k{float(payload['kappa']):g}"
    )


def worker(payload: dict) -> dict:
    started = time.perf_counter()
    thread_limit = int(payload["threads_per_worker"])
    context = (
        threadpool_limits(limits=thread_limit)
        if thread_limit > 0
        else nullcontext()
    )
    with context:
        raw = payload["benchmark_config"]
        config = BenchmarkConfig.from_mapping(
            raw,
            payload["scenario"],
            float(payload["kappa"]),
        )
        multiplier = int(payload["multiplier"])
        base_counts = payload["base_counts"]
        counts = {
            "train": int(base_counts["train_entities"]) * multiplier,
            "orientation_validation": (
                int(base_counts["orientation_validation_entities"]) * multiplier
            ),
            "test": int(base_counts["test_entities"]) * multiplier,
        }
        datasets = {}
        dataset_provenance = {}
        for split_name, split_id in SPLITS:
            generated = generate_stationary_row_audit(
                config,
                n_entities=counts[split_name],
                audit_seed=int(payload["audit_seed"]),
                split_id=split_id,
            )
            datasets[split_name] = {
                "features": generated["features"],
                "labels": generated["labels"],
            }
            dataset_provenance[split_name] = generated["metadata"]

        clean_rows = []
        base_random_state = (
            int(payload["audit_seed"]) * 100
            + int(round(10 * float(payload["kappa"])))
            + (1 if payload["scenario"].endswith("v2a") else 2)
        )
        for classifier_index, classifier in enumerate(CLASSIFIERS):
            evaluated = evaluate_classifier(
                datasets,
                classifier=classifier,
                hgb_max_iter=int(payload["hgb_max_iter"]),
                random_state=base_random_state + classifier_index,
            )
            clean_rows.append(
                {
                    "partition": payload["partition"],
                    "audit_seed": int(payload["audit_seed"]),
                    "validation_trial_ordinal": payload[
                        "validation_trial_ordinal"
                    ],
                    "scenario": payload["scenario"],
                    "kappa": float(payload["kappa"]),
                    "multiplier": multiplier,
                    **evaluated,
                }
            )

        leakage_rows = []
        if payload["include_leakage"]:
            ordinal = int(payload["validation_trial_ordinal"])
            for feature_index, feature in enumerate(FEATURES):
                for magnitude_index, magnitude in enumerate(
                    payload["leakage_magnitudes"]
                ):
                    operator = leakage_operator_for_trial(
                        feature=feature,
                        magnitude=float(magnitude),
                        validation_trial_ordinal=ordinal,
                    )
                    changed = {}
                    injection_metadata = {}
                    for split_name, split_id in SPLITS:
                        changed_features, metadata = inject_leakage(
                            datasets[split_name]["features"],
                            datasets[split_name]["labels"],
                            operator=operator,
                            config=config,
                            audit_seed=int(payload["audit_seed"]),
                            split_id=split_id,
                        )
                        changed[split_name] = {
                            "features": changed_features,
                            "labels": datasets[split_name]["labels"],
                        }
                        injection_metadata[split_name] = metadata
                    for classifier_index, classifier in enumerate(CLASSIFIERS):
                        evaluated = evaluate_classifier(
                            changed,
                            classifier=classifier,
                            hgb_max_iter=int(payload["hgb_max_iter"]),
                            random_state=(
                                base_random_state
                                + 10_000
                                + 1000 * feature_index
                                + 100 * magnitude_index
                                + classifier_index
                            ),
                        )
                        leakage_rows.append(
                            {
                                "partition": payload["partition"],
                                "audit_seed": int(payload["audit_seed"]),
                                "validation_trial_ordinal": ordinal,
                                "scenario": payload["scenario"],
                                "kappa": float(payload["kappa"]),
                                "multiplier": multiplier,
                                "feature": feature,
                                "magnitude": float(magnitude),
                                "target_label": operator.target_label,
                                "receiver_category": operator.receiver_category,
                                "operator_index": operator.operator_index,
                                **evaluated,
                                "train_changed_rows": injection_metadata[
                                    "train"
                                ]["changed_rows"],
                                "train_target_class_rows": injection_metadata[
                                    "train"
                                ]["target_class_rows"],
                                "train_realised_probability_change": (
                                    injection_metadata["train"][
                                        "realised_probability_change"
                                    ]
                                ),
                                "orientation_validation_changed_rows": (
                                    injection_metadata[
                                        "orientation_validation"
                                    ]["changed_rows"]
                                ),
                                "orientation_validation_target_class_rows": (
                                    injection_metadata[
                                        "orientation_validation"
                                    ]["target_class_rows"]
                                ),
                                "orientation_validation_realised_probability_change": (
                                    injection_metadata[
                                        "orientation_validation"
                                    ]["realised_probability_change"]
                                ),
                                "test_changed_rows": injection_metadata["test"][
                                    "changed_rows"
                                ],
                                "test_target_class_rows": injection_metadata[
                                    "test"
                                ]["target_class_rows"],
                                "test_realised_probability_change": (
                                    injection_metadata["test"][
                                        "realised_probability_change"
                                    ]
                                ),
                                "replacement_anchor": injection_metadata[
                                    "test"
                                ]["replacement_anchor"],
                            }
                        )
                    del changed

    return {
        "schema_version": "auroc-audit-cell-seed-checkpoint-v2.4",
        "task_key": task_key(payload),
        "partition": payload["partition"],
        "audit_seed": int(payload["audit_seed"]),
        "validation_trial_ordinal": payload["validation_trial_ordinal"],
        "scenario": payload["scenario"],
        "kappa": float(payload["kappa"]),
        "multiplier": int(payload["multiplier"]),
        "entity_counts": counts,
        "dataset_provenance": dataset_provenance,
        "clean_rows": clean_rows,
        "leakage_rows": leakage_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "threads_per_worker": thread_limit,
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def append_completion(path: Path, result: dict) -> None:
    summary = {
        "task_key": result["task_key"],
        "checkpoint": f"checkpoints/{result['task_key']}.json",
        "clean_rows": len(result["clean_rows"]),
        "leakage_rows": len(result["leakage_rows"]),
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_rss_kb": result["peak_rss_kb"],
    }
    with path.open("a") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_tasks(
    *,
    raw_v2_3: dict,
    raw_v2_4: dict,
    mode: str,
    threads_per_worker: int,
) -> list[dict]:
    cells = [
        (scenario, kappa)
        for scenario in raw_v2_3["scenarios"]
        for kappa in (0.0, 1.0)
    ]
    base = raw_v2_4["audit_base"]
    base_counts = {
        "train_entities": int(base["train_entities"]),
        "orientation_validation_entities": int(
            base["orientation_validation_entities"]
        ),
        "test_entities": int(base["test_entities"]),
    }
    magnitudes = [
        float(value) for value in raw_v2_4["step_b"]["leakage_magnitudes"]
    ]
    if mode == "smoke":
        calibration_seeds = [
            int(raw_v2_4["step_b"]["calibration_clean_seeds"]["start"])
        ]
        validation_seeds = [
            int(raw_v2_4["step_b"]["validation_clean_seeds"]["start"])
        ]
    else:
        calibration = raw_v2_4["step_b"]["calibration_clean_seeds"]
        validation = raw_v2_4["step_b"]["validation_clean_seeds"]
        calibration_seeds = list(
            range(
                int(calibration["start"]),
                int(calibration["start"]) + int(calibration["count"]),
            )
        )
        validation_seeds = list(
            range(
                int(validation["start"]),
                int(validation["start"]) + int(validation["count"]),
            )
        )

    common = {
        "benchmark_config": raw_v2_3,
        "base_counts": base_counts,
        "hgb_max_iter": int(raw_v2_3["auroc_calibration"]["hgb_max_iter"]),
        "leakage_magnitudes": magnitudes,
        "threads_per_worker": threads_per_worker,
    }
    tasks = []
    selected_multiplier = int(base["audit_multiplier"])
    for seed in calibration_seeds:
        for scenario, kappa in cells:
            tasks.append(
                {
                    **common,
                    "partition": "calibration",
                    "audit_seed": seed,
                    "validation_trial_ordinal": None,
                    "scenario": scenario,
                    "kappa": kappa,
                    "multiplier": selected_multiplier,
                    "include_leakage": False,
                }
            )
    validation_start = int(
        raw_v2_4["step_b"]["validation_clean_seeds"]["start"]
    )
    for seed in validation_seeds:
        ordinal = seed - validation_start
        for scenario, kappa in cells:
            tasks.append(
                {
                    **common,
                    "partition": "validation",
                    "audit_seed": seed,
                    "validation_trial_ordinal": ordinal,
                    "scenario": scenario,
                    "kappa": kappa,
                    "multiplier": selected_multiplier,
                    "include_leakage": True,
                }
            )
    if mode == "smoke":
        seed = calibration_seeds[0]
        for scenario, kappa in cells:
            tasks.append(
                {
                    **common,
                    "partition": "smoke_base",
                    "audit_seed": seed,
                    "validation_trial_ordinal": None,
                    "scenario": scenario,
                    "kappa": kappa,
                    "multiplier": 1,
                    "include_leakage": False,
                }
            )
    return tasks


def write_csv_exclusive(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_checkpoints(output: Path) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted((output / "checkpoints").glob("*.json"))
    ]


def integrated_clean_t(clean: pd.DataFrame, partition: str) -> pd.DataFrame:
    selected = clean[clean["partition"] == partition]
    counts = selected.groupby("audit_seed").size()
    if not counts.empty and not np.all(counts.to_numpy() == 8):
        raise ValueError(f"{partition} does not have eight clean rows per seed")
    rows = []
    for seed, group in selected.groupby("audit_seed", sort=True):
        rows.append(
            {
                "partition": partition,
                "audit_seed": int(seed),
                "T_global_8": float(
                    np.max(np.abs(group["test_auroc"].to_numpy() - 0.5))
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize(
    *,
    output: Path,
    raw_v2_4: dict,
    mode: str,
    expected_tasks: int,
    workers: int,
) -> dict:
    checkpoints = collect_checkpoints(output)
    if len(checkpoints) != expected_tasks:
        raise ValueError(
            f"expected {expected_tasks} checkpoints, found {len(checkpoints)}"
        )
    clean_rows = [
        row for checkpoint in checkpoints for row in checkpoint["clean_rows"]
    ]
    leakage_rows = [
        row for checkpoint in checkpoints for row in checkpoint["leakage_rows"]
    ]
    clean = pd.DataFrame(clean_rows)
    leakage = pd.DataFrame(leakage_rows)
    clean_t = pd.concat(
        [
            integrated_clean_t(clean, "calibration"),
            integrated_clean_t(clean, "validation"),
        ],
        ignore_index=True,
    )
    calibration_t = clean_t[clean_t["partition"] == "calibration"]
    validation_t = clean_t[clean_t["partition"] == "validation"]
    threshold = float(calibration_t["T_global_8"].max())
    validation_failures = int(
        np.sum(validation_t["T_global_8"].to_numpy() > threshold)
    )
    ff_low, ff_high = exact_binomial_interval(
        validation_failures,
        len(validation_t),
    )
    clean_decision = {
        "calibration_seeds": len(calibration_t),
        "calibration_order_statistic_rank": len(calibration_t),
        "threshold": threshold,
        "interpolation": "none",
        "validation_seeds": len(validation_t),
        "false_failures": validation_failures,
        "false_fail_rate": validation_failures / len(validation_t),
        "false_fail_ci_low": ff_low,
        "false_fail_ci_high": ff_high,
        "maximum_allowed_exact_upper": float(
            raw_v2_4["step_b"][
                "maximum_validation_false_fail_upper_bound"
            ]
        ),
    }

    clean_lookup = {
        (
            int(row.audit_seed),
            str(row.scenario),
            float(row.kappa),
            str(row.classifier),
        ): float(row.test_auroc)
        for row in clean[clean["partition"] == "validation"].itertuples(
            index=False
        )
    }
    leakage_t_rows = []
    if not leakage.empty:
        group_keys = ["feature", "classifier", "magnitude", "audit_seed"]
        for keys, group in leakage.groupby(group_keys, sort=True):
            feature, classifier, magnitude, seed = keys
            if len(group) != 4:
                raise ValueError("leakage target classifier lacks four cells")
            other = next(name for name in CLASSIFIERS if name != classifier)
            integrated_values = list(map(float, group["test_auroc"]))
            for scenario in sorted(group["scenario"].unique()):
                for kappa in (0.0, 1.0):
                    integrated_values.append(
                        clean_lookup[(int(seed), scenario, kappa, other)]
                    )
            if len(integrated_values) != 8:
                raise ValueError("leakage statistic must have eight values")
            first = group.iloc[0]
            statistic = float(
                np.max(np.abs(np.asarray(integrated_values) - 0.5))
            )
            leakage_t_rows.append(
                {
                    "feature": feature,
                    "classifier": classifier,
                    "magnitude": float(magnitude),
                    "audit_seed": int(seed),
                    "validation_trial_ordinal": int(
                        first["validation_trial_ordinal"]
                    ),
                    "target_label": int(first["target_label"]),
                    "receiver_category": (
                        None
                        if pd.isna(first["receiver_category"])
                        else int(first["receiver_category"])
                    ),
                    "operator_index": int(first["operator_index"]),
                    "T_global_8": statistic,
                    "threshold": threshold,
                    "detected": bool(statistic > threshold),
                }
            )
    power_rows = []
    operator_rows = []
    if leakage_t_rows:
        leakage_t = pd.DataFrame(leakage_t_rows)
        targets = {
            float(key): float(value)
            for key, value in raw_v2_4["step_b"]["required_power"].items()
        }
        for keys, group in leakage_t.groupby(
            ["feature", "classifier", "magnitude"],
            sort=True,
        ):
            feature, classifier, magnitude = keys
            detections = int(group["detected"].sum())
            low, high = exact_binomial_interval(detections, len(group))
            power_rows.append(
                {
                    "feature": feature,
                    "classifier": classifier,
                    "magnitude": float(magnitude),
                    "trials": len(group),
                    "detections": detections,
                    "detection_power": detections / len(group),
                    "detection_power_ci_low": low,
                    "detection_power_ci_high": high,
                    "required_power": targets[float(magnitude)],
                    "pass": low >= targets[float(magnitude)],
                    "detection_event": "global_T_strictly_gt_threshold",
                }
            )
        for keys, group in leakage_t.groupby(
            [
                "feature",
                "classifier",
                "magnitude",
                "target_label",
                "receiver_category",
                "operator_index",
            ],
            dropna=False,
            sort=True,
        ):
            (
                feature,
                classifier,
                magnitude,
                target_label,
                receiver_category,
                operator_index,
            ) = keys
            detections = int(group["detected"].sum())
            low, high = exact_binomial_interval(detections, len(group))
            operator_rows.append(
                {
                    "feature": feature,
                    "classifier": classifier,
                    "magnitude": float(magnitude),
                    "target_label": int(target_label),
                    "receiver_category": (
                        None
                        if pd.isna(receiver_category)
                        else int(receiver_category)
                    ),
                    "operator_index": int(operator_index),
                    "trials": len(group),
                    "detections": detections,
                    "detection_power": detections / len(group),
                    "detection_power_ci_low": low,
                    "detection_power_ci_high": high,
                    "status": "DESCRIPTIVE_OPERATOR_BREAKDOWN",
                }
            )
    else:
        leakage_t = pd.DataFrame()

    full_mode = mode == "full"
    clean_pass = (
        full_mode
        and len(calibration_t)
        == int(raw_v2_4["step_b"]["calibration_clean_seeds"]["count"])
        and len(validation_t)
        == int(raw_v2_4["step_b"]["validation_clean_seeds"]["count"])
        and ff_high
        <= float(
            raw_v2_4["step_b"][
                "maximum_validation_false_fail_upper_bound"
            ]
        )
    )
    power_pass = (
        full_mode
        and len(power_rows) == 12
        and all(bool(row["pass"]) for row in power_rows)
    )
    status = (
        "PASS"
        if clean_pass and power_pass
        else ("FAIL" if full_mode else "SMOKE_COMPLETE_NOT_GATE")
    )

    write_csv_exclusive(output / "clean_cell_results.csv", clean_rows)
    write_csv_exclusive(
        output / "clean_global_t_by_seed.csv",
        clean_t.to_dict("records"),
    )
    if leakage_rows:
        write_csv_exclusive(output / "leakage_cell_results.csv", leakage_rows)
        write_csv_exclusive(
            output / "leakage_global_t_by_seed.csv",
            leakage_t.to_dict("records"),
        )
        write_csv_exclusive(output / "leakage_power_summary.csv", power_rows)
        write_csv_exclusive(
            output / "leakage_operator_breakdown.csv",
            operator_rows,
        )

    elapsed_values = [float(row["elapsed_seconds"]) for row in checkpoints]
    peak_values = [int(row["peak_rss_kb"]) for row in checkpoints]
    manifest = {
        "schema_version": "auroc-gate-calibration-v2.4",
        "mode": mode,
        "status": status,
        "clean": clean_decision,
        "power": power_rows,
        "checkpoint_tasks": len(checkpoints),
        "workers": workers,
        "task_elapsed_seconds": {
            "minimum": min(elapsed_values),
            "median": float(np.median(elapsed_values)),
            "maximum": max(elapsed_values),
        },
        "maximum_worker_peak_rss_kb": max(peak_values),
        "learned_smoke_authorized": status == "PASS",
    }
    with (output / "calibration_manifest.json").open("x") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def write_run_manifest(
    *,
    output: Path,
    repo_root: Path,
    config_path: Path,
    source_config_path: Path,
    raw_v2_4: dict,
    mode: str,
    workers: int,
    threads_per_worker: int,
) -> None:
    path = output / "run_manifest.json"
    payload = {
        "schema_version": "auroc-audit-run-provenance-v2.4",
        "mode": mode,
        "source_commit_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip(),
        "config": str(config_path.relative_to(repo_root)),
        "config_sha256": sha256_file(config_path),
        "source_benchmark_config": str(source_config_path.relative_to(repo_root)),
        "source_benchmark_config_sha256": sha256_file(source_config_path),
        "code_sha256": {
            "scripts/calibrate_auroc_gate_v2_4.py": sha256_file(
                Path(__file__).resolve()
            ),
            "benchmarks/auroc_audit_v2_4.py": sha256_file(
                repo_root / "benchmarks/auroc_audit_v2_4.py"
            ),
        },
        "calibration_seed_list": list(
            range(
                int(raw_v2_4["step_b"]["calibration_clean_seeds"]["start"]),
                int(raw_v2_4["step_b"]["calibration_clean_seeds"]["start"])
                + (
                    1
                    if mode == "smoke"
                    else int(
                        raw_v2_4["step_b"]["calibration_clean_seeds"]["count"]
                    )
                ),
            )
        ),
        "validation_seed_list": list(
            range(
                int(raw_v2_4["step_b"]["validation_clean_seeds"]["start"]),
                int(raw_v2_4["step_b"]["validation_clean_seeds"]["start"])
                + (
                    1
                    if mode == "smoke"
                    else int(
                        raw_v2_4["step_b"]["validation_clean_seeds"]["count"]
                    )
                ),
            )
        ),
        "audit_multiplier": int(raw_v2_4["audit_base"]["audit_multiplier"]),
        "audit_entity_counts": raw_v2_4["audit_base"][
            "selected_entity_counts"
        ],
        "generator_training_entities_unchanged": int(
            raw_v2_4["unchanged_contract"]["generator_training_entities"]
        ),
        "audit_dataset_path": str(
            output.relative_to(repo_root) / "checkpoints"
        ),
        "audit_dataset_storage": (
            "deterministic ephemeral arrays; per-cell-seed provenance checkpoint"
        ),
        "workers": workers,
        "logical_cpus": os.cpu_count(),
        "threads_per_worker": threads_per_worker,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threshold": {
            "statistic": "global_max_over_4_cells_times_2_classifiers",
            "calibration_order_statistic": "maximum",
            "rank": 200,
            "interpolation": "none",
            "detection_event": "T_strictly_greater_than_threshold",
        },
        "leakage_operator_config": raw_v2_4["leakage_operators"],
        "created_unix_time": time.time(),
    }
    if path.exists():
        existing = json.loads(path.read_text())
        comparable = dict(existing)
        comparable.pop("created_unix_time", None)
        candidate = dict(payload)
        candidate.pop("created_unix_time", None)
        if comparable != candidate:
            raise RuntimeError("resume provenance does not match existing run")
        return
    with path.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_smoke_projection(
    *,
    output: Path,
    raw_v2_4: dict,
    workers: int,
) -> dict:
    checkpoints = collect_checkpoints(output)
    selected_calibration = [
        row for row in checkpoints if row["partition"] == "calibration"
    ]
    selected_validation = [
        row for row in checkpoints if row["partition"] == "validation"
    ]
    base = [row for row in checkpoints if row["partition"] == "smoke_base"]
    clean = pd.DataFrame(
        [item for row in checkpoints for item in row["clean_rows"]]
    )
    base_t = integrated_clean_t(clean, "smoke_base")
    selected_t = integrated_clean_t(clean, "calibration")
    if len(base_t) != 1 or len(selected_t) != 1:
        raise ValueError("smoke variance diagnostic requires one integrated seed")
    multiplier = int(raw_v2_4["audit_base"]["audit_multiplier"])
    base_value = float(base_t.iloc[0]["T_global_8"])
    selected_value = float(selected_t.iloc[0]["T_global_8"])
    projected = base_value / math.sqrt(multiplier)
    calibration_tasks = (
        int(raw_v2_4["step_b"]["calibration_clean_seeds"]["count"]) * 4
    )
    validation_tasks = (
        int(raw_v2_4["step_b"]["validation_clean_seeds"]["count"]) * 4
    )
    calibration_mean = float(
        np.mean([row["elapsed_seconds"] for row in selected_calibration])
    )
    validation_mean = float(
        np.mean([row["elapsed_seconds"] for row in selected_validation])
    )
    projected_worker_seconds = (
        calibration_tasks * calibration_mean
        + validation_tasks * validation_mean
    )
    payload = {
        "schema_version": "auroc-gate-step-b-smoke-projection-v2.4",
        "status": "SMOKE_COMPLETE",
        "selected_multiplier": multiplier,
        "runtime": {
            "selected_calibration_cell_mean_seconds": calibration_mean,
            "selected_validation_cell_mean_seconds": validation_mean,
            "full_calibration_tasks": calibration_tasks,
            "full_validation_tasks": validation_tasks,
            "workers": workers,
            "projected_full_wall_seconds_ideal": (
                projected_worker_seconds / workers
            ),
            "projected_full_wall_hours_ideal": (
                projected_worker_seconds / workers / 3600
            ),
        },
        "memory": {
            "maximum_worker_peak_rss_kb": max(
                int(row["peak_rss_kb"]) for row in checkpoints
            ),
            "projected_concurrent_peak_rss_kb_upper": (
                max(int(row["peak_rss_kb"]) for row in checkpoints) * workers
            ),
        },
        "variance_floor_risk": {
            "base_N_global_T": base_value,
            "selected_N_global_T": selected_value,
            "inverse_sqrt_projected_global_T": projected,
            "selected_to_projected_ratio": (
                selected_value / projected if projected else None
            ),
            "adverse_signal": selected_value > projected,
            "interpretation": (
                "one nested seed is a risk signal only and cannot estimate "
                "a variance floor"
            ),
        },
        "base_task_count": len(base),
    }
    with (output / "smoke_projection.json").open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/benchmark_v2/main_v2_4_candidate.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()
    raw_v2_4 = yaml.safe_load(config_path.read_text())
    source_config_path = (
        repo_root / raw_v2_4["source"]["v2_3_config"]
    ).resolve()
    raw_v2_3 = yaml.safe_load(source_config_path.read_text())
    default_output = (
        "artifacts/benchmark_v2_4/step_b_smoke"
        if args.mode == "smoke"
        else "artifacts/benchmark_v2_4/auroc_calibration"
    )
    output = (repo_root / (args.output_root or default_output)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    logical_cpus = os.cpu_count() or 1
    threads_per_worker = max(1, logical_cpus // args.workers)
    write_run_manifest(
        output=output,
        repo_root=repo_root,
        config_path=config_path,
        source_config_path=source_config_path,
        raw_v2_4=raw_v2_4,
        mode=args.mode,
        workers=args.workers,
        threads_per_worker=threads_per_worker,
    )
    tasks = build_tasks(
        raw_v2_3=raw_v2_3,
        raw_v2_4=raw_v2_4,
        mode=args.mode,
        threads_per_worker=threads_per_worker,
    )
    pending = [
        payload
        for payload in tasks
        if not (output / "checkpoints" / f"{task_key(payload)}.json").exists()
    ]
    print(
        json.dumps(
            {
                "mode": args.mode,
                "tasks_total": len(tasks),
                "tasks_completed": len(tasks) - len(pending),
                "tasks_pending": len(pending),
                "workers": args.workers,
                "threads_per_worker": threads_per_worker,
                "output": str(output.relative_to(repo_root)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    completion_log = output / "completed_tasks.jsonl"
    if pending:
        completed_now = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(worker, payload): payload for payload in pending}
            for future in as_completed(futures):
                payload = futures[future]
                result = future.result()
                checkpoint = output / "checkpoints" / f"{result['task_key']}.json"
                if checkpoint.exists():
                    raise FileExistsError(f"checkpoint collision: {checkpoint}")
                atomic_write_json(checkpoint, result)
                append_completion(completion_log, result)
                completed_now += 1
                print(
                    json.dumps(
                        {
                            "completed_now": completed_now,
                            "pending_at_start": len(pending),
                            "task_key": result["task_key"],
                            "elapsed_seconds": result["elapsed_seconds"],
                            "peak_rss_kb": result["peak_rss_kb"],
                            "checkpoint": str(checkpoint.relative_to(repo_root)),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    final_manifest = output / "calibration_manifest.json"
    if final_manifest.exists():
        manifest = json.loads(final_manifest.read_text())
    else:
        manifest = summarize(
            output=output,
            raw_v2_4=raw_v2_4,
            mode=args.mode,
            expected_tasks=len(tasks),
            workers=args.workers,
        )
    if args.mode == "smoke":
        projection_path = output / "smoke_projection.json"
        projection = (
            json.loads(projection_path.read_text())
            if projection_path.exists()
            else write_smoke_projection(
                output=output,
                raw_v2_4=raw_v2_4,
                workers=args.workers,
            )
        )
        manifest["smoke_projection"] = projection
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
