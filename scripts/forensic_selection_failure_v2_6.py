from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.validation_selection_v2_6 import load_development_context
from generators.empirical_conditional_iid import EmpiricalConditionalIID
from generators.sampling_plan import SamplingPlan
from scripts.forensic_row_marginal_v2_5 import (
    audit_sample_contract,
    independent_row_guard_statistics,
)


METRIC_KEYS = (
    "amount_ks",
    "gap_ks",
    "amount_abs_standardized_label_effect",
    "gap_abs_standardized_label_effect",
    "receiver_max_abs_signed_frequency",
)
MODEL_IDS = (
    "ctgan_separate_class",
    "tvae_separate_class",
    "cof_seqgen",
)
FORBIDDEN_PATH_PARTS = {
    "test",
    "test.npz",
    "test_split",
    "fresh_test",
    "fresh-test",
}
FORENSIC_SCHEMA = "benchmark-v2.6-selection-failure-forensic-v1"


class ForensicSelectionContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_read_path(path: Path, *, role: str) -> Path:
    resolved = path.resolve()
    if {part.lower() for part in resolved.parts} & FORBIDDEN_PATH_PARTS:
        raise ForensicSelectionContractError(
            f"{role} points to a forbidden test path: {resolved}"
        )
    if not resolved.is_file():
        raise ForensicSelectionContractError(
            f"{role} is missing: {resolved}"
        )
    return resolved


def _read_json(path: Path) -> Mapping[str, Any]:
    path = _safe_read_path(path, role="forensic JSON input")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForensicSelectionContractError(
            f"cannot read JSON: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise ForensicSelectionContractError(
            f"JSON root is not an object: {path}"
        )
    return value


def _read_yaml(path: Path) -> Mapping[str, Any]:
    path = _safe_read_path(path, role="forensic YAML input")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ForensicSelectionContractError(
            f"cannot read YAML: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise ForensicSelectionContractError(
            f"YAML root is not an object: {path}"
        )
    return value


def _load_synthetic(path: Path) -> SyntheticBatch:
    path = _safe_read_path(path, role="stored validation sample")
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {
                field: archive[field]
                for field in SyntheticBatch.__dataclass_fields__
            }
    except (OSError, ValueError, KeyError) as error:
        raise ForensicSelectionContractError(
            f"cannot load stored validation sample: {path}"
        ) from error
    return SyntheticBatch(**values)


def tree_digest(root: Path) -> Mapping[str, Any]:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in paths:
        digest.update(
            (
                f"{sha256_file(path)}  "
                f"{path.relative_to(root).as_posix()}\n"
            ).encode()
        )
    return {
        "algorithm": (
            "SHA256 over sorted '<file_sha256>  <relative_path>\\n' records"
        ),
        "sha256": digest.hexdigest(),
        "files": len(paths),
        "bytes": sum(path.stat().st_size for path in paths),
    }


def _batch_arrays(
    batch: SequenceBatch | SyntheticBatch,
) -> dict[str, np.ndarray]:
    return {
        field: getattr(batch, field)
        for field in SyntheticBatch.__dataclass_fields__
    }


def _signed_standardized_effect(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    groups = [np.asarray(values)[labels == label] for label in (0, 1)]
    if any(len(group) < 2 for group in groups):
        return float("inf")
    degrees = len(groups[0]) + len(groups[1]) - 2
    pooled = (
        (len(groups[0]) - 1) * groups[0].var(ddof=1)
        + (len(groups[1]) - 1) * groups[1].var(ddof=1)
    ) / degrees
    difference = float(groups[1].mean() - groups[0].mean())
    if pooled <= 0:
        return 0.0 if difference == 0 else float(
            np.copysign(np.inf, difference)
        )
    return float(difference / np.sqrt(pooled))


def _continuous_summary(values: np.ndarray) -> Mapping[str, Any]:
    quantile_levels = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    return {
        "rows": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "quantile_levels": list(quantile_levels),
        "quantiles": [
            float(value)
            for value in np.quantile(values, quantile_levels)
        ],
    }


def _frequency(values: np.ndarray, categories: int) -> list[float]:
    return (
        np.bincount(values, minlength=categories).astype(float)
        / len(values)
    ).tolist()


def decompose_batch(
    batch: SequenceBatch | SyntheticBatch,
    *,
    tau: np.ndarray,
    receiver_categories: int,
) -> Mapping[str, Any]:
    """Summarize the stored row channels without invoking a model/evaluator."""
    valid = batch.valid_mask
    labels = np.repeat(batch.y_entity, batch.lengths)
    amount = np.asarray(batch.x_num[..., 0][valid], dtype=float)
    gap_bins = np.asarray(batch.dt_bin[valid], dtype=np.int64)
    gap = np.asarray(tau, dtype=float)[gap_bins]
    receivers = np.asarray(
        batch.x_cat[..., 0][valid],
        dtype=np.int64,
    )
    gap_categories = len(tau)
    entity_receiver = np.zeros(
        (len(batch.lengths), receiver_categories),
        dtype=float,
    )
    for index, length_value in enumerate(batch.lengths):
        length = int(length_value)
        entity_receiver[index] = (
            np.bincount(
                batch.x_cat[index, :length, 0],
                minlength=receiver_categories,
            )
            / length
        )
    receiver_by_label = {
        str(label): entity_receiver[batch.y_entity == label].mean(0)
        for label in (0, 1)
    }
    receiver_signed = receiver_by_label["1"] - receiver_by_label["0"]
    return {
        "entities": int(len(batch.lengths)),
        "valid_rows": int(valid.sum()),
        "label_prevalence": float(batch.y_entity.mean()),
        "amount": {
            "overall": _continuous_summary(amount),
            "by_label": {
                str(label): _continuous_summary(amount[labels == label])
                for label in (0, 1)
            },
        },
        "gap": {
            "overall": _continuous_summary(gap),
            "by_label": {
                str(label): _continuous_summary(gap[labels == label])
                for label in (0, 1)
            },
        },
        "gap_bin_frequency": {
            "overall": _frequency(gap_bins, gap_categories),
            "by_label": {
                str(label): _frequency(
                    gap_bins[labels == label],
                    gap_categories,
                )
                for label in (0, 1)
            },
        },
        "receiver_frequency": {
            "overall": _frequency(receivers, receiver_categories),
            "by_label": {
                str(label): _frequency(
                    receivers[labels == label],
                    receiver_categories,
                )
                for label in (0, 1)
            },
        },
        "receiver_entity_frequency": {
            "by_label": {
                label: values.tolist()
                for label, values in receiver_by_label.items()
            },
        },
        "class_effect": {
            "amount_signed_standardized": _signed_standardized_effect(
                amount,
                labels,
            ),
            "gap_signed_standardized": _signed_standardized_effect(
                gap,
                labels,
            ),
            "receiver_signed_entity_frequency": (
                receiver_signed.tolist()
            ),
            "receiver_max_abs_signed_entity_frequency": float(
                np.max(np.abs(receiver_signed))
            ),
        },
    }


def training_cap_assessment(
    candidate_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Compare only the shared native 10k and 20k checkpoints."""
    models: dict[str, Any] = {}
    for model_id in sorted(
        {str(row["model_id"]) for row in candidate_results}
    ):
        by_id = {
            str(row["candidate_id"]): row
            for row in candidate_results
            if row["model_id"] == model_id
        }
        try:
            earlier = by_id["c00_native_checkpoint_10000"]
            later = by_id["c01_native_checkpoint_20000"]
        except KeyError as error:
            raise ValueError(
                f"{model_id} lacks the shared native 10k/20k pair"
            ) from error
        deltas = {
            key: float(earlier["statistics"][key])
            - float(later["statistics"][key])
            for key in METRIC_KEYS
        }
        improved = {
            key: value > 0
            for key, value in deltas.items()
        }
        models[model_id] = {
            "checkpoint_10000": {
                key: float(earlier["statistics"][key])
                for key in METRIC_KEYS
            },
            "checkpoint_20000": {
                key: float(later["statistics"][key])
                for key in METRIC_KEYS
            },
            "improvement_10000_minus_20000": deltas,
            "guard_improved": improved,
            "improved_guard_count": sum(improved.values()),
            "all_five_guards_improved": all(improved.values()),
        }
    all_improved = bool(models) and all(
        model["all_five_guards_improved"]
        for model in models.values()
    )
    return {
        "comparison": "shared_native_trajectory_10000_to_20000",
        "lower_is_better": True,
        "all_models_all_guards_improved": all_improved,
        "hypothesis": "SUPPORTED" if all_improved else "REFUTED",
        "models": models,
    }


def empirical_iid_feasibility(
    *,
    train: SequenceBatch,
    validation: SequenceBatch,
    plan: SamplingPlan,
    tau: np.ndarray,
    receiver_categories: int,
    thresholds: Mapping[str, float],
    seeds: Sequence[int],
) -> Mapping[str, Any]:
    """Replay the train-fitted label-conditional IID reference.

    The function receives in-memory train/validation objects and a frozen
    train-derived plan. It has no filesystem path and therefore cannot open a
    test split or mutate candidate runtime artifacts.
    """
    missing = set(METRIC_KEYS) - set(thresholds)
    if missing:
        raise ValueError(f"missing five-guard thresholds: {sorted(missing)}")
    reference = EmpiricalConditionalIID()
    reference.fit(train, config={}, seed=0)
    train_arrays = _batch_arrays(train)
    validation_arrays = _batch_arrays(validation)
    plan_arrays = {
        "y_entity": plan.y_entity,
        "lengths": plan.lengths,
        "valid_mask": plan.valid_mask,
    }
    trials = []
    for seed in seeds:
        sample = reference.sample(plan, seed=int(seed))
        sample_arrays = _batch_arrays(sample)
        contract = audit_sample_contract(
            sample_arrays,
            plan=plan_arrays,
            train=train_arrays,
        )
        statistics = independent_row_guard_statistics(
            validation_arrays,
            sample_arrays,
            tau=np.asarray(tau),
            receiver_categories=receiver_categories,
        )
        checks = {
            key: (
                "PASS"
                if np.isfinite(statistics[key])
                and statistics[key] <= float(thresholds[key])
                else "FAIL"
            )
            for key in METRIC_KEYS
        }
        all_pass = (
            contract["status"] == "PASS"
            and all(value == "PASS" for value in checks.values())
        )
        trials.append(
            {
                "seed": int(seed),
                "all_five_guards_pass": all_pass,
                "statistics": statistics,
                "thresholds": {
                    key: float(thresholds[key])
                    for key in METRIC_KEYS
                },
                "checks": checks,
                "sample_contract": contract["checks"],
            }
        )
    return {
        "reference": "train_fitted_label_conditional_empirical_iid",
        "sampling_plan_hash": plan.plan_hash,
        "all_trials_pass": bool(trials) and all(
            trial["all_five_guards_pass"] for trial in trials
        ),
        "trials": trials,
    }


def compare_decompositions(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Mapping[str, Any]:
    reference_amount = reference["amount"]["overall"]
    candidate_amount = candidate["amount"]["overall"]
    reference_gap = np.asarray(
        reference["gap_bin_frequency"]["overall"],
        dtype=float,
    )
    candidate_gap = np.asarray(
        candidate["gap_bin_frequency"]["overall"],
        dtype=float,
    )
    reference_receiver = np.asarray(
        reference["receiver_frequency"]["overall"],
        dtype=float,
    )
    candidate_receiver = np.asarray(
        candidate["receiver_frequency"]["overall"],
        dtype=float,
    )
    return {
        "amount_mean_delta": float(
            candidate_amount["mean"] - reference_amount["mean"]
        ),
        "amount_std_ratio": float(
            candidate_amount["std"] / reference_amount["std"]
        ),
        "amount_quantile_delta": [
            float(candidate_value - reference_value)
            for candidate_value, reference_value in zip(
                candidate_amount["quantiles"],
                reference_amount["quantiles"],
            )
        ],
        "gap_bin_total_variation": float(
            0.5 * np.abs(candidate_gap - reference_gap).sum()
        ),
        "gap_bin_max_abs_frequency_delta": float(
            np.max(np.abs(candidate_gap - reference_gap))
        ),
        "receiver_total_variation": float(
            0.5
            * np.abs(candidate_receiver - reference_receiver).sum()
        ),
        "receiver_max_abs_frequency_delta": float(
            np.max(np.abs(candidate_receiver - reference_receiver))
        ),
        "amount_class_effect_real": float(
            reference["class_effect"]["amount_signed_standardized"]
        ),
        "amount_class_effect_synthetic": float(
            candidate["class_effect"]["amount_signed_standardized"]
        ),
        "gap_class_effect_real": float(
            reference["class_effect"]["gap_signed_standardized"]
        ),
        "gap_class_effect_synthetic": float(
            candidate["class_effect"]["gap_signed_standardized"]
        ),
        "receiver_class_effect_real": float(
            reference["class_effect"][
                "receiver_max_abs_signed_entity_frequency"
            ]
        ),
        "receiver_class_effect_synthetic": float(
            candidate["class_effect"][
                "receiver_max_abs_signed_entity_frequency"
            ]
        ),
    }


def _inventory_file(repository_root: Path, relative: str) -> Mapping[str, Any]:
    path = repository_root / relative
    if not path.is_file():
        raise ForensicSelectionContractError(
            f"inventory input is missing: {relative}"
        )
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _source_evidence(repository_root: Path) -> Mapping[str, Any]:
    records = {
        "candidate_adapter": {
            "path": "generators/candidate_adapters_v2_6.py",
            "boundaries": [
                "29-108 train-only z-score fit/transform/inverse",
                "299-345 train-only fit and validation sampling/inverse",
            ],
        },
        "candidate_backends": {
            "path": "generators/candidate_model_backends_v2_6.py",
            "boundaries": [
                "57-107 CTGAN categorical activation/temperature",
                "109-137 CTGAN restored class-pair train transformer",
                "140-313 TVAE weighted loss and decode/inverse",
                "459-676 CoF loss/sampling boundary",
            ],
        },
        "class_pair_adapter": {
            "path": "generators/conditional_ctgan.py",
            "boundaries": [
                "137-247 valid-row flattening and y=0/y=1 fit",
                "252-263 class-pair sampling and SamplingPlan assembly",
            ],
        },
        "five_guard_evaluator": {
            "path": "eval/model_guards_v2_5.py",
            "boundaries": ["81-123 five row-guard statistics"],
        },
        "selection_harness": {
            "path": "eval/validation_selection_v2_6.py",
            "boundaries": [
                "496-591 frozen train/validation and SamplingPlan load",
                "613-829 stored sample validation and five-guard evaluation",
            ],
        },
    }
    for record in records.values():
        record["sha256"] = sha256_file(
            repository_root / record["path"]
        )
    return records


def _candidate_csv_row(
    result: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> Mapping[str, Any]:
    statistics = result["statistics"]
    checks = result["checks"]
    return {
        "row_kind": "learned_candidate",
        "model_id": result["model_id"],
        "candidate_id": result["candidate_id"],
        "seed": result["selection_seed"],
        "checkpoint_step": result["sampled_checkpoint_step"],
        **{
            key: statistics[key]
            for key in METRIC_KEYS
        },
        **{
            f"{key}_check": checks[key]
            for key in METRIC_KEYS
        },
        **comparison,
        "all_five_guards_pass": result["all_five_guards_pass"],
        "validation_sample_sha256": result["provenance"][
            "validation_sample_sha256"
        ],
    }


def _reference_csv_row(trial: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "row_kind": "empirical_iid_reference",
        "model_id": "empirical_iid",
        "candidate_id": "train_fitted_label_conditional",
        "seed": trial["seed"],
        "checkpoint_step": "",
        **{
            key: trial["statistics"][key]
            for key in METRIC_KEYS
        },
        **{
            f"{key}_check": trial["checks"][key]
            for key in METRIC_KEYS
        },
        "all_five_guards_pass": trial["all_five_guards_pass"],
        "validation_sample_sha256": "",
    }


def build_forensic_report(
    repository_root: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    repository_root = repository_root.resolve()
    config_path = repository_root / "configs/benchmark_v2/selection_v2_6.yaml"
    development_path = (
        repository_root / "configs/benchmark_v2/development_data_v2_6.yaml"
    )
    config = _read_yaml(config_path)
    context = load_development_context(
        repository_root=repository_root,
        manifest_path=development_path,
    )
    selection_report_path = (
        repository_root
        / "artifacts/benchmark_v2_6/selection/aggregate_attempt_001"
        / "selection_report.json"
    )
    selection_report = _read_json(selection_report_path)
    candidate_root = (
        repository_root
        / "artifacts/benchmark_v2_6/selection/candidates"
    )
    thresholds = config["validation_gate"]["thresholds"]
    real_summary = decompose_batch(
        context.validation,
        tau=context.tau,
        receiver_categories=context.receiver_categories,
    )
    candidate_records = []
    csv_rows: list[Mapping[str, Any]] = []
    result_rows = selection_report.get("candidate_results")
    if not isinstance(result_rows, list) or len(result_rows) != 12:
        raise ForensicSelectionContractError(
            "selection report must contain 12 primary candidate results"
        )
    for result in result_rows:
        model_id = str(result["model_id"])
        candidate_id = str(result["candidate_id"])
        if model_id not in MODEL_IDS:
            raise ForensicSelectionContractError(
                f"unexpected primary candidate model: {model_id}"
            )
        attempt = (
            candidate_root
            / model_id
            / candidate_id
            / "seed_2601"
            / "attempt_001"
        )
        manifest_path = attempt / "candidate_result.json"
        manifest = _read_json(manifest_path)
        sample_path = _safe_read_path(
            repository_root / str(manifest["validation_sample_path"]),
            role="candidate validation sample",
        )
        sample_hash = sha256_file(sample_path)
        if (
            sample_hash != manifest.get("validation_sample_sha256")
            or sample_hash != result.get("validation_sample_sha256")
        ):
            raise ForensicSelectionContractError(
                f"sample hash mismatch: {model_id}/{candidate_id}"
            )
        sample = _load_synthetic(sample_path)
        independent = independent_row_guard_statistics(
            _batch_arrays(context.validation),
            _batch_arrays(sample),
            tau=context.tau,
            receiver_categories=context.receiver_categories,
        )
        maximum_replay_error = max(
            abs(float(independent[key]) - float(result["statistics"][key]))
            for key in METRIC_KEYS
        )
        if maximum_replay_error > 1e-12:
            raise ForensicSelectionContractError(
                f"independent guard replay differs: "
                f"{model_id}/{candidate_id}"
            )
        contract = audit_sample_contract(
            _batch_arrays(sample),
            plan={
                "y_entity": context.plan.y_entity,
                "lengths": context.plan.lengths,
                "valid_mask": context.plan.valid_mask,
            },
            train=_batch_arrays(context.train),
        )
        if contract["status"] != "PASS":
            raise ForensicSelectionContractError(
                f"stored candidate sample contract failed: "
                f"{model_id}/{candidate_id}"
            )
        sample_summary = decompose_batch(
            sample,
            tau=context.tau,
            receiver_categories=context.receiver_categories,
        )
        comparison = compare_decompositions(
            real_summary,
            sample_summary,
        )
        record = {
            "model_id": model_id,
            "candidate_id": candidate_id,
            "selection_seed": int(result["selection_seed"]),
            "sampled_checkpoint_step": int(
                result["sampled_checkpoint_step"]
            ),
            "all_five_guards_pass": bool(
                result["all_five_guards_pass"]
            ),
            "statistics": {
                key: float(independent[key])
                for key in METRIC_KEYS
            },
            "checks": dict(result["checks"]),
            "sample_contract": contract,
            "distribution": sample_summary,
            "real_vs_synthetic": comparison,
            "provenance": {
                "candidate_result_path": str(
                    manifest_path.relative_to(repository_root)
                ),
                "candidate_result_sha256": sha256_file(manifest_path),
                "validation_sample_path": str(
                    sample_path.relative_to(repository_root)
                ),
                "validation_sample_sha256": sample_hash,
                "checkpoint_path": manifest["checkpoint_path"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "training_source_commit": manifest.get(
                    "training_source_commit",
                    manifest["source_commit"],
                ),
                "evaluation_source_commit": manifest.get(
                    "evaluation_source_commit",
                    manifest["source_commit"],
                ),
                "selection_config_sha256": manifest[
                    "selection_config_sha256"
                ],
                "development_manifest_sha256": manifest[
                    "development_manifest_sha256"
                ],
                "selection_plan_sha256": manifest[
                    "selection_plan_sha256"
                ],
            },
            "independent_replay_max_abs_error": maximum_replay_error,
        }
        candidate_records.append(record)
        csv_rows.append(_candidate_csv_row(record, comparison))

    reference = empirical_iid_feasibility(
        train=context.train,
        validation=context.validation,
        plan=context.plan,
        tau=context.tau,
        receiver_categories=context.receiver_categories,
        thresholds=thresholds,
        seeds=(2601, 2602, 2603, 2604, 2605),
    )
    csv_rows.extend(_reference_csv_row(row) for row in reference["trials"])
    cap = training_cap_assessment(candidate_records)
    source_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()
    aggregate_root = (
        repository_root
        / "artifacts/benchmark_v2_6/selection/aggregate_attempt_001"
    )
    inventory_paths = (
        "artifacts/benchmark_v2_6/selection/aggregate_attempt_001/"
        "AGGREGATE_COMPLETE.json",
        "artifacts/benchmark_v2_6/selection/aggregate_attempt_001/"
        "selection_report.json",
        "artifacts/benchmark_v2_6/selection/aggregate_attempt_001/"
        "selection_manifest.json",
        "artifacts/benchmark_v2_6/selection/aggregate_attempt_001/"
        "artifact_index.json",
        "artifacts/benchmark_v2_6/selection/aggregate_attempt_001/"
        "checksum_manifest.json",
        "artifacts/benchmark_v2_6/selection/authorization_history/"
        "authorization_f02004aa_primary_selection_attempt_001.json",
        "configs/benchmark_v2/selection_v2_6.yaml",
        "configs/benchmark_v2/development_data_v2_6.yaml",
        "configs/benchmark_v2/full_v2_5.yaml",
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00/"
        "data_manifest.json",
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00/"
        "train.npz",
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00/"
        "validation.npz",
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00/"
        "shared_sampling_plan.npz",
        "artifacts/benchmark_v2_5/full/FINAL_COMPLETE.json",
    )
    inventory = {
        path: _inventory_file(repository_root, path)
        for path in inventory_paths
    }
    if sha256_file(config_path) != (
        "0884a0144f74f6317ae9e636c0cf5657dedac21ff12cdf545a6b6e5e29be4c4d"
    ):
        raise ForensicSelectionContractError(
            "selection config does not match the frozen hash"
        )
    report = {
        "schema_version": FORENSIC_SCHEMA,
        "scope": {
            "source_head_at_analysis": source_head,
            "selection_status": "NO_PASSING_CANDIDATE",
            "primary_models": list(MODEL_IDS),
            "test_split_access": "FORBIDDEN_AND_NOT_ACCESSED",
            "fresh_test_runs": 0,
            "five_seed_or_full_runs": 0,
            "gpu_queries": 0,
            "gpu_training_runs": 0,
            "learned_fit_calls": 0,
            "learned_sample_calls": 0,
            "data_generation_runs": 0,
        },
        "inventory": {
            "files": inventory,
            "candidate_runtime_tree": tree_digest(candidate_root),
            "aggregate_tree": tree_digest(aggregate_root),
        },
        "frozen_development": {
            "train_file_sha256": context.train_file_sha256,
            "train_content_sha256": context.train_content_sha256,
            "validation_file_sha256": context.validation_file_sha256,
            "validation_content_sha256": (
                context.validation_content_sha256
            ),
            "development_manifest_sha256": context.manifest_sha256,
            "sampling_plan_sha256": context.plan.plan_hash,
            "validation_entities": len(context.validation.lengths),
            "validation_rows": int(context.validation.valid_mask.sum()),
        },
        "thresholds": {
            key: float(thresholds[key])
            for key in METRIC_KEYS
        },
        "real_validation_distribution": real_summary,
        "empirical_iid_feasibility": reference,
        "candidate_records": candidate_records,
        "training_cap_assessment": cap,
        "source_evidence": _source_evidence(repository_root),
        "hypotheses": {
            "A_threshold_feasibility": {
                "status": (
                    "REFUTED"
                    if reference["all_trials_pass"]
                    else "SUPPORTED"
                ),
                "basis": (
                    "all five fixed-seed train-fitted empirical-IID "
                    "replays passed the unchanged five guards"
                    if reference["all_trials_pass"]
                    else "at least one feasibility replay failed"
                ),
            },
            "B_numeric_decode_class_pair": {
                "status": "SUPPORTED",
                "basis": (
                    "all stored samples pass mask/plan/support, independent "
                    "guard replay exactly matches selection, and the "
                    "model/channel failure fingerprints are already present "
                    "in the stored decoded rows"
                ),
                "causal_sub_boundary": "INCONCLUSIVE",
                "limitation": (
                    "single validation seed and bundled c02/c03 changes do "
                    "not identify one operation as the sole causal boundary"
                ),
            },
            "C_training_cap_only": {
                "status": cap["hypothesis"],
                "basis": (
                    "the shared native 10k to 20k checkpoint transition "
                    "does not improve all five guards for any primary model"
                ),
            },
        },
    }
    return report, csv_rows


def write_forensic_outputs(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    repository_root: Path,
    json_path: Path,
    csv_path: Path,
) -> Mapping[str, Any]:
    repository_root = repository_root.resolve()
    docs_root = (repository_root / "docs").resolve()
    resolved_json = json_path.resolve()
    resolved_csv = csv_path.resolve()
    if (
        docs_root not in resolved_json.parents
        or docs_root not in resolved_csv.parents
    ):
        raise ForensicSelectionContractError(
            "forensic outputs are restricted to the versioned docs tree"
        )
    if resolved_json.exists() or resolved_csv.exists():
        raise ForensicSelectionContractError(
            "forensic output is append-only and already exists"
        )
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    fieldnames = sorted(
        {key for row in rows for key in row}
    )
    with resolved_csv.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return {
        "json": {
            "path": str(resolved_json.relative_to(repository_root)),
            "sha256": sha256_file(resolved_json),
        },
        "csv": {
            "path": str(resolved_csv.relative_to(repository_root)),
            "sha256": sha256_file(resolved_csv),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only v2.6 stored-sample selection-failure forensics"
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "docs/benchmark_v2/forensic_selection_failure_v2_6.json"
        ),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path(
            "docs/benchmark_v2/forensic_selection_failure_v2_6.csv"
        ),
    )
    args = parser.parse_args()
    root = args.repository_root.resolve()
    report, rows = build_forensic_report(root)
    result = write_forensic_outputs(
        report,
        rows,
        repository_root=root,
        json_path=(
            args.json_output
            if args.json_output.is_absolute()
            else root / args.json_output
        ),
        csv_path=(
            args.csv_output
            if args.csv_output.is_absolute()
            else root / args.csv_output
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
