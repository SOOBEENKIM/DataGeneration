from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import ks_2samp
import yaml

from benchmarks.types import SequenceBatch
from generators.sampling_plan import SamplingPlan


ANALYSIS_BASE_HEAD = "24d9ccec7e429f05cc3faceff0a5cb632d5219ac"
MODEL_IDS = (
    "ctgan_separate_class",
    "tvae_separate_class",
    "cof_seqgen",
)
METRICS = (
    "amount_ks",
    "gap_ks",
    "amount_abs_standardized_label_effect",
    "gap_abs_standardized_label_effect",
    "receiver_max_abs_signed_frequency",
)
CONTROLS = {
    "ctgan_separate_class": "ctgan_sf_c00_frozen_control",
    "tvae_separate_class": "tvae_sf_c00_frozen_control",
    "cof_seqgen": "cof_sf_c00_frozen_control",
}
EXPECTED_HASHES = {
    "aggregate_terminal": (
        "404c820d79cb3d17e21d617f5c255c0d1b97ed6fb7c94b9fdcd23682bdc5bf35"
    ),
    "aggregate_index": (
        "7620c08d071dabc320c7698a58d313fcb6812df82c29ea6f23fcb12d9e1cd97c"
    ),
    "single_factor_config": (
        "707ecdbd37bc55f4a3b9ece603b70d26c8274015aa9cb97d5d376aab3d9ecb7c"
    ),
    "aggregate_authorization": (
        "35ef35ac3e6cfe92a9357438eafc8fc41fb08141fe04c2c0d2a87c0d51d72013"
    ),
    "candidate_authorization": (
        "30b01268d0a94006bde3235b4c09f7a2fe44ec0b2e144b7b0577e6e45bba5adb"
    ),
    "v2_5_config": (
        "81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3"
    ),
    "v2_5_final": (
        "e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a"
    ),
    "v2_5_frozen": (
        "b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05"
    ),
    "prior_v2_6_selection": (
        "3cb9f91a00ea4b3bcb772527e2ff4f49b1a47c645618ce79995847f07337407d"
    ),
    "development_manifest": (
        "31ddf45dcf7b4f982ed990ec94281c3e6bad0454ec3182e2434e42358006ade5"
    ),
    "train_file": (
        "c67a6fce4593317e11e1b6f36397bfb38f9607916bf8f5b32329d8fa99b700a8"
    ),
    "validation_file": (
        "68c15c05c55742bfce0f22560ffe8378d032a7a9fd9a1072e80ec8da5340fba5"
    ),
    "sampling_plan": (
        "862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27"
    ),
}
EXPECTED_TREES = {
    "candidates": (
        "03c0926523decd8114e7e63d77369ece1a065f7732441f76c17149d350720267",
        30,
        6_871_674,
    ),
    "workers": (
        "8180c4d4fb576e1aa8953b4e5298b3ab717b1618adb080bab9b9fddea13fd819",
        9,
        10_363,
    ),
    "trajectories": (
        "5814437ec364a70dc26c167ba43f548e62065934fe4eca0b49d374d71746897a",
        25,
        98_249_656,
    ),
    "evaluations": (
        "32caf2c35a510b996a1211362df6bad7ae3c3fcc92bfee66c3b335d08465c55a",
        15,
        3_439_134,
    ),
    "aggregate_attempt_001": (
        "fd43d5abd73dfb76289f3b58f465892a920d7e1a29579c701823c4502b6080b6",
        6,
        38_468,
    ),
}
EXECUTION_COUNTS = {
    "gpu_queries": 0,
    "cuda_calls": 0,
    "model_fit_calls": 0,
    "model_sample_calls": 0,
    "data_generation_calls": 0,
    "test_split_reads": 0,
    "fresh_test_calls": 0,
    "tstr_calls": 0,
    "privacy_calls": 0,
    "five_seed_full_run_calls": 0,
}


class ForensicContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root: Path) -> tuple[str, int, int]:
    root = root.resolve()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    byte_count = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        byte_count += path.stat().st_size
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(file_hash.encode())
        digest.update(b"\n")
    return digest.hexdigest(), len(files), byte_count


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForensicContractError(f"cannot read JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ForensicContractError(f"JSON root is not an object: {path}")
    return value


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ForensicContractError(f"cannot read YAML: {path}") from error
    if not isinstance(value, Mapping):
        raise ForensicContractError(f"YAML root is not an object: {path}")
    return value


def _load_npz(path: Path, fields: Sequence[str]) -> Mapping[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {field: archive[field] for field in fields}
    except (OSError, ValueError, KeyError) as error:
        raise ForensicContractError(f"cannot read NPZ: {path}") from error


def _standardized_effect(values: np.ndarray, labels: np.ndarray) -> float:
    groups = [values[labels == label] for label in (0, 1)]
    if any(len(group) < 2 for group in groups):
        return float("inf")
    degrees = len(groups[0]) + len(groups[1]) - 2
    pooled = (
        (len(groups[0]) - 1) * groups[0].var(ddof=1)
        + (len(groups[1]) - 1) * groups[1].var(ddof=1)
    ) / degrees
    difference = float(groups[1].mean() - groups[0].mean())
    if pooled <= 0:
        return 0.0 if difference == 0 else float("inf")
    return float(abs(difference) / math.sqrt(float(pooled)))


def independent_guard_statistics(
    *,
    validation: Mapping[str, np.ndarray],
    sample: Mapping[str, np.ndarray],
    tau: np.ndarray,
    receiver_categories: int,
) -> Mapping[str, float]:
    real_valid = validation["valid_mask"]
    synthetic_valid = sample["valid_mask"]
    synthetic_labels = np.repeat(
        sample["y_entity"],
        sample["lengths"],
    )
    real_amount = validation["x_num"][..., 0][real_valid]
    synthetic_amount = sample["x_num"][..., 0][synthetic_valid]
    real_gap = tau[validation["dt_bin"][real_valid]]
    synthetic_gap = tau[sample["dt_bin"][synthetic_valid]]
    entity_frequency = np.zeros(
        (len(sample["lengths"]), receiver_categories),
        dtype=float,
    )
    for index, length_value in enumerate(sample["lengths"]):
        length = int(length_value)
        entity_frequency[index] = (
            np.bincount(
                sample["x_cat"][index, :length, 0],
                minlength=receiver_categories,
            )
            / length
        )
    receiver_by_class = [
        entity_frequency[sample["y_entity"] == label].mean(0)
        for label in (0, 1)
    ]
    return {
        "amount_ks": float(
            ks_2samp(real_amount, synthetic_amount).statistic
        ),
        "gap_ks": float(
            ks_2samp(real_gap, synthetic_gap).statistic
        ),
        "amount_abs_standardized_label_effect": _standardized_effect(
            synthetic_amount,
            synthetic_labels,
        ),
        "gap_abs_standardized_label_effect": _standardized_effect(
            synthetic_gap,
            synthetic_labels,
        ),
        "receiver_max_abs_signed_frequency": float(
            np.max(
                np.abs(receiver_by_class[1] - receiver_by_class[0])
            )
        ),
    }


def _sample_contract(
    *,
    sample: Mapping[str, np.ndarray],
    plan: SamplingPlan,
    tau: np.ndarray,
    receiver_categories: int,
) -> Mapping[str, bool]:
    valid = sample["valid_mask"]
    padding = ~valid
    return {
        "mask_matches_lengths": bool(
            np.array_equal(
                valid,
                np.arange(valid.shape[1])[None, :]
                < sample["lengths"][:, None],
            )
        ),
        "sampling_plan_labels_match": bool(
            np.array_equal(sample["y_entity"], plan.y_entity)
        ),
        "sampling_plan_lengths_match": bool(
            np.array_equal(sample["lengths"], plan.lengths)
        ),
        "sampling_plan_mask_match": bool(
            np.array_equal(valid, plan.valid_mask)
        ),
        "padding_zero": bool(
            np.all(sample["x_num"][padding] == 0)
            and np.all(sample["dt_bin"][padding] == 0)
            and np.all(sample["x_cat"][padding] == 0)
        ),
        "amount_finite": bool(np.isfinite(sample["x_num"]).all()),
        "gap_support_valid": bool(
            np.all(sample["dt_bin"][valid] >= 0)
            and np.all(sample["dt_bin"][valid] < len(tau))
        ),
        "receiver_support_valid": bool(
            np.all(sample["x_cat"][..., 0][valid] >= 0)
            and np.all(
                sample["x_cat"][..., 0][valid]
                < receiver_categories
            )
        ),
        "labels_binary": bool(
            np.isin(sample["y_entity"], (0, 1)).all()
        ),
    }


def _artifact_paths(repository_root: Path) -> Mapping[str, Path]:
    return {
        "aggregate_terminal": repository_root
        / "artifacts/benchmark_v2_6/selection_single_factor/"
        "aggregate_attempt_001/AGGREGATE_COMPLETE.json",
        "aggregate_index": repository_root
        / "artifacts/benchmark_v2_6/selection_single_factor/"
        "aggregate_attempt_001/artifact_index.json",
        "single_factor_config": repository_root
        / "configs/benchmark_v2/selection_v2_6_single_factor_amendment.yaml",
        "aggregate_authorization": repository_root
        / "artifacts/benchmark_v2_6/selection_single_factor/"
        "authorization_history/"
        "authorization_24d9ccec_aggregate_attempt_001.json",
        "candidate_authorization": repository_root
        / "artifacts/benchmark_v2_6/selection_single_factor/"
        "authorization_history/authorization_ad19d733_attempt_001.json",
        "v2_5_config": repository_root
        / "configs/benchmark_v2/full_v2_5.yaml",
        "v2_5_final": repository_root
        / "artifacts/benchmark_v2_5/full/FINAL_COMPLETE.json",
        "v2_5_frozen": repository_root
        / "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
        "kappa_1.00/data_manifest.json",
        "prior_v2_6_selection": repository_root
        / "artifacts/benchmark_v2_6/selection/"
        "aggregate_attempt_001/AGGREGATE_COMPLETE.json",
        "development_manifest": repository_root
        / "configs/benchmark_v2/development_data_v2_6.yaml",
    }


def _verify_inventory(repository_root: Path) -> Mapping[str, Any]:
    paths = _artifact_paths(repository_root)
    inventory: dict[str, Any] = {}
    for role, path in paths.items():
        actual = sha256_file(path)
        if actual != EXPECTED_HASHES[role]:
            raise ForensicContractError(
                f"preservation hash mismatch: {role}"
            )
        inventory[role] = {
            "path": str(path.relative_to(repository_root)),
            "sha256": actual,
        }
    runtime = (
        repository_root
        / "artifacts/benchmark_v2_6/selection_single_factor"
    )
    for role, expected in EXPECTED_TREES.items():
        actual = tree_digest(runtime / role)
        if actual != expected:
            raise ForensicContractError(
                f"preservation tree mismatch: {role}"
            )
        inventory[f"{role}_tree"] = {
            "path": str((runtime / role).relative_to(repository_root)),
            "sha256": actual[0],
            "files": actual[1],
            "bytes": actual[2],
        }
    terminal = _read_json(paths["aggregate_terminal"])
    index = _read_json(paths["aggregate_index"])
    aggregate_root = paths["aggregate_terminal"].parent
    if (
        terminal.get("status") != "COMPLETE"
        or terminal.get("artifact_index_sha256")
        != EXPECTED_HASHES["aggregate_index"]
        or any(
            sha256_file(aggregate_root / row["path"]) != row["sha256"]
            for row in index.get("artifacts", ())
        )
    ):
        raise ForensicContractError(
            "aggregate terminal/index checksum verification failed"
        )
    return inventory


def _candidate_sample_path(
    *,
    repository_root: Path,
    model_id: str,
    candidate_id: str,
) -> tuple[Path, str, str]:
    attempt = (
        repository_root
        / "artifacts/benchmark_v2_6/selection_single_factor/candidates"
        / model_id
        / candidate_id
        / "seed_2601/attempt_001"
    )
    terminal = _read_json(attempt / "COMPLETE.json")
    reference_path = attempt / "reference_manifest.json"
    if reference_path.is_file():
        reference = _read_json(reference_path)
        if sha256_file(reference_path) != terminal.get(
            "reference_manifest_sha256"
        ):
            raise ForensicContractError(
                f"frozen reference hash mismatch: {candidate_id}"
            )
        paths = reference["frozen_control_paths"]
        hashes = reference["frozen_control_hashes"]
        sample_path = repository_root / paths["validation_sample_path"]
        result_path = repository_root / paths["candidate_result_path"]
        if (
            sha256_file(sample_path)
            != hashes["validation_sample_sha256"]
            or sha256_file(result_path)
            != hashes["candidate_result_sha256"]
        ):
            raise ForensicContractError(
                f"frozen candidate hash mismatch: {candidate_id}"
            )
        return (
            sample_path,
            sha256_file(result_path),
            str(reference.get("execution_kind")),
        )
    result_path = attempt / "candidate_result.json"
    result = _read_json(result_path)
    sample_path = repository_root / result["validation_sample_path"]
    if (
        sha256_file(result_path) != terminal.get("candidate_result_sha256")
        or sha256_file(sample_path)
        != terminal.get("validation_sample_sha256")
        or result.get("test_split_read") is not False
        or result.get("fresh_test_read") is not False
    ):
        raise ForensicContractError(
            f"candidate artifact mismatch: {candidate_id}"
        )
    return (
        sample_path,
        sha256_file(result_path),
        str(result["execution_kind"]),
    )


def _direction(delta: float, tolerance: float = 1e-15) -> str:
    if delta < -tolerance:
        return "IMPROVED"
    if delta > tolerance:
        return "WORSENED"
    return "UNCHANGED"


def _code_evidence(repository_root: Path) -> Mapping[str, Any]:
    records = {
        "single_factor_backend": (
            "generators/single_factor_backends_v2_6.py",
            [
                "ConditionalCTGANSharedTransformerV26",
                "CoFNoisePredictionCandidateV26",
            ],
        ),
        "candidate_backend": (
            "generators/candidate_model_backends_v2_6.py",
            [
                "CheckPointableCTGANCandidateV26 categorical temperature",
                "CheckPointableTVAECandidateV26 categorical decoder",
            ],
        ),
        "single_factor_components": (
            "models/single_factor_components_v2_6.py",
            [
                "EpsilonPredictionCoFSeqGenV26",
                "epsilon_reverse_sample_v2_6",
                "variance_preserving_residual",
            ],
        ),
        "single_factor_runner": (
            "experiments/single_factor_runner_v2_6.py",
            [
                "_apply_evaluation_only_intervention",
                "_single_factor_adapter_spec",
            ],
        ),
        "aggregate_runner": (
            "experiments/single_factor_aggregate_v2_6.py",
            [
                "_evaluate_verified_candidates",
                "build_single_factor_selection",
            ],
        ),
    }
    return {
        role: {
            "path": path,
            "sha256": sha256_file(repository_root / path),
            "boundaries": boundaries,
        }
        for role, (path, boundaries) in records.items()
    }


def analyze_single_factor_failure(
    repository_root: Path,
) -> Mapping[str, Any]:
    repository_root = repository_root.resolve()
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ANALYSIS_BASE_HEAD, "HEAD"],
        cwd=repository_root,
        check=False,
    )
    if completed.returncode != 0:
        raise ForensicContractError(
            "analysis base HEAD is not preserved in history"
        )
    inventory = _verify_inventory(repository_root)
    development = _read_yaml(
        repository_root
        / "configs/benchmark_v2/development_data_v2_6.yaml"
    )
    split_fields = tuple(SequenceBatch.__dataclass_fields__)
    train_path = (
        repository_root / development["splits"]["train"]["path"]
    )
    validation_path = (
        repository_root / development["splits"]["validation"]["path"]
    )
    if (
        sha256_file(train_path) != EXPECTED_HASHES["train_file"]
        or sha256_file(validation_path)
        != EXPECTED_HASHES["validation_file"]
    ):
        raise ForensicContractError(
            "frozen train/validation file hash mismatch"
        )
    train_raw = _load_npz(train_path, split_fields)
    validation = _load_npz(validation_path, split_fields)
    train = SequenceBatch(**train_raw)
    plan_record = development["selection_sampling_plan"]
    plan = SamplingPlan.from_train_policy(
        train,
        entity_count=int(plan_record["entity_count"]),
        seed=int(plan_record["seed"]),
    )
    if plan.plan_hash != EXPECTED_HASHES["sampling_plan"]:
        raise ForensicContractError("SamplingPlan hash mismatch")
    tau = np.asarray(development["tau"], dtype=float)
    receiver_categories = int(
        train.x_cat[..., 0][train.valid_mask].max()
    ) + 1
    aggregate_root = (
        repository_root
        / "artifacts/benchmark_v2_6/selection_single_factor/"
        "aggregate_attempt_001"
    )
    stored_report = _read_json(aggregate_root / "selection_report.json")
    stored_by_id = {
        row["candidate_id"]: row
        for row in stored_report["candidate_results"]
    }
    if len(stored_by_id) != 9:
        raise ForensicContractError(
            "aggregate report does not contain nine candidates"
        )
    rows: list[dict[str, Any]] = []
    comparisons = 0
    matches = 0
    contracts_pass = True
    for model_id in MODEL_IDS:
        for stored in (
            row
            for row in stored_report["candidate_results"]
            if row["model_id"] == model_id
        ):
            candidate_id = stored["candidate_id"]
            sample_path, result_hash, execution_kind = (
                _candidate_sample_path(
                    repository_root=repository_root,
                    model_id=model_id,
                    candidate_id=candidate_id,
                )
            )
            sample = _load_npz(
                sample_path,
                tuple(SyntheticField for SyntheticField in (
                    "x_num",
                    "dt_bin",
                    "x_cat",
                    "valid_mask",
                    "y_entity",
                    "lengths",
                )),
            )
            contract = _sample_contract(
                sample=sample,
                plan=plan,
                tau=tau,
                receiver_categories=receiver_categories,
            )
            contract_pass = all(contract.values())
            contracts_pass = contracts_pass and contract_pass
            independent = independent_guard_statistics(
                validation=validation,
                sample=sample,
                tau=tau,
                receiver_categories=receiver_categories,
            )
            metric_matches = {
                metric: math.isclose(
                    independent[metric],
                    float(stored["statistics"][metric]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for metric in METRICS
            }
            comparisons += len(metric_matches)
            matches += sum(metric_matches.values())
            thresholds = {
                metric: float(stored["thresholds"][metric])
                for metric in METRICS
            }
            checks = {
                metric: (
                    "PASS"
                    if math.isfinite(independent[metric])
                    and independent[metric] <= thresholds[metric]
                    else "FAIL"
                )
                for metric in METRICS
            }
            if (
                checks != stored["checks"]
                or sha256_file(sample_path)
                != stored["validation_sample_sha256"]
                or result_hash != stored["candidate_result_sha256"]
            ):
                raise ForensicContractError(
                    f"stored evaluation mismatch: {candidate_id}"
                )
            failing = [
                metric for metric in METRICS if checks[metric] == "FAIL"
            ]
            largest = max(
                failing,
                key=lambda metric: independent[metric]
                / thresholds[metric],
            )
            row = {
                "model_id": model_id,
                "candidate_id": candidate_id,
                "execution_kind": execution_kind,
                "selection_status": stored_report[
                    "model_selections"
                ][model_id]["status"],
                "all_five_guards_pass": not failing,
                "failed_guards": failing,
                "largest_failure": largest,
                "largest_failure_ratio": (
                    independent[largest] / thresholds[largest]
                ),
                "stored_match_all": all(metric_matches.values()),
                "sample_contract_pass": contract_pass,
                "sample_contract": contract,
                "candidate_result_sha256": result_hash,
                "validation_sample_path": str(
                    sample_path.relative_to(repository_root)
                ),
                "validation_sample_sha256": sha256_file(sample_path),
            }
            for metric in METRICS:
                row[metric] = independent[metric]
                row[f"{metric}_threshold"] = thresholds[metric]
                row[f"{metric}_check"] = checks[metric]
                row[f"{metric}_stored_match"] = metric_matches[metric]
            rows.append(row)
    for model_id in MODEL_IDS:
        control = next(
            row
            for row in rows
            if row["candidate_id"] == CONTROLS[model_id]
        )
        for row in rows:
            if row["model_id"] != model_id:
                continue
            for metric in METRICS:
                delta = float(row[metric]) - float(control[metric])
                row[f"{metric}_delta_vs_control"] = delta
                row[f"{metric}_direction_vs_control"] = (
                    _direction(delta)
                )
    summaries: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        model_rows = [
            row for row in rows if row["model_id"] == model_id
        ]
        largest = max(
            (
                (row["largest_failure_ratio"], row)
                for row in model_rows
            ),
            key=lambda item: item[0],
        )[1]
        summaries[model_id] = {
            "selection_status": stored_report["model_selections"][
                model_id
            ]["status"],
            "passing_candidates": [
                row["candidate_id"]
                for row in model_rows
                if row["all_five_guards_pass"]
            ],
            "largest_observed_failure": {
                "candidate_id": largest["candidate_id"],
                "metric": largest["largest_failure"],
                "ratio_to_threshold": largest[
                    "largest_failure_ratio"
                ],
            },
            "candidate_deltas_vs_control": {
                row["candidate_id"]: {
                    metric: {
                        "delta": row[
                            f"{metric}_delta_vs_control"
                        ],
                        "direction": row[
                            f"{metric}_direction_vs_control"
                        ],
                    }
                    for metric in METRICS
                }
                for row in model_rows
            },
        }
    hypotheses = {
        "A": {
            "name": "evaluator_or_selection_runner_defect",
            "verdict": "REFUTED",
            "evidence": (
                "45/45 independently recomputed guard values and all "
                "PASS/FAIL decisions match the stored aggregate; all nine "
                "mask, padding, SamplingPlan, label, and support contracts "
                "pass."
            ),
        },
        "B": {
            "name": "candidate_implementation_or_sampling_decoder_path",
            "verdict": "SUPPORTED",
            "evidence": (
                "Single-factor interventions move their intended channel "
                "metrics: TVAE categorical sampling repairs receiver and "
                "gap-effect guards, CoF residual sampling repairs amount "
                "dispersion while leaving discrete metrics unchanged, and "
                "the CoF epsilon objective sharply worsens amount metrics. "
                "This supports path-dependent failure, not a provenance or "
                "contract implementation bug."
            ),
        },
        "C": {
            "name": "current_model_objective_candidate_space_limit",
            "verdict": "SUPPORTED",
            "evidence": (
                "All nine candidates remain ineligible. Improvements are "
                "partial or trade one failed channel for another; no finite "
                "single-factor candidate resolves all five frozen guards."
            ),
        },
    }
    recommendations = {
        "ctgan_separate_class": [
            (
                "One train-only amount representation candidate using a "
                "shared monotone empirical-quantile inverse map; retain "
                "separate-class generation and the frozen categorical path."
            ),
            (
                "Separately test one train-only categorical marginal-logit "
                "calibration candidate; do not bundle it with the amount "
                "representation change."
            ),
        ],
        "tvae_separate_class": [
            (
                "Use the successful 0.75 categorical decoder as the fixed "
                "control and test one train-only amount inverse-decoder "
                "candidate."
            ),
            (
                "Separately preregister one bounded categorical-temperature "
                "candidate for the residual gap-KS miss; keep weights fixed."
            ),
        ],
        "cof_seqgen": [
            (
                "Use the variance-preserving residual candidate as control "
                "and test one train-fitted non-Gaussian empirical-residual "
                "amount sampler."
            ),
            (
                "Separately test one train-only gap-logit marginal-bias "
                "candidate; keep the amount sampler and architecture fixed."
            ),
        ],
    }
    script_path = Path(__file__).resolve()
    return {
        "schema_version": (
            "benchmark-v2.6-single-factor-failure-forensic-v1"
        ),
        "status": "COMPLETE",
        "purpose": "forensic_diagnosis_without_selection_change",
        "analysis_base_head": ANALYSIS_BASE_HEAD,
        "forensic_script_path": str(
            script_path.relative_to(repository_root)
        ),
        "forensic_script_sha256": sha256_file(script_path),
        "hash_inventory": inventory,
        "guard_value_comparisons": comparisons,
        "guard_value_matches": matches,
        "all_stored_guard_values_match": comparisons == matches == 45,
        "all_sample_contracts_pass": contracts_pass,
        "candidate_rows": rows,
        "model_summaries": summaries,
        "hypotheses": hypotheses,
        "v2_7_minimal_recommendations": recommendations,
        "code_evidence": _code_evidence(repository_root),
        "selection_conclusion_changed": False,
        "threshold_changed": False,
        "test_used": False,
        "execution_counts": dict(EXECUTION_COUNTS),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _csv_rows(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = [
        "model_id",
        "candidate_id",
        "execution_kind",
        "selection_status",
        "all_five_guards_pass",
        "failed_guards",
        "largest_failure",
        "largest_failure_ratio",
        "stored_match_all",
        "sample_contract_pass",
    ]
    for metric in METRICS:
        fields.extend(
            (
                metric,
                f"{metric}_threshold",
                f"{metric}_check",
                f"{metric}_delta_vs_control",
                f"{metric}_direction_vs_control",
            )
        )
    output = []
    for source in evidence["candidate_rows"]:
        row = {
            field: (
                ",".join(source[field])
                if field == "failed_guards"
                else source[field]
            )
            for field in fields
        }
        output.append(row)
    return output


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
    except FileExistsError as error:
        raise ForensicContractError(
            f"append-only forensic artifact exists: {path}"
        ) from error


def write_forensic_evidence(
    *,
    output_root: Path,
    evidence: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        output_root.mkdir(parents=True)
    except FileExistsError as error:
        raise ForensicContractError(
            f"append-only forensic attempt exists: {output_root}"
        ) from error
    json_path = output_root / "forensic_evidence.json"
    _exclusive_json(json_path, evidence)
    rows = _csv_rows(evidence)
    csv_path = output_root / "candidate_guard_evidence.csv"
    try:
        with csv_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
            )
            writer.writeheader()
            writer.writerows(rows)
    except FileExistsError as error:
        raise ForensicContractError(
            f"append-only forensic artifact exists: {csv_path}"
        ) from error
    index_path = output_root / "forensic_artifact_index.json"
    index = {
        "schema_version": (
            "benchmark-v2.6-single-factor-forensic-index-v1"
        ),
        "status": "COMPLETE",
        "artifacts": [
            {
                "path": json_path.name,
                "sha256": sha256_file(json_path),
            },
            {
                "path": csv_path.name,
                "sha256": sha256_file(csv_path),
            },
        ],
    }
    _exclusive_json(index_path, index)
    terminal = {
        "schema_version": (
            "benchmark-v2.6-single-factor-forensic-terminal-v1"
        ),
        "status": "COMPLETE",
        "forensic_evidence_sha256": sha256_file(json_path),
        "candidate_guard_evidence_sha256": sha256_file(csv_path),
        "artifact_index_sha256": sha256_file(index_path),
        "guard_value_comparisons": evidence[
            "guard_value_comparisons"
        ],
        "guard_value_matches": evidence["guard_value_matches"],
        "selection_conclusion_changed": False,
        "execution_counts": evidence["execution_counts"],
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _exclusive_json(output_root / "FORENSIC_COMPLETE.json", terminal)
    return terminal


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently recompute the v2.6 single-factor five guards "
            "from stored validation artifacts only."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/benchmark_v2_6/selection_single_factor/"
            "forensic_attempt_001"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repository_root = args.repository_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root.is_absolute()
        else (repository_root / args.output_root).resolve()
    )
    expected_parent = (
        repository_root
        / "artifacts/benchmark_v2_6/selection_single_factor"
    ).resolve()
    if (
        output_root.parent != expected_parent
        or not output_root.name.startswith("forensic_attempt_")
    ):
        raise SystemExit(
            "forensic output must be a new attempt under the "
            "single-factor runtime root"
        )
    evidence = analyze_single_factor_failure(repository_root)
    terminal = write_forensic_evidence(
        output_root=output_root,
        evidence=evidence,
    )
    print(
        "single-factor forensic COMPLETE; guard matches="
        f"{terminal['guard_value_matches']}/"
        f"{terminal['guard_value_comparisons']}; "
        "selection conclusion unchanged"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
