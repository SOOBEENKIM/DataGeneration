from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


ANALYSIS_BASE_HEAD = "fbaaac55cd2506e806e22bb7c26cdbaba1ed2997"
MODEL_IDS = (
    "ctgan_separate_class",
    "tvae_separate_class",
    "cof_seqgen",
)
BLOCKING_MODEL_IDS = (
    "ctgan_separate_class",
    "cof_seqgen",
)
METRICS = (
    "amount_ks",
    "gap_ks",
    "amount_abs_standardized_label_effect",
    "gap_abs_standardized_label_effect",
    "receiver_max_abs_signed_frequency",
)
EXPECTED_HASHES = {
    "aggregate_complete": (
        "artifacts/benchmark_v2_7/candidate_selection/"
        "aggregate_attempt_001/AGGREGATE_COMPLETE.json",
        "a40630f51e1a07fb509f8f676eabf52f9bd18affbd7a9de47c510e2e1c0792b2",
    ),
    "aggregate_index": (
        "artifacts/benchmark_v2_7/candidate_selection/"
        "aggregate_attempt_001/artifact_index.json",
        "2539c02a5cec95ec85b21e2d1dec2798d905e63aac21723dfcdc103195a07b89",
    ),
    "aggregate_checksum": (
        "artifacts/benchmark_v2_7/candidate_selection/"
        "aggregate_attempt_001/checksum_manifest.json",
        "da1b2244d718ac9000758c4c6a5411b83e0cd7badd6cd1372ca58f064b1b2496",
    ),
    "selection_report": (
        "artifacts/benchmark_v2_7/candidate_selection/"
        "aggregate_attempt_001/selection_report.json",
        "3af94b53f5dbaaeca917a04d2b490c0267f70aa915e38e2f751b449451122ec3",
    ),
    "selection_manifest": (
        "artifacts/benchmark_v2_7/candidate_selection/"
        "aggregate_attempt_001/selection_manifest.json",
        "53807d3b399d74aea77c2a1b9cced1534bcf6735b7b27cf8fa8b35b74063a07b",
    ),
    "aggregate_authorization": (
        "artifacts/benchmark_v2_7/authorizations/"
        "authorization_fbaaac55_aggregate_attempt_001.json",
        "5961ce4b042553c7ff058984e37d38b69176118a1e0b3b92898f8f61bccb5d64",
    ),
    "execution_authorization": (
        "artifacts/benchmark_v2_7/authorizations/"
        "authorization_de57f79_attempt_001.json",
        "ff8bfdab5177d4b067e74f2cf192590b152236406b2eb4c1664878c20b5beea5",
    ),
    "runner_config": (
        "configs/benchmark_v2/evaluation_only_v2_7.yaml",
        "05435443f69f25f56e6d738dcae7c5a62f0c99096483b5ac26c6f525cffae68c",
    ),
    "candidate_config": (
        "configs/benchmark_v2/selection_v2_7_source_preparation.yaml",
        "2737723a05ce8628b7c9b8e1afbe7edae323a06971fbaf1eb00465224e0d654b",
    ),
    "v2_5_config": (
        "configs/benchmark_v2/full_v2_5.yaml",
        "81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3",
    ),
    "v2_5_final": (
        "artifacts/benchmark_v2_5/full/FINAL_COMPLETE.json",
        "e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a",
    ),
    "v2_5_frozen_manifest": (
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
        "kappa_1.00/data_manifest.json",
        "b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05",
    ),
    "v2_6_forensic_terminal": (
        "artifacts/benchmark_v2_6/selection_single_factor/"
        "forensic_attempt_001/FORENSIC_COMPLETE.json",
        "0c2e89a4124d35ccfa565ab504fbedf8819bed2dfe649370f811efe51a408703",
    ),
    "train_split": (
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
        "kappa_1.00/train.npz",
        "c67a6fce4593317e11e1b6f36397bfb38f9607916bf8f5b32329d8fa99b700a8",
    ),
    "validation_split": (
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
        "kappa_1.00/validation.npz",
        "68c15c05c55742bfce0f22560ffe8378d032a7a9fd9a1072e80ec8da5340fba5",
    ),
}
EXPECTED_TREES = {
    "candidate_input": {
        "sha256": (
            "de3fda560550cb8c815774a28be9bcc2350db32075a6776750fc66061b3a520e"
        ),
        "files": 51,
        "bytes": 6_993_542,
    },
    "aggregate_bundle": {
        "sha256": (
            "18d8e4d0fc5dfd148dd6e67036e063b6ad6db1a1f640c322b50bc825aa824378"
        ),
        "files": 6,
        "bytes": 34_223,
    },
}
EVALUATION_PATHS = {
    "ctgan_v27_c01_amount_quantile_inverse": (
        "artifacts/benchmark_v2_7/candidate_selection/evaluations/"
        "ctgan_separate_class/ctgan_v27_c01_amount_quantile_inverse/"
        "seed_2601/attempt_001/evaluation.json"
    ),
    "ctgan_v27_c02_categorical_logit": (
        "artifacts/benchmark_v2_7/candidate_selection/evaluations/"
        "ctgan_separate_class/ctgan_v27_c02_categorical_logit/"
        "seed_2601/attempt_001/evaluation.json"
    ),
    "tvae_v27_c01_amount_inverse_decoder": (
        "artifacts/benchmark_v2_7/candidate_selection/evaluations/"
        "tvae_separate_class/tvae_v27_c01_amount_inverse_decoder/"
        "seed_2601/attempt_001/evaluation.json"
    ),
    "tvae_v27_c02_bounded_temperature": (
        "artifacts/benchmark_v2_7/candidate_selection/evaluations/"
        "tvae_separate_class/tvae_v27_c02_bounded_temperature/"
        "seed_2601/attempt_001/evaluation.json"
    ),
    "cof_v27_c01_empirical_residual": (
        "artifacts/benchmark_v2_7/candidate_selection/evaluations/"
        "cof_seqgen/cof_v27_c01_empirical_residual/"
        "seed_2601/attempt_001/evaluation.json"
    ),
    "cof_v27_c02_gap_logit_bias": (
        "artifacts/benchmark_v2_7/candidate_selection/evaluations/"
        "cof_seqgen/cof_v27_c02_gap_logit_bias/"
        "seed_2601/attempt_001/evaluation.json"
    ),
}
EXECUTION_COUNTS = {
    "gpu_queries": 0,
    "cuda_calls": 0,
    "training_calls": 0,
    "model_restore_calls": 0,
    "model_sample_calls": 0,
    "candidate_reexecution_calls": 0,
    "guard_recalculation_calls": 0,
    "validation_sample_reads": 0,
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


def tree_digest(
    roots: Sequence[Path],
    *,
    relative_root: Path,
) -> Mapping[str, Any]:
    files = sorted(
        {
            path.resolve()
            for root in roots
            for path in root.rglob("*")
            if path.is_file()
        },
        key=lambda path: path.relative_to(relative_root).as_posix(),
    )
    digest = hashlib.sha256()
    byte_count = 0
    for path in files:
        relative = path.relative_to(relative_root).as_posix()
        byte_count += path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return {
        "sha256": digest.hexdigest(),
        "files": len(files),
        "bytes": byte_count,
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.name not in {"selection_report.json", "evaluation.json"}:
        raise ForensicContractError(
            "forensic data input must be stored selection_report.json "
            "or evaluation.json"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForensicContractError(f"cannot read JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ForensicContractError(f"JSON root is not an object: {path}")
    return value


def _verify_base_in_history(repository_root: Path) -> None:
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            ANALYSIS_BASE_HEAD,
            "HEAD",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ForensicContractError(
            "frozen v2.7 aggregate source is not preserved in history"
        )


def _verify_hash_inventory(
    repository_root: Path,
) -> Mapping[str, Any]:
    inventory: dict[str, Any] = {}
    for role, (relative, expected) in EXPECTED_HASHES.items():
        path = repository_root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ForensicContractError(f"{role} SHA-256 mismatch")
        inventory[role] = {
            "path": relative,
            "sha256": actual,
            "bytes": path.stat().st_size,
        }

    runtime = repository_root / (
        "artifacts/benchmark_v2_7/candidate_selection"
    )
    candidate_tree = tree_digest(
        (runtime / "workers", runtime / "evaluations"),
        relative_root=runtime,
    )
    aggregate_tree = tree_digest(
        (runtime / "aggregate_attempt_001",),
        relative_root=runtime / "aggregate_attempt_001",
    )
    if candidate_tree != EXPECTED_TREES["candidate_input"]:
        raise ForensicContractError("v2.7 candidate input tree mismatch")
    if aggregate_tree != EXPECTED_TREES["aggregate_bundle"]:
        raise ForensicContractError("v2.7 aggregate bundle tree mismatch")
    inventory["candidate_input_tree"] = candidate_tree
    inventory["aggregate_bundle_tree"] = aggregate_tree
    inventory["sampling_plan_sha256"] = (
        "862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27"
    )
    inventory["train_content_sha256"] = (
        "0c03f179930cd4d89ccc8a8b66283e620313e16d15fd53f0674c133bcb80c09d"
    )
    inventory["validation_content_sha256"] = (
        "aa1576b4ccae6a2ca30d0472f0782e1231ecc61ab3d224590a772dc3448b9e66"
    )
    return inventory


def _as_float(value: Any, *, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForensicContractError(f"{role} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ForensicContractError(f"{role} is not finite")
    return result


def _candidate_rows(
    *,
    report: Mapping[str, Any],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int, int, int, int]:
    raw_results = report.get("candidate_results")
    if not isinstance(raw_results, list) or len(raw_results) != 9:
        raise ForensicContractError(
            "selection report must contain exactly nine candidates"
        )
    selected = next(
        (
            item
            for item in raw_results
            if item.get("candidate_id")
            == "tvae_v27_c01_amount_inverse_decoder"
        ),
        None,
    )
    if not isinstance(selected, Mapping):
        raise ForensicContractError("TVAE selected comparator is missing")
    selected_stats = selected.get("statistics")
    if not isinstance(selected_stats, Mapping):
        raise ForensicContractError("TVAE comparator statistics missing")

    rows: list[dict[str, Any]] = []
    value_comparisons = 0
    value_matches = 0
    decision_comparisons = 0
    decision_matches = 0
    for result in raw_results:
        if not isinstance(result, Mapping):
            raise ForensicContractError("candidate result is not an object")
        candidate_id = str(result.get("candidate_id"))
        model_id = str(result.get("model_id"))
        if model_id not in MODEL_IDS:
            raise ForensicContractError("unexpected model in report")
        statistics = result.get("statistics")
        thresholds = result.get("thresholds")
        checks = result.get("checks")
        if not all(
            isinstance(value, Mapping)
            for value in (statistics, thresholds, checks)
        ):
            raise ForensicContractError(
                f"guard payload missing for {candidate_id}"
            )
        evaluation = evaluations.get(candidate_id)
        if evaluation is not None:
            if (
                evaluation.get("candidate_id") != candidate_id
                or evaluation.get("model_id") != model_id
                or evaluation.get("test_split_read") is not False
                or evaluation.get("evaluation_split") != "validation"
            ):
                raise ForensicContractError(
                    f"evaluation contract mismatch: {candidate_id}"
                )
            if (
                evaluation.get("sampling_plan_sha256")
                != report.get("sampling_plan_sha256")
            ):
                raise ForensicContractError(
                    f"SamplingPlan mismatch: {candidate_id}"
                )
            for metric in METRICS:
                value_comparisons += 1
                if _as_float(
                    evaluation["statistics"][metric],
                    role=f"{candidate_id}.{metric}",
                ) == _as_float(
                    statistics[metric],
                    role=f"report.{candidate_id}.{metric}",
                ):
                    value_matches += 1
            if (
                evaluation.get("checks") != checks
                or evaluation.get("thresholds") != thresholds
                or evaluation.get("status") != result.get("status")
                or evaluation.get("all_five_guards_pass")
                != result.get("all_five_guards_pass")
            ):
                raise ForensicContractError(
                    f"evaluation/report decision mismatch: {candidate_id}"
                )

        failed: list[str] = []
        ratios: dict[str, float] = {}
        row: dict[str, Any] = {
            "model_id": model_id,
            "candidate_id": candidate_id,
            "source_kind": str(result.get("source_kind")),
            "all_five_guards_pass": bool(
                result.get("all_five_guards_pass")
            ),
            "status": str(result.get("status")),
            "evaluation_sha256": str(result.get("evaluation_sha256")),
            "candidate_result_sha256": str(
                result.get("candidate_result_sha256")
            ),
            "checkpoint_sha256": str(
                result.get("checkpoint_sha256")
            ),
        }
        for metric in METRICS:
            statistic = _as_float(
                statistics[metric],
                role=f"{candidate_id}.{metric}",
            )
            threshold = _as_float(
                thresholds[metric],
                role=f"{candidate_id}.{metric}.threshold",
            )
            expected_check = "PASS" if statistic <= threshold else "FAIL"
            actual_check = str(checks[metric])
            decision_comparisons += 1
            if expected_check == actual_check:
                decision_matches += 1
            else:
                raise ForensicContractError(
                    f"stored threshold decision mismatch: "
                    f"{candidate_id}.{metric}"
                )
            if actual_check == "FAIL":
                failed.append(metric)
            ratios[metric] = statistic / threshold
            row[metric] = statistic
            row[f"{metric}_threshold"] = threshold
            row[f"{metric}_check"] = actual_check
            row[f"{metric}_ratio_to_threshold"] = ratios[metric]
            row[f"{metric}_delta_vs_tvae_selected"] = (
                statistic - float(selected_stats[metric])
            )
        row["failed_guards"] = failed
        row["largest_failure_component"] = max(
            failed,
            key=lambda metric: ratios[metric],
            default=None,
        )
        row["largest_failure_ratio"] = (
            ratios[row["largest_failure_component"]]
            if row["largest_failure_component"] is not None
            else 0.0
        )
        rows.append(row)
    return (
        rows,
        value_comparisons,
        value_matches,
        decision_comparisons,
        decision_matches,
    )


def _model_summaries(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    selections = report.get("model_selections")
    if not isinstance(selections, Mapping):
        raise ForensicContractError("model selections missing")
    summaries: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        model_rows = [
            row for row in rows if row["model_id"] == model_id
        ]
        selection = selections.get(model_id)
        if not isinstance(selection, Mapping):
            raise ForensicContractError(f"selection missing: {model_id}")
        summaries[model_id] = {
            "selection_status": str(selection.get("status")),
            "selected_candidate_id": selection.get(
                "selected_candidate_id"
            ),
            "all_five_guard_pass_candidates": [
                row["candidate_id"]
                for row in model_rows
                if row["all_five_guards_pass"]
            ],
            "candidate_failure_components": {
                row["candidate_id"]: row["failed_guards"]
                for row in model_rows
            },
        }
    return summaries


def _minimum_failure_components(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, list[str]]:
    output: dict[str, list[str]] = {}
    for model_id in BLOCKING_MODEL_IDS:
        failure_sets = [
            set(row["failed_guards"])
            for row in rows
            if row["model_id"] == model_id
        ]
        if not failure_sets:
            raise ForensicContractError(
                f"no candidate rows for {model_id}"
            )
        output[model_id] = sorted(set.intersection(*failure_sets))
    return output


def analyze_no_passing_candidates(
    repository_root: Path,
    *,
    extra_input_paths: Sequence[Path] = (),
) -> Mapping[str, Any]:
    repository_root = repository_root.resolve()
    _verify_base_in_history(repository_root)
    for path in extra_input_paths:
        if path.name not in {"selection_report.json", "evaluation.json"}:
            raise ForensicContractError(
                "forensic data input must be stored "
                "selection_report.json or evaluation.json"
            )

    inventory = _verify_hash_inventory(repository_root)
    report_path = repository_root / EXPECTED_HASHES[
        "selection_report"
    ][0]
    report = _read_json(report_path)
    evaluations: dict[str, Mapping[str, Any]] = {}
    evaluation_hashes: dict[str, Any] = {}
    report_by_id = {
        str(item["candidate_id"]): item
        for item in report["candidate_results"]
    }
    for candidate_id, relative in EVALUATION_PATHS.items():
        path = repository_root / relative
        actual_hash = sha256_file(path)
        expected_hash = str(
            report_by_id[candidate_id]["evaluation_sha256"]
        )
        if actual_hash != expected_hash:
            raise ForensicContractError(
                f"stored evaluation SHA-256 mismatch: {candidate_id}"
            )
        evaluations[candidate_id] = _read_json(path)
        evaluation_hashes[candidate_id] = {
            "path": relative,
            "sha256": actual_hash,
            "bytes": path.stat().st_size,
        }
    inventory["stored_evaluations"] = evaluation_hashes

    (
        rows,
        value_comparisons,
        value_matches,
        decision_comparisons,
        decision_matches,
    ) = _candidate_rows(report=report, evaluations=evaluations)
    summaries = _model_summaries(report, rows)
    minimum = _minimum_failure_components(rows)
    if (
        summaries["ctgan_separate_class"]["selection_status"]
        != "NO_PASSING_CANDIDATE"
        or summaries["cof_seqgen"]["selection_status"]
        != "NO_PASSING_CANDIDATE"
        or summaries["tvae_separate_class"]["selected_candidate_id"]
        != "tvae_v27_c01_amount_inverse_decoder"
        or report.get("primary_c2_selection_ready") is not False
    ):
        raise ForensicContractError(
            "frozen aggregate conclusion changed"
        )

    hypotheses = {
        "aggregate_implementation_defect": {
            "verdict": "REFUTED",
            "basis": (
                "All 30 non-control evaluation statistics match the "
                "selection report exactly; all 45 stored statistic/threshold "
                "decisions are internally consistent, and the TVAE passing "
                "control demonstrates the same aggregate path can select."
            ),
        },
        "evaluator_implementation_defect": {
            "verdict": "INCONCLUSIVE",
            "basis": (
                "No stored evaluation/report inconsistency supports an "
                "evaluator defect, and every stored sample contract is PASS. "
                "However, the requested read-only scope forbids recomputing "
                "guards from samples, so evaluator formula correctness is "
                "not independently re-established here."
            ),
        },
        "numeric_decode_sampling_path_limit": {
            "verdict": "SUPPORTED",
            "basis": (
                "CTGAN amount inversion repairs both amount guards while "
                "gap KS and receiver remain failed; its categorical-logit "
                "candidate leaves both common discrete blockers failed. CoF "
                "empirical residual sampling repairs both amount guards but "
                "leaves both gap guards failed, while gap-logit bias improves "
                "gap KS but worsens gap class effect."
            ),
        },
        "current_candidate_space_limit": {
            "verdict": "SUPPORTED",
            "basis": (
                "Every CTGAN candidate fails both gap KS and receiver "
                "frequency; every CoF candidate fails both gap KS and gap "
                "class effect. The finite preregistered candidates therefore "
                "contain no all-five-guard intersection."
            ),
        },
    }
    proposals = {
        "ctgan_separate_class": {
            "candidate_count": 1,
            "single_factor": "joint_discrete_decoder",
            "proposal": (
                "One train-only class-conditional joint (gap_bin, receiver) "
                "categorical decoder calibration, treated as one frozen "
                "discrete-decoder factor."
            ),
            "unchanged": (
                "amount quantile inverse, checkpoint, separate-class "
                "generators, thresholds, SamplingPlan, endpoint, and DGP"
            ),
            "rationale": (
                "gap KS and receiver frequency are each universal CTGAN "
                "blockers; one joint discrete-decoder boundary targets them "
                "without combining a numeric-path change."
            ),
        },
        "cof_seqgen": {
            "candidate_count": 1,
            "single_factor": "gap_sampler",
            "proposal": (
                "One train-only class-conditional gap-distribution sampler "
                "calibration replacing only the current gap-logit sampling "
                "map."
            ),
            "unchanged": (
                "empirical amount residual sampler, receiver path, "
                "checkpoint, architecture/objective, thresholds, "
                "SamplingPlan, endpoint, and DGP"
            ),
            "rationale": (
                "gap KS and gap class effect are each universal CoF blockers; "
                "a single class-conditional gap sampler addresses the shared "
                "channel without an amount or architecture change."
            ),
        },
    }
    return {
        "schema_version": (
            "benchmark-v2.7-no-passing-candidate-forensic-v1"
        ),
        "status": "COMPLETE",
        "purpose": "read_only_stored_evaluation_forensics",
        "analysis_base_head": ANALYSIS_BASE_HEAD,
        "hash_inventory": inventory,
        "data_sources": {
            "selection_reports": 1,
            "stored_evaluations": 6,
            "validation_samples": 0,
            "test_splits": 0,
        },
        "stored_evaluation_value_comparisons": value_comparisons,
        "stored_evaluation_value_matches": value_matches,
        "stored_guard_decision_comparisons": decision_comparisons,
        "stored_guard_decision_matches": decision_matches,
        "candidate_rows": rows,
        "model_summaries": summaries,
        "minimum_failure_components": minimum,
        "tvae_selected_comparator": (
            "tvae_v27_c01_amount_inverse_decoder"
        ),
        "hypotheses": hypotheses,
        "next_single_factor_proposals": proposals,
        "selection_conclusion_changed": False,
        "primary_c2_selection_ready": False,
        "fresh_test_authorized": False,
        "five_seed_full_run_authorized": False,
        "tstr_authorized": False,
        "privacy_authorized": False,
        "threshold_changed": False,
        "test_based_tuning": False,
        "execution_counts": dict(EXECUTION_COUNTS),
    }


def _csv_rows(
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    fields = [
        "model_id",
        "candidate_id",
        "source_kind",
        "status",
        "all_five_guards_pass",
        "failed_guards",
        "largest_failure_component",
        "largest_failure_ratio",
    ]
    for metric in METRICS:
        fields.extend(
            (
                metric,
                f"{metric}_threshold",
                f"{metric}_check",
                f"{metric}_ratio_to_threshold",
                f"{metric}_delta_vs_tvae_selected",
            )
        )
    rows = []
    for source in evidence["candidate_rows"]:
        rows.append(
            {
                field: (
                    ",".join(source[field])
                    if field == "failed_guards"
                    else source[field]
                )
                for field in fields
            }
        )
    return rows


def write_forensic_outputs(
    *,
    evidence: Mapping[str, Any],
    json_path: Path,
    csv_path: Path,
) -> Mapping[str, str]:
    if (
        json_path.name != "forensic_no_passing_candidate_v2_7.json"
        or csv_path.name != "forensic_no_passing_candidate_v2_7.csv"
        or json_path.parent.resolve() != csv_path.parent.resolve()
    ):
        raise ForensicContractError(
            "forensic outputs require the fixed JSON/CSV names"
        )
    if json_path.exists() or csv_path.exists():
        raise ForensicContractError(
            "append-only forensic output already exists"
        )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with json_path.open("x", encoding="utf-8") as handle:
            json.dump(
                evidence,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        rows = _csv_rows(evidence)
        with csv_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
    except FileExistsError as error:
        raise ForensicContractError(
            "append-only forensic output already exists"
        ) from error
    return {
        "json_path": str(json_path),
        "json_sha256": sha256_file(json_path),
        "csv_path": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explain the frozen v2.7 CTGAN/CoF selection failures from "
            "stored evaluation.json and selection_report.json only."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "docs/benchmark_v2/"
            "forensic_no_passing_candidate_v2_7.json"
        ),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path(
            "docs/benchmark_v2/"
            "forensic_no_passing_candidate_v2_7.csv"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.repository_root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path)

    evidence = analyze_no_passing_candidates(root)
    written = write_forensic_outputs(
        evidence=evidence,
        json_path=resolve(args.json_output),
        csv_path=resolve(args.csv_output),
    )
    print(json.dumps(written, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
