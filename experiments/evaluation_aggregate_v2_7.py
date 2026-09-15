from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from eval.candidate_preparation_v2_7 import (
    canonical_sha256,
    sha256_file,
)
from experiments.evaluation_only_runner_v2_7 import (
    MODEL_IDS,
    EvaluationExecutionPlan,
    build_evaluation_execution_plan,
)
from experiments.candidate_preparation_runner_v2_7 import (
    build_preparation_execution_plan,
    dry_run_preparation,
)


class EvaluationAggregateContractError(RuntimeError):
    pass


METRIC_KEYS = (
    "amount_ks",
    "gap_ks",
    "amount_abs_standardized_label_effect",
    "gap_abs_standardized_label_effect",
    "receiver_max_abs_signed_frequency",
)
AGGREGATE_SCOPE = {
    "validation_selection": True,
    "aggregate_only": True,
    "gpu_query": False,
    "cuda": False,
    "training": False,
    "model_fit": False,
    "model_sample": False,
    "candidate_reexecution": False,
    "data_generation": False,
    "test_split_access": False,
    "threshold_change": False,
    "fresh_test": False,
    "tstr": False,
    "privacy": False,
    "five_seed_full_run": False,
}
AGGREGATE_RELEVANT_SOURCE_PATHS = (
    "configs/benchmark_v2/evaluation_only_v2_7.yaml",
    "configs/benchmark_v2/selection_v2_7_source_preparation.yaml",
    "eval/candidate_preparation_v2_7.py",
    "experiments/candidate_preparation_runner_v2_7.py",
    "experiments/evaluation_only_runner_v2_7.py",
    "experiments/evaluation_aggregate_v2_7.py",
    "scripts/aggregate_evaluation_selection_v2_7.py",
)
PRESERVED_V27_INPUT_TREE = {
    "sha256": (
        "de3fda560550cb8c815774a28be9bcc2350db32075a6776750fc66061b3a520e"
    ),
    "files": 51,
    "bytes": 6_993_542,
}
PRESERVED_EXECUTION_AUTHORIZATION_SHA256 = (
    "ff8bfdab5177d4b067e74f2cf192590b152236406b2eb4c1664878c20b5beea5"
)


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
        raise EvaluationAggregateContractError(
            f"append-only aggregate artifact exists: {path}"
        ) from error


def _read_json(path: Path, *, role: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationAggregateContractError(
            f"cannot read {role}: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise EvaluationAggregateContractError(
            f"{role} is not a JSON object: {path}"
        )
    return value


def _tree_record(
    *,
    relative_root: Path,
    roots: Sequence[Path],
) -> Mapping[str, Any]:
    files = sorted(
        {
            path
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


def inspect_v27_terminal_inventory(
    *,
    repository_root: Path,
    runtime_root: Path,
    plan: EvaluationExecutionPlan,
) -> Mapping[str, Any]:
    repository_root = repository_root.resolve()
    runtime_root = runtime_root.resolve()
    expected_runtime = (
        repository_root
        / "artifacts/benchmark_v2_7/candidate_selection"
    ).resolve()
    if runtime_root != expected_runtime:
        raise EvaluationAggregateContractError(
            "aggregate input is outside the v2.7 runtime root"
        )
    expected_by_model = {
        model_id: [
            operation
            for operation in plan.operations
            if operation.model_id == model_id
        ]
        for model_id in MODEL_IDS
    }
    candidates_by_id = {
        candidate.candidate_id: candidate
        for candidate in plan.candidate_plan.candidates
    }
    worker_records: dict[str, Any] = {}
    candidate_records: dict[str, Any] = {}
    common_source_commits: set[str] = set()
    common_source_hashes: set[str] = set()
    common_authorization_paths: set[str] = set()
    common_authorization_hashes: set[str] = set()
    for model_id in MODEL_IDS:
        worker_root = runtime_root / "workers" / model_id
        terminal_paths = [
            path
            for name in ("WORKER_COMPLETE.json", "WORKER_FAILED.json")
            if (path := worker_root / name).is_file()
        ]
        if len(terminal_paths) != 1:
            raise EvaluationAggregateContractError(
                f"{model_id} must have exactly one terminal marker"
            )
        terminal_path = terminal_paths[0]
        terminal = _read_json(
            terminal_path,
            role=f"{model_id} worker terminal",
        )
        ownership_path = worker_root / "ownership.lock"
        ownership = _read_json(
            ownership_path,
            role=f"{model_id} ownership",
        )
        expected_ids = [
            operation.candidate_id
            for operation in expected_by_model[model_id]
        ]
        terminal_rows = terminal.get("terminals")
        if (
            terminal_path.name != "WORKER_COMPLETE.json"
            or terminal.get("status") != "COMPLETE"
            or terminal.get("model_id") != model_id
            or terminal.get("test_split_read") is not False
            or not isinstance(terminal_rows, list)
            or [row.get("candidate_id") for row in terminal_rows]
            != expected_ids
            or any(
                row.get("status") != "COMPLETE"
                for row in terminal_rows
            )
        ):
            raise EvaluationAggregateContractError(
                f"{model_id} worker terminal contract mismatch"
            )
        for operation in expected_by_model[model_id]:
            seed_root = (
                runtime_root
                / "evaluations"
                / model_id
                / operation.candidate_id
                / "seed_2601"
            )
            attempts = sorted(
                path
                for path in seed_root.glob("attempt_*")
                if path.is_dir()
            )
            if len(attempts) != 1:
                raise EvaluationAggregateContractError(
                    "aggregate requires exactly one candidate attempt: "
                    f"{operation.candidate_id}"
                )
            attempt = attempts[0]
            expected_attempt = attempt.relative_to(
                repository_root
            ).as_posix()
            terminal_row = next(
                row
                for row in terminal_rows
                if row["candidate_id"] == operation.candidate_id
            )
            if terminal_row.get("attempt") != expected_attempt:
                raise EvaluationAggregateContractError(
                    f"candidate attempt path mismatch: "
                    f"{operation.candidate_id}"
                )
            manifest_path = attempt / "manifest.json"
            complete_path = attempt / "COMPLETE.json"
            if (
                not manifest_path.is_file()
                or not complete_path.is_file()
                or (attempt / "FAILED.json").exists()
            ):
                raise EvaluationAggregateContractError(
                    f"candidate is not COMPLETE-only: "
                    f"{operation.candidate_id}"
                )
            manifest = _read_json(
                manifest_path,
                role=f"{operation.candidate_id} manifest",
            )
            complete = _read_json(
                complete_path,
                role=f"{operation.candidate_id} terminal",
            )
            expected_manifest = {
                "model_id": model_id,
                "candidate_id": operation.candidate_id,
                "runner_config_sha256": plan.runner_config_sha256,
                "candidate_config_sha256": (
                    plan.candidate_plan.config_sha256
                ),
                "operation_definition_sha256": (
                    operation.definition_sha256
                ),
                "checkpoint_sha256": operation.checkpoint_sha256,
                "sampling_plan_sha256": (
                    plan.candidate_plan.sampling_plan_sha256
                ),
                "test_split_read": False,
            }
            if (
                complete.get("status") != "COMPLETE"
                or any(
                    manifest.get(key) != value
                    for key, value in expected_manifest.items()
                )
            ):
                raise EvaluationAggregateContractError(
                    f"candidate manifest/provenance mismatch: "
                    f"{operation.candidate_id}"
                )
            common_source_commits.add(
                str(manifest.get("source_commit"))
            )
            common_source_hashes.add(
                str(manifest.get("relevant_source_sha256"))
            )
            common_authorization_paths.add(
                str(manifest.get("authorization_path"))
            )
            common_authorization_hashes.add(
                str(manifest.get("authorization_sha256"))
            )
            file_hashes = {
                path.name: sha256_file(path)
                for path in sorted(attempt.iterdir())
                if path.is_file()
            }
            if operation.is_control:
                reference_path = attempt / "control_reference.json"
                reference = _read_json(
                    reference_path,
                    role=f"{operation.candidate_id} control reference",
                )
                candidate = candidates_by_id[operation.candidate_id]
                if (
                    reference.get("model_id") != model_id
                    or reference.get("candidate_id")
                    != operation.candidate_id
                    or reference.get("frozen_control")
                    != dict(candidate.frozen_control)
                    or complete.get("control_reference_sha256")
                    != sha256_file(reference_path)
                ):
                    raise EvaluationAggregateContractError(
                        f"frozen control reference mismatch: "
                        f"{operation.candidate_id}"
                    )
            else:
                evaluation_path = attempt / "evaluation.json"
                result_path = attempt / "candidate_result.json"
                sample_path = attempt / "validation_sample.npz"
                fit_state_path = attempt / "fit_state.json"
                evaluation = _read_json(
                    evaluation_path,
                    role=f"{operation.candidate_id} evaluation",
                )
                result = _read_json(
                    result_path,
                    role=f"{operation.candidate_id} result",
                )
                expected_hashes = {
                    "fit_state_sha256": sha256_file(fit_state_path),
                    "validation_sample_sha256": sha256_file(
                        sample_path
                    ),
                    "evaluation_sha256": sha256_file(evaluation_path),
                    "candidate_result_sha256": sha256_file(result_path),
                }
                expected_checks = set(
                    plan.validation_contract["thresholds"]
                )
                if (
                    any(
                        complete.get(key) != value
                        for key, value in expected_hashes.items()
                    )
                    or evaluation.get("model_id") != model_id
                    or evaluation.get("candidate_id")
                    != operation.candidate_id
                    or evaluation.get("evaluation_split")
                    != "validation"
                    or evaluation.get("fit_split") != "train"
                    or evaluation.get("thresholds")
                    != plan.validation_contract["thresholds"]
                    or set(evaluation.get("checks", {}))
                    != expected_checks
                    or evaluation.get("test_split_read") is not False
                    or result.get("model_id") != model_id
                    or result.get("candidate_id")
                    != operation.candidate_id
                    or result.get("validation_sample_sha256")
                    != expected_hashes["validation_sample_sha256"]
                    or result.get("all_five_guards_pass")
                    != evaluation.get("all_five_guards_pass")
                    or result.get("test_split_read") is not False
                ):
                    raise EvaluationAggregateContractError(
                        f"stored evaluation/result mismatch: "
                        f"{operation.candidate_id}"
                    )
            candidate_records[operation.candidate_id] = {
                "model_id": model_id,
                "attempt_path": expected_attempt,
                "manifest_sha256": sha256_file(manifest_path),
                "complete_sha256": sha256_file(complete_path),
                "artifacts": file_hashes,
            }
        if ownership.get("model_id") != model_id:
            raise EvaluationAggregateContractError(
                f"{model_id} ownership identity mismatch"
            )
        worker_records[model_id] = {
            "path": terminal_path.relative_to(
                repository_root
            ).as_posix(),
            "sha256": sha256_file(terminal_path),
            "ownership_path": ownership_path.relative_to(
                repository_root
            ).as_posix(),
            "ownership_sha256": sha256_file(ownership_path),
        }
    if (
        len(common_source_commits) != 1
        or len(common_source_hashes) != 1
        or len(common_authorization_paths) != 1
        or len(common_authorization_hashes) != 1
        or len(candidate_records) != 9
    ):
        raise EvaluationAggregateContractError(
            "candidate execution provenance is inconsistent"
        )
    source_commit = next(iter(common_source_commits))
    source_hash = next(iter(common_source_hashes))
    authorization_relative = next(iter(common_authorization_paths))
    authorization_hash = next(iter(common_authorization_hashes))
    authorization_path = (
        repository_root / authorization_relative
    ).resolve()
    if (
        not authorization_path.is_file()
        or sha256_file(authorization_path) != authorization_hash
    ):
        raise EvaluationAggregateContractError(
            "candidate execution authorization hash mismatch"
        )
    for model_id, record in worker_records.items():
        expected_provenance = canonical_sha256(
            {
                "source_commit": source_commit,
                "relevant_source_sha256": source_hash,
                "runner_config_sha256": plan.runner_config_sha256,
                "candidate_config_sha256": (
                    plan.candidate_plan.config_sha256
                ),
                "authorization_sha256": authorization_hash,
                "model_id": model_id,
            }
        )
        terminal = _read_json(
            repository_root / record["path"],
            role=f"{model_id} worker terminal",
        )
        ownership = _read_json(
            repository_root / record["ownership_path"],
            role=f"{model_id} ownership",
        )
        if (
            terminal.get("provenance_sha256")
            != expected_provenance
            or ownership.get("provenance_sha256")
            != expected_provenance
        ):
            raise EvaluationAggregateContractError(
                f"{model_id} worker provenance hash mismatch"
            )
    input_tree = _tree_record(
        relative_root=runtime_root,
        roots=(
            runtime_root / "workers",
            runtime_root / "evaluations",
        ),
    )
    inventory = {
        "schema_version": (
            "benchmark-v2.7-aggregate-input-inventory-v1"
        ),
        "status": "PASS",
        "worker_count": 3,
        "candidate_count": 9,
        "worker_terminals": worker_records,
        "candidate_artifacts": candidate_records,
        "candidate_source_commit": source_commit,
        "candidate_relevant_source_sha256": source_hash,
        "candidate_execution_authorization_path": (
            authorization_relative
        ),
        "candidate_execution_authorization_sha256": (
            authorization_hash
        ),
        "input_tree": input_tree,
        "test_split_read": False,
        "gpu_queries": 0,
        "cuda_calls": 0,
        "model_fit_calls": 0,
        "model_sample_calls": 0,
        "candidate_reexecution_calls": 0,
    }
    return {
        **inventory,
        "inventory_sha256": canonical_sha256(inventory),
    }


def _verified_guard_row(
    *,
    value: Mapping[str, Any],
    model_id: str,
    candidate_id: str,
    thresholds: Mapping[str, float],
    source_kind: str,
    candidate_result_sha256: str,
    validation_sample_sha256: str,
    evaluation_sha256: str,
    checkpoint_sha256: str,
) -> Mapping[str, Any]:
    statistics = value.get("statistics")
    checks = value.get("checks")
    if (
        not isinstance(statistics, Mapping)
        or not isinstance(checks, Mapping)
        or set(statistics) != set(METRIC_KEYS)
        or set(checks) != set(METRIC_KEYS)
        or value.get("thresholds") != thresholds
    ):
        raise EvaluationAggregateContractError(
            f"stored five-guard schema mismatch: {candidate_id}"
        )
    numeric = {
        key: float(statistics[key])
        for key in METRIC_KEYS
    }
    expected_checks = {
        key: (
            "PASS"
            if math.isfinite(numeric[key])
            and numeric[key] <= float(thresholds[key])
            else "FAIL"
        )
        for key in METRIC_KEYS
    }
    all_pass = all(
        expected_checks[key] == "PASS" for key in METRIC_KEYS
    )
    if (
        dict(checks) != expected_checks
        or value.get("all_five_guards_pass") is not all_pass
    ):
        raise EvaluationAggregateContractError(
            f"stored five-guard decision mismatch: {candidate_id}"
        )
    return {
        "model_id": model_id,
        "candidate_id": candidate_id,
        "status": "PASS" if all_pass else "FAIL",
        "all_five_guards_pass": all_pass,
        "statistics": numeric,
        "thresholds": dict(thresholds),
        "checks": expected_checks,
        "continuous_ks_max": max(
            numeric["amount_ks"],
            numeric["gap_ks"],
        ),
        "continuous_ks_sum": (
            numeric["amount_ks"] + numeric["gap_ks"]
        ),
        "source_kind": source_kind,
        "candidate_result_sha256": candidate_result_sha256,
        "validation_sample_sha256": validation_sample_sha256,
        "evaluation_sha256": evaluation_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }


def load_v27_stored_guard_results(
    *,
    repository_root: Path,
    runtime_root: Path,
    plan: EvaluationExecutionPlan,
    inventory: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    repository_root = repository_root.resolve()
    runtime_root = runtime_root.resolve()
    if (
        inventory.get("status") != "PASS"
        or inventory.get("candidate_count") != 9
        or inventory.get("test_split_read") is not False
    ):
        raise EvaluationAggregateContractError(
            "aggregate inventory is not ready"
        )
    prior_root = (
        repository_root
        / "artifacts/benchmark_v2_6/selection_single_factor/"
        "aggregate_attempt_001"
    )
    prior_terminal_path = prior_root / "AGGREGATE_COMPLETE.json"
    prior_report_path = prior_root / "selection_report.json"
    prior_terminal = _read_json(
        prior_terminal_path,
        role="frozen v2.6 aggregate terminal",
    )
    prior_report = _read_json(
        prior_report_path,
        role="frozen v2.6 selection report",
    )
    prior_candidates = prior_report.get("candidate_results")
    if (
        prior_terminal.get("status") != "COMPLETE"
        or prior_terminal.get("selection_report_sha256")
        != sha256_file(prior_report_path)
        or not isinstance(prior_candidates, list)
    ):
        raise EvaluationAggregateContractError(
            "frozen control guard evidence is not hash-bound"
        )
    operation_by_id = {
        operation.candidate_id: operation
        for operation in plan.operations
    }
    thresholds = dict(plan.validation_contract["thresholds"])
    results: list[Mapping[str, Any]] = []
    for candidate_id, record in inventory[
        "candidate_artifacts"
    ].items():
        operation = operation_by_id[candidate_id]
        attempt = repository_root / record["attempt_path"]
        if operation.is_control:
            reference_path = attempt / "control_reference.json"
            reference = _read_json(
                reference_path,
                role=f"{candidate_id} control reference",
            )
            frozen = reference["frozen_control"]
            matching = [
                row
                for row in prior_candidates
                if isinstance(row, Mapping)
                and row.get("model_id") == operation.model_id
                and row.get("candidate_result_sha256")
                == frozen["candidate_result_sha256"]
                and row.get("validation_sample_sha256")
                == frozen["validation_sample_sha256"]
                and row.get("checkpoint_sha256")
                == frozen["checkpoint_sha256"]
            ]
            if len(matching) != 1:
                raise EvaluationAggregateContractError(
                    f"frozen guard evidence is ambiguous: {candidate_id}"
                )
            results.append(
                _verified_guard_row(
                    value=matching[0],
                    model_id=operation.model_id,
                    candidate_id=candidate_id,
                    thresholds=thresholds,
                    source_kind="frozen_control",
                    candidate_result_sha256=frozen[
                        "candidate_result_sha256"
                    ],
                    validation_sample_sha256=frozen[
                        "validation_sample_sha256"
                    ],
                    evaluation_sha256=sha256_file(prior_report_path),
                    checkpoint_sha256=frozen["checkpoint_sha256"],
                )
            )
            continue
        evaluation_path = attempt / "evaluation.json"
        result_path = attempt / "candidate_result.json"
        evaluation = _read_json(
            evaluation_path,
            role=f"{candidate_id} stored evaluation",
        )
        candidate_result = _read_json(
            result_path,
            role=f"{candidate_id} stored result",
        )
        artifacts = record["artifacts"]
        if (
            artifacts.get("evaluation.json")
            != sha256_file(evaluation_path)
            or artifacts.get("candidate_result.json")
            != sha256_file(result_path)
            or candidate_result.get("all_five_guards_pass")
            != evaluation.get("all_five_guards_pass")
            or candidate_result.get("guard_status")
            != evaluation.get("status")
        ):
            raise EvaluationAggregateContractError(
                f"stored result hash/status mismatch: {candidate_id}"
            )
        results.append(
            _verified_guard_row(
                value=evaluation,
                model_id=operation.model_id,
                candidate_id=candidate_id,
                thresholds=thresholds,
                source_kind="stored_v2_7_evaluation",
                candidate_result_sha256=sha256_file(result_path),
                validation_sample_sha256=artifacts[
                    "validation_sample.npz"
                ],
                evaluation_sha256=sha256_file(evaluation_path),
                checkpoint_sha256=operation.checkpoint_sha256,
            )
        )
    if len(results) != 9:
        raise EvaluationAggregateContractError(
            "stored guard family is incomplete"
        )
    frozen = plan.candidate_plan.frozen_contract
    provenance = {
        "runner_config_sha256": plan.runner_config_sha256,
        "candidate_config_sha256": (
            plan.candidate_plan.config_sha256
        ),
        "development_manifest_sha256": frozen[
            "development_manifest_sha256"
        ],
        "train_file_sha256": frozen["train_file_sha256"],
        "train_content_sha256": frozen["train_content_sha256"],
        "validation_file_sha256": frozen[
            "validation_file_sha256"
        ],
        "validation_content_sha256": frozen[
            "validation_content_sha256"
        ],
        "sampling_plan_sha256": frozen["sampling_plan_sha256"],
        "thresholds": thresholds,
        "selection_rule": {
            "eligibility": "all_five_row_marginal_guards_PASS",
            "primary_objective": "minimize_max_amount_ks_gap_ks",
            "secondary_objective": "minimize_sum_amount_ks_gap_ks",
            "final_tiebreaker": "lexicographic_candidate_id",
        },
        "frozen_control_evidence_path": prior_report_path.relative_to(
            repository_root
        ).as_posix(),
        "frozen_control_evidence_sha256": sha256_file(
            prior_report_path
        ),
        "validation_samples_read": 0,
        "guard_evaluations_executed": 0,
        "test_split_read": False,
    }
    return results, provenance


def validate_v27_aggregate_authorization(
    *,
    plan: EvaluationExecutionPlan,
    inventory: Mapping[str, Any],
    authorization: Mapping[str, Any],
    source_commit: str,
    relevant_source_sha256: str,
) -> None:
    frozen = plan.candidate_plan.frozen_contract
    frozen_keys = (
        "development_manifest_sha256",
        "train_file_sha256",
        "train_content_sha256",
        "validation_file_sha256",
        "validation_content_sha256",
        "sampling_plan_sha256",
        "v2_5_frozen_manifest_sha256",
        "v2_5_final_complete_sha256",
        "v2_6_aggregate_complete_sha256",
        "v2_6_aggregate_index_sha256",
    )
    expected = {
        "schema_version": (
            "benchmark-v2.7-aggregate-only-authorization-v1"
        ),
        "status": "AUTHORIZED",
        "aggregate_source_commit": source_commit,
        "aggregate_relevant_source_sha256": relevant_source_sha256,
        "runner_config_sha256": plan.runner_config_sha256,
        "candidate_config_sha256": (
            plan.candidate_plan.config_sha256
        ),
        "candidate_source_commit": inventory[
            "candidate_source_commit"
        ],
        "candidate_relevant_source_sha256": inventory[
            "candidate_relevant_source_sha256"
        ],
        "candidate_execution_authorization_path": inventory[
            "candidate_execution_authorization_path"
        ],
        "candidate_execution_authorization_sha256": inventory[
            "candidate_execution_authorization_sha256"
        ],
        "input_inventory_sha256": inventory["inventory_sha256"],
        "input_tree": inventory["input_tree"],
        "approved_worker_terminals": inventory[
            "worker_terminals"
        ],
        "approved_candidate_artifacts": inventory[
            "candidate_artifacts"
        ],
        "frozen_provenance": {
            key: frozen[key] for key in frozen_keys
        },
        "scope": AGGREGATE_SCOPE,
    }
    mismatches = {
        key: {"expected": value, "actual": authorization.get(key)}
        for key, value in expected.items()
        if authorization.get(key) != value
    }
    if mismatches or not str(
        authorization.get("approval_text", "")
    ).strip():
        raise EvaluationAggregateContractError(
            f"aggregate authorization mismatch: {mismatches}"
        )


def v27_aggregate_relevant_source_sha256(
    repository_root: Path,
) -> str:
    repository_root = repository_root.resolve()
    digest = hashlib.sha256()
    for relative in AGGREGATE_RELEVANT_SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise EvaluationAggregateContractError(
                f"v2.7 aggregate source is missing: {relative}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _current_head(repository_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise EvaluationAggregateContractError(
            "cannot resolve current aggregate source commit"
        ) from error
    if len(value) != 40:
        raise EvaluationAggregateContractError(
            "current aggregate source commit is malformed"
        )
    return value


def execute_v27_validation_aggregate(
    *,
    repository_root: Path,
    config_path: Path,
    authorization_path: Path,
    output_root: Path,
    source_commit: str,
    mode: str,
) -> Mapping[str, Any]:
    if mode not in {"plan", "dry-run", "execute"}:
        raise EvaluationAggregateContractError(
            f"unknown v2.7 aggregate mode: {mode}"
        )
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    authorization_path = authorization_path.resolve()
    output_root = output_root.resolve()
    runtime_root = (
        repository_root
        / "artifacts/benchmark_v2_7/candidate_selection"
    ).resolve()
    authorization_root = (
        repository_root
        / "artifacts/benchmark_v2_7/authorizations"
    ).resolve()
    if (
        authorization_path.parent != authorization_root
        or "aggregate" not in authorization_path.name
        or not authorization_path.is_file()
    ):
        raise EvaluationAggregateContractError(
            "aggregate authorization is outside append-only history"
        )
    if (
        output_root.parent != runtime_root
        or not output_root.name.startswith("aggregate_attempt_")
    ):
        raise EvaluationAggregateContractError(
            "aggregate output is outside the v2.7 runtime root"
        )
    if _current_head(repository_root) != source_commit:
        raise EvaluationAggregateContractError(
            "aggregate source commit is not current HEAD"
        )
    plan = build_evaluation_execution_plan(config_path)
    if plan.repository_root != repository_root:
        raise EvaluationAggregateContractError(
            "aggregate config belongs to another repository"
        )
    try:
        frozen_report = dry_run_preparation(
            build_preparation_execution_plan(
                plan.candidate_plan.config_path
            )
        )
    except Exception as error:
        raise EvaluationAggregateContractError(
            "frozen train/validation provenance verification failed"
        ) from error
    if (
        frozen_report.get("frozen_provenance_verified") is not True
        or frozen_report.get("runtime_artifact_created") is not False
        or frozen_report.get("current_execution_counts", {}).get(
            "test_split_reads"
        )
        != 0
    ):
        raise EvaluationAggregateContractError(
            "frozen provenance report violates aggregate scope"
        )
    inventory = inspect_v27_terminal_inventory(
        repository_root=repository_root,
        runtime_root=runtime_root,
        plan=plan,
    )
    if (
        inventory["input_tree"] != PRESERVED_V27_INPUT_TREE
        or inventory["candidate_execution_authorization_sha256"]
        != PRESERVED_EXECUTION_AUTHORIZATION_SHA256
    ):
        raise EvaluationAggregateContractError(
            "preserved v2.7 candidate input tree mismatch"
        )
    authorization = _read_json(
        authorization_path,
        role="v2.7 aggregate authorization",
    )
    relevant_source_hash = v27_aggregate_relevant_source_sha256(
        repository_root
    )
    validate_v27_aggregate_authorization(
        plan=plan,
        inventory=inventory,
        authorization=authorization,
        source_commit=source_commit,
        relevant_source_sha256=relevant_source_hash,
    )
    readiness = {
        **inventory,
        "aggregate_source_commit": source_commit,
        "aggregate_relevant_source_sha256": relevant_source_hash,
        "aggregate_authorization_path": (
            authorization_path.relative_to(repository_root).as_posix()
        ),
        "aggregate_authorization_sha256": sha256_file(
            authorization_path
        ),
        "frozen_provenance_verified": True,
    }
    if mode == "plan":
        return {
            "mode": "plan",
            "readiness": readiness,
            "report": None,
            "selection_manifest": None,
            "terminal": None,
            "artifacts_written": False,
        }
    results, provenance = load_v27_stored_guard_results(
        repository_root=repository_root,
        runtime_root=runtime_root,
        plan=plan,
        inventory=inventory,
    )
    report, selection_manifest = build_v27_selection(
        candidate_results=results,
        provenance={
            **provenance,
            "aggregate_source_commit": source_commit,
            "aggregate_relevant_source_sha256": relevant_source_hash,
            "aggregate_authorization_sha256": sha256_file(
                authorization_path
            ),
            "candidate_source_commit": inventory[
                "candidate_source_commit"
            ],
            "candidate_relevant_source_sha256": inventory[
                "candidate_relevant_source_sha256"
            ],
        },
    )
    if mode == "dry-run":
        return {
            "mode": "dry-run",
            "readiness": readiness,
            "report": report,
            "selection_manifest": selection_manifest,
            "terminal": None,
            "artifacts_written": False,
        }
    terminal = write_v27_aggregate_bundle(
        output_root=output_root,
        authorization_path=authorization_path.relative_to(
            repository_root
        ),
        authorization_sha256=sha256_file(authorization_path),
        readiness=readiness,
        report=report,
        selection_manifest=selection_manifest,
        source_commit=source_commit,
        relevant_source_sha256=relevant_source_hash,
    )
    return {
        "mode": "execute",
        "readiness": readiness,
        "report": report,
        "selection_manifest": selection_manifest,
        "terminal": terminal,
        "artifacts_written": True,
    }


def _choose_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not candidates:
        raise EvaluationAggregateContractError(
            "candidate family is empty"
        )
    model_ids = {str(row.get("model_id")) for row in candidates}
    if len(model_ids) != 1:
        raise EvaluationAggregateContractError(
            "candidate family mixes models"
        )
    model_id = next(iter(model_ids))
    eligible = [
        row
        for row in candidates
        if row.get("all_five_guards_pass") is True
        and all(
            value == "PASS"
            for value in row.get("checks", {}).values()
        )
        and len(row.get("checks", {})) == 5
    ]
    if not eligible:
        return {
            "model_id": model_id,
            "status": "NO_PASSING_CANDIDATE",
            "selected_candidate_id": None,
            "reason": "no candidate passed all five validation guards",
        }
    selected = min(
        eligible,
        key=lambda row: (
            float(row["continuous_ks_max"]),
            float(row["continuous_ks_sum"]),
            str(row["candidate_id"]),
        ),
    )
    return {
        "model_id": model_id,
        "status": "SELECTED",
        "selected_candidate_id": selected["candidate_id"],
        "continuous_ks_max": selected["continuous_ks_max"],
        "continuous_ks_sum": selected["continuous_ks_sum"],
        "candidate_result_sha256": selected.get(
            "candidate_result_sha256"
        ),
        "validation_sample_sha256": selected.get(
            "validation_sample_sha256"
        ),
        "evaluation_sha256": selected.get("evaluation_sha256"),
    }


def build_v27_selection(
    *,
    candidate_results: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    by_model = {
        model_id: [
            row
            for row in candidate_results
            if row.get("model_id") == model_id
        ]
        for model_id in MODEL_IDS
    }
    if (
        len(candidate_results) != 9
        or any(len(rows) != 3 for rows in by_model.values())
    ):
        raise EvaluationAggregateContractError(
            "v2.7 selection requires exactly 3 x 3 candidates"
        )
    selections = {
        model_id: _choose_candidate(rows)
        for model_id, rows in by_model.items()
    }
    blocking = [
        model_id
        for model_id in MODEL_IDS
        if selections[model_id]["status"] != "SELECTED"
    ]
    ready = not blocking
    downstream = {
        "test_split_read": False,
        "fresh_test_authorized": False,
        "tstr_authorized": False,
        "privacy_authorized": False,
        "five_seed_full_run_authorized": False,
        "requires_separate_fresh_test_authorization": ready,
    }
    report = {
        "schema_version": "benchmark-v2.7-validation-selection-v1",
        "status": (
            "SELECTION_PASS_AWAITING_FRESH_TEST_AUTHORIZATION"
            if ready
            else "SELECTION_FAILED_PRIMARY_NO_FULL_RUN"
        ),
        "candidate_results": [dict(row) for row in candidate_results],
        "model_selections": selections,
        "primary_model_ids": list(MODEL_IDS),
        "primary_c2_selection_ready": ready,
        "blocking_primary_models": blocking,
        **downstream,
        **dict(provenance),
    }
    selected = (
        {
            model_id: dict(selections[model_id])
            for model_id in MODEL_IDS
        }
        if ready
        else {}
    )
    manifest = {
        "schema_version": (
            "benchmark-v2.7-validation-selection-manifest-v1"
        ),
        "status": (
            "FROZEN_AWAITING_SEPARATE_FRESH_TEST_AUTHORIZATION"
            if ready
            else "PRIMARY_SELECTION_FAILED_NO_FRESH_TEST"
        ),
        "selected_candidates": selected,
        "model_selection_status": {
            model_id: selections[model_id]["status"]
            for model_id in MODEL_IDS
        },
        "blocking_primary_models": blocking,
        "primary_c2_selection_ready": ready,
        **downstream,
        **dict(provenance),
    }
    return report, manifest


def write_v27_aggregate_bundle(
    *,
    output_root: Path,
    authorization_path: Path,
    authorization_sha256: str,
    readiness: Mapping[str, Any],
    report: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    source_commit: str,
    relevant_source_sha256: str,
) -> Mapping[str, Any]:
    output_root = output_root.resolve()
    try:
        output_root.mkdir(parents=True)
    except FileExistsError as error:
        raise EvaluationAggregateContractError(
            f"append-only aggregate attempt exists: {output_root}"
        ) from error
    values = (
        ("aggregate_readiness.json", readiness),
        ("selection_report.json", report),
        ("selection_manifest.json", selection_manifest),
    )
    for name, value in values:
        _exclusive_json(output_root / name, value)
    checksums = [
        {"path": name, "sha256": sha256_file(output_root / name)}
        for name, _ in values
    ]
    checksum_path = output_root / "checksum_manifest.json"
    _exclusive_json(
        checksum_path,
        {
            "schema_version": (
                "benchmark-v2.7-aggregate-checksum-manifest-v1"
            ),
            "status": "COMPLETE",
            "artifacts": checksums,
        },
    )
    index_records = [
        *checksums,
        {
            "path": checksum_path.name,
            "sha256": sha256_file(checksum_path),
        },
    ]
    index_path = output_root / "artifact_index.json"
    _exclusive_json(
        index_path,
        {
            "schema_version": (
                "benchmark-v2.7-aggregate-artifact-index-v1"
            ),
            "status": "COMPLETE",
            "source_commit": source_commit,
            "relevant_source_sha256": relevant_source_sha256,
            "authorization_path": str(authorization_path),
            "authorization_sha256": authorization_sha256,
            "artifacts": index_records,
        },
    )
    terminal = {
        "schema_version": "benchmark-v2.7-aggregate-terminal-v1",
        "status": "COMPLETE",
        "worker_count": 3,
        "candidate_count": 9,
        "authorization_sha256": authorization_sha256,
        "selection_report_sha256": sha256_file(
            output_root / "selection_report.json"
        ),
        "selection_manifest_sha256": sha256_file(
            output_root / "selection_manifest.json"
        ),
        "checksum_manifest_sha256": sha256_file(checksum_path),
        "artifact_index_sha256": sha256_file(index_path),
        "primary_c2_selection_ready": bool(
            report["primary_c2_selection_ready"]
        ),
        "blocking_primary_models": list(
            report["blocking_primary_models"]
        ),
        "test_split_read": False,
        "fresh_test_authorized": False,
        "tstr_authorized": False,
        "privacy_authorized": False,
        "five_seed_full_run_authorized": False,
        "gpu_queries": 0,
        "cuda_calls": 0,
        "model_fit_calls": 0,
        "model_sample_calls": 0,
        "candidate_reexecution_calls": 0,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _exclusive_json(output_root / "AGGREGATE_COMPLETE.json", terminal)
    return terminal
