from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from benchmarks.types import SyntheticBatch
from eval.model_guards_v2_5 import row_guard_statistics
from eval.single_factor_amendment_v2_6 import (
    FROZEN_HASHES,
    FROZEN_THRESHOLDS,
    MODEL_IDS,
    sha256_file,
)
from eval.validation_selection_v2_6 import (
    CONTINUOUS_KS_KEYS,
    METRIC_KEYS,
    load_development_context,
)
from generators.contracts_v2_5 import validate_synthetic_contract
from experiments.single_factor_runner_v2_6 import (
    DEVELOPMENT_MANIFEST_SHA256,
    PRIOR_SELECTION_COMPLETE_SHA256,
    RUNTIME_ROOT,
    V2_5_CONFIG_SHA256,
    V2_5_FINAL_COMPLETE_SHA256,
    V2_5_FROZEN_MANIFEST_SHA256,
    SingleFactorExecutionPlan,
    build_single_factor_execution_plan,
    single_factor_model_tree_sha256,
    single_factor_relevant_source_sha256,
    verify_single_factor_worker_candidate_artifacts,
)


AUTHORIZATION_SCHEMA = (
    "benchmark-v2.6-single-factor-aggregate-authorization-v1"
)
AGGREGATE_SCHEMA = "benchmark-v2.6-single-factor-selection-v1"
PRIMARY_MODEL_IDS = MODEL_IDS
EXPECTED_SCOPE = {
    "validation_selection": True,
    "aggregate_only": True,
    "gpu_query": False,
    "cuda": False,
    "model_fit": False,
    "model_sample": False,
    "data_generation": False,
    "test_split_access": False,
    "fresh_test": False,
    "tstr": False,
    "privacy": False,
    "five_seed_full_run": False,
}


class SingleFactorAggregateContractError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, *, role: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SingleFactorAggregateContractError(
            f"cannot read {role}: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise SingleFactorAggregateContractError(
            f"{role} is not a JSON object: {path}"
        )
    return value


def _read_yaml(path: Path, *, role: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SingleFactorAggregateContractError(
            f"cannot read {role}: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise SingleFactorAggregateContractError(
            f"{role} is not a YAML object: {path}"
        )
    return value


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
        raise SingleFactorAggregateContractError(
            f"append-only aggregate artifact exists: {path}"
        ) from error


def _current_head(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise SingleFactorAggregateContractError(
            "cannot resolve current source commit"
        )
    return value


def _below(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _repository_path(
    repository_root: Path,
    value: Any,
    *,
    role: str,
    root: Path | None = None,
) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise SingleFactorAggregateContractError(
            f"{role} must be repository-relative"
        )
    path = (repository_root / relative).resolve()
    lowered = {part.lower() for part in path.parts}
    if lowered & {
        "test",
        "test.npz",
        "test_split",
        "fresh_test",
        "fresh-test",
        "heldout_test",
    }:
        raise SingleFactorAggregateContractError(
            f"{role} accesses a forbidden test path"
        )
    if root is not None and not _below(path, root):
        raise SingleFactorAggregateContractError(
            f"{role} escapes its approved root"
        )
    if not path.is_file():
        raise SingleFactorAggregateContractError(
            f"{role} is missing: {path}"
        )
    return path


def _load_synthetic(path: Path) -> SyntheticBatch:
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {
                field: archive[field]
                for field in SyntheticBatch.__dataclass_fields__
            }
    except (OSError, ValueError, KeyError) as error:
        raise SingleFactorAggregateContractError(
            f"cannot load validation candidate: {path}"
        ) from error
    return SyntheticBatch(**values)


def _choose(results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not results:
        raise SingleFactorAggregateContractError(
            "candidate result family is empty"
        )
    model_ids = {str(result.get("model_id")) for result in results}
    if len(model_ids) != 1:
        raise SingleFactorAggregateContractError(
            "candidate family mixes models"
        )
    model_id = next(iter(model_ids))
    eligible = [
        result
        for result in results
        if result.get("all_five_guards_pass") is True
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
        key=lambda result: (
            float(result["continuous_ks_max"]),
            float(result["continuous_ks_sum"]),
            str(result["candidate_id"]),
        ),
    )
    return {
        "model_id": model_id,
        "status": "SELECTED",
        "selected_candidate_id": selected["candidate_id"],
        "continuous_ks_max": selected["continuous_ks_max"],
        "continuous_ks_sum": selected["continuous_ks_sum"],
        "candidate_definition_sha256": selected.get(
            "candidate_definition_sha256"
        ),
        "candidate_result_sha256": selected.get(
            "candidate_result_sha256"
        ),
        "validation_sample_sha256": selected.get(
            "validation_sample_sha256"
        ),
        "checkpoint_sha256": selected.get("checkpoint_sha256"),
    }


def build_single_factor_selection(
    *,
    candidate_results: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    by_model = {
        model_id: [
            result
            for result in candidate_results
            if result.get("model_id") == model_id
        ]
        for model_id in PRIMARY_MODEL_IDS
    }
    if (
        any(len(results) != 3 for results in by_model.values())
        or len(candidate_results) != 9
    ):
        raise SingleFactorAggregateContractError(
            "single-factor selection requires exactly 3 x 3 candidates"
        )
    selections = {
        model_id: _choose(results)
        for model_id, results in by_model.items()
    }
    blocking = [
        model_id
        for model_id in PRIMARY_MODEL_IDS
        if selections[model_id]["status"] != "SELECTED"
    ]
    ready = not blocking
    report = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": (
            "SELECTION_PASS_AWAITING_FRESH_TEST_AUTHORIZATION"
            if ready
            else "SELECTION_FAILED_PRIMARY_NO_FULL_RUN"
        ),
        "candidate_results": [
            dict(result) for result in candidate_results
        ],
        "model_selections": selections,
        "primary_model_ids": list(PRIMARY_MODEL_IDS),
        "primary_c2_selection_ready": ready,
        "blocking_primary_models": blocking,
        "test_split_read": False,
        "fresh_test_authorized": False,
        "tstr_authorized": False,
        "privacy_authorized": False,
        "five_seed_full_run_authorized": False,
        "requires_separate_fresh_test_authorization": ready,
        **dict(provenance),
    }
    if ready:
        selected = {
            model_id: {
                key: value
                for key, value in selections[model_id].items()
                if key
                in {
                    "selected_candidate_id",
                    "continuous_ks_max",
                    "continuous_ks_sum",
                    "candidate_definition_sha256",
                    "candidate_result_sha256",
                    "validation_sample_sha256",
                    "checkpoint_sha256",
                }
            }
            for model_id in PRIMARY_MODEL_IDS
        }
        manifest_status = (
            "FROZEN_AWAITING_SEPARATE_FRESH_TEST_AUTHORIZATION"
        )
    else:
        selected = {}
        manifest_status = "PRIMARY_SELECTION_FAILED_NO_FRESH_TEST"
    selection_manifest = {
        "schema_version": (
            "benchmark-v2.6-single-factor-selection-manifest-v1"
        ),
        "status": manifest_status,
        "selected_candidates": selected,
        "model_selection_status": {
            model_id: selections[model_id]["status"]
            for model_id in PRIMARY_MODEL_IDS
        },
        "blocking_primary_models": blocking,
        "primary_c2_selection_ready": ready,
        "test_split_read": False,
        "fresh_test_authorized": False,
        "tstr_authorized": False,
        "privacy_authorized": False,
        "five_seed_full_run_authorized": False,
        "requires_separate_fresh_test_authorization": ready,
        **dict(provenance),
    }
    return report, selection_manifest


def validate_single_factor_aggregate_authorization(
    *,
    repository_root: Path,
    authorization_path: Path,
    plan: SingleFactorExecutionPlan,
    source_commit: str,
    relevant_source_sha256: str,
) -> Mapping[str, Any]:
    runtime_root = (repository_root / RUNTIME_ROOT).resolve()
    authorization_path = authorization_path.resolve()
    if not _below(
        authorization_path,
        runtime_root / "authorization_history",
    ):
        raise SingleFactorAggregateContractError(
            "aggregate authorization is outside append-only history"
        )
    authorization = _read_json(
        authorization_path,
        role="single-factor aggregate authorization",
    )
    expected = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "AUTHORIZED",
        "source_commit": source_commit,
        "relevant_source_sha256": relevant_source_sha256,
        "single_factor_config_sha256": plan.config_sha256,
        "base_selection_config_sha256": (
            plan.base_selection_config_sha256
        ),
        "development_manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
        "v2_5_config_sha256": V2_5_CONFIG_SHA256,
        "v2_5_final_complete_sha256": V2_5_FINAL_COMPLETE_SHA256,
        "v2_5_frozen_manifest_sha256": (
            V2_5_FROZEN_MANIFEST_SHA256
        ),
        "prior_v2_6_selection_complete_sha256": (
            PRIOR_SELECTION_COMPLETE_SHA256
        ),
        "scope": EXPECTED_SCOPE,
    }
    mismatches = {
        key: {"expected": value, "actual": authorization.get(key)}
        for key, value in expected.items()
        if authorization.get(key) != value
    }
    if mismatches or not str(
        authorization.get("approval_text", "")
    ).strip():
        raise SingleFactorAggregateContractError(
            f"aggregate authorization mismatch: {mismatches}"
        )
    preserved_provenance = {
        "development manifest": (
            plan.development_manifest_path,
            DEVELOPMENT_MANIFEST_SHA256,
        ),
        "v2.5 config": (
            repository_root / "configs/benchmark_v2/full_v2_5.yaml",
            V2_5_CONFIG_SHA256,
        ),
        "v2.5 FINAL_COMPLETE": (
            repository_root
            / "artifacts/benchmark_v2_5/full/FINAL_COMPLETE.json",
            V2_5_FINAL_COMPLETE_SHA256,
        ),
        "v2.5 frozen manifest": (
            repository_root
            / "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
            "kappa_1.00/data_manifest.json",
            V2_5_FROZEN_MANIFEST_SHA256,
        ),
        "prior v2.6 selection terminal": (
            repository_root
            / "artifacts/benchmark_v2_6/selection/"
            "aggregate_attempt_001/AGGREGATE_COMPLETE.json",
            PRIOR_SELECTION_COMPLETE_SHA256,
        ),
    }
    for role, (path, expected_hash) in preserved_provenance.items():
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise SingleFactorAggregateContractError(
                f"{role} preservation hash mismatch"
            )
    if tuple(authorization.get("model_ids", ())) != PRIMARY_MODEL_IDS:
        raise SingleFactorAggregateContractError(
            "aggregate authorization model order changed"
        )
    worker_records = authorization.get("approved_worker_terminals")
    if not isinstance(worker_records, Mapping) or set(
        worker_records
    ) != set(PRIMARY_MODEL_IDS):
        raise SingleFactorAggregateContractError(
            "aggregate authorization worker scope is incomplete"
        )
    return authorization


def validate_single_factor_aggregate_readiness(
    *,
    repository_root: Path,
    runtime_root: Path,
    plan: SingleFactorExecutionPlan,
    authorization: Mapping[str, Any],
) -> Mapping[str, Any]:
    execution_authorization_hash = authorization.get(
        "candidate_execution_authorization_sha256"
    )
    candidate_source_commit = authorization.get(
        "candidate_source_commit"
    )
    candidate_source_hash = authorization.get(
        "candidate_relevant_source_sha256"
    )
    if (
        not isinstance(execution_authorization_hash, str)
        or len(execution_authorization_hash) != 64
        or not isinstance(candidate_source_commit, str)
        or len(candidate_source_commit) != 40
        or not isinstance(candidate_source_hash, str)
        or len(candidate_source_hash) != 64
    ):
        raise SingleFactorAggregateContractError(
            "candidate execution provenance is incomplete"
        )
    execution_authorization = _repository_path(
        repository_root,
        authorization.get("candidate_execution_authorization_path"),
        role="candidate execution authorization",
        root=runtime_root / "authorization_history",
    )
    if sha256_file(execution_authorization) != (
        execution_authorization_hash
    ):
        raise SingleFactorAggregateContractError(
            "candidate execution authorization hash mismatch"
        )
    approved = authorization["approved_worker_terminals"]
    verified_models: dict[str, Any] = {}
    for model_id in PRIMARY_MODEL_IDS:
        worker = approved[model_id]
        if not isinstance(worker, Mapping):
            raise SingleFactorAggregateContractError(
                f"malformed worker authorization: {model_id}"
            )
        expected_path = (
            runtime_root
            / "workers"
            / model_id
            / "attempt_001"
            / "WORKER_COMPLETE.json"
        ).resolve()
        terminal_path = _repository_path(
            repository_root,
            worker.get("path"),
            role=f"{model_id} worker terminal",
            root=runtime_root,
        )
        if terminal_path != expected_path:
            raise SingleFactorAggregateContractError(
                f"{model_id} approved worker attempt changed"
            )
        terminal_hash = sha256_file(terminal_path)
        if terminal_hash != worker.get("sha256"):
            raise SingleFactorAggregateContractError(
                f"{model_id} worker terminal hash mismatch"
            )
        tree_hash, tree_files, tree_bytes = (
            single_factor_model_tree_sha256(
                runtime_root=runtime_root,
                model_id=model_id,
            )
        )
        if (
            tree_hash != worker.get("tree_sha256")
            or tree_files != int(worker.get("tree_files", -1))
            or tree_bytes != int(worker.get("tree_bytes", -1))
        ):
            raise SingleFactorAggregateContractError(
                f"{model_id} worker tree preservation mismatch"
            )
        terminal = _read_json(
            terminal_path,
            role=f"{model_id} worker terminal",
        )
        if (
            terminal.get("status") != "COMPLETE"
            or terminal.get("model_id") != model_id
            or terminal.get("source_commit")
            != candidate_source_commit
            or terminal.get("relevant_source_sha256")
            != candidate_source_hash
            or terminal.get("single_factor_config_sha256")
            != plan.config_sha256
            or terminal.get("authorization_sha256")
            != execution_authorization_hash
            or terminal.get("selection_aggregate_created") is not False
            or terminal.get("test_split_read") is not False
        ):
            raise SingleFactorAggregateContractError(
                f"{model_id} worker provenance mismatch"
            )
        results = terminal.get("results")
        if not isinstance(results, list):
            raise SingleFactorAggregateContractError(
                f"{model_id} worker results are missing"
            )
        artifacts = verify_single_factor_worker_candidate_artifacts(
            repository_root=repository_root,
            runtime_root=runtime_root,
            plan=plan,
            model_id=model_id,
            results=results,
            source_commit=candidate_source_commit,
            source_hash=candidate_source_hash,
            authorization_hash=execution_authorization_hash,
        )
        verified_models[model_id] = {
            "worker_terminal_path": str(
                terminal_path.relative_to(repository_root)
            ),
            "worker_terminal_sha256": terminal_hash,
            "worker_tree_sha256": tree_hash,
            "worker_tree_files": tree_files,
            "worker_tree_bytes": tree_bytes,
            "candidate_artifacts": artifacts,
        }
    return {
        "schema_version": (
            "benchmark-v2.6-single-factor-aggregate-readiness-v1"
        ),
        "status": "PASS",
        "worker_count": 3,
        "candidate_count": 9,
        "worker_models": verified_models,
        "candidate_source_commit": candidate_source_commit,
        "candidate_relevant_source_sha256": candidate_source_hash,
        "candidate_execution_authorization_sha256": (
            execution_authorization_hash
        ),
        "test_split_read": False,
        "gpu_queries": 0,
        "cuda_calls": 0,
        "model_fit_calls": 0,
        "model_sample_calls": 0,
    }


def _evaluate_verified_candidates(
    *,
    repository_root: Path,
    runtime_root: Path,
    plan: SingleFactorExecutionPlan,
    readiness: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    context = load_development_context(
        repository_root=repository_root,
        manifest_path=plan.development_manifest_path,
    )
    if (
        context.manifest_sha256 != DEVELOPMENT_MANIFEST_SHA256
        or context.train_file_sha256
        != FROZEN_HASHES["train_file_sha256"]
        or context.train_content_sha256
        != FROZEN_HASHES["train_content_sha256"]
        or context.validation_file_sha256
        != FROZEN_HASHES["validation_file_sha256"]
        or context.validation_content_sha256
        != FROZEN_HASHES["validation_content_sha256"]
        or context.plan.plan_hash
        != FROZEN_HASHES["sampling_plan_sha256"]
    ):
        raise SingleFactorAggregateContractError(
            "frozen train/validation/SamplingPlan provenance mismatch"
        )
    by_operation = {
        (operation.model_id, operation.candidate_id): operation
        for operation in plan.operations
    }
    results: list[Mapping[str, Any]] = []
    verified_models = readiness["worker_models"]
    for model_id in PRIMARY_MODEL_IDS:
        artifacts = verified_models[model_id]["candidate_artifacts"]
        for candidate_id, record in artifacts.items():
            operation = by_operation[(model_id, candidate_id)]
            if operation.execution_kind == "reuse_frozen_control":
                reference_path = _repository_path(
                    repository_root,
                    record["reference_manifest_path"],
                    role="frozen-control reference",
                    root=runtime_root,
                )
                reference = _read_json(
                    reference_path,
                    role="frozen-control reference",
                )
                paths = reference.get("frozen_control_paths")
                hashes = reference.get("frozen_control_hashes")
                if not isinstance(paths, Mapping) or not isinstance(
                    hashes,
                    Mapping,
                ):
                    raise SingleFactorAggregateContractError(
                        "frozen-control provenance is incomplete"
                    )
                sample_path = _repository_path(
                    repository_root,
                    paths.get("validation_sample_path"),
                    role="frozen validation sample",
                )
                result_path = _repository_path(
                    repository_root,
                    paths.get("candidate_result_path"),
                    role="frozen candidate result",
                )
                checkpoint_path = _repository_path(
                    repository_root,
                    paths.get("checkpoint_path"),
                    role="frozen checkpoint",
                )
                expected_hashes = {
                    sample_path: hashes.get(
                        "validation_sample_sha256"
                    ),
                    result_path: hashes.get(
                        "candidate_result_sha256"
                    ),
                    checkpoint_path: hashes.get(
                        "checkpoint_sha256"
                    ),
                }
                if any(
                    sha256_file(path) != expected
                    for path, expected in expected_hashes.items()
                ):
                    raise SingleFactorAggregateContractError(
                        "frozen-control artifact hash mismatch"
                    )
                candidate_result_hash = sha256_file(result_path)
                checkpoint_hash = sha256_file(checkpoint_path)
            else:
                result_path = _repository_path(
                    repository_root,
                    record["candidate_result_path"],
                    role="single-factor candidate result",
                    root=runtime_root,
                )
                candidate_result = _read_json(
                    result_path,
                    role="single-factor candidate result",
                )
                if (
                    candidate_result.get(
                        "candidate_definition_sha256"
                    )
                    != operation.definition_sha256
                    or candidate_result.get("execution_kind")
                    != operation.execution_kind
                    or candidate_result.get(
                        "validation_selection_executed"
                    )
                    is not False
                    or candidate_result.get("test_split_read") is not False
                    or candidate_result.get("fresh_test_read") is not False
                ):
                    raise SingleFactorAggregateContractError(
                        f"candidate definition/provenance mismatch: "
                        f"{candidate_id}"
                    )
                sample_path = _repository_path(
                    repository_root,
                    record["validation_sample_path"],
                    role="single-factor validation sample",
                    root=runtime_root,
                )
                checkpoint_path = _repository_path(
                    repository_root,
                    candidate_result.get("checkpoint_path"),
                    role="single-factor checkpoint",
                )
                checkpoint_hash = sha256_file(checkpoint_path)
                if checkpoint_hash != candidate_result.get(
                    "checkpoint_sha256"
                ):
                    raise SingleFactorAggregateContractError(
                        f"candidate checkpoint hash mismatch: "
                        f"{candidate_id}"
                    )
                candidate_result_hash = sha256_file(result_path)
            sample = _load_synthetic(sample_path)
            sample_contract = validate_synthetic_contract(
                sample,
                plan=context.plan,
                train=context.train,
            )
            statistics = {
                key: float(value)
                for key, value in row_guard_statistics(
                    context.validation,
                    sample,
                    tau=context.tau,
                    receiver_categories=context.receiver_categories,
                ).items()
            }
            checks = {
                key: (
                    "PASS"
                    if np.isfinite(statistics[key])
                    and statistics[key]
                    <= float(FROZEN_THRESHOLDS[key])
                    else "FAIL"
                )
                for key in METRIC_KEYS
            }
            all_pass = all(value == "PASS" for value in checks.values())
            results.append(
                {
                    "model_id": model_id,
                    "candidate_id": candidate_id,
                    "status": "PASS" if all_pass else "FAIL",
                    "all_five_guards_pass": all_pass,
                    "statistics": statistics,
                    "thresholds": dict(FROZEN_THRESHOLDS),
                    "checks": checks,
                    "continuous_ks_max": max(
                        statistics[key]
                        for key in CONTINUOUS_KS_KEYS
                    ),
                    "continuous_ks_sum": sum(
                        statistics[key]
                        for key in CONTINUOUS_KS_KEYS
                    ),
                    "sample_contract": sample_contract,
                    "candidate_definition_sha256": (
                        operation.definition_sha256
                    ),
                    "candidate_result_sha256": (
                        candidate_result_hash
                    ),
                    "validation_sample_sha256": sha256_file(
                        sample_path
                    ),
                    "checkpoint_sha256": checkpoint_hash,
                    "execution_kind": operation.execution_kind,
                }
            )
    provenance = {
        "single_factor_config_sha256": plan.config_sha256,
        "base_selection_config_sha256": (
            plan.base_selection_config_sha256
        ),
        "development_manifest_sha256": context.manifest_sha256,
        "train_file_sha256": context.train_file_sha256,
        "train_content_sha256": context.train_content_sha256,
        "validation_file_sha256": context.validation_file_sha256,
        "validation_content_sha256": context.validation_content_sha256,
        "selection_plan_sha256": context.plan.plan_hash,
        "thresholds": dict(FROZEN_THRESHOLDS),
        "selection_rule": {
            "eligibility": "all_five_row_marginal_guards_PASS",
            "primary_objective": "minimize_max_amount_ks_gap_ks",
            "secondary_objective": "minimize_sum_amount_ks_gap_ks",
            "final_tiebreaker": "lexicographic_candidate_id",
        },
        "test_split_read": False,
    }
    return results, provenance


def write_single_factor_aggregate_bundle(
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
        raise SingleFactorAggregateContractError(
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
                "benchmark-v2.6-single-factor-checksum-manifest-v1"
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
                "benchmark-v2.6-single-factor-artifact-index-v1"
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
        "schema_version": (
            "benchmark-v2.6-single-factor-aggregate-terminal-v1"
        ),
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
        "finished_at_utc": _utc_now(),
    }
    _exclusive_json(output_root / "AGGREGATE_COMPLETE.json", terminal)
    return terminal


def execute_single_factor_validation_aggregate(
    *,
    repository_root: Path,
    config_path: Path,
    base_selection_config_path: Path,
    authorization_path: Path,
    output_root: Path,
    source_commit: str,
    mode: str,
) -> Mapping[str, Any]:
    if mode not in {"plan", "dry-run", "execute"}:
        raise SingleFactorAggregateContractError(
            f"unknown aggregate mode: {mode}"
        )
    repository_root = repository_root.resolve()
    runtime_root = (repository_root / RUNTIME_ROOT).resolve()
    output_root = output_root.resolve()
    if (
        output_root.parent != runtime_root
        or not output_root.name.startswith("aggregate_attempt_")
    ):
        raise SingleFactorAggregateContractError(
            "aggregate output is outside the single-factor root"
        )
    if _current_head(repository_root) != source_commit:
        raise SingleFactorAggregateContractError(
            "aggregate source commit is not current HEAD"
        )
    plan = build_single_factor_execution_plan(
        config_path=config_path,
        base_selection_config_path=base_selection_config_path,
    )
    source_hash = single_factor_relevant_source_sha256(repository_root)
    authorization = validate_single_factor_aggregate_authorization(
        repository_root=repository_root,
        authorization_path=authorization_path,
        plan=plan,
        source_commit=source_commit,
        relevant_source_sha256=source_hash,
    )
    readiness = validate_single_factor_aggregate_readiness(
        repository_root=repository_root,
        runtime_root=runtime_root,
        plan=plan,
        authorization=authorization,
    )
    if mode == "plan":
        return {
            "mode": "plan",
            "readiness": readiness,
            "report": None,
            "selection_manifest": None,
            "terminal": None,
            "artifacts_written": False,
        }
    results, provenance = _evaluate_verified_candidates(
        repository_root=repository_root,
        runtime_root=runtime_root,
        plan=plan,
        readiness=readiness,
    )
    report, selection_manifest = build_single_factor_selection(
        candidate_results=results,
        provenance={
            **provenance,
            "aggregate_source_commit": source_commit,
            "aggregate_relevant_source_sha256": source_hash,
            "candidate_source_commit": readiness[
                "candidate_source_commit"
            ],
            "candidate_relevant_source_sha256": readiness[
                "candidate_relevant_source_sha256"
            ],
            "authorization_sha256": sha256_file(
                authorization_path
            ),
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
    terminal = write_single_factor_aggregate_bundle(
        output_root=output_root,
        authorization_path=authorization_path.relative_to(
            repository_root
        ),
        authorization_sha256=sha256_file(authorization_path),
        readiness=readiness,
        report=report,
        selection_manifest=selection_manifest,
        source_commit=source_commit,
        relevant_source_sha256=source_hash,
    )
    return {
        "mode": "execute",
        "readiness": readiness,
        "report": report,
        "selection_manifest": selection_manifest,
        "terminal": terminal,
        "artifacts_written": True,
    }
