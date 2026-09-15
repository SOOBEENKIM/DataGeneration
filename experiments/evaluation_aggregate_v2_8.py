from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml

from eval.candidate_preparation_v2_8 import sha256_file


class EvaluationAggregateV28ContractError(RuntimeError):
    pass


FORBIDDEN_CALL_COUNTS = {
    "gpu_query": 0,
    "cuda": 0,
    "training": 0,
    "sampling": 0,
    "guard_recalculation": 0,
    "candidate_reexecution": 0,
    "test_split_access": 0,
}
SOURCE_ONLY_FALSE_FLAGS = (
    "authorization_creation_authorized",
    "execution_authorized",
    "gpu_query_authorized",
    "cuda_authorized",
    "training_authorized",
    "sampling_authorized",
    "guard_recalculation_authorized",
    "candidate_reexecution_authorized",
    "fresh_test_authorized",
    "tstr_authorized",
    "privacy_authorized",
    "five_seed_full_run_authorized",
)
AGGREGATE_SCOPE = {
    "aggregate_only": True,
    "stored_evaluation_only": True,
    "gpu_query": False,
    "cuda": False,
    "training": False,
    "sampling": False,
    "guard_recalculation": False,
    "candidate_reexecution": False,
    "test_split_access": False,
    "fresh_test": False,
    "tstr": False,
    "privacy": False,
    "five_seed_full_run": False,
}
AGGREGATE_RELEVANT_SOURCE_PATHS = (
    "configs/benchmark_v2/evaluation_aggregate_v2_8.yaml",
    "configs/benchmark_v2/evaluation_only_v2_8.yaml",
    "configs/benchmark_v2/selection_v2_8_source_amendment.yaml",
    "eval/candidate_preparation_v2_8.py",
    "experiments/evaluation_aggregate_v2_8.py",
    "scripts/aggregate_evaluation_selection_v2_8.py",
)


@dataclass(frozen=True)
class AggregateCandidate:
    model_id: str
    candidate_id: str
    source_kind: str
    attempt_path: str
    worker_terminal_path: str | None = None


@dataclass(frozen=True)
class AggregatePlan:
    config_path: Path
    config_sha256: str
    runtime_root: str
    future_output_root: str
    candidates: tuple[AggregateCandidate, ...]
    inputs: Mapping[str, Mapping[str, Any]]
    frozen_provenance: Mapping[str, Any]
    selection_rule: Mapping[str, Any]
    forbidden_call_counts: Mapping[str, int]


def _read_config(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EvaluationAggregateV28ContractError(
            f"cannot read v2.8 aggregate config: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise EvaluationAggregateV28ContractError(
            "v2.8 aggregate config is not a mapping"
        )
    return value


def build_v28_aggregate_plan(config_path: Path) -> AggregatePlan:
    config_path = config_path.resolve()
    raw = _read_config(config_path)
    if (
        raw.get("schema_version")
        != "benchmark-v2.8-aggregate-source-only-v1"
        or raw.get("mode") != "SOURCE_ONLY_IMPLEMENTATION"
        or any(raw.get(flag) is not False for flag in SOURCE_ONLY_FALSE_FLAGS)
        or raw.get("test_split_access") != "FORBIDDEN"
    ):
        raise EvaluationAggregateV28ContractError(
            "v2.8 aggregate source-only scope mismatch"
        )
    raw_inputs = raw.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise EvaluationAggregateV28ContractError(
            "v2.8 aggregate inputs are missing"
        )
    order = (
        "ctgan_separate_class",
        "cof_seqgen",
        "tvae_separate_class",
    )
    candidates = tuple(
        AggregateCandidate(
            model_id=model_id,
            candidate_id=str(raw_inputs[model_id]["candidate_id"]),
            source_kind=str(raw_inputs[model_id]["source_kind"]),
            attempt_path=str(raw_inputs[model_id]["attempt_path"]),
            worker_terminal_path=raw_inputs[model_id].get(
                "worker_terminal_path"
            ),
        )
        for model_id in order
    )
    return AggregatePlan(
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        runtime_root=str(raw["runtime_root"]),
        future_output_root=str(raw["future_output_root"]),
        candidates=candidates,
        inputs={
            str(key): dict(value)
            for key, value in raw_inputs.items()
        },
        frozen_provenance=dict(raw["frozen_provenance"]),
        selection_rule=dict(raw["selection_rule"]),
        forbidden_call_counts=dict(FORBIDDEN_CALL_COUNTS),
    )


def _read_json(path: Path, *, role: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationAggregateV28ContractError(
            f"cannot read {role}: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise EvaluationAggregateV28ContractError(
            f"{role} is not a JSON object: {path}"
        )
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def inspect_v28_terminal_inventory(
    *,
    repository_root: Path,
    plan: AggregatePlan,
) -> Mapping[str, Any]:
    root = repository_root.resolve()
    runtime = (root / plan.runtime_root).resolve()
    expected_runtime = (
        root / "artifacts/benchmark_v2_8/candidate_selection"
    ).resolve()
    if runtime != expected_runtime:
        raise EvaluationAggregateV28ContractError(
            "aggregate input is outside the v2.8 runtime root"
        )
    actual_worker_terminals = sorted(
        tuple(runtime.glob("workers/*/WORKER_COMPLETE.json"))
        + tuple(runtime.glob("workers/*/WORKER_FAILED.json"))
    )
    actual_candidate_terminals = sorted(
        runtime.glob("evaluations/*/*/seed_2801/attempt_*/COMPLETE.json")
    )
    if len(actual_worker_terminals) != 2:
        raise EvaluationAggregateV28ContractError(
            "v2.8 aggregate requires exactly two worker terminals"
        )
    if len(actual_candidate_terminals) != 2:
        raise EvaluationAggregateV28ContractError(
            "v2.8 aggregate requires exactly two candidate terminals"
        )

    records: dict[str, Any] = {}
    common_authorizations: set[str] = set()
    common_authorization_hashes: set[str] = set()
    common_source_commits: set[str] = set()
    common_source_hashes: set[str] = set()
    for candidate in plan.candidates:
        if candidate.source_kind != "stored_v2_8_evaluation":
            continue
        spec = plan.inputs[candidate.model_id]
        attempt = (root / candidate.attempt_path).resolve()
        worker_path = (root / str(candidate.worker_terminal_path)).resolve()
        if (
            attempt.parent.name != "seed_2801"
            or attempt.name != "attempt_001"
            or worker_path not in actual_worker_terminals
        ):
            raise EvaluationAggregateV28ContractError(
                f"unexpected terminal path: {candidate.candidate_id}"
            )
        required = {
            name: attempt / name
            for name in (
                "manifest.json",
                "COMPLETE.json",
                "candidate_result.json",
                "evaluation.json",
                "fit_state.json",
                "validation_sample.npz",
            )
        }
        if any(not path.is_file() for path in required.values()):
            raise EvaluationAggregateV28ContractError(
                f"incomplete candidate artifact: {candidate.candidate_id}"
            )
        if (attempt / "FAILED.json").exists():
            raise EvaluationAggregateV28ContractError(
                f"candidate has conflicting terminal: {candidate.candidate_id}"
            )
        manifest = _read_json(
            required["manifest.json"],
            role=f"{candidate.candidate_id} manifest",
        )
        complete = _read_json(
            required["COMPLETE.json"],
            role=f"{candidate.candidate_id} terminal",
        )
        result = _read_json(
            required["candidate_result.json"],
            role=f"{candidate.candidate_id} result",
        )
        evaluation = _read_json(
            required["evaluation.json"],
            role=f"{candidate.candidate_id} evaluation",
        )
        worker = _read_json(
            worker_path,
            role=f"{candidate.model_id} worker terminal",
        )
        ownership_path = worker_path.parent / "ownership.lock"
        ownership = _read_json(
            ownership_path,
            role=f"{candidate.model_id} worker ownership",
        )
        expected_hashes = {
            "candidate_result_sha256": sha256_file(
                required["candidate_result.json"]
            ),
            "evaluation_sha256": sha256_file(
                required["evaluation.json"]
            ),
            "fit_state_sha256": sha256_file(required["fit_state.json"]),
            "validation_sample_sha256": sha256_file(
                required["validation_sample.npz"]
            ),
        }
        if (
            complete.get("status") != "COMPLETE"
            or any(
                complete.get(key) != value
                for key, value in expected_hashes.items()
            )
            or manifest.get("model_id") != candidate.model_id
            or manifest.get("candidate_id") != candidate.candidate_id
            or manifest.get("runner_config_sha256")
            != plan.frozen_provenance["runner_config_sha256"]
            or manifest.get("candidate_config_sha256")
            != plan.frozen_provenance["candidate_config_sha256"]
            or manifest.get("sampling_plan_sha256")
            != plan.frozen_provenance["sampling_plan_sha256"]
            or manifest.get("test_split_read") is not False
            or result.get("all_five_guards_pass")
            != evaluation.get("all_five_guards_pass")
            or result.get("guard_status") != evaluation.get("status")
            or result.get("test_split_read") is not False
            or evaluation.get("test_split_read") is not False
            or evaluation.get("validation_selection_executed") is not False
            or worker.get("status") != "COMPLETE"
            or worker.get("candidate_id") != candidate.candidate_id
            or worker.get("attempt")
            != attempt.relative_to(root).as_posix()
            or worker.get("test_split_read") is not False
            or ownership.get("provenance_sha256")
            != worker.get("provenance_sha256")
        ):
            raise EvaluationAggregateV28ContractError(
                f"stored artifact contract mismatch: {candidate.candidate_id}"
            )
        common_authorizations.add(str(manifest.get("authorization_path")))
        common_authorization_hashes.add(
            str(manifest.get("authorization_sha256"))
        )
        common_source_commits.add(str(manifest.get("source_commit")))
        common_source_hashes.add(
            str(manifest.get("relevant_source_sha256"))
        )
        records[candidate.candidate_id] = {
            "model_id": candidate.model_id,
            "attempt_path": attempt.relative_to(root).as_posix(),
            "worker_terminal_path": worker_path.relative_to(root).as_posix(),
            "worker_terminal_sha256": sha256_file(worker_path),
            "manifest_sha256": sha256_file(required["manifest.json"]),
            "complete_sha256": sha256_file(required["COMPLETE.json"]),
            **expected_hashes,
        }

    expected_authorization_hash = plan.frozen_provenance[
        "execution_authorization_sha256"
    ]
    if (
        len(common_authorizations) != 1
        or len(common_authorization_hashes) != 1
        or next(iter(common_authorization_hashes))
        != expected_authorization_hash
        or len(common_source_commits) != 1
        or len(common_source_hashes) != 1
    ):
        raise EvaluationAggregateV28ContractError(
            "v2.8 candidate provenance is not a single frozen family"
        )
    execution_authorization_path = (
        root / next(iter(common_authorizations))
    )
    if (
        not execution_authorization_path.is_file()
        or sha256_file(execution_authorization_path)
        != expected_authorization_hash
    ):
        raise EvaluationAggregateV28ContractError(
            "v2.8 execution authorization hash mismatch"
        )

    frozen_candidate = next(
        candidate
        for candidate in plan.candidates
        if candidate.source_kind == "frozen_v2_7_selected_reference"
    )
    frozen_spec = plan.inputs[frozen_candidate.model_id]
    frozen_attempt = (root / frozen_candidate.attempt_path).resolve()
    frozen_paths = {
        "complete_sha256": frozen_attempt / "COMPLETE.json",
        "candidate_result_sha256": frozen_attempt / "candidate_result.json",
        "evaluation_sha256": frozen_attempt / "evaluation.json",
        "validation_sample_sha256": frozen_attempt / "validation_sample.npz",
    }
    if any(
        not path.is_file()
        or sha256_file(path) != frozen_spec[key]
        for key, path in frozen_paths.items()
    ):
        raise EvaluationAggregateV28ContractError(
            "frozen TVAE reference hash mismatch"
        )
    frozen_evaluation = _read_json(
        frozen_paths["evaluation_sha256"],
        role="frozen TVAE evaluation",
    )
    frozen_result = _read_json(
        frozen_paths["candidate_result_sha256"],
        role="frozen TVAE candidate result",
    )
    prior_root = (
        root
        / "artifacts/benchmark_v2_7/candidate_selection/"
        "aggregate_attempt_001"
    )
    prior_terminal_path = prior_root / "AGGREGATE_COMPLETE.json"
    prior_report_path = prior_root / "selection_report.json"
    if (
        sha256_file(prior_terminal_path)
        != plan.frozen_provenance["v2_7_aggregate_complete_sha256"]
        or sha256_file(prior_report_path)
        != plan.frozen_provenance["v2_7_selection_report_sha256"]
    ):
        raise EvaluationAggregateV28ContractError(
            "frozen v2.7 aggregate evidence hash mismatch"
        )
    prior_report = _read_json(
        prior_report_path,
        role="frozen v2.7 selection report",
    )
    prior_selection = prior_report.get("model_selections", {}).get(
        "tvae_separate_class", {}
    )
    if (
        frozen_evaluation.get("all_five_guards_pass") is not True
        or frozen_result.get("all_five_guards_pass") is not True
        or prior_selection.get("status") != "SELECTED"
        or prior_selection.get("selected_candidate_id")
        != frozen_candidate.candidate_id
    ):
        raise EvaluationAggregateV28ContractError(
            "frozen TVAE reference is not the selected all-pass evidence"
        )
    frozen_record = {
        "model_id": frozen_candidate.model_id,
        "candidate_id": frozen_candidate.candidate_id,
        "attempt_path": frozen_attempt.relative_to(root).as_posix(),
        **{key: frozen_spec[key] for key in frozen_paths},
        "v2_7_selection_report_sha256": sha256_file(prior_report_path),
    }
    inventory = {
        "schema_version": "benchmark-v2.8-aggregate-input-inventory-v1",
        "worker_terminal_count": len(actual_worker_terminals),
        "candidate_terminal_count": len(actual_candidate_terminals),
        "stored_evaluation_count": len(records),
        "frozen_reference_count": 1,
        "candidate_artifacts": records,
        "frozen_reference": frozen_record,
        "candidate_execution_authorization_path": (
            execution_authorization_path.relative_to(root).as_posix()
        ),
        "candidate_execution_authorization_sha256": (
            expected_authorization_hash
        ),
        "candidate_source_commit": next(iter(common_source_commits)),
        "candidate_relevant_source_sha256": next(
            iter(common_source_hashes)
        ),
        "runner_config_sha256": plan.frozen_provenance[
            "runner_config_sha256"
        ],
        "candidate_config_sha256": plan.frozen_provenance[
            "candidate_config_sha256"
        ],
        "sampling_plan_sha256": plan.frozen_provenance[
            "sampling_plan_sha256"
        ],
        "test_split_read": False,
    }
    return {
        **inventory,
        "input_inventory_sha256": _canonical_sha256(inventory),
    }


def _stored_guard_row(
    *,
    evaluation: Mapping[str, Any],
    candidate: AggregateCandidate,
    source_kind: str,
    evaluation_sha256: str,
    candidate_result_sha256: str,
    validation_sample_sha256: str,
    checkpoint_sha256: str,
    thresholds: Mapping[str, Any],
) -> Mapping[str, Any]:
    checks = evaluation.get("checks")
    expected_keys = set(thresholds)
    if (
        not isinstance(checks, Mapping)
        or set(checks) != expected_keys
        or any(status not in {"PASS", "FAIL"} for status in checks.values())
        or evaluation.get("thresholds") != thresholds
        or evaluation.get("fit_split") != "train"
        or evaluation.get("evaluation_split") != "validation"
        or evaluation.get("test_split_read") is not False
    ):
        raise EvaluationAggregateV28ContractError(
            f"stored guard evidence contract mismatch: "
            f"{candidate.candidate_id}"
        )
    all_pass = all(status == "PASS" for status in checks.values())
    if (
        evaluation.get("all_five_guards_pass") is not all_pass
        or evaluation.get("status")
        != ("PASS" if all_pass else "FAIL")
    ):
        raise EvaluationAggregateV28ContractError(
            f"stored guard decision is internally inconsistent: "
            f"{candidate.candidate_id}"
        )
    statistics = evaluation.get("statistics")
    if not isinstance(statistics, Mapping) or set(statistics) != expected_keys:
        raise EvaluationAggregateV28ContractError(
            f"stored guard statistics are incomplete: "
            f"{candidate.candidate_id}"
        )
    amount_ks = float(statistics["amount_ks"])
    gap_ks = float(statistics["gap_ks"])
    return {
        "model_id": candidate.model_id,
        "candidate_id": candidate.candidate_id,
        "source_kind": source_kind,
        "status": evaluation["status"],
        "all_five_guards_pass": all_pass,
        "checks": dict(checks),
        "statistics": dict(statistics),
        "thresholds": dict(thresholds),
        "continuous_ks_max": max(amount_ks, gap_ks),
        "continuous_ks_sum": amount_ks + gap_ks,
        "evaluation_sha256": evaluation_sha256,
        "candidate_result_sha256": candidate_result_sha256,
        "validation_sample_sha256": validation_sample_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }


def load_v28_stored_guard_results(
    *,
    repository_root: Path,
    plan: AggregatePlan,
    inventory: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    root = repository_root.resolve()
    if (
        inventory.get("candidate_terminal_count") != 2
        or inventory.get("worker_terminal_count") != 2
        or inventory.get("frozen_reference_count") != 1
        or inventory.get("test_split_read") is not False
    ):
        raise EvaluationAggregateV28ContractError(
            "v2.8 stored guard inventory is incomplete"
        )
    thresholds = plan.selection_rule.get("thresholds")
    if not isinstance(thresholds, Mapping) or len(thresholds) != 5:
        raise EvaluationAggregateV28ContractError(
            "v2.8 frozen five-guard threshold family is invalid"
        )
    results: list[Mapping[str, Any]] = []
    for candidate in plan.candidates:
        if candidate.source_kind == "stored_v2_8_evaluation":
            record = inventory["candidate_artifacts"].get(
                candidate.candidate_id
            )
            if not isinstance(record, Mapping):
                raise EvaluationAggregateV28ContractError(
                    f"candidate inventory missing: {candidate.candidate_id}"
                )
            attempt = root / str(record["attempt_path"])
            evaluation_path = attempt / "evaluation.json"
            result_path = attempt / "candidate_result.json"
            result = _read_json(
                result_path,
                role=f"{candidate.candidate_id} stored result",
            )
            evaluation = _read_json(
                evaluation_path,
                role=f"{candidate.candidate_id} stored evaluation",
            )
            if (
                sha256_file(evaluation_path)
                != record["evaluation_sha256"]
                or sha256_file(result_path)
                != record["candidate_result_sha256"]
                or result.get("all_five_guards_pass")
                != evaluation.get("all_five_guards_pass")
                or result.get("guard_status") != evaluation.get("status")
            ):
                raise EvaluationAggregateV28ContractError(
                    f"candidate result hash/status mismatch: "
                    f"{candidate.candidate_id}"
                )
            manifest = _read_json(
                attempt / "manifest.json",
                role=f"{candidate.candidate_id} manifest",
            )
            results.append(
                _stored_guard_row(
                    evaluation=evaluation,
                    candidate=candidate,
                    source_kind=candidate.source_kind,
                    evaluation_sha256=record["evaluation_sha256"],
                    candidate_result_sha256=record[
                        "candidate_result_sha256"
                    ],
                    validation_sample_sha256=record[
                        "validation_sample_sha256"
                    ],
                    checkpoint_sha256=str(
                        manifest["checkpoint_sha256"]
                    ),
                    thresholds=thresholds,
                )
            )
            continue
        record = inventory["frozen_reference"]
        attempt = root / str(record["attempt_path"])
        evaluation = _read_json(
            attempt / "evaluation.json",
            role="frozen TVAE stored evaluation",
        )
        result = _read_json(
            attempt / "candidate_result.json",
            role="frozen TVAE stored result",
        )
        manifest = _read_json(
            attempt / "manifest.json",
            role="frozen TVAE manifest",
        )
        if (
            result.get("all_five_guards_pass")
            != evaluation.get("all_five_guards_pass")
            or result.get("guard_status") != evaluation.get("status")
        ):
            raise EvaluationAggregateV28ContractError(
                "frozen TVAE stored decision mismatch"
            )
        results.append(
            _stored_guard_row(
                evaluation=evaluation,
                candidate=candidate,
                source_kind=candidate.source_kind,
                evaluation_sha256=record["evaluation_sha256"],
                candidate_result_sha256=record[
                    "candidate_result_sha256"
                ],
                validation_sample_sha256=record[
                    "validation_sample_sha256"
                ],
                checkpoint_sha256=str(manifest["checkpoint_sha256"]),
                thresholds=thresholds,
            )
        )
    provenance = {
        "aggregate_config_sha256": plan.config_sha256,
        "input_inventory_sha256": inventory["input_inventory_sha256"],
        "runner_config_sha256": inventory["runner_config_sha256"],
        "candidate_config_sha256": inventory[
            "candidate_config_sha256"
        ],
        "candidate_execution_authorization_sha256": inventory[
            "candidate_execution_authorization_sha256"
        ],
        "sampling_plan_sha256": inventory["sampling_plan_sha256"],
        "selection_rule": dict(plan.selection_rule),
        "stored_evaluations_read": 3,
        "validation_samples_read": 0,
        "guard_recalculations_executed": 0,
        "test_split_read": False,
    }
    return results, provenance


def build_v28_selection(
    *,
    candidate_results: list[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    model_ids = (
        "ctgan_separate_class",
        "tvae_separate_class",
        "cof_seqgen",
    )
    if (
        len(candidate_results) != 3
        or provenance.get("test_split_read") is not False
        or provenance.get("guard_recalculations_executed") != 0
    ):
        raise EvaluationAggregateV28ContractError(
            "v2.8 selection requires three stored-only primary results"
        )
    selections: dict[str, Any] = {}
    for model_id in model_ids:
        eligible = [
            row
            for row in candidate_results
            if row.get("model_id") == model_id
            and row.get("all_five_guards_pass") is True
        ]
        eligible.sort(
            key=lambda row: (
                float(row["continuous_ks_max"]),
                float(row["continuous_ks_sum"]),
                str(row["candidate_id"]),
            )
        )
        selections[model_id] = (
            {
                "status": "SELECTED",
                "selected_candidate_id": eligible[0]["candidate_id"],
            }
            if eligible
            else {
                "status": "NO_PASSING_CANDIDATE",
                "selected_candidate_id": None,
            }
        )
    blocking = [
        model_id
        for model_id in model_ids
        if selections[model_id]["status"] != "SELECTED"
    ]
    ready = not blocking
    report = {
        "schema_version": "benchmark-v2.8-validation-selection-v1",
        "status": "COMPLETE",
        "candidate_results": [dict(row) for row in candidate_results],
        "model_selections": selections,
        "blocking_primary_models": blocking,
        "primary_c2_selection_ready": ready,
        "provenance": dict(provenance),
        "fresh_test_authorized": False,
        "tstr_authorized": False,
        "privacy_authorized": False,
        "five_seed_full_run_authorized": False,
    }
    manifest = {
        "schema_version": (
            "benchmark-v2.8-validation-selection-manifest-v1"
        ),
        "status": (
            "FROZEN_AWAITING_SEPARATE_FRESH_TEST_AUTHORIZATION"
            if ready
            else "FROZEN_NO_PASSING_PRIMARY_CANDIDATE"
        ),
        "selected_candidates": {
            model_id: value["selected_candidate_id"]
            for model_id, value in selections.items()
            if value["status"] == "SELECTED"
        },
        "model_selection_status": {
            model_id: value["status"]
            for model_id, value in selections.items()
        },
        "blocking_primary_models": blocking,
        "primary_c2_selection_ready": ready,
        "fresh_test_authorized": False,
        "test_split_read": False,
    }
    return report, manifest


def expected_v28_aggregate_authorization(
    *,
    plan: AggregatePlan,
    inventory: Mapping[str, Any],
    source_commit: str,
    relevant_source_sha256: str,
) -> Mapping[str, Any]:
    return {
        "schema_version": (
            "benchmark-v2.8-aggregate-only-authorization-v1"
        ),
        "status": "AUTHORIZED",
        "approval_text": (
            "Authorize exactly one v2.8 stored-evidence aggregate-only "
            "selection; all GPU, sampling, guard recalculation, test, "
            "and downstream experiment actions remain forbidden."
        ),
        "source_commit": source_commit,
        "relevant_source_sha256": relevant_source_sha256,
        "aggregate_config_sha256": plan.config_sha256,
        "input_inventory_sha256": inventory[
            "input_inventory_sha256"
        ],
        "candidate_execution_authorization_sha256": inventory[
            "candidate_execution_authorization_sha256"
        ],
        "runner_config_sha256": inventory["runner_config_sha256"],
        "candidate_config_sha256": inventory[
            "candidate_config_sha256"
        ],
        "sampling_plan_sha256": inventory["sampling_plan_sha256"],
        "candidate_terminal_count": 2,
        "worker_terminal_count": 2,
        "frozen_reference_count": 1,
        "scope": dict(AGGREGATE_SCOPE),
    }


def validate_v28_aggregate_authorization(
    *,
    plan: AggregatePlan,
    inventory: Mapping[str, Any],
    authorization: Mapping[str, Any],
    source_commit: str,
    relevant_source_sha256: str,
) -> None:
    expected = expected_v28_aggregate_authorization(
        plan=plan,
        inventory=inventory,
        source_commit=source_commit,
        relevant_source_sha256=relevant_source_sha256,
    )
    mismatches = {
        key: {
            "expected": value,
            "actual": authorization.get(key),
        }
        for key, value in expected.items()
        if authorization.get(key) != value
    }
    if mismatches or set(authorization) != set(expected):
        raise EvaluationAggregateV28ContractError(
            f"aggregate authorization mismatch: {mismatches}"
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
        raise EvaluationAggregateV28ContractError(
            f"append-only aggregate artifact exists: {path}"
        ) from error


def write_v28_aggregate_bundle(
    *,
    output_root: Path,
    authorization_path: Path,
    authorization_sha256: str,
    input_inventory: Mapping[str, Any],
    selection_report: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    output_root = output_root.resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise EvaluationAggregateV28ContractError(
            f"append-only aggregate output exists: {output_root}"
        ) from error
    report_path = output_root / "selection_report.json"
    manifest_path = output_root / "selection_manifest.json"
    _exclusive_json(report_path, selection_report)
    _exclusive_json(manifest_path, selection_manifest)
    checksum_path = output_root / "checksum_manifest.json"
    checksum = {
        "schema_version": "benchmark-v2.8-aggregate-checksums-v1",
        "selection_report.json": sha256_file(report_path),
        "selection_manifest.json": sha256_file(manifest_path),
    }
    _exclusive_json(checksum_path, checksum)
    index_path = output_root / "artifact_index.json"
    index = {
        "schema_version": "benchmark-v2.8-aggregate-index-v1",
        "authorization_path": str(authorization_path),
        "authorization_sha256": authorization_sha256,
        "input_inventory_sha256": input_inventory[
            "input_inventory_sha256"
        ],
        "artifacts": {
            "selection_report.json": sha256_file(report_path),
            "selection_manifest.json": sha256_file(manifest_path),
            "checksum_manifest.json": sha256_file(checksum_path),
        },
        "gpu_queries": 0,
        "cuda_calls": 0,
        "training_calls": 0,
        "sampling_calls": 0,
        "guard_recalculations": 0,
        "candidate_reexecution_calls": 0,
        "test_split_read": False,
    }
    _exclusive_json(index_path, index)
    terminal_path = output_root / "AGGREGATE_COMPLETE.json"
    terminal = {
        "schema_version": "benchmark-v2.8-aggregate-terminal-v1",
        "status": "COMPLETE",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_sha256": authorization_sha256,
        "input_inventory_sha256": input_inventory[
            "input_inventory_sha256"
        ],
        "selection_report_sha256": sha256_file(report_path),
        "selection_manifest_sha256": sha256_file(manifest_path),
        "checksum_manifest_sha256": sha256_file(checksum_path),
        "artifact_index_sha256": sha256_file(index_path),
        "primary_c2_selection_ready": bool(
            selection_report["primary_c2_selection_ready"]
        ),
        "blocking_primary_models": list(
            selection_report["blocking_primary_models"]
        ),
        "gpu_queries": 0,
        "cuda_calls": 0,
        "training_calls": 0,
        "sampling_calls": 0,
        "guard_recalculations": 0,
        "candidate_reexecution_calls": 0,
        "fresh_test_authorized": False,
        "tstr_authorized": False,
        "privacy_authorized": False,
        "five_seed_full_run_authorized": False,
        "test_split_read": False,
    }
    _exclusive_json(terminal_path, terminal)
    return terminal


def v28_aggregate_relevant_source_sha256(
    repository_root: Path,
) -> str:
    root = repository_root.resolve()
    records = []
    for relative in AGGREGATE_RELEVANT_SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise EvaluationAggregateV28ContractError(
                f"aggregate source path missing: {relative}"
            )
        records.append(
            {"path": relative, "sha256": sha256_file(path)}
        )
    return _canonical_sha256(records)


def _current_head(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def execute_v28_validation_aggregate(
    *,
    repository_root: Path,
    config_path: Path,
    authorization_path: Path | None,
    output_root: Path,
    source_commit: str,
    mode: str,
) -> Mapping[str, Any]:
    if mode not in {"plan", "dry-run", "execute"}:
        raise EvaluationAggregateV28ContractError(
            f"unknown aggregate mode: {mode}"
        )
    root = repository_root.resolve()
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    plan = build_v28_aggregate_plan(config_path)
    if _current_head(root) != source_commit:
        raise EvaluationAggregateV28ContractError(
            "aggregate source commit differs from HEAD"
        )
    relevant_hash = v28_aggregate_relevant_source_sha256(root)
    base = {
        "schema_version": "benchmark-v2.8-aggregate-plan-v1",
        "source_commit": source_commit,
        "relevant_source_sha256": relevant_hash,
        "aggregate_config_sha256": plan.config_sha256,
        "candidate_terminal_plan": 2,
        "worker_terminal_plan": 2,
        "frozen_reference_plan": 1,
        "candidate_ids": [
            candidate.candidate_id for candidate in plan.candidates
        ],
        "authorization_created": False,
        "runtime_artifact_created": False,
        "selection_executed": False,
        "execution_counts": dict(FORBIDDEN_CALL_COUNTS),
    }
    if mode == "plan":
        return {**base, "mode": "PLAN"}

    evaluation_runner_path = (
        root / "configs/benchmark_v2/evaluation_only_v2_8.yaml"
    )
    if (
        sha256_file(evaluation_runner_path)
        != plan.frozen_provenance["runner_config_sha256"]
        or sha256_file(
            root
            / "configs/benchmark_v2/"
            "selection_v2_8_source_amendment.yaml"
        )
        != plan.frozen_provenance["candidate_config_sha256"]
    ):
        raise EvaluationAggregateV28ContractError(
            "v2.8 frozen source config hash mismatch"
        )
    from experiments.evaluation_only_runner_v2_8 import (
        build_evaluation_execution_plan,
        dry_run_evaluation,
    )

    frozen_dry_run = dry_run_evaluation(
        build_evaluation_execution_plan(evaluation_runner_path)
    )
    if (
        frozen_dry_run.get("frozen_provenance_verified") is not True
        or frozen_dry_run.get("test_split_read") is not False
    ):
        raise EvaluationAggregateV28ContractError(
            "v2.8 frozen provenance dry-run did not pass"
        )
    inventory = inspect_v28_terminal_inventory(
        repository_root=root,
        plan=plan,
    )
    results, provenance = load_v28_stored_guard_results(
        repository_root=root,
        plan=plan,
        inventory=inventory,
    )
    dry_run_result = {
        **base,
        "schema_version": "benchmark-v2.8-aggregate-dry-run-v1",
        "mode": "DRY_RUN",
        "input_inventory_verified": True,
        "input_inventory_sha256": inventory[
            "input_inventory_sha256"
        ],
        "frozen_provenance_verified": True,
        "stored_candidate_results_verified": len(results),
        "guard_recalculations_executed": provenance[
            "guard_recalculations_executed"
        ],
        "validation_samples_read": provenance[
            "validation_samples_read"
        ],
        "test_split_read": False,
    }
    if mode == "dry-run":
        return dry_run_result

    expected_output = (root / plan.future_output_root).resolve()
    if output_root != expected_output:
        raise EvaluationAggregateV28ContractError(
            "aggregate output is outside the v2.8 append-only root"
        )
    if authorization_path is None:
        raise EvaluationAggregateV28ContractError(
            "execute requires aggregate-only authorization"
        )
    authorization_path = authorization_path.resolve()
    authorization_root = (
        root / "artifacts/benchmark_v2_8/authorizations"
    ).resolve()
    if (
        authorization_path.parent != authorization_root
        or "aggregate" not in authorization_path.name
        or not authorization_path.is_file()
    ):
        raise EvaluationAggregateV28ContractError(
            "aggregate authorization is outside append-only history"
        )
    authorization = _read_json(
        authorization_path,
        role="v2.8 aggregate-only authorization",
    )
    validate_v28_aggregate_authorization(
        plan=plan,
        inventory=inventory,
        authorization=authorization,
        source_commit=source_commit,
        relevant_source_sha256=relevant_hash,
    )
    report, selection_manifest = build_v28_selection(
        candidate_results=results,
        provenance={
            **provenance,
            "aggregate_source_commit": source_commit,
            "aggregate_relevant_source_sha256": relevant_hash,
            "aggregate_authorization_sha256": sha256_file(
                authorization_path
            ),
        },
    )
    terminal = write_v28_aggregate_bundle(
        output_root=output_root,
        authorization_path=authorization_path.relative_to(root),
        authorization_sha256=sha256_file(authorization_path),
        input_inventory=inventory,
        selection_report=report,
        selection_manifest=selection_manifest,
    )
    return {
        **dry_run_result,
        "schema_version": "benchmark-v2.8-aggregate-execution-v1",
        "mode": "EXECUTE",
        "selection_executed": True,
        "runtime_artifact_created": True,
        "aggregate_terminal": terminal,
        "model_selections": report["model_selections"],
        "primary_c2_selection_ready": report[
            "primary_c2_selection_ready"
        ],
    }
