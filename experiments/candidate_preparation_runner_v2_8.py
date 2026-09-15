from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from eval.candidate_preparation_v2_8 import (
    V28Definition,
    V28PreparationContractError,
    load_v28_definition,
    sha256_file,
)


ZERO_EXECUTION_COUNTS = {
    "gpu_queries": 0,
    "cuda_calls": 0,
    "training_calls": 0,
    "checkpoint_update_calls": 0,
    "optimizer_updates": 0,
    "model_restore_calls": 0,
    "model_sample_calls": 0,
    "evaluation_calls": 0,
    "selection_calls": 0,
    "test_split_reads": 0,
    "fresh_test_calls": 0,
    "tstr_calls": 0,
    "privacy_calls": 0,
    "five_seed_full_run_calls": 0,
}
RELEVANT_SOURCE_PATHS = (
    "configs/benchmark_v2/selection_v2_8_source_amendment.yaml",
    "eval/candidate_preparation_v2_8.py",
    "experiments/candidate_preparation_runner_v2_8.py",
    "scripts/prepare_candidates_v2_8.py",
)
FROZEN_FILE_PATHS = {
    "train_file_sha256": (
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
        "kappa_1.00/train.npz"
    ),
    "validation_file_sha256": (
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
        "kappa_1.00/validation.npz"
    ),
    "v2_5_config_sha256": "configs/benchmark_v2/full_v2_5.yaml",
    "v2_5_final_complete_sha256": (
        "artifacts/benchmark_v2_5/full/FINAL_COMPLETE.json"
    ),
    "v2_5_frozen_manifest_sha256": (
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
        "kappa_1.00/data_manifest.json"
    ),
    "v2_6_forensic_terminal_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/"
        "forensic_attempt_001/FORENSIC_COMPLETE.json"
    ),
    "v2_7_runner_config_sha256": (
        "configs/benchmark_v2/evaluation_only_v2_7.yaml"
    ),
    "v2_7_candidate_config_sha256": (
        "configs/benchmark_v2/selection_v2_7_source_preparation.yaml"
    ),
    "v2_7_execution_authorization_sha256": (
        "artifacts/benchmark_v2_7/authorizations/"
        "authorization_de57f79_attempt_001.json"
    ),
    "v2_7_aggregate_authorization_sha256": (
        "artifacts/benchmark_v2_7/authorizations/"
        "authorization_fbaaac55_aggregate_attempt_001.json"
    ),
    "v2_7_aggregate_complete_sha256": (
        "artifacts/benchmark_v2_7/candidate_selection/"
        "aggregate_attempt_001/AGGREGATE_COMPLETE.json"
    ),
    "v2_7_selection_report_sha256": (
        "artifacts/benchmark_v2_7/candidate_selection/"
        "aggregate_attempt_001/selection_report.json"
    ),
    "v2_7_forensic_report_sha256": (
        "docs/benchmark_v2/forensic_no_passing_candidate_v2_7.md"
    ),
    "v2_7_forensic_json_sha256": (
        "docs/benchmark_v2/forensic_no_passing_candidate_v2_7.json"
    ),
    "v2_7_forensic_csv_sha256": (
        "docs/benchmark_v2/forensic_no_passing_candidate_v2_7.csv"
    ),
}
PARENT_AUTHORIZATION_PATH = (
    "artifacts/benchmark_v2_7/authorizations/"
    "authorization_de57f79_attempt_001.json"
)


@dataclass(frozen=True)
class V28Plan:
    definition: V28Definition

    @property
    def repository_root(self) -> Path:
        return self.definition.repository_root


def build_v28_plan(config_path: Path) -> V28Plan:
    return V28Plan(definition=load_v28_definition(config_path))


def v28_relevant_source_sha256(repository_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in RELEVANT_SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise V28PreparationContractError(
                f"v2.8 relevant source is missing: {relative}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def v28_plan_report(plan: V28Plan) -> Mapping[str, Any]:
    definition = plan.definition
    candidates = definition.candidates
    raw_contract = definition.raw["future_evaluation_contract"]
    execution_contract = {
        key: raw_contract[key]
        for key in (
            "checkpoint_access",
            "existing_checkpoint_required",
            "optimizer_updates",
            "training_calls",
            "checkpoint_writes",
        )
    }
    return {
        "schema_version": "benchmark-v2.8-source-plan-v1",
        "status": "PASS",
        "mode": "SOURCE_ONLY_PREPARATION",
        "config_path": str(
            definition.config_path.relative_to(
                definition.repository_root
            )
        ),
        "config_sha256": definition.config_sha256,
        "relevant_source_sha256": v28_relevant_source_sha256(
            definition.repository_root
        ),
        "models": list(definition.model_ids),
        "counts": {
            "models": len(definition.model_ids),
            "candidates": len(candidates),
            "frozen_parent_controls": sum(
                candidate.is_control for candidate in candidates
            ),
            "future_evaluation_only": sum(
                not candidate.is_control for candidate in candidates
            ),
            "new_training_trajectories": 0,
        },
        "execution_contract": execution_contract,
        "operations": [
            {
                "model_id": candidate.model_id,
                "candidate_id": candidate.candidate_id,
                "execution_kind": candidate.execution_kind,
                "factor": candidate.factor,
                "changed_dimensions": list(
                    candidate.changed_dimensions
                ),
                "future_sampling_required": (
                    candidate.future_sampling_required
                ),
                "parent_candidate_id": str(
                    candidate.frozen_parent["candidate_id"]
                ),
                "checkpoint_path": str(
                    candidate.frozen_parent["checkpoint_path"]
                ),
                "checkpoint_sha256": str(
                    candidate.frozen_parent["checkpoint_sha256"]
                ),
                **execution_contract,
            }
            for candidate in candidates
        ],
        "current_execution_counts": dict(ZERO_EXECUTION_COUNTS),
        "authorization_created": False,
        "runtime_artifact_created": False,
        "execution_authorized": False,
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V28PreparationContractError(
            f"cannot read frozen parent JSON: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise V28PreparationContractError(
            f"frozen parent JSON is not an object: {path}"
        )
    return value


def _tree_record(
    *,
    relative_root: Path,
    roots: tuple[Path, ...],
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


def _verify_parent_history(
    repository_root: Path,
    parent_commit: str,
) -> None:
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            parent_commit,
            "HEAD",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise V28PreparationContractError(
            "v2.7 forensic parent is not preserved in history"
        )


def _verify_parent(
    *,
    root: Path,
    model_id: str,
    parent: Mapping[str, Any],
) -> tuple[Mapping[str, Any], int]:
    attempt = (root / str(parent["attempt_path"])).resolve()
    expected_attempt_parent = (
        root
        / "artifacts/benchmark_v2_7/candidate_selection/evaluations"
        / model_id
        / str(parent["candidate_id"])
        / "seed_2601"
    ).resolve()
    if attempt.parent != expected_attempt_parent:
        raise V28PreparationContractError(
            f"frozen parent path mismatch: {model_id}"
        )
    files = {
        "manifest": ("manifest.json", "manifest_sha256"),
        "complete": ("COMPLETE.json", "complete_sha256"),
        "candidate_result": (
            "candidate_result.json",
            "candidate_result_sha256",
        ),
        "evaluation": ("evaluation.json", "evaluation_sha256"),
        "fit_state": ("fit_state.json", "fit_state_sha256"),
        "validation_sample": (
            "validation_sample.npz",
            "validation_sample_sha256",
        ),
    }
    record: dict[str, Any] = {}
    for role, (name, key) in files.items():
        path = attempt / name
        actual = sha256_file(path)
        if actual != parent.get(key):
            raise V28PreparationContractError(
                f"frozen parent {role} mismatch: {model_id}"
            )
        record[role] = {
            "path": str(path.relative_to(root)),
            "sha256": actual,
        }
    checkpoint = root / str(parent["checkpoint_path"])
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != parent.get("checkpoint_sha256"):
        raise V28PreparationContractError(
            f"frozen parent checkpoint mismatch: {model_id}"
        )
    record["checkpoint"] = {
        "path": str(checkpoint.relative_to(root)),
        "sha256": checkpoint_hash,
    }
    authorization = root / PARENT_AUTHORIZATION_PATH
    authorization_hash = sha256_file(authorization)
    if authorization_hash != parent.get("authorization_sha256"):
        raise V28PreparationContractError(
            f"frozen parent authorization mismatch: {model_id}"
        )
    record["authorization"] = {
        "path": PARENT_AUTHORIZATION_PATH,
        "sha256": authorization_hash,
    }

    manifest = _read_json(attempt / "manifest.json")
    complete = _read_json(attempt / "COMPLETE.json")
    candidate_result = _read_json(attempt / "candidate_result.json")
    evaluation = _read_json(attempt / "evaluation.json")
    fit_state = _read_json(attempt / "fit_state.json")
    common_checks = (
        manifest.get("candidate_id") == parent.get("candidate_id"),
        manifest.get("model_id") == model_id,
        manifest.get("checkpoint_path") == parent.get("checkpoint_path"),
        manifest.get("checkpoint_sha256")
        == parent.get("checkpoint_sha256"),
        manifest.get("source_commit") == parent.get("source_commit"),
        manifest.get("relevant_source_sha256")
        == parent.get("relevant_source_sha256"),
        manifest.get("candidate_config_sha256")
        == parent.get("candidate_config_sha256"),
        manifest.get("runner_config_sha256")
        == parent.get("runner_config_sha256"),
        manifest.get("sampling_plan_sha256")
        == parent.get("sampling_plan_sha256"),
        manifest.get("authorization_sha256")
        == parent.get("authorization_sha256"),
        manifest.get("optimizer_updates") == 0,
        manifest.get("training_calls") == 0,
        manifest.get("test_split_read") is False,
        complete.get("status") == "COMPLETE",
        complete.get("candidate_result_sha256")
        == parent.get("candidate_result_sha256"),
        complete.get("evaluation_sha256")
        == parent.get("evaluation_sha256"),
        complete.get("fit_state_sha256")
        == parent.get("fit_state_sha256"),
        complete.get("validation_sample_sha256")
        == parent.get("validation_sample_sha256"),
        candidate_result.get("candidate_id")
        == parent.get("candidate_id"),
        candidate_result.get("checkpoint_sha256")
        == parent.get("checkpoint_sha256"),
        candidate_result.get("optimizer_updates") == 0,
        candidate_result.get("training_calls") == 0,
        candidate_result.get("test_split_read") is False,
        evaluation.get("candidate_id") == parent.get("candidate_id"),
        evaluation.get("evaluation_split") == "validation",
        evaluation.get("test_split_read") is False,
        fit_state.get("candidate_id") == parent.get("candidate_id"),
        fit_state.get("fit_split") == "train",
        fit_state.get("validation_rows_used") == 0,
        fit_state.get("test_rows_used") == 0,
    )
    if not all(common_checks):
        raise V28PreparationContractError(
            f"frozen parent provenance mismatch: {model_id}"
        )
    return record, len(record)


def dry_run_v28(plan: V28Plan) -> Mapping[str, Any]:
    definition = plan.definition
    root = definition.repository_root
    if sha256_file(definition.config_path) != definition.config_sha256:
        raise V28PreparationContractError(
            "v2.8 config changed after plan construction"
        )
    _verify_parent_history(
        root,
        str(definition.raw["parent_forensic_commit"]),
    )
    frozen = definition.raw["frozen_contract"]
    inventory: dict[str, Any] = {}
    for contract_key, relative in FROZEN_FILE_PATHS.items():
        path = root / relative
        actual = sha256_file(path)
        if actual != frozen.get(contract_key):
            raise V28PreparationContractError(
                f"frozen provenance mismatch: {contract_key}"
            )
        inventory[contract_key] = {
            "path": relative,
            "sha256": actual,
        }

    v27_root = root / (
        "artifacts/benchmark_v2_7/candidate_selection"
    )
    candidate_tree = _tree_record(
        relative_root=v27_root,
        roots=(v27_root / "workers", v27_root / "evaluations"),
    )
    aggregate_root = v27_root / "aggregate_attempt_001"
    aggregate_tree = _tree_record(
        relative_root=aggregate_root,
        roots=(aggregate_root,),
    )
    if candidate_tree != frozen.get("v2_7_candidate_input_tree"):
        raise V28PreparationContractError(
            "v2.7 candidate tree changed"
        )
    if aggregate_tree != frozen.get("v2_7_aggregate_bundle_tree"):
        raise V28PreparationContractError(
            "v2.7 aggregate tree changed"
        )
    inventory["v2_7_candidate_input_tree"] = candidate_tree
    inventory["v2_7_aggregate_bundle_tree"] = aggregate_tree

    parents: dict[str, Any] = {}
    parent_artifact_count = 0
    for model_id in definition.model_ids:
        parent = definition.raw["models"][model_id]["parent"]
        record, count = _verify_parent(
            root=root,
            model_id=model_id,
            parent=parent,
        )
        parents[model_id] = record
        parent_artifact_count += count

    return {
        **v28_plan_report(plan),
        "schema_version": "benchmark-v2.8-source-dry-run-v1",
        "status": "PASS",
        "parent_provenance_verified": True,
        "parents_verified": len(parents),
        "parent_artifacts_verified": parent_artifact_count,
        "parent_inventory": parents,
        "frozen_inventory": inventory,
        "fit_state_created": False,
        "authorization_created": False,
        "runtime_artifact_created": False,
    }
