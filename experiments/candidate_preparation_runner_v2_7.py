from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import yaml

from benchmarks.types import SequenceBatch
from eval.candidate_preparation_v2_7 import (
    V27PreparationContractError,
    V27PreparationPlan,
    load_and_validate_preparation,
    sha256_file,
    validate_candidate_io_path,
)
from experiments.provenance_v2_5 import hash_batch
from generators.sampling_plan import SamplingPlan


ZERO_EXECUTION_COUNTS = {
    "gpu_queries": 0,
    "cuda_calls": 0,
    "candidate_training_calls": 0,
    "model_fit_calls": 0,
    "model_sample_calls": 0,
    "validation_selection_calls": 0,
    "test_split_reads": 0,
    "fresh_test_calls": 0,
    "tstr_calls": 0,
    "privacy_calls": 0,
    "five_seed_full_run_calls": 0,
}
FROZEN_FILE_PATHS = {
    "v2_5_config_sha256": "configs/benchmark_v2/full_v2_5.yaml",
    "v2_5_final_complete_sha256": (
        "artifacts/benchmark_v2_5/full/FINAL_COMPLETE.json"
    ),
    "v2_5_final_artifact_index_sha256": (
        "artifacts/benchmark_v2_5/full/finalization_attempt_001/"
        "artifact_index.json"
    ),
    "v2_5_final_checksum_sha256": (
        "artifacts/benchmark_v2_5/full/finalization_attempt_001/"
        "checksum_manifest_report.json"
    ),
    "v2_5_frozen_manifest_sha256": (
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
        "kappa_1.00/data_manifest.json"
    ),
    "v2_6_single_factor_config_sha256": (
        "configs/benchmark_v2/selection_v2_6_single_factor_amendment.yaml"
    ),
    "v2_6_candidate_authorization_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/"
        "authorization_history/authorization_ad19d733_attempt_001.json"
    ),
    "v2_6_aggregate_authorization_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/"
        "authorization_history/"
        "authorization_24d9ccec_aggregate_attempt_001.json"
    ),
    "v2_6_aggregate_complete_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/"
        "aggregate_attempt_001/AGGREGATE_COMPLETE.json"
    ),
    "v2_6_aggregate_index_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/"
        "aggregate_attempt_001/artifact_index.json"
    ),
    "v2_6_forensic_complete_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/"
        "forensic_attempt_001/FORENSIC_COMPLETE.json"
    ),
    "v2_6_forensic_index_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/"
        "forensic_attempt_001/forensic_artifact_index.json"
    ),
    "v2_6_forensic_evidence_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/"
        "forensic_attempt_001/forensic_evidence.json"
    ),
    "v2_6_forensic_csv_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/"
        "forensic_attempt_001/candidate_guard_evidence.csv"
    ),
}
FROZEN_TREE_PATHS = {
    "v2_6_candidate_tree_record_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/candidates"
    ),
    "v2_6_worker_tree_record_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/workers"
    ),
    "v2_6_trajectory_tree_record_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/trajectories"
    ),
    "v2_6_evaluation_tree_record_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/evaluations"
    ),
    "v2_6_aggregate_tree_record_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/"
        "aggregate_attempt_001"
    ),
    "v2_6_forensic_tree_record_sha256": (
        "artifacts/benchmark_v2_6/selection_single_factor/"
        "forensic_attempt_001"
    ),
}
RELEVANT_SOURCE_PATHS = (
    "configs/benchmark_v2/selection_v2_7_source_preparation.yaml",
    "eval/candidate_preparation_v2_7.py",
    "experiments/candidate_preparation_runner_v2_7.py",
    "scripts/prepare_candidates_v2_7.py",
)


@dataclass(frozen=True)
class V27PreparationExecutionPlan:
    definition: V27PreparationPlan

    @property
    def repository_root(self) -> Path:
        return self.definition.repository_root


def build_preparation_execution_plan(
    config_path: Path,
) -> V27PreparationExecutionPlan:
    return V27PreparationExecutionPlan(
        definition=load_and_validate_preparation(config_path)
    )


def v27_relevant_source_sha256(repository_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in RELEVANT_SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise V27PreparationContractError(
                f"v2.7 relevant source is missing: {relative}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def preparation_plan_report(
    plan: V27PreparationExecutionPlan,
) -> Mapping[str, Any]:
    candidates = plan.definition.candidates
    return {
        "schema_version": "benchmark-v2.7-source-plan-v1",
        "status": "PASS",
        "mode": "SOURCE_ONLY_PREPARATION",
        "config_path": str(
            plan.definition.config_path.relative_to(
                plan.definition.repository_root
            )
        ),
        "config_sha256": plan.definition.config_sha256,
        "relevant_source_sha256": v27_relevant_source_sha256(
            plan.definition.repository_root
        ),
        "models": list(plan.definition.model_ids),
        "counts": {
            "models": len(plan.definition.model_ids),
            "candidates": len(candidates),
            "frozen_controls": sum(
                candidate.is_control for candidate in candidates
            ),
            "future_evaluation_only": sum(
                not candidate.is_control for candidate in candidates
            ),
            "new_training_trajectories": sum(
                candidate.future_training_required
                for candidate in candidates
            ),
        },
        "operations": [
            {
                "model_id": candidate.model_id,
                "candidate_id": candidate.candidate_id,
                "execution_kind": candidate.execution_kind,
                "factor": candidate.factor,
                "changed_dimensions": list(
                    candidate.changed_dimensions
                ),
                "future_training_required": (
                    candidate.future_training_required
                ),
                "future_sampling_required": (
                    candidate.future_sampling_required
                ),
            }
            for candidate in candidates
        ],
        "sampling_plan_sha256": (
            plan.definition.sampling_plan_sha256
        ),
        "thresholds": dict(plan.definition.thresholds),
        "current_execution_counts": dict(ZERO_EXECUTION_COUNTS),
        "authorization_created": False,
        "execution_authorized": False,
    }


def _tree_record_sha256(root: Path) -> tuple[str, int, int]:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    byte_count = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        byte_count += path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files), byte_count


def _load_sequence_batch(path: Path) -> SequenceBatch:
    with np.load(path, allow_pickle=False) as archive:
        values = {
            field: archive[field]
            for field in SequenceBatch.__dataclass_fields__
        }
    return SequenceBatch(**values)


def _current_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()


def _verify_parent_in_history(
    repository_root: Path,
    parent: str,
) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", parent, "HEAD"],
        cwd=repository_root,
        check=False,
    )
    if result.returncode != 0:
        raise V27PreparationContractError(
            "frozen forensic parent is not preserved in history"
        )


def dry_run_preparation(
    plan: V27PreparationExecutionPlan,
) -> Mapping[str, Any]:
    definition = plan.definition
    root = definition.repository_root
    if sha256_file(definition.config_path) != definition.config_sha256:
        raise V27PreparationContractError(
            "v2.7 config changed after plan construction"
        )
    _verify_parent_in_history(root, definition.parent_forensic_commit)
    inventory: dict[str, Any] = {}
    for contract_key, relative in FROZEN_FILE_PATHS.items():
        path = root / relative
        actual = sha256_file(path)
        if actual != definition.frozen_contract.get(contract_key):
            raise V27PreparationContractError(
                f"frozen file provenance mismatch: {contract_key}"
            )
        inventory[contract_key] = {
            "path": relative,
            "sha256": actual,
        }
    for contract_key, relative in FROZEN_TREE_PATHS.items():
        actual, files, bytes_count = _tree_record_sha256(
            root / relative
        )
        if actual != definition.frozen_contract.get(contract_key):
            raise V27PreparationContractError(
                f"frozen tree provenance mismatch: {contract_key}"
            )
        inventory[contract_key] = {
            "path": relative,
            "sha256": actual,
            "files": files,
            "bytes": bytes_count,
        }
    for candidate in definition.candidates:
        if not candidate.is_control:
            continue
        for role in (
            "candidate_result",
            "checkpoint",
            "validation_sample",
        ):
            relative = str(candidate.frozen_control[f"{role}_path"])
            actual = sha256_file(root / relative)
            expected = str(
                candidate.frozen_control[f"{role}_sha256"]
            )
            if actual != expected:
                raise V27PreparationContractError(
                    f"frozen control mismatch: "
                    f"{candidate.model_id}/{role}"
                )
            inventory[
                f"{candidate.model_id}_{role}_sha256"
            ] = {
                "path": relative,
                "sha256": actual,
            }
    development_path = (
        root / "configs/benchmark_v2/development_data_v2_6.yaml"
    )
    if (
        sha256_file(development_path)
        != definition.frozen_contract["development_manifest_sha256"]
    ):
        raise V27PreparationContractError(
            "development manifest provenance mismatch"
        )
    development = yaml.safe_load(
        development_path.read_text(encoding="utf-8")
    )
    batches: dict[str, SequenceBatch] = {}
    for split in ("train", "validation"):
        record = development["splits"][split]
        split_path = validate_candidate_io_path(
            repository_root=root,
            path=root / record["path"],
            access="read",
        )
        if sha256_file(split_path) != definition.frozen_contract[
            f"{split}_file_sha256"
        ]:
            raise V27PreparationContractError(
                f"{split} file provenance mismatch"
            )
        batch = _load_sequence_batch(split_path)
        if hash_batch(batch) != definition.frozen_contract[
            f"{split}_content_sha256"
        ]:
            raise V27PreparationContractError(
                f"{split} content provenance mismatch"
            )
        batches[split] = batch
    plan_record = development["selection_sampling_plan"]
    sampling_plan = SamplingPlan.from_train_policy(
        batches["train"],
        entity_count=int(plan_record["entity_count"]),
        seed=int(plan_record["seed"]),
    )
    if sampling_plan.plan_hash != definition.sampling_plan_sha256:
        raise V27PreparationContractError(
            "train-only SamplingPlan provenance mismatch"
        )
    report = preparation_plan_report(plan)
    return {
        **report,
        "schema_version": "benchmark-v2.7-source-dry-run-v1",
        "source_commit": _current_head(root),
        "frozen_provenance_verified": True,
        "inventory": inventory,
        "train_content_sha256": hash_batch(batches["train"]),
        "validation_content_sha256": hash_batch(
            batches["validation"]
        ),
        "fit_state_created": False,
        "runtime_artifact_created": False,
        "authorization_created": False,
        "current_execution_counts": dict(ZERO_EXECUTION_COUNTS),
    }
