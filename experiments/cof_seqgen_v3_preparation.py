"""Read-only plan/dry-run orchestration for CoF-SeqGen v3 source prep."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from eval.cof_seqgen_v3_contract import (
    V3ContractError,
    V3Definition,
    load_v3_definition,
    sha256_file,
)
from models.cof_seqgen_v3 import CoFSeqDenoiserV3, CoFSeqGenV3


ZERO_EXECUTION_COUNTS = {
    "gpu_queries": 0,
    "cuda_calls": 0,
    "model_fit_calls": 0,
    "optimizer_updates": 0,
    "checkpoint_writes": 0,
    "model_sample_calls": 0,
    "validation_execution_calls": 0,
    "selection_calls": 0,
    "test_split_reads": 0,
    "fresh_test_calls": 0,
    "tstr_calls": 0,
    "privacy_calls": 0,
    "five_seed_full_run_calls": 0,
}
RELEVANT_SOURCE_PATHS = (
    "configs/benchmark_v2/cof_seqgen_v3_source_preparation.yaml",
    "models/cof_seqgen_v3.py",
    "eval/cof_seqgen_v3_contract.py",
    "experiments/cof_seqgen_v3_preparation.py",
    "scripts/prepare_cof_seqgen_v3.py",
)


@dataclass(frozen=True)
class V3Plan:
    definition: V3Definition


def build_v3_plan(config_path: Path) -> V3Plan:
    return V3Plan(load_v3_definition(config_path))


def v3_relevant_source_sha256(repository_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in RELEVANT_SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise V3ContractError(f"v3 source is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def v3_plan_report(plan: V3Plan) -> Mapping[str, Any]:
    definition = plan.definition
    candidates = definition.candidates
    return {
        "schema_version": "cof-seqgen-v3-source-plan-v1",
        "status": "PASS",
        "mode": "SOURCE_ONLY_IMPLEMENTATION",
        "config_path": str(
            definition.config_path.relative_to(
                definition.repository_root
            )
        ),
        "config_sha256": definition.config_sha256,
        "relevant_source_sha256": v3_relevant_source_sha256(
            definition.repository_root
        ),
        "counts": {
            "candidates": len(candidates),
            "frozen_hash_references": sum(
                candidate.execution_kind == "hash_reference_only"
                for candidate in candidates
            ),
            "future_training_trajectories": sum(
                candidate.training_required for candidate in candidates
            ),
            "future_validation_candidates": sum(
                bool(candidate.raw.get("validation_required"))
                for candidate in candidates
            ),
        },
        "operations": [
            {
                "candidate_id": candidate.candidate_id,
                "architecture": candidate.architecture,
                "execution_kind": candidate.execution_kind,
                "training_required": candidate.training_required,
                "sampling_required": bool(
                    candidate.raw.get("sampling_required")
                ),
                "validation_required": bool(
                    candidate.raw.get("validation_required")
                ),
                "paired_discrete_mask": (
                    False
                    if not candidate.training_required
                    else True
                ),
                "post_hoc_calibration": "FORBIDDEN",
            }
            for candidate in candidates
        ],
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),
        "authorization_created": False,
        "runtime_artifact_created": False,
        "execution_authorized": False,
    }


def _v28_candidate_tree_sha256(repository_root: Path) -> str:
    artifact_root = repository_root / "artifacts/benchmark_v2_8"
    excluded_aggregate = (
        artifact_root
        / "candidate_selection/aggregate_attempt_001"
    ).resolve()
    excluded_authorization = (
        artifact_root
        / "authorizations/"
        "aggregate_authorization_3b4770f_attempt_001.json"
    ).resolve()
    files = sorted(
        (
            path
            for path in artifact_root.rglob("*")
            if path.is_file()
            and excluded_aggregate not in path.resolve().parents
            and path.resolve() != excluded_authorization
        ),
        key=lambda path: path.relative_to(repository_root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(repository_root).as_posix()
        digest.update(
            f"{sha256_file(path)}  {relative}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V3ContractError(f"cannot read frozen JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise V3ContractError(f"frozen JSON is not an object: {path}")
    return value


def dry_run_v3(plan: V3Plan) -> Mapping[str, Any]:
    definition = plan.definition
    root = definition.repository_root
    if sha256_file(definition.config_path) != definition.config_sha256:
        raise V3ContractError("v3 config changed after plan construction")
    raw = definition.raw
    specification = root / str(raw["specification_path"])
    if sha256_file(specification) != raw.get("specification_sha256"):
        raise V3ContractError("v3 specification hash mismatch")
    frozen = raw["frozen_contract"]
    inventory: dict[str, Mapping[str, str]] = {}
    path_hash_pairs = (
        ("train_file_path", "train_file_sha256"),
        ("validation_file_path", "validation_file_sha256"),
        ("v2_5_config_path", "v2_5_config_sha256"),
        ("v2_5_final_complete_path", "v2_5_final_complete_sha256"),
        (
            "v2_5_frozen_manifest_path",
            "v2_5_frozen_manifest_sha256",
        ),
        (
            "v2_8_candidate_config_path",
            "v2_8_candidate_config_sha256",
        ),
        (
            "v2_8_runner_config_path",
            "v2_8_runner_config_sha256",
        ),
        (
            "v2_8_aggregate_config_path",
            "v2_8_aggregate_config_sha256",
        ),
        (
            "v2_8_aggregate_complete_path",
            "v2_8_aggregate_complete_sha256",
        ),
        (
            "v2_8_selection_report_path",
            "v2_8_selection_report_sha256",
        ),
        (
            "v2_8_selection_manifest_path",
            "v2_8_selection_manifest_sha256",
        ),
    )
    for path_key, hash_key in path_hash_pairs:
        relative = str(frozen[path_key])
        actual = sha256_file(root / relative)
        if actual != frozen.get(hash_key):
            raise V3ContractError(f"frozen provenance mismatch: {hash_key}")
        inventory[hash_key] = {"path": relative, "sha256": actual}
    tree_hash = _v28_candidate_tree_sha256(root)
    if tree_hash != frozen.get("v2_8_candidate_tree_sha256"):
        raise V3ContractError("v2.8 candidate tree changed")

    reference = definition.candidates[0].raw["frozen_reference"]
    attempt = root / str(reference["attempt_path"])
    reference_files = (
        ("manifest.json", "manifest_sha256"),
        ("COMPLETE.json", "complete_sha256"),
        ("candidate_result.json", "candidate_result_sha256"),
        ("evaluation.json", "evaluation_sha256"),
        ("validation_sample.npz", "validation_sample_sha256"),
    )
    for filename, hash_key in reference_files:
        actual = sha256_file(attempt / filename)
        if actual != reference.get(hash_key):
            raise V3ContractError(
                f"frozen v2.8 CoF reference mismatch: {filename}"
            )

    report = _read_json(root / str(frozen["v2_8_selection_report_path"]))
    selections = report.get("model_selections", {})
    cof_result = next(
        (
            value
            for value in report.get("candidate_results", [])
            if value.get("candidate_id")
            == "cof_v28_c01_gap_distribution_sampler"
        ),
        None,
    )
    if (
        report.get("primary_c2_selection_ready") is not False
        or selections.get("ctgan_separate_class", {}).get("status")
        != "SELECTED"
        or selections.get("tvae_separate_class", {}).get("status")
        != "SELECTED"
        or selections.get("cof_seqgen", {}).get("status")
        != "NO_PASSING_CANDIDATE"
        or not isinstance(cof_result, Mapping)
        or cof_result.get("statistics", {}).get(
            "receiver_max_abs_signed_frequency"
        )
        != 0.021126555312304892
        or cof_result.get("checks", {}).get(
            "receiver_max_abs_signed_frequency"
        )
        != "FAIL"
    ):
        raise V3ContractError("v2.8 official conclusion changed")
    if (
        CoFSeqDenoiserV3.CANDIDATES
        != ("direct_joint", "factorized_joint")
        or CoFSeqGenV3.AMOUNT_CONTRACT
        != "train_fitted_centered_empirical_residual_257_v2_8_frozen"
    ):
        raise V3ContractError("v3 architecture import contract changed")

    return {
        **v3_plan_report(plan),
        "schema_version": "cof-seqgen-v3-source-dry-run-v1",
        "frozen_provenance_verified": True,
        "v2_8_official_conclusion_verified": True,
        "v2_8_candidate_tree_sha256": tree_hash,
        "frozen_reference_artifacts_verified": len(reference_files),
        "architecture_import_verified": True,
        "frozen_inventory": inventory,
        "authorization_created": False,
        "runtime_artifact_created": False,
    }
