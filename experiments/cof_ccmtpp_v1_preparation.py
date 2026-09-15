"""Read-only plan/dry-run orchestration for ``cof_ccmtpp_v1``."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping

from eval.cof_ccmtpp_v1_contract import (
    CCMTPPContractError,
    CCMTPPDefinition,
    load_ccmtpp_definition,
    sha256_file,
)
from models.cof_ccmtpp_v1 import (
    CausalEventDecoder,
    CoFCCMTPPV1,
    HierarchicalCopyReceiverDecoder,
    MixtureLogisticGapHead,
)


ZERO_EXECUTION_COUNTS = {
    "gpu_queries": 0,
    "cuda_calls": 0,
    "model_fit_calls": 0,
    "optimizer_updates": 0,
    "checkpoint_writes": 0,
    "model_sample_calls": 0,
    "evaluation_calls": 0,
    "selection_calls": 0,
    "internal_test_reads": 0,
    "sparkov_fraud_test_reads": 0,
    "tstr_calls": 0,
    "privacy_calls": 0,
    "full_run_calls": 0,
}
RELEVANT_SOURCE_PATHS = (
    "configs/benchmark_v2/cof_ccmtpp_v1_source_only.yaml",
    "models/cof_ccmtpp_v1.py",
    "eval/cof_ccmtpp_v1_contract.py",
    "experiments/cof_ccmtpp_v1_preparation.py",
    "scripts/prepare_cof_ccmtpp_v1.py",
)


@dataclass(frozen=True)
class CCMTPPPlan:
    definition: CCMTPPDefinition


def build_ccmtpp_plan(config_path: Path) -> CCMTPPPlan:
    return CCMTPPPlan(load_ccmtpp_definition(config_path))


def relevant_source_sha256(repository_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in RELEVANT_SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise CCMTPPContractError(f"source file is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def ccmtpp_plan_report(plan: CCMTPPPlan) -> Mapping[str, Any]:
    definition = plan.definition
    return {
        "schema_version": "cof-ccmtpp-v1-plan-v1",
        "status": "PASS",
        "mode": "SOURCE_ONLY",
        "config_sha256": definition.config_sha256,
        "relevant_source_sha256": relevant_source_sha256(definition.repository_root),
        "counts": {
            "frozen_controls": 1,
            "implemented_candidates": 4,
            "eligible_future_candidates_now": 1,
            "gated_future_candidates": 3,
            "unimplemented_candidates": 1,
        },
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "model_id": candidate.model_id,
                "status": candidate.status,
                "implemented": candidate.implemented,
                "execution_authorized": False,
            }
            for candidate in definition.candidates
        ],
        "deferred_C5": dict(definition.raw["deferred_C5"]),
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),
        "authorization_created": False,
        "runtime_artifact_created": False,
    }


def _v28_candidate_tree_sha256(repository_root: Path) -> str:
    artifact_root = repository_root / "artifacts/benchmark_v2_8"
    excluded_aggregate = (
        artifact_root / "candidate_selection/aggregate_attempt_001"
    ).resolve()
    excluded_authorization = (
        artifact_root / "authorizations/aggregate_authorization_3b4770f_attempt_001.json"
    ).resolve()
    files = sorted(
        (
            path for path in artifact_root.rglob("*")
            if path.is_file()
            and excluded_aggregate not in path.resolve().parents
            and path.resolve() != excluded_authorization
        ),
        key=lambda path: path.relative_to(repository_root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(repository_root).as_posix()
        digest.update(f"{sha256_file(path)}  {relative}\n".encode("utf-8"))
    return digest.hexdigest()


def _external_bundle_digest(
    repository_root: Path, bundle_relative: str, artifact_relative: str
) -> str:
    digest = hashlib.sha256()
    for label in (bundle_relative, artifact_relative):
        root = repository_root / label
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = f"{label}:{path.relative_to(root)}".encode("utf-8")
            payload = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def _git_blob_sha256(root: Path, commit: str, relative: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CCMTPPContractError("cannot read frozen non-v3 source blob") from error
    return hashlib.sha256(completed.stdout).hexdigest()


def dry_run_ccmtpp(plan: CCMTPPPlan) -> Mapping[str, Any]:
    definition = plan.definition
    root = definition.repository_root
    if sha256_file(definition.config_path) != definition.config_sha256:
        raise CCMTPPContractError("config changed after plan construction")
    preserved = definition.raw["preservation"]
    verified_files = 0
    for record in preserved.values():
        if isinstance(record, Mapping) and "path" in record:
            actual = sha256_file(root / str(record["path"]))
            if actual != record.get("sha256"):
                raise CCMTPPContractError(f"preserved file changed: {record['path']}")
            verified_files += 1
    v28 = _v28_candidate_tree_sha256(root)
    if v28 != preserved["v2_8_candidate_tree_sha256"]:
        raise CCMTPPContractError("preserved v2.8 candidate tree changed")
    external_hashes = {}
    for dataset in ("amlsim", "sparkov"):
        record = preserved[f"{dataset}_bundle"]
        actual = _external_bundle_digest(
            root, str(record["bundle_root"]), str(record["artifact_root"])
        )
        if actual != record["tree_sha256"]:
            raise CCMTPPContractError(f"preserved {dataset} bundle changed")
        external_hashes[dataset] = actual
    legacy = preserved["frozen_non_v3_cof"]
    for path_key, hash_key in (
        ("model_path", "model_sha256"),
        ("adapter_path", "adapter_sha256"),
        ("denoiser_path", "denoiser_sha256"),
    ):
        if _git_blob_sha256(root, legacy["source_commit"], legacy[path_key]) != legacy[hash_key]:
            raise CCMTPPContractError("frozen non-v3 source blob changed")
    # Imports above verify that the architecture surface exists. No model is
    # instantiated and no fit/sample/device operation is called.
    if not all((CausalEventDecoder, MixtureLogisticGapHead,
                HierarchicalCopyReceiverDecoder, CoFCCMTPPV1)):
        raise CCMTPPContractError("architecture import failed")
    report = dict(ccmtpp_plan_report(plan))
    report.update({
        "schema_version": "cof-ccmtpp-v1-dry-run-v1",
        "preservation_verified": True,
        "preserved_file_hashes_verified": verified_files,
        "preserved_v2_8_tree_sha256": v28,
        "preserved_external_bundle_sha256": external_hashes,
        "frozen_non_v3_source_blobs_verified": 3,
        "architecture_import_verified": True,
        "fit_sample_cuda_instrumentation": dict(ZERO_EXECUTION_COUNTS),
    })
    return report
