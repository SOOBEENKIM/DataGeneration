from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
from typing import Any, Callable, Mapping

import yaml

from eval.candidate_preparation_v2_8 import (
    V28Candidate,
    V28Definition,
    canonical_sha256,
    load_v28_definition,
    sha256_file,
)


MODEL_IDS = ("ctgan_separate_class", "cof_seqgen")
FROZEN_THRESHOLDS = {
    "amount_ks": 0.006081138155655141,
    "gap_ks": 0.006387882975686154,
    "amount_abs_standardized_label_effect": 0.0363693454591819,
    "gap_abs_standardized_label_effect": 0.051540527275560376,
    "receiver_max_abs_signed_frequency": 0.02,
}
FROZEN_SAMPLING_PLAN_SHA256 = (
    "862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27"
)
EXPECTED_RESTORE_MODES = {
    "ctgan_separate_class": "cpu_first_then_explicit_selected_device",
    "cof_seqgen": "selected_device_restore",
}
ZERO_EXECUTION_COUNTS = {
    "gpu_queries": 0,
    "cuda_calls": 0,
    "model_fit_calls": 0,
    "model_sample_calls": 0,
    "optimizer_updates": 0,
    "evaluation_calls": 0,
    "selection_calls": 0,
    "test_split_reads": 0,
}
RELEVANT_SOURCE_PATHS = (
    "configs/benchmark_v2/evaluation_only_v2_8.yaml",
    "configs/benchmark_v2/selection_v2_8_source_amendment.yaml",
    "eval/candidate_preparation_v2_8.py",
    "eval/evaluation_only_v2_8.py",
    "experiments/evaluation_only_runner_v2_8.py",
    "generators/evaluation_only_transforms_v2_8.py",
    "scripts/run_evaluation_only_v2_8.py",
)


class EvaluationOnlyContractError(RuntimeError):
    pass


class EvaluationOnlyArtifactStore:
    def __init__(self, repository_root: Path):
        self.repository_root = repository_root.resolve()
        self.root = (
            self.repository_root
            / "artifacts/benchmark_v2_8/candidate_selection"
        )

    def claim_model(
        self,
        *,
        model_id: str,
        provenance_sha256: str,
    ) -> Path:
        if model_id not in MODEL_IDS or len(provenance_sha256) != 64:
            raise EvaluationOnlyContractError(
                "invalid v2.8 evaluation ownership claim"
            )
        worker_root = self.root / "workers" / model_id
        worker_root.mkdir(parents=True, exist_ok=True)
        claim = worker_root / "ownership.lock"
        self.write_json(
            claim,
            {
                "schema_version": "benchmark-v2.8-evaluation-owner-v1",
                "model_id": model_id,
                "provenance_sha256": provenance_sha256,
            },
        )
        return claim

    def create_evaluation_attempt(
        self,
        *,
        model_id: str,
        candidate_id: str,
        seed: int,
    ) -> Path:
        expected_prefix = {
            "ctgan_separate_class": "ctgan_v28_",
            "cof_seqgen": "cof_v28_",
        }
        if (
            model_id not in MODEL_IDS
            or not candidate_id.startswith(expected_prefix[model_id])
            or not (
                self.root / "workers" / model_id / "ownership.lock"
            ).is_file()
        ):
            raise EvaluationOnlyContractError(
                "evaluation attempt requires its exclusive model owner"
            )
        parent = (
            self.root
            / "evaluations"
            / model_id
            / candidate_id
            / f"seed_{seed}"
        )
        parent.mkdir(parents=True, exist_ok=True)
        for number in range(1, 1_000_000):
            attempt = parent / f"attempt_{number:03d}"
            try:
                attempt.mkdir()
            except FileExistsError:
                continue
            return attempt
        raise EvaluationOnlyContractError(
            "append-only evaluation attempt namespace exhausted"
        )

    def write_json(
        self,
        path: Path,
        value: Mapping[str, Any],
    ) -> None:
        path = path.resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as error:
            raise EvaluationOnlyContractError(
                "evaluation artifact escapes append-only root"
            ) from error
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(
                    value,
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
        except FileExistsError as error:
            raise EvaluationOnlyContractError(
                f"append-only evaluation artifact exists: {path}"
            ) from error


@dataclass(frozen=True)
class EvaluationOperation:
    model_id: str
    candidate_id: str
    factor: str
    checkpoint_path: str
    checkpoint_sha256: str
    restore_mode: str
    optimizer_updates: int = 0
    training_calls: int = 0
    checkpoint_writes: int = 0


@dataclass(frozen=True)
class FrozenReference:
    model_id: str
    candidate_id: str


@dataclass(frozen=True)
class EvaluationExecutionPlan:
    repository_root: Path
    runner_config_path: Path
    runner_config_sha256: str
    candidate_config_sha256: str
    runtime_root: Path
    definition: V28Definition
    operations: tuple[EvaluationOperation, ...]
    references: tuple[FrozenReference, ...]
    execution_contract: Mapping[str, Any]
    validation_contract: Mapping[str, Any]
    training_trajectories: tuple[Any, ...] = ()


def _operation(
    candidate: V28Candidate,
    restore: Mapping[str, Any],
) -> EvaluationOperation:
    return EvaluationOperation(
        model_id=candidate.model_id,
        candidate_id=candidate.candidate_id,
        factor=candidate.factor,
        checkpoint_path=str(restore["checkpoint_path"]),
        checkpoint_sha256=str(restore["checkpoint_sha256"]),
        restore_mode=str(restore["mode"]),
    )


def _reference(candidate: V28Candidate) -> FrozenReference:
    return FrozenReference(
        model_id=candidate.model_id,
        candidate_id=candidate.candidate_id,
    )


def build_evaluation_execution_plan(
    runner_config_path: Path,
) -> EvaluationExecutionPlan:
    runner_config_path = runner_config_path.resolve()
    repository_root = runner_config_path.parents[2]
    try:
        raw = yaml.safe_load(
            runner_config_path.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as error:
        raise EvaluationOnlyContractError(
            "cannot read v2.8 evaluation runner config"
        ) from error
    false_flags = (
        "execution_authorized",
        "authorization_creation_authorized",
        "gpu_query_authorized",
        "cuda_authorized",
        "training_authorized",
        "checkpoint_update_authorized",
        "sampling_authorized",
        "evaluation_authorized",
        "selection_authorized",
        "fresh_test_authorized",
        "tstr_authorized",
        "privacy_authorized",
        "five_seed_full_run_authorized",
    )
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema_version")
        != "benchmark-v2.8-evaluation-only-runner-v1"
        or raw.get("mode") != "SOURCE_ONLY_IMPLEMENTATION"
        or raw.get("authorization_required_for_execute") is not True
        or raw.get("test_split_access") != "FORBIDDEN"
        or any(raw.get(flag) is not False for flag in false_flags)
    ):
        raise EvaluationOnlyContractError(
            "v2.8 execution authorization boundary changed"
        )
    candidate_config = (
        repository_root / str(raw["candidate_config_path"])
    ).resolve()
    if sha256_file(candidate_config) != raw.get(
        "candidate_config_sha256"
    ):
        raise EvaluationOnlyContractError(
            "v2.8 candidate config provenance mismatch"
        )
    definition = load_v28_definition(candidate_config)
    restore_contract = raw.get("restore_contract")
    execution_contract = raw.get("execution_contract")
    validation_contract = raw.get("validation_contract")
    if (
        not isinstance(restore_contract, Mapping)
        or tuple(restore_contract) != MODEL_IDS
        or not isinstance(execution_contract, Mapping)
        or not isinstance(validation_contract, Mapping)
        or validation_contract.get("thresholds") != FROZEN_THRESHOLDS
        or validation_contract.get("sampling_plan_sha256")
        != FROZEN_SAMPLING_PLAN_SHA256
        or validation_contract.get(
            "candidate_selection_inside_worker"
        )
        != "FORBIDDEN"
        or validation_contract.get(
            "result_contingent_candidate_change"
        )
        != "FORBIDDEN"
    ):
        raise EvaluationOnlyContractError(
            "v2.8 restore or validation contract changed"
        )
    operations = []
    for candidate in definition.candidates:
        if candidate.is_control:
            continue
        restore = restore_contract.get(candidate.model_id)
        if (
            not isinstance(restore, Mapping)
            or restore.get("mode")
            != EXPECTED_RESTORE_MODES[candidate.model_id]
            or restore.get("checkpoint_path")
            != candidate.frozen_parent["checkpoint_path"]
            or restore.get("checkpoint_sha256")
            != candidate.frozen_parent["checkpoint_sha256"]
        ):
            raise EvaluationOnlyContractError(
                f"restore provenance changed: {candidate.model_id}"
            )
        operations.append(_operation(candidate, restore))
    if (
        [(item.model_id, item.candidate_id) for item in operations]
        != [
            (
                "ctgan_separate_class",
                "ctgan_v28_c01_joint_gap_receiver_decoder",
            ),
            ("cof_seqgen", "cof_v28_c01_gap_distribution_sampler"),
        ]
        or any(
            item.optimizer_updates
            or item.training_calls
            or item.checkpoint_writes
            for item in operations
        )
    ):
        raise EvaluationOnlyContractError(
            "v2.8 evaluation operation family changed"
        )
    return EvaluationExecutionPlan(
        repository_root=repository_root,
        runner_config_path=runner_config_path,
        runner_config_sha256=sha256_file(runner_config_path),
        candidate_config_sha256=definition.config_sha256,
        runtime_root=(
            repository_root / str(raw["future_runtime_root"])
        ).resolve(),
        definition=definition,
        operations=tuple(
            operations
        ),
        references=tuple(
            _reference(candidate)
            for candidate in definition.candidates
            if candidate.is_control
        ),
        execution_contract=dict(execution_contract),
        validation_contract=dict(validation_contract),
    )


def expected_execution_authorization(
    *,
    plan: EvaluationExecutionPlan,
    source_commit: str,
    relevant_source_sha256: str,
) -> Mapping[str, Any]:
    frozen = plan.definition.raw["frozen_contract"]
    provenance_keys = (
        "train_file_sha256",
        "train_content_sha256",
        "validation_file_sha256",
        "validation_content_sha256",
        "sampling_plan_sha256",
        "v2_5_config_sha256",
        "v2_5_final_complete_sha256",
        "v2_5_frozen_manifest_sha256",
        "v2_6_forensic_terminal_sha256",
        "v2_7_runner_config_sha256",
        "v2_7_candidate_config_sha256",
        "v2_7_execution_authorization_sha256",
        "v2_7_aggregate_authorization_sha256",
        "v2_7_aggregate_complete_sha256",
        "v2_7_selection_report_sha256",
    )
    parent_references = {}
    for model_id in plan.definition.model_ids:
        parent = plan.definition.raw["models"][model_id]["parent"]
        parent_references[model_id] = {
            key: parent[key]
            for key in (
                "candidate_id",
                "source_commit",
                "relevant_source_sha256",
                "authorization_sha256",
                "candidate_config_sha256",
                "runner_config_sha256",
                "sampling_plan_sha256",
                "attempt_path",
                "manifest_sha256",
                "complete_sha256",
                "candidate_result_sha256",
                "evaluation_sha256",
                "fit_state_sha256",
                "validation_sample_sha256",
                "checkpoint_path",
                "checkpoint_sha256",
            )
        }
    return {
        "schema_version": (
            "benchmark-v2.8-evaluation-execution-authorization-v1"
        ),
        "status": "AUTHORIZED",
        "source_commit": source_commit,
        "relevant_source_sha256": relevant_source_sha256,
        "runner_config_sha256": plan.runner_config_sha256,
        "candidate_config_sha256": plan.candidate_config_sha256,
        "runtime_root": "artifacts/benchmark_v2_8/candidate_selection",
        "models": list(MODEL_IDS),
        "candidate_ids": [
            operation.candidate_id for operation in plan.operations
        ],
        "checkpoint_hashes": {
            operation.model_id: operation.checkpoint_sha256
            for operation in plan.operations
        },
        "frozen_provenance": {
            key: frozen[key] for key in provenance_keys
        },
        "frozen_provenance_sha256": canonical_sha256(frozen),
        "parent_references": parent_references,
        "counts": {
            "frozen_references": 3,
            "evaluation_only_candidates": 2,
            "training_trajectories": 0,
        },
        "authorization": {
            "frozen_reference": True,
            "evaluation_only_sampling": True,
            "validation_five_guard_evaluation": True,
            "cuda_evaluation": True,
            "gpu_inventory_query": False,
            "training": False,
            "optimizer_updates": False,
            "checkpoint_writes": False,
            "tvae_rerun": False,
            "selection": False,
            "test_split_access": False,
            "fresh_test": False,
            "tstr": False,
            "privacy": False,
            "five_seed_full_run": False,
        },
    }


def validate_execution_authorization(
    *,
    plan: EvaluationExecutionPlan,
    authorization: Mapping[str, Any],
    source_commit: str,
    relevant_source_sha256: str,
) -> None:
    expected = expected_execution_authorization(
        plan=plan,
        source_commit=source_commit,
        relevant_source_sha256=relevant_source_sha256,
    )
    mismatches = {
        key: {"expected": value, "actual": authorization.get(key)}
        for key, value in expected.items()
        if authorization.get(key) != value
    }
    if mismatches or set(authorization) != set(expected):
        raise EvaluationOnlyContractError(
            f"v2.8 evaluation authorization mismatch: {mismatches}"
        )


def validate_checkpoint_provenance(
    plan: EvaluationExecutionPlan,
    operation: EvaluationOperation,
) -> Path:
    expected = [
        item
        for item in plan.operations
        if (item.model_id, item.candidate_id)
        == (operation.model_id, operation.candidate_id)
    ]
    if len(expected) != 1 or expected[0] != operation:
        raise EvaluationOnlyContractError(
            "checkpoint operation differs from the frozen plan"
        )
    checkpoint = (
        plan.repository_root / operation.checkpoint_path
    ).resolve()
    try:
        checkpoint.relative_to(plan.repository_root)
    except ValueError as error:
        raise EvaluationOnlyContractError(
            "checkpoint escapes repository"
        ) from error
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint) != operation.checkpoint_sha256
    ):
        raise EvaluationOnlyContractError(
            "checkpoint provenance hash mismatch"
        )
    return checkpoint


def restore_checkpoint_for_evaluation(
    *,
    plan: EvaluationExecutionPlan,
    operation: EvaluationOperation,
    device: str,
    train: Any,
    loader=None,
) -> Mapping[str, Any]:
    if not device.startswith("cuda:"):
        raise EvaluationOnlyContractError(
            "evaluation restore requires explicit selected CUDA device"
        )
    checkpoint = validate_checkpoint_provenance(plan, operation)
    if loader is None:
        from experiments.candidate_runner_v2_6 import (
            restore_candidate_checkpoint_for_sampling,
        )

        loader = restore_candidate_checkpoint_for_sampling
    bundle = loader(
        checkpoint,
        model_id=operation.model_id,
        device=device,
        train=train,
    )
    if not isinstance(bundle, Mapping) or bundle.get("backend") is None:
        raise EvaluationOnlyContractError(
            "restored checkpoint bundle has no backend"
        )
    return {
        "checkpoint_bundle": bundle,
        "checkpoint_path": operation.checkpoint_path,
        "checkpoint_sha256": operation.checkpoint_sha256,
        "restore_mode": operation.restore_mode,
        "training_calls": 0,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
    }


def validate_evaluation_boundary(
    *,
    plan: EvaluationExecutionPlan,
    data_paths: tuple[Path, ...],
    thresholds: Mapping[str, float],
    sampling_plan_sha256: str,
    selection_requested: bool,
    candidate_overrides: Mapping[str, Any],
) -> None:
    if (
        dict(thresholds) != FROZEN_THRESHOLDS
        or sampling_plan_sha256 != FROZEN_SAMPLING_PLAN_SHA256
        or selection_requested
        or bool(candidate_overrides)
    ):
        raise EvaluationOnlyContractError(
            "threshold, SamplingPlan, selection, or override changed"
        )
    data_root = (
        plan.repository_root
        / "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00"
    ).resolve()
    expected = {
        (data_root / "train.npz").resolve(),
        (data_root / "validation.npz").resolve(),
    }
    resolved = {path.resolve() for path in data_paths}
    if resolved != expected:
        raise EvaluationOnlyContractError(
            "evaluation may read exactly frozen train and validation"
        )
    for path in resolved:
        try:
            path.relative_to(data_root)
        except ValueError as error:
            raise EvaluationOnlyContractError(
                "data path escapes frozen train/validation root"
            ) from error
        if "test" in path.name.lower() or not path.is_file():
            raise EvaluationOnlyContractError(
                "test or missing data access is forbidden"
            )


def evaluation_relevant_source_sha256(
    repository_root: Path,
) -> str:
    digest = hashlib.sha256()
    for relative in RELEVANT_SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise EvaluationOnlyContractError(
                f"v2.8 evaluation source is missing: {relative}"
            )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _current_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()


def evaluation_plan_report(
    plan: EvaluationExecutionPlan,
) -> Mapping[str, Any]:
    return {
        "schema_version": "benchmark-v2.8-evaluation-plan-v1",
        "status": "PASS",
        "mode": "PLAN_ONLY",
        "source_commit": _current_head(plan.repository_root),
        "relevant_source_sha256": evaluation_relevant_source_sha256(
            plan.repository_root
        ),
        "runner_config_sha256": plan.runner_config_sha256,
        "candidate_config_sha256": plan.candidate_config_sha256,
        "runtime_root": str(
            plan.runtime_root.relative_to(plan.repository_root)
        ),
        "counts": {
            "models": 2,
            "frozen_references": 3,
            "evaluation_only_candidates": 2,
            "training_trajectories": 0,
        },
        "operations": [
            {
                "model_id": operation.model_id,
                "candidate_id": operation.candidate_id,
                "factor": operation.factor,
                "checkpoint_path": operation.checkpoint_path,
                "checkpoint_sha256": operation.checkpoint_sha256,
                "restore_mode": operation.restore_mode,
                "optimizer_updates": operation.optimizer_updates,
                "training_calls": operation.training_calls,
                "checkpoint_writes": operation.checkpoint_writes,
            }
            for operation in plan.operations
        ],
        "frozen_references": [
            {
                "model_id": reference.model_id,
                "candidate_id": reference.candidate_id,
            }
            for reference in plan.references
        ],
        "sampling_plan_sha256": FROZEN_SAMPLING_PLAN_SHA256,
        "thresholds": dict(FROZEN_THRESHOLDS),
        "authorization_created": False,
        "runtime_artifact_created": False,
        "execution_authorized": False,
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),
    }


def dry_run_evaluation(
    plan: EvaluationExecutionPlan,
) -> Mapping[str, Any]:
    from experiments.candidate_preparation_runner_v2_8 import (
        build_v28_plan,
        dry_run_v28,
    )

    if sha256_file(plan.runner_config_path) != plan.runner_config_sha256:
        raise EvaluationOnlyContractError(
            "runner config changed after plan construction"
        )
    preparation = dry_run_v28(
        build_v28_plan(plan.definition.config_path)
    )
    checkpoint_hashes = {}
    for operation in plan.operations:
        validate_checkpoint_provenance(plan, operation)
        checkpoint_hashes[operation.model_id] = (
            operation.checkpoint_sha256
        )
    report = evaluation_plan_report(plan)
    return {
        **report,
        "schema_version": "benchmark-v2.8-evaluation-dry-run-v1",
        "mode": "DRY_RUN",
        "frozen_provenance_verified": True,
        "frozen_inventory": preparation["frozen_inventory"],
        "parent_inventory": preparation["parent_inventory"],
        "checkpoint_hashes_verified": checkpoint_hashes,
        "test_split_read": False,
        "fit_state_created": False,
        "authorization_created": False,
        "runtime_artifact_created": False,
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),
    }


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationOnlyContractError(
            f"cannot read JSON provenance: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise EvaluationOnlyContractError(
            f"JSON provenance is not an object: {path}"
        )
    return value


def _candidate_definition(
    plan: EvaluationExecutionPlan,
    operation: EvaluationOperation,
) -> V28Candidate:
    candidates = [
        candidate
        for candidate in plan.definition.candidates
        if (
            candidate.model_id,
            candidate.candidate_id,
        )
        == (
            operation.model_id,
            operation.candidate_id,
        )
    ]
    if len(candidates) != 1 or candidates[0].is_control:
        raise EvaluationOnlyContractError(
            "future evaluation candidate does not resolve exactly once"
        )
    return candidates[0]


def _load_parent_intervention(
    *,
    plan: EvaluationExecutionPlan,
    operation: EvaluationOperation,
):
    from eval.candidate_preparation_v2_7 import (
        load_and_validate_preparation,
    )

    candidate = _candidate_definition(plan, operation)
    parent = candidate.frozen_parent
    attempt = (
        plan.repository_root / str(parent["attempt_path"])
    ).resolve()
    fit_state_path = attempt / "fit_state.json"
    if (
        not fit_state_path.is_file()
        or sha256_file(fit_state_path) != parent["fit_state_sha256"]
    ):
        raise EvaluationOnlyContractError(
            "frozen parent fit-state hash mismatch"
        )
    v27_definition = load_and_validate_preparation(
        plan.repository_root
        / "configs/benchmark_v2/selection_v2_7_source_preparation.yaml"
    )
    matches = [
        value
        for value in v27_definition.candidates
        if (
            value.model_id,
            value.candidate_id,
        )
        == (
            operation.model_id,
            parent["candidate_id"],
        )
    ]
    if len(matches) != 1:
        raise EvaluationOnlyContractError(
            "frozen v2.7 parent definition does not resolve once"
        )
    return matches[0], _read_json_mapping(fit_state_path)


def _sample_restored_bundle(
    bundle: Mapping[str, Any],
    *,
    sampling_plan: Any,
    seed: int,
) -> Any:
    backend = bundle.get("backend")
    if backend is None or not callable(getattr(backend, "sample", None)):
        raise EvaluationOnlyContractError(
            "restored checkpoint backend cannot sample"
        )
    sample = backend.sample(sampling_plan, seed=seed)
    transform = bundle.get("train_only_zscore")
    if transform is not None:
        sample = transform.inverse_synthetic(sample)
    return sample


def _save_synthetic_npz(path: Path, sample: Any) -> None:
    import numpy as np

    if path.exists():
        raise EvaluationOnlyContractError(
            f"append-only sample path exists: {path}"
        )
    np.savez_compressed(
        path,
        **{
            field: getattr(sample, field)
            for field in sample.__dataclass_fields__
        },
    )


def _verify_development_context(
    plan: EvaluationExecutionPlan,
    context: Any,
) -> None:
    frozen = plan.definition.raw["frozen_contract"]
    observed = {
        "train_file_sha256": context.train_file_sha256,
        "train_content_sha256": context.train_content_sha256,
        "validation_file_sha256": context.validation_file_sha256,
        "validation_content_sha256": context.validation_content_sha256,
        "sampling_plan_sha256": context.plan.plan_hash,
    }
    mismatches = {
        key: {"expected": frozen[key], "actual": value}
        for key, value in observed.items()
        if value != frozen[key]
    }
    if mismatches:
        raise EvaluationOnlyContractError(
            f"frozen train/validation provenance changed: {mismatches}"
        )


def _execute_evaluation_child(
    payload: Mapping[str, Any],
    *,
    completion_notifier: Callable[[], None] | None = None,
) -> None:
    import torch

    from eval.evaluation_only_v2_8 import evaluate_validation_guards
    from eval.validation_selection_v2_6 import load_development_context
    from experiments.candidate_runner_v2_6 import (
        load_train_only_candidate_context,
    )
    from generators.evaluation_only_transforms_v2_7 import (
        apply_post_sample_intervention as apply_parent_intervention,
    )
    from generators.evaluation_only_transforms_v2_8 import (
        apply_post_sample_intervention,
        fit_train_only_intervention,
    )

    root = Path(str(payload["repository_root"])).resolve()
    plan = build_evaluation_execution_plan(
        Path(str(payload["runner_config_path"])).resolve()
    )
    operation = next(
        (
            item
            for item in plan.operations
            if item.model_id == payload["model_id"]
            and item.candidate_id == payload["candidate_id"]
        ),
        None,
    )
    if operation is None:
        raise EvaluationOnlyContractError(
            "evaluation child received an unknown operation"
        )
    source_commit = str(payload["source_commit"])
    relevant_hash = str(payload["relevant_source_sha256"])
    if (
        _current_head(root) != source_commit
        or evaluation_relevant_source_sha256(root) != relevant_hash
    ):
        raise EvaluationOnlyContractError(
            "source changed after authorization"
        )
    authorization_path = Path(
        str(payload["authorization_path"])
    ).resolve()
    authorization = _read_json_mapping(authorization_path)
    validate_execution_authorization(
        plan=plan,
        authorization=authorization,
        source_commit=source_commit,
        relevant_source_sha256=relevant_hash,
    )
    device = str(payload["device"])
    if not device.startswith("cuda:"):
        raise EvaluationOnlyContractError(
            "authorized evaluation requires an explicit CUDA device"
        )
    torch.cuda.set_device(torch.device(device))
    train_context = load_train_only_candidate_context(
        repository_root=root,
        development_manifest_path=(
            root / "configs/benchmark_v2/development_data_v2_6.yaml"
        ),
    )
    _verify_development_context(plan, train_context)
    data_root = (
        root
        / "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00"
    )
    validate_evaluation_boundary(
        plan=plan,
        data_paths=(
            data_root / "train.npz",
            data_root / "validation.npz",
        ),
        thresholds=FROZEN_THRESHOLDS,
        sampling_plan_sha256=train_context.plan.plan_hash,
        selection_requested=False,
        candidate_overrides={},
    )
    candidate = _candidate_definition(plan, operation)
    restored = restore_checkpoint_for_evaluation(
        plan=plan,
        operation=operation,
        device=device,
        train=train_context.train,
    )
    bundle = restored["checkpoint_bundle"]
    calibration_sample = _sample_restored_bundle(
        bundle,
        sampling_plan=train_context.plan,
        seed=int(plan.execution_contract["fit_sample_seed"]),
    )
    fit_state = fit_train_only_intervention(
        candidate=candidate,
        train=train_context.train,
        calibration_sample=calibration_sample,
        source_commit=source_commit,
        config_sha256=plan.candidate_config_sha256,
        train_file_sha256=train_context.train_file_sha256,
        train_content_sha256=train_context.train_content_sha256,
        sampling_plan_sha256=train_context.plan.plan_hash,
    )
    sample = _sample_restored_bundle(
        bundle,
        sampling_plan=train_context.plan,
        seed=int(plan.execution_contract["validation_sample_seed"]),
    )
    parent_candidate, parent_fit_state = _load_parent_intervention(
        plan=plan,
        operation=operation,
    )
    sample = apply_parent_intervention(
        candidate=parent_candidate,
        sample=sample,
        fit_state=parent_fit_state,
    )
    sample = apply_post_sample_intervention(
        candidate=candidate,
        sample=sample,
        fit_state=fit_state,
    )
    # Validation is opened only after train-only fitting and sampling finish.
    context = load_development_context(
        repository_root=root,
        manifest_path=(
            root / "configs/benchmark_v2/development_data_v2_6.yaml"
        ),
    )
    _verify_development_context(plan, context)
    guards = evaluate_validation_guards(
        train=context.train,
        validation=context.validation,
        sample=sample,
        sampling_plan=context.plan,
        tau=context.tau,
        receiver_categories=context.receiver_categories,
        thresholds=FROZEN_THRESHOLDS,
    )
    attempt = Path(str(payload["attempt_path"])).resolve()
    try:
        attempt.relative_to(plan.runtime_root.resolve())
    except ValueError as error:
        raise EvaluationOnlyContractError(
            "evaluation artifact path escapes v2.8 runtime"
        ) from error
    store = EvaluationOnlyArtifactStore(root)
    sample_path = attempt / "validation_sample.npz"
    _save_synthetic_npz(sample_path, sample)
    store.write_json(attempt / "fit_state.json", fit_state)
    store.write_json(
        attempt / "evaluation.json",
        {
            **guards,
            "model_id": operation.model_id,
            "candidate_id": operation.candidate_id,
        },
    )
    result = {
        "schema_version": (
            "benchmark-v2.8-evaluation-only-candidate-result-v1"
        ),
        "status": "COMPLETE",
        "source_commit": source_commit,
        "relevant_source_sha256": relevant_hash,
        "runner_config_sha256": plan.runner_config_sha256,
        "candidate_config_sha256": plan.candidate_config_sha256,
        "authorization_sha256": sha256_file(authorization_path),
        "model_id": operation.model_id,
        "candidate_id": operation.candidate_id,
        "factor": operation.factor,
        "checkpoint_path": operation.checkpoint_path,
        "checkpoint_sha256": operation.checkpoint_sha256,
        "restore_mode": operation.restore_mode,
        "fit_state_sha256": fit_state["state_sha256"],
        "validation_sample_path": str(sample_path.relative_to(root)),
        "validation_sample_sha256": sha256_file(sample_path),
        "guard_status": guards["status"],
        "all_five_guards_pass": guards["all_five_guards_pass"],
        "training_calls": 0,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "sampling_calls": 2,
        "evaluation_calls": 1,
        "selection_calls": 0,
        "test_split_read": False,
    }
    store.write_json(attempt / "candidate_result.json", result)
    store.write_json(
        attempt / "COMPLETE.json",
        {
            "schema_version": (
                "benchmark-v2.8-evaluation-attempt-terminal-v1"
            ),
            "status": "COMPLETE",
            "candidate_result_sha256": sha256_file(
                attempt / "candidate_result.json"
            ),
            "evaluation_sha256": sha256_file(
                attempt / "evaluation.json"
            ),
            "fit_state_sha256": sha256_file(
                attempt / "fit_state.json"
            ),
            "validation_sample_sha256": sha256_file(sample_path),
        },
    )
    if completion_notifier is not None:
        completion_notifier()


def _evaluation_child_entry(
    payload: Mapping[str, Any],
    result_queue: Any,
) -> None:
    completion_sent = False

    def notify_complete() -> None:
        nonlocal completion_sent
        result_queue.put({"status": "COMPLETE"})
        completion_sent = True

    try:
        _execute_evaluation_child(
            payload,
            completion_notifier=notify_complete,
        )
        if not completion_sent:
            notify_complete()
    except Exception as error:
        result_queue.put(
            {
                "status": "FAILED",
                "failure_class": "code",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )


def execute_model_worker(
    *,
    plan: EvaluationExecutionPlan,
    authorization_path: Path,
    model_id: str,
    device: str,
) -> Mapping[str, Any]:
    if model_id not in MODEL_IDS:
        raise EvaluationOnlyContractError(
            f"unknown v2.8 evaluation model: {model_id}"
        )
    authorization_path = authorization_path.resolve()
    try:
        relative_authorization = authorization_path.relative_to(
            plan.repository_root
        )
    except ValueError as error:
        raise EvaluationOnlyContractError(
            "authorization must be inside the repository"
        ) from error
    if (
        not authorization_path.is_file()
        or "authorization" not in relative_authorization.as_posix()
        or not device.startswith("cuda:")
    ):
        raise EvaluationOnlyContractError(
            "authorization path or explicit CUDA device is invalid"
        )
    source_commit = _current_head(plan.repository_root)
    relevant_hash = evaluation_relevant_source_sha256(
        plan.repository_root
    )
    authorization = _read_json_mapping(authorization_path)
    validate_execution_authorization(
        plan=plan,
        authorization=authorization,
        source_commit=source_commit,
        relevant_source_sha256=relevant_hash,
    )
    if sha256_file(plan.runner_config_path) != plan.runner_config_sha256:
        raise EvaluationOnlyContractError(
            "runner config changed after authorization"
        )
    operation = next(
        item for item in plan.operations if item.model_id == model_id
    )
    validate_checkpoint_provenance(plan, operation)
    provenance = canonical_sha256(
        {
            "source_commit": source_commit,
            "relevant_source_sha256": relevant_hash,
            "runner_config_sha256": plan.runner_config_sha256,
            "candidate_config_sha256": plan.candidate_config_sha256,
            "authorization_sha256": sha256_file(authorization_path),
            "model_id": model_id,
            "candidate_id": operation.candidate_id,
        }
    )
    store = EvaluationOnlyArtifactStore(plan.repository_root)
    store.claim_model(
        model_id=model_id,
        provenance_sha256=provenance,
    )
    attempt = store.create_evaluation_attempt(
        model_id=model_id,
        candidate_id=operation.candidate_id,
        seed=int(plan.execution_contract["validation_sample_seed"]),
    )
    store.write_json(
        attempt / "manifest.json",
        {
            "schema_version": (
                "benchmark-v2.8-evaluation-attempt-manifest-v1"
            ),
            "status": "RUNNING",
            "source_commit": source_commit,
            "relevant_source_sha256": relevant_hash,
            "runner_config_sha256": plan.runner_config_sha256,
            "candidate_config_sha256": plan.candidate_config_sha256,
            "authorization_path": str(relative_authorization),
            "authorization_sha256": sha256_file(authorization_path),
            "model_id": model_id,
            "candidate_id": operation.candidate_id,
            "factor": operation.factor,
            "checkpoint_path": operation.checkpoint_path,
            "checkpoint_sha256": operation.checkpoint_sha256,
            "sampling_plan_sha256": FROZEN_SAMPLING_PLAN_SHA256,
            "training_calls": 0,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
            "test_split_read": False,
            "device": device,
        },
    )
    payload = {
        "repository_root": str(plan.repository_root),
        "runner_config_path": str(plan.runner_config_path),
        "authorization_path": str(authorization_path),
        "attempt_path": str(attempt),
        "source_commit": source_commit,
        "relevant_source_sha256": relevant_hash,
        "model_id": model_id,
        "candidate_id": operation.candidate_id,
        "device": device,
    }
    from experiments.evaluation_only_runner_v2_7 import (
        run_evaluation_child_until_terminal,
    )

    terminal = run_evaluation_child_until_terminal(
        child_target=_evaluation_child_entry,
        payload=payload,
        attempt_path=attempt,
        max_wall_seconds=float(
            plan.execution_contract["max_wall_seconds_per_candidate"]
        ),
        termination_grace_seconds=float(
            plan.execution_contract["termination_grace_seconds"]
        ),
    )
    worker_root = store.root / "workers" / model_id
    if terminal["status"] == "COMPLETE":
        marker = worker_root / "WORKER_COMPLETE.json"
        marker_status = "COMPLETE"
    else:
        failed = {
            "schema_version": (
                "benchmark-v2.8-evaluation-attempt-terminal-v1"
            ),
            "status": "FAILED",
            "failure_class": terminal.get("failure_class", "code"),
            "details": dict(terminal),
        }
        if not (attempt / "FAILED.json").exists():
            store.write_json(attempt / "FAILED.json", failed)
        marker = worker_root / "WORKER_FAILED.json"
        marker_status = "FAILED"
    store.write_json(
        marker,
        {
            "schema_version": "benchmark-v2.8-evaluation-worker-v1",
            "status": marker_status,
            "model_id": model_id,
            "candidate_id": operation.candidate_id,
            "provenance_sha256": provenance,
            "attempt": str(attempt.relative_to(plan.repository_root)),
            "terminal": dict(terminal),
            "training_calls": 0,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
            "selection_calls": 0,
            "test_split_read": False,
        },
    )
    return {
        "status": marker_status,
        "model_id": model_id,
        "candidate_id": operation.candidate_id,
        "worker_terminal_path": str(
            marker.relative_to(plan.repository_root)
        ),
    }
