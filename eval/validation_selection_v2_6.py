from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.model_guards_v2_5 import row_guard_statistics
from experiments.provenance_v2_5 import hash_batch
from generators.contracts_v2_5 import validate_synthetic_contract
from generators.sampling_plan import SamplingPlan


SCHEMA_VERSION = "benchmark-v2.6-validation-selection-v1"
DEVELOPMENT_SCHEMA = "benchmark-v2.6-development-data-v1"
CANDIDATE_RESULT_SCHEMA = "benchmark-v2.6-candidate-result-v1"
FREEZE_SCHEMA = "benchmark-v2.6-selection-freeze-v1"
MODEL_IDS = (
    "ctgan_separate_class",
    "tvae_separate_class",
    "neural_sequence",
    "cof_seqgen",
)
PRIMARY_MODEL_IDS = (
    "ctgan_separate_class",
    "tvae_separate_class",
    "cof_seqgen",
)
SECONDARY_MODEL_IDS = ("neural_sequence",)
FIXED_REFERENCE_IDS = ("empirical_iid",)
METRIC_KEYS = (
    "amount_ks",
    "gap_ks",
    "amount_abs_standardized_label_effect",
    "gap_abs_standardized_label_effect",
    "receiver_max_abs_signed_frequency",
)
CONTINUOUS_KS_KEYS = ("amount_ks", "gap_ks")
FORBIDDEN_SPLIT_NAMES = {
    "test",
    "test.npz",
    "test_split",
    "heldout_test",
}


class SelectionContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class DevelopmentContext:
    train: SequenceBatch
    validation: SequenceBatch
    plan: SamplingPlan
    tau: np.ndarray
    receiver_categories: int
    manifest_sha256: str
    train_file_sha256: str
    validation_file_sha256: str
    train_content_sha256: str
    validation_content_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SelectionContractError(f"cannot read YAML: {path}") from error
    if not isinstance(value, Mapping):
        raise SelectionContractError(f"YAML root is not a mapping: {path}")
    return value


def read_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionContractError(f"cannot read JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise SelectionContractError(f"JSON root is not a mapping: {path}")
    return value


def _assert_not_test_path(path: Path, *, role: str) -> None:
    lowered = {part.lower() for part in path.parts}
    if lowered & FORBIDDEN_SPLIT_NAMES:
        raise SelectionContractError(
            f"{role} resolves to a forbidden test-split path: {path}"
        )


def _resolve_read_path(
    repository_root: Path,
    value: str,
    *,
    role: str,
    within: Path | None = None,
) -> Path:
    candidate = Path(value)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (repository_root / candidate).resolve()
    )
    _assert_not_test_path(resolved, role=role)
    if within is not None:
        within = within.resolve()
        if resolved != within and within not in resolved.parents:
            raise SelectionContractError(
                f"{role} escapes its allowed root: {resolved}"
            )
    if not resolved.is_file():
        raise SelectionContractError(f"{role} is missing: {resolved}")
    return resolved


def _load_sequence_batch(path: Path) -> SequenceBatch:
    _assert_not_test_path(path, role="development split")
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {
                field: archive[field]
                for field in SequenceBatch.__dataclass_fields__
            }
    except (OSError, ValueError, KeyError) as error:
        raise SelectionContractError(
            f"cannot load development split: {path}"
        ) from error
    return SequenceBatch(**values)


def _load_synthetic_batch(path: Path) -> SyntheticBatch:
    _assert_not_test_path(path, role="candidate validation sample")
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {
                field: archive[field]
                for field in SyntheticBatch.__dataclass_fields__
            }
    except (OSError, ValueError, KeyError) as error:
        raise SelectionContractError(
            f"cannot load candidate validation sample: {path}"
        ) from error
    return SyntheticBatch(**values)


def validate_selection_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise SelectionContractError("wrong v2.6 selection schema")
    if config.get("mode") != "PREPARATION_ONLY":
        raise SelectionContractError("v2.6 config must be preparation-only")
    if config.get("test_split_access") != "FORBIDDEN":
        raise SelectionContractError("test split access must be FORBIDDEN")
    model_order = tuple(config.get("model_order", ()))
    if model_order != MODEL_IDS:
        raise SelectionContractError("v2.6 model order is not frozen")
    roles = config.get("model_roles")
    expected_roles = {
        "primary_c2_learned": list(PRIMARY_MODEL_IDS),
        "secondary": list(SECONDARY_MODEL_IDS),
        "fixed_reference": list(FIXED_REFERENCE_IDS),
    }
    if roles != expected_roles:
        raise SelectionContractError(
            "v2.6 primary, secondary, and reference roles are not frozen"
        )
    models = config.get("models")
    if not isinstance(models, Mapping) or set(models) != set(MODEL_IDS):
        raise SelectionContractError("v2.6 config must define four models")

    budget = config.get("selection_budget")
    if not isinstance(budget, Mapping):
        raise SelectionContractError("selection budget is missing")
    candidate_count = int(budget.get("candidates_per_model", -1))
    requested_updates = int(
        budget.get("requested_updates_per_trajectory", -1)
    )
    max_wall_seconds = float(
        budget.get("max_gpu_wall_seconds_per_trajectory", -1)
    )
    checkpoints = tuple(
        int(value)
        for value in budget.get("eligible_checkpoint_steps", ())
    )
    if (
        candidate_count < 1
        or requested_updates < 1
        or max_wall_seconds <= 0
        or checkpoints != (10_000, 20_000)
        or requested_updates != 20_000
        or int(budget.get("evaluation_candidates_per_model", -1)) != 4
        or int(budget.get("evaluation_candidates_all_models", -1)) != 16
        or int(budget.get("training_trajectories_per_model", -1)) != 3
        or int(budget.get("training_trajectories_all_models", -1)) != 12
        or float(budget.get("max_gpu_hours_per_trajectory", -1)) != 2.0
        or float(budget.get("max_gpu_hours_per_model", -1)) != 6.0
        or float(budget.get("max_gpu_hours_all_models", -1)) != 24.0
        or int(budget.get("checkpoint_interval_steps", -1)) != 100
    ):
        raise SelectionContractError("selection budget differs from preregistration")
    seeds = tuple(int(seed) for seed in budget.get("selection_seeds", ()))
    if seeds != (2601,):
        raise SelectionContractError("selection seed must be exactly [2601]")

    allowed_candidate_keys = {
        "candidate_id",
        "numeric_representation",
        "channel_loss_weights",
        "sampling_rule",
        "training_schedule",
        "implementation_contract",
    }
    definitions: dict[str, list[Mapping[str, Any]]] = {}
    trajectory_count = 0
    expected_adapter_paths = {
        "ctgan_separate_class": (
            "generators.candidate_model_backends_v2_6."
            "ConditionalCTGANCandidateV26"
        ),
        "tvae_separate_class": (
            "generators.candidate_model_backends_v2_6."
            "ConditionalTVAECandidateV26"
        ),
        "neural_sequence": (
            "generators.candidate_model_backends_v2_6."
            "NeuralSequenceCandidateV26"
        ),
        "cof_seqgen": (
            "generators.candidate_model_backends_v2_6."
            "CoFSeqGenCandidateV26"
        ),
    }
    for model_id in MODEL_IDS:
        model = models[model_id]
        if not isinstance(model, Mapping):
            raise SelectionContractError(f"invalid model block: {model_id}")
        base_config = model.get("adapter_base_config")
        if (
            not isinstance(base_config, Mapping)
            or base_config.get("adapter") != expected_adapter_paths[model_id]
            or any(
                key in base_config
                for key in (
                    "validation_batch",
                    "validation_labels",
                    "validation_lengths",
                    "test_path",
                    "fresh_test_path",
                )
            )
        ):
            raise SelectionContractError(
                f"{model_id} v2.6 adapter base config is invalid"
            )
        candidates = model.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != candidate_count:
            raise SelectionContractError(
                f"{model_id} must define exactly {candidate_count} candidates"
            )
        ids: list[str] = []
        trajectory_ids: list[str] = []
        definitions[model_id] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise SelectionContractError(
                    f"invalid candidate in {model_id}"
                )
            if set(candidate) != allowed_candidate_keys:
                raise SelectionContractError(
                    f"candidate keys differ from frozen dimensions: {model_id}"
                )
            candidate_id = str(candidate["candidate_id"])
            ids.append(candidate_id)
            schedule = candidate["training_schedule"]
            if not isinstance(schedule, Mapping):
                raise SelectionContractError("candidate schedule is missing")
            if int(schedule.get("requested_updates", -1)) != requested_updates:
                raise SelectionContractError(
                    f"{model_id}/{candidate_id} update budget differs"
                )
            trajectory_id = str(schedule.get("shared_trajectory_id", ""))
            if not trajectory_id or Path(trajectory_id).name != trajectory_id:
                raise SelectionContractError(
                    f"{model_id}/{candidate_id} trajectory ID is invalid"
                )
            trajectory_ids.append(trajectory_id)
            if float(schedule.get("max_gpu_wall_seconds", -1)) != (
                max_wall_seconds
            ):
                raise SelectionContractError(
                    f"{model_id}/{candidate_id} wall budget differs"
                )
            checkpoint = int(schedule.get("selection_checkpoint_step", -1))
            if checkpoint not in checkpoints:
                raise SelectionContractError(
                    f"{model_id}/{candidate_id} checkpoint is not eligible"
                )
            if bool(schedule.get("early_stopping", True)):
                raise SelectionContractError("early stopping must be false")
            if int(schedule.get("checkpoint_interval_steps", -1)) != 100:
                raise SelectionContractError(
                    "candidate checkpoint interval must be 100"
                )
            implementation = candidate["implementation_contract"]
            if (
                not isinstance(implementation, Mapping)
                or implementation.get("v2_5_files_modified") is not False
                or implementation.get(
                    "required_before_candidate_execution"
                )
                is not True
            ):
                raise SelectionContractError(
                    "candidate implementation must preserve v2.5 and "
                    "remain required before execution"
                )
            class_updates = schedule.get("class_updates")
            class_wall = schedule.get("class_wall_seconds")
            if model_id in {
                "ctgan_separate_class",
                "tvae_separate_class",
            }:
                if (
                    not isinstance(class_updates, Mapping)
                    or sum(int(value) for value in class_updates.values())
                    != requested_updates
                    or set(class_updates) != {"0", "1"}
                    or not isinstance(class_wall, Mapping)
                    or sum(float(value) for value in class_wall.values())
                    != max_wall_seconds
                    or set(class_wall) != {"0", "1"}
                ):
                    raise SelectionContractError(
                        f"{model_id}/{candidate_id} class budget differs"
                    )
            elif class_updates is not None or class_wall is not None:
                raise SelectionContractError(
                    f"{model_id}/{candidate_id} has unexpected class budget"
                )
            definitions[model_id].append(candidate)
        if len(set(ids)) != candidate_count:
            raise SelectionContractError(
                f"duplicate candidate id in {model_id}"
            )
        by_id = {
            str(candidate["candidate_id"]): candidate
            for candidate in definitions[model_id]
        }
        native_ids = {
            "c00_native_checkpoint_10000",
            "c01_native_checkpoint_20000",
        }
        if not native_ids <= set(by_id):
            raise SelectionContractError(
                f"{model_id} must retain c00/c01 native checkpoint candidates"
            )
        c00 = by_id["c00_native_checkpoint_10000"]
        c01 = by_id["c01_native_checkpoint_20000"]
        if (
            c00["training_schedule"]["shared_trajectory_id"]
            != c01["training_schedule"]["shared_trajectory_id"]
        ):
            raise SelectionContractError(
                f"{model_id} c00/c01 must share one native trajectory"
            )
        for key in (
            "numeric_representation",
            "channel_loss_weights",
            "sampling_rule",
            "implementation_contract",
        ):
            if c00[key] != c01[key]:
                raise SelectionContractError(
                    f"{model_id} c00/c01 differ outside checkpoint selection"
                )
        c00_schedule = dict(c00["training_schedule"])
        c01_schedule = dict(c01["training_schedule"])
        c00_step = c00_schedule.pop("selection_checkpoint_step")
        c01_step = c01_schedule.pop("selection_checkpoint_step")
        if (
            c00_schedule != c01_schedule
            or c00_step != 10_000
            or c01_step != 20_000
        ):
            raise SelectionContractError(
                f"{model_id} c00/c01 resource schedules are not shared"
            )
        if len(set(trajectory_ids)) != 3:
            raise SelectionContractError(
                f"{model_id} must define exactly three training trajectories"
            )
        trajectory_count += len(set(trajectory_ids))

    gate = config.get("validation_gate")
    if not isinstance(gate, Mapping):
        raise SelectionContractError("validation gate is missing")
    thresholds = gate.get("thresholds")
    if not isinstance(thresholds, Mapping) or not all(
        key in thresholds for key in METRIC_KEYS
    ):
        raise SelectionContractError("all five validation thresholds are required")
    if any(float(thresholds[key]) < 0 for key in METRIC_KEYS):
        raise SelectionContractError("validation thresholds must be nonnegative")
    rule = gate.get("selection_rule")
    expected_rule = {
        "eligibility": "all_five_row_marginal_guards_PASS",
        "primary_objective": "minimize_max_amount_ks_gap_ks",
        "secondary_objective": "minimize_sum_amount_ks_gap_ks",
        "final_tiebreaker": "lexicographic_candidate_id",
        "primary_no_pass_action": "PROHIBIT_V2_6_PRIMARY_C2_FULL_RUN",
        "secondary_no_pass_action": (
            "RECORD_NOT_EVALUABLE_WITHOUT_BLOCKING_PRIMARY_C2"
        ),
    }
    if rule != expected_rule:
        raise SelectionContractError("candidate selection rule is not frozen")
    return {
        "candidate_count": candidate_count,
        "requested_updates": requested_updates,
        "max_gpu_wall_seconds": max_wall_seconds,
        "eligible_checkpoint_steps": checkpoints,
        "selection_seed": seeds[0],
        "training_trajectory_count": trajectory_count,
        "evaluation_candidate_count": candidate_count * len(MODEL_IDS),
        "definitions": definitions,
        "thresholds": {
            key: float(thresholds[key])
            for key in METRIC_KEYS
        },
    }


def selection_readiness(
    selections: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    selection_ids = set(selections)
    if (
        selection_ids != set(MODEL_IDS)
        and selection_ids != set(PRIMARY_MODEL_IDS)
    ):
        raise SelectionContractError("selection family is incomplete")
    blocking_primary = [
        model_id
        for model_id in PRIMARY_MODEL_IDS
        if selections[model_id].get("status") != "SELECTED"
    ]
    primary_ready = not blocking_primary
    neural_selected = (
        "neural_sequence" in selections
        and selections["neural_sequence"].get("status") == "SELECTED"
    )
    if not primary_ready:
        status = "SELECTION_FAILED_PRIMARY_NO_FULL_RUN"
    elif neural_selected:
        status = "SELECTION_PASS"
    else:
        status = "PRIMARY_SELECTION_PASS_SECONDARY_NOT_EVALUABLE"
    return {
        "status": status,
        "primary_c2_selection_ready": primary_ready,
        "blocking_primary_models": blocking_primary,
        "neural_secondary_comparison": (
            "EVALUABLE_AWAITING_AUTHORIZATION"
            if neural_selected
            else "NOT_EVALUABLE"
        ),
        "all_models_selected": primary_ready and neural_selected,
    }


def load_development_context(
    *,
    repository_root: Path,
    manifest_path: Path,
) -> DevelopmentContext:
    repository_root = repository_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = read_yaml_mapping(manifest_path)
    if manifest.get("schema_version") != DEVELOPMENT_SCHEMA:
        raise SelectionContractError("wrong development-data schema")
    if manifest.get("status") != "FROZEN_FOR_SELECTION":
        raise SelectionContractError("development data is not frozen")
    if manifest.get("test_split_access") != "FORBIDDEN":
        raise SelectionContractError("development manifest permits test access")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {
        "train",
        "validation",
    }:
        raise SelectionContractError(
            "development manifest must contain train and validation only"
        )
    loaded: dict[str, SequenceBatch] = {}
    file_hashes: dict[str, str] = {}
    content_hashes: dict[str, str] = {}
    for split in ("train", "validation"):
        record = splits[split]
        if not isinstance(record, Mapping):
            raise SelectionContractError(f"malformed {split} record")
        path = _resolve_read_path(
            repository_root,
            str(record["path"]),
            role=f"{split} split",
        )
        file_hash = sha256_file(path)
        if file_hash != record.get("file_sha256"):
            raise SelectionContractError(f"{split} file hash mismatch")
        batch = _load_sequence_batch(path)
        content_hash = hash_batch(batch)
        if content_hash != record.get("content_sha256"):
            raise SelectionContractError(f"{split} content hash mismatch")
        if (
            int(record.get("entities", -1)) != len(batch.lengths)
            or int(record.get("valid_rows", -1))
            != int(batch.valid_mask.sum())
        ):
            raise SelectionContractError(f"{split} shape inventory mismatch")
        loaded[split] = batch
        file_hashes[split] = file_hash
        content_hashes[split] = content_hash

    plan_record = manifest.get("selection_sampling_plan")
    if not isinstance(plan_record, Mapping):
        raise SelectionContractError("selection SamplingPlan is missing")
    if plan_record.get("fit_split") != "train":
        raise SelectionContractError("selection plan must be fit on train")
    entity_count = int(plan_record.get("entity_count", -1))
    if entity_count != len(loaded["validation"].lengths):
        raise SelectionContractError(
            "selection plan entity count must equal validation entity count"
        )
    plan = SamplingPlan.from_train_policy(
        loaded["train"],
        entity_count=entity_count,
        seed=int(plan_record["seed"]),
    )
    if plan.plan_hash != plan_record.get("content_sha256"):
        raise SelectionContractError("selection SamplingPlan hash mismatch")

    tau = np.asarray(manifest.get("tau"), dtype=float)
    if tau.ndim != 1 or not len(tau) or not np.isfinite(tau).all():
        raise SelectionContractError("development tau must be finite 1-D")
    train_valid = loaded["train"].valid_mask
    expected_bins = int(loaded["train"].dt_bin[train_valid].max()) + 1
    if len(tau) != expected_bins:
        raise SelectionContractError("tau does not match train gap support")
    receiver_categories = int(
        loaded["train"].x_cat[..., 0][train_valid].max()
    ) + 1
    return DevelopmentContext(
        train=loaded["train"],
        validation=loaded["validation"],
        plan=plan,
        tau=tau,
        receiver_categories=receiver_categories,
        manifest_sha256=sha256_file(manifest_path),
        train_file_sha256=file_hashes["train"],
        validation_file_sha256=file_hashes["validation"],
        train_content_sha256=content_hashes["train"],
        validation_content_sha256=content_hashes["validation"],
    )


def _candidate_attempt_path(
    candidate_root: Path,
    *,
    model_id: str,
    candidate_id: str,
    seed: int,
) -> Path:
    return (
        candidate_root
        / model_id
        / candidate_id
        / f"seed_{seed}"
        / "attempt_001"
    )


def evaluate_candidate_artifact(
    *,
    repository_root: Path,
    candidate_root: Path,
    model_id: str,
    candidate: Mapping[str, Any],
    selection_seed: int,
    config_sha256: str,
    context: DevelopmentContext,
    thresholds: Mapping[str, float],
) -> Mapping[str, Any]:
    if model_id not in MODEL_IDS:
        raise SelectionContractError(f"unknown v2.6 model: {model_id}")
    candidate_id = str(candidate["candidate_id"])
    attempt = _candidate_attempt_path(
        candidate_root.resolve(),
        model_id=model_id,
        candidate_id=candidate_id,
        seed=selection_seed,
    ).resolve()
    if candidate_root.resolve() not in attempt.parents:
        raise SelectionContractError("candidate attempt escapes root")
    _assert_not_test_path(attempt, role="candidate attempt")
    manifest_path = attempt / "candidate_result.json"
    manifest = read_json_mapping(manifest_path)
    expected = {
        "schema_version": CANDIDATE_RESULT_SCHEMA,
        "status": "COMPLETE",
        "model_id": model_id,
        "candidate_id": candidate_id,
        "selection_seed": selection_seed,
        "selection_config_sha256": config_sha256,
        "candidate_definition_sha256": canonical_sha256(candidate),
        "shared_trajectory_id": candidate["training_schedule"][
            "shared_trajectory_id"
        ],
        "train_file_sha256": context.train_file_sha256,
        "train_content_sha256": context.train_content_sha256,
        "validation_file_sha256": context.validation_file_sha256,
        "validation_content_sha256": context.validation_content_sha256,
        "development_manifest_sha256": context.manifest_sha256,
        "selection_plan_sha256": context.plan.plan_hash,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise SelectionContractError(
            f"candidate result provenance mismatch: {mismatches}"
        )
    source_commit = str(manifest.get("source_commit", ""))
    source_hash = str(manifest.get("relevant_source_sha256", ""))
    if len(source_commit) != 40 or len(source_hash) != 64:
        raise SelectionContractError("candidate source provenance is invalid")
    training_source_commit = str(
        manifest.get("training_source_commit", source_commit)
    )
    training_source_hash = str(
        manifest.get("training_relevant_source_sha256", source_hash)
    )
    if (
        len(training_source_commit) != 40
        or len(training_source_hash) != 64
    ):
        raise SelectionContractError(
            "candidate training source provenance is invalid"
        )
    if "training_source_commit" in manifest and (
        manifest.get("evaluation_source_commit") != source_commit
        or manifest.get("evaluation_relevant_source_sha256")
        != source_hash
        or not str(
            manifest.get("continuation_authorization_sha256", "")
        )
    ):
        raise SelectionContractError(
            "candidate continuation provenance is incomplete"
        )

    schedule = candidate["training_schedule"]
    if int(manifest.get("requested_updates", -1)) != int(
        schedule["requested_updates"]
    ):
        raise SelectionContractError("candidate requested updates mismatch")
    if int(manifest.get("actual_updates", -1)) != int(
        schedule["requested_updates"]
    ):
        raise SelectionContractError("candidate did not finish requested updates")
    if int(manifest.get("sampled_checkpoint_step", -1)) != int(
        schedule["selection_checkpoint_step"]
    ):
        raise SelectionContractError("candidate sampled wrong checkpoint")
    if bool(manifest.get("wall_cap_reached", True)):
        raise SelectionContractError("candidate reached wall cap")
    if manifest.get("trajectory_completed_requested_updates") is not True:
        raise SelectionContractError(
            "candidate trajectory completion is not recorded"
        )

    trajectory_manifest_path = _resolve_read_path(
        repository_root,
        str(manifest["trajectory_manifest_path"]),
        role="shared trajectory manifest",
        within=candidate_root,
    )
    trajectory_complete_path = _resolve_read_path(
        repository_root,
        str(manifest["trajectory_complete_path"]),
        role="shared trajectory completion",
        within=candidate_root,
    )
    if (
        sha256_file(trajectory_manifest_path)
        != manifest.get("trajectory_manifest_sha256")
        or sha256_file(trajectory_complete_path)
        != manifest.get("trajectory_complete_sha256")
    ):
        raise SelectionContractError(
            "shared trajectory artifact hash mismatch"
        )
    trajectory_manifest = read_json_mapping(trajectory_manifest_path)
    trajectory_complete = read_json_mapping(trajectory_complete_path)
    trajectory_expected = {
        "source_commit": training_source_commit,
        "relevant_source_sha256": training_source_hash,
        "selection_config_sha256": config_sha256,
        "development_manifest_sha256": context.manifest_sha256,
        "train_file_sha256": context.train_file_sha256,
        "train_content_sha256": context.train_content_sha256,
        "validation_file_sha256": context.validation_file_sha256,
        "validation_content_sha256": context.validation_content_sha256,
        "selection_plan_sha256": context.plan.plan_hash,
        "model_id": model_id,
        "shared_trajectory_id": schedule["shared_trajectory_id"],
        "requested_updates": schedule["requested_updates"],
    }
    if any(
        trajectory_manifest.get(key) != value
        for key, value in trajectory_expected.items()
    ):
        raise SelectionContractError(
            "shared trajectory provenance mismatch"
        )
    if (
        trajectory_complete.get("status") != "COMPLETE"
        or trajectory_complete.get("shared_trajectory_id")
        != schedule["shared_trajectory_id"]
        or int(trajectory_complete.get("requested_updates", -1))
        != int(schedule["requested_updates"])
        or int(trajectory_complete.get("actual_updates", -1))
        != int(schedule["requested_updates"])
        or trajectory_complete.get("completed_hard_cap_contract") is not True
        or trajectory_complete.get("wall_cap_reached") is not False
    ):
        raise SelectionContractError(
            "shared trajectory did not complete its frozen contract"
        )

    sample_path = _resolve_read_path(
        repository_root,
        str(manifest["validation_sample_path"]),
        role="candidate validation sample",
        within=candidate_root,
    )
    checkpoint_path = _resolve_read_path(
        repository_root,
        str(manifest["checkpoint_path"]),
        role="candidate checkpoint",
        within=candidate_root,
    )
    if sha256_file(sample_path) != manifest.get("validation_sample_sha256"):
        raise SelectionContractError("candidate validation sample hash mismatch")
    if sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256"):
        raise SelectionContractError("candidate checkpoint hash mismatch")
    sample = _load_synthetic_batch(sample_path)
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
            and statistics[key] <= float(thresholds[key])
            else "FAIL"
        )
        for key in METRIC_KEYS
    }
    all_pass = all(value == "PASS" for value in checks.values())
    return {
        "model_id": model_id,
        "candidate_id": candidate_id,
        "selection_seed": selection_seed,
        "status": "PASS" if all_pass else "FAIL",
        "all_five_guards_pass": all_pass,
        "statistics": statistics,
        "thresholds": {
            key: float(thresholds[key])
            for key in METRIC_KEYS
        },
        "checks": checks,
        "continuous_ks_max": max(
            statistics[key] for key in CONTINUOUS_KS_KEYS
        ),
        "continuous_ks_sum": sum(
            statistics[key] for key in CONTINUOUS_KS_KEYS
        ),
        "sample_contract": sample_contract,
        "source_commit": source_commit,
        "relevant_source_sha256": source_hash,
        "training_source_commit": training_source_commit,
        "training_relevant_source_sha256": training_source_hash,
        "candidate_definition_sha256": canonical_sha256(candidate),
        "candidate_result_sha256": sha256_file(manifest_path),
        "shared_trajectory_id": schedule["shared_trajectory_id"],
        "trajectory_manifest_sha256": sha256_file(
            trajectory_manifest_path
        ),
        "trajectory_complete_sha256": sha256_file(
            trajectory_complete_path
        ),
        "validation_sample_sha256": sha256_file(sample_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "sampled_checkpoint_step": int(
            manifest["sampled_checkpoint_step"]
        ),
    }


def choose_candidate(
    results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not results:
        raise SelectionContractError("candidate result family is empty")
    model_ids = {str(result["model_id"]) for result in results}
    if len(model_ids) != 1:
        raise SelectionContractError("candidate family mixes models")
    eligible = [
        result
        for result in results
        if result.get("all_five_guards_pass") is True
    ]
    if not eligible:
        return {
            "model_id": next(iter(model_ids)),
            "status": "NO_PASSING_CANDIDATE",
            "selected_candidate_id": None,
            "v2_6_full_run_permitted": False,
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
        "model_id": selected["model_id"],
        "status": "SELECTED",
        "selected_candidate_id": selected["candidate_id"],
        "v2_6_full_run_permitted": False,
        "reason": (
            "selection complete; source/config/development provenance must "
            "be frozen and separately authorized"
        ),
        "continuous_ks_max": selected["continuous_ks_max"],
        "continuous_ks_sum": selected["continuous_ks_sum"],
        "candidate_definition_sha256": selected[
            "candidate_definition_sha256"
        ],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "sampled_checkpoint_step": selected["sampled_checkpoint_step"],
        "source_commit": selected["source_commit"],
        "relevant_source_sha256": selected[
            "relevant_source_sha256"
        ],
    }


def run_validation_selection(
    *,
    repository_root: Path,
    config_path: Path,
    development_manifest_path: Path,
    candidate_root: Path,
    evaluated_model_ids: Sequence[str] = MODEL_IDS,
    unavailable_model_ids: Sequence[str] = (),
) -> Mapping[str, Any]:
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    candidate_root = candidate_root.resolve()
    _assert_not_test_path(candidate_root, role="candidate root")
    config = read_yaml_mapping(config_path)
    validated = validate_selection_config(config)
    evaluated_ids = tuple(evaluated_model_ids)
    unavailable_ids = tuple(unavailable_model_ids)
    if (
        len(set(evaluated_ids)) != len(evaluated_ids)
        or len(set(unavailable_ids)) != len(unavailable_ids)
        or set(evaluated_ids) & set(unavailable_ids)
        or set(evaluated_ids) | set(unavailable_ids) != set(MODEL_IDS)
        or not set(unavailable_ids) <= set(SECONDARY_MODEL_IDS)
        or not set(PRIMARY_MODEL_IDS) <= set(evaluated_ids)
    ):
        raise SelectionContractError(
            "validation selection model scope is invalid"
        )
    context = load_development_context(
        repository_root=repository_root,
        manifest_path=development_manifest_path,
    )
    config_hash = sha256_file(config_path)
    results: list[Mapping[str, Any]] = []
    selections: dict[str, Mapping[str, Any]] = {}
    for model_id in evaluated_ids:
        model_results = [
            evaluate_candidate_artifact(
                repository_root=repository_root,
                candidate_root=candidate_root,
                model_id=model_id,
                candidate=candidate,
                selection_seed=validated["selection_seed"],
                config_sha256=config_hash,
                context=context,
                thresholds=validated["thresholds"],
            )
            for candidate in validated["definitions"][model_id]
        ]
        results.extend(model_results)
        selections[model_id] = choose_candidate(model_results)
    for model_id in unavailable_ids:
        selections[model_id] = {
            "model_id": model_id,
            "status": "NOT_EVALUABLE",
            "selected_candidate_id": None,
            "v2_6_full_run_permitted": False,
            "reason": (
                "secondary model was outside the authorized primary "
                "candidate execution scope"
            ),
        }
    readiness = selection_readiness(selections)
    source_pairs = sorted(
        {
        (
            str(result["source_commit"]),
            str(result["relevant_source_sha256"]),
        )
        for result in results
        }
    )
    single_source = len(source_pairs) == 1
    source_commit = source_pairs[0][0] if single_source else None
    source_hash = source_pairs[0][1] if single_source else None
    return {
        "schema_version": SCHEMA_VERSION,
        "status": readiness["status"],
        "test_split_read": False,
        "v2_5_test_artifacts_used_for_selection": False,
        "v2_6_full_run_permitted": False,
        "requires_separate_user_authorization": True,
        "selection_config_sha256": config_hash,
        "development_manifest_sha256": context.manifest_sha256,
        "train_file_sha256": context.train_file_sha256,
        "validation_file_sha256": context.validation_file_sha256,
        "train_content_sha256": context.train_content_sha256,
        "validation_content_sha256": context.validation_content_sha256,
        "selection_plan_sha256": context.plan.plan_hash,
        "candidate_source_commit": source_commit,
        "candidate_relevant_source_sha256": source_hash,
        "candidate_source_states": [
            {
                "source_commit": pair[0],
                "relevant_source_sha256": pair[1],
            }
            for pair in source_pairs
        ],
        "mixed_candidate_source_provenance": not single_source,
        "candidate_results": results,
        "model_selections": selections,
        "evaluated_model_ids": list(evaluated_ids),
        "unavailable_model_ids": list(unavailable_ids),
        "primary_model_ids": list(PRIMARY_MODEL_IDS),
        "secondary_model_ids": list(SECONDARY_MODEL_IDS),
        "fixed_reference_ids": list(FIXED_REFERENCE_IDS),
        "primary_c2_selection_ready": readiness[
            "primary_c2_selection_ready"
        ],
        "blocking_primary_models": readiness["blocking_primary_models"],
        "neural_secondary_comparison": readiness[
            "neural_secondary_comparison"
        ],
        "all_models_selected": readiness["all_models_selected"],
        "no_pass_action": (
            None
            if readiness["primary_c2_selection_ready"]
            else "PROHIBIT_V2_6_PRIMARY_C2_FULL_RUN_AND_REPORT_CAUSE"
        ),
    }


def build_selection_freeze(
    selection_report: Mapping[str, Any],
) -> Mapping[str, Any]:
    if (
        selection_report.get("status")
        not in {
            "SELECTION_PASS",
            "PRIMARY_SELECTION_PASS_SECONDARY_NOT_EVALUABLE",
        }
        or selection_report.get("primary_c2_selection_ready") is not True
        or selection_report.get("test_split_read") is not False
    ):
        raise SelectionContractError(
            "cannot freeze an incomplete or failed selection"
        )
    selections = selection_report["model_selections"]
    if not isinstance(selections, Mapping) or set(selections) != set(MODEL_IDS):
        raise SelectionContractError("selection family is incomplete")
    selected_model_ids = [
        model_id
        for model_id in MODEL_IDS
        if selections[model_id].get("status") == "SELECTED"
    ]
    if not set(PRIMARY_MODEL_IDS) <= set(selected_model_ids):
        raise SelectionContractError(
            "cannot freeze without all primary C2 candidates"
        )
    return {
        "schema_version": FREEZE_SCHEMA,
        "status": "FROZEN_AWAITING_SEPARATE_AUTHORIZATION",
        "selected_candidates": {
            model_id: {
                key: selections[model_id][key]
                for key in (
                    "selected_candidate_id",
                    "candidate_definition_sha256",
                    "checkpoint_sha256",
                    "sampled_checkpoint_step",
                )
            }
            for model_id in selected_model_ids
        },
        "model_selection_status": {
            model_id: selections[model_id]["status"]
            for model_id in MODEL_IDS
        },
        "primary_c2_selection_ready": True,
        "neural_secondary_comparison": selection_report[
            "neural_secondary_comparison"
        ],
        "selection_config_sha256": selection_report[
            "selection_config_sha256"
        ],
        "candidate_source_commit": selection_report[
            "candidate_source_commit"
        ],
        "candidate_relevant_source_sha256": selection_report[
            "candidate_relevant_source_sha256"
        ],
        "candidate_source_states": selection_report.get(
            "candidate_source_states",
            [
                {
                    "source_commit": selection_report[
                        "candidate_source_commit"
                    ],
                    "relevant_source_sha256": selection_report[
                        "candidate_relevant_source_sha256"
                    ],
                }
            ],
        ),
        "mixed_candidate_source_provenance": selection_report.get(
            "mixed_candidate_source_provenance",
            False,
        ),
        "development_manifest_sha256": selection_report[
            "development_manifest_sha256"
        ],
        "train_file_sha256": selection_report["train_file_sha256"],
        "validation_file_sha256": selection_report[
            "validation_file_sha256"
        ],
        "train_content_sha256": selection_report[
            "train_content_sha256"
        ],
        "validation_content_sha256": selection_report[
            "validation_content_sha256"
        ],
        "selection_plan_sha256": selection_report[
            "selection_plan_sha256"
        ],
        "test_split_hash": None,
        "test_split_read": False,
        "fresh_test_authorized": False,
        "five_seed_full_experiment_authorized": False,
        "requires_separate_user_authorization": True,
    }


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise SelectionContractError(
            f"selection output already exists: {path}"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
