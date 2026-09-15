from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


MODEL_IDS = (
    "ctgan_separate_class",
    "tvae_separate_class",
    "cof_seqgen",
)
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
EXPECTED_CANDIDATES = {
    "ctgan_separate_class": {
        "ctgan_v27_c00_frozen_standard": "none",
        "ctgan_v27_c01_amount_quantile_inverse": "amount_inverse_map",
        "ctgan_v27_c02_categorical_logit": (
            "categorical_logit_calibration"
        ),
    },
    "tvae_separate_class": {
        "tvae_v27_c00_frozen_temperature_0_75": "none",
        "tvae_v27_c01_amount_inverse_decoder": "amount_inverse_decoder",
        "tvae_v27_c02_bounded_temperature": "categorical_temperature",
    },
    "cof_seqgen": {
        "cof_v27_c00_frozen_variance_residual": "none",
        "cof_v27_c01_empirical_residual": "amount_residual_sampler",
        "cof_v27_c02_gap_logit_bias": "gap_logit_calibration",
    },
}
EXPECTED_FACTOR_PARAMETERS = {
    "ctgan_v27_c01_amount_quantile_inverse": {
        "rule": "class_conditional_monotone_empirical_quantile_inverse",
        "quantile_grid_size": 257,
        "interpolation": "linear_monotone",
        "tail_rule": "clamp_to_train_extrema",
    },
    "ctgan_v27_c02_categorical_logit": {
        "rule": "class_channel_marginal_log_probability_offset",
        "additive_smoothing": 0.5,
        "logit_offset_clip": [-2.0, 2.0],
        "base_distribution": "frozen_train_plan_output",
    },
    "tvae_v27_c01_amount_inverse_decoder": {
        "rule": "class_conditional_monotone_empirical_quantile_inverse",
        "quantile_grid_size": 257,
        "interpolation": "linear_monotone",
        "tail_rule": "clamp_to_train_extrema",
    },
    "tvae_v27_c02_bounded_temperature": {
        "rule": "train_only_bounded_grid",
        "fixed_grid": [0.5, 0.625, 0.75, 0.875, 1.0],
        "objective": (
            "minimize_max_gap_receiver_train_frequency_error"
        ),
        "tiebreak": "closest_to_0_75_then_lower",
    },
    "cof_v27_c01_empirical_residual": {
        "rule": "class_conditional_centered_empirical_quantile_residual",
        "quantile_grid_size": 257,
        "resampling": "deterministic_inverse_ecdf",
        "variance_guard": "preserve_train_residual_variance",
    },
    "cof_v27_c02_gap_logit_bias": {
        "rule": "class_conditional_gap_marginal_log_probability_offset",
        "additive_smoothing": 0.5,
        "logit_offset_clip": [-2.0, 2.0],
        "base_distribution": "frozen_train_plan_output",
    },
}
EXPECTED_FORENSIC = {
    "evaluator_or_selection_runner_defect": "REFUTED",
    "numeric_sampling_decoder_path_limit": "SUPPORTED",
    "current_objective_candidate_space_limit": "SUPPORTED",
    "threshold_relaxation": "FORBIDDEN",
    "test_based_calibration": "FORBIDDEN",
    "result_contingent_candidate_change": "FORBIDDEN",
}
EXPECTED_FIT_STATE_CONTRACT = {
    "fit_split": "train",
    "fit_input": "valid_train_rows_and_train_only_sampling_plan",
    "validation_refit": "FORBIDDEN",
    "test_access": "FORBIDDEN",
    "state_destination": "checkpoint_or_candidate_artifact",
    "hash_algorithm": "sha256_canonical_json",
    "source_config_data_plan_provenance_required": True,
}
EXPECTED_BENCHMARK_CONTRACT = {
    "scenario": "joint_semimarkov_v2b",
    "kappa": 1.0,
    "endpoint": "continuous_association_recovery_error",
    "dgp": "benchmark_v2_joint_semimarkov_v2b_kappa_1_00_frozen",
    "c2_rule": "unchanged_v2_6_primary_three_model_rule",
}
EXPECTED_SELECTION_RULE = {
    "eligibility": "all_five_row_marginal_guards_PASS",
    "primary_objective": "minimize_max_amount_ks_gap_ks",
    "secondary_objective": "minimize_sum_amount_ks_gap_ks",
    "final_tiebreaker": "lexicographic_candidate_id",
    "no_pass_action": "PROHIBIT_V2_7_PRIMARY_C2_FULL_RUN",
}
EXPECTED_BENCHMARK_CONTRACT = {
    "scenario": "joint_semimarkov_v2b",
    "kappa": 1.0,
    "endpoint": "continuous_association_recovery_error",
    "dgp": "benchmark_v2_joint_semimarkov_v2b_kappa_1_00_frozen",
    "c2_rule": "unchanged_v2_6_primary_three_model_rule",
}
EXPECTED_SELECTION_RULE = {
    "eligibility": "all_five_row_marginal_guards_PASS",
    "primary_objective": "minimize_max_amount_ks_gap_ks",
    "secondary_objective": "minimize_sum_amount_ks_gap_ks",
    "final_tiebreaker": "lexicographic_candidate_id",
    "no_pass_action": "PROHIBIT_V2_7_PRIMARY_C2_FULL_RUN",
}


class V27PreparationContractError(RuntimeError):
    pass


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class V27Candidate:
    model_id: str
    candidate_id: str
    execution_kind: str
    factor: str
    changed_dimensions: tuple[str, ...]
    baseline_dimensions: Mapping[str, Any]
    effective_dimensions: Mapping[str, Any]
    fit_parameters: Mapping[str, Any]
    frozen_control: Mapping[str, Any]
    future_training_required: bool
    future_sampling_required: bool

    @property
    def is_control(self) -> bool:
        return self.execution_kind == "reuse_frozen_control"


@dataclass(frozen=True)
class V27PreparationPlan:
    config_path: Path
    config_sha256: str
    repository_root: Path
    future_artifact_root: Path
    candidates: tuple[V27Candidate, ...]
    thresholds: Mapping[str, float]
    sampling_plan_sha256: str
    fit_state_contract: Mapping[str, Any]
    frozen_contract: Mapping[str, Any]
    parent_forensic_commit: str

    @property
    def model_ids(self) -> tuple[str, ...]:
        return MODEL_IDS


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise V27PreparationContractError(
            f"cannot read v2.7 preparation config: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise V27PreparationContractError(
            "v2.7 preparation config root must be a mapping"
        )
    return value


def _changed_dimensions(
    baseline: Mapping[str, Any],
    effective: Mapping[str, Any],
) -> tuple[str, ...]:
    if set(baseline) != set(effective):
        raise V27PreparationContractError(
            "candidate effective dimensions changed their key set"
        )
    return tuple(
        key
        for key in baseline
        if canonical_sha256(baseline[key])
        != canonical_sha256(effective[key])
    )


def validate_preparation_definition(
    raw: Mapping[str, Any],
) -> tuple[V27Candidate, ...]:
    required_false = (
        "execution_authorized",
        "authorization_creation_authorized",
        "gpu_execution_authorized",
        "candidate_training_authorized",
        "candidate_sampling_authorized",
        "validation_selection_authorized",
        "fresh_test_authorized",
        "tstr_authorized",
        "privacy_authorized",
        "five_seed_full_run_authorized",
    )
    if (
        raw.get("schema_version")
        != "benchmark-v2.7-source-preparation-v1"
        or raw.get("mode") != "SOURCE_ONLY_PREPARATION"
        or raw.get("test_split_access") != "FORBIDDEN"
        or any(raw.get(key) is not False for key in required_false)
    ):
        raise V27PreparationContractError(
            "v2.7 source-only authorization boundary changed"
        )
    if raw.get("frozen_forensic_conclusions") != EXPECTED_FORENSIC:
        raise V27PreparationContractError(
            "frozen forensic conclusion changed"
        )
    frozen = raw.get("frozen_contract", {})
    if (
        frozen.get("thresholds") != FROZEN_THRESHOLDS
        or frozen.get("sampling_plan_sha256")
        != FROZEN_SAMPLING_PLAN_SHA256
    ):
        raise V27PreparationContractError(
            "five-guard threshold or SamplingPlan changed"
        )
    if any(
        frozen.get(key) != value
        for key, value in EXPECTED_BENCHMARK_CONTRACT.items()
    ):
        raise V27PreparationContractError(
            "frozen benchmark contract changed"
        )
    if raw.get("selection_rule") != EXPECTED_SELECTION_RULE:
        raise V27PreparationContractError(
            "frozen selection rule changed"
        )
    if any(
        frozen.get(key) != value
        for key, value in EXPECTED_BENCHMARK_CONTRACT.items()
    ):
        raise V27PreparationContractError(
            "frozen benchmark contract changed"
        )
    if raw.get("selection_rule") != EXPECTED_SELECTION_RULE:
        raise V27PreparationContractError(
            "frozen selection rule changed"
        )
    if raw.get("fit_state_contract") != EXPECTED_FIT_STATE_CONTRACT:
        raise V27PreparationContractError(
            "train-only fit-state contract changed"
        )
    models = raw.get("models")
    if not isinstance(models, Mapping) or tuple(models) != MODEL_IDS:
        raise V27PreparationContractError(
            "v2.7 model family changed"
        )
    candidates: list[V27Candidate] = []
    for model_id in MODEL_IDS:
        model = models[model_id]
        baseline = model.get("baseline_dimensions")
        frozen_control = model.get("frozen_control")
        rows = model.get("candidates")
        if (
            not isinstance(baseline, Mapping)
            or not isinstance(frozen_control, Mapping)
            or not isinstance(rows, list)
            or len(rows) != 3
        ):
            raise V27PreparationContractError(
                f"invalid model candidate definition: {model_id}"
            )
        expected = EXPECTED_CANDIDATES[model_id]
        if {
            str(row.get("candidate_id")): str(row.get("factor"))
            for row in rows
        } != expected:
            raise V27PreparationContractError(
                f"v2.7 candidate family changed: {model_id}"
            )
        for row in rows:
            effective = row.get("effective_dimensions")
            if not isinstance(effective, Mapping):
                raise V27PreparationContractError(
                    "candidate dimensions must be a mapping"
                )
            changed = _changed_dimensions(baseline, effective)
            factor = str(row["factor"])
            execution_kind = str(row.get("execution_kind"))
            is_control = execution_kind == "reuse_frozen_control"
            if is_control:
                valid = (
                    factor == "none"
                    and changed == ()
                    and row.get("future_training_required") is False
                    and row.get("future_sampling_required") is False
                    and "fit_parameters" not in row
                )
            else:
                valid = (
                    execution_kind
                    == "future_evaluate_frozen_checkpoint"
                    and changed == (factor,)
                    and row.get("future_training_required") is False
                    and row.get("future_sampling_required") is True
                    and isinstance(row.get("fit_parameters"), Mapping)
                )
            if not valid:
                raise V27PreparationContractError(
                    "candidate is not an exact single-factor operation: "
                    f"{model_id}/{row.get('candidate_id')}"
                )
            if (
                not is_control
                and row.get("fit_parameters")
                != EXPECTED_FACTOR_PARAMETERS[row["candidate_id"]]
            ):
                raise V27PreparationContractError(
                    "preregistered factor parameters changed: "
                    f"{model_id}/{row['candidate_id']}"
                )
            candidates.append(
                V27Candidate(
                    model_id=model_id,
                    candidate_id=str(row["candidate_id"]),
                    execution_kind=execution_kind,
                    factor=factor,
                    changed_dimensions=changed,
                    baseline_dimensions=dict(baseline),
                    effective_dimensions=dict(effective),
                    fit_parameters=dict(row.get("fit_parameters", {})),
                    frozen_control=dict(frozen_control),
                    future_training_required=bool(
                        row["future_training_required"]
                    ),
                    future_sampling_required=bool(
                        row["future_sampling_required"]
                    ),
                )
            )
    return tuple(candidates)


def load_and_validate_preparation(
    config_path: Path,
) -> V27PreparationPlan:
    config_path = config_path.resolve()
    raw = _read_yaml(config_path)
    candidates = validate_preparation_definition(raw)
    repository_root = config_path.parents[2]
    return V27PreparationPlan(
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        repository_root=repository_root,
        future_artifact_root=(
            repository_root / str(raw["future_artifact_root"])
        ).resolve(),
        candidates=candidates,
        thresholds=dict(raw["frozen_contract"]["thresholds"]),
        sampling_plan_sha256=str(
            raw["frozen_contract"]["sampling_plan_sha256"]
        ),
        fit_state_contract=dict(raw["fit_state_contract"]),
        frozen_contract=dict(raw["frozen_contract"]),
        parent_forensic_commit=str(raw["parent_forensic_commit"]),
    )


def build_fit_state_record(
    *,
    candidate: V27Candidate,
    parameters: Mapping[str, Any],
    source_commit: str,
    config_sha256: str,
    train_file_sha256: str,
    train_content_sha256: str,
    sampling_plan_sha256: str,
    destination: str,
) -> Mapping[str, Any]:
    if candidate.is_control:
        raise V27PreparationContractError(
            "frozen controls cannot create fit state"
        )
    if destination not in {"checkpoint", "candidate_artifact"}:
        raise V27PreparationContractError(
            "fit state destination must be checkpoint or candidate artifact"
        )
    provenance = {
        "source_commit": source_commit,
        "config_sha256": config_sha256,
        "train_file_sha256": train_file_sha256,
        "train_content_sha256": train_content_sha256,
        "sampling_plan_sha256": sampling_plan_sha256,
    }
    if (
        len(source_commit) != 40
        or any(
            len(value) != 64
            for key, value in provenance.items()
            if key != "source_commit"
        )
    ):
        raise V27PreparationContractError(
            "fit state provenance hash shape is invalid"
        )
    state = {
        "schema_version": "benchmark-v2.7-train-fit-state-v1",
        "candidate_id": candidate.candidate_id,
        "model_id": candidate.model_id,
        "factor": candidate.factor,
        "fit_split": "train",
        "fit_input": "valid_train_rows_and_train_only_sampling_plan",
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "destination": destination,
        "parameters": dict(parameters),
        "provenance": provenance,
    }
    return {
        **state,
        "state_sha256": canonical_sha256(state),
    }


def validate_fit_state_record(
    *,
    record: Mapping[str, Any],
    candidate: V27Candidate,
    source_commit: str,
    config_sha256: str,
    train_file_sha256: str,
    train_content_sha256: str,
    sampling_plan_sha256: str,
) -> None:
    expected_provenance = {
        "source_commit": source_commit,
        "config_sha256": config_sha256,
        "train_file_sha256": train_file_sha256,
        "train_content_sha256": train_content_sha256,
        "sampling_plan_sha256": sampling_plan_sha256,
    }
    if (
        record.get("candidate_id") != candidate.candidate_id
        or record.get("model_id") != candidate.model_id
        or record.get("factor") != candidate.factor
        or record.get("fit_split") != "train"
        or record.get("validation_rows_used") != 0
        or record.get("test_rows_used") != 0
        or record.get("destination")
        not in {"checkpoint", "candidate_artifact"}
        or record.get("provenance") != expected_provenance
    ):
        raise V27PreparationContractError(
            "fit state contract or provenance mismatch"
        )
    state = {
        key: value
        for key, value in record.items()
        if key != "state_sha256"
    }
    if record.get("state_sha256") != canonical_sha256(state):
        raise V27PreparationContractError(
            "fit state hash mismatch"
        )


def validate_candidate_io_path(
    *,
    repository_root: Path,
    path: Path,
    access: str,
) -> Path:
    repository_root = repository_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError as error:
        raise V27PreparationContractError(
            "candidate path escapes the repository"
        ) from error
    lower_parts = tuple(part.lower() for part in relative.parts)
    if any("test" in part for part in lower_parts):
        raise V27PreparationContractError(
            "test and fresh-test paths are forbidden"
        )
    if access == "read":
        if relative.name not in {"train.npz", "validation.npz"}:
            raise V27PreparationContractError(
                "candidate preparation may read only train/validation data"
            )
        return resolved
    if access != "write":
        raise V27PreparationContractError(
            "candidate path access must be read or write"
        )
    future_root = (
        repository_root / "artifacts/benchmark_v2_7/candidate_selection"
    ).resolve()
    try:
        resolved.relative_to(future_root)
    except ValueError as error:
        raise V27PreparationContractError(
            "v2.5/v2.6 artifacts and frozen data are read-only"
        ) from error
    return resolved


class AppendOnlyV27Store:
    def __init__(self, repository_root: Path):
        self.repository_root = repository_root.resolve()
        self.root = (
            self.repository_root
            / "artifacts/benchmark_v2_7/candidate_selection"
        )

    def claim_worker(
        self,
        *,
        model_id: str,
        provenance_sha256: str,
    ) -> Path:
        if model_id not in MODEL_IDS or len(provenance_sha256) != 64:
            raise V27PreparationContractError(
                "invalid worker ownership claim"
            )
        worker_root = self.root / "workers" / model_id
        worker_root.mkdir(parents=True, exist_ok=True)
        claim = worker_root / "ownership.lock"
        value = {
            "schema_version": "benchmark-v2.7-worker-ownership-v1",
            "model_id": model_id,
            "provenance_sha256": provenance_sha256,
        }
        try:
            with claim.open("x", encoding="utf-8") as handle:
                json.dump(
                    value,
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
        except FileExistsError as error:
            raise V27PreparationContractError(
                f"model worker is already owned: {model_id}"
            ) from error
        return claim

    def create_candidate_attempt(
        self,
        *,
        model_id: str,
        candidate_id: str,
        seed: int,
    ) -> Path:
        if (
            model_id not in MODEL_IDS
            or candidate_id not in EXPECTED_CANDIDATES[model_id]
            or not (
                self.root
                / "workers"
                / model_id
                / "ownership.lock"
            ).is_file()
        ):
            raise V27PreparationContractError(
                "candidate attempt requires its exclusive model owner"
            )
        parent = (
            self.root
            / "candidates"
            / model_id
            / candidate_id
            / f"seed_{seed}"
        )
        parent.mkdir(parents=True, exist_ok=True)
        for attempt_number in range(1, 1000):
            attempt = parent / f"attempt_{attempt_number:03d}"
            try:
                attempt.mkdir()
            except FileExistsError:
                continue
            return attempt
        raise V27PreparationContractError(
            "append-only candidate attempt namespace exhausted"
        )

    def write_json(
        self,
        path: Path,
        value: Mapping[str, Any],
    ) -> None:
        validate_candidate_io_path(
            repository_root=self.repository_root,
            path=path,
            access="write",
        )
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
            raise V27PreparationContractError(
                f"append-only artifact already exists: {path}"
            ) from error
