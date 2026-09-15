from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import yaml

from benchmarks.types import SyntheticBatch


SCHEMA_VERSION = "benchmark-v2.6-single-factor-amendment-v1"
MODEL_IDS = (
    "ctgan_separate_class",
    "tvae_separate_class",
    "cof_seqgen",
)
EXECUTION_KINDS = {
    "reuse_frozen_control",
    "evaluate_existing_checkpoint",
    "train_new_trajectory",
}
FROZEN_THRESHOLDS = {
    "amount_ks": 0.006081138155655141,
    "gap_ks": 0.006387882975686154,
    "amount_abs_standardized_label_effect": 0.0363693454591819,
    "gap_abs_standardized_label_effect": 0.051540527275560376,
    "receiver_max_abs_signed_frequency": 0.02,
}
FROZEN_HASHES = {
    "selection_config_sha256": (
        "0884a0144f74f6317ae9e636c0cf5657dedac21ff12cdf545a6b6e5e29be4c4d"
    ),
    "development_manifest_sha256": (
        "31ddf45dcf7b4f982ed990ec94281c3e6bad0454ec3182e2434e42358006ade5"
    ),
    "train_file_sha256": (
        "c67a6fce4593317e11e1b6f36397bfb38f9607916bf8f5b32329d8fa99b700a8"
    ),
    "train_content_sha256": (
        "0c03f179930cd4d89ccc8a8b66283e620313e16d15fd53f0674c133bcb80c09d"
    ),
    "validation_file_sha256": (
        "68c15c05c55742bfce0f22560ffe8378d032a7a9fd9a1072e80ec8da5340fba5"
    ),
    "validation_content_sha256": (
        "aa1576b4ccae6a2ca30d0472f0782e1231ecc61ab3d224590a772dc3448b9e66"
    ),
    "sampling_plan_sha256": (
        "862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27"
    ),
}
EXPECTED_CANDIDATES = {
    "ctgan_separate_class": {
        "ctgan_sf_c00_frozen_control": ("reuse_frozen_control", "none"),
        "ctgan_sf_c01_shared_transformer": (
            "train_new_trajectory",
            "transformer_scope",
        ),
        "ctgan_sf_c02_temperature_only": (
            "evaluate_existing_checkpoint",
            "categorical_temperature",
        ),
    },
    "tvae_separate_class": {
        "tvae_sf_c00_frozen_control": ("reuse_frozen_control", "none"),
        "tvae_sf_c01_categorical_decode_only": (
            "evaluate_existing_checkpoint",
            "categorical_decode",
        ),
        "tvae_sf_c02_channel_weight_only": (
            "train_new_trajectory",
            "channel_weights",
        ),
    },
    "cof_seqgen": {
        "cof_sf_c00_frozen_control": ("reuse_frozen_control", "none"),
        "cof_sf_c01_noise_prediction": (
            "train_new_trajectory",
            "amount_diffusion_parameterization",
        ),
        "cof_sf_c02_variance_preserving_residual": (
            "evaluate_existing_checkpoint",
            "amount_residual_sampling",
        ),
    },
}
EXPECTED_DIAGNOSTICS = {
    "ctgan_separate_class": (
        "amount_mean_by_class",
        "amount_std_by_class",
        "amount_signed_standardized_class_effect",
        "gap_frequency_by_class",
        "gap_signed_standardized_class_effect",
        "receiver_frequency_by_class",
        "receiver_signed_frequency_by_category",
    ),
    "tvae_separate_class": (
        "gap_raw_frequency_by_bin",
        "receiver_raw_frequency_by_category",
        "gap_max_category_frequency",
        "receiver_max_category_frequency",
        "gap_frequency_hhi",
        "receiver_frequency_hhi",
        "gap_frequency_by_class",
        "receiver_frequency_by_class",
    ),
    "cof_seqgen": (
        "generated_amount_mean",
        "generated_amount_std",
        "generated_amount_q01",
        "generated_amount_q05",
        "generated_amount_q25",
        "generated_amount_q50",
        "generated_amount_q75",
        "generated_amount_q95",
        "generated_amount_q99",
    ),
}
EXPECTED_BASELINES = {
    "ctgan_separate_class": {
        "class_handling": "separate_y0_y1_generators",
        "transformer_scope": "per_class_train_fitted",
        "numeric_representation": "native_then_bayesian_gmm",
        "categorical_temperature": 0.2,
        "requested_updates": 20_000,
    },
    "tvae_separate_class": {
        "class_handling": "separate_y0_y1_generators",
        "transformer_scope": "per_class_train_fitted",
        "numeric_representation": "native_then_bayesian_gmm",
        "categorical_decode": "upstream_argmax_inverse",
        "channel_weights": {
            "amount": 1.0,
            "gap": 1.0,
            "receiver": 1.0,
        },
        "latent_scale": 1.0,
        "requested_updates": 20_000,
    },
    "cof_seqgen": {
        "architecture": "frozen_v2_6_seq_denoiser",
        "discrete_gap_receiver_path": "frozen_mask_feedback_argmax",
        "numeric_representation": "native_raw_amount",
        "amount_diffusion_parameterization": (
            "clean_x0_mse_matching_ddim"
        ),
        "amount_residual_sampling": "none",
        "requested_updates": 20_000,
    },
}
EXPECTED_INTERVENTION_VALUES = {
    "ctgan_sf_c01_shared_transformer": (
        "transformer_scope",
        "shared_all_train_rows_fitted_once",
    ),
    "ctgan_sf_c02_temperature_only": (
        "categorical_temperature",
        0.5,
    ),
    "tvae_sf_c01_categorical_decode_only": (
        "categorical_decode",
        "temperature_multinomial_gap_receiver_0.75",
    ),
    "tvae_sf_c02_channel_weight_only": (
        "channel_weights",
        {"amount": 2.0, "gap": 2.0, "receiver": 1.0},
    ),
    "cof_sf_c01_noise_prediction": (
        "amount_diffusion_parameterization",
        "epsilon_mse_matching_reverse_diffusion",
    ),
    "cof_sf_c02_variance_preserving_residual": (
        "amount_residual_sampling",
        "train_fitted_zero_mean_variance_deficit",
    ),
}
EXPECTED_TRAJECTORIES = {
    "ctgan_sf_c01_shared_transformer": (
        "ctgan_shared_transformer_seed_2601"
    ),
    "tvae_sf_c02_channel_weight_only": (
        "tvae_channel_weight_seed_2601"
    ),
    "cof_sf_c01_noise_prediction": (
        "cof_noise_prediction_seed_2601"
    ),
}
EXPECTED_RESIDUAL_CONTRACT = {
    "fit_split": "train",
    "rule": "add_zero_mean_gaussian_variance_deficit",
    "variance_target": "valid_train_amount_variance",
    "base_variance_source": "train_only_sampling_plan_sample",
    "variance": "max(train_variance_minus_base_variance, 0)",
    "validation_refit": "FORBIDDEN",
}


class SingleFactorContractError(RuntimeError):
    pass


def _is_below(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def validate_amendment_io_path(
    *,
    repository_root: Path,
    path: Path,
    role: str,
    access: str,
) -> Path:
    repository_root = repository_root.resolve()
    resolved = path.resolve()
    if not _is_below(resolved, repository_root):
        raise SingleFactorContractError(
            f"{role} escapes the repository"
        )
    relative = resolved.relative_to(repository_root)
    lowered = {part.lower() for part in relative.parts}
    if lowered & {
        "test",
        "test.npz",
        "test_split",
        "fresh_test",
        "fresh-test",
        "heldout_test",
    }:
        raise SingleFactorContractError(
            f"{role} accesses a forbidden test path"
        )
    if access == "read":
        if not resolved.is_file():
            raise SingleFactorContractError(
                f"{role} read input is missing"
            )
        if (
            relative.parts
            and relative.parts[0].lower() == "data"
            and resolved.name.lower() not in {
                "train.npz",
                "validation.npz",
            }
        ):
            raise SingleFactorContractError(
                f"{role} is not a frozen train/validation split"
            )
        return resolved
    if access != "write":
        raise ValueError("access must be read or write")
    output_root = (
        repository_root
        / "artifacts/benchmark_v2_6/selection_single_factor"
    )
    if not _is_below(resolved, output_root):
        raise SingleFactorContractError(
            f"{role} write is outside the append-only amendment root"
        )
    return resolved


class AppendOnlyAmendmentStore:
    """Exclusive-create store for a future separately authorized execution."""

    _SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.root = (
            self.repository_root
            / "artifacts/benchmark_v2_6/selection_single_factor"
        )

    def create_candidate_attempt(
        self,
        *,
        model_id: str,
        candidate_id: str,
        seed: int,
    ) -> Path:
        if (
            model_id not in MODEL_IDS
            or not self._SAFE_ID.fullmatch(candidate_id)
            or seed != 2601
        ):
            raise SingleFactorContractError(
                "candidate attempt identity is not preregistered"
            )
        seed_root = (
            self.root
            / "candidates"
            / model_id
            / candidate_id
            / f"seed_{seed}"
        )
        validate_amendment_io_path(
            repository_root=self.repository_root,
            path=seed_root,
            role="candidate attempt",
            access="write",
        )
        seed_root.mkdir(parents=True, exist_ok=True)
        for attempt_number in range(1, 1000):
            attempt = seed_root / f"attempt_{attempt_number:03d}"
            try:
                attempt.mkdir()
            except FileExistsError:
                continue
            return attempt
        raise SingleFactorContractError("candidate attempt space exhausted")

    def write_json(
        self,
        path: Path,
        value: Mapping[str, Any],
    ) -> None:
        resolved = validate_amendment_io_path(
            repository_root=self.repository_root,
            path=path,
            role="candidate JSON",
            access="write",
        )
        try:
            with resolved.open("x", encoding="utf-8") as handle:
                json.dump(
                    value,
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
        except FileExistsError as error:
            raise SingleFactorContractError(
                f"append-only candidate JSON already exists: {resolved}"
            ) from error


@dataclass(frozen=True)
class CandidateDefinition:
    model_id: str
    candidate_id: str
    execution_kind: str
    intervention_dimension: str
    changed_dimensions: tuple[str, ...]
    trajectory_id: str | None


@dataclass(frozen=True)
class AmendmentDefinition:
    model_ids: tuple[str, ...]
    candidates: tuple[CandidateDefinition, ...]
    diagnostics_by_model: Mapping[str, tuple[str, ...]]
    training_trajectory_count: int
    evaluation_only_count: int
    reused_control_count: int
    max_training_gpu_hours: float
    max_evaluation_only_gpu_hours: float
    max_total_gpu_hours: float


@dataclass(frozen=True)
class AmendmentPlan:
    config_path: Path
    config_sha256: str
    model_ids: tuple[str, ...]
    candidates: tuple[CandidateDefinition, ...]
    diagnostics_by_model: Mapping[str, tuple[str, ...]]
    training_trajectory_count: int
    evaluation_only_count: int
    reused_control_count: int
    max_training_gpu_hours: float
    max_evaluation_only_gpu_hours: float
    max_total_gpu_hours: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SingleFactorContractError(
            f"cannot read single-factor amendment: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise SingleFactorContractError("amendment root must be a mapping")
    return value


def _repository_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name != "benchmark_v2":
        raise SingleFactorContractError(
            "amendment config must be under configs/benchmark_v2"
        )
    root = resolved.parents[2]
    if not (root / ".git").exists():
        raise SingleFactorContractError("repository root cannot be resolved")
    return root


def _assert_frozen_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise SingleFactorContractError("wrong amendment schema")
    forbidden_true = (
        "execution_authorized",
        "candidate_training_authorized",
        "validation_selection_authorized",
        "fresh_test_authorized",
        "five_seed_full_run_authorized",
        "gpu_execution_authorized",
    )
    if any(config.get(key) is not False for key in forbidden_true):
        raise SingleFactorContractError(
            "execution authorization must remain false"
        )
    if (
        config.get("mode") != "PREPARATION_ONLY"
        or config.get("test_split_access") != "FORBIDDEN"
        or config.get("artifact_root")
        != "artifacts/benchmark_v2_6/selection_single_factor"
    ):
        raise SingleFactorContractError(
            "preparation/test/artifact scope is not frozen"
        )
    frozen = config.get("frozen_contract")
    if not isinstance(frozen, Mapping):
        raise SingleFactorContractError("frozen contract is missing")
    for key, expected in FROZEN_HASHES.items():
        if frozen.get(key) != expected:
            raise SingleFactorContractError(
                f"frozen provenance changed: {key}"
            )
    if frozen.get("thresholds") != FROZEN_THRESHOLDS:
        raise SingleFactorContractError("five-guard thresholds changed")
    expected_values = {
        "scenario": "joint_semimarkov_v2b",
        "kappa": 1.0,
        "selection_seed": 2601,
        "endpoint": "continuous_association_recovery_error",
        "dgp": "benchmark_v2_joint_semimarkov_v2b_kappa_1_00_frozen",
        "c2_rule": "unchanged_v2_6_primary_three_model_rule",
    }
    for key, expected in expected_values.items():
        if frozen.get(key) != expected:
            raise SingleFactorContractError(
                f"frozen benchmark contract changed: {key}"
            )
    expected_rule = {
        "eligibility": "all_five_row_marginal_guards_PASS",
        "primary_objective": "minimize_max_amount_ks_gap_ks",
        "secondary_objective": "minimize_sum_amount_ks_gap_ks",
        "final_tiebreaker": "lexicographic_candidate_id",
        "no_pass_action": "PROHIBIT_V2_6_PRIMARY_C2_FULL_RUN",
    }
    if config.get("selection_rule") != expected_rule:
        raise SingleFactorContractError("selection rule changed")


def _assert_budget(config: Mapping[str, Any]) -> None:
    budget = config.get("budget")
    expected = {
        "requested_updates_per_new_trajectory": 20_000,
        "checkpoint_interval_steps": 100,
        "max_gpu_wall_seconds_per_new_trajectory": 7_200,
        "new_training_trajectories": 3,
        "max_training_gpu_hours": 6.0,
        "evaluation_only_candidates": 3,
        "max_gpu_wall_seconds_per_evaluation_only_candidate": 1_800,
        "max_evaluation_only_gpu_hours": 1.5,
        "reused_control_candidates": 3,
        "total_evaluation_candidates": 9,
        "max_total_gpu_hours": 7.5,
        "early_stopping": False,
        "update_increase_candidates": 0,
        "result_based_extension": "FORBIDDEN",
    }
    if budget != expected:
        raise SingleFactorContractError("single-factor budget changed")


def _assert_reference(
    repository_root: Path,
    record: Mapping[str, Any],
    *,
    model_id: str,
) -> None:
    pairs = (
        ("candidate_result_path", "candidate_result_sha256"),
        ("checkpoint_path", "checkpoint_sha256"),
        ("validation_sample_path", "validation_sample_sha256"),
    )
    allowed_root = (
        repository_root
        / "artifacts/benchmark_v2_6/selection/candidates"
    ).resolve()
    for path_key, hash_key in pairs:
        path = (repository_root / str(record.get(path_key, ""))).resolve()
        if allowed_root not in path.parents or not path.is_file():
            raise SingleFactorContractError(
                f"{model_id} frozen control path is invalid: {path_key}"
            )
        if "test" in {part.lower() for part in path.parts}:
            raise SingleFactorContractError(
                f"{model_id} frozen control accesses test"
            )
        if sha256_file(path) != record.get(hash_key):
            raise SingleFactorContractError(
                f"{model_id} frozen control hash mismatch: {path_key}"
            )


def _changed_dimensions(
    baseline: Mapping[str, Any],
    effective: Mapping[str, Any],
) -> tuple[str, ...]:
    if set(baseline) != set(effective):
        raise SingleFactorContractError(
            "candidate dimensions differ from baseline schema"
        )
    return tuple(
        key
        for key in baseline
        if effective[key] != baseline[key]
    )


def _frequency(values: np.ndarray, categories: int) -> list[float]:
    if not len(values):
        raise SingleFactorContractError(
            "diagnostic frequency group is empty"
        )
    return (
        np.bincount(values, minlength=categories).astype(float)
        / len(values)
    ).tolist()


def _signed_standardized_effect(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    groups = [values[labels == label] for label in (0, 1)]
    if any(len(group) < 2 for group in groups):
        raise SingleFactorContractError(
            "diagnostic class group is too small"
        )
    degrees = len(groups[0]) + len(groups[1]) - 2
    pooled = (
        (len(groups[0]) - 1) * groups[0].var(ddof=1)
        + (len(groups[1]) - 1) * groups[1].var(ddof=1)
    ) / degrees
    difference = float(groups[1].mean() - groups[0].mean())
    if pooled <= 0:
        return 0.0 if difference == 0 else float(
            np.copysign(np.inf, difference)
        )
    return float(difference / np.sqrt(pooled))


def compute_model_diagnostics(
    *,
    model_id: str,
    sample: SyntheticBatch,
    tau: np.ndarray,
    receiver_categories: int,
) -> Mapping[str, Any]:
    if model_id not in MODEL_IDS:
        raise SingleFactorContractError(
            f"unknown diagnostic model: {model_id}"
        )
    if (
        receiver_categories < 1
        or np.asarray(tau).ndim != 1
        or not np.isfinite(tau).all()
    ):
        raise SingleFactorContractError(
            "diagnostic support metadata is invalid"
        )
    valid = sample.valid_mask
    labels = np.repeat(sample.y_entity, sample.lengths)
    amount = np.asarray(sample.x_num[..., 0][valid], dtype=float)
    gap_bin = np.asarray(sample.dt_bin[valid], dtype=np.int64)
    if np.any(gap_bin < 0) or np.any(gap_bin >= len(tau)):
        raise SingleFactorContractError(
            "diagnostic gap is outside frozen support"
        )
    receiver = np.asarray(
        sample.x_cat[..., 0][valid],
        dtype=np.int64,
    )
    if (
        np.any(receiver < 0)
        or np.any(receiver >= receiver_categories)
    ):
        raise SingleFactorContractError(
            "diagnostic receiver is outside frozen support"
        )
    if model_id == "ctgan_separate_class":
        gap = np.asarray(tau, dtype=float)[gap_bin]
        entity_frequencies = np.zeros(
            (len(sample.lengths), receiver_categories),
            dtype=float,
        )
        for index, length_value in enumerate(sample.lengths):
            length = int(length_value)
            entity_frequencies[index] = (
                np.bincount(
                    sample.x_cat[index, :length, 0],
                    minlength=receiver_categories,
                )
                / length
            )
        receiver_by_class = {
            str(label): entity_frequencies[
                sample.y_entity == label
            ].mean(0)
            for label in (0, 1)
        }
        return {
            "amount_mean_by_class": {
                str(label): float(amount[labels == label].mean())
                for label in (0, 1)
            },
            "amount_std_by_class": {
                str(label): float(amount[labels == label].std())
                for label in (0, 1)
            },
            "amount_signed_standardized_class_effect": (
                _signed_standardized_effect(amount, labels)
            ),
            "gap_frequency_by_class": {
                str(label): _frequency(
                    gap_bin[labels == label],
                    len(tau),
                )
                for label in (0, 1)
            },
            "gap_signed_standardized_class_effect": (
                _signed_standardized_effect(gap, labels)
            ),
            "receiver_frequency_by_class": {
                label: values.tolist()
                for label, values in receiver_by_class.items()
            },
            "receiver_signed_frequency_by_category": (
                receiver_by_class["1"] - receiver_by_class["0"]
            ).tolist(),
        }
    if model_id == "tvae_separate_class":
        gap_frequency = np.asarray(
            _frequency(gap_bin, len(tau)),
            dtype=float,
        )
        receiver_frequency = np.asarray(
            _frequency(receiver, receiver_categories),
            dtype=float,
        )
        return {
            "gap_raw_frequency_by_bin": gap_frequency.tolist(),
            "receiver_raw_frequency_by_category": (
                receiver_frequency.tolist()
            ),
            "gap_max_category_frequency": float(gap_frequency.max()),
            "receiver_max_category_frequency": float(
                receiver_frequency.max()
            ),
            "gap_frequency_hhi": float(np.square(gap_frequency).sum()),
            "receiver_frequency_hhi": float(
                np.square(receiver_frequency).sum()
            ),
            "gap_frequency_by_class": {
                str(label): _frequency(
                    gap_bin[labels == label],
                    len(tau),
                )
                for label in (0, 1)
            },
            "receiver_frequency_by_class": {
                str(label): _frequency(
                    receiver[labels == label],
                    receiver_categories,
                )
                for label in (0, 1)
            },
        }
    quantile_levels = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    quantiles = np.quantile(amount, quantile_levels)
    return {
        "generated_amount_mean": float(amount.mean()),
        "generated_amount_std": float(amount.std()),
        **{
            f"generated_amount_q{int(level * 100):02d}": float(value)
            for level, value in zip(quantile_levels, quantiles)
        },
    }


def validate_amendment_definition(
    config: Mapping[str, Any],
) -> AmendmentDefinition:
    _assert_frozen_contract(config)
    _assert_budget(config)
    models = config.get("models")
    if not isinstance(models, Mapping) or tuple(models) != MODEL_IDS:
        raise SingleFactorContractError(
            "single-factor model order/family changed"
        )
    definitions = []
    diagnostics_by_model: dict[str, tuple[str, ...]] = {}
    for model_id in MODEL_IDS:
        model = models[model_id]
        if not isinstance(model, Mapping):
            raise SingleFactorContractError(f"malformed model: {model_id}")
        baseline = model.get("baseline_dimensions")
        candidates = model.get("candidates")
        diagnostics = model.get("diagnostics")
        frozen_control = model.get("frozen_control")
        if (
            not isinstance(baseline, Mapping)
            or not isinstance(candidates, list)
            or not isinstance(diagnostics, list)
            or not isinstance(frozen_control, Mapping)
        ):
            raise SingleFactorContractError(
                f"incomplete model contract: {model_id}"
            )
        if baseline != EXPECTED_BASELINES[model_id]:
            raise SingleFactorContractError(
                f"baseline definition changed: {model_id}"
            )
        if tuple(diagnostics) != EXPECTED_DIAGNOSTICS[model_id]:
            raise SingleFactorContractError(
                f"diagnostic contract changed: {model_id}"
            )
        diagnostics_by_model[model_id] = tuple(diagnostics)
        expected = EXPECTED_CANDIDATES[model_id]
        if {
            str(candidate.get("candidate_id"))
            for candidate in candidates
        } != set(expected):
            raise SingleFactorContractError(
                f"candidate family changed: {model_id}"
            )
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            execution = str(candidate.get("execution_kind"))
            intervention = str(candidate.get("intervention_dimension"))
            if execution not in EXECUTION_KINDS:
                raise SingleFactorContractError(
                    f"invalid execution kind: {candidate_id}"
                )
            if (execution, intervention) != expected[candidate_id]:
                raise SingleFactorContractError(
                    f"candidate intervention changed: {candidate_id}"
                )
            effective = candidate.get("effective_dimensions")
            if not isinstance(effective, Mapping):
                raise SingleFactorContractError(
                    f"candidate dimensions missing: {candidate_id}"
                )
            changed = _changed_dimensions(baseline, effective)
            expected_changed = () if intervention == "none" else (
                intervention,
            )
            if changed != expected_changed:
                raise SingleFactorContractError(
                    f"candidate is not single-factor: {candidate_id}"
                )
            expected_effective = dict(baseline)
            if candidate_id in EXPECTED_INTERVENTION_VALUES:
                dimension, value = EXPECTED_INTERVENTION_VALUES[
                    candidate_id
                ]
                expected_effective[dimension] = value
            if effective != expected_effective:
                raise SingleFactorContractError(
                    f"candidate value changed: {candidate_id}"
                )
            if effective.get("requested_updates") != 20_000:
                raise SingleFactorContractError(
                    f"candidate changes update budget: {candidate_id}"
                )
            trajectory_id = candidate.get("trajectory_id")
            if (
                execution == "train_new_trajectory"
                and trajectory_id != EXPECTED_TRAJECTORIES.get(candidate_id)
            ):
                raise SingleFactorContractError(
                    f"training candidate lacks trajectory: {candidate_id}"
                )
            if (
                execution != "train_new_trajectory"
                and trajectory_id is not None
            ):
                raise SingleFactorContractError(
                    f"nontraining candidate owns trajectory: {candidate_id}"
                )
            if (
                execution == "evaluate_existing_checkpoint"
                and candidate.get("source_checkpoint") != "frozen_control"
            ):
                raise SingleFactorContractError(
                    f"evaluation candidate source changed: {candidate_id}"
                )
            if (
                execution != "evaluate_existing_checkpoint"
                and candidate.get("source_checkpoint") is not None
            ):
                raise SingleFactorContractError(
                    f"candidate has an unexpected source checkpoint: "
                    f"{candidate_id}"
                )
            residual_contract = candidate.get("residual_contract")
            if candidate_id == "cof_sf_c02_variance_preserving_residual":
                if residual_contract != EXPECTED_RESIDUAL_CONTRACT:
                    raise SingleFactorContractError(
                        "variance-preserving residual contract changed"
                    )
            elif residual_contract is not None:
                raise SingleFactorContractError(
                    f"unexpected residual contract: {candidate_id}"
                )
            definitions.append(
                CandidateDefinition(
                    model_id=model_id,
                    candidate_id=candidate_id,
                    execution_kind=execution,
                    intervention_dimension=intervention,
                    changed_dimensions=changed,
                    trajectory_id=trajectory_id,
                )
            )
    training = sum(
        item.execution_kind == "train_new_trajectory"
        for item in definitions
    )
    evaluation = sum(
        item.execution_kind == "evaluate_existing_checkpoint"
        for item in definitions
    )
    controls = sum(
        item.execution_kind == "reuse_frozen_control"
        for item in definitions
    )
    if (training, evaluation, controls) != (3, 3, 3):
        raise SingleFactorContractError(
            "candidate execution accounting changed"
        )
    budget = config["budget"]
    return AmendmentDefinition(
        model_ids=MODEL_IDS,
        candidates=tuple(definitions),
        diagnostics_by_model=diagnostics_by_model,
        training_trajectory_count=training,
        evaluation_only_count=evaluation,
        reused_control_count=controls,
        max_training_gpu_hours=float(budget["max_training_gpu_hours"]),
        max_evaluation_only_gpu_hours=float(
            budget["max_evaluation_only_gpu_hours"]
        ),
        max_total_gpu_hours=float(budget["max_total_gpu_hours"]),
    )


def load_and_validate_amendment(config_path: Path) -> AmendmentPlan:
    config_path = config_path.resolve()
    repository_root = _repository_root(config_path)
    config = _read_yaml(config_path)
    definition = validate_amendment_definition(config)
    models = config["models"]
    for model_id in MODEL_IDS:
        _assert_reference(
            repository_root,
            models[model_id]["frozen_control"],
            model_id=model_id,
        )
    return AmendmentPlan(
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        model_ids=definition.model_ids,
        candidates=definition.candidates,
        diagnostics_by_model=definition.diagnostics_by_model,
        training_trajectory_count=definition.training_trajectory_count,
        evaluation_only_count=definition.evaluation_only_count,
        reused_control_count=definition.reused_control_count,
        max_training_gpu_hours=definition.max_training_gpu_hours,
        max_evaluation_only_gpu_hours=(
            definition.max_evaluation_only_gpu_hours
        ),
        max_total_gpu_hours=definition.max_total_gpu_hours,
    )
