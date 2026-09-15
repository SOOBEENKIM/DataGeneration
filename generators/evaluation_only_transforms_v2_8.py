from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.candidate_preparation_v2_8 import (
    V28Candidate,
    V28PreparationContractError,
    build_v28_fit_state,
    canonical_sha256,
)


class EvaluationInterventionError(RuntimeError):
    pass


def _row_labels(batch: SequenceBatch | SyntheticBatch) -> np.ndarray:
    return np.broadcast_to(
        batch.y_entity[:, None],
        batch.valid_mask.shape,
    )[batch.valid_mask]


def _joint_rows(
    batch: SequenceBatch | SyntheticBatch,
) -> np.ndarray:
    return np.column_stack(
        (
            batch.dt_bin[batch.valid_mask],
            batch.x_cat[..., 0][batch.valid_mask],
        )
    ).astype(np.int64)


def _class_probabilities(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    label: int,
    support: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    selected = values[labels == label]
    if not len(selected):
        raise EvaluationInterventionError(
            "train-only calibration needs both entity classes"
        )
    counts = np.array(
        [
            np.count_nonzero(
                np.all(selected == item, axis=1)
                if selected.ndim == 2
                else selected == item
            )
            for item in support
        ],
        dtype=np.float64,
    )
    return (counts + smoothing) / (
        counts.sum() + smoothing * len(support)
    )


def _fit_joint(
    train: SequenceBatch,
    source: SyntheticBatch,
    *,
    smoothing: float,
    clip: tuple[float, float],
) -> Mapping[str, Any]:
    target_values = _joint_rows(train)
    source_values = _joint_rows(source)
    support = np.unique(target_values, axis=0)
    target_labels = _row_labels(train)
    source_labels = _row_labels(source)
    class_states = {}
    for label in (0, 1):
        target = _class_probabilities(
            target_values,
            target_labels,
            label=label,
            support=support,
            smoothing=smoothing,
        )
        source_probability = _class_probabilities(
            source_values,
            source_labels,
            label=label,
            support=support,
            smoothing=smoothing,
        )
        class_states[str(label)] = {
            "target_probability": target.tolist(),
            "source_probability": source_probability.tolist(),
            "logit_residual": np.clip(
                np.log(target) - np.log(source_probability),
                clip[0],
                clip[1],
            ).tolist(),
        }
    return {
        "support": support.tolist(),
        "additive_smoothing": smoothing,
        "logit_offset_clip": list(clip),
        "class_states": class_states,
    }


def _fit_gap(
    train: SequenceBatch,
    source: SyntheticBatch,
) -> Mapping[str, Any]:
    target_values = train.dt_bin[train.valid_mask]
    source_values = source.dt_bin[source.valid_mask]
    support = np.unique(target_values)
    target_labels = _row_labels(train)
    source_labels = _row_labels(source)
    class_states = {}
    for label in (0, 1):
        target = _class_probabilities(
            target_values,
            target_labels,
            label=label,
            support=support,
            smoothing=0.0,
        )
        source_probability = _class_probabilities(
            source_values,
            source_labels,
            label=label,
            support=support,
            smoothing=0.0,
        )
        class_states[str(label)] = {
            "target_probability": target.tolist(),
            "source_probability": source_probability.tolist(),
        }
    return {
        "support": support.tolist(),
        "interpolation": "discrete_monotone_quantile_transport",
        "tie_break": "lower_gap_bin",
        "class_states": class_states,
    }


def fit_train_only_intervention(
    *,
    candidate: V28Candidate,
    train: SequenceBatch,
    calibration_sample: SyntheticBatch,
    source_commit: str,
    config_sha256: str,
    train_file_sha256: str,
    train_content_sha256: str,
    sampling_plan_sha256: str,
) -> Mapping[str, Any]:
    if (
        candidate.is_control
        or not isinstance(train, SequenceBatch)
        or not isinstance(calibration_sample, SyntheticBatch)
    ):
        raise EvaluationInterventionError(
            "v2.8 calibration requires a future candidate, frozen train, "
            "and one train-plan sample"
        )
    if candidate.factor == "joint_gap_receiver_discrete_decoder":
        fit = candidate.fit_parameters
        clip = tuple(float(value) for value in fit["logit_offset_clip"])
        factor_state = _fit_joint(
            train,
            calibration_sample,
            smoothing=float(fit["additive_smoothing"]),
            clip=(clip[0], clip[1]),
        )
    elif candidate.factor == "gap_distribution_sampler":
        factor_state = _fit_gap(train, calibration_sample)
    else:
        raise EvaluationInterventionError(
            f"unsupported v2.8 factor: {candidate.factor}"
        )
    parameters = {
        "rule": candidate.fit_parameters["rule"],
        "changed_factor": candidate.factor,
        "factor_state": factor_state,
    }
    try:
        return build_v28_fit_state(
            candidate=candidate,
            parameters=parameters,
            source_commit=source_commit,
            config_sha256=config_sha256,
            train_file_sha256=train_file_sha256,
            train_content_sha256=train_content_sha256,
            sampling_plan_sha256=sampling_plan_sha256,
        )
    except V28PreparationContractError as error:
        raise EvaluationInterventionError(
            "v2.8 train-only fit-state provenance is invalid"
        ) from error


def _validated_parameters(
    candidate: V28Candidate,
    fit_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    unhashed = {
        key: value
        for key, value in fit_state.items()
        if key != "state_sha256"
    }
    parameters = fit_state.get("parameters")
    if (
        not isinstance(parameters, Mapping)
        or fit_state.get("candidate_id") != candidate.candidate_id
        or fit_state.get("model_id") != candidate.model_id
        or fit_state.get("factor") != candidate.factor
        or fit_state.get("fit_split") != "train"
        or fit_state.get("validation_rows_used") != 0
        or fit_state.get("test_rows_used") != 0
        or fit_state.get("state_sha256") != canonical_sha256(unhashed)
        or parameters.get("changed_factor") != candidate.factor
    ):
        raise EvaluationInterventionError(
            "v2.8 fit-state identity or hash mismatch"
        )
    return parameters


def _allocate(
    probability: np.ndarray,
    support: np.ndarray,
    count: int,
) -> np.ndarray:
    if count == 0:
        return np.empty((0,) + support.shape[1:], dtype=np.int64)
    positions = (np.arange(count, dtype=np.float64) + 0.5) / count
    indices = np.searchsorted(
        np.cumsum(probability),
        positions,
        side="left",
    )
    return support[np.minimum(indices, len(support) - 1)]


def apply_post_sample_intervention(
    *,
    candidate: V28Candidate,
    sample: SyntheticBatch,
    fit_state: Mapping[str, Any],
) -> SyntheticBatch:
    parameters = _validated_parameters(candidate, fit_state)
    factor_state = parameters.get("factor_state")
    if not isinstance(factor_state, Mapping):
        raise EvaluationInterventionError(
            "v2.8 factor fit state is missing"
        )
    dt_bin = sample.dt_bin.copy()
    categorical = sample.x_cat.copy()
    support = np.asarray(factor_state["support"], dtype=np.int64)
    for label in (0, 1):
        mask = sample.valid_mask & (sample.y_entity == label)[:, None]
        flat_positions = np.flatnonzero(mask)
        probability = np.asarray(
            factor_state["class_states"][str(label)][
                "target_probability"
            ],
            dtype=np.float64,
        )
        allocated = _allocate(probability, support, int(mask.sum()))
        if candidate.factor == "joint_gap_receiver_discrete_decoder":
            dt_bin[mask] = allocated[:, 0]
            categorical[..., 0][mask] = allocated[:, 1]
        elif candidate.factor == "gap_distribution_sampler":
            flat_gap = dt_bin.reshape(-1)
            order = np.argsort(
                flat_gap[flat_positions],
                kind="stable",
            )
            flat_gap[flat_positions[order]] = allocated
        else:
            raise EvaluationInterventionError(
                f"unsupported v2.8 factor: {candidate.factor}"
            )
    dt_bin[~sample.valid_mask] = 0
    categorical[~sample.valid_mask] = 0
    return SyntheticBatch(
        x_num=sample.x_num.copy(),
        dt_bin=dt_bin,
        x_cat=categorical,
        valid_mask=sample.valid_mask.copy(),
        y_entity=sample.y_entity.copy(),
        lengths=sample.lengths.copy(),
    )
