from __future__ import annotations

from functools import partial
from types import MethodType
from typing import Any, Mapping

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.candidate_preparation_v2_7 import (
    V27Candidate,
    V27PreparationContractError,
    build_fit_state_record,
    canonical_sha256,
)


class EvaluationInterventionError(RuntimeError):
    pass


_AMOUNT_FACTORS = {
    "amount_inverse_map",
    "amount_inverse_decoder",
    "amount_residual_sampler",
}
_LOGIT_FACTORS = {
    "categorical_logit_calibration",
    "gap_logit_calibration",
}


def _row_labels(batch: SequenceBatch | SyntheticBatch) -> np.ndarray:
    return np.broadcast_to(
        batch.y_entity[:, None],
        batch.valid_mask.shape,
    )[batch.valid_mask]


def _finite_amount_rows(
    batch: SequenceBatch | SyntheticBatch,
    *,
    label: int,
) -> np.ndarray:
    rows = batch.x_num[..., 0][batch.valid_mask]
    selected = rows[_row_labels(batch) == label].astype(np.float64)
    if not len(selected) or not np.isfinite(selected).all():
        raise EvaluationInterventionError(
            "train-only amount calibration needs finite rows per class"
        )
    return selected


def _quantile_pair(
    source: np.ndarray,
    target: np.ndarray,
    *,
    count: int,
    centered: bool,
) -> Mapping[str, Any]:
    if count < 3:
        raise EvaluationInterventionError(
            "quantile calibration grid must contain at least 3 points"
        )
    source_center = float(np.mean(source)) if centered else 0.0
    target_center = float(np.mean(target)) if centered else 0.0
    levels = np.linspace(0.0, 1.0, count, dtype=np.float64)
    source_quantiles = np.quantile(source - source_center, levels)
    target_quantiles = np.quantile(target - target_center, levels)
    if (
        not np.isfinite(source_quantiles).all()
        or not np.isfinite(target_quantiles).all()
    ):
        raise EvaluationInterventionError(
            "non-finite train-only quantile fit state"
        )
    return {
        "levels": levels.tolist(),
        "source_quantiles": source_quantiles.tolist(),
        "target_quantiles": target_quantiles.tolist(),
        "source_center": source_center,
        "target_center": target_center,
    }


def _category_probabilities(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    label: int,
    categories: int,
    smoothing: float,
) -> np.ndarray:
    selected = values[labels == label]
    if not len(selected):
        raise EvaluationInterventionError(
            "categorical calibration needs rows for both classes"
        )
    counts = np.bincount(
        selected.astype(np.int64),
        minlength=categories,
    ).astype(np.float64)
    return (counts + smoothing) / (
        counts.sum() + smoothing * categories
    )


def _logit_offsets(
    *,
    target: SequenceBatch,
    source: SyntheticBatch,
    channel: str,
    smoothing: float,
    clip: tuple[float, float],
) -> Mapping[str, Any]:
    if channel == "gap":
        target_values = target.dt_bin[target.valid_mask]
        source_values = source.dt_bin[source.valid_mask]
    elif channel == "receiver":
        target_values = target.x_cat[..., 0][target.valid_mask]
        source_values = source.x_cat[..., 0][source.valid_mask]
    else:
        raise EvaluationInterventionError(
            f"unknown categorical channel: {channel}"
        )
    categories = int(
        max(target_values.max(), source_values.max())
    ) + 1
    target_labels = _row_labels(target)
    source_labels = _row_labels(source)
    offsets: dict[str, list[float]] = {}
    for label in (0, 1):
        target_probability = _category_probabilities(
            target_values,
            target_labels,
            label=label,
            categories=categories,
            smoothing=smoothing,
        )
        source_probability = _category_probabilities(
            source_values,
            source_labels,
            label=label,
            categories=categories,
            smoothing=smoothing,
        )
        raw = np.log(target_probability) - np.log(source_probability)
        offsets[str(label)] = np.clip(
            raw,
            clip[0],
            clip[1],
        ).tolist()
    return {
        "categories": categories,
        "additive_smoothing": smoothing,
        "clip": list(clip),
        "offset_by_label": offsets,
    }


def _categorical_train_score(
    train: SequenceBatch,
    sample: SyntheticBatch,
) -> float:
    scores = []
    for channel in ("gap", "receiver"):
        if channel == "gap":
            target = train.dt_bin[train.valid_mask]
            source = sample.dt_bin[sample.valid_mask]
        else:
            target = train.x_cat[..., 0][train.valid_mask]
            source = sample.x_cat[..., 0][sample.valid_mask]
        categories = int(max(target.max(), source.max())) + 1
        target_labels = _row_labels(train)
        source_labels = _row_labels(sample)
        for label in (0, 1):
            target_counts = np.bincount(
                target[target_labels == label],
                minlength=categories,
            ).astype(np.float64)
            source_counts = np.bincount(
                source[source_labels == label],
                minlength=categories,
            ).astype(np.float64)
            scores.append(
                float(
                    np.max(
                        np.abs(
                            target_counts / target_counts.sum()
                            - source_counts / source_counts.sum()
                        )
                    )
                )
            )
    return float(max(scores))


def _validate_calibration_samples(
    train: SequenceBatch,
    samples: Mapping[str, SyntheticBatch],
) -> None:
    if not samples:
        raise EvaluationInterventionError(
            "train-only calibration samples are missing"
        )
    for name, sample in samples.items():
        if not isinstance(name, str) or not isinstance(
            sample,
            SyntheticBatch,
        ):
            raise EvaluationInterventionError(
                "calibration samples must be named SyntheticBatch values"
            )
        if set(np.unique(sample.y_entity)) != {0, 1}:
            raise EvaluationInterventionError(
                "calibration sample must preserve both entity labels"
            )
        if sample.valid_mask.shape[1] != train.valid_mask.shape[1]:
            raise EvaluationInterventionError(
                "calibration sample sequence length differs from train"
            )


def fit_train_only_intervention(
    *,
    candidate: V27Candidate,
    train: SequenceBatch,
    calibration_samples: Mapping[str, SyntheticBatch],
    source_commit: str,
    config_sha256: str,
    train_file_sha256: str,
    train_content_sha256: str,
    sampling_plan_sha256: str,
) -> Mapping[str, Any]:
    if candidate.is_control:
        raise EvaluationInterventionError(
            "frozen controls cannot fit an intervention"
        )
    if not isinstance(train, SequenceBatch):
        raise EvaluationInterventionError(
            "intervention fitting accepts the frozen train split only"
        )
    _validate_calibration_samples(train, calibration_samples)
    fit_parameters = dict(candidate.fit_parameters)
    factor = candidate.factor
    parameters: dict[str, Any] = {
        "rule": fit_parameters["rule"],
        "changed_factor": factor,
    }
    if factor in {
        "amount_inverse_map",
        "amount_inverse_decoder",
        "amount_residual_sampler",
    }:
        source = calibration_samples.get("base")
        if source is None or set(calibration_samples) != {"base"}:
            raise EvaluationInterventionError(
                "amount fit requires exactly one train-plan base sample"
            )
        count = int(fit_parameters["quantile_grid_size"])
        centered = factor == "amount_residual_sampler"
        parameters["class_quantile_maps"] = {
            str(label): _quantile_pair(
                _finite_amount_rows(source, label=label),
                _finite_amount_rows(train, label=label),
                count=count,
                centered=centered,
            )
            for label in (0, 1)
        }
    elif factor == "categorical_logit_calibration":
        source = calibration_samples.get("base")
        if source is None or set(calibration_samples) != {"base"}:
            raise EvaluationInterventionError(
                "categorical fit requires one train-plan base sample"
            )
        smoothing = float(fit_parameters["additive_smoothing"])
        clip_values = fit_parameters["logit_offset_clip"]
        clip = (float(clip_values[0]), float(clip_values[1]))
        parameters["logit_offsets"] = {
            channel: _logit_offsets(
                target=train,
                source=source,
                channel=channel,
                smoothing=smoothing,
                clip=clip,
            )
            for channel in ("gap", "receiver")
        }
    elif factor == "gap_logit_calibration":
        source = calibration_samples.get("base")
        if source is None or set(calibration_samples) != {"base"}:
            raise EvaluationInterventionError(
                "gap fit requires one train-plan base sample"
            )
        clip_values = fit_parameters["logit_offset_clip"]
        parameters["gap_logit_offsets"] = _logit_offsets(
            target=train,
            source=source,
            channel="gap",
            smoothing=float(fit_parameters["additive_smoothing"]),
            clip=(float(clip_values[0]), float(clip_values[1])),
        )
    elif factor == "categorical_temperature":
        grid = [float(value) for value in fit_parameters["fixed_grid"]]
        expected = {f"{value:g}" for value in grid}
        if set(calibration_samples) != expected:
            raise EvaluationInterventionError(
                "temperature fit samples differ from the frozen grid"
            )
        scored = {
            key: _categorical_train_score(train, calibration_samples[key])
            for key in sorted(expected, key=float)
        }
        chosen = min(
            grid,
            key=lambda value: (
                scored[f"{value:g}"],
                abs(value - 0.75),
                value,
            ),
        )
        parameters.update(
            {
                "fixed_grid": grid,
                "train_only_scores": scored,
                "selected_temperature": float(chosen),
            }
        )
    else:
        raise EvaluationInterventionError(
            f"unsupported v2.7 evaluation factor: {factor}"
        )
    try:
        return build_fit_state_record(
            candidate=candidate,
            parameters=parameters,
            source_commit=source_commit,
            config_sha256=config_sha256,
            train_file_sha256=train_file_sha256,
            train_content_sha256=train_content_sha256,
            sampling_plan_sha256=sampling_plan_sha256,
            destination="candidate_artifact",
        )
    except V27PreparationContractError as error:
        raise EvaluationInterventionError(
            "train-only fit-state provenance is invalid"
        ) from error


def _validated_parameters(
    candidate: V27Candidate,
    fit_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    state = {
        key: value
        for key, value in fit_state.items()
        if key != "state_sha256"
    }
    if (
        fit_state.get("candidate_id") != candidate.candidate_id
        or fit_state.get("model_id") != candidate.model_id
        or fit_state.get("factor") != candidate.factor
        or fit_state.get("fit_split") != "train"
        or fit_state.get("validation_rows_used") != 0
        or fit_state.get("test_rows_used") != 0
        or fit_state.get("state_sha256") != canonical_sha256(state)
    ):
        raise EvaluationInterventionError(
            "fit-state identity or hash mismatch"
        )
    parameters = fit_state.get("parameters")
    if not isinstance(parameters, Mapping):
        raise EvaluationInterventionError("fit-state parameters are missing")
    return parameters


def apply_post_sample_intervention(
    *,
    candidate: V27Candidate,
    sample: SyntheticBatch,
    fit_state: Mapping[str, Any],
) -> SyntheticBatch:
    parameters = _validated_parameters(candidate, fit_state)
    numerical = sample.x_num.copy()
    if candidate.factor in _AMOUNT_FACTORS:
        maps = parameters.get("class_quantile_maps")
        if not isinstance(maps, Mapping):
            raise EvaluationInterventionError(
                "amount fit state has no class quantile maps"
            )
        for label in (0, 1):
            entity_mask = sample.y_entity == label
            mask = sample.valid_mask & entity_mask[:, None]
            mapping = maps[str(label)]
            source = np.asarray(
                mapping["source_quantiles"],
                dtype=np.float64,
            )
            target = np.asarray(
                mapping["target_quantiles"],
                dtype=np.float64,
            )
            values = numerical[..., 0][mask].astype(np.float64)
            values -= float(mapping["source_center"])
            mapped = np.interp(values, source, target)
            mapped += float(mapping["target_center"])
            numerical[..., 0][mask] = mapped.astype(np.float32)
    elif (
        candidate.factor not in _LOGIT_FACTORS
        and candidate.factor != "categorical_temperature"
    ):
        raise EvaluationInterventionError(
            f"unsupported post-sampling factor: {candidate.factor}"
        )
    numerical[~sample.valid_mask] = 0
    return SyntheticBatch(
        x_num=numerical,
        dt_bin=sample.dt_bin.copy(),
        x_cat=sample.x_cat.copy(),
        valid_mask=sample.valid_mask.copy(),
        y_entity=sample.y_entity.copy(),
        lengths=sample.lengths.copy(),
    )


def _ctgan_v27_apply_activate(synthesizer, data):
    import torch

    transformed = []
    start = 0
    offsets = synthesizer._v27_categorical_logit_offsets
    channel_by_column = {1: "gap", 2: "receiver"}
    temperature = float(
        getattr(synthesizer, "v2_6_categorical_temperature", 0.2)
    )
    for column_index, column_info in enumerate(
        synthesizer._transformer.output_info_list
    ):
        for span_info in column_info:
            end = start + span_info.dim
            logits = data[:, start:end]
            if span_info.activation_fn == "tanh":
                transformed.append(torch.tanh(logits))
            elif span_info.activation_fn == "softmax":
                channel = channel_by_column.get(column_index)
                if channel is not None:
                    bias = torch.as_tensor(
                        offsets[channel],
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                    if bias.numel() != logits.shape[-1]:
                        raise EvaluationInterventionError(
                            "CTGAN categorical logit-bias width mismatch"
                        )
                    logits = logits + bias
                transformed.append(
                    synthesizer._gumbel_softmax(
                        logits,
                        tau=temperature,
                    )
                )
            else:
                raise EvaluationInterventionError(
                    "unexpected CTGAN activation in v2.7 restore"
                )
            start = end
    return torch.cat(transformed, dim=1)


def configure_backend_for_intervention(
    *,
    backend: Any,
    candidate: V27Candidate,
    fit_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    parameters = _validated_parameters(candidate, fit_state)
    factor = candidate.factor
    record: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "changed_factor": factor,
        "training_calls": 0,
        "optimizer_updates": 0,
    }
    if factor in _AMOUNT_FACTORS:
        record["sampling_path_change"] = "post_sample_amount_only"
        return record
    if factor == "categorical_logit_calibration":
        models = getattr(backend, "models", None)
        offsets = parameters.get("logit_offsets")
        if not isinstance(models, Mapping) or set(models) != {0, 1}:
            raise EvaluationInterventionError(
                "CTGAN restore lacks its separate class models"
            )
        if not isinstance(offsets, Mapping) or set(offsets) != {
            "gap",
            "receiver",
        }:
            raise EvaluationInterventionError(
                "CTGAN categorical fit-state is incomplete"
            )
        for label, synthesizer in models.items():
            label_offsets = {
                channel: tuple(
                    float(value)
                    for value in offsets[channel][
                        "offset_by_label"
                    ][str(label)]
                )
                for channel in ("gap", "receiver")
            }
            synthesizer._v27_categorical_logit_offsets = (
                label_offsets
            )
            synthesizer._v27_original_apply_activate = getattr(
                synthesizer,
                "_apply_activate",
                None,
            )
            synthesizer._apply_activate = MethodType(
                _ctgan_v27_apply_activate,
                synthesizer,
            )
        record["sampling_path_change"] = (
            "ctgan_gap_receiver_logits_only"
        )
        return record
    if factor == "categorical_temperature":
        models = getattr(backend, "models", None)
        selected = float(parameters["selected_temperature"])
        if (
            not isinstance(models, Mapping)
            or set(models) != {0, 1}
            or selected not in {
                float(value)
                for value in parameters["fixed_grid"]
            }
        ):
            raise EvaluationInterventionError(
                "TVAE temperature fit-state is invalid"
            )
        for synthesizer in models.values():
            synthesizer.set_candidate_contract(
                channel_weights=dict(
                    synthesizer.v2_6_channel_weights
                ),
                latent_scale=float(
                    synthesizer.v2_6_latent_scale
                ),
                categorical_temperature=selected,
            )
        record.update(
            {
                "sampling_path_change": (
                    "tvae_categorical_temperature_only"
                ),
                "selected_temperature": selected,
            }
        )
        return record
    if factor == "gap_logit_calibration":
        offsets = parameters.get("gap_logit_offsets")
        if not isinstance(offsets, Mapping):
            raise EvaluationInterventionError(
                "CoF gap fit-state is incomplete"
            )
        bias = {
            label: tuple(
                float(value)
                for value in offsets["offset_by_label"][str(label)]
            )
            for label in (0, 1)
        }
        from models.evaluation_only_components_v2_7 import (
            ddim_sample_gap_bias_v2_7,
        )

        backend.sampling_function = partial(
            ddim_sample_gap_bias_v2_7,
            gap_logit_bias_by_label=bias,
        )
        record["sampling_path_change"] = "cof_gap_logits_only"
        return record
    raise EvaluationInterventionError(
        f"unsupported backend intervention: {factor}"
    )
