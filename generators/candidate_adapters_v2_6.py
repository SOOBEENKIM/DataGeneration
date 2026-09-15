from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch
from generators.sampling_plan import SamplingPlan


class CandidateAdapterContractError(RuntimeError):
    pass


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TrainOnlyZScore:
    mean: np.ndarray
    std: np.ndarray
    valid_row_count: int

    @classmethod
    def fit(cls, train: SequenceBatch) -> "TrainOnlyZScore":
        if not isinstance(train, SequenceBatch):
            raise TypeError("z-score fitting accepts a train SequenceBatch only")
        if train.x_num.shape[-1] != 1:
            raise CandidateAdapterContractError(
                "v2.6 candidate z-score is preregistered for amount only"
            )
        rows = train.x_num[train.valid_mask].astype(np.float64)
        if not len(rows) or not np.isfinite(rows).all():
            raise CandidateAdapterContractError(
                "train valid numerical rows must be finite and nonempty"
            )
        mean = rows.mean(axis=0, dtype=np.float64)
        std = rows.std(axis=0, dtype=np.float64)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise CandidateAdapterContractError(
                "train-only z-score state is non-finite"
            )
        if np.any(std <= 0):
            raise CandidateAdapterContractError(
                "train-only z-score has a zero-variance channel"
            )
        return cls(
            mean=np.ascontiguousarray(mean),
            std=np.ascontiguousarray(std),
            valid_row_count=int(len(rows)),
        )

    def transform_train(self, train: SequenceBatch) -> SequenceBatch:
        if not isinstance(train, SequenceBatch):
            raise TypeError("only a train SequenceBatch can be transformed")
        transformed = np.zeros_like(train.x_num, dtype=np.float32)
        transformed[train.valid_mask] = (
            (
                train.x_num[train.valid_mask].astype(np.float64)
                - self.mean
            )
            / self.std
        ).astype(np.float32)
        return SequenceBatch(
            x_num=transformed,
            dt_bin=train.dt_bin.copy(),
            x_cat=train.x_cat.copy(),
            valid_mask=train.valid_mask.copy(),
            y_entity=train.y_entity.copy(),
            lengths=train.lengths.copy(),
            entity_ids=np.asarray(train.entity_ids).copy(),
        )

    def inverse_synthetic(
        self,
        sample: SequenceBatch | SyntheticBatch,
    ) -> SequenceBatch | SyntheticBatch:
        if not isinstance(sample, (SequenceBatch, SyntheticBatch)):
            raise TypeError("z-score inverse requires a sequence batch")
        restored = np.zeros_like(sample.x_num, dtype=np.float32)
        restored[sample.valid_mask] = (
            sample.x_num[sample.valid_mask].astype(np.float64) * self.std
            + self.mean
        ).astype(np.float32)
        common = {
            "x_num": restored,
            "dt_bin": sample.dt_bin.copy(),
            "x_cat": sample.x_cat.copy(),
            "valid_mask": sample.valid_mask.copy(),
            "y_entity": sample.y_entity.copy(),
            "lengths": sample.lengths.copy(),
        }
        if isinstance(sample, SequenceBatch):
            return SequenceBatch(
                **common,
                entity_ids=np.asarray(sample.entity_ids).copy(),
            )
        return SyntheticBatch(**common)

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {
            "schema_version": "benchmark-v2.6-train-only-zscore-v1",
            "fit_split": "train",
            "valid_row_count": self.valid_row_count,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "state_sha256": _canonical_sha256(
                {
                    "fit_split": "train",
                    "valid_row_count": self.valid_row_count,
                    "mean": self.mean.tolist(),
                    "std": self.std.tolist(),
                }
            ),
        }


@dataclass(frozen=True)
class CandidateAdapterSpec:
    model_id: str
    candidate_id: str
    shared_trajectory_id: str
    requested_updates: int
    sampled_checkpoint_step: int
    checkpoint_interval_steps: int
    max_wall_seconds: float
    numeric_representation: Mapping[str, Any]
    loss_weights: Mapping[str, float]
    sampling_rule: Mapping[str, Any]
    class_updates: Mapping[int, int] | None
    class_wall_seconds: Mapping[int, float] | None
    definition_sha256: str

    @property
    def uses_train_zscore(self) -> bool:
        return "zscore" in str(
            self.numeric_representation.get("amount", "")
        ).lower()


def candidate_adapter_spec(
    config: Mapping[str, Any],
    *,
    model_id: str,
    candidate_id: str,
) -> CandidateAdapterSpec:
    models = config.get("models")
    if not isinstance(models, Mapping) or model_id not in models:
        raise CandidateAdapterContractError(f"unknown candidate model: {model_id}")
    candidates = models[model_id].get("candidates")
    if not isinstance(candidates, list):
        raise CandidateAdapterContractError("candidate definitions are missing")
    matches = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise CandidateAdapterContractError(
            f"candidate ID must resolve exactly once: {model_id}/{candidate_id}"
        )
    candidate = matches[0]
    schedule = candidate["training_schedule"]
    raw_weights = candidate["channel_loss_weights"]
    loss_weights = {
        key: float(raw_weights[key])
        for key in ("amount", "gap", "receiver", "label")
        if key in raw_weights
    }
    class_updates = schedule.get("class_updates")
    class_wall = schedule.get("class_wall_seconds")
    return CandidateAdapterSpec(
        model_id=model_id,
        candidate_id=candidate_id,
        shared_trajectory_id=str(schedule["shared_trajectory_id"]),
        requested_updates=int(schedule["requested_updates"]),
        sampled_checkpoint_step=int(
            schedule["selection_checkpoint_step"]
        ),
        checkpoint_interval_steps=int(
            schedule["checkpoint_interval_steps"]
        ),
        max_wall_seconds=float(schedule["max_gpu_wall_seconds"]),
        numeric_representation=dict(candidate["numeric_representation"]),
        loss_weights=loss_weights,
        sampling_rule=dict(candidate["sampling_rule"]),
        class_updates=(
            {int(key): int(value) for key, value in class_updates.items()}
            if isinstance(class_updates, Mapping)
            else None
        ),
        class_wall_seconds=(
            {int(key): float(value) for key, value in class_wall.items()}
            if isinstance(class_wall, Mapping)
            else None
        ),
        definition_sha256=_canonical_sha256(candidate),
    )


BackendFactory = Callable[[], Any]


class CandidateAdapterV26:
    """Lazy v2.6 wrapper around a model-specific candidate backend.

    Construction and contract inspection never instantiate a backend or touch
    CUDA. ``fit_train_only`` is the sole model-training entrypoint and
    ``sample_validation`` requires the frozen train-fit SamplingPlan.
    """

    def __init__(
        self,
        spec: CandidateAdapterSpec,
        *,
        backend_factory: BackendFactory,
    ) -> None:
        self.spec = spec
        self._backend_factory = backend_factory
        self._backend: Any | None = None
        self._zscore: TrainOnlyZScore | None = None
        self._sampling_plan_sha256: str | None = None

    @property
    def loss_weights(self) -> Mapping[str, float]:
        return dict(self.spec.loss_weights)

    @property
    def sampling_rule(self) -> Mapping[str, Any]:
        return dict(self.spec.sampling_rule)

    def bind_train_only_sampling_plan(self, plan: SamplingPlan) -> None:
        if not isinstance(plan, SamplingPlan):
            raise TypeError("candidate sampling requires a SamplingPlan")
        if (
            self._sampling_plan_sha256 is not None
            and self._sampling_plan_sha256 != plan.plan_hash
        ):
            raise CandidateAdapterContractError(
                "a candidate cannot rebind its SamplingPlan"
            )
        self._sampling_plan_sha256 = plan.plan_hash

    def assert_sampling_plan(self, plan: SamplingPlan) -> None:
        if (
            self._sampling_plan_sha256 is None
            or plan.plan_hash != self._sampling_plan_sha256
        ):
            raise ValueError("SamplingPlan hash mismatch")

    def _backend_config(
        self,
        base_config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = dict(base_config)
        config.update(
            {
                "requested_steps": self.spec.requested_updates,
                "requested_steps_total": self.spec.requested_updates,
                "checkpoint_interval_steps": (
                    self.spec.checkpoint_interval_steps
                ),
                "max_wall_seconds": self.spec.max_wall_seconds,
                "max_wall_seconds_total": self.spec.max_wall_seconds,
                "v2_6_candidate_id": self.spec.candidate_id,
                "v2_6_shared_trajectory_id": (
                    self.spec.shared_trajectory_id
                ),
                "v2_6_channel_loss_weights": dict(
                    self.spec.loss_weights
                ),
                "v2_6_sampling_rule": dict(self.spec.sampling_rule),
                "v2_6_train_zscore_state": (
                    self._zscore.checkpoint_state()
                    if self._zscore is not None
                    else None
                ),
            }
        )
        if self.spec.class_updates is not None:
            config["requested_steps_per_class"] = dict(
                self.spec.class_updates
            )
            config["max_wall_seconds_per_class"] = dict(
                self.spec.class_wall_seconds or {}
            )
        return config

    def fit_train_only(
        self,
        train: SequenceBatch,
        *,
        base_config: Mapping[str, Any],
        seed: int,
    ) -> None:
        forbidden = {
            "validation_batch",
            "validation_labels",
            "validation_lengths",
            "test_batch",
            "test_path",
            "fresh_test_path",
        }
        if forbidden & set(base_config):
            raise CandidateAdapterContractError(
                "candidate fitting received validation/test material"
            )
        transformed = train
        if self.spec.uses_train_zscore:
            self._zscore = TrainOnlyZScore.fit(train)
            transformed = self._zscore.transform_train(train)
        self._backend = self._backend_factory()
        self._backend.fit(
            transformed,
            config=self._backend_config(base_config),
            seed=seed,
        )

    def sample_validation(
        self,
        plan: SamplingPlan,
        *,
        seed: int,
    ) -> SyntheticBatch:
        self.assert_sampling_plan(plan)
        if self._backend is None:
            raise CandidateAdapterContractError(
                "candidate backend is not fitted or restored"
            )
        sample = self._backend.sample(plan, seed=seed)
        if self._zscore is not None:
            restored = self._zscore.inverse_synthetic(sample)
            assert isinstance(restored, SyntheticBatch)
            sample = restored
        return sample

    def attach_trained_backend_for_sampling(
        self,
        backend: Any,
        *,
        train_only_zscore: TrainOnlyZScore | None,
    ) -> None:
        if backend is None:
            raise CandidateAdapterContractError(
                "restored candidate backend is missing"
            )
        if self.spec.uses_train_zscore != (
            train_only_zscore is not None
        ):
            raise CandidateAdapterContractError(
                "restored z-score state differs from candidate definition"
            )
        self._backend = backend
        self._zscore = train_only_zscore

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "schema_version": "benchmark-v2.6-candidate-adapter-state-v1",
            "model_id": self.spec.model_id,
            "candidate_id": self.spec.candidate_id,
            "shared_trajectory_id": self.spec.shared_trajectory_id,
            "requested_updates": self.spec.requested_updates,
            "sampled_checkpoint_step": self.spec.sampled_checkpoint_step,
            "sampling_plan_sha256": self._sampling_plan_sha256,
            "train_only_zscore": (
                self._zscore.checkpoint_state()
                if self._zscore is not None
                else None
            ),
        }


def _default_backend_factory(model_id: str) -> BackendFactory:
    def factory() -> Any:
        from generators.candidate_model_backends_v2_6 import (
            BACKEND_REGISTRY_V2_6,
        )

        return BACKEND_REGISTRY_V2_6[model_id]()

    return factory


def build_candidate_adapter(
    spec: CandidateAdapterSpec,
    *,
    backend_factory: BackendFactory | None = None,
) -> CandidateAdapterV26:
    if spec.model_id not in {
        "ctgan_separate_class",
        "tvae_separate_class",
        "neural_sequence",
        "cof_seqgen",
    }:
        raise CandidateAdapterContractError(
            f"no v2.6 candidate adapter for {spec.model_id}"
        )
    return CandidateAdapterV26(
        spec,
        backend_factory=(
            backend_factory
            if backend_factory is not None
            else _default_backend_factory(spec.model_id)
        ),
    )
