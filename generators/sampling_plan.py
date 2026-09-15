from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from benchmarks.types import SequenceBatch


@dataclass(frozen=True)
class SamplingPlan:
    y_entity: np.ndarray
    lengths: np.ndarray
    valid_mask: np.ndarray
    plan_hash: str

    @property
    def plan_id(self) -> str:
        """Backward-compatible name for the full immutable SHA-256."""
        return self.plan_hash

    @classmethod
    def from_batch(cls, batch: SequenceBatch) -> "SamplingPlan":
        payload = batch.y_entity.tobytes() + batch.lengths.tobytes()
        return cls(
            batch.y_entity.copy(), batch.lengths.copy(), batch.valid_mask.copy(),
            hashlib.sha256(payload).hexdigest(),
        )

    @classmethod
    def from_train_policy(
        cls,
        train: SequenceBatch,
        *,
        entity_count: int,
        seed: int,
    ) -> "SamplingPlan":
        """Fit label prevalence and conditional lengths using train only."""
        if entity_count < 1:
            raise ValueError("entity_count must be positive")
        rng = np.random.default_rng(seed)
        prevalence = float(train.y_entity.mean())
        labels = (rng.random(entity_count) < prevalence).astype(np.int64)
        lengths = np.empty(entity_count, dtype=np.int64)
        for label in (0, 1):
            pool = train.lengths[train.y_entity == label]
            if not len(pool):
                raise ValueError("both labels must occur in train")
            selected = labels == label
            lengths[selected] = rng.choice(
                pool,
                int(selected.sum()),
                replace=True,
            )
        max_length = train.valid_mask.shape[1]
        valid_mask = np.arange(max_length)[None, :] < lengths[:, None]
        payload = (
            labels.tobytes()
            + lengths.tobytes()
            + b"benchmark-v2.5-train-only-plan"
        )
        return cls(
            labels,
            lengths,
            valid_mask,
            hashlib.sha256(payload).hexdigest(),
        )

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, y_entity=self.y_entity, lengths=self.lengths,
            valid_mask=self.valid_mask,
            plan_hash=np.array(self.plan_hash),
        )
