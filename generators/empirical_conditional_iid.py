from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch
from .sampling_plan import SamplingPlan


class EmpiricalConditionalIID:
    name = "empirical_conditional_iid"

    def fit(self, train: SequenceBatch, *, config: Mapping[str, Any] = {}, seed: int = 0) -> None:
        self.pools = {}
        row_y = np.repeat(train.y_entity, train.lengths)
        self.pools["x_num"] = train.x_num[train.valid_mask]
        self.pools["dt_bin"] = train.dt_bin[train.valid_mask]
        self.pools["x_cat"] = train.x_cat[train.valid_mask]
        self.row_y = row_y

    def sample(self, plan: SamplingPlan, *, seed: int) -> SyntheticBatch:
        rng = np.random.default_rng(seed)
        n, length = plan.valid_mask.shape
        x_num = np.zeros((n, length, self.pools["x_num"].shape[-1]), np.float32)
        dt_bin = np.zeros((n, length), np.int64)
        x_cat = np.zeros((n, length, self.pools["x_cat"].shape[-1]), np.int64)
        for label in (0, 1):
            positions = np.argwhere(plan.valid_mask & (plan.y_entity[:, None] == label))
            pool = np.flatnonzero(self.row_y == label)
            chosen = rng.choice(pool, len(positions), replace=True)
            x_num[positions[:, 0], positions[:, 1]] = self.pools["x_num"][chosen]
            dt_bin[positions[:, 0], positions[:, 1]] = self.pools["dt_bin"][chosen]
            x_cat[positions[:, 0], positions[:, 1]] = self.pools["x_cat"][chosen]
        return SyntheticBatch(
            x_num, dt_bin, x_cat, plan.valid_mask.copy(),
            plan.y_entity.copy(), plan.lengths.copy(),
        )
