from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch
from .sampling_plan import SamplingPlan


class EmpiricalConditionalBlock:
    name = "empirical_conditional_block"

    def __init__(self, block_length: int | str):
        self.block_length = block_length

    def fit(self, train: SequenceBatch, *, config: Mapping[str, Any] = {}, seed: int = 0) -> None:
        self.train = train
        self.by_label = {y: np.flatnonzero(train.y_entity == y) for y in (0, 1)}

    def sample(self, plan: SamplingPlan, *, seed: int) -> SyntheticBatch:
        rng = np.random.default_rng(seed)
        n, max_l = plan.valid_mask.shape
        x_num = np.zeros((n, max_l, self.train.x_num.shape[-1]), np.float32)
        dt_bin = np.zeros((n, max_l), np.int64)
        x_cat = np.zeros((n, max_l, self.train.x_cat.shape[-1]), np.int64)
        for i, (label, target_length) in enumerate(zip(plan.y_entity, plan.lengths)):
            pos = 0
            while pos < target_length:
                source = int(rng.choice(self.by_label[int(label)]))
                source_len = int(self.train.lengths[source])
                block = source_len if self.block_length == "full" else int(self.block_length)
                take = min(block, int(target_length) - pos, source_len)
                start = int(rng.integers(0, source_len - take + 1))
                dest = slice(pos, pos + take)
                src = slice(start, start + take)
                x_num[i, dest], dt_bin[i, dest], x_cat[i, dest] = (
                    self.train.x_num[source, src],
                    self.train.dt_bin[source, src],
                    self.train.x_cat[source, src],
                )
                pos += take
        return SyntheticBatch(
            x_num, dt_bin, x_cat, plan.valid_mask.copy(),
            plan.y_entity.copy(), plan.lengths.copy(),
        )
