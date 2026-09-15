from __future__ import annotations

from typing import Any, Mapping, Protocol

from benchmarks.types import SequenceBatch, SyntheticBatch
from .sampling_plan import SamplingPlan


class SequenceGenerator(Protocol):
    name: str

    def fit(self, train: SequenceBatch, *, config: Mapping[str, Any], seed: int) -> None: ...
    def sample(self, plan: SamplingPlan, *, seed: int) -> SyntheticBatch: ...
