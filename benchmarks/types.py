from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _validate_arrays(obj: Any, has_ids: bool) -> None:
    n, length, d_num = obj.x_num.shape
    if obj.x_num.dtype != np.float32 or d_num < 1:
        raise TypeError("x_num must be float32 [N,L,d_num]")
    if obj.dt_bin.shape != (n, length) or obj.dt_bin.dtype != np.int64:
        raise TypeError("dt_bin must be int64 [N,L]")
    if obj.x_cat.shape[:2] != (n, length) or obj.x_cat.dtype != np.int64:
        raise TypeError("x_cat must be int64 [N,L,d_cat]")
    if obj.valid_mask.shape != (n, length) or obj.valid_mask.dtype != np.bool_:
        raise TypeError("valid_mask must be bool [N,L]")
    if obj.y_entity.shape != (n,) or obj.y_entity.dtype != np.int64:
        raise TypeError("y_entity must be int64 [N]")
    if obj.lengths.shape != (n,) or obj.lengths.dtype != np.int64:
        raise TypeError("lengths must be int64 [N]")
    expected = np.arange(length)[None, :] < obj.lengths[:, None]
    if not np.array_equal(obj.valid_mask, expected):
        raise ValueError("canonical v2 mask must be right-padded and prefix-contiguous")
    if not np.array_equal(obj.valid_mask.sum(1), obj.lengths):
        raise ValueError("mask/length mismatch")
    pad = ~obj.valid_mask
    if np.any(obj.x_num[pad] != 0) or np.any(obj.dt_bin[pad] != 0) or np.any(obj.x_cat[pad] != 0):
        raise ValueError("all padded values must be zero")
    if not np.isfinite(obj.x_num).all() or not np.isin(obj.y_entity, [0, 1]).all():
        raise ValueError("non-finite data or invalid labels")
    if has_ids and np.asarray(obj.entity_ids).shape != (n,):
        raise ValueError("entity_ids must have shape [N]")


@dataclass(frozen=True)
class SequenceBatch:
    x_num: np.ndarray
    dt_bin: np.ndarray
    x_cat: np.ndarray
    valid_mask: np.ndarray
    y_entity: np.ndarray
    lengths: np.ndarray
    entity_ids: np.ndarray

    def __post_init__(self) -> None:
        _validate_arrays(self, True)

    def y_position(self) -> np.ndarray:
        return self.y_entity[:, None] * self.valid_mask


@dataclass(frozen=True)
class SyntheticBatch:
    x_num: np.ndarray
    dt_bin: np.ndarray
    x_cat: np.ndarray
    valid_mask: np.ndarray
    y_entity: np.ndarray
    lengths: np.ndarray

    def __post_init__(self) -> None:
        _validate_arrays(self, False)


@dataclass(frozen=True)
class DatasetBundle:
    train: SequenceBatch
    test: SequenceBatch
    metadata: Mapping[str, Any]
