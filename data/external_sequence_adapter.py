"""Source-only external sequence protocol for AMLSim and Sparkov.

This module contains deterministic data-contract logic only.  It does not
import a model, initialize CUDA, or execute training/sampling/evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from benchmarks.types import SequenceBatch


class ExternalProtocolError(RuntimeError):
    """Raised when an external-data protocol invariant is violated."""


@dataclass(frozen=True)
class RawWindow:
    dataset: str
    entity_id: Any
    ordinal: int
    transaction_ids: tuple[Any, ...]
    rows: pd.DataFrame

    @property
    def length(self) -> int:
        return len(self.transaction_ids)

    @property
    def y_entity(self) -> int:
        label_column = "IS_FRAUD" if self.dataset == "amlsim" else "is_fraud"
        return int(self.rows[label_column].astype(int).max())


@dataclass(frozen=True)
class SplitAssignment:
    entity_to_split: dict[Any, str]
    entities_by_split: dict[str, tuple[Any, ...]]
    counts_by_label: dict[str, dict[str, int]]
    seed: int


@dataclass(frozen=True)
class ExternalTransformState:
    dataset: str
    fit_role: str
    fit_transaction_count: int
    fit_entity_count: int
    fit_transaction_ids_sha256: str
    fit_entity_ids_sha256: str
    amount_log_mean: float
    amount_log_std: float
    gap_edges: tuple[float, ...]
    gap_tau: tuple[float, ...]
    receiver_to_code: dict[Any, int]
    pad_code: int = 0
    unk_code: int = 1

    @property
    def state_sha256(self) -> str:
        finite_edges = ["+inf" if np.isposinf(value) else value for value in self.gap_edges]
        payload = {
            "dataset": self.dataset,
            "fit_role": self.fit_role,
            "fit_transaction_count": self.fit_transaction_count,
            "fit_entity_count": self.fit_entity_count,
            "fit_transaction_ids_sha256": self.fit_transaction_ids_sha256,
            "fit_entity_ids_sha256": self.fit_entity_ids_sha256,
            "amount_log_mean": self.amount_log_mean,
            "amount_log_std": self.amount_log_std,
            "gap_edges": finite_edges,
            "gap_tau": self.gap_tau,
            "receiver_vocabulary": sorted(
                ((str(value), code) for value, code in self.receiver_to_code.items())
            ),
            "pad_code": self.pad_code,
            "unk_code": self.unk_code,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PreparedExternalSplit:
    batch: SequenceBatch
    windows: tuple[RawWindow, ...]


@dataclass(frozen=True)
class PreparedExternalDataset:
    dataset: str
    source_role: str
    train: PreparedExternalSplit
    validation: PreparedExternalSplit
    internal_test: PreparedExternalSplit
    transforms: ExternalTransformState
    split_assignment: SplitAssignment
    leakage_audit: dict[str, int]


def _ordered_rows(raw: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, str, str]:
    if dataset == "amlsim":
        required = {
            "TX_ID",
            "SENDER_ACCOUNT_ID",
            "RECEIVER_ACCOUNT_ID",
            "TX_AMOUNT",
            "TIMESTAMP",
            "IS_FRAUD",
        }
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ExternalProtocolError(f"missing AMLSim columns: {missing}")
        ordered = raw.copy()
        ordered["_protocol_timestamp"] = pd.to_numeric(
            ordered["TIMESTAMP"], errors="raise"
        )
        ordered = ordered.sort_values(
            ["SENDER_ACCOUNT_ID", "TIMESTAMP", "TX_ID"],
            kind="mergesort",
        ).reset_index(drop=True)
        return ordered, "SENDER_ACCOUNT_ID", "TX_ID"
    if dataset == "sparkov":
        required = {
            "trans_num",
            "cc_num",
            "merchant",
            "amt",
            "trans_date_trans_time",
            "is_fraud",
        }
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ExternalProtocolError(f"missing Sparkov columns: {missing}")
        ordered = raw.copy()
        ordered["_source_row_order"] = range(len(ordered))
        parsed = pd.to_datetime(
            ordered["trans_date_trans_time"],
            format="%Y-%m-%d %H:%M:%S",
            errors="raise",
        )
        ordered["_protocol_timestamp"] = parsed.astype("int64") / 1_000_000_000
        ordered = ordered.sort_values(
            ["cc_num", "_protocol_timestamp", "_source_row_order"],
            kind="mergesort",
        ).reset_index(drop=True)
        return ordered, "cc_num", "trans_num"
    raise ExternalProtocolError(f"unsupported external dataset: {dataset}")


def build_raw_windows(
    raw: pd.DataFrame,
    *,
    dataset: str,
    window_size: int = 32,
    minimum_tail: int = 16,
) -> list[RawWindow]:
    """Build stable, non-overlapping raw windows without fitting transforms."""

    ordered, entity_column, transaction_column = _ordered_rows(raw, dataset)
    if ordered[transaction_column].isna().any() or ordered[transaction_column].duplicated().any():
        raise ExternalProtocolError("duplicate transaction identity or missing transaction ID")
    if ordered[entity_column].isna().any():
        raise ExternalProtocolError("missing entity identity")
    windows: list[RawWindow] = []
    for entity, group in ordered.groupby(entity_column, sort=False):
        group = group.reset_index(drop=True)
        for start in range(0, len(group), window_size):
            segment = group.iloc[start : start + window_size].copy()
            if len(segment) < minimum_tail:
                continue
            segment["_protocol_window_ordinal"] = start // window_size
            segment["_protocol_window_position"] = np.arange(len(segment))
            windows.append(
                RawWindow(
                    dataset=dataset,
                    entity_id=entity,
                    ordinal=start // window_size,
                    transaction_ids=tuple(segment[transaction_column].tolist()),
                    rows=segment,
                )
            )
    return windows


def group_stratified_entity_split(
    raw: pd.DataFrame,
    *,
    dataset: str,
    seed: int,
) -> SplitAssignment:
    """Assign whole entities to a deterministic 70/15/15 label-stratified split."""

    if dataset == "amlsim":
        entity_column, label_column = "SENDER_ACCOUNT_ID", "IS_FRAUD"
    elif dataset == "sparkov":
        entity_column, label_column = "cc_num", "is_fraud"
    else:
        raise ExternalProtocolError(f"unsupported external dataset: {dataset}")
    missing = [column for column in (entity_column, label_column) if column not in raw]
    if missing:
        raise ExternalProtocolError(f"missing split columns: {missing}")
    labels = pd.to_numeric(raw[label_column], errors="raise")
    if not set(labels.unique()).issubset({0, 1}):
        raise ExternalProtocolError("fraud label must be binary")
    entity_labels = (
        pd.DataFrame({"entity": raw[entity_column], "label": labels})
        .groupby("entity", sort=True)["label"]
        .max()
    )

    return _split_entity_labels(entity_labels, seed=seed)


def _split_entity_labels(entity_labels: pd.Series, *, seed: int) -> SplitAssignment:
    rng = np.random.default_rng(seed)
    by_split: dict[str, list[Any]] = {
        "train": [],
        "validation": [],
        "internal_test": [],
    }
    counts: dict[str, dict[str, int]] = {}
    for label in (0, 1):
        entities = entity_labels.index[entity_labels == label].to_numpy(copy=True)
        if len(entities) < 3:
            raise ExternalProtocolError(
                f"label {label} has fewer than three entities; 70/15/15 stratification impossible"
            )
        rng.shuffle(entities)
        n_train = int(len(entities) * 0.70)
        n_validation = max(1, int(len(entities) * 0.15))
        n_train = min(n_train, len(entities) - n_validation - 1)
        partitions = {
            "train": entities[:n_train],
            "validation": entities[n_train : n_train + n_validation],
            "internal_test": entities[n_train + n_validation :],
        }
        counts[str(label)] = {}
        for split, values in partitions.items():
            by_split[split].extend(values.tolist())
            counts[str(label)][split] = len(values)

    entity_to_split = {
        entity: split for split, entities in by_split.items() for entity in entities
    }
    if len(entity_to_split) != len(entity_labels):
        raise ExternalProtocolError("entity split is not exhaustive and disjoint")
    return SplitAssignment(
        entity_to_split=entity_to_split,
        entities_by_split={split: tuple(values) for split, values in by_split.items()},
        counts_by_label=counts,
        seed=seed,
    )


def _dataset_columns(dataset: str) -> tuple[str, str, str, str]:
    if dataset == "amlsim":
        return "SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "TX_AMOUNT", "TX_ID"
    if dataset == "sparkov":
        return "cc_num", "merchant", "amt", "trans_num"
    raise ExternalProtocolError(f"unsupported external dataset: {dataset}")


def _identity_hash(values: list[Any]) -> str:
    payload = [
        {"type": type(value).__name__, "value": str(value)}
        for value in values
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def fit_train_transforms(
    train_rows: pd.DataFrame,
    *,
    dataset: str,
    gap_bins: int,
    fit_role: str,
) -> ExternalTransformState:
    """Fit amount, gap, and receiver state from training rows only."""

    if fit_role != "train":
        raise ExternalProtocolError("external transforms may be fit on train only")
    if gap_bins < 2:
        raise ExternalProtocolError("gap_bins must be at least two")
    ordered, entity_column, transaction_column = _ordered_rows(train_rows, dataset)
    _, receiver_column, amount_column, _ = _dataset_columns(dataset)
    if ordered[transaction_column].duplicated().any():
        raise ExternalProtocolError("duplicate transaction identity in transform fit")
    amount = pd.to_numeric(ordered[amount_column], errors="raise").to_numpy(float)
    if not np.isfinite(amount).all() or np.any(amount < 0):
        raise ExternalProtocolError("amount must be finite and nonnegative")
    amount_log = np.log1p(amount)
    amount_std = float(amount_log.std(ddof=0))
    if amount_std <= 0:
        amount_std = 1.0

    gap_groups = [entity_column]
    if "_protocol_window_ordinal" in ordered:
        gap_groups.append("_protocol_window_ordinal")
    gaps = (
        ordered.groupby(gap_groups, sort=False)["_protocol_timestamp"]
        .diff()
        .fillna(0.0)
        .to_numpy(float)
    )
    if np.any(gaps < 0) or not np.isfinite(gaps).all():
        raise ExternalProtocolError("sorted train gaps must be finite and nonnegative")
    positive = gaps[gaps > 0]
    if positive.size == 0:
        raise ExternalProtocolError("train rows contain no positive gap")
    edges = np.quantile(positive, np.linspace(0.0, 1.0, gap_bins + 1)).astype(float)
    edges[0] = 0.0
    edges[-1] = np.inf
    bins = np.clip(np.digitize(gaps, edges[1:], right=False), 0, gap_bins - 1)
    tau = tuple(
        float(np.median(gaps[bins == index])) if np.any(bins == index) else 0.0
        for index in range(gap_bins)
    )

    receivers = sorted(
        pd.unique(ordered[receiver_column]).tolist(),
        key=lambda value: str(value),
    )
    if any(pd.isna(value) for value in receivers):
        raise ExternalProtocolError("receiver is missing in train rows")
    receiver_to_code = {value: index + 2 for index, value in enumerate(receivers)}
    return ExternalTransformState(
        dataset=dataset,
        fit_role=fit_role,
        fit_transaction_count=len(ordered),
        fit_entity_count=int(ordered[entity_column].nunique()),
        fit_transaction_ids_sha256=_identity_hash(
            ordered[transaction_column].tolist()
        ),
        fit_entity_ids_sha256=_identity_hash(
            sorted(pd.unique(ordered[entity_column]).tolist(), key=str)
        ),
        amount_log_mean=float(amount_log.mean()),
        amount_log_std=amount_std,
        gap_edges=tuple(float(value) for value in edges),
        gap_tau=tau,
        receiver_to_code=receiver_to_code,
    )


def encode_windows(
    windows: list[RawWindow],
    state: ExternalTransformState,
    *,
    window_size: int = 32,
) -> SequenceBatch:
    """Encode raw windows with frozen train-only state and an explicit UNK code."""

    if not windows:
        raise ExternalProtocolError("cannot encode an empty window list")
    if any(window.dataset != state.dataset for window in windows):
        raise ExternalProtocolError("window dataset does not match transform state")
    _, receiver_column, amount_column, _ = _dataset_columns(state.dataset)
    n = len(windows)
    x_num = np.zeros((n, window_size, 1), dtype=np.float32)
    dt_bin = np.zeros((n, window_size), dtype=np.int64)
    x_cat = np.zeros((n, window_size, 1), dtype=np.int64)
    valid_mask = np.zeros((n, window_size), dtype=np.bool_)
    y_entity = np.zeros(n, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    entity_ids: list[Any] = []
    edges = np.asarray(state.gap_edges, dtype=float)

    for index, window in enumerate(windows):
        length = window.length
        if not 1 <= length <= window_size:
            raise ExternalProtocolError("window length is outside the fixed contract")
        rows = window.rows
        amount = pd.to_numeric(rows[amount_column], errors="raise").to_numpy(float)
        if not np.isfinite(amount).all() or np.any(amount < 0):
            raise ExternalProtocolError("amount must be finite and nonnegative")
        amount_encoded = (
            np.log1p(amount) - state.amount_log_mean
        ) / state.amount_log_std
        timestamps = pd.to_numeric(rows["_protocol_timestamp"], errors="raise").to_numpy(float)
        gaps = np.zeros(length, dtype=float)
        gaps[1:] = np.diff(timestamps)
        if np.any(gaps < 0):
            raise ExternalProtocolError("window timestamps are not sorted")
        gap_codes = np.clip(
            np.digitize(gaps, edges[1:], right=False),
            0,
            len(edges) - 2,
        )
        receiver_codes = np.asarray(
            [state.receiver_to_code.get(value, state.unk_code) for value in rows[receiver_column]],
            dtype=np.int64,
        )
        x_num[index, :length, 0] = amount_encoded.astype(np.float32)
        dt_bin[index, :length] = gap_codes.astype(np.int64)
        x_cat[index, :length, 0] = receiver_codes
        valid_mask[index, :length] = True
        y_entity[index] = window.y_entity
        lengths[index] = length
        entity_ids.append(window.entity_id)

    return SequenceBatch(
        x_num=x_num,
        dt_bin=dt_bin,
        x_cat=x_cat,
        valid_mask=valid_mask,
        y_entity=y_entity,
        lengths=lengths,
        entity_ids=np.asarray(entity_ids),
    )


def read_development_csv(
    path: str | Path,
    *,
    dataset: str,
    source_role: str,
) -> pd.DataFrame:
    """Read an allowed development source, refusing Sparkov public test first."""

    _assert_source_role(dataset, source_role, str(path))
    return pd.read_csv(path, low_memory=False)


def _assert_source_role(dataset: str, source_role: str, path_string: str = "") -> None:
    if dataset == "sparkov":
        if source_role != "fraudTrain_development" or "fraudtest" in path_string.lower():
            raise ExternalProtocolError(
                "Sparkov public fraudTest is locked for later temporal authorization"
            )
    elif dataset == "amlsim":
        if source_role != "transactions_development":
            raise ExternalProtocolError("invalid AMLSim development source role")
    else:
        raise ExternalProtocolError(f"unsupported external dataset: {dataset}")


def _prepared_split(windows: list[RawWindow], state: ExternalTransformState) -> PreparedExternalSplit:
    if not windows:
        raise ExternalProtocolError("group-stratified split produced no eligible windows")
    return PreparedExternalSplit(
        batch=encode_windows(windows, state),
        windows=tuple(windows),
    )


def prepare_external_dataset(
    raw: pd.DataFrame,
    *,
    dataset: str,
    source_role: str,
    split_seed: int,
    gap_bins: int,
) -> PreparedExternalDataset:
    """Prepare train/validation/internal-test batches under the frozen protocol."""

    _assert_source_role(dataset, source_role)
    windows = build_raw_windows(raw, dataset=dataset)
    if not windows:
        raise ExternalProtocolError("raw source contains no eligible 16-32 row windows")
    entity_labels: dict[Any, int] = {}
    for window in windows:
        entity_labels[window.entity_id] = max(
            entity_labels.get(window.entity_id, 0),
            window.y_entity,
        )
    assignment = _split_entity_labels(
        pd.Series(entity_labels, dtype=np.int64),
        seed=split_seed,
    )
    windows_by_split: dict[str, list[RawWindow]] = {
        "train": [],
        "validation": [],
        "internal_test": [],
    }
    for window in windows:
        windows_by_split[assignment.entity_to_split[window.entity_id]].append(window)

    train_rows = pd.concat(
        [window.rows for window in windows_by_split["train"]],
        ignore_index=True,
    )
    transforms = fit_train_transforms(
        train_rows,
        dataset=dataset,
        gap_bins=gap_bins,
        fit_role="train",
    )
    prepared = {
        split: _prepared_split(split_windows, transforms)
        for split, split_windows in windows_by_split.items()
    }

    entity_sets = {
        split: {window.entity_id for window in split_windows}
        for split, split_windows in windows_by_split.items()
    }
    window_sets = {
        split: {(window.dataset, window.entity_id, window.ordinal) for window in split_windows}
        for split, split_windows in windows_by_split.items()
    }
    transaction_sets = {
        split: {
            transaction_id
            for window in split_windows
            for transaction_id in window.transaction_ids
        }
        for split, split_windows in windows_by_split.items()
    }

    def overlap_count(sets: Mapping[str, set[Any]]) -> int:
        return sum(
            len(sets[left] & sets[right])
            for left, right in (
                ("train", "validation"),
                ("train", "internal_test"),
                ("validation", "internal_test"),
            )
        )

    leakage_audit = {
        "entity_overlap_count": overlap_count(entity_sets),
        "window_overlap_count": overlap_count(window_sets),
        "transaction_overlap_count": overlap_count(transaction_sets),
    }
    if any(leakage_audit.values()):
        raise ExternalProtocolError(f"external split leakage detected: {leakage_audit}")
    return PreparedExternalDataset(
        dataset=dataset,
        source_role=source_role,
        train=prepared["train"],
        validation=prepared["validation"],
        internal_test=prepared["internal_test"],
        transforms=transforms,
        split_assignment=assignment,
        leakage_audit=leakage_audit,
    )
