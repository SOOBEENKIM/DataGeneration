"""Canonical, model-agnostic data contract for the CoFSeqGen-SAF family.

This module is the model-agnostic boundary used by acquisition and dataset
adapters.  It defines the frames that every adapter must produce,
deterministic entity-level split assignment, and provenance proving that
learned transforms were fit from the training split only.  It performs no file
or network I/O itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "cofseqgen-saf-canonical-entity-sequence-v1"
SPLIT_NAMES = ("train", "validation", "test")
CORE_EVENT_COLUMNS = (
    "entity_id",
    "event_id",
    "event_index",
    "timestamp",
    "gap",
    "receiver_or_mark",
    "amount_or_numeric_value",
)


class SAFDataContractError(ValueError):
    """Raised when a canonical dataset or its provenance violates the contract."""


@dataclass(frozen=True)
class ReservedCategoryCodes:
    """Codes shared by every future SAF materializer.

    Ordinary learned categorical or gap-support codes must start at
    ``first_learned_code``.  A source missing value and a value unseen during
    train fitting are intentionally different states.
    """

    pad: int = 0
    unk: int = 1
    missing: int = 2
    first_learned_code: int = 3

    def __post_init__(self) -> None:
        values = (self.pad, self.unk, self.missing, self.first_learned_code)
        if values != (0, 1, 2, 3) or len(set(values)) != len(values):
            raise SAFDataContractError("reserved PAD/UNK/MISSING codes changed")


RESERVED_CODES = ReservedCategoryCodes()


def _identity_key(value: Any) -> tuple[str, str]:
    if value is None or bool(pd.isna(value)):
        raise SAFDataContractError("identity values must be non-missing scalars")
    return type(value).__name__, str(value)


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_records(values: Sequence[Any]) -> list[dict[str, str]]:
    return [
        {"type": value_type, "value": value_string}
        for value_type, value_string in sorted(_identity_key(value) for value in values)
    ]


@dataclass(frozen=True)
class CanonicalSchema:
    """Dataset-specific declarations layered on the common event schema."""

    dataset_id: str
    time_representation: str
    timestamp_unit: str
    static_context_columns: tuple[str, ...] = ()
    auxiliary_numeric_columns: tuple[str, ...] = ()
    auxiliary_categorical_columns: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "static_context_columns", tuple(self.static_context_columns))
        object.__setattr__(
            self,
            "auxiliary_numeric_columns",
            tuple(self.auxiliary_numeric_columns),
        )
        object.__setattr__(
            self,
            "auxiliary_categorical_columns",
            tuple(self.auxiliary_categorical_columns),
        )
        if self.schema_version != SCHEMA_VERSION:
            raise SAFDataContractError("canonical schema version changed")
        if not self.dataset_id or not self.timestamp_unit:
            raise SAFDataContractError("dataset_id and timestamp_unit are required")
        if self.time_representation not in {"absolute", "relative"}:
            raise SAFDataContractError("time_representation must be absolute or relative")
        declared = (
            self.static_context_columns
            + self.auxiliary_numeric_columns
            + self.auxiliary_categorical_columns
        )
        if len(set(declared)) != len(declared):
            raise SAFDataContractError("declared canonical columns must be unique")
        reserved = set(CORE_EVENT_COLUMNS) | {"split"}
        if any(column in reserved for column in declared):
            raise SAFDataContractError("declared columns collide with canonical columns")

    @property
    def schema_sha256(self) -> str:
        return _hash_payload(
            {
                "schema_version": self.schema_version,
                "dataset_id": self.dataset_id,
                "time_representation": self.time_representation,
                "timestamp_unit": self.timestamp_unit,
                "static_context_columns": self.static_context_columns,
                "auxiliary_numeric_columns": self.auxiliary_numeric_columns,
                "auxiliary_categorical_columns": self.auxiliary_categorical_columns,
                "reserved_codes": RESERVED_CODES.__dict__,
            }
        )


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    observed = tuple(frame.columns)
    if observed != expected:
        raise SAFDataContractError(
            f"{frame_name} columns must be exactly {expected}, observed {observed}"
        )


def _entity_key_set(values: Sequence[Any]) -> set[tuple[str, str]]:
    return {_identity_key(value) for value in values}


@dataclass(frozen=True)
class CanonicalEntitySequenceDataset:
    """Three canonical frames with strict entity and temporal invariants.

    ``static_context`` has one row per entity. ``events`` contains ordered
    event rows. ``entity_splits`` assigns every entity to exactly one of the
    three development roles.  The first event's gap is NaN; zero is a valid
    observed gap only for later simultaneous events.
    """

    schema: CanonicalSchema
    static_context: pd.DataFrame
    events: pd.DataFrame
    entity_splits: pd.DataFrame
    gap_tolerance: float = 1e-9
    required_splits: tuple[str, ...] = SPLIT_NAMES
    source_split_assignment_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.schema, CanonicalSchema):
            raise SAFDataContractError("schema must be CanonicalSchema")
        object.__setattr__(self, "required_splits", tuple(self.required_splits))
        if not self.required_splits or not set(self.required_splits) <= set(SPLIT_NAMES):
            raise SAFDataContractError("required_splits must be a non-empty canonical subset")
        for name in ("static_context", "events", "entity_splits"):
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame):
                raise SAFDataContractError(f"{name} must be a pandas DataFrame")
            object.__setattr__(self, name, frame.copy(deep=True))
        if not np.isfinite(self.gap_tolerance) or self.gap_tolerance < 0:
            raise SAFDataContractError("gap_tolerance must be finite and nonnegative")
        self._validate()

    def _validate(self) -> None:
        static_columns = ("entity_id",) + self.schema.static_context_columns
        event_columns = (
            CORE_EVENT_COLUMNS
            + self.schema.auxiliary_numeric_columns
            + self.schema.auxiliary_categorical_columns
        )
        _require_exact_columns(
            self.static_context,
            static_columns,
            frame_name="static_context",
        )
        _require_exact_columns(self.events, event_columns, frame_name="events")
        _require_exact_columns(
            self.entity_splits,
            ("entity_id", "split"),
            frame_name="entity_splits",
        )
        if self.static_context.empty or self.events.empty or self.entity_splits.empty:
            raise SAFDataContractError("canonical frames must be non-empty")
        if self.static_context["entity_id"].duplicated().any():
            raise SAFDataContractError("static_context must have one row per entity")
        if self.entity_splits["entity_id"].duplicated().any():
            raise SAFDataContractError("an entity may belong to only one split")
        if self.events[["entity_id", "event_id"]].duplicated().any():
            raise SAFDataContractError("event_id must be unique within each entity")

        static_entities = _entity_key_set(self.static_context["entity_id"].tolist())
        event_entities = _entity_key_set(self.events["entity_id"].tolist())
        split_entities = _entity_key_set(self.entity_splits["entity_id"].tolist())
        if static_entities != event_entities or static_entities != split_entities:
            raise SAFDataContractError(
                "static, event, and split entity sets must match exactly"
            )

        split_values = set(self.entity_splits["split"].tolist())
        if split_values != set(self.required_splits):
            raise SAFDataContractError(
                "entity_splits must contain exactly the dataset view's required roles"
            )

        if self.events["event_id"].isna().any():
            raise SAFDataContractError("event_id must be non-missing")
        event_indices = pd.to_numeric(self.events["event_index"], errors="raise")
        if not np.isfinite(event_indices.to_numpy(float)).all() or np.any(
            event_indices.to_numpy(float) != np.floor(event_indices.to_numpy(float))
        ):
            raise SAFDataContractError("event_index must contain finite integers")
        timestamps = pd.to_numeric(self.events["timestamp"], errors="raise")
        if not np.isfinite(timestamps.to_numpy(float)).all():
            raise SAFDataContractError("timestamp must be finite in canonical units")
        amounts = pd.to_numeric(
            self.events["amount_or_numeric_value"],
            errors="raise",
        ).to_numpy(float)
        if np.isinf(amounts).any() or np.isnan(amounts).all():
            raise SAFDataContractError(
                "amount_or_numeric_value permits missing values but needs finite observations"
            )
        if self.events["receiver_or_mark"].notna().sum() == 0:
            raise SAFDataContractError("receiver_or_mark needs at least one observed value")
        for column in self.schema.auxiliary_numeric_columns:
            values = pd.to_numeric(self.events[column], errors="raise").to_numpy(float)
            if np.isinf(values).any() or np.isnan(values).all():
                raise SAFDataContractError(
                    f"auxiliary numeric column {column} needs finite observations"
                )

        for entity_id, group in self.events.groupby("entity_id", sort=False, dropna=False):
            _identity_key(entity_id)
            indices = pd.to_numeric(group["event_index"], errors="raise").to_numpy(int)
            if not np.array_equal(indices, np.arange(len(group), dtype=int)):
                raise SAFDataContractError(
                    "events must be stored in contiguous event_index order per entity"
                )
            entity_timestamps = pd.to_numeric(
                group["timestamp"], errors="raise"
            ).to_numpy(float)
            observed_differences = np.diff(entity_timestamps)
            if np.any(observed_differences < 0):
                raise SAFDataContractError("timestamps must be nondecreasing within entity")
            gaps = pd.to_numeric(group["gap"], errors="coerce").to_numpy(float)
            if not np.isnan(gaps[0]):
                raise SAFDataContractError(
                    "the first event gap must be missing, never encoded as zero"
                )
            if len(gaps) > 1:
                if not np.isfinite(gaps[1:]).all() or np.any(gaps[1:] < 0):
                    raise SAFDataContractError(
                        "non-first gaps must be finite and nonnegative"
                    )
                if not np.allclose(
                    gaps[1:],
                    observed_differences,
                    rtol=0.0,
                    atol=self.gap_tolerance,
                ):
                    raise SAFDataContractError(
                        "gap must equal the timestamp difference in canonical units"
                    )

    @property
    def split_assignment_sha256(self) -> str:
        if self.source_split_assignment_sha256 is not None:
            return self.source_split_assignment_sha256
        return hash_entity_split_assignment(self.entity_splits)

    def entity_ids_for_split(self, split: str) -> tuple[Any, ...]:
        if split not in self.required_splits:
            raise SAFDataContractError(f"split is sealed or unavailable in this view: {split}")
        values = self.entity_splits.loc[
            self.entity_splits["split"] == split,
            "entity_id",
        ].tolist()
        return tuple(sorted(values, key=_identity_key))


def hash_entity_split_assignment(entity_splits: pd.DataFrame) -> str:
    """Hash a complete assignment without reading any event or static content."""

    _require_exact_columns(
        entity_splits,
        ("entity_id", "split"),
        frame_name="entity_splits",
    )
    if entity_splits["entity_id"].duplicated().any():
        raise SAFDataContractError("an entity may belong to only one split")
    if set(entity_splits["split"]) != set(SPLIT_NAMES):
        raise SAFDataContractError("source split assignment must contain all canonical roles")
    rows = sorted(
        (
            *_identity_key(entity_id),
            str(split),
        )
        for entity_id, split in entity_splits.itertuples(index=False, name=None)
    )
    return _hash_payload(rows)


def _partition_sizes(n: int, train_fraction: float, validation_fraction: float) -> tuple[int, int, int]:
    if n < 3:
        raise SAFDataContractError("each split stratum needs at least three entities")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise SAFDataContractError("split fractions must lie strictly between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise SAFDataContractError("train and validation fractions leave no test split")
    n_train = max(1, int(np.floor(n * train_fraction)))
    n_validation = max(1, int(np.floor(n * validation_fraction)))
    if n_train + n_validation >= n:
        n_train = n - n_validation - 1
    if n_train < 1:
        raise SAFDataContractError("split fractions cannot allocate all three roles")
    return n_train, n_validation, n - n_train - n_validation


def build_entity_split_assignment(
    entity_ids: Sequence[Any],
    *,
    seed: int,
    strata: Mapping[Any, Any] | None = None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> pd.DataFrame:
    """Create a deterministic entity-disjoint split before any transform fit."""

    ids = list(entity_ids)
    keys = [_identity_key(value) for value in ids]
    if len(set(keys)) != len(keys):
        raise SAFDataContractError("entity_ids must be unique")
    if not ids:
        raise SAFDataContractError("entity_ids must be non-empty")
    if strata is None:
        grouped: dict[tuple[str, str], list[Any]] = {("str", "all"): ids}
    else:
        stratum_by_key = {_identity_key(entity): value for entity, value in strata.items()}
        if set(stratum_by_key) != set(keys):
            raise SAFDataContractError("strata must map every entity exactly once")
        grouped = {}
        for entity in ids:
            stratum = stratum_by_key[_identity_key(entity)]
            grouped.setdefault(_identity_key(stratum), []).append(entity)

    ordered_strata = sorted(grouped)
    total = len(ids)
    targets = list(_partition_sizes(total, train_fraction, validation_fraction))
    if any(target < len(grouped) for target in targets):
        raise SAFDataContractError(
            "too many strata to place at least one entity from each in every split"
        )
    fractions = np.asarray(
        [train_fraction, validation_fraction, 1 - train_fraction - validation_fraction]
    )
    allocations = {
        key: list(_partition_sizes(len(grouped[key]), train_fraction, validation_fraction))
        for key in ordered_strata
    }
    observed = np.sum(np.asarray(list(allocations.values()), dtype=int), axis=0)
    while not np.array_equal(observed, np.asarray(targets)):
        donors = [index for index in range(3) if observed[index] > targets[index]]
        recipients = [index for index in range(3) if observed[index] < targets[index]]
        moves: list[tuple[float, tuple[str, str], int, int]] = []
        for donor in donors:
            for recipient in recipients:
                for key in ordered_strata:
                    counts = allocations[key]
                    if counts[donor] <= 1:
                        continue
                    ideal = len(grouped[key]) * fractions
                    before = float(np.abs(np.asarray(counts) - ideal).sum())
                    changed = counts.copy()
                    changed[donor] -= 1
                    changed[recipient] += 1
                    after = float(np.abs(np.asarray(changed) - ideal).sum())
                    moves.append((after - before, key, donor, recipient))
        if not moves:
            raise SAFDataContractError("cannot apportion exact split totals across strata")
        _, key, donor, recipient = min(moves)
        allocations[key][donor] -= 1
        allocations[key][recipient] += 1
        observed[donor] -= 1
        observed[recipient] += 1

    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(len(grouped))
    rows: list[tuple[Any, str]] = []
    for stratum_key, child_seed in zip(ordered_strata, child_sequences):
        values = sorted(grouped[stratum_key], key=_identity_key)
        n_train, n_validation, _ = allocations[stratum_key]
        order = np.random.default_rng(child_seed).permutation(len(values))
        shuffled = [values[index] for index in order]
        rows.extend((entity, "train") for entity in shuffled[:n_train])
        rows.extend(
            (entity, "validation")
            for entity in shuffled[n_train : n_train + n_validation]
        )
        rows.extend((entity, "test") for entity in shuffled[n_train + n_validation :])
    rows.sort(key=lambda row: _identity_key(row[0]))
    return pd.DataFrame(rows, columns=("entity_id", "split"))


@dataclass(frozen=True)
class TrainOnlyFitProvenance:
    """Immutable identity evidence for a future learned preprocessing state."""

    schema_version: str
    schema_sha256: str
    split_assignment_sha256: str
    fit_split: str
    fit_entity_count: int
    fit_entity_ids_sha256: str
    fit_event_count: int
    fit_event_ids_sha256: str


def make_train_only_fit_provenance(
    dataset: CanonicalEntitySequenceDataset,
) -> TrainOnlyFitProvenance:
    train_ids = dataset.entity_ids_for_split("train")
    train_key_set = _entity_key_set(train_ids)
    train_events = dataset.events[
        dataset.events["entity_id"].map(_identity_key).isin(train_key_set)
    ]
    event_identities = sorted(
        (
            *_identity_key(entity_id),
            *_identity_key(event_id),
        )
        for entity_id, event_id in train_events[
            ["entity_id", "event_id"]
        ].itertuples(index=False, name=None)
    )
    return TrainOnlyFitProvenance(
        schema_version=SCHEMA_VERSION,
        schema_sha256=dataset.schema.schema_sha256,
        split_assignment_sha256=dataset.split_assignment_sha256,
        fit_split="train",
        fit_entity_count=len(train_ids),
        fit_entity_ids_sha256=_hash_payload(_identity_records(train_ids)),
        fit_event_count=len(train_events),
        fit_event_ids_sha256=_hash_payload(event_identities),
    )


def validate_train_only_fit_provenance(
    dataset: CanonicalEntitySequenceDataset,
    provenance: TrainOnlyFitProvenance,
) -> None:
    expected = make_train_only_fit_provenance(dataset)
    if provenance != expected:
        raise SAFDataContractError(
            "transform provenance is not the exact canonical training split"
        )
