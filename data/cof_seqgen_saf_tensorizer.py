"""Train-only tensorization for canonical CoFSeqGen-SAF datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from data.cof_seqgen_saf_contract import (
    CanonicalEntitySequenceDataset,
    CanonicalSchema,
    RESERVED_CODES,
    SAFDataContractError,
    SPLIT_NAMES,
    hash_entity_split_assignment,
)
from models.cof_seqgen_saf import GapSupportState, fit_train_only_gap_support


STATIC_COLUMN_ROLES: Mapping[str, Mapping[str, Tuple[str, ...]]] = {
    "controlled_coupling_dgp": {
        "categorical": ("entity_label",),
        "numeric": (),
    },
    "amlsim": {
        "categorical": ("account_is_fraud", "tx_behavior_id"),
        "numeric": ("initial_balance",),
    },
    "sparkov": {
        "categorical": (
            "cardholder_gender",
            "cardholder_state",
            "entity_any_fraud",
        ),
        "numeric": ("birth_year", "city_population"),
    },
    "berka": {
        "categorical": (
            "account_frequency",
            "account_district_id",
            "city",
            "region",
        ),
        "numeric": ("account_open_day",),
    },
    "hm": {
        "categorical": (
            "club_member_status",
            "fashion_news_frequency",
            "postal_code",
        ),
        "numeric": ("age",),
    },
    "citi_bike": {"categorical": (), "numeric": ()},
}


def _dataset_family(dataset_id: str) -> str:
    if dataset_id.startswith("controlled_coupling"):
        return "controlled_coupling_dgp"
    if dataset_id.startswith("berka"):
        return "berka"
    return dataset_id


def _typed_key(value: Any) -> Optional[str]:
    if value is None or bool(pd.isna(value)):
        return None
    # Fitting a pandas Series yields Python scalars in some paths, while
    # selecting a row yields NumPy scalars.  Canonicalize both before adding
    # the type tag so an observed static category cannot become <UNK> solely
    # because pandas changed its scalar wrapper.
    if isinstance(value, np.generic):
        value = value.item()
    return json.dumps(
        {"type": type(value).__name__, "value": str(value)},
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class CategoryCodec:
    learned_keys: Tuple[str, ...]
    fit_split: str = "train"

    def __post_init__(self) -> None:
        if self.fit_split != "train":
            raise SAFDataContractError("categorical codec must be train-only")
        if len(set(self.learned_keys)) != len(self.learned_keys):
            raise SAFDataContractError("categorical learned keys must be unique")

    @property
    def vocab_size(self) -> int:
        return RESERVED_CODES.first_learned_code + len(self.learned_keys)

    @classmethod
    def fit(cls, values: Iterable[Any]) -> "CategoryCodec":
        keys = sorted({_typed_key(value) for value in values} - {None})
        return cls(tuple(keys))

    def encode(self, values: Iterable[Any]) -> np.ndarray:
        lookup = {
            key: index + RESERVED_CODES.first_learned_code
            for index, key in enumerate(self.learned_keys)
        }
        encoded = []
        for value in values:
            key = _typed_key(value)
            if key is None:
                encoded.append(RESERVED_CODES.missing)
            else:
                encoded.append(lookup.get(key, RESERVED_CODES.unk))
        return np.asarray(encoded, dtype=np.int64)

    def decode(self, codes: Iterable[int]) -> list[Optional[str]]:
        values = [json.loads(key)["value"] for key in self.learned_keys]
        output: list[Optional[str]] = []
        for code in codes:
            code = int(code)
            if code == RESERVED_CODES.missing:
                output.append(None)
            elif code < RESERVED_CODES.first_learned_code:
                output.append("<UNK>")
            else:
                position = code - RESERVED_CODES.first_learned_code
                output.append(values[position] if position < len(values) else "<UNK>")
        return output


@dataclass(frozen=True)
class NumericCodec:
    mean: float
    scale: float
    transform: str
    fit_split: str = "train"

    def __post_init__(self) -> None:
        if self.fit_split != "train":
            raise SAFDataContractError("numeric codec must be train-only")
        if not np.isfinite(self.mean) or not np.isfinite(self.scale) or self.scale <= 0:
            raise SAFDataContractError("numeric codec parameters must be finite")
        if self.transform not in {"zscore", "signed_log1p_zscore"}:
            raise SAFDataContractError("unknown numeric transform")

    @staticmethod
    def _signed_log1p(values: np.ndarray) -> np.ndarray:
        return np.sign(values) * np.log1p(np.abs(values))

    @classmethod
    def fit(cls, values: Iterable[Any], *, transform: str) -> "NumericCodec":
        numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
        finite = numeric[np.isfinite(numeric)]
        if not finite.size:
            raise SAFDataContractError("numeric codec needs finite train observations")
        transformed = (
            cls._signed_log1p(finite)
            if transform == "signed_log1p_zscore"
            else finite
        )
        scale = float(np.std(transformed))
        return cls(
            mean=float(np.mean(transformed)),
            scale=max(scale, 1e-8),
            transform=transform,
        )

    def encode(self, values: Iterable[Any]) -> np.ndarray:
        numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
        transformed = (
            self._signed_log1p(numeric)
            if self.transform == "signed_log1p_zscore"
            else numeric
        )
        return ((transformed - self.mean) / self.scale).astype(np.float32)

    def decode(self, values: Iterable[float]) -> np.ndarray:
        transformed = np.asarray(list(values), dtype=np.float64) * self.scale + self.mean
        if self.transform == "signed_log1p_zscore":
            # Numerical guard only: preserve unbounded sampling semantics while
            # preventing expm1 overflow from creating invalid infinities.
            return np.sign(transformed) * np.expm1(np.minimum(np.abs(transformed), 700.0))
        return transformed


@dataclass(frozen=True)
class SAFTensorizerState:
    dataset_id: str
    schema_sha256: str
    split_assignment_sha256: str
    receiver_codec: CategoryCodec
    auxiliary_categorical_codecs: Tuple[Tuple[str, CategoryCodec], ...]
    event_numeric_codecs: Tuple[Tuple[str, NumericCodec], ...]
    static_categorical_codecs: Tuple[Tuple[str, CategoryCodec], ...]
    static_numeric_codecs: Tuple[Tuple[str, NumericCodec], ...]
    gap_support: GapSupportState
    fit_split: str = "train"

    def __post_init__(self) -> None:
        if self.fit_split != "train":
            raise SAFDataContractError("tensorizer state must be train-only")

    def model_config_kwargs(self) -> Dict[str, object]:
        return {
            "receiver_vocab_size": self.receiver_codec.vocab_size,
            "auxiliary_categorical_vocab_sizes": tuple(
                codec.vocab_size for _, codec in self.auxiliary_categorical_codecs
            ),
            "auxiliary_numeric_dim": max(0, len(self.event_numeric_codecs) - 1),
            "static_categorical_vocab_sizes": tuple(
                codec.vocab_size for _, codec in self.static_categorical_codecs
            ),
            "static_dim": len(self.static_numeric_codecs),
        }

    def to_dict(self) -> Dict[str, object]:
        def category(codec: CategoryCodec) -> Dict[str, object]:
            return asdict(codec)

        def numeric(codec: NumericCodec) -> Dict[str, object]:
            return asdict(codec)

        return {
            "schema_version": "cof-seqgen-saf-tensorizer-state-v2",
            "dataset_id": self.dataset_id,
            "schema_sha256": self.schema_sha256,
            "split_assignment_sha256": self.split_assignment_sha256,
            "fit_split": self.fit_split,
            "receiver_codec": category(self.receiver_codec),
            "auxiliary_categorical_codecs": {
                name: category(codec)
                for name, codec in self.auxiliary_categorical_codecs
            },
            "event_numeric_codecs": {
                name: numeric(codec) for name, codec in self.event_numeric_codecs
            },
            "static_categorical_codecs": {
                name: category(codec)
                for name, codec in self.static_categorical_codecs
            },
            "static_numeric_codecs": {
                name: numeric(codec) for name, codec in self.static_numeric_codecs
            },
            "gap_support": self.gap_support.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SAFTensorizerState":
        if raw.get("schema_version") not in {
            "cof-seqgen-saf-tensorizer-state-v1",
            "cof-seqgen-saf-tensorizer-state-v2",
        }:
            raise SAFDataContractError("unknown tensorizer state schema")

        def category(item: Mapping[str, Any]) -> CategoryCodec:
            return CategoryCodec(
                learned_keys=tuple(item["learned_keys"]),
                fit_split=item.get("fit_split", "train"),
            )

        def numeric(item: Mapping[str, Any]) -> NumericCodec:
            return NumericCodec(
                mean=float(item["mean"]),
                scale=float(item["scale"]),
                transform=str(item["transform"]),
                fit_split=item.get("fit_split", "train"),
            )

        support = raw["gap_support"]
        return cls(
            dataset_id=str(raw["dataset_id"]),
            schema_sha256=str(raw["schema_sha256"]),
            split_assignment_sha256=str(raw["split_assignment_sha256"]),
            receiver_codec=category(raw["receiver_codec"]),
            auxiliary_categorical_codecs=tuple(
                (name, category(item))
                for name, item in raw["auxiliary_categorical_codecs"].items()
            ),
            event_numeric_codecs=tuple(
                (name, numeric(item))
                for name, item in raw["event_numeric_codecs"].items()
            ),
            static_categorical_codecs=tuple(
                (name, category(item))
                for name, item in raw["static_categorical_codecs"].items()
            ),
            static_numeric_codecs=tuple(
                (name, numeric(item))
                for name, item in raw["static_numeric_codecs"].items()
            ),
            gap_support=GapSupportState(
                representatives=tuple(support["representatives"]),
                upper_bounds=tuple(
                    np.inf if value is None else value
                    for value in support["upper_bounds"]
                ),
                source_unique_count=int(support["source_unique_count"]),
                fit_split=support.get("fit_split", "train"),
                zero_is_explicit=bool(support["zero_is_explicit"]),
            ),
            fit_split=raw.get("fit_split", "train"),
        )


@dataclass(frozen=True)
class TensorizedSequence:
    entity_id: Any
    gap: np.ndarray
    receiver: np.ndarray
    numeric_value: np.ndarray
    auxiliary_categorical: Tuple[np.ndarray, ...]
    auxiliary_numeric: np.ndarray
    static_categorical: Tuple[int, ...]
    static_numeric: np.ndarray

    @property
    def length(self) -> int:
        return len(self.gap)


class SAFTensorizer:
    def __init__(self, state: SAFTensorizerState) -> None:
        self.state = state

    @classmethod
    def fit(
        cls,
        dataset: CanonicalEntitySequenceDataset,
        *,
        max_positive_gap_states: int = 31,
    ) -> "SAFTensorizer":
        family = _dataset_family(dataset.schema.dataset_id)
        if family not in STATIC_COLUMN_ROLES:
            raise SAFDataContractError(
                f"static roles are not preregistered for {dataset.schema.dataset_id}"
            )
        roles = STATIC_COLUMN_ROLES[family]
        declared = set(dataset.schema.static_context_columns)
        role_columns = set(roles["categorical"]) | set(roles["numeric"])
        if declared != role_columns:
            raise SAFDataContractError(
                f"static role declaration mismatch: {sorted(declared ^ role_columns)}"
            )
        train_ids = set(dataset.entity_ids_for_split("train"))
        train_events = dataset.events[dataset.events["entity_id"].isin(train_ids)]
        train_static = dataset.static_context[
            dataset.static_context["entity_id"].isin(train_ids)
        ]
        receiver = CategoryCodec.fit(train_events["receiver_or_mark"])
        auxiliary_categories = tuple(
            (column, CategoryCodec.fit(train_events[column]))
            for column in dataset.schema.auxiliary_categorical_columns
        )
        event_numeric_columns = (
            "amount_or_numeric_value",
        ) + dataset.schema.auxiliary_numeric_columns
        event_numeric = tuple(
            (
                column,
                NumericCodec.fit(
                    train_events[column],
                    transform=(
                        "signed_log1p_zscore"
                        if column in {"amount_or_numeric_value", "balance"}
                        else "zscore"
                    ),
                ),
            )
            for column in event_numeric_columns
        )
        static_categories = tuple(
            (column, CategoryCodec.fit(train_static[column]))
            for column in roles["categorical"]
        )
        static_numeric = tuple(
            (
                column,
                NumericCodec.fit(train_static[column], transform="zscore"),
            )
            for column in roles["numeric"]
        )
        support = fit_train_only_gap_support(
            train_events["gap"].dropna().to_numpy(float),
            max_positive_states=max_positive_gap_states,
        )
        return cls(
            SAFTensorizerState(
                dataset_id=dataset.schema.dataset_id,
                schema_sha256=dataset.schema.schema_sha256,
                split_assignment_sha256=dataset.split_assignment_sha256,
                receiver_codec=receiver,
                auxiliary_categorical_codecs=auxiliary_categories,
                event_numeric_codecs=event_numeric,
                static_categorical_codecs=static_categories,
                static_numeric_codecs=static_numeric,
                gap_support=support,
            )
        )

    def _validate_dataset(self, dataset: CanonicalEntitySequenceDataset) -> None:
        if dataset.schema.dataset_id != self.state.dataset_id:
            raise SAFDataContractError("tensorizer dataset identity mismatch")
        if dataset.schema.schema_sha256 != self.state.schema_sha256:
            raise SAFDataContractError("tensorizer schema hash mismatch")
        if dataset.split_assignment_sha256 != self.state.split_assignment_sha256:
            raise SAFDataContractError("tensorizer split hash mismatch")

    def transform_split(
        self,
        dataset: CanonicalEntitySequenceDataset,
        split: str,
        *,
        entity_limit: Optional[int] = None,
    ) -> Tuple[TensorizedSequence, ...]:
        self._validate_dataset(dataset)
        entity_ids = dataset.entity_ids_for_split(split)
        if entity_limit is not None:
            entity_ids = entity_ids[:entity_limit]
        static = dataset.static_context.set_index("entity_id")
        events = dataset.events[dataset.events["entity_id"].isin(set(entity_ids))]
        grouped = {entity: group for entity, group in events.groupby("entity_id", sort=False)}
        output = []
        for entity_id in entity_ids:
            group = grouped[entity_id].sort_values("event_index", kind="mergesort")
            static_row = static.loc[entity_id]
            primary_codec = self.state.event_numeric_codecs[0][1]
            aux_numeric = [
                codec.encode(group[column])
                for column, codec in self.state.event_numeric_codecs[1:]
            ]
            output.append(
                TensorizedSequence(
                    entity_id=entity_id,
                    gap=group["gap"].to_numpy(np.float32),
                    receiver=self.state.receiver_codec.encode(
                        group["receiver_or_mark"]
                    ),
                    numeric_value=primary_codec.encode(
                        group["amount_or_numeric_value"]
                    ),
                    auxiliary_categorical=tuple(
                        codec.encode(group[column])
                        for column, codec in self.state.auxiliary_categorical_codecs
                    ),
                    auxiliary_numeric=(
                        np.column_stack(aux_numeric).astype(np.float32)
                        if aux_numeric
                        else np.zeros((len(group), 0), dtype=np.float32)
                    ),
                    static_categorical=tuple(
                        int(codec.encode([static_row[column]])[0])
                        for column, codec in self.state.static_categorical_codecs
                    ),
                    static_numeric=np.asarray(
                        [
                            float(codec.encode([static_row[column]])[0])
                            for column, codec in self.state.static_numeric_codecs
                        ],
                        dtype=np.float32,
                    ),
                )
            )
        return tuple(output)

    def decode_generated(
        self,
        *,
        entity_ids: Sequence[Any],
        gap: torch.Tensor,
        receiver: torch.Tensor,
        numeric_value: torch.Tensor,
        auxiliary_categorical: Sequence[torch.Tensor],
        auxiliary_numeric: torch.Tensor,
        lengths: Sequence[int],
    ) -> pd.DataFrame:
        rows = []
        primary_codec = self.state.event_numeric_codecs[0][1]
        for entity_position, (entity_id, length) in enumerate(zip(entity_ids, lengths)):
            length = int(length)
            gap_values = gap[entity_position, :length].detach().cpu().numpy()
            receiver_values = self.state.receiver_codec.decode(
                receiver[entity_position, :length].detach().cpu().numpy()
            )
            primary_values = primary_codec.decode(
                numeric_value[entity_position, :length].detach().cpu().numpy()
            )
            aux_cat_values = [
                codec.decode(
                    auxiliary_categorical[index][entity_position, :length]
                    .detach()
                    .cpu()
                    .numpy()
                )
                for index, (_, codec) in enumerate(
                    self.state.auxiliary_categorical_codecs
                )
            ]
            aux_numeric_values = [
                codec.decode(
                    auxiliary_numeric[entity_position, :length, index]
                    .detach()
                    .cpu()
                    .numpy()
                )
                for index, (_, codec) in enumerate(
                    self.state.event_numeric_codecs[1:]
                )
            ]
            timestamps = np.zeros(length, dtype=float)
            if length > 1:
                timestamps[1:] = np.cumsum(
                    gap_values[1:].astype(np.float64),
                    dtype=np.float64,
                )
            for event_index in range(length):
                row = {
                    "entity_id": entity_id,
                    "event_id": f"{entity_id}-{event_index}",
                    "event_index": event_index,
                    "timestamp": float(timestamps[event_index]),
                    "gap": np.nan if event_index == 0 else float(gap_values[event_index]),
                    "receiver_or_mark": receiver_values[event_index],
                    "amount_or_numeric_value": float(primary_values[event_index]),
                }
                for index, (column, _) in enumerate(
                    self.state.event_numeric_codecs[1:]
                ):
                    row[column] = float(aux_numeric_values[index][event_index])
                for index, (column, _) in enumerate(
                    self.state.auxiliary_categorical_codecs
                ):
                    row[column] = aux_cat_values[index][event_index]
                rows.append(row)
        columns = [
            "entity_id",
            "event_id",
            "event_index",
            "timestamp",
            "gap",
            "receiver_or_mark",
            "amount_or_numeric_value",
            *[name for name, _ in self.state.event_numeric_codecs[1:]],
            *[name for name, _ in self.state.auxiliary_categorical_codecs],
        ]
        return pd.DataFrame(rows, columns=columns)


def load_canonical_dataset(
    dataset_dir: str | Path,
    *,
    allowed_splits: Sequence[str] = SPLIT_NAMES,
) -> CanonicalEntitySequenceDataset:
    """Load only allowed entity content while retaining the full split hash.

    Reading ``entity_splits`` is structural access. Static/event row bodies for
    sealed roles are excluded by Parquet predicates and never materialized.
    """

    path = Path(dataset_dir)
    allowed = tuple(allowed_splits)
    if not allowed or not set(allowed) <= set(SPLIT_NAMES):
        raise SAFDataContractError("allowed_splits must be a canonical non-empty subset")
    schema_raw = json.loads((path / "schema.json").read_text(encoding="utf-8"))
    schema = CanonicalSchema(
        dataset_id=schema_raw["dataset_id"],
        time_representation=schema_raw["time_representation"],
        timestamp_unit=schema_raw["timestamp_unit"],
        static_context_columns=tuple(schema_raw["static_context_columns"]),
        auxiliary_numeric_columns=tuple(schema_raw["auxiliary_numeric_columns"]),
        auxiliary_categorical_columns=tuple(
            schema_raw["auxiliary_categorical_columns"]
        ),
    )
    full_splits = pd.read_parquet(path / "entity_splits.parquet")
    source_split_hash = hash_entity_split_assignment(full_splits)
    selected_splits = full_splits[full_splits["split"].isin(allowed)].copy()
    selected_ids = selected_splits["entity_id"].tolist()
    parquet_filter = [("entity_id", "in", selected_ids)]
    return CanonicalEntitySequenceDataset(
        schema=schema,
        static_context=pd.read_parquet(
            path / "static_context.parquet",
            filters=parquet_filter,
        ),
        events=pd.read_parquet(path / "events.parquet", filters=parquet_filter),
        entity_splits=selected_splits,
        required_splits=allowed,
        source_split_assignment_sha256=source_split_hash,
    )
