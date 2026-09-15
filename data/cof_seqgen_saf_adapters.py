"""Dataset adapters for CoFSeqGen-SAF canonical entity sequences."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from data.cof_seqgen_saf_contract import (
    CORE_EVENT_COLUMNS,
    CanonicalEntitySequenceDataset,
    CanonicalSchema,
    SAFDataContractError,
    build_entity_split_assignment,
)


DEFAULT_SPLIT_SEED = 20260826
LENGTH_EDGES = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class AdapterResult:
    dataset_id: str
    canonical: CanonicalEntitySequenceDataset
    audit: dict[str, Any]
    stratification: dict[str, Any]
    oracle_events: pd.DataFrame | None = None
    mark_context: pd.DataFrame | None = None


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], dataset: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SAFDataContractError(f"{dataset} is missing required columns: {missing}")


def _clean_string(values: pd.Series) -> pd.Series:
    output = values.astype("string").str.strip()
    return output.mask(output.eq("") | output.eq("nan") | output.eq("<NA>"))


def _stable_value(value: Any) -> str:
    if value is None or bool(pd.isna(value)):
        return "<MISSING>"
    return f"{type(value).__name__}:{value}"


def stable_hash_rank(values: Sequence[Any], *, seed: int, namespace: str) -> list[Any]:
    """Order typed identities without depending on source row order."""

    def key(value: Any) -> tuple[str, str]:
        payload = json.dumps(
            {
                "namespace": namespace,
                "seed": seed,
                "type": type(value).__name__,
                "value": str(value),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), _stable_value(value)

    return sorted(values, key=key)


def _length_bin(length: int, *, coarse: bool = False) -> str:
    index = int(np.searchsorted(np.asarray(LENGTH_EDGES), length, side="right") - 1)
    index = max(0, min(index, len(LENGTH_EDGES) - 1))
    if coarse:
        index //= 2
    return f"L{index}"


def _label(kind: str, values: Sequence[Any]) -> str:
    return json.dumps(
        {"kind": kind, "values": [_stable_value(value) for value in values]},
        sort_keys=True,
        separators=(",", ":"),
    )


def build_hierarchical_strata(
    static_context: pd.DataFrame,
    lengths: pd.Series,
    *,
    static_columns: Sequence[str],
) -> tuple[dict[Any, str], dict[str, Any]]:
    """Create feasible static/length strata, deterministically merging cells <3."""

    entity_ids = static_context["entity_id"].tolist()
    if set(entity_ids) != set(lengths.index.tolist()):
        raise SAFDataContractError("stratification entity sets do not match")
    static_by_entity = static_context.set_index("entity_id")
    candidates: dict[Any, list[str]] = {}
    for entity in entity_ids:
        static_values = [static_by_entity.at[entity, column] for column in static_columns]
        length = int(lengths.at[entity])
        levels = [
            _label("static_x_fine_length", (*static_values, _length_bin(length))),
            _label("static_x_coarse_length", (*static_values, _length_bin(length, coarse=True))),
            _label("static_only", static_values),
            _label("coarse_length_only", (_length_bin(length, coarse=True),)),
            _label("global", ()),
        ]
        if not static_columns:
            levels = [levels[0], levels[1], levels[3], levels[4]]
        candidates[entity] = levels

    level = {entity: 0 for entity in entity_ids}
    for _ in range(8):
        labels = {entity: candidates[entity][level[entity]] for entity in entity_ids}
        counts = pd.Series(list(labels.values())).value_counts()
        rare = [entity for entity, value in labels.items() if counts[value] < 3]
        if not rare:
            break
        movable = [entity for entity in rare if level[entity] + 1 < len(candidates[entity])]
        if movable:
            for entity in movable:
                level[entity] += 1
            continue
        nonrare_counts = counts[counts >= 3]
        if nonrare_counts.empty:
            raise SAFDataContractError("fewer than three entities are available for splitting")
        destination = sorted(
            nonrare_counts.items(), key=lambda item: (-int(item[1]), item[0])
        )[0][0]
        for entity in rare:
            candidates[entity] = [destination]
            level[entity] = 0
    labels = {entity: candidates[entity][level[entity]] for entity in entity_ids}
    counts = pd.Series(list(labels.values())).value_counts().sort_index()
    if (counts < 3).any():
        raise SAFDataContractError("hierarchical stratum merging left a cell below three")
    return labels, {
        "static_columns": list(static_columns),
        "length_edges": list(LENGTH_EDGES) + ["+inf"],
        "stratum_count": int(len(counts)),
        "stratum_sizes": {key: int(value) for key, value in counts.items()},
    }


def _ordered_events(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    event_column: str,
    timestamp_column: str,
    receiver_column: str,
    amount_column: str,
    auxiliary_numeric: Sequence[str],
    auxiliary_categorical: Sequence[str],
) -> pd.DataFrame:
    ordered = frame.sort_values(
        [entity_column, timestamp_column, event_column], kind="mergesort"
    ).reset_index(drop=True)
    if ordered[event_column].isna().any() or ordered[
        [entity_column, event_column]
    ].duplicated().any():
        raise SAFDataContractError("event identity is missing or duplicated within entity")
    ordered["event_index"] = ordered.groupby(entity_column, sort=False).cumcount()
    ordered["gap"] = ordered.groupby(entity_column, sort=False)[timestamp_column].diff()
    canonical = pd.DataFrame(
        {
            "entity_id": ordered[entity_column],
            "event_id": ordered[event_column],
            "event_index": ordered["event_index"],
            "timestamp": ordered[timestamp_column],
            "gap": ordered["gap"],
            "receiver_or_mark": ordered[receiver_column],
            "amount_or_numeric_value": ordered[amount_column],
        }
    )
    for column in (*auxiliary_numeric, *auxiliary_categorical):
        canonical[column] = ordered[column]
    return canonical[
        list(CORE_EVENT_COLUMNS) + list(auxiliary_numeric) + list(auxiliary_categorical)
    ]


def _finalize(
    *,
    dataset_id: str,
    schema: CanonicalSchema,
    static_context: pd.DataFrame,
    events: pd.DataFrame,
    stratify_static_columns: Sequence[str],
    split_seed: int,
    audit: dict[str, Any],
) -> AdapterResult:
    lengths = events.groupby("entity_id", sort=True).size()
    strata, stratification = build_hierarchical_strata(
        static_context,
        lengths,
        static_columns=stratify_static_columns,
    )
    splits = build_entity_split_assignment(
        static_context["entity_id"].tolist(),
        seed=split_seed,
        strata=strata,
    )
    canonical = CanonicalEntitySequenceDataset(
        schema=schema,
        static_context=static_context,
        events=events,
        entity_splits=splits,
    )
    first_gap_count = int(events.loc[events["event_index"] == 0, "gap"].isna().sum())
    audit.update(
        {
            "canonical_entities": int(len(static_context)),
            "canonical_events": int(len(events)),
            "entity_leakage": 0,
            "negative_gap_count": int((events["gap"].dropna() < 0).sum()),
            "first_gap_missing_count": first_gap_count,
            "split_seed": split_seed,
            "split_assignment_sha256": canonical.split_assignment_sha256,
        }
    )
    return AdapterResult(dataset_id, canonical, audit, stratification)


def adapt_amlsim(raw_dir: Path, *, split_seed: int = DEFAULT_SPLIT_SEED) -> AdapterResult:
    accounts = pd.read_csv(raw_dir / "accounts.csv", low_memory=False)
    transactions = pd.read_csv(raw_dir / "transactions.csv", low_memory=False)
    _require_columns(
        transactions,
        [
            "TX_ID",
            "SENDER_ACCOUNT_ID",
            "RECEIVER_ACCOUNT_ID",
            "TX_TYPE",
            "TX_AMOUNT",
            "TIMESTAMP",
            "IS_FRAUD",
        ],
        "amlsim transactions",
    )
    _require_columns(
        accounts,
        ["ACCOUNT_ID", "INIT_BALANCE", "IS_FRAUD", "TX_BEHAVIOR_ID"],
        "amlsim accounts",
    )
    timestamp = pd.to_numeric(transactions["TIMESTAMP"], errors="coerce")
    amount = pd.to_numeric(transactions["TX_AMOUNT"], errors="coerce")
    parse_errors = int(timestamp.isna().sum())
    if parse_errors or amount.isna().any():
        raise SAFDataContractError("AMLSim timestamp or amount parsing failed")
    work = transactions.assign(
        _timestamp=timestamp.astype(float),
        _amount=amount.astype(float),
        _receiver=transactions["RECEIVER_ACCOUNT_ID"].astype("string"),
        transaction_type=_clean_string(transactions["TX_TYPE"]),
        event_is_fraud=transactions["IS_FRAUD"].astype(int).astype("string"),
    )
    entity_ids = pd.Index(pd.unique(work["SENDER_ACCOUNT_ID"]))
    static = accounts[accounts["ACCOUNT_ID"].isin(entity_ids)].copy()
    if static["ACCOUNT_ID"].duplicated().any() or len(static) != len(entity_ids):
        raise SAFDataContractError("AMLSim account context is not one-to-one with senders")
    static = static.rename(
        columns={
            "ACCOUNT_ID": "entity_id",
            "INIT_BALANCE": "initial_balance",
            "IS_FRAUD": "account_is_fraud",
            "TX_BEHAVIOR_ID": "tx_behavior_id",
        }
    )[["entity_id", "initial_balance", "account_is_fraud", "tx_behavior_id"]]
    static["account_is_fraud"] = static["account_is_fraud"].astype(int)
    events = _ordered_events(
        work,
        entity_column="SENDER_ACCOUNT_ID",
        event_column="TX_ID",
        timestamp_column="_timestamp",
        receiver_column="_receiver",
        amount_column="_amount",
        auxiliary_numeric=(),
        auxiliary_categorical=("transaction_type", "event_is_fraud"),
    )
    schema = CanonicalSchema(
        dataset_id="amlsim",
        time_representation="absolute",
        timestamp_unit="simulation_step",
        static_context_columns=("initial_balance", "account_is_fraud", "tx_behavior_id"),
        auxiliary_categorical_columns=("transaction_type", "event_is_fraud"),
    )
    return _finalize(
        dataset_id="amlsim",
        schema=schema,
        static_context=static.sort_values("entity_id").reset_index(drop=True),
        events=events,
        stratify_static_columns=("account_is_fraud",),
        split_seed=split_seed,
        audit={
            "raw_event_rows": int(len(transactions)),
            "raw_static_rows": int(len(accounts)),
            "timestamp_parse_errors": parse_errors,
            "duplicate_event_ids": int(transactions["TX_ID"].duplicated().sum()),
            "negative_amount_count": int((amount < 0).sum()),
        },
    )


def adapt_controlled_coupling_dgp(
    raw_dir: Path,
    *,
    generator_config: Mapping[str, Any],
    scenario: str,
    kappa: float,
    n_entities: int,
    generation_seed: int,
    split_seed: int = DEFAULT_SPLIT_SEED,
    cell_id: str,
) -> AdapterResult:
    """Materialize one observed DGP cell plus separately stored oracle latents."""

    del raw_dir  # The registered generator source, rather than a row file, is authoritative.
    from benchmarks.temporal_coupling_v2 import BenchmarkConfig, _split

    config = BenchmarkConfig.from_mapping(generator_config, scenario, kappa)
    config = BenchmarkConfig(
        **{
            **config.__dict__,
            "n_train": int(n_entities),
            "n_test": 0,
            "bin_edges": None,
        }
    )
    values, latent = _split(config, int(n_entities), int(generation_seed), 0)
    valid_entity, valid_position = np.nonzero(values["valid_mask"])
    entity_ids = np.asarray(
        [f"controlled-{index:06d}" for index in range(int(n_entities))],
        dtype=object,
    )
    row_entity_ids = entity_ids[valid_entity]
    event_ids = valid_entity.astype(np.int64) * int(config.max_length) + valid_position
    raw_gap = latent["raw_gap"][valid_entity, valid_position].astype(float)
    timestamps = np.cumsum(latent["raw_gap"].astype(float), axis=1)[
        valid_entity, valid_position
    ]
    event_index = valid_position.astype(np.int64)
    gaps = raw_gap.copy()
    gaps[event_index == 0] = np.nan
    events = pd.DataFrame(
        {
            "entity_id": row_entity_ids,
            "event_id": event_ids,
            "event_index": event_index,
            "timestamp": timestamps,
            "gap": gaps,
            "receiver_or_mark": pd.Series(
                values["x_cat"][valid_entity, valid_position, 0]
            ).map(lambda value: f"receiver-{int(value)}"),
            "amount_or_numeric_value": values["x_num"][
                valid_entity, valid_position, 0
            ].astype(float),
        }
    )[list(CORE_EVENT_COLUMNS)]
    static = pd.DataFrame(
        {
            "entity_id": entity_ids,
            "entity_label": values["y_entity"].astype(np.int64),
        }
    )
    oracle = pd.DataFrame(
        {
            "entity_id": row_entity_ids,
            "event_id": event_ids,
            "event_index": event_index,
            "gap_state": latent["gap_state"][valid_entity, valid_position].astype(
                np.int8
            ),
            "receiver_state": latent["receiver_state"][
                valid_entity, valid_position
            ].astype(np.int8),
            "receiver_repeat": latent["receiver_repeat"][
                valid_entity, valid_position
            ].astype(np.int8),
        }
    )
    schema = CanonicalSchema(
        dataset_id=f"controlled_coupling_{cell_id}",
        time_representation="absolute",
        timestamp_unit="continuous_simulation_time",
        static_context_columns=("entity_label",),
    )
    result = _finalize(
        dataset_id=schema.dataset_id,
        schema=schema,
        static_context=static,
        events=events,
        stratify_static_columns=("entity_label",),
        split_seed=split_seed,
        audit={
            "raw_event_rows": int(len(events)),
            "timestamp_parse_errors": 0,
            "duplicate_event_ids": int(events["event_id"].duplicated().sum()),
            "negative_amount_count": int(
                (events["amount_or_numeric_value"] < 0).sum()
            ),
            "scenario": scenario,
            "kappa": float(kappa),
            "generation_seed": int(generation_seed),
            "n_entities_requested": int(n_entities),
            "generator_schema_version": config.schema_version,
            "first_sampled_gap_role": "absolute_time_origin_offset_not_modeled_as_gap",
            "oracle_latents_separated_from_model_input": True,
        },
    )
    return AdapterResult(
        result.dataset_id,
        result.canonical,
        result.audit,
        result.stratification,
        oracle,
    )


def adapt_sparkov(raw_dir: Path, *, split_seed: int = DEFAULT_SPLIT_SEED) -> AdapterResult:
    raw = pd.read_csv(raw_dir / "fraudTrain.csv", low_memory=False)
    _require_columns(
        raw,
        [
            "trans_num",
            "cc_num",
            "merchant",
            "category",
            "amt",
            "gender",
            "state",
            "dob",
            "city_pop",
            "trans_date_trans_time",
            "unix_time",
            "merch_lat",
            "merch_long",
            "is_fraud",
        ],
        "sparkov",
    )
    parsed = pd.to_datetime(
        raw["trans_date_trans_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    parse_errors = int(parsed.isna().sum())
    if parse_errors:
        raise SAFDataContractError("Sparkov timestamp parsing failed")
    unix_from_text = parsed.astype("int64") / 1_000_000_000
    provided_unix = pd.to_numeric(raw["unix_time"], errors="coerce")
    unix_disagreement = int((np.abs(unix_from_text - provided_unix) > 1).sum())
    work = raw.assign(
        _timestamp=unix_from_text.astype(float),
        _amount=pd.to_numeric(raw["amt"], errors="coerce"),
        _receiver=_clean_string(raw["merchant"]),
        merchant_latitude=pd.to_numeric(raw["merch_lat"], errors="coerce"),
        merchant_longitude=pd.to_numeric(raw["merch_long"], errors="coerce"),
        event_is_fraud=raw["is_fraud"].astype(int).astype("string"),
        category=_clean_string(raw["category"]),
    )
    if work[["_amount", "merchant_latitude", "merchant_longitude"]].isna().any().any():
        raise SAFDataContractError("Sparkov numeric parsing failed")
    static_source = raw.sort_values(["cc_num", "trans_num"], kind="mergesort")
    for column in ("gender", "state", "dob", "city_pop"):
        if (static_source.groupby("cc_num")[column].nunique(dropna=False) > 1).any():
            raise SAFDataContractError(f"Sparkov static field varies within card: {column}")
    static = static_source.groupby("cc_num", sort=True).first().reset_index()
    static["entity_any_fraud"] = (
        raw.groupby("cc_num", sort=True)["is_fraud"].max().reindex(static["cc_num"]).to_numpy()
    )
    static["birth_year"] = pd.to_datetime(static["dob"], errors="coerce").dt.year
    static = static.rename(
        columns={
            "cc_num": "entity_id",
            "gender": "cardholder_gender",
            "state": "cardholder_state",
            "city_pop": "city_population",
        }
    )[
        [
            "entity_id",
            "cardholder_gender",
            "cardholder_state",
            "birth_year",
            "city_population",
            "entity_any_fraud",
        ]
    ]
    events = _ordered_events(
        work,
        entity_column="cc_num",
        event_column="trans_num",
        timestamp_column="_timestamp",
        receiver_column="_receiver",
        amount_column="_amount",
        auxiliary_numeric=("merchant_latitude", "merchant_longitude"),
        auxiliary_categorical=("category", "event_is_fraud"),
    )
    schema = CanonicalSchema(
        dataset_id="sparkov",
        time_representation="absolute",
        timestamp_unit="unix_second_from_trans_date_trans_time",
        static_context_columns=(
            "cardholder_gender",
            "cardholder_state",
            "birth_year",
            "city_population",
            "entity_any_fraud",
        ),
        auxiliary_numeric_columns=("merchant_latitude", "merchant_longitude"),
        auxiliary_categorical_columns=("category", "event_is_fraud"),
    )
    return _finalize(
        dataset_id="sparkov",
        schema=schema,
        static_context=static,
        events=events,
        stratify_static_columns=("entity_any_fraud", "cardholder_gender"),
        split_seed=split_seed,
        audit={
            "raw_event_rows": int(len(raw)),
            "timestamp_parse_errors": parse_errors,
            "provided_unix_disagreement_rows": unix_disagreement,
            "timestamp_authority": "trans_date_trans_time",
            "duplicate_event_ids": int(raw["trans_num"].duplicated().sum()),
            "negative_amount_count": int((work["_amount"] < 0).sum()),
            "fraud_test_pooled": False,
        },
    )


def _parse_berka_date(values: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    text = numeric.astype("Int64").astype("string").str.zfill(6)
    parsed = pd.to_datetime("19" + text, format="%Y%m%d", errors="coerce")
    if parsed.isna().any():
        raise SAFDataContractError(f"Berka {name} contains invalid dates")
    return parsed


def adapt_berka(raw_dir: Path, *, split_seed: int = DEFAULT_SPLIT_SEED) -> AdapterResult:
    account = pd.read_csv(raw_dir / "account.asc", sep=";", low_memory=False)
    trans = pd.read_csv(raw_dir / "trans.asc", sep=";", low_memory=False)
    district = pd.read_csv(raw_dir / "district.asc", sep=";", low_memory=False)
    _require_columns(
        trans,
        [
            "trans_id",
            "account_id",
            "date",
            "type",
            "operation",
            "amount",
            "balance",
            "k_symbol",
            "bank",
            "account",
        ],
        "berka transactions",
    )
    parsed = _parse_berka_date(trans["date"], name="transaction date")
    timestamp = parsed.astype("int64") / (86_400 * 1_000_000_000)
    operation = _clean_string(trans["operation"])
    tx_type = _clean_string(trans["type"])
    receiver_mark = operation.fillna(tx_type).fillna("<MISSING>")
    work = trans.assign(
        _timestamp=timestamp.astype(float),
        _amount=pd.to_numeric(trans["amount"], errors="coerce"),
        _receiver=receiver_mark,
        balance=pd.to_numeric(trans["balance"], errors="coerce"),
        transaction_type=tx_type,
        k_symbol=_clean_string(trans["k_symbol"]),
    )
    if work[["_amount", "balance"]].isna().any().any():
        raise SAFDataContractError("Berka amount or balance parsing failed")

    geography = district[["A1", "A2", "A3"]].rename(
        columns={"A1": "district_id", "A2": "city", "A3": "region"}
    )
    opened = _parse_berka_date(account["date"], name="account date")
    static = account.assign(
        account_open_day=(opened.astype("int64") / (86_400 * 1_000_000_000)).astype(float)
    ).merge(
        geography,
        on="district_id",
        how="left",
        validate="many_to_one",
    )
    static = static.rename(
        columns={
            "account_id": "entity_id",
            "district_id": "account_district_id",
            "frequency": "account_frequency",
        }
    )[
        [
            "entity_id",
            "account_frequency",
            "account_district_id",
            "account_open_day",
            "city",
            "region",
        ]
    ]
    events = _ordered_events(
        work,
        entity_column="account_id",
        event_column="trans_id",
        timestamp_column="_timestamp",
        receiver_column="_receiver",
        amount_column="_amount",
        auxiliary_numeric=("balance",),
        auxiliary_categorical=("transaction_type", "k_symbol"),
    )
    schema = CanonicalSchema(
        dataset_id="berka",
        time_representation="absolute",
        timestamp_unit="unix_day",
        static_context_columns=(
            "account_frequency",
            "account_district_id",
            "account_open_day",
            "city",
            "region",
        ),
        auxiliary_numeric_columns=("balance",),
        auxiliary_categorical_columns=("transaction_type", "k_symbol"),
    )
    return _finalize(
        dataset_id="berka",
        schema=schema,
        static_context=static.sort_values("entity_id").reset_index(drop=True),
        events=events,
        stratify_static_columns=("region", "account_frequency"),
        split_seed=split_seed,
        audit={
            "raw_event_rows": int(len(trans)),
            "raw_static_rows": int(len(account)),
            "timestamp_parse_errors": 0,
            "duplicate_event_ids": int(trans["trans_id"].duplicated().sum()),
            "negative_amount_count": int((work["_amount"] < 0).sum()),
            "tabdit_parent_fields": 5,
            "tabdit_child_fields_including_timestamp": 6,
            "excluded_from_comparison_view": [
                "counterparty_bank",
                "counterparty_account",
                "owner_sex",
                "owner_birth_year",
            ],
        },
    )


def adapt_citi_bike(raw_dir: Path, *, split_seed: int = DEFAULT_SPLIT_SEED) -> AdapterResult:
    candidates = sorted(raw_dir.glob("*citibike-tripdata*.csv"))
    if len(candidates) != 1:
        raise SAFDataContractError(f"expected one Citi Bike CSV, observed {candidates}")
    raw = pd.read_csv(candidates[0], low_memory=False)
    _require_columns(
        raw,
        [
            "tripduration",
            "starttime",
            "end station id",
            "start station id",
            "start station latitude",
            "start station longitude",
            "end station latitude",
            "end station longitude",
            "bikeid",
            "usertype",
            "birth year",
            "gender",
        ],
        "citi bike",
    )
    parsed = pd.to_datetime(raw["starttime"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    parse_errors = int(parsed.isna().sum())
    if parse_errors:
        raise SAFDataContractError("Citi Bike timestamp parsing failed")
    work = raw.assign(
        _event_id=np.arange(len(raw), dtype=np.int64),
        _timestamp=(parsed.astype("int64") / 1_000_000_000).astype(float),
        _amount=pd.to_numeric(raw["tripduration"], errors="coerce"),
        _receiver=pd.to_numeric(raw["end station id"], errors="coerce").astype("Int64").astype("string"),
        start_latitude=pd.to_numeric(raw["start station latitude"], errors="coerce"),
        start_longitude=pd.to_numeric(raw["start station longitude"], errors="coerce"),
        end_latitude=pd.to_numeric(raw["end station latitude"], errors="coerce"),
        end_longitude=pd.to_numeric(raw["end station longitude"], errors="coerce"),
        birth_year=pd.to_numeric(raw["birth year"], errors="coerce"),
        start_station_id=pd.to_numeric(raw["start station id"], errors="coerce").astype("Int64").astype("string"),
        user_type=_clean_string(raw["usertype"]),
        rider_gender=pd.to_numeric(raw["gender"], errors="coerce").astype("Int64").astype("string"),
    )
    required_numeric = [
        "_amount",
        "start_latitude",
        "start_longitude",
        "end_latitude",
        "end_longitude",
    ]
    if work[required_numeric].isna().any().any() or work["_receiver"].isna().any():
        raise SAFDataContractError("Citi Bike required numeric/station parsing failed")
    entity_ids = sorted(pd.unique(work["bikeid"]).tolist())
    static = pd.DataFrame({"entity_id": entity_ids})
    events = _ordered_events(
        work,
        entity_column="bikeid",
        event_column="_event_id",
        timestamp_column="_timestamp",
        receiver_column="_receiver",
        amount_column="_amount",
        auxiliary_numeric=(
            "start_latitude",
            "start_longitude",
            "end_latitude",
            "end_longitude",
            "birth_year",
        ),
        auxiliary_categorical=("start_station_id", "user_type", "rider_gender"),
    )
    schema = CanonicalSchema(
        dataset_id="citi_bike",
        time_representation="absolute",
        timestamp_unit="unix_second",
        auxiliary_numeric_columns=(
            "start_latitude",
            "start_longitude",
            "end_latitude",
            "end_longitude",
            "birth_year",
        ),
        auxiliary_categorical_columns=("start_station_id", "user_type", "rider_gender"),
    )
    return _finalize(
        dataset_id="citi_bike",
        schema=schema,
        static_context=static,
        events=events,
        stratify_static_columns=(),
        split_seed=split_seed,
        audit={
            "raw_event_rows": int(len(raw)),
            "timestamp_parse_errors": parse_errors,
            "duplicate_source_rows": int(raw.duplicated().sum()),
            "negative_duration_count": int((work["_amount"] < 0).sum()),
            "seq2synth_reported_rows": 559_644,
            "seq2synth_reported_trajectories": 5_793,
            "exact_seq2synth_row_identity_confirmed": False,
        },
    )


def adapt_hm(
    raw_dir: Path,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    sample_seed: int = 20260826,
    customer_count: int = 10_000,
    chunksize: int = 2_000_000,
) -> AdapterResult:
    transactions_path = raw_dir / "transactions_train.csv"
    customers_path = raw_dir / "customers.csv"
    articles_path = raw_dir / "articles.csv"
    for path in (transactions_path, customers_path, articles_path):
        if not path.is_file():
            raise SAFDataContractError(f"H&M source file is missing: {path}")
    active: set[str] = set()
    for chunk in pd.read_csv(
        transactions_path, usecols=["customer_id"], dtype={"customer_id": "string"}, chunksize=chunksize
    ):
        active.update(chunk["customer_id"].dropna().tolist())
    if len(active) < customer_count:
        raise SAFDataContractError("H&M has fewer active customers than requested")
    selected = stable_hash_rank(
        list(active), seed=sample_seed, namespace="cofseqgen-saf-hm-customer"
    )[:customer_count]
    selected_set = set(selected)
    frames = []
    row_offset = 0
    raw_rows = 0
    for chunk in pd.read_csv(transactions_path, chunksize=chunksize, low_memory=False):
        raw_rows += len(chunk)
        mask = chunk["customer_id"].isin(selected_set)
        kept = chunk.loc[mask, ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]].copy()
        kept["_event_id"] = np.arange(row_offset, row_offset + len(chunk), dtype=np.int64)[mask.to_numpy()]
        frames.append(kept)
        row_offset += len(chunk)
    raw = pd.concat(frames, ignore_index=True)
    parsed = pd.to_datetime(raw["t_dat"], format="%Y-%m-%d", errors="coerce")
    parse_errors = int(parsed.isna().sum())
    if parse_errors:
        raise SAFDataContractError("H&M timestamp parsing failed")
    work = raw.assign(
        _timestamp=(parsed.astype("int64") / (86_400 * 1_000_000_000)).astype(float),
        _amount=pd.to_numeric(raw["price"], errors="coerce"),
        _receiver=raw["article_id"].astype("string"),
        sales_channel_id=raw["sales_channel_id"].astype("Int64").astype("string"),
    )
    customers = pd.read_csv(customers_path, low_memory=False)
    _require_columns(
        customers,
        ["customer_id", "club_member_status", "fashion_news_frequency", "age", "postal_code"],
        "H&M customers",
    )
    static = customers[customers["customer_id"].isin(selected_set)].copy()
    if static["customer_id"].duplicated().any() or len(static) != customer_count:
        raise SAFDataContractError("H&M selected customer context is incomplete or duplicated")
    static = static.rename(columns={"customer_id": "entity_id"})[
        [
            "entity_id",
            "club_member_status",
            "fashion_news_frequency",
            "age",
            "postal_code",
        ]
    ]
    articles = pd.read_csv(articles_path, low_memory=False)
    _require_columns(articles, ["article_id"], "H&M articles")
    events = _ordered_events(
        work,
        entity_column="customer_id",
        event_column="_event_id",
        timestamp_column="_timestamp",
        receiver_column="_receiver",
        amount_column="_amount",
        auxiliary_numeric=(),
        auxiliary_categorical=("sales_channel_id",),
    )
    schema = CanonicalSchema(
        dataset_id="hm",
        time_representation="absolute",
        timestamp_unit="unix_day",
        static_context_columns=(
            "club_member_status",
            "fashion_news_frequency",
            "age",
            "postal_code",
        ),
        auxiliary_categorical_columns=("sales_channel_id",),
    )
    result = _finalize(
        dataset_id="hm",
        schema=schema,
        static_context=static.sort_values("entity_id").reset_index(drop=True),
        events=events,
        stratify_static_columns=("fashion_news_frequency",),
        split_seed=split_seed,
        audit={
            "raw_event_rows_scanned": raw_rows,
            "selected_customers": customer_count,
            "sample_seed": sample_seed,
            "selected_customer_ids_sha256": hashlib.sha256(
                "\n".join(selected).encode("utf-8")
            ).hexdigest(),
            "timestamp_parse_errors": parse_errors,
            "negative_price_count": int((work["_amount"] < 0).sum()),
            "images_accessed": False,
        },
    )
    observed_articles = set(work["_receiver"].dropna().tolist())
    mark_context = articles.assign(
        receiver_or_mark=articles["article_id"].astype("string")
    )
    mark_context = mark_context[
        mark_context["receiver_or_mark"].isin(observed_articles)
    ].drop(columns=["article_id"])
    if mark_context["receiver_or_mark"].duplicated().any() or set(
        mark_context["receiver_or_mark"].tolist()
    ) != observed_articles:
        raise SAFDataContractError("H&M article context is incomplete or duplicated")
    return AdapterResult(
        result.dataset_id,
        result.canonical,
        result.audit,
        result.stratification,
        mark_context=mark_context.sort_values("receiver_or_mark").reset_index(drop=True),
    )


ADAPTERS = {
    "controlled_coupling_dgp": adapt_controlled_coupling_dgp,
    "amlsim": adapt_amlsim,
    "sparkov": adapt_sparkov,
    "berka": adapt_berka,
    "citi_bike": adapt_citi_bike,
    "hm": adapt_hm,
}
