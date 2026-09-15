from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.cof_seqgen_saf_contract import (
    CanonicalEntitySequenceDataset,
    CanonicalSchema,
    RESERVED_CODES,
    SAFDataContractError,
    build_entity_split_assignment,
    make_train_only_fit_provenance,
    validate_train_only_fit_provenance,
)
from eval.cof_seqgen_saf_source_contract import load_saf_source_definition


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _valid_dataset() -> CanonicalEntitySequenceDataset:
    entity_ids = ["e0", "e1", "e2", "e3", "e4", "e5"]
    schema = CanonicalSchema(
        dataset_id="toy",
        time_representation="absolute",
        timestamp_unit="seconds",
        static_context_columns=("segment",),
        auxiliary_categorical_columns=("channel",),
    )
    static = pd.DataFrame(
        {
            "entity_id": entity_ids,
            "segment": ["a", "a", "b", "b", "c", "c"],
        }
    )
    split = pd.DataFrame(
        {
            "entity_id": entity_ids,
            "split": ["train", "train", "validation", "validation", "test", "test"],
        }
    )
    rows = []
    for entity_position, entity_id in enumerate(entity_ids):
        first_time = float(100 * entity_position)
        rows.extend(
            (
                {
                    "entity_id": entity_id,
                    "event_id": f"{entity_id}-0",
                    "event_index": 0,
                    "timestamp": first_time,
                    "gap": np.nan,
                    "receiver_or_mark": "receiver-a",
                    "amount_or_numeric_value": 1.0,
                    "channel": "web",
                },
                {
                    "entity_id": entity_id,
                    "event_id": f"{entity_id}-1",
                    "event_index": 1,
                    "timestamp": first_time + 5.0,
                    "gap": 5.0,
                    "receiver_or_mark": None if entity_id == "e5" else "receiver-b",
                    "amount_or_numeric_value": np.nan if entity_id == "e5" else 2.0,
                    "channel": "store",
                },
            )
        )
    events = pd.DataFrame(rows)[
        [
            "entity_id",
            "event_id",
            "event_index",
            "timestamp",
            "gap",
            "receiver_or_mark",
            "amount_or_numeric_value",
            "channel",
        ]
    ]
    return CanonicalEntitySequenceDataset(
        schema=schema,
        static_context=static,
        events=events,
        entity_splits=split,
    )


def test_source_protocol_is_new_fail_closed_family() -> None:
    definition = load_saf_source_definition(REPOSITORY_ROOT)
    assert definition.raw["family"] == "cof_seqgen_saf"
    assert definition.raw["safety"]["data_download_authorized"] is True
    assert definition.raw["safety"]["materialization_authorized"] is True
    assert definition.raw["safety"]["model_implementation_authorized"] is True
    assert definition.raw["safety"]["fit_authorized"] is False
    assert definition.raw["safety"]["baseline_install_authorized"] is False
    assert (
        definition.raw["endpoint_policy"]["metric_audit_artifact_sha256"]
        == "4c0cec50af9dceccad8a709cbf7393aa1523e8f1ff55e8f43db4d65638395ce4"
    )
    assert definition.raw["family_separation"]["h1_artifacts_mutable"] is False
    assert definition.raw["family_separation"]["d1_status"] == "SPECIFIED_NOT_IMPLEMENTED"
    assert len(definition.config_sha256) == 64


def test_canonical_dataset_preserves_missing_first_gap_and_distinct_codes() -> None:
    dataset = _valid_dataset()
    first_rows = dataset.events[dataset.events["event_index"] == 0]
    assert first_rows["gap"].isna().all()
    assert (
        RESERVED_CODES.pad,
        RESERVED_CODES.unk,
        RESERVED_CODES.missing,
        RESERVED_CODES.first_learned_code,
    ) == (0, 1, 2, 3)
    assert set(dataset.entity_ids_for_split("test")) == {"e4", "e5"}


def test_first_event_gap_zero_is_rejected() -> None:
    dataset = _valid_dataset()
    events = dataset.events.copy()
    events.loc[events["event_index"] == 0, "gap"] = 0.0
    with pytest.raises(SAFDataContractError, match="first event gap must be missing"):
        CanonicalEntitySequenceDataset(
            schema=dataset.schema,
            static_context=dataset.static_context,
            events=events,
            entity_splits=dataset.entity_splits,
        )


def test_timestamp_gap_mismatch_is_rejected() -> None:
    dataset = _valid_dataset()
    events = dataset.events.copy()
    events.loc[events["event_id"] == "e0-1", "gap"] = 6.0
    with pytest.raises(SAFDataContractError, match="timestamp difference"):
        CanonicalEntitySequenceDataset(
            schema=dataset.schema,
            static_context=dataset.static_context,
            events=events,
            entity_splits=dataset.entity_splits,
        )


def test_duplicate_or_missing_split_assignment_is_rejected() -> None:
    dataset = _valid_dataset()
    duplicate = pd.concat(
        [dataset.entity_splits, dataset.entity_splits.iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(SAFDataContractError, match="only one split"):
        CanonicalEntitySequenceDataset(
            schema=dataset.schema,
            static_context=dataset.static_context,
            events=dataset.events,
            entity_splits=duplicate,
        )


def test_entity_split_is_deterministic_disjoint_and_stratified() -> None:
    entity_ids = [f"e{index}" for index in range(20)]
    strata = {entity_id: index % 2 for index, entity_id in enumerate(entity_ids)}
    first = build_entity_split_assignment(entity_ids, seed=412, strata=strata)
    second = build_entity_split_assignment(entity_ids, seed=412, strata=strata)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == len(entity_ids)
    assert first["entity_id"].nunique() == len(entity_ids)
    assert set(first["split"]) == {"train", "validation", "test"}
    assert first["split"].value_counts().to_dict() == {
        "train": 14,
        "validation": 3,
        "test": 3,
    }
    joined = first.assign(stratum=first["entity_id"].map(strata))
    assert set(joined.groupby("stratum")["split"].unique().explode()) == {
        "train",
        "validation",
        "test",
    }


def test_train_only_fit_provenance_rejects_validation_participation() -> None:
    dataset = _valid_dataset()
    provenance = make_train_only_fit_provenance(dataset)
    validate_train_only_fit_provenance(dataset, provenance)
    tampered = replace(
        provenance,
        fit_split="train_plus_validation",
        fit_entity_count=provenance.fit_entity_count + 2,
    )
    with pytest.raises(SAFDataContractError, match="exact canonical training split"):
        validate_train_only_fit_provenance(dataset, tampered)


def test_split_or_schema_change_invalidates_fit_provenance() -> None:
    dataset = _valid_dataset()
    provenance = make_train_only_fit_provenance(dataset)
    changed_split = dataset.entity_splits.copy()
    changed_split.loc[changed_split["entity_id"] == "e1", "split"] = "validation"
    changed_split.loc[changed_split["entity_id"] == "e2", "split"] = "train"
    changed_dataset = CanonicalEntitySequenceDataset(
        schema=dataset.schema,
        static_context=dataset.static_context,
        events=dataset.events,
        entity_splits=changed_split,
    )
    with pytest.raises(SAFDataContractError, match="exact canonical training split"):
        validate_train_only_fit_provenance(changed_dataset, provenance)
