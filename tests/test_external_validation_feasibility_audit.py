import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_external_validation_feasibility import (
    AuditError,
    DatasetColumns,
    audit_frame,
    assert_calibration_sources,
    deterministic_entity_split,
    receiver_oov_by_split,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_amlsim_fixture_uses_actual_columns_and_stable_event_windows():
    frame = pd.DataFrame(
        {
            "TX_ID": [1, 2, 3, 4, 5, 6],
            "SENDER_ACCOUNT_ID": [10, 10, 10, 20, 20, 20],
            "RECEIVER_ACCOUNT_ID": [20, 30, 20, 10, 30, 40],
            "TX_AMOUNT": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "TIMESTAMP": [0, 0, 1, 0, 2, 3],
            "IS_FRAUD": [False, True, False, False, False, False],
        }
    )
    columns = DatasetColumns(
        transaction_id="TX_ID",
        entity="SENDER_ACCOUNT_ID",
        receiver="RECEIVER_ACCOUNT_ID",
        amount="TX_AMOUNT",
        timestamp="TIMESTAMP",
        fraud="IS_FRAUD",
    )

    audit = audit_frame(frame, columns, event_window_size=2, time_window_width=2)

    assert audit["rows"] == 6
    assert audit["entities"] == 2
    assert audit["receiver_cardinality"] == 4
    assert audit["transaction_fraud"]["count"] == 1
    assert audit["entity_fraud"]["positive_count"] == 1
    assert audit["duplicate_timestamps_within_entity"]["row_count"] == 2
    assert audit["ordering"]["stable_tie_breaker"] == "TX_ID"
    assert audit["fixed_event_windows"]["window_size"] == 2
    assert audit["fixed_event_windows"]["window_count"] == 4
    assert audit["fixed_event_windows"]["positive_count"] == 1


def test_sparkov_fixture_treats_merchant_as_receiver_not_category():
    frame = pd.DataFrame(
        {
            "trans_num": ["a", "b", "c", "d"],
            "cc_num": [101, 101, 202, 202],
            "merchant": ["m1", "m2", "m1", "m3"],
            "category": ["grocery", "grocery", "grocery", "fuel"],
            "amt": [10.0, 11.0, 12.0, 13.0],
            "unix_time": [1, 2, 1, 3],
            "is_fraud": [0, 1, 0, 0],
        }
    )
    columns = DatasetColumns(
        transaction_id="trans_num",
        entity="cc_num",
        receiver="merchant",
        amount="amt",
        timestamp="unix_time",
        fraud="is_fraud",
        receiver_secondary="category",
    )

    audit = audit_frame(frame, columns, event_window_size=2, time_window_width=2)

    assert audit["receiver_column"] == "merchant"
    assert audit["receiver_cardinality"] == 3
    assert audit["secondary_receiver"]["column"] == "category"
    assert audit["secondary_receiver"]["cardinality"] == 2


def test_entity_split_and_receiver_vocabulary_are_train_only():
    entities = pd.Series(range(1000), name="entity")
    split = deterministic_entity_split(entities)

    assert set(split.unique()) == {"train", "validation", "internal_test"}
    for entity in entities:
        assert split[entities == entity].nunique() == 1

    frame = pd.DataFrame(
        {
            "entity": [1, 1, 2, 3],
            "receiver": ["known", "train_only", "known", "unseen"],
            "split": ["train", "train", "validation", "internal_test"],
        }
    )
    oov = receiver_oov_by_split(frame, "receiver", "split")
    assert oov["vocabulary_source"] == "train"
    assert oov["validation"]["oov_rows"] == 0
    assert oov["internal_test"]["oov_rows"] == 1


def test_original_test_is_forbidden_as_calibration_input():
    assert_calibration_sources([Path("fraudTrain.csv")])
    with pytest.raises(AuditError, match="test source is forbidden for calibration"):
        assert_calibration_sources([Path("fraudTest.csv")])


def test_committed_audit_records_actual_schema_and_frozen_non_v3_cof():
    report_path = (
        REPOSITORY_ROOT
        / "docs/benchmark_v2/external_validation_feasibility_audit.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["datasets"]["amlsim"]["files"]["transactions"]["rows"] == 1_323_234
    assert report["datasets"]["sparkov"]["files"]["train"]["rows"] == 1_296_675
    assert report["datasets"]["sparkov"]["files"]["test"]["rows"] == 555_719
    assert report["datasets"]["amlsim"]["mapping"]["receiver"] == "RECEIVER_ACCOUNT_ID"
    assert report["datasets"]["sparkov"]["mapping"]["receiver"] == "merchant"
    assert report["datasets"]["sparkov"]["mapping"]["receiver_secondary"] == "category"

    baseline = report["frozen_external_cof_baseline"]
    assert baseline["source_commit"] == "99a445f6dc893a8c2240d950de4f92877cc07f8a"
    assert baseline["config_sha256"] == (
        "81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3"
    )
    assert baseline["model_family"] == "existing CoF-SeqGen v2.5 (non-v3)"
    preservation = report["preservation_inventory"]
    assert preservation["v2_5_final_complete"]["sha256"] == (
        "e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a"
    )
    assert preservation["v3_direct_attempt_002"]["tree_sha256"] == (
        "a5ede6b1c13b0a59d328e094444bda9cad54489b8fc3c49282004bd033dc830d"
    )
    assert report["existing_preparation_code_audit"]["sparkov"]["status"] == (
        "NOT_ADMISSIBLE_FOR_PROPOSED_EXTERNAL_PROTOCOL"
    )
    assert all(
        value == 0
        for value in report["datasets"]["sparkov"]["files"]["train"][
            "missing_by_raw_column"
        ].values()
    )
    assert all(value == 0 for value in report["execution_counts"].values())
