import pandas as pd
import pytest
from pathlib import Path

from data.external_sequence_adapter import (
    ExternalProtocolError,
    build_raw_windows,
    encode_windows,
    fit_train_transforms,
    group_stratified_entity_split,
    prepare_external_dataset,
    read_development_csv,
)
from scripts.plan_external_sequence_protocol import build_external_protocol_plan


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_amlsim_windows_use_timestamp_then_tx_id_and_do_not_overlap():
    rows = []
    for tx_id in range(64):
        rows.append(
            {
                "TX_ID": tx_id,
                "SENDER_ACCOUNT_ID": "sender-a",
                "RECEIVER_ACCOUNT_ID": f"receiver-{tx_id % 5}",
                "TX_AMOUNT": float(tx_id + 1),
                "TIMESTAMP": tx_id // 3,
                "IS_FRAUD": tx_id == 35,
            }
        )
    raw = pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)

    windows = build_raw_windows(raw, dataset="amlsim")

    assert [window.length for window in windows] == [32, 32]
    assert windows[0].transaction_ids == tuple(range(32))
    assert windows[1].transaction_ids == tuple(range(32, 64))
    assert set(windows[0].transaction_ids).isdisjoint(windows[1].transaction_ids)


def test_sparkov_uses_datetime_stable_source_order_and_window_any_fraud_label():
    rows = []
    for minute in reversed(range(24)):
        for within_time in ("second", "first"):
            rows.append(
                {
                    "trans_num": f"{minute:02d}-{within_time}",
                    "cc_num": 1234,
                    "merchant": f"merchant-{minute % 3}",
                    "amt": float(minute + 1),
                    "trans_date_trans_time": f"2020-01-01 00:{minute:02d}:00",
                    "is_fraud": minute == 23 and within_time == "first",
                }
            )
    raw = pd.DataFrame(rows)

    windows = build_raw_windows(raw, dataset="sparkov")

    assert [window.length for window in windows] == [32, 16]
    assert windows[0].transaction_ids[:4] == (
        "00-second",
        "00-first",
        "01-second",
        "01-first",
    )
    assert windows[0].y_entity == 0
    assert windows[1].y_entity == 1


def test_tail_shorter_than_sixteen_is_excluded():
    raw = pd.DataFrame(
        {
            "TX_ID": range(47),
            "SENDER_ACCOUNT_ID": ["sender-a"] * 47,
            "RECEIVER_ACCOUNT_ID": ["receiver-a"] * 47,
            "TX_AMOUNT": [1.0] * 47,
            "TIMESTAMP": range(47),
            "IS_FRAUD": [0] * 46 + [1],
        }
    )

    windows = build_raw_windows(raw, dataset="amlsim")

    assert [window.length for window in windows] == [32]
    assert windows[0].transaction_ids == tuple(range(32))
    assert windows[0].y_entity == 0


def test_group_stratified_split_is_entity_disjoint_and_70_15_15_per_label():
    raw = pd.DataFrame(
        {
            "TX_ID": range(60),
            "SENDER_ACCOUNT_ID": [f"sender-{index}" for index in range(60)],
            "IS_FRAUD": [1] * 20 + [0] * 40,
        }
    )

    assignment = group_stratified_entity_split(raw, dataset="amlsim", seed=20260801)

    assert assignment.counts_by_label == {
        "0": {"train": 28, "validation": 6, "internal_test": 6},
        "1": {"train": 14, "validation": 3, "internal_test": 3},
    }
    split_sets = {
        split: set(assignment.entities_by_split[split])
        for split in ("train", "validation", "internal_test")
    }
    assert split_sets["train"].isdisjoint(split_sets["validation"])
    assert split_sets["train"].isdisjoint(split_sets["internal_test"])
    assert split_sets["validation"].isdisjoint(split_sets["internal_test"])


def test_duplicate_transaction_identity_fails_closed_before_windowing():
    raw = pd.DataFrame(
        {
            "TX_ID": [7] * 16,
            "SENDER_ACCOUNT_ID": ["sender-a"] * 16,
            "RECEIVER_ACCOUNT_ID": ["receiver-a"] * 16,
            "TX_AMOUNT": [1.0] * 16,
            "TIMESTAMP": list(range(16)),
            "IS_FRAUD": [0] * 16,
        }
    )

    with pytest.raises(ExternalProtocolError, match="duplicate transaction identity"):
        build_raw_windows(raw, dataset="amlsim")


def test_amount_gap_and_receiver_state_fit_train_only_with_unk_on_validation():
    train = pd.DataFrame(
        {
            "TX_ID": range(32),
            "SENDER_ACCOUNT_ID": ["train-sender"] * 32,
            "RECEIVER_ACCOUNT_ID": [f"known-{index % 2}" for index in range(32)],
            "TX_AMOUNT": [float(index + 1) for index in range(32)],
            "TIMESTAMP": list(range(32)),
            "IS_FRAUD": [0] * 32,
        }
    )
    validation = pd.DataFrame(
        {
            "TX_ID": range(100, 116),
            "SENDER_ACCOUNT_ID": ["validation-sender"] * 16,
            "RECEIVER_ACCOUNT_ID": ["never-seen-in-train"] * 16,
            "TX_AMOUNT": [1_000_000.0] * 16,
            "TIMESTAMP": [index * 10_000 for index in range(16)],
            "IS_FRAUD": [0] * 16,
        }
    )

    state = fit_train_transforms(train, dataset="amlsim", gap_bins=4, fit_role="train")
    encoded = encode_windows(
        build_raw_windows(validation, dataset="amlsim"),
        state,
    )

    assert state.receiver_to_code == {"known-0": 2, "known-1": 3}
    assert state.unk_code == 1
    assert state.fit_role == "train"
    assert state.fit_transaction_count == 32
    assert len(state.fit_transaction_ids_sha256) == 64
    assert len(state.fit_entity_ids_sha256) == 64
    assert 1_000_000.0 not in state.gap_edges
    assert set(encoded.x_cat[encoded.valid_mask, 0]) == {state.unk_code}
    assert encoded.x_num[encoded.valid_mask].mean() > 1.0


def test_sparkov_public_test_is_rejected_before_csv_read(monkeypatch, tmp_path):
    calls = []

    def forbidden_read(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("public test CSV must not be read")

    monkeypatch.setattr(pd, "read_csv", forbidden_read)

    with pytest.raises(ExternalProtocolError, match="public fraudTest is locked"):
        read_development_csv(
            tmp_path / "fraudTest.csv",
            dataset="sparkov",
            source_role="public_temporal_test",
        )
    assert calls == []


def test_prepared_dataset_has_no_entity_window_or_transaction_leakage():
    rows = []
    tx_id = 0
    for entity_index in range(12):
        for position in range(16):
            rows.append(
                {
                    "TX_ID": tx_id,
                    "SENDER_ACCOUNT_ID": f"sender-{entity_index}",
                    "RECEIVER_ACCOUNT_ID": f"receiver-{entity_index % 4}",
                    "TX_AMOUNT": float(position + 1),
                    "TIMESTAMP": position,
                    "IS_FRAUD": int(entity_index < 6 and position == 8),
                }
            )
            tx_id += 1
    raw = pd.DataFrame(rows)

    prepared = prepare_external_dataset(
        raw,
        dataset="amlsim",
        source_role="transactions_development",
        split_seed=20260801,
        gap_bins=4,
    )

    assert prepared.leakage_audit == {
        "entity_overlap_count": 0,
        "window_overlap_count": 0,
        "transaction_overlap_count": 0,
    }
    assert prepared.transforms.fit_role == "train"
    assert prepared.transforms.fit_transaction_count == 8 * 16
    assert prepared.split_assignment.counts_by_label == {
        "0": {"train": 4, "validation": 1, "internal_test": 1},
        "1": {"train": 4, "validation": 1, "internal_test": 1},
    }
    assert prepared.train.batch.valid_mask.sum() == 8 * 16
    assert prepared.validation.batch.valid_mask.sum() == 2 * 16
    assert prepared.internal_test.batch.valid_mask.sum() == 2 * 16


def test_public_adapter_transform_hash_is_unchanged_by_validation_rows():
    rows = []
    tx_id = 0
    for entity_index in range(12):
        for position in range(16):
            rows.append(
                {
                    "TX_ID": tx_id,
                    "SENDER_ACCOUNT_ID": f"sender-{entity_index}",
                    "RECEIVER_ACCOUNT_ID": f"receiver-{entity_index}",
                    "TX_AMOUNT": float(position + 1),
                    "TIMESTAMP": position,
                    "IS_FRAUD": int(entity_index < 6 and position == 8),
                }
            )
            tx_id += 1
    raw = pd.DataFrame(rows)
    first = prepare_external_dataset(
        raw,
        dataset="amlsim",
        source_role="transactions_development",
        split_seed=20260801,
        gap_bins=4,
    )
    validation_entities = set(first.split_assignment.entities_by_split["validation"])
    mutated = raw.copy()
    validation_rows = mutated["SENDER_ACCOUNT_ID"].isin(validation_entities)
    mutated.loc[validation_rows, "TX_AMOUNT"] = 99_999_999.0
    mutated.loc[validation_rows, "RECEIVER_ACCOUNT_ID"] = "validation-only-receiver"

    second = prepare_external_dataset(
        mutated,
        dataset="amlsim",
        source_role="transactions_development",
        split_seed=20260801,
        gap_bins=4,
    )

    assert second.transforms.state_sha256 == first.transforms.state_sha256
    assert set(second.validation.batch.x_cat[second.validation.batch.valid_mask, 0]) == {
        second.transforms.unk_code
    }


def test_source_only_plan_freezes_non_v3_baseline_and_protocol_without_execution():
    plan = build_external_protocol_plan(
        repo_root=REPOSITORY_ROOT,
        config_path=REPOSITORY_ROOT
        / "configs/benchmark_v2/external_sequence_protocol_v1.yaml",
        mode="plan",
    )

    assert plan["frozen_baseline"]["source_commit"] == (
        "99a445f6dc893a8c2240d950de4f92877cc07f8a"
    )
    assert plan["frozen_baseline"]["config_sha256"] == (
        "81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3"
    )
    assert plan["datasets"]["amlsim"]["sort"] == ["TIMESTAMP", "TX_ID"]
    assert plan["datasets"]["sparkov"]["receiver"] == "merchant"
    assert plan["datasets"]["sparkov"]["public_test_role"] == (
        "locked_temporal_robustness_only"
    )
    assert plan["window"] == {
        "length": 32,
        "minimum_tail": 16,
        "stride": 32,
        "padding": "right_zero_with_prefix_valid_mask",
    }
    assert plan["external_thresholds"]["execution_enabled"] is False
    assert plan["preservation"]["v2_5_final_complete"]["sha256"] == (
        "e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a"
    )
    assert plan["preservation"]["v3_forensic_evidence"]["sha256"] == (
        "e89b6456f5ed404ff9e1fa8ba8fb9f901ade9ced7c4c501fb3b8f66a24f3a538"
    )
    assert all(value == 0 for value in plan["execution_counts"].values())


def test_dry_run_reads_only_hashes_and_headers_and_creates_no_runtime():
    dry_run = build_external_protocol_plan(
        repo_root=REPOSITORY_ROOT,
        config_path=REPOSITORY_ROOT
        / "configs/benchmark_v2/external_sequence_protocol_v1.yaml",
        mode="dry-run",
    )

    assert dry_run["raw_read_counts"] == {
        "raw_hash_reads": 3,
        "raw_header_reads": 3,
    }
    assert dry_run["verified_raw_sources"]["sparkov_public_test_locked"][
        "access"
    ] == "read_only_hash_and_header"
    assert all(
        source["schema_contract"] == "PASS"
        for source in dry_run["verified_raw_sources"].values()
    )
    assert dry_run["runtime_artifacts_created"] is False
    assert dry_run["execution_authorized"] is False
    assert all(value == 0 for value in dry_run["execution_counts"].values())
