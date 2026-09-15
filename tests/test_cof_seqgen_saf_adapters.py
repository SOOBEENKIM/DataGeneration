from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from data.cof_seqgen_saf_adapters import (
    AdapterResult,
    adapt_amlsim,
    adapt_berka,
    adapt_controlled_coupling_dgp,
    adapt_hm,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _generator_config() -> dict:
    return yaml.safe_load(
        (REPOSITORY_ROOT / "configs/benchmark_v2/full_v2_5.yaml").read_text(
            encoding="utf-8"
        )
    )


def _controlled(kappa: float):
    return adapt_controlled_coupling_dgp(
        REPOSITORY_ROOT / "data/cof_seqgen_saf/raw/controlled_coupling_dgp",
        generator_config=_generator_config(),
        scenario="joint_semimarkov_v2b",
        kappa=kappa,
        n_entities=240,
        generation_seed=42,
        split_seed=20260826,
        cell_id=f"test_kappa_{int(kappa)}",
    )


def test_controlled_dgp_is_deterministic_and_separates_oracle_latents() -> None:
    first = _controlled(1.0)
    second = _controlled(1.0)
    pd.testing.assert_frame_equal(first.canonical.static_context, second.canonical.static_context)
    pd.testing.assert_frame_equal(first.canonical.events, second.canonical.events)
    pd.testing.assert_frame_equal(first.canonical.entity_splits, second.canonical.entity_splits)
    pd.testing.assert_frame_equal(first.oracle_events, second.oracle_events)
    assert first.canonical.events.loc[
        first.canonical.events["event_index"] == 0, "gap"
    ].isna().all()
    assert not {"gap_state", "receiver_state", "receiver_repeat"}.intersection(
        first.canonical.events.columns
    )
    assert len(first.oracle_events) == len(first.canonical.events)
    assert first.audit["negative_gap_count"] == 0


def test_amlsim_adapter_returns_materializable_result(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "ACCOUNT_ID": [1, 2, 3],
            "INIT_BALANCE": [100.0, 100.0, 100.0],
            "IS_FRAUD": [False, False, False],
            "TX_BEHAVIOR_ID": [1, 1, 1],
        }
    ).to_csv(tmp_path / "accounts.csv", index=False)
    rows = []
    for sender in (1, 2, 3):
        rows.extend(
            [
                {
                    "TX_ID": sender * 10,
                    "SENDER_ACCOUNT_ID": sender,
                    "RECEIVER_ACCOUNT_ID": (sender % 3) + 1,
                    "TX_TYPE": "TRANSFER",
                    "TX_AMOUNT": 5.0,
                    "TIMESTAMP": 0,
                    "IS_FRAUD": False,
                },
                {
                    "TX_ID": sender * 10 + 1,
                    "SENDER_ACCOUNT_ID": sender,
                    "RECEIVER_ACCOUNT_ID": (sender % 3) + 1,
                    "TX_TYPE": "TRANSFER",
                    "TX_AMOUNT": 6.0,
                    "TIMESTAMP": 1,
                    "IS_FRAUD": False,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(tmp_path / "transactions.csv", index=False)
    result = adapt_amlsim(tmp_path, split_seed=3)
    assert isinstance(result, AdapterResult)
    assert result.audit["canonical_entities"] == 3


def test_controlled_cells_keep_paired_entity_identity_and_split() -> None:
    uncoupled = _controlled(0.0)
    coupled = _controlled(1.0)
    pd.testing.assert_frame_equal(
        uncoupled.canonical.static_context,
        coupled.canonical.static_context,
    )
    pd.testing.assert_frame_equal(
        uncoupled.canonical.entity_splits,
        coupled.canonical.entity_splits,
    )
    assert (
        uncoupled.canonical.split_assignment_sha256
        == coupled.canonical.split_assignment_sha256
    )
    assert not uncoupled.canonical.events.equals(coupled.canonical.events)


def test_berka_projection_matches_tabdit_parent_and_child_field_families(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "account_id": [1, 2, 3],
            "district_id": [10, 10, 10],
            "frequency": ["MONTHLY"] * 3,
            "date": [930101, 930101, 930101],
        }
    ).to_csv(tmp_path / "account.asc", sep=";", index=False)
    pd.DataFrame({"A1": [10], "A2": ["City"], "A3": ["Region"]}).to_csv(
        tmp_path / "district.asc", sep=";", index=False
    )
    rows = []
    for entity in (1, 2, 3):
        for position, date in enumerate((930101, 930102)):
            rows.append(
                {
                    "trans_id": 10 * entity + position,
                    "account_id": entity,
                    "date": date,
                    "type": "CREDIT",
                    "operation": "TRANSFER",
                    "amount": 10.0 + position,
                    "balance": 100.0 + position,
                    "k_symbol": "HOUSEHOLD",
                    "bank": "AB",
                    "account": 1000 + entity,
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "trans.asc", sep=";", index=False)
    result = adapt_berka(tmp_path, split_seed=3)
    assert result.canonical.schema.static_context_columns == (
        "account_frequency",
        "account_district_id",
        "account_open_day",
        "city",
        "region",
    )
    assert result.canonical.schema.auxiliary_numeric_columns == ("balance",)
    assert result.canonical.schema.auxiliary_categorical_columns == (
        "transaction_type",
        "k_symbol",
    )
    assert set(result.canonical.events["receiver_or_mark"]) == {"TRANSFER"}
    assert result.audit["tabdit_parent_fields"] == 5
    assert result.audit["tabdit_child_fields_including_timestamp"] == 6


def test_hm_adapter_preserves_selected_article_context(tmp_path: Path) -> None:
    customers = [f"customer-{index}" for index in range(6)]
    transactions = []
    for index, customer in enumerate(customers):
        transactions.extend(
            [
                {
                    "t_dat": "2020-01-01",
                    "customer_id": customer,
                    "article_id": 100 + index % 2,
                    "price": 0.1,
                    "sales_channel_id": 1,
                },
                {
                    "t_dat": "2020-01-02",
                    "customer_id": customer,
                    "article_id": 101 - index % 2,
                    "price": 0.2,
                    "sales_channel_id": 2,
                },
            ]
        )
    pd.DataFrame(transactions).to_csv(
        tmp_path / "transactions_train.csv", index=False
    )
    pd.DataFrame(
        {
            "customer_id": customers,
            "club_member_status": ["ACTIVE"] * 6,
            "fashion_news_frequency": ["Regularly"] * 6,
            "age": [30] * 6,
            "postal_code": ["postal"] * 6,
        }
    ).to_csv(tmp_path / "customers.csv", index=False)
    pd.DataFrame(
        {
            "article_id": [100, 101, 999],
            "product_type_no": [1, 2, 3],
            "colour_group_code": [10, 20, 30],
        }
    ).to_csv(tmp_path / "articles.csv", index=False)
    result = adapt_hm(
        tmp_path,
        split_seed=3,
        sample_seed=4,
        customer_count=6,
        chunksize=4,
    )
    assert result.mark_context is not None
    assert set(result.mark_context["receiver_or_mark"]) == {"100", "101"}
    assert "product_type_no" in result.mark_context.columns
    assert set(result.canonical.events["receiver_or_mark"]) == {"100", "101"}
