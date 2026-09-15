"""Read-only feasibility audit for AMLSim and Sparkov external validation.

The module reads public raw CSV files and emits documentation evidence.  It
does not import model code, initialize CUDA, generate data, or execute an
external-data experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


AUDIT_SOURCE_HEAD = "91acaf97fee2ae6ada305670c06698b98a247561"
FROZEN_COF_SOURCE_COMMIT = "99a445f6dc893a8c2240d950de4f92877cc07f8a"
FROZEN_COF_CONFIG_SHA256 = (
    "81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3"
)
FROZEN_COF_MODEL_SHA256 = (
    "caf5fdb3baf367a28c6081a8ba5f4a89e59dacbba2377636acba5c051ae7006e"
)
FROZEN_COF_ADAPTER_SHA256 = (
    "48e9e1280bc23a27abf0b5f6447f6682e3b846d0d1773fb96d99abcf407dd7f3"
)
EXECUTION_COUNTS = {
    "gpu_inventory_queries": 0,
    "cuda_calls": 0,
    "model_fit_calls": 0,
    "model_sample_calls": 0,
    "data_generation_calls": 0,
    "external_validation_experiment_calls": 0,
    "fresh_test_calls": 0,
    "tstr_calls": 0,
    "privacy_calls": 0,
    "full_run_calls": 0,
}
PRESERVED_FILES = {
    "v2_5_final_complete": (
        "artifacts/benchmark_v2_5/full/FINAL_COMPLETE.json",
        "e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a",
    ),
    "v2_5_frozen_manifest": (
        "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00/data_manifest.json",
        "b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05",
    ),
    "v2_8_aggregate_complete": (
        "artifacts/benchmark_v2_8/candidate_selection/aggregate_attempt_001/AGGREGATE_COMPLETE.json",
        "58573dfb93ebed71843f286405e205669140b8cc66811ce61deebe808de513e9",
    ),
    "v3_forensic_evidence": (
        "docs/benchmark_v2/forensic_v3_attempt_002_evidence.json",
        "e89b6456f5ed404ff9e1fa8ba8fb9f901ade9ced7c4c501fb3b8f66a24f3a538",
    ),
}
PRESERVED_TREES = {
    "v2_8_candidate_selection": (
        "artifacts/benchmark_v2_8/candidate_selection",
        "7f60deb8a10cd9946e2e8a814b21c463fbf753954e7e03c44b6a7d9efb02e700",
        21,
        2_065_996,
    ),
    "v3_direct_attempt_002": (
        "artifacts/benchmark_v3/candidate_selection/candidates/"
        "cof_v3_c01_direct_joint/seed_3001/attempt_002",
        "a5ede6b1c13b0a59d328e094444bda9cad54489b8fc3c49282004bd033dc830d",
        52,
        186_589_677,
    ),
    "v3_factorized_attempt_002": (
        "artifacts/benchmark_v3/candidate_selection/candidates/"
        "cof_v3_c02_factorized_joint/seed_3001/attempt_002",
        "28f159a587fbf5d9c656bc09417bd84c283b7a5e4537280c0314b23c4d1e0da1",
        52,
        164_781_479,
    ),
}


class AuditError(RuntimeError):
    """Raised when audit input or a read-only protocol invariant is invalid."""


@dataclass(frozen=True)
class DatasetColumns:
    transaction_id: str
    entity: str
    receiver: str
    amount: str
    timestamp: str
    fraud: str
    receiver_secondary: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob_sha256(repo_root: Path, commit: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return _sha256_bytes(result.stdout)


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _tree_inventory(root: Path) -> Mapping[str, Any]:
    files: dict[str, Mapping[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files[path.relative_to(root).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "tree_sha256": _canonical_sha256({"files": files}),
        "file_count": len(files),
        "byte_count": sum(value["bytes"] for value in files.values()),
    }


def _preservation_inventory(repo_root: Path) -> Mapping[str, Any]:
    inventory: dict[str, Any] = {}
    for name, (relative, expected_hash) in PRESERVED_FILES.items():
        observed = sha256_file(repo_root / relative)
        if observed != expected_hash:
            raise AuditError(f"preserved file hash mismatch: {relative}: {observed}")
        inventory[name] = {"path": relative, "sha256": observed}
    for name, (relative, expected_hash, expected_files, expected_bytes) in PRESERVED_TREES.items():
        observed = _tree_inventory(repo_root / relative)
        expected = {
            "tree_sha256": expected_hash,
            "file_count": expected_files,
            "byte_count": expected_bytes,
        }
        if observed != expected:
            raise AuditError(f"preserved tree mismatch: {relative}: {observed}")
        inventory[name] = {"path": relative, **observed}
    return inventory


def _binary_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(np.int8)
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="raise")
        values = set(numeric.dropna().unique().tolist())
        if not values.issubset({0, 1, 0.0, 1.0}):
            raise AuditError(f"fraud column is not binary: {sorted(values)[:10]}")
        return numeric.astype(np.int8)
    normalized = series.astype("string").str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
    unknown = sorted(set(normalized.dropna()) - set(mapping))
    if unknown:
        raise AuditError(f"fraud column is not binary: {unknown[:10]}")
    return normalized.map(mapping).astype(np.int8)


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _distribution(values: pd.Series) -> Mapping[str, Any]:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    quantiles = numeric.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "count": int(numeric.size),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "min": float(quantiles.loc[0.0]),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(quantiles.loc[1.0]),
    }


def deterministic_entity_split(entities: pd.Series) -> pd.Series:
    """Stable 70/15/15 entity partition proposed for development data."""

    unique = pd.unique(entities)
    assignment: dict[Any, str] = {}
    for entity in unique:
        bucket = int(hashlib.sha256(str(entity).encode("utf-8")).hexdigest()[:16], 16) % 100
        assignment[entity] = (
            "train" if bucket < 70 else "validation" if bucket < 85 else "internal_test"
        )
    return entities.map(assignment).astype("string")


def receiver_oov_by_split(
    frame: pd.DataFrame,
    receiver_column: str,
    split_column: str,
) -> Mapping[str, Any]:
    train_vocab = set(frame.loc[frame[split_column] == "train", receiver_column].dropna())
    result: dict[str, Any] = {
        "vocabulary_source": "train",
        "train_vocabulary_cardinality": len(train_vocab),
    }
    for split in ("train", "validation", "internal_test"):
        values = frame.loc[frame[split_column] == split, receiver_column]
        oov = ~values.isin(train_vocab)
        result[split] = {
            "rows": int(values.size),
            "unique_receivers": int(values.nunique(dropna=True)),
            "oov_rows": int(oov.sum()),
            "oov_row_rate": _rate(int(oov.sum()), int(values.size)),
            "oov_categories": int(values.loc[oov].nunique(dropna=True)),
        }
    return result


def assert_calibration_sources(paths: Iterable[Path]) -> None:
    for path in paths:
        if "test" in path.name.lower():
            raise AuditError(f"test source is forbidden for calibration: {path}")


def _window_summary(
    ordered: pd.DataFrame,
    columns: DatasetColumns,
    window_id: pd.Series,
) -> Mapping[str, Any]:
    labels = _binary_labels(ordered[columns.fraud])
    scratch = pd.DataFrame(
        {
            "entity": ordered[columns.entity].to_numpy(),
            "window": window_id.to_numpy(),
            "fraud": labels.to_numpy(),
        }
    )
    windows = scratch.groupby(["entity", "window"], sort=False)["fraud"].agg(
        ["size", "max"]
    )
    positives = int(windows["max"].sum())
    return {
        "window_count": int(len(windows)),
        "positive_count": positives,
        "positive_prevalence": _rate(positives, len(windows)),
        "rows_per_window": _distribution(windows["size"]),
    }


def audit_frame(
    frame: pd.DataFrame,
    columns: DatasetColumns,
    *,
    event_window_size: int,
    time_window_width: int,
) -> Mapping[str, Any]:
    required = [
        columns.transaction_id,
        columns.entity,
        columns.receiver,
        columns.amount,
        columns.timestamp,
        columns.fraud,
    ]
    if columns.receiver_secondary:
        required.append(columns.receiver_secondary)
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise AuditError(f"missing required raw columns: {missing_columns}")
    if frame.empty:
        raise AuditError("raw frame is empty")

    label = _binary_labels(frame[columns.fraud])
    timestamp = pd.to_numeric(frame[columns.timestamp], errors="raise")
    amount = pd.to_numeric(frame[columns.amount], errors="raise")
    entity_counts = frame.groupby(columns.entity, sort=False).size()
    entity_labels = pd.DataFrame(
        {"entity": frame[columns.entity].to_numpy(), "fraud": label.to_numpy()}
    ).groupby("entity", sort=False)["fraud"].max()

    duplicate_timestamp_rows = frame.duplicated(
        [columns.entity, columns.timestamp], keep=False
    )
    duplicate_groups = (
        frame.loc[duplicate_timestamp_rows, [columns.entity, columns.timestamp]]
        .drop_duplicates()
        .shape[0]
    )
    duplicate_id_rows = frame.duplicated(columns.transaction_id, keep=False)

    original_order = pd.DataFrame(
        {"entity": frame[columns.entity].to_numpy(), "timestamp": timestamp.to_numpy()}
    )
    within_entity_monotone = bool(
        original_order.groupby("entity", sort=False)["timestamp"]
        .apply(lambda values: bool(values.is_monotonic_increasing))
        .all()
    )

    ordered = frame.sort_values(
        [columns.entity, columns.timestamp, columns.transaction_id],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered_time = pd.to_numeric(ordered[columns.timestamp], errors="raise")
    event_index = ordered.groupby(columns.entity, sort=False).cumcount()
    event_window_id = event_index // event_window_size
    event_windows = dict(_window_summary(ordered, columns, event_window_id))
    event_windows["window_size"] = event_window_size
    event_windows["tail_policy"] = "retain_nonempty_tail_with_mask"

    entity_min_time = ordered_time.groupby(ordered[columns.entity], sort=False).transform("min")
    time_window_id = ((ordered_time - entity_min_time) // time_window_width).astype(np.int64)
    time_windows = dict(_window_summary(ordered, columns, time_window_id))
    time_windows["width"] = time_window_width
    time_windows["origin"] = "per_entity_minimum_timestamp"

    fraud_count = int(label.sum())
    positive_entities = int(entity_labels.sum())
    gaps = ordered_time.groupby(ordered[columns.entity], sort=False).diff().dropna()
    audit: dict[str, Any] = {
        "rows": int(len(frame)),
        "entities": int(frame[columns.entity].nunique(dropna=True)),
        "receiver_column": columns.receiver,
        "receiver_cardinality": int(frame[columns.receiver].nunique(dropna=True)),
        "missing_protocol_columns": {
            column: int(frame[column].isna().sum()) for column in required
        },
        "duplicate_transaction_ids": {
            "row_count": int(duplicate_id_rows.sum()),
            "id_count": int(frame.loc[duplicate_id_rows, columns.transaction_id].nunique()),
        },
        "exact_duplicate_rows": int(frame.duplicated(keep=False).sum()),
        "duplicate_timestamps_global": {
            "row_count": int(frame.duplicated(columns.timestamp, keep=False).sum()),
            "timestamp_count": int(
                frame.loc[
                    frame.duplicated(columns.timestamp, keep=False), columns.timestamp
                ].nunique()
            ),
        },
        "duplicate_timestamps_within_entity": {
            "row_count": int(duplicate_timestamp_rows.sum()),
            "entity_timestamp_group_count": int(duplicate_groups),
        },
        "ordering": {
            "input_timestamp_globally_monotone": bool(timestamp.is_monotonic_increasing),
            "input_timestamp_monotone_within_entity": within_entity_monotone,
            "stable_sort_possible": bool(not frame[columns.transaction_id].isna().any()),
            "stable_sort_keys": [
                columns.entity,
                columns.timestamp,
                columns.transaction_id,
            ],
            "stable_tie_breaker": columns.transaction_id,
        },
        "entity_transaction_counts": _distribution(entity_counts),
        "transaction_fraud": {
            "count": fraud_count,
            "prevalence": _rate(fraud_count, len(frame)),
        },
        "entity_fraud": {
            "positive_count": positive_entities,
            "positive_prevalence": _rate(positive_entities, len(entity_labels)),
            "definition": "any transaction fraud within entity",
        },
        "fixed_event_windows": event_windows,
        "non_overlapping_time_windows": time_windows,
        "amount": _distribution(amount),
        "within_entity_gap": _distribution(gaps) if len(gaps) else None,
    }
    if columns.receiver_secondary:
        audit["secondary_receiver"] = {
            "column": columns.receiver_secondary,
            "cardinality": int(frame[columns.receiver_secondary].nunique(dropna=True)),
            "missing": int(frame[columns.receiver_secondary].isna().sum()),
        }
    return audit


def _raw_missing_counts(path: Path) -> Mapping[str, int]:
    counts: pd.Series | None = None
    for chunk in pd.read_csv(path, chunksize=250_000, low_memory=False):
        missing = chunk.isna().sum()
        counts = missing if counts is None else counts.add(missing, fill_value=0)
    if counts is None:
        raise AuditError(f"raw CSV is empty: {path}")
    return {str(column): int(value) for column, value in counts.items()}


def _file_record(
    path: Path,
    frame: pd.DataFrame,
    *,
    missing_by_raw_column: Mapping[str, int] | None = None,
) -> Mapping[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "missing_by_raw_column": dict(
            missing_by_raw_column
            if missing_by_raw_column is not None
            else {str(column): int(value) for column, value in frame.isna().sum().items()}
        ),
    }


def _split_report(
    frame: pd.DataFrame,
    columns: DatasetColumns,
) -> Mapping[str, Any]:
    working = frame[[columns.entity, columns.receiver, columns.fraud]].copy()
    working["split"] = deterministic_entity_split(working[columns.entity])
    working["fraud"] = _binary_labels(working[columns.fraud])
    summary: dict[str, Any] = {
        "rule": "sha256(string(entity)) first 64 bits modulo 100: 0-69 train, 70-84 validation, 85-99 internal_test",
        "entity_disjoint": True,
        "receiver_vocabulary": receiver_oov_by_split(
            working.rename(columns={columns.receiver: "receiver"}),
            "receiver",
            "split",
        ),
    }
    for split in ("train", "validation", "internal_test"):
        subset = working.loc[working["split"] == split]
        positives = int(subset["fraud"].sum())
        entity_label = subset.groupby(columns.entity, sort=False)["fraud"].max()
        summary[split] = {
            "rows": int(len(subset)),
            "entities": int(subset[columns.entity].nunique()),
            "fraud_transactions": positives,
            "transaction_fraud_prevalence": _rate(positives, len(subset)),
            "fraud_positive_entities": int(entity_label.sum()),
            "entity_fraud_prevalence": _rate(int(entity_label.sum()), len(entity_label)),
        }
    return summary


def _read_protocol_csv(path: Path, columns: DatasetColumns) -> tuple[pd.DataFrame, list[str]]:
    raw_columns = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [
        columns.transaction_id,
        columns.entity,
        columns.receiver,
        columns.amount,
        columns.timestamp,
        columns.fraud,
    ]
    if columns.receiver_secondary:
        usecols.append(columns.receiver_secondary)
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    return frame, raw_columns


def _audit_amlsim(raw_root: Path) -> Mapping[str, Any]:
    root = raw_root / "ibm_amlsim_example_accounts_transactions_alerts"
    transactions_path = root / "transactions.csv"
    accounts_path = root / "accounts.csv"
    alerts_path = root / "alerts.csv"
    columns = DatasetColumns(
        transaction_id="TX_ID",
        entity="SENDER_ACCOUNT_ID",
        receiver="RECEIVER_ACCOUNT_ID",
        amount="TX_AMOUNT",
        timestamp="TIMESTAMP",
        fraud="IS_FRAUD",
    )
    transactions, raw_columns = _read_protocol_csv(transactions_path, columns)
    accounts = pd.read_csv(accounts_path, low_memory=False)
    alerts = pd.read_csv(alerts_path, low_memory=False)
    audit = audit_frame(
        transactions,
        columns,
        event_window_size=32,
        time_window_width=10,
    )
    split = _split_report(transactions, columns)

    fraud = _binary_labels(transactions[columns.fraud])
    has_alert = transactions["TX_ID"].isin(set(alerts["TX_ID"]))
    fraud_ids = set(transactions.loc[fraud == 1, "TX_ID"])
    alert_ids = set(alerts["TX_ID"])
    return {
        "files": {
            "transactions": {
                **_file_record(
                    transactions_path,
                    transactions,
                    missing_by_raw_column=_raw_missing_counts(transactions_path),
                ),
                "columns": raw_columns,
            },
            "accounts": _file_record(accounts_path, accounts),
            "alerts": _file_record(alerts_path, alerts),
        },
        "mapping": {
            "transaction_id": columns.transaction_id,
            "timestamp": columns.timestamp,
            "entity": columns.entity,
            "receiver": columns.receiver,
            "amount": columns.amount,
            "fraud_label": columns.fraud,
        },
        "audit": audit,
        "development_entity_split": split,
        "companion_label_checks": {
            "account_is_fraud_positive": int(_binary_labels(accounts["IS_FRAUD"]).sum()),
            "alert_rows": int(len(alerts)),
            "alert_unique_ids": int(alerts["ALERT_ID"].nunique()),
            "transactions_linked_to_alert_rows": int(has_alert.sum()),
            "fraud_transaction_ids_equal_alert_transaction_ids": fraud_ids == alert_ids,
            "fraud_ids_not_in_alerts": len(fraud_ids - alert_ids),
            "alert_ids_not_marked_transaction_fraud": len(alert_ids - fraud_ids),
        },
        "label_mapping_candidates": {
            "recommended": "fixed-window Y = any transaction IS_FRAUD",
            "alternatives_not_selected": [
                "sender ACCOUNT.IS_FRAUD (entity-constant and not temporally localized)",
                "any alert membership (identical transaction-ID support to IS_FRAUD in this raw release and therefore not an independent label)",
            ],
        },
        "recommendation": {
            "entity": "SENDER_ACCOUNT_ID",
            "sequence_unit": "stable (SENDER_ACCOUNT_ID, TIMESTAMP, TX_ID) order, non-overlapping 32-event windows; retain a nonempty tail with a valid-length mask",
            "receiver": "RECEIVER_ACCOUNT_ID",
            "sequence_label_y": "1 iff any IS_FRAUD=1 transaction occurs in the window",
            "split": "entity-disjoint SHA-256 partition by SENDER_ACCOUNT_ID; fit transforms/vocabulary on train only",
            "exclude_from_features": ["IS_FRAUD", "ALERT_ID", "TX_ID"],
            "risks": [
                "TIMESTAMP is a coarse integer simulation step and has many within-entity ties; TX_ID must be the frozen tie-breaker.",
                "A receiver can be a sender in another split, so receiver vocabulary/OOV handling must not import validation/test categories into training.",
                "Account-level IS_FRAUD and alert identifiers can directly leak the target and must not enter model inputs.",
            ],
            "exclusion_criteria": [
                "missing or non-binary IS_FRAUD",
                "missing entity/timestamp/amount/receiver",
                "non-unique TX_ID preventing deterministic tie-breaking",
                "a preregistered split with no positive validation or locked-test windows",
            ],
        },
    }


def _audit_sparkov(raw_root: Path) -> Mapping[str, Any]:
    root = raw_root / "Sparkov_fraud_train_test"
    train_path = root / "fraudTrain.csv"
    test_path = root / "fraudTest.csv"
    columns = DatasetColumns(
        transaction_id="trans_num",
        entity="cc_num",
        receiver="merchant",
        amount="amt",
        timestamp="unix_time",
        fraud="is_fraud",
        receiver_secondary="category",
    )
    train, train_raw_columns = _read_protocol_csv(train_path, columns)
    test, test_raw_columns = _read_protocol_csv(test_path, columns)
    train_audit = audit_frame(
        train,
        columns,
        event_window_size=32,
        time_window_width=7 * 24 * 60 * 60,
    )
    test_audit = audit_frame(
        test,
        columns,
        event_window_size=32,
        time_window_width=7 * 24 * 60 * 60,
    )
    split = _split_report(train, columns)
    train_entities = set(train[columns.entity])
    test_entities = set(test[columns.entity])
    train_receivers = set(train[columns.receiver])
    test_receiver_oov = ~test[columns.receiver].isin(train_receivers)
    train_categories = set(train[columns.receiver_secondary])
    test_category_oov = ~test[columns.receiver_secondary].isin(train_categories)
    return {
        "files": {
            "train": {
                **_file_record(
                    train_path,
                    train,
                    missing_by_raw_column=_raw_missing_counts(train_path),
                ),
                "columns": train_raw_columns,
            },
            "test": {
                **_file_record(
                    test_path,
                    test,
                    missing_by_raw_column=_raw_missing_counts(test_path),
                ),
                "columns": test_raw_columns,
            },
        },
        "mapping": {
            "transaction_id": columns.transaction_id,
            "timestamp": columns.timestamp,
            "timestamp_human": "trans_date_trans_time",
            "entity": columns.entity,
            "receiver": columns.receiver,
            "receiver_secondary": columns.receiver_secondary,
            "amount": columns.amount,
            "fraud_label": columns.fraud,
        },
        "train_audit": train_audit,
        "locked_original_test_audit": test_audit,
        "development_entity_split_from_fraudTrain_only": split,
        "published_partition_relationship": {
            "train_entities": len(train_entities),
            "test_entities": len(test_entities),
            "overlapping_entities": len(train_entities & test_entities),
            "test_entity_overlap_fraction": _rate(
                len(train_entities & test_entities), len(test_entities)
            ),
            "test_merchant_oov_rows_vs_train": int(test_receiver_oov.sum()),
            "test_merchant_oov_row_rate_vs_train": _rate(
                int(test_receiver_oov.sum()), len(test)
            ),
            "test_merchant_oov_categories_vs_train": int(
                test.loc[test_receiver_oov, columns.receiver].nunique()
            ),
            "test_category_oov_rows_vs_train": int(test_category_oov.sum()),
            "original_test_role": "locked temporal robustness evaluation only; never calibration or candidate selection",
        },
        "label_mapping_candidates": {
            "recommended": "fixed-window Y = any transaction is_fraud",
            "alternatives_not_selected": [
                "card-level any-fraud Y (77.52% positive in fraudTrain and not temporally localized)",
                "window fraud count/rate regression (not compatible with the current binary sequence-Y contract without a separately preregistered model/interface change)",
            ],
        },
        "recommendation": {
            "entity": "cc_num",
            "sequence_unit": "stable (cc_num, unix_time, trans_num) order, non-overlapping 32-event windows; retain a nonempty tail with a valid-length mask",
            "receiver": "merchant",
            "receiver_secondary": "category is a low-cardinality merchant-type diagnostic, not the primary receiver identity",
            "sequence_label_y": "1 iff any is_fraud=1 transaction occurs in the window",
            "split": "use fraudTrain.csv only for entity-disjoint SHA-256 train/validation/internal-test development; keep fraudTest.csv untouched for a separately preregistered temporal robustness test",
            "exclude_from_features": [
                "is_fraud",
                "trans_num",
                "the blank CSV index column",
                "names/addresses/demographics unless separately justified",
            ],
            "risks": [
                "The published train/test files reuse card entities, so they are not an entity-leakage-free split.",
                "Fraud prevalence is low at transaction level and a window-level any-fraud label may remain imbalanced.",
                "merchant can contain train-only vocabulary gaps; an explicit UNK contract is required.",
                "merchant/category and amount may make Y detectable from row marginals without temporal information.",
            ],
            "exclusion_criteria": [
                "missing or non-binary is_fraud",
                "missing cc_num/unix_time/amt/merchant",
                "non-unique trans_num preventing deterministic tie-breaking",
                "a preregistered entity split with no positive validation or internal-test windows",
                "using fraudTest.csv to fit transforms, vocabulary, thresholds, or candidate choices",
            ],
        },
    }


def _frozen_baseline(repo_root: Path) -> Mapping[str, Any]:
    config_path = repo_root / "configs/benchmark_v2/full_v2_5.yaml"
    observed = {
        "model": _git_blob_sha256(
            repo_root, FROZEN_COF_SOURCE_COMMIT, "models/cof_seqgen.py"
        ),
        "adapter": _git_blob_sha256(
            repo_root,
            FROZEN_COF_SOURCE_COMMIT,
            "generators/cof_seqgen_adapter.py",
        ),
        "config": _git_blob_sha256(
            repo_root,
            FROZEN_COF_SOURCE_COMMIT,
            "configs/benchmark_v2/full_v2_5.yaml",
        ),
    }
    expected = {
        "model": FROZEN_COF_MODEL_SHA256,
        "adapter": FROZEN_COF_ADAPTER_SHA256,
        "config": FROZEN_COF_CONFIG_SHA256,
    }
    if observed != expected or sha256_file(config_path) != FROZEN_COF_CONFIG_SHA256:
        raise AuditError(f"frozen non-v3 CoF provenance mismatch: {observed}")
    return {
        "model_family": "existing CoF-SeqGen v2.5 (non-v3)",
        "source_commit": FROZEN_COF_SOURCE_COMMIT,
        "source_commit_role": "source state used for the completed v2.5 full execution and frozen here as the external baseline identity",
        "model_path": "models/cof_seqgen.py",
        "model_sha256": observed["model"],
        "adapter_path": "generators/cof_seqgen_adapter.py",
        "adapter_sha256": observed["adapter"],
        "config_path": "configs/benchmark_v2/full_v2_5.yaml",
        "config_sha256": observed["config"],
        "cof_config": {
            "architecture": "CoF-SeqGen",
            "d_model": 128,
            "n_layers": 2,
            "batch_size": 256,
            "optimizer": {"name": "Adam", "lr": 0.001, "weight_decay": 0.0},
            "requested_steps": 20000,
            "max_wall_seconds": 7200,
            "early_stopping": False,
            "coherence_lambda": 0.0,
            "cfg_dropout": 0.15,
            "guidance_scale": 2.0,
            "diffusion_steps": 50,
            "discrete_mask_max": 0.7,
            "start_from_mask": True,
            "feedback_discrete": True,
            "feedback_after": 0.3,
            "loss_positions": "all_valid",
        },
        "audit_constraint": "identity only: this audit does not adapt, train, sample, or authorize this baseline; external-data preprocessing and evaluation require a separate preregistration",
    }


def build_audit(repo_root: Path, raw_root: Path) -> Mapping[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != AUDIT_SOURCE_HEAD:
        raise AuditError(f"unexpected audit base HEAD: {head}")
    report = {
        "schema_version": "external-validation-feasibility-audit-v1",
        "audit_base_head": head,
        "scope": "read-only raw-data feasibility and source-only protocol proposal",
        "mapping_status": "recommended candidates based on observed raw columns and distributions; not an experiment authorization",
        "datasets": {
            "amlsim": _audit_amlsim(raw_root),
            "sparkov": _audit_sparkov(raw_root),
        },
        "frozen_external_cof_baseline": _frozen_baseline(repo_root),
        "existing_preparation_code_audit": {
            "amlsim": {
                "path": "data/prepare_amlsim.py",
                "sha256": sha256_file(repo_root / "data/prepare_amlsim.py"),
                "status": "REQUIRES_PROTOCOL_AMENDMENT_BEFORE_USE",
                "finding": "Its sender/receiver/amount/timestamp mapping agrees with the raw schema, but final ordering uses only (SENDER_ACCOUNT_ID, TIMESTAMP); the observed timestamp ties require frozen TX_ID tie-breaking and an entity-disjoint split contract.",
            },
            "sparkov": {
                "path": "data/prepare_sparkov.py",
                "sha256": sha256_file(repo_root / "data/prepare_sparkov.py"),
                "status": "NOT_ADMISSIBLE_FOR_PROPOSED_EXTERNAL_PROTOCOL",
                "finding": "It concatenates all provided files and drops merchant in favor of category. That would mix the published test into preparation and replace counterparty identity with merchant type; it must not be used unchanged.",
            },
        },
        "preservation_inventory": _preservation_inventory(repo_root),
        "external_protocol_proposal": {
            "preprocessing": [
                "Fit log1p/standardization parameters on valid train rows only and freeze their hashes before validation.",
                "Fit receiver vocabulary on train only; reserve explicit UNK for validation/locked-test categories.",
                "Choose timestamp bin edges, maximum length, tail policy, and class/window sampling from train only.",
            ],
            "row_shuffle_negative_control": {
                "unit": "within each fixed entity window",
                "operator": "keep timestamp positions, entity Y, sequence length, and row-feature marginals fixed; independently permute aligned (amount, receiver and other non-time row features) across valid positions, then recompute gap-aligned summaries without exposing row fraud labels",
                "decision_use": "compare an order-sensitive validation statistic/classifier with the unshuffled sequence; unchanged performance indicates row-marginal rather than sequential signal",
                "forbidden": "do not orient, tune, or select the shuffle statistic on the locked test",
            },
            "controlled_benchmark_separation": {
                "five_guard": "Do not copy v2 controlled-benchmark numeric thresholds to external data. Preserve the guard families as diagnostics, but calibrate any external hard thresholds with train-only entity bootstrap in a new preregistration.",
                "fidelity": [
                    "amount and gap marginals/effect sizes by Y",
                    "receiver PMF and UNK rate by Y",
                    "sequence length and Y prevalence",
                    "train-support validity and padding/mask contract",
                ],
                "coherence": [
                    "receiver repeat/run length and transition fidelity",
                    "joint gap-receiver distribution by Y",
                    "amount-gap and amount-receiver dependence by Y",
                    "original-versus-row-shuffled order-sensitive delta",
                ],
                "locked_test": "Use once after train-only fitting and validation-only protocol selection; no recalibration or candidate changes afterward.",
            },
        },
        "feasibility": {
            "amlsim": "FEASIBLE_WITH_PREREGISTRATION",
            "sparkov": "FEASIBLE_WITH_PREREGISTRATION_AND_SPLIT_DECISION",
            "experiment_authorized": False,
        },
        "required_user_decisions": [
            "Approve non-overlapping 32-event windows (with masked tails) versus a separately preregistered time-window alternative.",
            "Approve Sparkov merchant as primary receiver and category as secondary diagnostic.",
            "Choose whether Sparkov fraudTest.csv is a final temporal robustness evaluation despite entity overlap, or whether only the fraudTrain-derived entity-disjoint internal test is confirmatory.",
            "Approve train-only external-data thresholds and the row-shuffle coherence endpoint before any model execution.",
            "Approve the frozen v2.5 non-v3 CoF source/config identity and a separate external execution budget; neither is authorized by this audit.",
        ],
        "execution_counts": dict(EXECUTION_COUNTS),
    }
    return report


def _pct(value: float) -> str:
    return f"{100.0 * value:.6f}%"


def render_markdown(report: Mapping[str, Any]) -> str:
    aml = report["datasets"]["amlsim"]
    sp = report["datasets"]["sparkov"]
    lines = [
        "# AMLSim and Sparkov external-validation feasibility audit",
        "",
        "## Decision",
        "",
        "Both datasets are technically usable for a separately preregistered external sequence study, but this audit does **not** authorize an experiment. AMLSim is `FEASIBLE_WITH_PREREGISTRATION`; Sparkov additionally needs a decision about the published test split because card entities overlap. CoF-SeqGen v3 is closed and is not the external baseline.",
        "",
        "All reported statistics came from read-only CSV inspection. GPU/CUDA, model fit/sample, data generation, validation/test execution, TSTR, privacy, and full runs were all zero.",
        "",
        "## Frozen existing CoF baseline (non-v3)",
        "",
        f"- Source commit: `{report['frozen_external_cof_baseline']['source_commit']}`",
        f"- Model: `models/cof_seqgen.py`, SHA-256 `{report['frozen_external_cof_baseline']['model_sha256']}`",
        f"- Adapter: `generators/cof_seqgen_adapter.py`, SHA-256 `{report['frozen_external_cof_baseline']['adapter_sha256']}`",
        f"- Config: `configs/benchmark_v2/full_v2_5.yaml`, SHA-256 `{report['frozen_external_cof_baseline']['config_sha256']}`",
        "- Identity: existing CoF-SeqGen v2.5, not v3. The exact frozen model block is 20,000 requested updates, 7,200 s cap, d_model 128, 2 layers, batch 256, Adam 1e-3, diffusion steps 50, and coherence_lambda 0.0.",
        "- This freezes identity only. No external preprocessing, execution budget, fit, sample, or test is authorized.",
        "",
        "## AMLSim: observed raw schema and distributions",
        "",
        f"Raw transaction file: `{aml['files']['transactions']['path']}`; SHA-256 `{aml['files']['transactions']['sha256']}`; {aml['files']['transactions']['rows']:,} rows.",
        "",
        "Actual transaction columns: `" + "`, `".join(aml["files"]["transactions"]["columns"]) + "`.",
        "",
        "| Field | Actual column | Audit choice |",
        "|---|---|---|",
        "| Timestamp | `TIMESTAMP` | Coarse integer simulation step |",
        "| Entity | `SENDER_ACCOUNT_ID` | Recommended sequence owner |",
        "| Receiver | `RECEIVER_ACCOUNT_ID` | Recommended counterparty |",
        "| Amount | `TX_AMOUNT` | Train-only `log1p` then standardization is feasible |",
        "| Row fraud | `IS_FRAUD` | Never an input; window Y is any-fraud |",
        "| Stable tie-breaker | `TX_ID` | Required because timestamps repeat |",
        "",
        f"- Entities: {aml['audit']['entities']:,}; receivers: {aml['audit']['receiver_cardinality']:,}.",
        f"- Transaction fraud: {aml['audit']['transaction_fraud']['count']:,}/{aml['audit']['rows']:,} ({_pct(aml['audit']['transaction_fraud']['prevalence'])}).",
        f"- Entity any-fraud: {aml['audit']['entity_fraud']['positive_count']:,}/{aml['audit']['entities']:,} ({_pct(aml['audit']['entity_fraud']['positive_prevalence'])}).",
        f"- Non-overlapping 32-event window any-fraud: {aml['audit']['fixed_event_windows']['positive_count']:,}/{aml['audit']['fixed_event_windows']['window_count']:,} ({_pct(aml['audit']['fixed_event_windows']['positive_prevalence'])}).",
        f"- Non-overlapping 10-step window any-fraud: {aml['audit']['non_overlapping_time_windows']['positive_count']:,}/{aml['audit']['non_overlapping_time_windows']['window_count']:,} ({_pct(aml['audit']['non_overlapping_time_windows']['positive_prevalence'])}).",
        f"- Within-entity duplicate-timestamp rows: {aml['audit']['duplicate_timestamps_within_entity']['row_count']:,}; duplicate entity-time groups: {aml['audit']['duplicate_timestamps_within_entity']['entity_timestamp_group_count']:,}.",
        f"- Missing protocol fields: `{json.dumps(aml['audit']['missing_protocol_columns'], sort_keys=True)}`; duplicate transaction-ID rows: {aml['audit']['duplicate_transaction_ids']['row_count']}.",
        "",
        "| AMLSim entity transactions | min | p05 | median | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Count | {aml['audit']['entity_transaction_counts']['min']:.0f} | {aml['audit']['entity_transaction_counts']['p05']:.0f} | {aml['audit']['entity_transaction_counts']['median']:.0f} | {aml['audit']['entity_transaction_counts']['p95']:.0f} | {aml['audit']['entity_transaction_counts']['p99']:.0f} | {aml['audit']['entity_transaction_counts']['max']:.0f} |",
        "",
        f"The proposed entity split has {aml['development_entity_split']['train']['entities']:,}/{aml['development_entity_split']['validation']['entities']:,}/{aml['development_entity_split']['internal_test']['entities']:,} train/validation/internal-test senders. A train-only receiver vocabulary has {aml['development_entity_split']['receiver_vocabulary']['train_vocabulary_cardinality']:,} categories; validation OOV is {aml['development_entity_split']['receiver_vocabulary']['validation']['oov_rows']:,} rows ({_pct(aml['development_entity_split']['receiver_vocabulary']['validation']['oov_row_rate'])}) and internal-test OOV is {aml['development_entity_split']['receiver_vocabulary']['internal_test']['oov_rows']:,} rows ({_pct(aml['development_entity_split']['receiver_vocabulary']['internal_test']['oov_row_rate'])}).",
        "",
        "Recommended mapping: sender account entity, stable event order `(SENDER_ACCOUNT_ID, TIMESTAMP, TX_ID)`, non-overlapping 32-event windows with masked tails, receiver account, and `Y=1` iff any row in the window has `IS_FRAUD=1`. Split sender entities by a frozen hash. `ALERT_ID`, `IS_FRAUD`, and `TX_ID` are excluded from model features. Receiver OOV is handled by a train-fitted vocabulary and explicit UNK.",
        "",
        "## Sparkov: observed raw schema and distributions",
        "",
        f"Train file: `{sp['files']['train']['path']}`; SHA-256 `{sp['files']['train']['sha256']}`; {sp['files']['train']['rows']:,} rows. Locked published test: `{sp['files']['test']['path']}`; SHA-256 `{sp['files']['test']['sha256']}`; {sp['files']['test']['rows']:,} rows.",
        "",
        "Actual columns: `" + "`, `".join(sp["files"]["train"]["columns"]) + "`. The first empty CSV header is an exported index and is not a feature.",
        "",
        "| Field | Actual column | Audit choice |",
        "|---|---|---|",
        "| Timestamp | `unix_time` (`trans_date_trans_time` is human-readable) | Stable numeric order |",
        "| Entity | `cc_num` | Recommended card sequence owner |",
        "| Receiver | `merchant` | Recommended counterparty identity |",
        "| Secondary receiver diagnostic | `category` | Merchant type, not receiver identity |",
        "| Amount | `amt` | Train-only `log1p` then standardization is feasible |",
        "| Row fraud | `is_fraud` | Never an input; window Y is any-fraud |",
        "| Stable tie-breaker | `trans_num` | Deterministic tie-breaker |",
        "",
        f"- Train entities: {sp['train_audit']['entities']:,}; merchants: {sp['train_audit']['receiver_cardinality']:,}; categories: {sp['train_audit']['secondary_receiver']['cardinality']:,}.",
        f"- Train transaction fraud: {sp['train_audit']['transaction_fraud']['count']:,}/{sp['train_audit']['rows']:,} ({_pct(sp['train_audit']['transaction_fraud']['prevalence'])}).",
        f"- Train entity any-fraud: {sp['train_audit']['entity_fraud']['positive_count']:,}/{sp['train_audit']['entities']:,} ({_pct(sp['train_audit']['entity_fraud']['positive_prevalence'])}).",
        f"- Train 32-event window any-fraud: {sp['train_audit']['fixed_event_windows']['positive_count']:,}/{sp['train_audit']['fixed_event_windows']['window_count']:,} ({_pct(sp['train_audit']['fixed_event_windows']['positive_prevalence'])}).",
        f"- Locked-test transaction fraud: {sp['locked_original_test_audit']['transaction_fraud']['count']:,}/{sp['locked_original_test_audit']['rows']:,} ({_pct(sp['locked_original_test_audit']['transaction_fraud']['prevalence'])}).",
        f"- Published test entities overlapping train: {sp['published_partition_relationship']['overlapping_entities']:,}/{sp['published_partition_relationship']['test_entities']:,} ({_pct(sp['published_partition_relationship']['test_entity_overlap_fraction'])}).",
        f"- Missing train protocol fields: `{json.dumps(sp['train_audit']['missing_protocol_columns'], sort_keys=True)}`; duplicate transaction-ID rows: {sp['train_audit']['duplicate_transaction_ids']['row_count']}.",
        "",
        "| Sparkov train entity transactions | min | p05 | median | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Count | {sp['train_audit']['entity_transaction_counts']['min']:.0f} | {sp['train_audit']['entity_transaction_counts']['p05']:.1f} | {sp['train_audit']['entity_transaction_counts']['median']:.1f} | {sp['train_audit']['entity_transaction_counts']['p95']:.1f} | {sp['train_audit']['entity_transaction_counts']['p99']:.1f} | {sp['train_audit']['entity_transaction_counts']['max']:.0f} |",
        "",
        f"The fraudTrain-only entity split has {sp['development_entity_split_from_fraudTrain_only']['train']['entities']:,}/{sp['development_entity_split_from_fraudTrain_only']['validation']['entities']:,}/{sp['development_entity_split_from_fraudTrain_only']['internal_test']['entities']:,} train/validation/internal-test cards. All 693 merchants appear in the proposed train split, giving zero merchant OOV rows in those feasibility partitions; this observed zero must not remove the required UNK contract. The published test also has zero merchant OOV rows relative to fraudTrain, but 98.27% entity overlap remains the dominant split concern.",
        "",
        "Recommended mapping: card entity, stable `(cc_num, unix_time, trans_num)` order, non-overlapping 32-event windows with masked tails, `merchant` receiver, and window any-fraud Y. Development uses only `fraudTrain.csv` with an entity-disjoint hash split. `fraudTest.csv` remains untouched until a separately preregistered final temporal robustness evaluation; it is not entity-disjoint and must never calibrate transforms, vocabulary, thresholds, or candidates.",
        "",
        "## Train-only transforms, vocabulary, and leakage control",
        "",
        "For both datasets, fit amount `log1p`/standardization, time or gap bins, receiver vocabulary/UNK, sequence length/tail policy, and any external hard thresholds on the training entities only. Freeze source/data/config hashes before validation. Validation selects only a preregistered protocol; the locked test is read once. Entity IDs themselves are not model features.",
        "",
        "The proposed deterministic feasibility split is SHA-256(entity) modulo 100: 70% train, 15% validation, 15% internal test. The exact split counts and receiver OOV rates are in the JSON evidence. This is a feasibility calculation, not a final preregistration.",
        "",
        "## Existing preparation-code compatibility",
        "",
        f"`data/prepare_amlsim.py` (SHA-256 `{report['existing_preparation_code_audit']['amlsim']['sha256']}`) agrees with the raw sender/receiver/amount/timestamp mapping, but its final order omits `TX_ID`; it is not admissible unchanged because 888,787 rows participate in within-sender timestamp ties. `data/prepare_sparkov.py` (SHA-256 `{report['existing_preparation_code_audit']['sparkov']['sha256']}`) concatenates supplied files and drops `merchant` in favor of `category`; it is not admissible because that can mix the published test into preparation and discards the actual counterparty identity.",
        "",
        "## Row-shuffle negative control",
        "",
        "Within every entity window, keep timestamp positions, Y, length, padding mask, and row-feature marginals fixed; independently permute aligned non-time feature tuples such as `(amount, receiver)` over valid positions. Recompute gap-aligned summaries without exposing row-level fraud labels. Compare the same order-sensitive statistic/classifier on original and shuffled sequences. If discrimination or coherence does not fall, the external signal is primarily row-level and a sequence claim is unsupported. Orientation, thresholding, and selection are train/validation-only.",
        "",
        "## External fidelity/coherence protocol (proposal only)",
        "",
        "The controlled benchmark's five metric families remain useful diagnostics, but its numeric thresholds must **not** be transplanted. External hard thresholds require train-only entity bootstrap and preregistration. Report row fidelity (amount/gap marginals and Y effects, receiver PMF/UNK, length/Y prevalence, support/mask) separately from sequence coherence (repeat/run/transition, joint gap-receiver, cross-channel dependence, original-versus-shuffled delta). The locked test cannot revise either family.",
        "",
        "## Risks, exclusion criteria, and decisions still required",
        "",
    ]
    for decision in report["required_user_decisions"]:
        lines.append(f"- {decision}")
    lines.extend(
        [
            "",
            "Dataset-specific risks and exclusion criteria are recorded verbatim in the JSON evidence. If any required protocol field is missing, the transaction ID cannot deterministically break timestamp ties, a frozen split lacks positive validation/test windows, or Sparkov's published test is used for calibration, execution must be refused.",
            "",
            "## Preservation and execution accounting",
            "",
            f"- Audit base HEAD: `{report['audit_base_head']}`",
            "- Existing v2.5-v3 runtime, data, configs, checkpoints, and forensic artifacts were read only and were not regenerated or moved.",
            "- Preservation anchors reverified: v2.5 FINAL_COMPLETE/frozen manifest, v2.8 candidate tree, both v3 attempt_002 trees, and the v3 forensic evidence. Full paths, hashes, counts, and byte sizes are in the JSON evidence.",
            "- GPU inventory query, CUDA, model fit/sample, data generation, external validation/test, TSTR, privacy, and full run counts: all zero.",
            "",
        ]
    )
    return "\n".join(lines)


def write_bundle(report: Mapping[str, Any], json_path: Path, markdown_path: Path) -> None:
    for path in (json_path, markdown_path):
        if path.exists():
            raise AuditError(f"append-only output exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown_text = render_markdown(report)
    json_path.write_text(json_text, encoding="utf-8", newline="\n")
    markdown_path.write_text(markdown_text, encoding="utf-8", newline="\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_audit(args.repo_root.resolve(), args.raw_root.resolve())
    write_bundle(report, args.json.resolve(), args.markdown.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
