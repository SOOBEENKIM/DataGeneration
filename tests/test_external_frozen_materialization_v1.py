import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.external_sequence_adapter import prepare_external_dataset
from scripts.materialize_external_sequence_protocol_v1 import (
    ExternalMaterializationError,
    audit_frozen_cof_compatibility,
    build_materialization_plan,
    load_materialization_raw,
    validate_materialization_authorization,
    validate_prepared_materialization,
    write_frozen_dataset_attempt,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs/benchmark_v2/external_frozen_materialization_v1.yaml"
)


def _amlsim_fixture() -> pd.DataFrame:
    rows = []
    transaction_id = 0
    for entity_index in range(12):
        for position in range(16):
            rows.append(
                {
                    "TX_ID": transaction_id,
                    "SENDER_ACCOUNT_ID": f"sender-{entity_index}",
                    "RECEIVER_ACCOUNT_ID": f"receiver-{entity_index}",
                    "TX_AMOUNT": float(position + entity_index + 1),
                    "TIMESTAMP": position * (entity_index + 1),
                    "IS_FRAUD": int(entity_index < 6 and position == 8),
                }
            )
            transaction_id += 1
    return pd.DataFrame(rows)


def test_plan_and_dry_run_validate_inputs_without_writes_or_execution():
    plan = build_materialization_plan(
        repo_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
        mode="plan",
    )
    dry_run = build_materialization_plan(
        repo_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
        mode="dry-run",
    )

    assert set(plan["datasets"]) == {"amlsim", "sparkov"}
    assert plan["protocol_config_sha256"] == (
        "434e0694df60a8b85ba073e4bcaed9f6c5a1bf43ca2da61be69d87c845e1e3cc"
    )
    assert plan["attempt"] == "attempt_001"
    assert dry_run["raw_read_counts"] == {
        "raw_hash_reads": 2,
        "raw_header_reads": 2,
        "raw_body_reads": 0,
    }
    assert set(dry_run["verified_raw_sources"]) == {
        "amlsim_development",
        "sparkov_development",
    }
    assert dry_run["sparkov_public_test_access"] == {
        "hash_reads": 0,
        "header_reads": 0,
        "body_reads": 0,
        "csv_parse_calls": 0,
    }
    assert dry_run["expected_counts"]["amlsim"][
        "pre_minimum_tail_candidate_windows"
    ] == 45501
    assert dry_run["expected_counts"]["sparkov"][
        "pre_minimum_tail_candidate_windows"
    ] == 41019
    assert dry_run["expected_counts"]["sparkov"][
        "protocol_retained_window_conservative_range"
    ] == {"minimum": 40036, "maximum": 41019}
    assert dry_run["expected_counts"]["exact_split_and_window_counts"] == (
        "deferred_to_authorized_body_read"
    )
    assert plan["runtime_artifacts_created"] is False
    assert dry_run["runtime_artifacts_created"] is False
    assert all(value == 0 for value in plan["execution_counts"].values())
    assert all(value == 0 for value in dry_run["execution_counts"].values())
    for dataset_plan in plan["datasets"].values():
        data_attempt = REPOSITORY_ROOT / dataset_plan["data_attempt_path"]
        artifact_attempt = REPOSITORY_ROOT / dataset_plan["artifact_attempt_path"]
        # Plan/dry-run must tolerate a prior immutable COMPLETE attempt without
        # treating it as a writable target.  A fresh checkout may have neither.
        assert data_attempt.exists() == artifact_attempt.exists()
        if artifact_attempt.exists():
            complete = json.loads(
                (artifact_attempt / "COMPLETE.json").read_text(encoding="utf-8")
            )
            assert complete["status"] == "COMPLETE"


def test_frozen_cof_accepts_dynamic_external_shapes_without_architecture_change():
    audit = audit_frozen_cof_compatibility(REPOSITORY_ROOT)

    assert audit["status"] == "PASS"
    assert audit["frozen_source_commit"] == (
        "99a445f6dc893a8c2240d950de4f92877cc07f8a"
    )
    assert audit["amount_input_width"] == 1
    assert audit["gap_bin_cardinality"] == 16
    assert audit["receiver_input_width"] == 1
    assert audit["sequence_length"] == 32
    assert audit["datasets"]["amlsim"]["receiver_cardinality_upper_bound"] == 9928
    assert audit["datasets"]["sparkov"]["receiver_cardinality_upper_bound"] == 695
    assert all(item["compatible"] for item in audit["datasets"].values())
    assert audit["architecture_change_required"] is False
    assert audit["training_runner_created"] is False


def test_fixture_materialization_writes_complete_append_only_contract(tmp_path):
    raw = _amlsim_fixture()
    prepared = prepare_external_dataset(
        raw,
        dataset="amlsim",
        source_role="transactions_development",
        split_seed=20260801,
        gap_bins=16,
    )
    raw_bytes = raw.to_csv(index=False).encode("utf-8")
    data_attempt = tmp_path / "data" / "amlsim" / "attempt_001"
    artifact_attempt = tmp_path / "artifacts" / "amlsim" / "attempt_001"

    result = write_frozen_dataset_attempt(
        prepared=prepared,
        data_attempt=data_attempt,
        artifact_attempt=artifact_attempt,
        raw_manifest={
            "path": "/fixture/amlsim.csv",
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "rows": len(raw),
            "columns": raw.columns.tolist(),
        },
        provenance={
            "source_commit": "fixture-source",
            "materialization_config_sha256": "a" * 64,
            "protocol_config_sha256": "434e0694" + "0" * 56,
        },
    )

    required_data = {
        "raw_schema_manifest.json",
        "entity_split_manifest.json",
        "window_manifest.json",
        "train.npz",
        "validation.npz",
        "internal_test.npz",
        "train_transform_state.json",
        "summary.json",
        "leakage_audit.json",
        "provenance_manifest.json",
    }
    assert required_data <= {path.name for path in data_attempt.iterdir()}
    assert {
        "RUNNING.json",
        "checksum_manifest.json",
        "artifact_index.json",
        "COMPLETE.json",
    } <= {path.name for path in artifact_attempt.iterdir()}
    assert not (artifact_attempt / "FAILED.json").exists()
    complete = json.loads((artifact_attempt / "COMPLETE.json").read_text())
    assert complete["status"] == "COMPLETE"
    assert complete["dataset"] == "amlsim"
    checksums = json.loads(
        (artifact_attempt / "checksum_manifest.json").read_text()
    )
    assert set(required_data) <= set(checksums["data_files"])
    for split in ("train", "validation", "internal_test"):
        with np.load(data_attempt / f"{split}.npz", allow_pickle=False) as arrays:
            assert set(arrays.files) == {
                "x_num",
                "dt_bin",
                "x_cat",
                "valid_mask",
                "y_entity",
                "lengths",
                "entity_ids",
            }
    summary = json.loads((data_attempt / "summary.json").read_text())
    assert summary["receiver_vocabulary_cardinality"] == (
        len(prepared.transforms.receiver_to_code) + 2
    )
    assert summary["splits"]["train"]["pad_count"] == 8 * 16
    assert summary["splits"]["validation"]["unk_count"] == 2 * 16
    assert summary["splits"]["train"]["transaction_fraud_count"] == 4
    assert summary["splits"]["train"]["transaction_fraud_prevalence"] == 4 / 128
    transform = json.loads(
        (data_attempt / "train_transform_state.json").read_text()
    )
    assert transform["fit_role"] == "train"
    assert transform["gap_edges"][-1] == "+inf"
    assert transform["state_sha256"] == prepared.transforms.state_sha256
    assert result["status"] == "COMPLETE"

    original_complete_hash = hashlib.sha256(
        (artifact_attempt / "COMPLETE.json").read_bytes()
    ).hexdigest()
    with pytest.raises(ExternalMaterializationError, match="already exists"):
        write_frozen_dataset_attempt(
            prepared=prepared,
            data_attempt=data_attempt,
            artifact_attempt=artifact_attempt,
            raw_manifest={"sha256": "b" * 64},
            provenance={"source_commit": "fixture-source"},
        )
    assert hashlib.sha256(
        (artifact_attempt / "COMPLETE.json").read_bytes()
    ).hexdigest() == original_complete_hash


def test_materializer_rejects_sparkov_public_test_before_csv_parse(monkeypatch, tmp_path):
    calls = []

    def forbidden_read(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Sparkov fraudTest body must not be parsed")

    monkeypatch.setattr(pd, "read_csv", forbidden_read)
    with pytest.raises(Exception, match="public fraudTest is locked"):
        load_materialization_raw(
            tmp_path / "fraudTest.csv",
            dataset="sparkov",
            source_role="locked_temporal_robustness_only",
        )
    assert calls == []


def test_materializer_fails_closed_on_leakage_or_non_unk_unseen_receiver():
    prepared = prepare_external_dataset(
        _amlsim_fixture(),
        dataset="amlsim",
        source_role="transactions_development",
        split_seed=20260801,
        gap_bins=16,
    )
    leaked = replace(
        prepared,
        leakage_audit={
            "entity_overlap_count": 1,
            "window_overlap_count": 0,
            "transaction_overlap_count": 0,
        },
    )
    with pytest.raises(ExternalMaterializationError, match="leakage detected"):
        validate_prepared_materialization(leaked)

    forged_clean_audit = replace(
        prepared,
        validation=prepared.train,
        leakage_audit={
            "entity_overlap_count": 0,
            "window_overlap_count": 0,
            "transaction_overlap_count": 0,
        },
    )
    with pytest.raises(ExternalMaterializationError, match="independent leakage audit"):
        validate_prepared_materialization(forged_clean_audit)

    first_valid = np.argwhere(prepared.validation.batch.valid_mask)[0]
    prepared.validation.batch.x_cat[first_valid[0], first_valid[1], 0] = 999_999
    with pytest.raises(ExternalMaterializationError, match="outside train vocabulary"):
        validate_prepared_materialization(prepared)


def test_execution_authorization_is_hash_bound_and_mismatch_fails_before_raw_read(
    monkeypatch,
):
    protocol_path = (
        REPOSITORY_ROOT / "configs/benchmark_v2/external_sequence_protocol_v1.yaml"
    )
    materializer_path = (
        REPOSITORY_ROOT / "scripts/materialize_external_sequence_protocol_v1.py"
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    protocol = __import__("yaml").safe_load(protocol_path.read_text())
    authorization = {
        "schema_version": "external-frozen-materialization-authorization-v1",
        "scope": "external_frozen_data_materialization_v1_only",
        "source_commit": source_commit,
        "materializer_source_sha256": hashlib.sha256(
            materializer_path.read_bytes()
        ).hexdigest(),
        "materialization_config_sha256": hashlib.sha256(
            CONFIG_PATH.read_bytes()
        ).hexdigest(),
        "protocol_config_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "attempt": "attempt_001",
        "datasets": {
            "amlsim": {
                "path": protocol["datasets"]["amlsim"]["development_source"]["path"],
                "sha256": protocol["datasets"]["amlsim"]["development_source"]["sha256"],
                "source_role": "transactions_development",
            },
            "sparkov": {
                "path": protocol["datasets"]["sparkov"]["development_source"]["path"],
                "sha256": protocol["datasets"]["sparkov"]["development_source"]["sha256"],
                "source_role": "fraudTrain_development",
            },
        },
        "allowed_operations": ["frozen_data_materialization"],
        "forbidden_operations": [
            "gpu_inventory_query",
            "cuda",
            "model_fit",
            "model_sample",
            "fidelity",
            "coherence",
            "tstr",
            "privacy",
            "external_test_evaluation",
            "sparkov_public_fraudTest_parse",
        ],
    }
    assert validate_materialization_authorization(
        repo_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
        authorization=authorization,
    )["status"] == "PASS"

    raw_calls = []
    monkeypatch.setattr(
        "scripts.materialize_external_sequence_protocol_v1.load_materialization_raw",
        lambda *args, **kwargs: raw_calls.append((args, kwargs)),
    )
    authorization["datasets"]["sparkov"]["sha256"] = "0" * 64
    with pytest.raises(ExternalMaterializationError, match="authorization dataset mismatch"):
        validate_materialization_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            authorization=authorization,
        )
    assert raw_calls == []
