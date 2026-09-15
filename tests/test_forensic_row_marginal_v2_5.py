import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.model_guards_v2_5 import row_guard_statistics
from scripts.forensic_row_marginal_v2_5 import (
    ForensicContractError,
    _static_sampling_path_audit,
    attempt_tree_digest,
    audit_sample_contract,
    independent_row_guard_statistics,
    latest_terminal_attempt,
    write_forensic_artifacts,
)


def fixture_arrays():
    lengths = np.asarray([3, 2, 3, 2], dtype=np.int64)
    valid = np.arange(3)[None, :] < lengths[:, None]
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    x_num = np.zeros((4, 3, 1), dtype=np.float32)
    dt_bin = np.zeros((4, 3), dtype=np.int64)
    x_cat = np.zeros((4, 3, 1), dtype=np.int64)
    x_num[..., 0][valid] = np.asarray(
        [0.0, 0.5, 1.0, 0.2, 0.7, 0.1, 0.4, 0.9, 0.3, 0.8],
        dtype=np.float32,
    )
    dt_bin[valid] = np.asarray([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    x_cat[..., 0][valid] = np.asarray([0, 1, 2, 1, 0, 2, 1, 0, 2, 1])
    return {
        "x_num": x_num,
        "dt_bin": dt_bin,
        "x_cat": x_cat,
        "valid_mask": valid,
        "y_entity": y,
        "lengths": lengths,
    }


def test_independent_forensic_replay_matches_evaluator_statistics():
    reference = fixture_arrays()
    candidate = {
        key: value.copy()
        for key, value in fixture_arrays().items()
    }
    candidate["x_num"][0, 0, 0] += 0.25
    tau = np.asarray([0.5, 2.0])
    sequence = SequenceBatch(
        **reference,
        entity_ids=np.asarray(["a", "b", "c", "d"]),
    )
    synthetic = SyntheticBatch(**candidate)

    independent = independent_row_guard_statistics(
        reference,
        candidate,
        tau=tau,
        receiver_categories=3,
    )
    evaluator = row_guard_statistics(
        sequence,
        synthetic,
        tau=tau,
        receiver_categories=3,
    )

    assert independent == evaluator


def test_forensic_contract_checks_plan_mask_padding_support_and_row_order():
    train = fixture_arrays()
    sample = {
        key: value.copy()
        for key, value in train.items()
    }
    plan = {
        key: sample[key].copy()
        for key in ("y_entity", "lengths", "valid_mask")
    }

    passed = audit_sample_contract(sample, plan=plan, train=train)
    assert passed["status"] == "PASS"
    assert all(passed["checks"].values())

    broken = {
        key: value.copy()
        for key, value in sample.items()
    }
    broken["y_entity"][0] = 1
    failed = audit_sample_contract(broken, plan=plan, train=train)
    assert failed["status"] == "FAIL"
    assert failed["checks"]["labels_equal_plan"] is False


def test_static_audit_finds_direct_runner_path_and_distinct_learned_samplers():
    repository = Path(__file__).resolve().parents[1]

    audit = _static_sampling_path_audit(repository)

    assert audit["audit_pass"] is True
    assert audit[
        "runner_passes_adapter_sample_directly_to_evaluator"
    ] is True
    assert audit["sample_reassignments_between"] == []
    assert audit["distinct_learned_sample_implementation_groups"] == 3
    assert audit["tvae_inherits_ctgan_sample"] is True
    assert all(audit["required_sample_definitions_present"].values())


def test_forensic_latest_attempt_requires_one_terminal_marker(tmp_path):
    seed_root = tmp_path / "seed_1"
    first = seed_root / "attempt_001"
    second = seed_root / "attempt_002"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "COMPLETE.json").write_text('{"status":"COMPLETE"}\n')
    (second / "INVALID.json").write_text('{"status":"INVALID"}\n')

    attempt, status, marker = latest_terminal_attempt(seed_root)

    assert attempt == second
    assert status == "INVALID"
    assert marker == second / "INVALID.json"

    (second / "FAILED.json").write_text('{"status":"FAILED"}\n')
    with pytest.raises(ForensicContractError, match="singly terminal"):
        latest_terminal_attempt(seed_root)


def test_attempt_tree_digest_matches_documented_line_algorithm(tmp_path):
    full_root = tmp_path / "full"
    first = full_root / "generator_a/seed_1/attempt_002"
    second = full_root / "generator_b/seed_1/attempt_002"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "COMPLETE.json").write_text(
        '{"status":"COMPLETE"}\n',
        encoding="utf-8",
    )
    (first / "metrics.json").write_text("{}\n", encoding="utf-8")
    (second / "INVALID.json").write_text(
        '{"status":"INVALID"}\n',
        encoding="utf-8",
    )
    (second / "attempt_003.txt").write_text(
        "not an attempt_003 path component\n",
        encoding="utf-8",
    )

    complete = attempt_tree_digest(full_root, attempt=2)
    terminals = attempt_tree_digest(
        full_root,
        attempt=2,
        terminal_markers_only=True,
    )

    assert complete["files"] == 4
    assert terminals["files"] == 2
    assert terminals["paths"] == [
        "generator_a/seed_1/attempt_002/COMPLETE.json",
        "generator_b/seed_1/attempt_002/INVALID.json",
    ]
    assert complete["sha256"] != terminals["sha256"]


def test_forensic_writer_refuses_runtime_artifact_and_data_roots(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    report = {
        "component_rows": [
            {"generator": "fixture", "component": "amount_ks"}
        ]
    }
    output_json = repository / "docs/report.json"

    with pytest.raises(ForensicContractError, match="runtime"):
        write_forensic_artifacts(
            report,
            csv_path=repository / "artifacts/report.csv",
            json_path=output_json,
            repository_root=repository,
        )
    with pytest.raises(ForensicContractError, match="runtime"):
        write_forensic_artifacts(
            report,
            csv_path=repository / "docs/report.csv",
            json_path=repository / "data/report.json",
            repository_root=repository,
        )
    assert not output_json.exists()


def test_forensic_writer_outputs_strict_json_and_csv(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    csv_path = repository / "docs/report.csv"
    json_path = repository / "docs/report.json"
    report = {
        "component_rows": [
            {
                "generator": "fixture",
                "seed": 1,
                "component": "amount_ks",
            }
        ]
    }

    written = write_forensic_artifacts(
        report,
        csv_path=csv_path,
        json_path=json_path,
        repository_root=repository,
    )

    assert json.loads(json_path.read_text()) == report
    assert csv_path.read_text().splitlines()[0] == (
        "generator,seed,component"
    )
    assert b"\r" not in csv_path.read_bytes()
    assert written["csv"]["sha256"]
    assert written["json"]["sha256"]
