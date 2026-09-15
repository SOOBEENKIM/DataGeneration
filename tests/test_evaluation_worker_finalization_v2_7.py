from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import threading
import time

import pytest

from experiments.evaluation_only_runner_v2_7 import (
    EvaluationOnlyArtifactStore,
    EvaluationOnlyContractError,
    build_evaluation_execution_plan,
    build_finalization_recovery_inventory,
    finalize_evaluation_worker,
    recover_evaluation_worker_finalization,
    run_evaluation_child_until_terminal,
    validate_finalization_only_authorization,
)
from scripts.run_evaluation_only_v2_7 import parse_args

REPOSITORY = Path(__file__).resolve().parents[1]
RUNNER_CONFIG = (
    REPOSITORY
    / "configs/benchmark_v2/evaluation_only_v2_7.yaml"
)

def _complete_before_delayed_cleanup(payload, result_queue):
    attempt = Path(payload["attempt_path"])
    (attempt / "COMPLETE.json").write_text(
        json.dumps({"status": "COMPLETE"}) + "\n",
        encoding="utf-8",
    )
    result_queue.put({"status": "COMPLETE"})
    threading.Thread(
        target=time.sleep,
        args=(float(payload["cleanup_delay_seconds"]),),
        daemon=False,
    ).start()


def _exit_without_terminal(payload, result_queue):
    del payload, result_queue
    os._exit(7)


def test_worker_finalizes_before_completed_child_cleanup(tmp_path):
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()
    worker_complete = tmp_path / "WORKER_COMPLETE.json"
    callback_times = []
    started = time.monotonic()

    result = run_evaluation_child_until_terminal(
        child_target=_complete_before_delayed_cleanup,
        payload={
            "attempt_path": str(attempt),
            "cleanup_delay_seconds": 2.0,
        },
        attempt_path=attempt,
        max_wall_seconds=5.0,
        termination_grace_seconds=0.2,
        target_complete_cleanup_grace_seconds=0.05,
        on_target_complete=lambda value: (
            callback_times.append(time.monotonic()),
            worker_complete.write_text(
                json.dumps(
                    {
                        "status": value["status"],
                        "target_completed": value[
                            "target_completed"
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            ),
        ),
    )

    marker = json.loads(worker_complete.read_text(encoding="utf-8"))
    assert marker == {
        "status": "COMPLETE",
        "target_completed": True,
    }
    assert result["status"] == "COMPLETE"
    assert result["cleanup_forced"] is True
    assert callback_times[0] - started < 1.0
    assert time.monotonic() - started < 1.0


def test_abnormal_child_exit_records_worker_failed(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    store = EvaluationOnlyArtifactStore(repository)
    store.claim_model(
        model_id="ctgan_separate_class",
        provenance_sha256="a" * 64,
    )
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()

    result = run_evaluation_child_until_terminal(
        child_target=_exit_without_terminal,
        payload={},
        attempt_path=attempt,
        max_wall_seconds=2.0,
        termination_grace_seconds=0.2,
    )
    marker = finalize_evaluation_worker(
        store=store,
        model_id="ctgan_separate_class",
        provenance_sha256="a" * 64,
        terminals=[
            {
                "candidate_id": (
                    "ctgan_v27_c01_amount_quantile_inverse"
                ),
                "status": result["status"],
                "failure_class": result["failure_class"],
                "attempt": "attempt_001",
            }
        ],
        expected_candidate_ids=(
            "ctgan_v27_c00_frozen_standard",
            "ctgan_v27_c01_amount_quantile_inverse",
            "ctgan_v27_c02_categorical_logit",
        ),
        finalization_kind="normal",
    )

    value = json.loads(marker.read_text(encoding="utf-8"))
    assert result["failure_class"] == "child_exit"
    assert marker.name == "WORKER_FAILED.json"
    assert value["status"] == "FAILED"
    assert value["terminals"][0]["failure_class"] == "child_exit"


def test_missing_worker_marker_recovery_requires_separate_authorization(
    tmp_path,
):
    repository = tmp_path / "repository"
    repository.mkdir()
    store = EvaluationOnlyArtifactStore(repository)
    store.claim_model(
        model_id="ctgan_separate_class",
        provenance_sha256="a" * 64,
    )
    candidate_ids = (
        "ctgan_v27_c00_frozen_standard",
        "ctgan_v27_c01_amount_quantile_inverse",
        "ctgan_v27_c02_categorical_logit",
    )
    for candidate_id in candidate_ids:
        attempt = store.create_evaluation_attempt(
            model_id="ctgan_separate_class",
            candidate_id=candidate_id,
            seed=2601,
        )
        store.write_json(
            attempt / "manifest.json",
            {
                "model_id": "ctgan_separate_class",
                "candidate_id": candidate_id,
                "source_commit": "d" * 40,
                "authorization_path": (
                    "artifacts/benchmark_v2_7/authorizations/"
                    "execution.json"
                ),
                "authorization_sha256": "e" * 64,
            },
        )
        store.write_json(
            attempt / "COMPLETE.json",
            {"status": "COMPLETE"},
        )
    inventory = build_finalization_recovery_inventory(
        runtime_root=store.root,
        model_id="ctgan_separate_class",
        expected_candidate_ids=candidate_ids,
    )
    before = inventory["preservation_tree_sha256"]
    plan = build_evaluation_execution_plan(RUNNER_CONFIG)

    with pytest.raises(
        EvaluationOnlyContractError,
        match="finalization-only authorization",
    ):
        validate_finalization_only_authorization(
            plan=plan,
            authorization={},
            inventory=inventory,
            source_commit="f" * 40,
            relevant_source_sha256="1" * 64,
            original_authorization_sha256="e" * 64,
        )

    after = build_finalization_recovery_inventory(
        runtime_root=store.root,
        model_id="ctgan_separate_class",
        expected_candidate_ids=candidate_ids,
    )
    assert after["preservation_tree_sha256"] == before
    assert not (
        store.root
        / "workers/ctgan_separate_class/WORKER_COMPLETE.json"
    ).exists()


def test_authorized_recovery_reuses_complete_candidates_without_replay(
    tmp_path,
):
    repository = tmp_path / "repository"
    repository.mkdir()
    store = EvaluationOnlyArtifactStore(repository)
    store.claim_model(
        model_id="ctgan_separate_class",
        provenance_sha256="a" * 64,
    )
    authorization_root = (
        repository
        / "artifacts/benchmark_v2_7/authorizations"
    )
    authorization_root.mkdir(parents=True)
    original_authorization = authorization_root / "execution.json"
    original_authorization.write_text(
        '{"status":"AUTHORIZED"}\n',
        encoding="utf-8",
    )
    original_hash = hashlib.sha256(
        original_authorization.read_bytes()
    ).hexdigest()
    candidate_ids = (
        "ctgan_v27_c00_frozen_standard",
        "ctgan_v27_c01_amount_quantile_inverse",
        "ctgan_v27_c02_categorical_logit",
    )
    for candidate_id in candidate_ids:
        attempt = store.create_evaluation_attempt(
            model_id="ctgan_separate_class",
            candidate_id=candidate_id,
            seed=2601,
        )
        store.write_json(
            attempt / "manifest.json",
            {
                "model_id": "ctgan_separate_class",
                "candidate_id": candidate_id,
                "source_commit": "d" * 40,
                "authorization_path": str(
                    original_authorization.relative_to(repository)
                ),
                "authorization_sha256": original_hash,
            },
        )
        store.write_json(
            attempt / "COMPLETE.json",
            {"status": "COMPLETE"},
        )
    inventory = build_finalization_recovery_inventory(
        runtime_root=store.root,
        model_id="ctgan_separate_class",
        expected_candidate_ids=candidate_ids,
    )
    attempt_hashes = {
        row["candidate_id"]: row["attempt_tree_sha256"]
        for row in inventory["candidate_attempts"]
    }
    plan = build_evaluation_execution_plan(RUNNER_CONFIG)
    actions = {
        "write_worker_terminal_marker": True,
        "candidate_execution": False,
        "candidate_replay": False,
        "sampling": False,
        "evaluation": False,
        "training": False,
        "optimizer_updates": False,
        "selection": False,
        "fresh_test": False,
        "tstr": False,
        "privacy": False,
        "five_seed_full_run": False,
    }
    recovery_authorization = authorization_root / "finalization.json"
    recovery_authorization.write_text(
        json.dumps(
            {
                "schema_version": (
                    "benchmark-v2.7-evaluation-finalization-only-"
                    "authorization-v1"
                ),
                "status": "AUTHORIZED",
                "finalizer_source_commit": "f" * 40,
                "finalizer_relevant_source_sha256": "1" * 64,
                "runner_config_sha256": plan.runner_config_sha256,
                "candidate_config_sha256": (
                    plan.candidate_plan.config_sha256
                ),
                "model_id": "ctgan_separate_class",
                "candidate_source_commit": "d" * 40,
                "original_execution_authorization_sha256": (
                    original_hash
                ),
                "ownership_lock_sha256": inventory[
                    "ownership_lock_sha256"
                ],
                "preservation_tree_sha256": inventory[
                    "preservation_tree_sha256"
                ],
                "inventory_sha256": inventory["inventory_sha256"],
                "candidate_ids": list(candidate_ids),
                "candidate_attempts": {
                    row["candidate_id"]: {
                        "attempt_path": row["attempt_path"],
                        "attempt_tree_sha256": row[
                            "attempt_tree_sha256"
                        ],
                        "manifest_sha256": row["manifest_sha256"],
                        "complete_sha256": row["complete_sha256"],
                    }
                    for row in inventory["candidate_attempts"]
                },
                "actions": actions,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = recover_evaluation_worker_finalization(
        plan=plan,
        store=store,
        authorization_path=recovery_authorization,
        model_id="ctgan_separate_class",
        source_commit="f" * 40,
        relevant_source_sha256="1" * 64,
    )

    assert result["status"] == "COMPLETE"
    assert result["candidate_execution_calls"] == 0
    assert result["sampling_calls"] == 0
    assert result["evaluation_calls"] == 0
    assert not list(store.root.rglob("attempt_002"))
    assert {
        row["candidate_id"]: row["attempt_tree_sha256"]
        for row in build_finalization_recovery_inventory(
            runtime_root=store.root,
            model_id="ctgan_separate_class",
            expected_candidate_ids=candidate_ids,
            allow_existing_worker_complete=True,
        )["candidate_attempts"]
    } == attempt_hashes


def test_finalization_only_cli_requires_separate_authorization():
    parsed = parse_args(
        [
            "--mode",
            "finalize-only",
            "--authorization",
            "finalization_authorization.json",
            "--model",
            "ctgan_separate_class",
        ]
    )
    assert parsed.mode == "finalize-only"
    assert parsed.device is None
    with pytest.raises(SystemExit):
        parse_args(["--mode", "finalize-only"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "finalize-only",
                "--authorization",
                "execution_authorization.json",
                "--model",
                "ctgan_separate_class",
                "--device",
                "cuda:0",
            ]
        )
