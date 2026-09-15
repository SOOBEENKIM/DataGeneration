from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import threading
import time

import pytest

from experiments.candidate_runner_v2_6 import run_bounded_process
from experiments.single_factor_runner_v2_6 import (
    SINGLE_FACTOR_MODEL_IDS,
    SingleFactorRunnerContractError,
    _apply_evaluation_only_intervention,
    _backend_factory,
    _single_factor_adapter_spec,
    build_single_factor_execution_plan,
    dry_run_single_factor_worker,
    finalize_single_factor_worker_attempt,
    record_single_factor_worker_failure,
    recover_single_factor_worker_finalization,
    single_factor_model_tree_sha256,
    single_factor_relevant_source_sha256,
    validate_single_factor_authorization,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = (
    REPOSITORY
    / "configs/benchmark_v2/selection_v2_6_single_factor_amendment.yaml"
)
BASE_CONFIG = REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
DEVELOPMENT = (
    REPOSITORY / "configs/benchmark_v2/development_data_v2_6.yaml"
)


def _target_that_finishes_before_process_cleanup(delay_seconds):
    threading.Thread(
        target=time.sleep,
        args=(delay_seconds,),
        daemon=False,
    ).start()


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _complete_candidate_fixture(
    *,
    repository,
    runtime_root,
    operation,
    source_commit,
    source_hash,
    authorization_hash,
):
    attempt = (
        runtime_root
        / "candidates"
        / operation.model_id
        / operation.candidate_id
        / "seed_2601"
        / "attempt_001"
    )
    attempt.mkdir(parents=True)
    if operation.execution_kind == "reuse_frozen_control":
        reference = attempt / "reference_manifest.json"
        _write_json(
            reference,
            {
                "status": "COMPLETE",
                "source_commit": source_commit,
                "relevant_source_sha256": source_hash,
                "single_factor_config_sha256": (
                    "707ecdbd37bc55f4a3b9ece603b70d26c8274015aa9cb97d5d376aab3d9ecb7c"
                ),
                "authorization_sha256": authorization_hash,
                "model_id": operation.model_id,
                "candidate_id": operation.candidate_id,
                "execution_kind": "reuse_frozen_control",
                "training_calls": 0,
                "sampling_calls": 0,
                "test_split_read": False,
                "validation_selection_executed": False,
            },
        )
        complete = attempt / "COMPLETE.json"
        _write_json(
            complete,
            {
                "status": "COMPLETE",
                "reference_manifest_sha256": _sha256(reference),
            },
        )
        return {
            "candidate_id": operation.candidate_id,
            "status": "COMPLETE",
            "operation": "reuse_frozen_control",
            "candidate_attempt_path": str(
                attempt.relative_to(repository)
            ),
            "reference_manifest_sha256": _sha256(reference),
            "complete_sha256": _sha256(complete),
        }
    sample = attempt / "validation_sample.npz"
    sample.write_bytes(b"validation-only-fixture")
    diagnostics = attempt / "diagnostics.json"
    _write_json(diagnostics, {"diagnostic": "preserved"})
    result = attempt / "candidate_result.json"
    _write_json(
        result,
        {
            "status": "COMPLETE",
            "source_commit": source_commit,
            "relevant_source_sha256": source_hash,
            "single_factor_config_sha256": (
                "707ecdbd37bc55f4a3b9ece603b70d26c8274015aa9cb97d5d376aab3d9ecb7c"
            ),
            "authorization_sha256": authorization_hash,
            "model_id": operation.model_id,
            "candidate_id": operation.candidate_id,
            "test_split_read": False,
            "validation_selection_executed": False,
            "validation_sample_sha256": _sha256(sample),
        },
    )
    complete = attempt / "COMPLETE.json"
    _write_json(
        complete,
        {
            "status": "COMPLETE",
            "candidate_result_sha256": _sha256(result),
            "diagnostics_sha256": _sha256(diagnostics),
            "validation_sample_sha256": _sha256(sample),
        },
    )
    if operation.execution_kind == "train_new_trajectory":
        operation_attempt = (
            runtime_root
            / "trajectories"
            / operation.model_id
            / str(operation.trajectory_id)
            / "seed_2601"
            / "attempt_001"
        )
    else:
        operation_attempt = (
            runtime_root
            / "evaluations"
            / operation.model_id
            / operation.candidate_id
            / "seed_2601"
            / "attempt_001"
        )
    operation_attempt.mkdir(parents=True)
    _write_json(
        operation_attempt / "CANDIDATE_ARTIFACT.json",
        {
            "candidate_attempt_path": str(
                attempt.relative_to(repository)
            ),
            "candidate_result_sha256": _sha256(result),
            "complete_sha256": _sha256(complete),
            "validation_sample_sha256": _sha256(sample),
        },
    )
    return {
        "candidate_id": operation.candidate_id,
        "status": "COMPLETE",
        "operation": operation.execution_kind,
        "attempt_path": str(operation_attempt.relative_to(repository)),
    }


def _authorization(plan, *, source_commit: str) -> dict:
    return {
        "schema_version": (
            "benchmark-v2.6-single-factor-execution-authorization-v1"
        ),
        "status": "AUTHORIZED",
        "approval_scope": (
            "validation-only single-factor candidate execution: exactly "
            "three frozen controls, three checkpoint-only evaluations, "
            "and three new trajectories; test/fresh-test/selection/"
            "five-seed/full execution remain forbidden"
        ),
        "source_commit": source_commit,
        "relevant_source_sha256": single_factor_relevant_source_sha256(
            REPOSITORY
        ),
        "single_factor_config_sha256": plan.config_sha256,
        "base_selection_config_sha256": (
            "0884a0144f74f6317ae9e636c0cf5657dedac21ff12cdf545a6b6e5e29be4c4d"
        ),
        "development_manifest_sha256": (
            "31ddf45dcf7b4f982ed990ec94281c3e6bad0454ec3182e2434e42358006ade5"
        ),
        "train_file_sha256": (
            "c67a6fce4593317e11e1b6f36397bfb38f9607916bf8f5b32329d8fa99b700a8"
        ),
        "train_content_sha256": (
            "0c03f179930cd4d89ccc8a8b66283e620313e16d15fd53f0674c133bcb80c09d"
        ),
        "validation_file_sha256": (
            "68c15c05c55742bfce0f22560ffe8378d032a7a9fd9a1072e80ec8da5340fba5"
        ),
        "validation_content_sha256": (
            "aa1576b4ccae6a2ca30d0472f0782e1231ecc61ab3d224590a772dc3448b9e66"
        ),
        "sampling_plan_sha256": (
            "862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27"
        ),
        "v2_5_config_sha256": (
            "81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3"
        ),
        "v2_5_final_complete_sha256": (
            "e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a"
        ),
        "v2_5_frozen_manifest_sha256": (
            "b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05"
        ),
        "prior_v2_6_selection_complete_sha256": (
            "3cb9f91a00ea4b3bcb772527e2ff4f49b1a47c645618ce79995847f07337407d"
        ),
        "prior_v2_6_candidate_tree_sha256": (
            "fixture-tree-digest"
        ),
        "runtime_root": (
            "artifacts/benchmark_v2_6/selection_single_factor"
        ),
        "models": list(SINGLE_FACTOR_MODEL_IDS),
        "candidate_ids": [
            operation.candidate_id
            for operation in plan.operations
        ],
        "new_trajectory_ids": [
            operation.trajectory_id
            for operation in plan.operations
            if operation.execution_kind == "train_new_trajectory"
        ],
        "counts": {
            "candidates": 9,
            "reused_controls": 3,
            "evaluation_only": 3,
            "new_trajectories": 3,
        },
        "frozen_control_hashes": {
            operation.model_id: dict(operation.frozen_control_hashes)
            for operation in plan.operations
            if operation.execution_kind == "reuse_frozen_control"
        },
        "authorization": {
            "candidate_execution": True,
            "gpu_training_for_three_new_trajectories": True,
            "gpu_sampling_for_three_evaluation_candidates": True,
            "validation_selection": False,
            "test_split_access": False,
            "fresh_test": False,
            "five_seed_full_run": False,
            "data_generation": False,
            "dgp_execution": False,
        },
    }


def test_execution_plan_is_exactly_nine_single_factor_candidates():
    plan = build_single_factor_execution_plan(
        config_path=CONFIG,
        base_selection_config_path=BASE_CONFIG,
    )

    assert plan.model_ids == SINGLE_FACTOR_MODEL_IDS
    assert len(plan.operations) == 9
    assert sum(
        operation.execution_kind == "reuse_frozen_control"
        for operation in plan.operations
    ) == 3
    assert sum(
        operation.execution_kind == "evaluate_existing_checkpoint"
        for operation in plan.operations
    ) == 3
    assert sum(
        operation.execution_kind == "train_new_trajectory"
        for operation in plan.operations
    ) == 3
    assert len(
        {
            operation.trajectory_id
            for operation in plan.operations
            if operation.trajectory_id is not None
        }
    ) == 3
    for model_id in SINGLE_FACTOR_MODEL_IDS:
        model = [
            operation
            for operation in plan.operations
            if operation.model_id == model_id
        ]
        assert len(model) == 3
        assert {operation.execution_kind for operation in model} == {
            "reuse_frozen_control",
            "evaluate_existing_checkpoint",
            "train_new_trajectory",
        }


def test_controls_and_evaluation_only_operations_never_retrain():
    plan = build_single_factor_execution_plan(
        config_path=CONFIG,
        base_selection_config_path=BASE_CONFIG,
    )

    for operation in plan.operations:
        if operation.execution_kind == "reuse_frozen_control":
            assert operation.training_required is False
            assert operation.sampling_required is False
        elif operation.execution_kind == "evaluate_existing_checkpoint":
            assert operation.training_required is False
            assert operation.sampling_required is True
            assert operation.checkpoint_path is not None
            assert operation.checkpoint_sha256 is not None
        else:
            assert operation.training_required is True
            assert operation.sampling_required is True
            assert operation.checkpoint_path is None


def test_authorization_and_provenance_mismatch_fail_closed():
    plan = build_single_factor_execution_plan(
        config_path=CONFIG,
        base_selection_config_path=BASE_CONFIG,
    )
    source_commit = "a" * 40
    authorization = _authorization(plan, source_commit=source_commit)

    validate_single_factor_authorization(
        authorization,
        plan=plan,
        source_commit=source_commit,
        relevant_source_sha256=single_factor_relevant_source_sha256(
            REPOSITORY
        ),
        prior_candidate_tree_sha256="fixture-tree-digest",
    )

    for key, bad_value in (
        ("source_commit", "b" * 40),
        ("single_factor_config_sha256", "0" * 64),
        ("sampling_plan_sha256", "1" * 64),
        ("prior_v2_6_candidate_tree_sha256", "2" * 64),
    ):
        tampered = copy.deepcopy(authorization)
        tampered[key] = bad_value
        with pytest.raises(SingleFactorRunnerContractError):
            validate_single_factor_authorization(
                tampered,
                plan=plan,
                source_commit=source_commit,
                relevant_source_sha256=(
                    single_factor_relevant_source_sha256(REPOSITORY)
                ),
                prior_candidate_tree_sha256="fixture-tree-digest",
            )


def test_dry_run_is_validation_only_and_has_zero_execution_calls():
    plan = build_single_factor_execution_plan(
        config_path=CONFIG,
        base_selection_config_path=BASE_CONFIG,
    )
    source_commit = "a" * 40
    authorization = _authorization(plan, source_commit=source_commit)
    counters = {
        "test_split_reads": 0,
        "fresh_test_reads": 0,
        "gpu_queries": 0,
        "cuda_calls": 0,
        "dgp_calls": 0,
        "fit_calls": 0,
        "sample_calls": 0,
        "selection_calls": 0,
    }

    result = dry_run_single_factor_worker(
        repository_root=REPOSITORY,
        plan=plan,
        authorization=authorization,
        source_commit=source_commit,
        prior_candidate_tree_sha256="fixture-tree-digest",
        model_id="ctgan_separate_class",
        counters=counters,
        verify_files=False,
    )

    assert result["status"] == "DRY_RUN_PASS"
    assert result["operation_count"] == 3
    assert result["new_trajectory_count"] == 1
    assert result["evaluation_only_count"] == 1
    assert result["reused_control_count"] == 1
    assert result["test_split_accesses"] == 0
    assert result["fresh_test_accesses"] == 0
    assert result["execution_calls"] == counters
    assert all(value == 0 for value in counters.values())


def test_test_and_fresh_test_paths_are_not_part_of_the_plan():
    plan = build_single_factor_execution_plan(
        config_path=CONFIG,
        base_selection_config_path=BASE_CONFIG,
    )

    serialized = repr(plan).lower()
    assert "test.npz" not in serialized
    assert "fresh_test" not in serialized
    assert "fresh-test" not in serialized
    assert plan.development_manifest_path == DEVELOPMENT.resolve()


def test_target_completion_callback_runs_before_delayed_child_cleanup(
    tmp_path,
):
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()
    worker_terminal = tmp_path / "WORKER_COMPLETE.json"
    callback_times = []
    started = time.monotonic()

    result = run_bounded_process(
        target=_target_that_finishes_before_process_cleanup,
        attempt_path=attempt,
        max_wall_seconds=5.0,
        termination_grace_seconds=0.2,
        args=(2.0,),
        target_complete_cleanup_grace_seconds=0.05,
        on_target_complete=lambda value: (
            callback_times.append(time.monotonic()),
            worker_terminal.write_text(
                str(value["status"]),
                encoding="utf-8",
            ),
        ),
    )

    assert result["status"] == "COMPLETE"
    assert result["cleanup_forced"] is True
    assert worker_terminal.read_text(encoding="utf-8") == "COMPLETE"
    assert callback_times[0] - started < 1.0
    assert time.monotonic() - started < 1.0


@pytest.mark.parametrize("model_id", SINGLE_FACTOR_MODEL_IDS)
def test_complete_candidate_artifacts_finalize_worker_append_only(
    tmp_path,
    model_id,
):
    plan = build_single_factor_execution_plan(
        config_path=CONFIG,
        base_selection_config_path=BASE_CONFIG,
    )
    repository = tmp_path / "repository"
    runtime_root = (
        repository
        / "artifacts/benchmark_v2_6/selection_single_factor"
    )
    worker_attempt = (
        runtime_root
        / "workers"
        / model_id
        / "attempt_001"
    )
    worker_attempt.mkdir(parents=True)
    source_commit = "a" * 40
    source_hash = "b" * 64
    authorization_hash = "c" * 64
    results = [
        _complete_candidate_fixture(
            repository=repository,
            runtime_root=runtime_root,
            operation=operation,
            source_commit=source_commit,
            source_hash=source_hash,
            authorization_hash=authorization_hash,
        )
        for operation in plan.operations
        if operation.model_id == model_id
    ]

    terminal = finalize_single_factor_worker_attempt(
        repository_root=repository,
        runtime_root=runtime_root,
        worker_attempt=worker_attempt,
        plan=plan,
        model_id=model_id,
        results=results,
        source_commit=source_commit,
        source_hash=source_hash,
        authorization_hash=authorization_hash,
        finalization_kind="normal",
    )

    assert terminal["status"] == "COMPLETE"
    assert terminal["candidate_count"] == 3
    assert set(terminal["candidate_artifacts"]) == {
        operation.candidate_id
        for operation in plan.operations
        if operation.model_id == model_id
    }
    assert (worker_attempt / "WORKER_COMPLETE.json").is_file()
    assert not (worker_attempt / "WORKER_FAILED.json").exists()
    with pytest.raises(
        SingleFactorRunnerContractError,
        match="already terminal",
    ):
        finalize_single_factor_worker_attempt(
            repository_root=repository,
            runtime_root=runtime_root,
            worker_attempt=worker_attempt,
            plan=plan,
            model_id=model_id,
            results=results,
            source_commit=source_commit,
            source_hash=source_hash,
            authorization_hash=authorization_hash,
            finalization_kind="normal",
        )


def test_abnormal_candidate_child_records_worker_failed(tmp_path):
    worker_attempt = tmp_path / "workers/attempt_001"
    worker_attempt.mkdir(parents=True)

    terminal = record_single_factor_worker_failure(
        worker_attempt=worker_attempt,
        model_id="ctgan_separate_class",
        results=[
            {
                "candidate_id": "ctgan_sf_c01_shared_transformer",
                "status": "FAILED",
                "failure_class": "infrastructure_or_code",
            }
        ],
        failure_class="infrastructure_or_code",
        exception="child exited 17 without terminal metadata",
        source_commit="a" * 40,
        source_hash="b" * 64,
        config_hash=(
            "707ecdbd37bc55f4a3b9ece603b70d26c8274015aa9cb97d5d376aab3d9ecb7c"
        ),
        authorization_hash="c" * 64,
    )

    assert terminal["status"] == "FAILED"
    assert terminal["failure_class"] == "infrastructure_or_code"
    assert terminal["exception"] == (
        "child exited 17 without terminal metadata"
    )
    assert (worker_attempt / "WORKER_FAILED.json").is_file()
    assert not (worker_attempt / "WORKER_COMPLETE.json").exists()


def test_finalization_only_recovery_reuses_complete_candidates(
    tmp_path,
):
    plan = build_single_factor_execution_plan(
        config_path=CONFIG,
        base_selection_config_path=BASE_CONFIG,
    )
    repository = tmp_path / "repository"
    runtime_root = (
        repository
        / "artifacts/benchmark_v2_6/selection_single_factor"
    )
    workers = (
        runtime_root / "workers/ctgan_separate_class"
    )
    prior_worker = workers / "attempt_001"
    prior_worker.mkdir(parents=True)
    _write_json(
        workers / "ownership.lock",
        {"source_commit": "a" * 40},
    )
    _write_json(
        prior_worker / "manifest.json",
        {
            "status": "RUNNING",
            "model_id": "ctgan_separate_class",
        },
    )
    candidate_source_commit = "a" * 40
    candidate_source_hash = "b" * 64
    execution_authorization_hash = "c" * 64
    results = [
        _complete_candidate_fixture(
            repository=repository,
            runtime_root=runtime_root,
            operation=operation,
            source_commit=candidate_source_commit,
            source_hash=candidate_source_hash,
            authorization_hash=execution_authorization_hash,
        )
        for operation in plan.operations
        if operation.model_id == "ctgan_separate_class"
    ]
    preservation_hash, preservation_files, preservation_bytes = (
        single_factor_model_tree_sha256(
            runtime_root=runtime_root,
            model_id="ctgan_separate_class",
        )
    )
    recovery_source_commit = "d" * 40
    recovery_source_hash = "e" * 64
    recovery_authorization_hash = "f" * 64
    authorization = {
        "schema_version": (
            "benchmark-v2.6-single-factor-finalization-recovery-authorization-v1"
        ),
        "status": "AUTHORIZED",
        "authorized_action": "FINALIZATION_ONLY",
        "model_id": "ctgan_separate_class",
        "recovery_source_commit": recovery_source_commit,
        "recovery_relevant_source_sha256": recovery_source_hash,
        "candidate_source_commit": candidate_source_commit,
        "candidate_relevant_source_sha256": candidate_source_hash,
        "single_factor_config_sha256": plan.config_sha256,
        "execution_authorization_sha256": (
            execution_authorization_hash
        ),
        "prior_worker_attempt_path": str(
            prior_worker.relative_to(repository)
        ),
        "prior_model_tree_sha256": preservation_hash,
        "prior_model_tree_files": preservation_files,
        "prior_model_tree_bytes": preservation_bytes,
        "results": results,
        "authorization": {
            "terminal_marker_write": True,
            "gpu_query": False,
            "training": False,
            "sampling": False,
            "selection": False,
            "test_split_access": False,
            "fresh_test": False,
            "full_run": False,
        },
    }

    terminal = recover_single_factor_worker_finalization(
        repository_root=repository,
        runtime_root=runtime_root,
        plan=plan,
        authorization=authorization,
        recovery_authorization_hash=recovery_authorization_hash,
        current_source_commit=recovery_source_commit,
        current_source_hash=recovery_source_hash,
    )

    recovery_attempt = workers / "attempt_002"
    assert terminal["status"] == "COMPLETE"
    assert terminal["finalization_kind"] == "recovery"
    assert terminal["recovery_authorization_sha256"] == (
        recovery_authorization_hash
    )
    assert (recovery_attempt / "recovery_manifest.json").is_file()
    assert (recovery_attempt / "WORKER_COMPLETE.json").is_file()
    assert not (prior_worker / "WORKER_COMPLETE.json").exists()
    assert terminal["training_calls_during_finalization"] == 0
    assert terminal["sampling_calls_during_finalization"] == 0


def test_single_factor_specs_bind_the_exact_preregistered_backend_change():
    plan = build_single_factor_execution_plan(
        config_path=CONFIG,
        base_selection_config_path=BASE_CONFIG,
    )
    operations = {
        operation.candidate_id: operation
        for operation in plan.operations
    }

    shared = operations["ctgan_sf_c01_shared_transformer"]
    assert _backend_factory(shared).__name__ == (
        "ConditionalCTGANSharedTransformerV26"
    )
    assert (
        _single_factor_adapter_spec(shared).sampling_rule[
            "categorical_temperature"
        ]
        == 0.2
    )

    weighted = operations["tvae_sf_c02_channel_weight_only"]
    assert _backend_factory(weighted) is None
    assert _single_factor_adapter_spec(weighted).loss_weights == {
        "amount": 2.0,
        "gap": 2.0,
        "receiver": 1.0,
    }
    assert (
        "categorical_temperature"
        not in _single_factor_adapter_spec(weighted).sampling_rule
    )

    epsilon = operations["cof_sf_c01_noise_prediction"]
    assert _backend_factory(epsilon).__name__ == (
        "CoFNoisePredictionCandidateV26"
    )
    assert (
        _single_factor_adapter_spec(epsilon).sampling_rule["amount"]
        == "epsilon_reverse_diffusion_50_steps"
    )


class _FakeCTGANSynthesizer:
    def __init__(self):
        self.temperature = None

    def set_categorical_temperature(self, value):
        self.temperature = value


class _FakeTVAESynthesizer:
    def __init__(self):
        self.contract = None

    def set_candidate_contract(self, **kwargs):
        self.contract = kwargs


class _FakeBackend:
    def __init__(self, synthesizer):
        self.models = {
            0: synthesizer(),
            1: synthesizer(),
        }


def test_evaluation_only_interventions_modify_sampling_contract_only():
    plan = build_single_factor_execution_plan(
        config_path=CONFIG,
        base_selection_config_path=BASE_CONFIG,
    )
    operations = {
        operation.candidate_id: operation
        for operation in plan.operations
    }
    ctgan = _FakeBackend(_FakeCTGANSynthesizer)
    _apply_evaluation_only_intervention(
        operation=operations["ctgan_sf_c02_temperature_only"],
        backend=ctgan,
    )
    assert {
        model.temperature for model in ctgan.models.values()
    } == {0.5}

    tvae = _FakeBackend(_FakeTVAESynthesizer)
    _apply_evaluation_only_intervention(
        operation=operations[
            "tvae_sf_c01_categorical_decode_only"
        ],
        backend=tvae,
    )
    assert all(
        model.contract
        == {
            "channel_weights": {
                "amount": 1.0,
                "gap": 1.0,
                "receiver": 1.0,
            },
            "latent_scale": 1.0,
            "categorical_temperature": 0.75,
        }
        for model in tvae.models.values()
    )
