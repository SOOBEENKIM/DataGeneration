import json
from pathlib import Path
from dataclasses import replace
import time

import pytest
import torch

from experiments.cof_seqgen_v3_execution_runner import (
    RunnerChildSpec,
    V3AttemptStore,
    build_v3_candidate_adapter,
    build_v3_checkpoint_provenance,
    build_v3_aggregate_input_manifest,
    build_v3_execution_plan,
    execute_candidate_stages,
    dry_run_v3_execution,
    read_only_tree_inventory,
    run_runner_owned_child,
    validate_v3_execution_authorization,
    validate_v3_checkpoint_provenance,
    validate_v3_runtime_model,
    v3_execution_plan_report,
)
from eval.cof_seqgen_v3_contract import (
    V3ContractError,
    canonical_sha256,
    sha256_file,
)
from generators.cof_seqgen_v3_candidate import (
    CoFSeqGenV3CandidateAdapter,
)
from generators.sampling_plan import SamplingPlan
from models.cof_seqgen_v3 import (
    CoFSeqDenoiserV3,
    CoFSeqGenV3,
    JointStateCodec,
)
from scripts.run_cof_seqgen_v3_validation import parse_args


REPOSITORY = Path(__file__).resolve().parents[1]
RUNNER_CONFIG = (
    REPOSITORY
    / "configs/benchmark_v2/cof_seqgen_v3_validation_runner.yaml"
)


def _fixture_child_never_reports_setup(payload, event_queue):
    del event_queue
    time.sleep(float(payload["sleep_seconds"]))


def _fixture_child_setup_then_hangs(payload, event_queue):
    event_queue.put({"kind": "setup_complete"})
    time.sleep(float(payload["sleep_seconds"]))


def _fixture_child_raises(payload, event_queue):
    event_queue.put({"kind": "setup_complete"})
    raise RuntimeError(str(payload["message"]))


def _fixture_child_interrupts(payload, event_queue):
    event_queue.put({"kind": "setup_complete"})
    raise KeyboardInterrupt(str(payload["message"]))


def _exclusive_fixture_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _fixture_child_completes_artifacts(payload, event_queue):
    attempt = Path(payload["attempt_path"])
    event_queue.put({"kind": "setup_complete"})
    _exclusive_fixture_bytes(
        attempt / "progress.jsonl",
        b'{"step":1,"status":"running"}\n',
    )
    event_queue.put({"kind": "progress", "step": 1})
    time.sleep(0.01)
    _exclusive_fixture_bytes(attempt / "checkpoint.pt", b"checkpoint")
    values = {
        "runtime.json": {"status": "COMPLETE", "actual_updates": 1},
        "checkpoint_provenance.json": {"checkpoint_sha256": "a" * 64},
        "joint_support_state.json": {"state_sha256": "b" * 64},
        "amount_fit_state.json": {"state_sha256": "c" * 64},
        "conditioning_binding.json": {"binding_sha256": "d" * 64},
        "metrics.json": {"all_five_guards_pass": True},
        "evaluation.json": {
            "status": "VALID",
            "all_five_guards_pass": True,
            "checks": {
                "amount_ks": "PASS",
                "gap_ks": "PASS",
                "amount_abs_standardized_label_effect": "PASS",
                "gap_abs_standardized_label_effect": "PASS",
                "receiver_max_abs_signed_frequency": "PASS",
            },
        },
    }
    for name, value in values.items():
        _exclusive_fixture_bytes(
            attempt / name,
            (json.dumps(value, sort_keys=True) + "\n").encode("utf-8"),
        )
    _exclusive_fixture_bytes(attempt / "sample.npz", b"sample")
    return {"terminal_status": "COMPLETE"}


def test_execution_plan_has_exactly_two_frozen_v3_candidates_and_splits():
    plan = build_v3_execution_plan(RUNNER_CONFIG)
    report = v3_execution_plan_report(plan)

    assert [operation.candidate_id for operation in plan.operations] == [
        "cof_v3_c01_direct_joint",
        "cof_v3_c02_factorized_joint",
    ]
    assert [operation.architecture for operation in plan.operations] == [
        "direct_joint",
        "factorized_joint",
    ]
    assert all(operation.training_split == "train" for operation in plan.operations)
    assert all(
        operation.validation_use == "post_generation_five_guard_only"
        for operation in plan.operations
    )
    assert all(operation.seed == 3001 for operation in plan.operations)
    assert all(operation.requested_updates == 20_000 for operation in plan.operations)
    assert all(operation.max_wall_seconds == 7_200 for operation in plan.operations)
    assert report["sampling_plan_sha256"] == (
        "862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27"
    )
    assert report["execution_counts"] == {
        "gpu_queries": 0,
        "cuda_calls": 0,
        "model_fit_calls": 0,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "model_sample_calls": 0,
        "validation_execution_calls": 0,
        "data_generation_calls": 0,
        "selection_calls": 0,
        "test_split_reads": 0,
        "fresh_test_calls": 0,
        "tstr_calls": 0,
        "privacy_calls": 0,
        "five_seed_full_run_calls": 0,
    }
    assert report["authorization_created"] is False
    assert report["runtime_artifact_created"] is False


def test_candidate_pipeline_fits_train_then_uses_validation_only_for_evaluation():
    operation = build_v3_execution_plan(RUNNER_CONFIG).operations[0]
    calls = []

    class FixtureAdapter:
        architecture = "direct_joint"

        def fit_train_only(self, batch, *, split, operation):
            calls.append(("fit", batch, split, operation.candidate_id))
            return {"checkpoint": "fixture"}

        def sample_validation(self, plan, *, checkpoint, operation):
            calls.append(("sample", plan, checkpoint, operation.candidate_id))
            return "synthetic-validation"

    def evaluator(real_validation, synthetic, *, operation):
        calls.append(
            ("evaluate", real_validation, synthetic, operation.candidate_id)
        )
        return {"status": "VALID"}

    result = execute_candidate_stages(
        operation=operation,
        train_batch="train-batch",
        validation_batch="validation-batch",
        sampling_plan="frozen-plan",
        adapter=FixtureAdapter(),
        evaluator=evaluator,
    )

    assert result == {"status": "VALID"}
    assert calls == [
        (
            "fit",
            "train-batch",
            "train",
            "cof_v3_c01_direct_joint",
        ),
        (
            "sample",
            "frozen-plan",
            {"checkpoint": "fixture"},
            "cof_v3_c01_direct_joint",
        ),
        (
            "evaluate",
            "validation-batch",
            "synthetic-validation",
            "cof_v3_c01_direct_joint",
        ),
    ]

    with pytest.raises(V3ContractError, match="train-only"):
        execute_candidate_stages(
            operation=replace(operation, training_split="validation"),
            train_batch="train-batch",
            validation_batch="validation-batch",
            sampling_plan="frozen-plan",
            adapter=FixtureAdapter(),
            evaluator=evaluator,
        )


def test_runtime_model_is_bound_to_candidate_path_and_rejects_forbidden_hooks():
    plan = build_v3_execution_plan(RUNNER_CONFIG)
    codec = JointStateCodec(gap_bins=16, receiver_classes=64)

    for operation in plan.operations:
        denoiser = CoFSeqDenoiserV3(
            d_num=1,
            codec=codec,
            candidate=operation.architecture,
            d_model=8,
            n_heads=2,
            n_layers=1,
            max_length=4,
        )
        model = CoFSeqGenV3(
            denoiser=denoiser,
            codec=codec,
            joint_support_mask=torch.ones(
                codec.state_count,
                dtype=torch.bool,
            ),
            joint_support_sha256="a" * 64,
            coherence_lambda=0.0,
        )
        report = validate_v3_runtime_model(
            operation=operation,
            model=model,
            enabled_hooks=(),
        )
        assert report["architecture"] == operation.architecture
        assert report["legacy_independent_heads_present"] is False
        assert report["post_hoc_calibration_hooks"] == []

        with pytest.raises(V3ContractError, match="post-hoc"):
            validate_v3_runtime_model(
                operation=operation,
                model=model,
                enabled_hooks=("gap_distribution_transport",),
            )

    legacy_like = type(
        "LegacyLike",
        (),
        {
            "denoiser": type(
                "LegacyDenoiser",
                (),
                {"bin_head": object(), "cat_heads": [object()]},
            )(),
        },
    )()
    with pytest.raises(V3ContractError, match="legacy"):
        validate_v3_runtime_model(
            operation=plan.operations[0],
            model=legacy_like,
            enabled_hooks=(),
        )


def test_checkpoint_provenance_binds_support_amount_and_frozen_hashes():
    plan = build_v3_execution_plan(RUNNER_CONFIG)
    operation = plan.operations[0]
    support_payload = {
        "schema_version": "cof-seqgen-v3-joint-support-v1",
        "fit_split": "train",
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "provenance": {
            "train_file_sha256": (
                plan.raw["data_contract"]["train_file_sha256"]
            ),
            "train_content_sha256": (
                plan.raw["data_contract"]["train_content_sha256"]
            ),
            "sampling_plan_sha256": (
                plan.raw["data_contract"]["sampling_plan_sha256"]
            ),
        },
        "support_mask": [True, False, True],
    }
    support_state = {
        **support_payload,
        "state_sha256": canonical_sha256(support_payload),
    }
    amount_payload = {
        "schema_version": "cof-seqgen-v3-amount-fit-state-v1",
        "algorithm": "train_fitted_centered_empirical_residual",
        "quantile_grid_size": 257,
        "fit_split": "train",
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "parameters_sha256": "d" * 64,
    }
    amount_state = {
        **amount_payload,
        "state_sha256": canonical_sha256(amount_payload),
    }

    provenance = build_v3_checkpoint_provenance(
        plan=plan,
        operation=operation,
        source_commit="a" * 40,
        train_manifest_sha256="b" * 64,
        joint_support_state=support_state,
        amount_fit_state=amount_state,
    )

    assert provenance["candidate_id"] == operation.candidate_id
    assert provenance["joint_support_state_sha256"] == (
        support_state["state_sha256"]
    )
    assert provenance["amount_fit_state_sha256"] == (
        amount_state["state_sha256"]
    )
    assert provenance["sampling_plan_sha256"] == (
        plan.raw["data_contract"]["sampling_plan_sha256"]
    )
    validate_v3_checkpoint_provenance(
        provenance=provenance,
        plan=plan,
        operation=operation,
        source_commit="a" * 40,
        train_manifest_sha256="b" * 64,
        joint_support_state=support_state,
        amount_fit_state=amount_state,
    )

    corrupted = dict(provenance)
    corrupted["joint_support_state_sha256"] = "0" * 64
    with pytest.raises(V3ContractError, match="checkpoint provenance"):
        validate_v3_checkpoint_provenance(
            provenance=corrupted,
            plan=plan,
            operation=operation,
            source_commit="a" * 40,
            train_manifest_sha256="b" * 64,
            joint_support_state=support_state,
            amount_fit_state=amount_state,
        )


def test_artifact_store_is_exclusive_append_only_and_terminal_complete(tmp_path):
    plan = build_v3_execution_plan(RUNNER_CONFIG)
    operation = plan.operations[0]
    runtime_root = tmp_path / "candidate_selection"
    store = V3AttemptStore.claim(
        runtime_root=runtime_root,
        operation=operation,
        ownership_id="worker-a",
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    with pytest.raises(V3ContractError, match="ownership"):
        V3AttemptStore.claim(
            runtime_root=runtime_root,
            operation=operation,
            ownership_id="worker-b",
            authorization_sha256="a" * 64,
            provenance_sha256="b" * 64,
        )

    attempt = store.allocate_attempt(
        manifest={
            "schema_version": "cof-seqgen-v3-candidate-manifest-v1",
            "candidate_id": operation.candidate_id,
            "architecture": operation.architecture,
            "authorization_sha256": "a" * 64,
            "provenance_sha256": "b" * 64,
        }
    )
    assert attempt.name == "attempt_001"
    with pytest.raises(FileExistsError):
        store.write_json("manifest.json", {"changed": True})

    store.write_json("runtime.json", {"elapsed_seconds": 1.0})
    store.write_json(
        "checkpoint_provenance.json",
        {"checkpoint_sha256": "c" * 64},
    )
    store.write_json(
        "joint_support_state.json",
        {"state_sha256": "d" * 64},
    )
    store.write_json(
        "amount_fit_state.json",
        {"state_sha256": "e" * 64},
    )
    store.write_json(
        "conditioning_binding.json",
        {"binding_sha256": "f" * 64},
    )
    store.write_bytes("checkpoint.pt", b"fixture-checkpoint")
    store.write_bytes("sample.npz", b"fixture-sample")
    store.write_json(
        "metrics.json",
        {"all_five_guards_pass": True},
    )
    store.write_json(
        "evaluation.json",
        {
            "status": "VALID",
            "all_five_guards_pass": True,
            "checks": {
                "amount_ks": "PASS",
                "gap_ks": "PASS",
                "amount_abs_standardized_label_effect": "PASS",
                "gap_abs_standardized_label_effect": "PASS",
                "receiver_max_abs_signed_frequency": "PASS",
            },
        },
    )
    terminal = store.finalize("COMPLETE")

    assert terminal["status"] == "COMPLETE"
    assert (attempt / "COMPLETE.json").is_file()
    assert set(terminal["artifact_sha256"]) == {
        "manifest.json",
        "runtime.json",
        "checkpoint.pt",
        "checkpoint_provenance.json",
        "joint_support_state.json",
        "amount_fit_state.json",
        "conditioning_binding.json",
        "sample.npz",
        "metrics.json",
        "evaluation.json",
    }
    with pytest.raises(V3ContractError, match="terminal"):
        store.finalize("INVALID")
    assert json.loads((attempt / "manifest.json").read_text())[
        "candidate_id"
    ] == operation.candidate_id


def test_runner_owned_child_wall_cap_records_failed_without_unbounded_join(
    tmp_path,
):
    operation = build_v3_execution_plan(RUNNER_CONFIG).operations[0]
    store = V3AttemptStore.claim(
        runtime_root=tmp_path / "candidate_selection",
        operation=operation,
        ownership_id="wall-cap-worker",
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    attempt = store.allocate_attempt(
        manifest={
            "schema_version": "cof-seqgen-v3-candidate-manifest-v1",
            "candidate_id": operation.candidate_id,
            "architecture": operation.architecture,
            "authorization_sha256": "a" * 64,
            "provenance_sha256": "b" * 64,
        }
    )

    started = time.monotonic()
    result = run_runner_owned_child(
        child_spec=RunnerChildSpec(
            target=_fixture_child_setup_then_hangs,
            payload={"sleep_seconds": 10.0},
        ),
        store=store,
        max_wall_seconds=2.5,
        startup_timeout_seconds=5.0,
        termination_grace_seconds=0.02,
    )

    assert time.monotonic() - started < 4.0
    assert result["status"] == "FAILED"
    assert result["failure_class"] == "wall_cap"
    assert result["runner_owned_child_only"] is True
    assert (attempt / "FAILED.json").is_file()


def test_runner_owned_child_setup_timeout_records_terminal_failure(tmp_path):
    operation = build_v3_execution_plan(RUNNER_CONFIG).operations[0]
    store = V3AttemptStore.claim(
        runtime_root=tmp_path / "candidate_selection",
        operation=operation,
        ownership_id="startup-timeout-worker",
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    attempt = store.allocate_attempt(
        manifest={
            "schema_version": "cof-seqgen-v3-candidate-manifest-v1",
            "candidate_id": operation.candidate_id,
            "architecture": operation.architecture,
            "authorization_sha256": "a" * 64,
            "provenance_sha256": "b" * 64,
        }
    )

    started = time.monotonic()
    result = run_runner_owned_child(
        child_spec=RunnerChildSpec(
            target=_fixture_child_never_reports_setup,
            payload={"sleep_seconds": 10.0},
        ),
        store=store,
        max_wall_seconds=10.0,
        startup_timeout_seconds=0.05,
        termination_grace_seconds=0.02,
    )

    assert time.monotonic() - started < 1.0
    assert result["status"] == "FAILED"
    assert result["failure_class"] == "startup_timeout"
    assert result["setup_signal_received"] is False
    assert (attempt / "FAILED.json").is_file()


def test_runner_owned_child_progress_timeout_after_setup_is_terminal(tmp_path):
    operation = build_v3_execution_plan(RUNNER_CONFIG).operations[0]
    store = V3AttemptStore.claim(
        runtime_root=tmp_path / "candidate_selection",
        operation=operation,
        ownership_id="progress-timeout-worker",
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    attempt = store.allocate_attempt(
        manifest={
            "schema_version": "cof-seqgen-v3-candidate-manifest-v1",
            "candidate_id": operation.candidate_id,
            "architecture": operation.architecture,
        }
    )
    result = run_runner_owned_child(
        child_spec=RunnerChildSpec(
            target=_fixture_child_setup_then_hangs,
            payload={"sleep_seconds": 10.0},
        ),
        store=store,
        max_wall_seconds=10.0,
        startup_timeout_seconds=5.0,
        progress_timeout_seconds=0.05,
        termination_grace_seconds=0.02,
    )

    assert result["status"] == "FAILED"
    assert result["failure_class"] == "progress_timeout"
    assert result["setup_signal_received"] is True
    assert (attempt / "FAILED.json").is_file()


@pytest.mark.parametrize(
    ("target", "failure_class"),
    [
        (_fixture_child_raises, "child_exception"),
        (_fixture_child_interrupts, "operator_interrupt"),
    ],
)
def test_child_exception_and_interrupt_leave_append_only_failed_marker(
    tmp_path,
    target,
    failure_class,
):
    operation = build_v3_execution_plan(RUNNER_CONFIG).operations[0]
    store = V3AttemptStore.claim(
        runtime_root=tmp_path / failure_class,
        operation=operation,
        ownership_id=f"{failure_class}-worker",
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    attempt = store.allocate_attempt(
        manifest={
            "schema_version": "cof-seqgen-v3-candidate-manifest-v1",
            "candidate_id": operation.candidate_id,
            "architecture": operation.architecture,
        }
    )
    result = run_runner_owned_child(
        child_spec=RunnerChildSpec(
            target=target,
            payload={"message": "fixture failure"},
        ),
        store=store,
        max_wall_seconds=10.0,
        startup_timeout_seconds=5.0,
        progress_timeout_seconds=1.0,
        termination_grace_seconds=0.1,
    )

    assert result["status"] == "FAILED"
    assert result["failure_class"] == failure_class
    assert "fixture failure" in result["exception"]
    assert (attempt / "FAILED.json").is_file()
    with pytest.raises(V3ContractError, match="terminal"):
        store.finalize("FAILED")


def test_child_spawn_error_is_recorded_instead_of_leaving_markerless_attempt(
    tmp_path,
):
    operation = build_v3_execution_plan(RUNNER_CONFIG).operations[0]
    store = V3AttemptStore.claim(
        runtime_root=tmp_path / "spawn-error",
        operation=operation,
        ownership_id="spawn-error-worker",
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    attempt = store.allocate_attempt(
        manifest={
            "schema_version": "cof-seqgen-v3-candidate-manifest-v1",
            "candidate_id": operation.candidate_id,
            "architecture": operation.architecture,
        }
    )

    result = run_runner_owned_child(
        child_spec=RunnerChildSpec(
            target=lambda payload, events: None,
            payload={},
        ),
        store=store,
        max_wall_seconds=1.0,
        startup_timeout_seconds=0.5,
        termination_grace_seconds=0.05,
    )

    assert result["status"] == "FAILED"
    assert result["failure_class"] == "child_start_error"
    assert (attempt / "FAILED.json").is_file()


def test_normal_child_finalizes_candidate_before_worker_terminal(tmp_path):
    operation = build_v3_execution_plan(RUNNER_CONFIG).operations[0]
    store = V3AttemptStore.claim(
        runtime_root=tmp_path / "candidate_selection",
        operation=operation,
        ownership_id="normal-worker",
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    attempt = store.allocate_attempt(
        manifest={
            "schema_version": "cof-seqgen-v3-candidate-manifest-v1",
            "candidate_id": operation.candidate_id,
            "architecture": operation.architecture,
            "authorization_sha256": "a" * 64,
            "provenance_sha256": "b" * 64,
        }
    )

    result = run_runner_owned_child(
        child_spec=RunnerChildSpec(
            target=_fixture_child_completes_artifacts,
            payload={"attempt_path": str(attempt)},
        ),
        store=store,
        max_wall_seconds=10.0,
        startup_timeout_seconds=5.0,
        progress_timeout_seconds=2.0,
        termination_grace_seconds=0.1,
    )
    worker = store.finalize_worker(result)

    candidate_terminal = attempt / "COMPLETE.json"
    worker_terminal = (
        store.ownership_path.parent
        / attempt.name
        / "WORKER_COMPLETE.json"
    )
    assert result["status"] == "COMPLETE"
    assert worker["status"] == "COMPLETE"
    assert worker["candidate_terminal_sha256"] == sha256_file(
        candidate_terminal
    )
    assert candidate_terminal.stat().st_mtime_ns <= (
        worker_terminal.stat().st_mtime_ns
    )


def test_attempt_002_requires_hash_bound_continuation_and_preserves_attempt_001(
    tmp_path,
):
    operation = build_v3_execution_plan(RUNNER_CONFIG).operations[0]
    runtime_root = tmp_path / "candidate_selection"
    first = V3AttemptStore.claim(
        runtime_root=runtime_root,
        operation=operation,
        ownership_id="attempt-001-worker",
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    attempt_001 = first.allocate_attempt(
        manifest={
            "schema_version": "cof-seqgen-v3-candidate-manifest-v1",
            "candidate_id": operation.candidate_id,
            "architecture": operation.architecture,
            "authorization_sha256": "a" * 64,
        }
    )
    first.write_json("runtime.json", {"status": "FAILED"})
    first.write_bytes("checkpoint.pt", b"failed")
    for name in (
        "checkpoint_provenance.json",
        "joint_support_state.json",
        "amount_fit_state.json",
        "conditioning_binding.json",
    ):
        first.write_json(name, {"status": "UNAVAILABLE"})
    first.write_bytes("sample.npz", b"failed")
    first.write_json("metrics.json", {"status": "NOT_COMPUTED"})
    first.write_json(
        "evaluation.json",
        {"status": "NOT_COMPUTED", "all_five_guards_pass": False},
    )
    failed = first.finalize(
        "FAILED",
        metadata={"failure_class": "startup_timeout"},
    )
    before = read_only_tree_inventory(attempt_001)
    continuation = {
        "schema_version": "cof-seqgen-v3-continuation-v1",
        "previous_attempt": "attempt_001",
        "previous_attempt_tree_sha256": before["tree_sha256"],
        "previous_terminal_status": "FAILED",
        "previous_terminal_sha256": sha256_file(attempt_001 / "FAILED.json"),
        "previous_authorization_sha256": "a" * 64,
        "next_attempt": "attempt_002",
    }

    with pytest.raises(V3ContractError, match="continuation"):
        V3AttemptStore.claim(
            runtime_root=runtime_root,
            operation=operation,
            ownership_id="attempt-002-without-authorization",
            authorization_sha256="c" * 64,
            provenance_sha256="d" * 64,
        )
    with pytest.raises(V3ContractError, match="continuation"):
        V3AttemptStore.claim(
            runtime_root=runtime_root,
            operation=operation,
            ownership_id="attempt-002-corrupt",
            authorization_sha256="c" * 64,
            provenance_sha256="d" * 64,
            continuation={
                **continuation,
                "previous_attempt_tree_sha256": "0" * 64,
            },
        )

    second = V3AttemptStore.claim(
        runtime_root=runtime_root,
        operation=operation,
        ownership_id="attempt-002-worker",
        authorization_sha256="c" * 64,
        provenance_sha256="d" * 64,
        continuation=continuation,
    )
    attempt_002 = second.allocate_attempt(
        manifest={
            "schema_version": "cof-seqgen-v3-candidate-manifest-v1",
            "candidate_id": operation.candidate_id,
            "architecture": operation.architecture,
            "authorization_sha256": "c" * 64,
        }
    )

    assert attempt_002.name == "attempt_002"
    assert second.ownership_path.name == "ownership_attempt_002.json"
    assert read_only_tree_inventory(attempt_001) == before
    assert failed["status"] == "FAILED"


def test_future_aggregate_schema_selects_only_complete_all_five_pass():
    checks_pass = {
        "amount_ks": "PASS",
        "gap_ks": "PASS",
        "amount_abs_standardized_label_effect": "PASS",
        "gap_abs_standardized_label_effect": "PASS",
        "receiver_max_abs_signed_frequency": "PASS",
    }
    records = {
        "cof_v3_c01_direct_joint": {
            "terminal": {
                "status": "COMPLETE",
                "aggregate_eligible": True,
                "artifact_sha256": {
                    name: "a" * 64
                    for name in V3AttemptStore.REQUIRED_ARTIFACTS
                },
            },
            "evaluation": {
                "status": "VALID",
                "all_five_guards_pass": True,
                "checks": checks_pass,
            },
        },
        "cof_v3_c02_factorized_joint": {
            "terminal": {
                "status": "INVALID",
                "aggregate_eligible": False,
                "artifact_sha256": {
                    name: "b" * 64
                    for name in V3AttemptStore.REQUIRED_ARTIFACTS
                },
            },
            "evaluation": {
                "status": "INVALID",
                "all_five_guards_pass": False,
                "checks": {**checks_pass, "gap_ks": "FAIL"},
            },
        },
    }

    manifest = build_v3_aggregate_input_manifest(records)

    assert manifest["eligible_candidates"] == [
        "cof_v3_c01_direct_joint"
    ]
    assert manifest["selection_executed"] is False
    assert manifest["candidate_status"] == {
        "cof_v3_c01_direct_joint": "ELIGIBLE",
        "cof_v3_c02_factorized_joint": "INELIGIBLE",
    }

    incomplete = dict(records)
    incomplete.pop("cof_v3_c02_factorized_joint")
    with pytest.raises(V3ContractError, match="both v3 candidates"):
        build_v3_aggregate_input_manifest(incomplete)


def test_real_v3_candidate_adapters_bind_exact_paths_without_execution(
    monkeypatch,
):
    plan = build_v3_execution_plan(RUNNER_CONFIG)
    fixture_plan = SamplingPlan(
        y_entity=torch.tensor([0, 1]).numpy().astype("int64"),
        lengths=torch.tensor([2, 1]).numpy().astype("int64"),
        valid_mask=torch.tensor(
            [[True, True], [True, False]]
        ).numpy(),
        plan_hash=plan.raw["data_contract"]["sampling_plan_sha256"],
    )
    calls = {
        "cuda": 0,
        "fit": 0,
        "sample": 0,
    }

    def forbidden(*args, **kwargs):
        calls["cuda"] += 1
        raise AssertionError("CUDA must not be touched while planning")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    for operation in plan.operations:
        adapter = build_v3_candidate_adapter(
            plan=plan,
            operation=operation,
            sampling_plan=fixture_plan,
            device="cuda:0",
        )
        assert isinstance(adapter, CoFSeqGenV3CandidateAdapter)
        assert adapter.architecture == operation.architecture
        assert adapter.sampling_plan.plan_hash == fixture_plan.plan_hash
    assert calls == {"cuda": 0, "fit": 0, "sample": 0}


def test_execution_dry_run_verifies_train_validation_without_runtime(
    tmp_path,
    monkeypatch,
):
    plan = build_v3_execution_plan(RUNNER_CONFIG)
    calls = {"cuda": 0, "fit": 0, "sample": 0}

    def forbidden(*args, **kwargs):
        calls["cuda"] += 1
        raise AssertionError("dry-run cannot query CUDA")

    runtime_root = (
        plan.repository_root / plan.raw["future_runtime_root"]
    )
    runtime_before = read_only_tree_inventory(runtime_root)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    report = dry_run_v3_execution(plan)

    assert report["status"] == "PASS"
    assert report["train_loaded_for_contract_validation"] is True
    assert report["validation_file_hash_verified"] is True
    assert report["validation_rows_loaded"] == 0
    assert report["test_split_reads"] == 0
    assert report["execution_counts"] == {
        "gpu_queries": 0,
        "cuda_calls": 0,
        "model_fit_calls": 0,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "model_sample_calls": 0,
        "validation_execution_calls": 0,
        "data_generation_calls": 0,
        "selection_calls": 0,
        "test_split_reads": 0,
        "fresh_test_calls": 0,
        "tstr_calls": 0,
        "privacy_calls": 0,
        "five_seed_full_run_calls": 0,
    }
    assert read_only_tree_inventory(runtime_root) == runtime_before
    assert report["runtime_artifact_created"] is False
    assert report["preserved_runtime_tree_sha256"] == (
        runtime_before["tree_sha256"]
    )
    assert calls == {"cuda": 0, "fit": 0, "sample": 0}


def test_execution_authorization_is_exact_and_test_fail_closed():
    plan = build_v3_execution_plan(RUNNER_CONFIG)
    source_commit = "1" * 40
    runner_hash = "2" * 64
    data = plan.raw["data_contract"]
    authorization = {
        "schema_version": "cof-seqgen-v3-execution-authorization-v1",
        "status": "AUTHORIZED",
        "source_commit": source_commit,
        "runner_relevant_source_sha256": runner_hash,
        "runner_config_sha256": plan.runner_config_sha256,
        "source_config_sha256": plan.raw["source_config_sha256"],
        "approved_candidates": [
            "cof_v3_c01_direct_joint",
            "cof_v3_c02_factorized_joint",
        ],
        "provenance": {
            "development_manifest_sha256": data[
                "development_manifest_sha256"
            ],
            "train_file_sha256": data["train_file_sha256"],
            "train_content_sha256": data["train_content_sha256"],
            "validation_file_sha256": data["validation_file_sha256"],
            "validation_content_sha256": data[
                "validation_content_sha256"
            ],
            "sampling_plan_sha256": data["sampling_plan_sha256"],
        },
        "scope": {
            "gpu_query": False,
            "cuda": True,
            "model_fit": True,
            "optimizer_updates": True,
            "checkpoint_write": True,
            "model_sample": True,
            "validation_five_guard_evaluation": True,
            "aggregate_selection": False,
            "test_split_access": False,
            "fresh_test": False,
            "tstr": False,
            "privacy": False,
            "five_seed_full_run": False,
            "sweep": False,
            "early_stopping": False,
        },
    }
    report = validate_v3_execution_authorization(
        plan=plan,
        authorization=authorization,
        source_commit=source_commit,
        runner_relevant_source_sha256=runner_hash,
    )
    assert report["status"] == "AUTHORIZED"
    assert report["test_split_access"] is False

    changed = json.loads(json.dumps(authorization))
    changed["scope"]["test_split_access"] = True
    with pytest.raises(V3ContractError, match="authorization"):
        validate_v3_execution_authorization(
            plan=plan,
            authorization=changed,
            source_commit=source_commit,
            runner_relevant_source_sha256=runner_hash,
        )


def test_execute_cli_requires_authorization_candidate_and_device():
    with pytest.raises(SystemExit):
        parse_args(["execute"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "execute",
                "--candidate",
                "cof_v3_c01_direct_joint",
                "--device",
                "cuda:0",
            ]
        )
    parsed = parse_args(
        [
            "execute",
            "--candidate",
            "cof_v3_c01_direct_joint",
            "--device",
            "cuda:0",
            "--authorization",
            "/future/authorization.json",
            "--ownership-id",
            "fixture-worker",
        ]
    )
    assert parsed.mode == "execute"
    assert parsed.candidate == "cof_v3_c01_direct_joint"
