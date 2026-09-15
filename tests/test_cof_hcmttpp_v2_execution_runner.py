from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from experiments.cof_hcmttpp_v2_execution_runner import (
    ExecutionDependencies,
    H1AttemptStore,
    H1RunnerContractError,
    RunnerChildSpec,
    ZERO_EXECUTION_COUNTS,
    build_expected_authorization_claim,
    build_checkpoint_provenance,
    build_execution_plan,
    dry_run_execution,
    execute_authorized,
    evaluate_h1_gate,
    resolve_dataset_access,
    run_runner_owned_child,
    validate_authorization_document,
    validate_checkpoint_provenance,
    _authorized_dependency_child,
)
from tests.cof_hcmttpp_v2_child_fixtures import (
    boundary_build_model,
    boundary_load_train_body,
    boundary_run_job,
    boundary_select_device,
    child_exception,
    progress_hang,
    startup_hang,
    successful_child,
)
from generators.cof_hcmttpp_v2_execution_backend import (
    validate_runtime_h1_factor_isolation,
)
from scripts.run_cof_hcmttpp_v2 import parse_args


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmark_v2/cof_hcmttpp_v2_execution_runner.yaml"


def test_first_valid_plan_is_exact_attempt_003_and_binds_both_failed_attempts():
    plan = build_execution_plan(CONFIG, dataset="amlsim", candidate_id="H1")
    job = plan.jobs[0]
    claim = build_expected_authorization_claim(plan)

    assert job.attempt == "attempt_003"
    assert claim["attempt"] == "attempt_003"
    assert claim["scope"]["attempt"] == "attempt_003"
    correction = claim["implementation_corrections"]
    assert correction["execution_class"] == (
        "FIRST_VALID_H1_EVALUATION_AFTER_TWO_DETERMINISTIC_IMPLEMENTATION_FAILURES"
    )
    assert correction["authorized_attempt"] == "attempt_003"
    assert correction["generic_retry"] is False
    assert correction["attempt_sweep"] is False
    assert set(correction["failed_attempts"]) == {"attempt_001", "attempt_002"}
    assert set(correction["failed_attempts"]["attempt_001"]) == {
        "amlsim",
        "sparkov",
    }
    assert set(correction["failed_attempts"]["attempt_002"]) == {
        "amlsim",
        "sparkov",
    }


def test_plan_and_dry_run_are_one_dataset_h1_only_and_side_effect_free():
    model_was_imported = "models.cof_hcmttpp_v2" in sys.modules
    runtime_root = ROOT / "artifacts/cof_hcmttpp_v2"
    runtime_existed = runtime_root.exists()
    amlsim = build_execution_plan(CONFIG, dataset="amlsim", candidate_id="H1")
    sparkov = build_execution_plan(CONFIG, dataset="sparkov", candidate_id="H1")

    assert len(amlsim.jobs) == len(sparkov.jobs) == 1
    assert amlsim.jobs[0].dataset == "amlsim"
    assert sparkov.jobs[0].dataset == "sparkov"
    for plan in (amlsim, sparkov):
        job = plan.jobs[0]
        assert job.candidate_id == "H1"
        assert job.seed == 4001
        assert job.attempt == "attempt_003"
        report = dry_run_execution(plan)
        assert report["dataset_scope"] == job.dataset
        assert report["job_count"] == 1
        assert report["jobs"] == [
            {
                "dataset": job.dataset,
                "candidate_id": "H1",
                "seed": 4001,
                "attempt": "attempt_003",
            }
        ]
        assert report["execution_counts"] == ZERO_EXECUTION_COUNTS
        assert report["runtime_artifacts_created"] is False
        assert report["implementation_correction_evidence"]["preservation_status"] == "PASS"
        assert set(
            report["implementation_correction_evidence"]["failed_attempts"]
        ) == {"attempt_001", "attempt_002"}

    if not model_was_imported:
        assert "models.cof_hcmttpp_v2" not in sys.modules
    assert runtime_root.exists() is runtime_existed

    with pytest.raises(H1RunnerContractError, match="only H1"):
        build_execution_plan(CONFIG, dataset="amlsim", candidate_id="H2")
    with pytest.raises(H1RunnerContractError, match="dataset"):
        build_execution_plan(CONFIG, dataset="both", candidate_id="H1")


def test_authorization_is_dataset_scoped_and_precedes_every_dependency(tmp_path):
    amlsim = build_execution_plan(CONFIG, dataset="amlsim", candidate_id="H1")
    sparkov = build_execution_plan(CONFIG, dataset="sparkov", candidate_id="H1")
    amended = dict(amlsim.raw)
    amended["authorization_history_root"] = str(tmp_path)
    amlsim = replace(amlsim, raw=amended)
    wrong = build_expected_authorization_claim(sparkov)
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps(wrong), encoding="utf-8")
    dependency_calls = []

    with pytest.raises(H1RunnerContractError, match="dataset-scoped"):
        execute_authorized(
            plan=amlsim,
            authorization_path=authorization,
            dependencies=lambda: dependency_calls.append("called"),
        )
    assert dependency_calls == []
    with pytest.raises(H1RunnerContractError, match="authorization"):
        execute_authorized(
            plan=amlsim,
            authorization_path=None,
            dependencies=lambda: dependency_calls.append("called"),
        )
    assert dependency_calls == []


def test_attempt_003_preserves_prior_attempts_and_rejects_arbitrary_attempt(tmp_path):
    plan = build_execution_plan(CONFIG, dataset="amlsim", candidate_id="H1")
    job = plan.jobs[0]
    old_lock = tmp_path / "workers/amlsim/H1/ownership.lock"
    old_attempt = tmp_path / "amlsim/H1/seed_4001/attempt_001"
    old_lock.parent.mkdir(parents=True)
    old_attempt.mkdir(parents=True)
    old_lock.write_bytes(b"preserved-lock\n")
    (old_attempt / "FAILED.json").write_bytes(b"preserved-failure\n")

    store = H1AttemptStore.claim(
        runtime_root=tmp_path,
        job=job,
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    allocated = store.allocate_attempt(
        {"dataset": "amlsim", "candidate_id": "H1", "seed": 4001}
    )
    assert allocated.name == "attempt_003"
    assert store.ownership_path.name == "ownership_attempt_003.lock"
    assert old_lock.read_bytes() == b"preserved-lock\n"
    assert (old_attempt / "FAILED.json").read_bytes() == b"preserved-failure\n"

    approval = {"approved": True, "text": "Explicit H1 attempt_003 approval"}
    corrupted = build_expected_authorization_claim(plan, approval=approval)
    corrupted["attempt"] = "attempt_004"
    with pytest.raises(H1RunnerContractError, match="hash-bound"):
        validate_authorization_document(plan, corrupted)

def test_split_access_is_train_first_validation_after_checkpoint_and_test_blocked():
    plan = build_execution_plan(CONFIG, dataset="sparkov", candidate_id="H1")
    train = resolve_dataset_access(
        plan=plan,
        split="train",
        purpose="tail_state_fit_and_training",
        final_checkpoint_complete=False,
    )
    assert train.name == "train.npz"

    with pytest.raises(H1RunnerContractError, match="final checkpoint"):
        resolve_dataset_access(
            plan=plan,
            split="validation",
            purpose="fixed_plan_sampling_and_evaluation",
            final_checkpoint_complete=False,
        )
    validation = resolve_dataset_access(
        plan=plan,
        split="validation",
        purpose="fixed_plan_sampling_and_evaluation",
        final_checkpoint_complete=True,
    )
    assert validation.name == "validation.npz"

    for forbidden in ("internal_test", "fraudTest", "fraud_test", "test"):
        with pytest.raises(H1RunnerContractError, match="forbidden"):
            resolve_dataset_access(
                plan=plan,
                split=forbidden,
                purpose="anything",
                final_checkpoint_complete=True,
            )


def _valid_h1_diagnostics():
    return {
        "gap": {
            key: {}
            for key in (
                "zero_rate_by_class",
                "zero_rate_absolute_error_by_class",
                "overall_gap_ks_by_class",
                "positive_only_gap_ks_by_class",
                "positive_quantile_differences_by_class",
                "central_tail_counts_by_class",
                "tail_mass_error_by_class",
                "conditional_tail_gate_by_class_and_history_stratum",
                "conditional_tail_scale_by_class_and_history_stratum",
                "signed_scale_diagnostics",
                "positive_cdf_total_mass_error",
                "threshold_cdf_continuity_error",
                "finite_nll_count_by_route",
                "finite_density_count",
                "sampled_min_max",
                "lower_boundary_rate",
                "nonfinite_sample_count",
                "exact_upper_clip_count",
                "frozen_bin_mass",
                "raw_vs_mapped_ks_difference",
            )
        },
        "receiver": {
            key: {}
            for key in (
                "classwise_receiver_tv",
                "full_receiver_tv",
                "head_receiver_tv",
                "tail_receiver_tv",
                "unk_rate_error",
                "head",
                "tail",
                "unk",
                "repeat",
                "new",
            )
        },
        "amount": {"contract": "frozen"},
        "hard_validity": {"status": "PASS"},
        "forbidden_access": {"internal_test": 0, "sparkov_fraud_test": 0},
    }


def _write_complete_h1_artifacts(store, diagnostics=None):
    store.append_progress({"step": 1, "status": "TRAINING"})
    store.write_json("frozen_parent_references.json", {"status": "PASS"})
    store.write_json("gap_hurdle_state.json", {"fit_split": "train"})
    store.write_json("positive_gap_spline_state.json", {"bins": 16})
    store.write_json("positive_gap_tail_state.json", {"fit_split": "train"})
    store.write_json("amount_transform_reference.json", {"rows": "train"})
    store.write_json("receiver_vocabulary_reference.json", {"path": "flat"})
    store.write_json("conditioning_plan.json", {"source": "fixed"})
    store.write_bytes("checkpoints/final.pt", b"checkpoint")
    store.write_bytes("checkpoints/latest", b"final.pt\n")
    store.write_json("checkpoint_provenance.json", {"status": "PASS"})
    store.write_bytes("validation_sample.npz", b"sample")
    store.write_json("diagnostics.json", diagnostics or _valid_h1_diagnostics())
    store.write_json("metrics.json", {"finite": True})
    store.write_json("evaluation.json", {"status": "VALID"})
    store.write_json("runtime.json", {"status": "COMPLETE"})
    store.write_json("gate_decision.json", {"status": "PASS"})


def test_append_only_store_is_dataset_owned_checksum_bound_and_terminal_last(tmp_path):
    plan = build_execution_plan(CONFIG, dataset="amlsim", candidate_id="H1")
    job = plan.jobs[0]
    store = H1AttemptStore.claim(
        runtime_root=tmp_path,
        job=job,
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    attempt = store.allocate_attempt(
        {"dataset": "amlsim", "candidate_id": "H1", "seed": 4001}
    )
    with pytest.raises(H1RunnerContractError, match="missing"):
        store.finalize("COMPLETE")
    _write_complete_h1_artifacts(store)
    terminal = store.finalize("COMPLETE")
    assert terminal["status"] == "COMPLETE"
    assert (attempt / "artifact_index.json").is_file()
    assert (attempt / "checksum_manifest.json").is_file()
    assert (attempt / "COMPLETE.json").is_file()
    index = json.loads((attempt / "artifact_index.json").read_text())
    assert "COMPLETE.json" not in index["files"]
    with pytest.raises(H1RunnerContractError, match="immutable"):
        store.write_json("late.json", {"forbidden": True})
    with pytest.raises(H1RunnerContractError, match="ownership"):
        H1AttemptStore.claim(
            runtime_root=tmp_path,
            job=job,
            authorization_sha256="a" * 64,
            provenance_sha256="b" * 64,
        )


def test_complete_fails_closed_when_h1_diagnostics_are_incomplete(tmp_path):
    _, store, _ = _fresh_store(tmp_path)
    _write_complete_h1_artifacts(store, diagnostics={"gap": {}, "receiver": {}})
    with pytest.raises(H1RunnerContractError, match="diagnostic"):
        store.finalize("COMPLETE")


def test_h1_gate_uses_unrounded_c0_c1_references_and_fail_stop_state():
    references = {
        "c0_gap_ks": {"y0": 0.20, "y1": 0.30},
        "c1_positive_gap_ks": {"y0": 0.40},
        "c1_coherence": {"y0": 0.10, "y1": 0.20},
        "c1_receiver_tv": {"y0": 0.30, "y1": 0.40},
        "c1_full_receiver_tv": 0.35,
    }
    passing = {
        "gap_ks": {"y0": 0.20, "y1": 0.29},
        "positive_gap_ks": {"y0": 0.399999},
        "coherence": {"y0": 0.10, "y1": 0.19},
        "receiver_tv": {"y0": 0.29, "y1": 0.40},
        "full_receiver_tv": 0.35,
        "hard_validity": True,
    }
    assert evaluate_h1_gate(passing, references)["status"] == "PASS"
    for key, mutation in (
        ("gap", {"gap_ks": {"y0": 0.200001, "y1": 0.29}}),
        ("positive", {"positive_gap_ks": {"y0": 0.40}}),
        ("coherence", {"coherence": {"y0": 0.100001, "y1": 0.19}}),
        ("receiver", {"full_receiver_tv": 0.350001}),
        ("validity", {"hard_validity": False}),
    ):
        observed = dict(passing)
        observed.update(mutation)
        result = evaluate_h1_gate(observed, references)
        assert result["status"] == "FAIL", key
        assert result["family_state"] == "STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL"


def _fresh_store(tmp_path, dataset="amlsim"):
    plan = build_execution_plan(CONFIG, dataset=dataset, candidate_id="H1")
    job = plan.jobs[0]
    store = H1AttemptStore.claim(
        runtime_root=tmp_path,
        job=job,
        authorization_sha256="c" * 64,
        provenance_sha256="d" * 64,
    )
    attempt = store.allocate_attempt(
        {"dataset": dataset, "candidate_id": "H1", "seed": 4001}
    )
    return job, store, attempt


def _run_child(target, store, payload=None):
    return run_runner_owned_child(
        child_spec=RunnerChildSpec(target=target, payload=payload or {}),
        store=store,
        max_wall_seconds=3,
        startup_timeout_seconds=0.15,
        progress_timeout_seconds=0.15,
        queue_poll_seconds=0.02,
        termination_grace_seconds=0.1,
        kill_grace_seconds=0.1,
    )


def test_spawn_watchdogs_and_child_exception_leave_failed_terminal(tmp_path):
    for index, (target, failure_class) in enumerate(
        (
            (startup_hang, "startup_watchdog"),
            (progress_hang, "progress_watchdog"),
            (child_exception, "child_exception"),
        )
    ):
        _, store, attempt = _fresh_store(tmp_path / str(index))
        terminal = _run_child(target, store)
        assert terminal["status"] == "FAILED"
        assert terminal["failure_class"] == failure_class
        assert terminal["child_alive_after_cleanup"] is False
        assert (attempt / "FAILED.json").is_file()
        assert (store.ownership_path.parent / "attempt_003/WORKER_FAILED.json").is_file()


def test_normal_child_orders_progress_checkpoint_candidate_and_worker_terminal(tmp_path):
    job, store, attempt = _fresh_store(tmp_path)
    payload = {
        "runtime_root": str(tmp_path),
        "ownership_path": str(store.ownership_path),
        "attempt_path": str(attempt),
        "job": {
            "dataset": job.dataset,
            "candidate_id": job.candidate_id,
            "seed": job.seed,
            "attempt": job.attempt,
        },
    }
    terminal = _run_child(successful_child, store, payload)
    assert terminal["status"] == "COMPLETE"
    assert (attempt / "progress.jsonl").is_file()
    assert (attempt / "checkpoints/final.pt").is_file()
    assert (attempt / "COMPLETE.json").is_file()
    worker = store.ownership_path.parent / "attempt_003/WORKER_COMPLETE.json"
    assert worker.is_file()


@pytest.mark.parametrize("dataset", ["amlsim", "sparkov"])
def test_parent_spawn_payload_and_backend_share_attempt_003_ownership(
    tmp_path, dataset
):
    plan = build_execution_plan(CONFIG, dataset=dataset, candidate_id="H1")
    job = plan.jobs[0]
    store = H1AttemptStore.claim(
        runtime_root=tmp_path,
        job=job,
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    attempt = store.allocate_attempt(
        {"dataset": dataset, "candidate_id": "H1", "seed": 4001}
    )
    dependencies = ExecutionDependencies(
        load_train_body=boundary_load_train_body,
        build_model=boundary_build_model,
        select_device=boundary_select_device,
        run_job=boundary_run_job,
    )
    terminal = _run_child(
        _authorized_dependency_child,
        store,
        {
            "dependencies": dependencies,
            "plan": plan,
            "job": job,
            "authorization": {"dataset_scope": dataset},
            "runtime_root": str(tmp_path),
            "ownership_path": str(store.ownership_path),
            "attempt_path": str(attempt),
        },
    )

    assert terminal["status"] == "COMPLETE"
    assert (attempt / "COMPLETE.json").is_file()
    assert store.ownership_path.name == "ownership_attempt_003.lock"


def test_spawn_child_rejects_mismatched_ownership_record_identity(tmp_path):
    plan = build_execution_plan(CONFIG, dataset="amlsim", candidate_id="H1")
    job = plan.jobs[0]
    store = H1AttemptStore.claim(
        runtime_root=tmp_path,
        job=job,
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    attempt = store.allocate_attempt(
        {"dataset": "amlsim", "candidate_id": "H1", "seed": 4001}
    )
    corrupted = json.loads(store.ownership_path.read_text(encoding="utf-8"))
    corrupted["dataset"] = "sparkov"
    store.ownership_path.write_text(
        json.dumps(corrupted, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    dependencies = ExecutionDependencies(
        load_train_body=boundary_load_train_body,
        build_model=boundary_build_model,
        select_device=boundary_select_device,
        run_job=boundary_run_job,
    )
    terminal = _run_child(
        _authorized_dependency_child,
        store,
        {
            "dependencies": dependencies,
            "plan": plan,
            "job": job,
            "authorization": {"dataset_scope": "amlsim"},
            "runtime_root": str(tmp_path),
            "ownership_path": str(store.ownership_path),
            "attempt_path": str(attempt),
        },
    )

    assert terminal["status"] == "FAILED"
    assert terminal["failure_class"] == "child_exception"
    assert "ownership record identity mismatch" in terminal["failure_event"][
        "message"
    ]


def test_checkpoint_provenance_is_train_only_and_hash_bound():
    plan = build_execution_plan(CONFIG, dataset="amlsim", candidate_id="H1")
    binding = {
        "fit_split": "train",
        "train_sha256": "1" * 64,
        "transform_sha256": "2" * 64,
        "receiver_vocabulary_sha256": "3" * 64,
        "tail_state_sha256": "4" * 64,
        "amount_contract_sha256": "5" * 64,
        "sampling_plan_sha256": "6" * 64,
        "threshold_sha256": "7" * 64,
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
    }
    provenance = build_checkpoint_provenance(
        plan=plan,
        job=plan.jobs[0],
        authorization_sha256="8" * 64,
        train_state_binding=binding,
        actual_updates=20_000,
        checkpoint_sha256="9" * 64,
    )
    assert validate_checkpoint_provenance(
        provenance,
        plan=plan,
        job=plan.jobs[0],
        authorization_sha256="8" * 64,
        train_state_binding=binding,
    )["status"] == "PASS"
    corrupted = dict(provenance)
    corrupted["tail_state_sha256"] = "0" * 64
    with pytest.raises(H1RunnerContractError, match="checkpoint provenance"):
        validate_checkpoint_provenance(
            corrupted,
            plan=plan,
            job=plan.jobs[0],
            authorization_sha256="8" * 64,
            train_state_binding=binding,
        )


def test_runtime_factor_isolation_rejects_every_non_h1_hook():
    plan = build_execution_plan(CONFIG, dataset="sparkov", candidate_id="H1")
    FlatReceiver = type("FlatReceiverDecoder", (), {})
    GapDecoder = type("H1HurdleRQSGapDecoder", (), {})
    Model = type(
        "CoFHCMTTPPV2H1",
        (),
        {
            "model_config": lambda self: {
                "candidate": "H1",
                "changed_factors": ["gap_decoder"],
                "central_bins": 16,
                "receiver_path": "flat_no_copy",
                "y_balanced_likelihood": False,
                "structure_loss": None,
                "permanent_v1_chain_state": "STOP_CCMTPP_V1_CHAIN_C1_GATE_FAIL",
            }
        },
    )
    model = Model()
    model.receiver_decoder = FlatReceiver()
    model.gap_decoder = GapDecoder()
    assert validate_runtime_h1_factor_isolation(
        model=model, plan=plan
    )["status"] == "PASS"
    for hook in (
        "pointer",
        "hierarchy",
        "y_balanced_likelihood",
        "coherence_loss",
        "posthoc_calibration",
        "c1_v1_retry",
        "C2",
        "C3",
        "C4",
        "H2",
    ):
        with pytest.raises(H1RunnerContractError, match="factor-isolation"):
            validate_runtime_h1_factor_isolation(
                model=model, plan=plan, enabled_hooks=(hook,)
            )


def test_cli_requires_one_dataset_and_exposes_only_h1():
    parsed = parse_args(
        ["--mode", "dry-run", "--dataset", "amlsim", "--candidate", "H1"]
    )
    assert parsed.dataset == "amlsim"
    with pytest.raises(SystemExit):
        parse_args(["--mode", "execute", "--candidate", "H1"])
    with pytest.raises(SystemExit):
        parse_args(
            ["--mode", "execute", "--dataset", "amlsim", "--candidate", "H2"]
        )
