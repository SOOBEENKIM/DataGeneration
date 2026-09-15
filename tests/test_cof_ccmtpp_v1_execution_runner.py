import json
import hashlib
from dataclasses import replace
import subprocess
import sys
import time
from pathlib import Path

import pytest
import numpy as np

from tests.cof_ccmtpp_v1_child_fixtures import (
    build_fixture_model,
    child_completes as _child_completes,
    child_interrupts as _child_interrupts,
    child_never_reports as _child_never_reports,
    child_raises as _child_raises,
    child_setup_then_hangs as _child_setup_then_hangs,
    load_fixture_data,
    query_fixture_device,
    run_fixture_job,
)
from eval.cof_ccmtpp_v1_contract import CCMTPPContractError
from experiments.cof_ccmtpp_v1_execution_runner import (
    ExecutionDependencies,
    CCMTPPAttemptStore,
    RunnerChildSpec,
    build_expected_authorization_claim,
    build_checkpoint_provenance,
    build_execution_plan,
    dry_run_execution,
    execute_authorized,
    execution_plan_report,
    fit_train_only_states,
    resolve_dataset_access,
    run_runner_owned_child,
    validate_authorization_document,
    validate_authorization_path,
    validate_checkpoint_provenance,
    validate_parent_gate_evidence,
    validate_runtime_candidate,
)
from models.cof_ccmtpp_v1 import CoFCCMTPPV1, ReceiverHierarchyState
from scripts.run_cof_ccmtpp_v1 import parse_args


REPOSITORY = Path(__file__).resolve().parents[1]
RUNNER_CONFIG = (
    REPOSITORY
    / "configs/benchmark_v2/cof_ccmtpp_v1_execution_runner.yaml"
)


def test_initial_plan_is_one_explicit_c1_dataset_cell():
    plan = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
    )
    report = execution_plan_report(plan)

    assert [(job.dataset, job.candidate_id) for job in plan.jobs] == [
        ("amlsim", "C1"),
    ]
    assert report["dataset_scope"] == "amlsim"
    assert plan.raw["worker_scope"] == {
        "datasets": ["amlsim", "sparkov"],
        "jobs_per_process": 1,
        "authorization_datasets": 1,
        "visible_devices_per_process": 1,
    }
    assert all(job.seed == 4001 for job in plan.jobs)
    assert all(job.attempt == "attempt_001" for job in plan.jobs)
    assert report["initial_candidate"] == "C1"
    assert report["direct_execution_allowed"] is False
    assert report["authorization_created"] is False
    assert report["runtime_artifact_created"] is False
    assert all(value == 0 for value in report["execution_counts"].values())


def test_execute_without_authorization_rejects_before_data_model_or_gpu_access():
    calls = {"data": 0, "model": 0, "gpu": 0}

    def touched(role):
        def callback(*args, **kwargs):
            del args, kwargs
            calls[role] += 1
            raise AssertionError(f"{role} must not be touched")
        return callback

    dependencies = ExecutionDependencies(
        load_data_body=touched("data"),
        build_model=touched("model"),
        query_device=touched("gpu"),
        run_job=touched("model"),
    )
    with pytest.raises(CCMTPPContractError, match="authorization"):
        execute_authorized(
            plan=build_execution_plan(
                RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
            ),
            authorization_path=None,
            dependencies=dependencies,
        )
    assert calls == {"data": 0, "model": 0, "gpu": 0}

    factory_calls = 0

    def dependency_factory():
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("dependency factory must remain authorization-gated")

    with pytest.raises(CCMTPPContractError, match="authorization"):
        execute_authorized(
            plan=build_execution_plan(
                RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
            ),
            authorization_path=None,
            dependencies=dependency_factory,
        )
    assert factory_calls == 0


def test_runner_import_and_unauthorized_cli_do_not_import_model_module():
    code = (
        "import sys; "
        "import experiments.cof_ccmtpp_v1_execution_runner; "
        "print(int('models.cof_ccmtpp_v1' in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "0"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys; "
                "from scripts.run_cof_ccmtpp_v1 import main; "
                "main(['--mode','dry-run','--candidate','C1','--dataset','amlsim']); "
                "print(json.dumps({'model':int('models.cof_ccmtpp_v1' in sys.modules),"
                "'backend':int('generators.cof_ccmtpp_v1_execution_backend' in sys.modules)}))"
            ),
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    import_state = json.loads(completed.stdout.splitlines()[-1])
    assert import_state == {"model": 0, "backend": 0}


def test_cli_has_plan_dry_run_execute_but_no_authorization_creation_mode():
    assert parse_args(
        ["--mode", "plan", "--candidate", "C1", "--dataset", "amlsim"]
    ).dataset == "amlsim"
    assert parse_args(
        ["--mode", "dry-run", "--candidate", "C1", "--dataset", "sparkov"]
    ).dataset == "sparkov"
    execute = parse_args(
        [
            "--mode", "execute", "--candidate", "C1",
            "--dataset", "amlsim",
            "--authorization", "authorization.json",
        ]
    )
    assert execute.mode == "execute"
    with pytest.raises(SystemExit):
        parse_args(["--mode", "execute", "--candidate", "C1"])
    with pytest.raises(SystemExit):
        parse_args([
            "--mode", "create-authorization", "--candidate", "C1",
            "--dataset", "amlsim",
        ])
    with pytest.raises(SystemExit):
        parse_args([
            "--mode", "execute", "--candidate", "C5",
            "--dataset", "amlsim",
        ])


def test_default_backend_factory_is_lazy_and_has_exact_execution_stages():
    code = (
        "import sys; "
        "from generators.cof_ccmtpp_v1_execution_backend import "
        "build_default_execution_dependencies; "
        "d=build_default_execution_dependencies(); "
        "print(d.load_data_body.__name__,d.build_model.__name__,"
        "d.query_device.__name__,d.run_job.__name__,"
        "int('models.cof_ccmtpp_v1' in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == (
        "load_authorized_train_body build_authorized_model "
        "select_explicit_device run_authorized_job 0"
    )


def test_c1_authorization_is_exactly_hash_bound_and_cannot_expand_scope():
    plan = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
    )
    authorization = build_expected_authorization_claim(
        plan,
        approval={
            "approved": True,
            "text": "Explicit approval for initial C1 external validation only.",
        },
    )
    validated = validate_authorization_document(plan, authorization)
    assert validated["candidate_id"] == "C1"
    assert authorization["dataset_scope"] == "amlsim"
    assert validated["dataset_scope"] == "amlsim"
    assert validated["datasets"] == ["amlsim"]
    assert validated["attempts"] == {"amlsim": "attempt_001"}
    assert validated["authorization_sha256"] == authorization["authorization_sha256"]

    for field, value in (
        ("candidate_id", "C2"),
        ("source_commit", "0" * 40),
        ("runner_config_sha256", "0" * 64),
    ):
        changed = json.loads(json.dumps(authorization))
        changed[field] = value
        with pytest.raises(CCMTPPContractError, match="authorization"):
            validate_authorization_document(plan, changed)

    changed = json.loads(json.dumps(authorization))
    changed["scope"]["internal_test"] = True
    with pytest.raises(CCMTPPContractError, match="scope"):
        validate_authorization_document(plan, changed)


def test_single_dataset_authorization_cannot_cross_dataset_boundary():
    amlsim_plan = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
    )
    sparkov_plan = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="sparkov"
    )
    authorization = build_expected_authorization_claim(
        amlsim_plan,
        approval={
            "approved": True,
            "text": "Explicit approval for AMLSim C1 only.",
        },
    )
    assert authorization["datasets"] == ["amlsim"]
    assert [record["dataset"] for record in authorization["jobs"]] == ["amlsim"]
    assert set(authorization["parent_evidence"]) == {"amlsim"}
    validate_authorization_document(amlsim_plan, authorization)

    with pytest.raises(CCMTPPContractError, match="authorization"):
        validate_authorization_document(sparkov_plan, authorization)


def test_dataset_worker_scope_rejects_cross_dataset_plan_and_data_access(tmp_path):
    amlsim_plan = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
    )
    cross_dataset_plan = replace(
        amlsim_plan,
        jobs=(
            amlsim_plan.jobs[0],
            replace(amlsim_plan.jobs[0], dataset="sparkov"),
        ),
    )
    with pytest.raises(CCMTPPContractError, match="one dataset"):
        execution_plan_report(cross_dataset_plan)

    sparkov = {
        **amlsim_plan.raw["datasets"]["sparkov"],
        "train_path": str(tmp_path / "must-not-be-opened.npz"),
    }
    isolated_plan = replace(
        amlsim_plan,
        raw={
            **amlsim_plan.raw,
            "datasets": {
                **amlsim_plan.raw["datasets"],
                "sparkov": sparkov,
            },
        },
    )
    with pytest.raises(CCMTPPContractError, match="dataset scope"):
        resolve_dataset_access(
            plan=isolated_plan,
            dataset="sparkov",
            split="train",
            purpose="fit",
            fit_complete=False,
        )
    assert not Path(sparkov["train_path"]).exists()


def _write_authorization(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_authorization_path_is_append_only_history_bound_before_any_dependency(tmp_path):
    base = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
    )
    plan = replace(
        base,
        raw={
            **base.raw,
            "authorization_history_root": str(tmp_path / "authorization_history"),
        },
    )
    authorization = build_expected_authorization_claim(
        plan,
        approval={
            "approved": True,
            "text": "Explicit approval for initial C1 external validation only.",
        },
    )
    authorized_path = tmp_path / "authorization_history" / "attempt_001" / "authorization.json"
    _write_authorization(authorized_path, authorization)
    result = validate_authorization_path(plan, authorized_path)
    assert result["status"] == "PASS"
    assert result["authorization_sha256"] == authorization["authorization_sha256"]
    assert result["authorization_file_sha256"] == _sha(authorized_path)

    outside = tmp_path / "outside.json"
    _write_authorization(outside, authorization)
    with pytest.raises(CCMTPPContractError, match="history"):
        validate_authorization_path(plan, outside)


def test_dry_run_is_read_only_and_reports_c1_initial_lock_and_zero_execution():
    c1 = dry_run_execution(
        build_execution_plan(RUNNER_CONFIG, candidate_id="C1", dataset="amlsim")
    )
    assert c1["status"] == "PASS"
    assert c1["candidate_gate"] == "SEPARATE_AUTHORIZATION_REQUIRED"
    assert c1["data_body_files_opened"] == []
    assert c1["runtime_artifact_created"] is False
    assert all(value == 0 for value in c1["execution_counts"].values())
    assert [item["dataset"] for item in c1["dataset_provenance"]] == ["amlsim"]

    c2 = dry_run_execution(
        build_execution_plan(RUNNER_CONFIG, candidate_id="C2", dataset="amlsim")
    )
    assert c2["candidate_gate"] == "LOCKED_PENDING_C1_COMPLETE_AND_PASS"
    assert all(value == 0 for value in c2["execution_counts"].values())


def test_checkpoint_provenance_binds_all_parent_unlock_hashes():
    plan = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
    )
    job = plan.jobs[0]
    state = _train_state_binding(plan)
    provenance = build_checkpoint_provenance(
        plan=plan,
        job=job,
        authorization_sha256="a" * 64,
        train_state_binding=state,
        actual_updates=20_000,
        checkpoint_sha256="b" * 64,
    )
    validate_checkpoint_provenance(
        provenance=provenance,
        plan=plan,
        job=job,
        authorization_sha256="a" * 64,
        train_state_binding=state,
        actual_updates=20_000,
        checkpoint_sha256="b" * 64,
    )
    changed = dict(provenance)
    changed["sampling_plan_sha256"] = "0" * 64
    with pytest.raises(CCMTPPContractError, match="checkpoint provenance"):
        validate_checkpoint_provenance(
            provenance=changed,
            plan=plan,
            job=job,
            authorization_sha256="a" * 64,
            train_state_binding=state,
            actual_updates=20_000,
            checkpoint_sha256="b" * 64,
        )


def test_authorized_execute_uses_owned_spawn_and_terminal_last(tmp_path):
    base = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
    )
    plan = replace(
        base,
        raw={
            **base.raw,
            "runtime_root": str(tmp_path / "runtime"),
            "authorization_history_root": str(tmp_path / "authorization_history"),
        },
    )
    authorization = build_expected_authorization_claim(
        plan,
        approval={
            "approved": True,
            "text": "Explicit approval for initial C1 external validation only.",
        },
    )
    path = tmp_path / "authorization_history" / "attempt_001" / "authorization.json"
    _write_authorization(path, authorization)
    result = execute_authorized(
        plan=plan,
        authorization_path=path,
        dependencies=ExecutionDependencies(
            load_data_body=load_fixture_data,
            build_model=build_fixture_model,
            query_device=query_fixture_device,
            run_job=run_fixture_job,
        ),
    )
    assert result["status"] == "COMPLETE"
    assert len(result["jobs"]) == 1
    assert all(record["status"] == "COMPLETE" for record in result["jobs"])
    for job in plan.jobs:
        attempt = (
            tmp_path / "runtime" / job.dataset / "C1" / "seed_4001" / "attempt_001"
        )
        worker = (
            tmp_path / "runtime" / "workers" / job.dataset / "C1"
            / "attempt_001" / "WORKER_COMPLETE.json"
        )
        assert (attempt / "COMPLETE.json").is_file()
        assert worker.is_file()
        assert (attempt / "COMPLETE.json").stat().st_mtime_ns <= worker.stat().st_mtime_ns


def test_c2_to_c4_authorization_requires_parent_evidence_not_direct_cli_scope():
    for candidate in ("C2", "C3", "C4"):
        plan = build_execution_plan(
            RUNNER_CONFIG, candidate_id=candidate, dataset="amlsim"
        )
        authorization = build_expected_authorization_claim(
            plan,
            approval={
                "approved": True,
                "text": f"Explicit approval for gated {candidate} only.",
            },
        )
        with pytest.raises(CCMTPPContractError, match="parent"):
            validate_authorization_document(plan, authorization)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _parent_gate_fixture(tmp_path: Path, plan, parent_id: str):
    root = tmp_path / "runtime"
    plan = replace(plan, raw={**plan.raw, "runtime_root": str(root)})
    source_hash = plan.raw.get("fixture_source_hash")
    if source_hash is None:
        from experiments.cof_ccmtpp_v1_execution_runner import (
            current_source_commit,
            execution_relevant_source_sha256,
        )
        source_hash = execution_relevant_source_sha256(plan.repository_root)
        source_commit = current_source_commit(plan.repository_root)
    evidence = {}
    for job in plan.jobs:
        attempt = (
            root / job.dataset / parent_id / "seed_4001" / "attempt_001"
        )
        dataset = plan.raw["datasets"][job.dataset]
        provenance = {
            "source_commit": source_commit,
            "runner_relevant_source_sha256": source_hash,
            "source_config_sha256": plan.source_definition.config_sha256,
            "runner_config_sha256": plan.config_sha256,
            "train_sha256": dataset["train_sha256"],
            "train_transform_sha256": dataset["transform_sha256"],
            "threshold_sha256": dataset["threshold_sha256"],
            "sampling_plan_sha256": dataset["sampling_plan_sha256"],
        }
        _json(attempt / "manifest.json", {
            "candidate_id": parent_id,
            "dataset": job.dataset,
            "provenance": provenance,
        })
        _json(attempt / "checkpoint_provenance.json", provenance)
        _json(attempt / "gate_decision.json", {
            "candidate_id": parent_id,
            "status": "PASS",
            "stop_criterion_id": parent_id,
        })
        files = {
            name: _sha(attempt / name)
            for name in (
                "manifest.json",
                "checkpoint_provenance.json",
                "gate_decision.json",
            )
        }
        _json(attempt / "artifact_index.json", {"files": sorted(files)})
        files["artifact_index.json"] = _sha(attempt / "artifact_index.json")
        _json(attempt / "checksum_manifest.json", {"files": files})
        _json(attempt / "COMPLETE.json", {
            "status": "COMPLETE",
            "candidate_id": parent_id,
            "dataset": job.dataset,
            "checksum_manifest_sha256": _sha(attempt / "checksum_manifest.json"),
            "artifact_index_sha256": _sha(attempt / "artifact_index.json"),
        })
        evidence[job.dataset] = {
            "attempt_path": str(attempt),
            "complete_sha256": _sha(attempt / "COMPLETE.json"),
            "checksum_manifest_sha256": _sha(attempt / "checksum_manifest.json"),
            "gate_decision_sha256": _sha(attempt / "gate_decision.json"),
            "checkpoint_provenance_sha256": _sha(
                attempt / "checkpoint_provenance.json"
            ),
        }
    return plan, evidence


def test_parent_complete_gate_and_all_provenance_hashes_unlock_only_next_candidate(tmp_path):
    plan = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C2", dataset="amlsim"
    )
    plan, evidence = _parent_gate_fixture(tmp_path, plan, "C1")

    result = validate_parent_gate_evidence(plan, evidence)
    assert result == {
        "status": "PASS",
        "parent_candidate_id": "C1",
        "datasets": ["amlsim"],
    }

    broken = json.loads(json.dumps(evidence))
    broken["amlsim"]["gate_decision_sha256"] = "0" * 64
    with pytest.raises(CCMTPPContractError, match="parent"):
        validate_parent_gate_evidence(plan, broken)

    wrong_plan = replace(plan, candidate_id="C3")
    with pytest.raises(CCMTPPContractError, match="parent"):
        validate_parent_gate_evidence(wrong_plan, evidence)


def _runtime_hierarchy() -> ReceiverHierarchyState:
    return ReceiverHierarchyState(
        receiver_classes=7,
        pad_code=0,
        unk_code=1,
        head_codes=(2, 3),
        tail_clusters=((4, 6), (5,)),
        counts=(0, 0, 10, 8, 6, 4, 2),
        fit_split="train",
        provenance=(("train_sha256", "a" * 64),),
        state_sha256="b" * 64,
    )


def _candidate_model(candidate: str, *, d_model: int = 128):
    return CoFCCMTPPV1(
        candidate=candidate,
        receiver_classes=7,
        hierarchy=_runtime_hierarchy() if candidate in {"C3", "C4"} else None,
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        max_length=64,
        gap_components=5,
        gap_max=100.0,
        dropout=0.0,
    )


def _train_state_binding(plan, dataset="amlsim"):
    record = plan.raw["datasets"][dataset]
    return {
        "fit_split": "train",
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
        "train_sha256": record["train_sha256"],
        "receiver_vocabulary_sha256": "c" * 64,
        "receiver_hierarchy_sha256": "b" * 64,
        "gap_support_sha256": "d" * 64,
        "gap_support_max": 100.0,
        "amount_transform_sha256": record["transform_sha256"],
        "sampling_plan_sha256": record["sampling_plan_sha256"],
        "threshold_sha256": record["threshold_sha256"],
    }


def _training_hyperparameters(plan):
    model = plan.source_definition.raw["model"]
    return {
        key: model[key]
        for key in (
            "optimizer", "learning_rate", "weight_decay", "batch_size",
            "requested_updates", "max_wall_seconds",
            "checkpoint_interval_updates", "early_stopping", "update_sweep",
        )
    }


def test_runtime_candidate_factor_isolation_and_hyperparameters_are_exact():
    for candidate in ("C1", "C2", "C3", "C4"):
        plan = build_execution_plan(
            RUNNER_CONFIG, candidate_id=candidate, dataset="amlsim"
        )
        report = validate_runtime_candidate(
            plan=plan,
            job=plan.jobs[0],
            model=_candidate_model(candidate),
            train_state_binding=_train_state_binding(plan),
            training_hyperparameters=_training_hyperparameters(plan),
            enabled_hooks=(),
        )
        assert report["candidate_id"] == candidate
        assert report["factor_isolation"] == "PASS"
        assert report["train_only_state_binding"] == "PASS"

    plan = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
    )
    with pytest.raises(CCMTPPContractError, match="hyperparameter"):
        validate_runtime_candidate(
            plan=plan,
            job=plan.jobs[0],
            model=_candidate_model("C1", d_model=64),
            train_state_binding=_train_state_binding(plan),
            training_hyperparameters=_training_hyperparameters(plan),
            enabled_hooks=(),
        )
    with pytest.raises(CCMTPPContractError, match="factor"):
        validate_runtime_candidate(
            plan=plan,
            job=plan.jobs[0],
            model=_candidate_model("C2"),
            train_state_binding=_train_state_binding(plan),
            training_hyperparameters=_training_hyperparameters(plan),
            enabled_hooks=(),
        )
    with pytest.raises(CCMTPPContractError, match="hook"):
        validate_runtime_candidate(
            plan=plan,
            job=plan.jobs[0],
            model=_candidate_model("C1"),
            train_state_binding=_train_state_binding(plan),
            training_hyperparameters=_training_hyperparameters(plan),
            enabled_hooks=("post_hoc_receiver_calibration",),
        )


def test_validation_or_test_rows_cannot_enter_train_state_binding():
    plan = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
    )
    for field in (
        "validation_rows_used",
        "internal_test_rows_used",
        "fraud_test_rows_used",
    ):
        state = _train_state_binding(plan)
        state[field] = 1
        with pytest.raises(CCMTPPContractError, match="train-only"):
            validate_runtime_candidate(
                plan=plan,
                job=plan.jobs[0],
                model=_candidate_model("C1"),
                train_state_binding=state,
                training_hyperparameters=_training_hyperparameters(plan),
                enabled_hooks=(),
            )


class _TrainBatchFixture:
    x_num = np.asarray(
        [[[0.1], [0.2], [0.0]], [[-0.1], [0.3], [0.4]]],
        dtype=np.float32,
    )
    dt_bin = np.asarray([[0, 1, 0], [2, 1, 0]], dtype=np.int64)
    x_cat = np.asarray([[[2], [3], [0]], [[4], [5], [2]]], dtype=np.int64)
    valid_mask = np.asarray(
        [[True, True, False], [True, True, True]], dtype=bool
    )
    y_entity = np.asarray([0, 1], dtype=np.int64)
    lengths = np.asarray([2, 3], dtype=np.int64)

    @property
    def entity_ids(self):
        raise AssertionError("entity identifiers must never be accessed")


def _transform_fixture():
    return {
        "fit_role": "train",
        "pad_code": 0,
        "unk_code": 1,
        "receiver_vocabulary": [
            {"value": f"r{code}", "code": code} for code in range(2, 6)
        ],
        "gap_tau": [0.0, 1.5, 3.0],
        "amount_log_mean": 2.0,
        "amount_log_std": 0.5,
        "state_sha256": "e" * 64,
    }


def test_train_only_state_fit_builds_vocab_hierarchy_gap_and_amount_binding():
    plan = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
    )
    states = fit_train_only_states(
        plan=plan,
        job=plan.jobs[0],
        batch=_TrainBatchFixture(),
        transform_state=_transform_fixture(),
        transform_file_sha256=plan.raw["datasets"]["amlsim"]["transform_sha256"],
        split="train",
    )
    assert states.binding["fit_split"] == "train"
    assert states.binding["validation_rows_used"] == 0
    assert states.binding["internal_test_rows_used"] == 0
    assert states.binding["fraud_test_rows_used"] == 0
    assert states.receiver_vocabulary["receiver_classes"] == 6
    assert states.hierarchy.fit_split == "train"
    assert states.gap_support["max_gap"] == 3.0
    assert np.array_equal(
        states.continuous_gap[states.valid_mask],
        np.asarray([0.0, 1.5, 3.0, 1.5, 0.0]),
    )
    with pytest.raises(CCMTPPContractError, match="train-only"):
        fit_train_only_states(
            plan=plan,
            job=plan.jobs[0],
            batch=_TrainBatchFixture(),
            transform_state=_transform_fixture(),
            transform_file_sha256=plan.raw["datasets"]["amlsim"]["transform_sha256"],
            split="validation",
        )


def test_internal_test_and_fraudtest_are_blocked_before_path_resolution():
    plan = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
    )
    for split in ("internal_test", "fraudTest", "fresh_test"):
        with pytest.raises(CCMTPPContractError, match="forbidden before path resolution"):
            resolve_dataset_access(
                plan=plan,
                dataset="amlsim",
                split=split,
                purpose="fit",
                fit_complete=False,
            )
    with pytest.raises(CCMTPPContractError, match="after fit"):
        resolve_dataset_access(
            plan=plan,
            dataset="amlsim",
            split="validation",
            purpose="evaluation",
            fit_complete=False,
        )


def _complete_attempt_files(
    store: CCMTPPAttemptStore, candidate="C1", *, include_full_tv=True
):
    store.append_progress({"step": 1, "loss": 1.0, "elapsed_seconds": 0.1})
    for name, value in (
        ("receiver_vocabulary_state.json", {"state_sha256": "1" * 64}),
        ("receiver_hierarchy_state.json", {"state_sha256": "2" * 64}),
        ("gap_support_state.json", {"state_sha256": "3" * 64}),
        ("amount_transform_state.json", {"state_sha256": "4" * 64}),
        ("conditioning_plan.json", {"plan_sha256": "5" * 64}),
        ("checkpoint_provenance.json", {"checkpoint_sha256": "6" * 64}),
        ("runtime.json", {"actual_updates": 1, "elapsed_seconds": 0.2}),
        ("metrics.json", {"loss": 0.5}),
        ("evaluation.json", {"status": "VALID"}),
        (
            "diagnostics.json",
            {
                "receiver": {
                    "overall_nll": 1.0,
                    "head_nll": 1.0,
                    "tail_nll": 1.0,
                    "unk_nll": 1.0,
                    "repeat_nll": 1.0,
                    "new_nll": 1.0,
                    "head_count": 1,
                    "tail_count": 1,
                    "unk_count": 0,
                    "repeat_count": 1,
                    "new_count": 1,
                },
                "fidelity": {
                    **({"full_receiver_tv": 0.2} if include_full_tv else {}),
                    "head_receiver_tv": 0.1,
                    "tail_receiver_tv": 0.3,
                    "unk_rate_error": 0.0,
                },
            },
        ),
        (
            "gate_decision.json",
            {"candidate_id": candidate, "status": "PASS", "stop_criterion_id": candidate},
        ),
    ):
        store.write_json(name, value)
    store.write_bytes("checkpoints/final.pt", b"checkpoint")
    store.write_bytes("validation_sample.npz", b"sample")


def test_artifact_store_is_append_only_checksum_bound_and_terminal_last(tmp_path):
    plan = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
    )
    job = plan.jobs[0]
    store = CCMTPPAttemptStore.claim(
        runtime_root=tmp_path / "runtime",
        job=job,
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    attempt = store.allocate_attempt(
        {
            "candidate_id": "C1",
            "dataset": "amlsim",
            "seed": 4001,
            "authorization_sha256": "a" * 64,
            "provenance_sha256": "b" * 64,
        }
    )
    _complete_attempt_files(store)
    terminal = store.finalize("COMPLETE")

    assert terminal["status"] == "COMPLETE"
    assert (attempt / "artifact_index.json").is_file()
    assert (attempt / "checksum_manifest.json").is_file()
    assert (attempt / "COMPLETE.json").is_file()
    assert not (attempt / "INVALID.json").exists()
    assert not (attempt / "FAILED.json").exists()
    assert (attempt / "COMPLETE.json").stat().st_mtime_ns >= (
        attempt / "checksum_manifest.json"
    ).stat().st_mtime_ns
    with pytest.raises(CCMTPPContractError, match="terminal"):
        store.write_json("late.json", {"forbidden": True})
    with pytest.raises(CCMTPPContractError, match="ownership"):
        CCMTPPAttemptStore.claim(
            runtime_root=tmp_path / "runtime",
            job=job,
            authorization_sha256="a" * 64,
            provenance_sha256="b" * 64,
        )


def test_amlsim_and_sparkov_workers_have_independent_ownership_and_attempts(tmp_path):
    runtime = tmp_path / "runtime"
    stores = []
    for dataset, authorization_token in (
        ("amlsim", "a"),
        ("sparkov", "c"),
    ):
        plan = build_execution_plan(
            RUNNER_CONFIG, candidate_id="C1", dataset=dataset
        )
        store = CCMTPPAttemptStore.claim(
            runtime_root=runtime,
            job=plan.jobs[0],
            authorization_sha256=authorization_token * 64,
            provenance_sha256="b" * 64,
        )
        attempt = store.allocate_attempt(
            {
                "candidate_id": "C1",
                "dataset": dataset,
                "seed": 4001,
                "authorization_sha256": authorization_token * 64,
                "provenance_sha256": "b" * 64,
            }
        )
        stores.append((store, attempt))

    assert stores[0][0].ownership_path != stores[1][0].ownership_path
    assert stores[0][1] != stores[1][1]
    assert stores[0][0].ownership_path == (
        runtime / "workers/amlsim/C1/ownership.lock"
    ).resolve()
    assert stores[1][0].ownership_path == (
        runtime / "workers/sparkov/C1/ownership.lock"
    ).resolve()
    assert stores[0][1] == (
        runtime / "amlsim/C1/seed_4001/attempt_001"
    ).resolve()
    assert stores[1][1] == (
        runtime / "sparkov/C1/seed_4001/attempt_001"
    ).resolve()


def test_complete_rejects_missing_receiver_decomposition_or_full_tv(tmp_path):
    plan = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
    )
    store = CCMTPPAttemptStore.claim(
        runtime_root=tmp_path / "runtime",
        job=plan.jobs[0],
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    store.allocate_attempt(
        {
            "candidate_id": "C1",
            "dataset": "amlsim",
            "seed": 4001,
            "authorization_sha256": "a" * 64,
            "provenance_sha256": "b" * 64,
        }
    )
    _complete_attempt_files(store, include_full_tv=False)
    with pytest.raises(CCMTPPContractError, match="diagnostic"):
        store.finalize("COMPLETE")


def _watchdog_store(tmp_path):
    plan = build_execution_plan(
        RUNNER_CONFIG, candidate_id="C1", dataset="amlsim"
    )
    store = CCMTPPAttemptStore.claim(
        runtime_root=tmp_path / "runtime",
        job=plan.jobs[0],
        authorization_sha256="a" * 64,
        provenance_sha256="b" * 64,
    )
    attempt = store.allocate_attempt(
        {
            "candidate_id": "C1",
            "dataset": "amlsim",
            "seed": 4001,
            "authorization_sha256": "a" * 64,
            "provenance_sha256": "b" * 64,
        }
    )
    return store, attempt


def test_startup_watchdog_terminates_owned_child_and_writes_failed_markers(tmp_path):
    store, attempt = _watchdog_store(tmp_path)
    started = time.monotonic()
    result = run_runner_owned_child(
        child_spec=RunnerChildSpec(
            target=_child_never_reports,
            payload={"sleep_seconds": 5.0},
        ),
        store=store,
        max_wall_seconds=1.0,
        startup_timeout_seconds=0.15,
        progress_timeout_seconds=0.2,
        queue_poll_seconds=0.02,
        termination_grace_seconds=0.1,
        kill_grace_seconds=0.1,
    )
    assert time.monotonic() - started < 2.0
    assert result["status"] == "FAILED"
    assert result["failure_class"] == "startup_watchdog"
    failed = json.loads((attempt / "FAILED.json").read_text())
    assert failed["child_terminated"] is True
    assert (store.ownership_path.parent / "attempt_001/WORKER_FAILED.json").is_file()


def test_progress_watchdog_and_child_exception_never_leave_markerless_attempt(tmp_path):
    store, attempt = _watchdog_store(tmp_path / "hang")
    result = run_runner_owned_child(
        child_spec=RunnerChildSpec(
            target=_child_setup_then_hangs,
            payload={"sleep_seconds": 5.0},
        ),
        store=store,
        max_wall_seconds=1.0,
        startup_timeout_seconds=0.2,
        progress_timeout_seconds=0.15,
        queue_poll_seconds=0.02,
        termination_grace_seconds=0.1,
        kill_grace_seconds=0.1,
    )
    assert result["failure_class"] == "progress_watchdog"
    assert (attempt / "FAILED.json").is_file()

    store, attempt = _watchdog_store(tmp_path / "exception")
    result = run_runner_owned_child(
        child_spec=RunnerChildSpec(
            target=_child_raises,
            payload={"message": "fixture boom"},
        ),
        store=store,
        max_wall_seconds=1.0,
        startup_timeout_seconds=0.3,
        progress_timeout_seconds=0.3,
        queue_poll_seconds=0.02,
        termination_grace_seconds=0.1,
        kill_grace_seconds=0.1,
    )
    assert result["failure_class"] == "child_exception"
    assert (attempt / "FAILED.json").is_file()


def test_normal_child_order_and_interrupt_terminal_contract(tmp_path):
    store, attempt = _watchdog_store(tmp_path / "complete")
    result = run_runner_owned_child(
        child_spec=RunnerChildSpec(
            target=_child_completes,
            payload={"attempt_path": str(attempt), "candidate_id": "C1"},
        ),
        store=store,
        max_wall_seconds=2.0,
        startup_timeout_seconds=0.5,
        progress_timeout_seconds=0.5,
        queue_poll_seconds=0.02,
        termination_grace_seconds=0.1,
        kill_grace_seconds=0.1,
    )
    worker = store.ownership_path.parent / "attempt_001/WORKER_COMPLETE.json"
    assert result["status"] == "COMPLETE"
    assert worker.is_file()
    assert (attempt / "progress.jsonl").stat().st_mtime_ns <= (
        attempt / "checkpoints/final.pt"
    ).stat().st_mtime_ns
    assert (attempt / "checkpoints/final.pt").stat().st_mtime_ns <= (
        attempt / "COMPLETE.json"
    ).stat().st_mtime_ns
    assert (attempt / "COMPLETE.json").stat().st_mtime_ns <= worker.stat().st_mtime_ns

    store, attempt = _watchdog_store(tmp_path / "interrupt")
    result = run_runner_owned_child(
        child_spec=RunnerChildSpec(
            target=_child_interrupts,
            payload={"message": "fixture interrupt"},
        ),
        store=store,
        max_wall_seconds=1.0,
        startup_timeout_seconds=0.3,
        progress_timeout_seconds=0.3,
        queue_poll_seconds=0.02,
        termination_grace_seconds=0.1,
        kill_grace_seconds=0.1,
    )
    assert result["failure_class"] == "child_interrupted"
    assert (attempt / "FAILED.json").is_file()
