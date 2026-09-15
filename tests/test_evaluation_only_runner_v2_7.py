from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.candidate_preparation_v2_7 import (
    FROZEN_SAMPLING_PLAN_SHA256,
    FROZEN_THRESHOLDS,
)
from eval.evaluation_only_v2_7 import evaluate_validation_guards
from experiments.evaluation_only_runner_v2_7 import (
    EvaluationOnlyArtifactStore,
    EvaluationOnlyContractError,
    build_control_reference,
    build_evaluation_execution_plan,
    dry_run_evaluation,
    evaluation_plan_report,
    restore_checkpoint_for_evaluation,
    validate_checkpoint_provenance,
    validate_evaluation_boundary,
    validate_execution_authorization,
    validate_operation_single_factor,
)
from generators.sampling_plan import SamplingPlan
from generators.evaluation_only_transforms_v2_7 import (
    apply_post_sample_intervention,
    configure_backend_for_intervention,
    fit_train_only_intervention,
)
from scripts.run_evaluation_only_v2_7 import parse_args


REPOSITORY = Path(__file__).resolve().parents[1]
RUNNER_CONFIG = (
    REPOSITORY
    / "configs/benchmark_v2/evaluation_only_v2_7.yaml"
)


def _validation_fixture():
    lengths = np.asarray([2, 2, 2, 2], dtype=np.int64)
    mask = np.ones((4, 2), dtype=np.bool_)
    x_num = np.asarray(
        [[[0.0], [0.1]], [[0.1], [0.2]], [[0.2], [0.3]], [[0.3], [0.4]]],
        dtype=np.float32,
    )
    dt_bin = np.asarray(
        [[0, 1], [1, 0], [0, 1], [1, 0]],
        dtype=np.int64,
    )
    x_cat = dt_bin[..., None].copy()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    validation = SequenceBatch(
        x_num=x_num,
        dt_bin=dt_bin,
        x_cat=x_cat,
        valid_mask=mask,
        y_entity=labels,
        lengths=lengths,
        entity_ids=np.asarray(["a", "b", "c", "d"]),
    )
    sample = SyntheticBatch(
        x_num=x_num.copy(),
        dt_bin=dt_bin.copy(),
        x_cat=x_cat.copy(),
        valid_mask=mask.copy(),
        y_entity=labels.copy(),
        lengths=lengths.copy(),
    )
    plan = SamplingPlan(
        y_entity=labels.copy(),
        lengths=lengths.copy(),
        valid_mask=mask.copy(),
        plan_hash=FROZEN_SAMPLING_PLAN_SHA256,
    )
    return validation, sample, plan


def test_plan_reuses_three_controls_and_six_checkpoint_only_candidates():
    plan = build_evaluation_execution_plan(RUNNER_CONFIG)

    assert len(plan.operations) == 9
    assert sum(operation.is_control for operation in plan.operations) == 3
    assert sum(
        operation.sampling_required for operation in plan.operations
    ) == 6
    assert all(
        operation.training_required is False
        for operation in plan.operations
    )
    assert len(
        {
            (operation.model_id, operation.checkpoint_sha256)
            for operation in plan.operations
        }
    ) == 3


def test_checkpoint_provenance_mismatch_fails_before_execution():
    plan = build_evaluation_execution_plan(RUNNER_CONFIG)
    operation = next(
        operation
        for operation in plan.operations
        if operation.candidate_id
        == "ctgan_v27_c01_amount_quantile_inverse"
    )

    validate_checkpoint_provenance(plan, operation)
    with pytest.raises(
        EvaluationOnlyContractError,
        match="checkpoint provenance",
    ):
        validate_checkpoint_provenance(
            plan,
            replace(operation, checkpoint_sha256="0" * 64),
        )


def test_control_references_frozen_bytes_without_new_sample():
    plan = build_evaluation_execution_plan(RUNNER_CONFIG)

    for operation in plan.operations:
        if not operation.is_control:
            continue
        reference = build_control_reference(
            plan=plan,
            operation=operation,
            source_commit="a" * 40,
            relevant_source_sha256="b" * 64,
        )
        assert reference["training_calls"] == 0
        assert reference["optimizer_updates"] == 0
        assert reference["sampling_calls"] == 0
        assert reference["fit_state_created"] is False
        assert "new_validation_sample_path" not in reference
        assert (
            reference["frozen_control"]["checkpoint_sha256"]
            == operation.checkpoint_sha256
        )


def test_six_evaluation_operations_reject_factor_drift():
    plan = build_evaluation_execution_plan(RUNNER_CONFIG)
    operations = [
        operation
        for operation in plan.operations
        if not operation.is_control
    ]

    assert len(operations) == 6
    for operation in operations:
        validate_operation_single_factor(plan, operation)
    with pytest.raises(EvaluationOnlyContractError, match="single factor"):
        validate_operation_single_factor(
            plan,
            replace(operations[0], factor="categorical_logit_calibration"),
        )


def test_restore_delegates_to_frozen_cpu_first_or_cof_device_contract():
    plan = build_evaluation_execution_plan(RUNNER_CONFIG)
    calls = []

    def loader(path, *, model_id, device, train):
        calls.append((path, model_id, device, train))
        return {"backend": f"{model_id}-backend"}

    for model_id in (
        "ctgan_separate_class",
        "tvae_separate_class",
        "cof_seqgen",
    ):
        operation = next(
            operation
            for operation in plan.operations
            if operation.model_id == model_id and not operation.is_control
        )
        restored = restore_checkpoint_for_evaluation(
            plan=plan,
            operation=operation,
            device="cuda:7",
            train="train-fixture",
            loader=loader,
        )
        assert restored["training_calls"] == 0
        assert restored["optimizer_updates"] == 0
        assert restored["restore_mode"] == operation.restore_mode

    assert [call[1] for call in calls] == [
        "ctgan_separate_class",
        "tvae_separate_class",
        "cof_seqgen",
    ]


def test_validation_guards_use_frozen_thresholds_and_sampling_plan():
    validation, sample, sampling_plan = _validation_fixture()

    result = evaluate_validation_guards(
        train=validation,
        validation=validation,
        sample=sample,
        sampling_plan=sampling_plan,
        tau=np.asarray([0.5, 2.0]),
        receiver_categories=2,
        thresholds=FROZEN_THRESHOLDS,
    )

    assert result["thresholds"] == FROZEN_THRESHOLDS
    assert result["sampling_plan_sha256"] == FROZEN_SAMPLING_PLAN_SHA256
    assert set(result["checks"]) == set(FROZEN_THRESHOLDS)
    assert result["validation_selection_executed"] is False
    assert result["test_split_read"] is False


def test_test_threshold_and_result_based_selection_requests_fail_closed():
    plan = build_evaluation_execution_plan(RUNNER_CONFIG)
    train_path = (
        REPOSITORY
        / "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/"
        "kappa_1.00/train.npz"
    )
    validation_path = train_path.with_name("validation.npz")
    test_path = train_path.with_name("test.npz")

    validate_evaluation_boundary(
        plan=plan,
        data_paths=(train_path, validation_path),
        thresholds=FROZEN_THRESHOLDS,
        sampling_plan_sha256=FROZEN_SAMPLING_PLAN_SHA256,
        selection_requested=False,
        candidate_overrides={},
    )
    rejected = (
        {
            "data_paths": (train_path, test_path),
            "thresholds": FROZEN_THRESHOLDS,
            "selection_requested": False,
            "candidate_overrides": {},
        },
        {
            "data_paths": (train_path, validation_path),
            "thresholds": {**FROZEN_THRESHOLDS, "amount_ks": 1.0},
            "selection_requested": False,
            "candidate_overrides": {},
        },
        {
            "data_paths": (train_path, validation_path),
            "thresholds": FROZEN_THRESHOLDS,
            "selection_requested": True,
            "candidate_overrides": {},
        },
        {
            "data_paths": (train_path, validation_path),
            "thresholds": FROZEN_THRESHOLDS,
            "selection_requested": False,
            "candidate_overrides": {"choose_after_results": True},
        },
    )
    for request in rejected:
        with pytest.raises(EvaluationOnlyContractError):
            validate_evaluation_boundary(
                plan=plan,
                sampling_plan_sha256=FROZEN_SAMPLING_PLAN_SHA256,
                **request,
            )


def test_evaluation_store_is_append_only_with_exclusive_model_owner(
    tmp_path,
):
    repository = tmp_path / "repository"
    repository.mkdir()
    store = EvaluationOnlyArtifactStore(repository)

    ctgan_owner = store.claim_model(
        model_id="ctgan_separate_class",
        provenance_sha256="a" * 64,
    )
    tvae_owner = store.claim_model(
        model_id="tvae_separate_class",
        provenance_sha256="a" * 64,
    )
    assert ctgan_owner != tvae_owner
    with pytest.raises(EvaluationOnlyContractError, match="owned"):
        store.claim_model(
            model_id="ctgan_separate_class",
            provenance_sha256="a" * 64,
        )

    attempt = store.create_evaluation_attempt(
        model_id="ctgan_separate_class",
        candidate_id="ctgan_v27_c01_amount_quantile_inverse",
        seed=2601,
    )
    store.write_json(attempt / "manifest.json", {"status": "RUNNING"})
    with pytest.raises(EvaluationOnlyContractError, match="exists"):
        store.write_json(attempt / "manifest.json", {"status": "CHANGED"})


def test_execute_cli_requires_authorization_model_and_device():
    assert parse_args(["--mode", "plan"]).mode == "plan"
    assert parse_args(["--mode", "dry-run"]).mode == "dry-run"
    with pytest.raises(SystemExit):
        parse_args(["--mode", "execute"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "execute",
                "--model",
                "ctgan_separate_class",
                "--device",
                "cuda:0",
            ]
        )


def test_authorization_must_pin_source_config_operations_and_checkpoints():
    plan = build_evaluation_execution_plan(RUNNER_CONFIG)
    authorization = {
        "schema_version": (
            "benchmark-v2.7-evaluation-execution-authorization-v1"
        ),
        "status": "AUTHORIZED",
        "source_commit": "a" * 40,
        "relevant_source_sha256": "b" * 64,
        "runner_config_sha256": plan.runner_config_sha256,
        "candidate_config_sha256": plan.candidate_plan.config_sha256,
        "development_manifest_sha256": (
            plan.candidate_plan.frozen_contract[
                "development_manifest_sha256"
            ]
        ),
        "train_file_sha256": plan.candidate_plan.frozen_contract[
            "train_file_sha256"
        ],
        "train_content_sha256": plan.candidate_plan.frozen_contract[
            "train_content_sha256"
        ],
        "validation_file_sha256": (
            plan.candidate_plan.frozen_contract[
                "validation_file_sha256"
            ]
        ),
        "validation_content_sha256": (
            plan.candidate_plan.frozen_contract[
                "validation_content_sha256"
            ]
        ),
        "sampling_plan_sha256": FROZEN_SAMPLING_PLAN_SHA256,
        "runtime_root": (
            "artifacts/benchmark_v2_7/candidate_selection"
        ),
        "models": list(plan.candidate_plan.model_ids),
        "candidate_ids": [
            operation.candidate_id for operation in plan.operations
        ],
        "checkpoint_hashes": {
            operation.model_id: operation.checkpoint_sha256
            for operation in plan.operations
        },
        "counts": {
            "controls": 3,
            "evaluation_only_candidates": 6,
            "training_trajectories": 0,
        },
        "authorization": {
            "control_reference": True,
            "evaluation_only_sampling": True,
            "training": False,
            "optimizer_updates": False,
            "checkpoint_retraining": False,
            "validation_selection": False,
            "test_split_access": False,
            "fresh_test": False,
            "tstr": False,
            "privacy": False,
            "five_seed_full_run": False,
        },
    }

    validate_execution_authorization(
        plan=plan,
        authorization=authorization,
        source_commit="a" * 40,
        relevant_source_sha256="b" * 64,
    )
    tampered = {
        **authorization,
        "checkpoint_hashes": {
            **authorization["checkpoint_hashes"],
            "ctgan_separate_class": "0" * 64,
        },
    }
    with pytest.raises(EvaluationOnlyContractError, match="authorization"):
        validate_execution_authorization(
            plan=plan,
            authorization=tampered,
            source_commit="a" * 40,
            relevant_source_sha256="b" * 64,
        )
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "plan",
                "--authorization",
                "future.json",
            ]
        )


def test_six_train_only_fit_states_are_hashed_and_change_one_channel():
    plan = build_evaluation_execution_plan(RUNNER_CONFIG)
    train, base, _ = _validation_fixture()
    temperature_samples = {
        f"{temperature:g}": base
        for temperature in (0.5, 0.625, 0.75, 0.875, 1.0)
    }
    candidates = {
        candidate.candidate_id: candidate
        for candidate in plan.candidate_plan.candidates
    }

    for operation in plan.operations:
        if operation.is_control:
            continue
        calibration = (
            temperature_samples
            if operation.candidate_id
            == "tvae_v27_c02_bounded_temperature"
            else {"base": base}
        )
        state = fit_train_only_intervention(
            candidate=candidates[operation.candidate_id],
            train=train,
            calibration_samples=calibration,
            source_commit="a" * 40,
            config_sha256=plan.candidate_plan.config_sha256,
            train_file_sha256=(
                plan.candidate_plan.frozen_contract[
                    "train_file_sha256"
                ]
            ),
            train_content_sha256=(
                plan.candidate_plan.frozen_contract[
                    "train_content_sha256"
                ]
            ),
            sampling_plan_sha256=FROZEN_SAMPLING_PLAN_SHA256,
        )
        assert state["fit_split"] == "train"
        assert state["validation_rows_used"] == 0
        assert state["test_rows_used"] == 0
        assert len(state["state_sha256"]) == 64
        adjusted = apply_post_sample_intervention(
            candidate=candidates[operation.candidate_id],
            sample=base,
            fit_state=state,
        )
        changed = {
            "amount": not np.array_equal(adjusted.x_num, base.x_num),
            "gap": not np.array_equal(adjusted.dt_bin, base.dt_bin),
            "receiver": not np.array_equal(adjusted.x_cat, base.x_cat),
        }
        if operation.factor in {
            "amount_inverse_map",
            "amount_inverse_decoder",
            "amount_residual_sampler",
        }:
            assert changed["gap"] is False
            assert changed["receiver"] is False
        else:
            # Logit/temperature factors are installed before sampling;
            # the post-sampling layer must not introduce a second factor.
            assert changed == {
                "amount": False,
                "gap": False,
                "receiver": False,
            }


def test_plan_and_dry_run_are_read_only_and_make_zero_model_calls(
    monkeypatch,
):
    plan = build_evaluation_execution_plan(RUNNER_CONFIG)
    runtime_existed = plan.runtime_root.exists()
    calls = {
        "restore": 0,
        "fit": 0,
        "sample": 0,
        "cuda": 0,
        "gpu_query": 0,
    }

    def forbidden(*args, **kwargs):
        calls["restore"] += 1
        raise AssertionError("plan/dry-run touched model execution")

    monkeypatch.setattr(
        "experiments.evaluation_only_runner_v2_7."
        "restore_checkpoint_for_evaluation",
        forbidden,
    )
    report = evaluation_plan_report(plan)
    dry_run = dry_run_evaluation(plan)

    assert report["counts"] == {
        "models": 3,
        "candidates": 9,
        "frozen_controls": 3,
        "evaluation_only_candidates": 6,
        "training_trajectories": 0,
    }
    assert dry_run["status"] == "PASS"
    assert dry_run["frozen_provenance_verified"] is True
    assert dry_run["authorization_created"] is False
    assert dry_run["runtime_artifact_created"] is False
    assert dry_run["execution_counts"] == {
        "gpu_queries": 0,
        "cuda_calls": 0,
        "model_fit_calls": 0,
        "model_sample_calls": 0,
        "optimizer_updates": 0,
        "validation_selection_runs": 0,
        "fresh_test_runs": 0,
        "five_seed_full_runs": 0,
    }
    assert calls == {
        "restore": 0,
        "fit": 0,
        "sample": 0,
        "cuda": 0,
        "gpu_query": 0,
    }
    assert plan.runtime_root.exists() is runtime_existed


def test_discrete_interventions_configure_only_the_frozen_sampling_path():
    plan = build_evaluation_execution_plan(RUNNER_CONFIG)
    train, base, _ = _validation_fixture()
    candidates = {
        candidate.candidate_id: candidate
        for candidate in plan.candidate_plan.candidates
    }

    class TabularSynthesizer:
        def __init__(self):
            self.v2_6_channel_weights = {
                "amount": 1.0,
                "gap": 1.0,
                "receiver": 1.0,
            }
            self.v2_6_latent_scale = 1.0
            self.calls = []

        def set_candidate_contract(self, **values):
            self.calls.append(values)

    class Backend:
        def __init__(self):
            self.models = {0: TabularSynthesizer(), 1: TabularSynthesizer()}

    provenance = {
        "source_commit": "a" * 40,
        "config_sha256": plan.candidate_plan.config_sha256,
        "train_file_sha256": (
            plan.candidate_plan.frozen_contract["train_file_sha256"]
        ),
        "train_content_sha256": (
            plan.candidate_plan.frozen_contract[
                "train_content_sha256"
            ]
        ),
        "sampling_plan_sha256": FROZEN_SAMPLING_PLAN_SHA256,
    }
    ctgan_candidate = candidates["ctgan_v27_c02_categorical_logit"]
    ctgan_state = fit_train_only_intervention(
        candidate=ctgan_candidate,
        train=train,
        calibration_samples={"base": base},
        **provenance,
    )
    ctgan = Backend()
    configured = configure_backend_for_intervention(
        backend=ctgan,
        candidate=ctgan_candidate,
        fit_state=ctgan_state,
    )
    assert configured["changed_factor"] == "categorical_logit_calibration"
    assert all(
        hasattr(model, "_v27_categorical_logit_offsets")
        for model in ctgan.models.values()
    )

    tvae_candidate = candidates["tvae_v27_c02_bounded_temperature"]
    tvae_state = fit_train_only_intervention(
        candidate=tvae_candidate,
        train=train,
        calibration_samples={
            f"{value:g}": base
            for value in (0.5, 0.625, 0.75, 0.875, 1.0)
        },
        **provenance,
    )
    tvae = Backend()
    configured = configure_backend_for_intervention(
        backend=tvae,
        candidate=tvae_candidate,
        fit_state=tvae_state,
    )
    assert configured["selected_temperature"] == 0.75
    assert all(len(model.calls) == 1 for model in tvae.models.values())

    cof_candidate = candidates["cof_v27_c02_gap_logit_bias"]
    cof_state = fit_train_only_intervention(
        candidate=cof_candidate,
        train=train,
        calibration_samples={"base": base},
        **provenance,
    )
    cof = type("CoFBackend", (), {"sampling_function": object()})()
    configured = configure_backend_for_intervention(
        backend=cof,
        candidate=cof_candidate,
        fit_state=cof_state,
    )
    assert configured["changed_factor"] == "gap_logit_calibration"
    assert callable(cof.sampling_function)
