from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.candidate_preparation_v2_8 import load_v28_definition
from experiments.evaluation_only_runner_v2_8 import (
    EvaluationOnlyArtifactStore,
    EvaluationOnlyContractError,
    build_evaluation_execution_plan,
    dry_run_evaluation,
    expected_execution_authorization,
    execute_model_worker,
    restore_checkpoint_for_evaluation,
    validate_evaluation_boundary,
    validate_execution_authorization,
)
from scripts.run_evaluation_only_v2_8 import parse_args
from generators.evaluation_only_transforms_v2_8 import (
    apply_post_sample_intervention,
    fit_train_only_intervention,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = (
    REPOSITORY
    / "configs/benchmark_v2/evaluation_only_v2_8.yaml"
)


def test_v28_execution_plan_has_only_two_future_evaluations():
    plan = build_evaluation_execution_plan(CONFIG)

    assert [
        (operation.model_id, operation.candidate_id)
        for operation in plan.operations
    ] == [
        (
            "ctgan_separate_class",
            "ctgan_v28_c01_joint_gap_receiver_decoder",
        ),
        ("cof_seqgen", "cof_v28_c01_gap_distribution_sampler"),
    ]
    assert [reference.candidate_id for reference in plan.references] == [
        "ctgan_v28_c00_frozen_amount_inverse",
        "tvae_v28_c00_frozen_selected_amount_inverse",
        "cof_v28_c00_frozen_empirical_residual",
    ]
    assert plan.training_trajectories == ()
    assert all(operation.optimizer_updates == 0 for operation in plan.operations)


def test_v28_execute_authorization_is_exact_and_fail_closed():
    plan = build_evaluation_execution_plan(CONFIG)
    authorization = expected_execution_authorization(
        plan=plan,
        source_commit="a" * 40,
        relevant_source_sha256="b" * 64,
    )

    validate_execution_authorization(
        plan=plan,
        authorization=authorization,
        source_commit="a" * 40,
        relevant_source_sha256="b" * 64,
    )
    authorization["authorization"]["training"] = True
    with pytest.raises(EvaluationOnlyContractError):
        validate_execution_authorization(
            plan=plan,
            authorization=authorization,
            source_commit="a" * 40,
            relevant_source_sha256="b" * 64,
        )


def _batches():
    mask = np.ones((4, 4), dtype=np.bool_)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    lengths = np.full(4, 4, dtype=np.int64)
    train = SequenceBatch(
        x_num=np.arange(16, dtype=np.float32).reshape(4, 4, 1),
        dt_bin=np.array(
            [
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 2, 2, 2],
                [1, 1, 2, 2],
            ],
            dtype=np.int64,
        ),
        x_cat=np.array(
            [
                [[0], [0], [1], [1]],
                [[0], [1], [1], [1]],
                [[1], [2], [2], [2]],
                [[1], [1], [2], [2]],
            ],
            dtype=np.int64,
        ),
        valid_mask=mask,
        y_entity=labels,
        lengths=lengths,
        entity_ids=np.arange(4, dtype=np.int64),
    )
    sample = SyntheticBatch(
        x_num=train.x_num.copy(),
        dt_bin=np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 2],
                [1, 1, 2, 2],
            ],
            dtype=np.int64,
        ),
        x_cat=np.array(
            [
                [[0], [0], [0], [1]],
                [[0], [0], [1], [1]],
                [[1], [1], [1], [2]],
                [[1], [1], [2], [2]],
            ],
            dtype=np.int64,
        ),
        valid_mask=mask.copy(),
        y_entity=labels.copy(),
        lengths=lengths.copy(),
    )
    return train, sample


@pytest.mark.parametrize(
    ("candidate_id", "changed_channels"),
    [
        (
            "ctgan_v28_c01_joint_gap_receiver_decoder",
            {"gap", "receiver"},
        ),
        ("cof_v28_c01_gap_distribution_sampler", {"gap"}),
    ],
)
def test_v28_train_only_interventions_change_exactly_one_factor(
    candidate_id,
    changed_channels,
):
    plan = build_evaluation_execution_plan(CONFIG)
    candidate = next(
        value
        for value in plan.definition.candidates
        if value.candidate_id == candidate_id
    )
    train, sample = _batches()
    state = fit_train_only_intervention(
        candidate=candidate,
        train=train,
        calibration_sample=sample,
        source_commit="a" * 40,
        config_sha256=plan.candidate_config_sha256,
        train_file_sha256="b" * 64,
        train_content_sha256="c" * 64,
        sampling_plan_sha256="d" * 64,
    )
    adjusted = apply_post_sample_intervention(
        candidate=candidate,
        sample=sample,
        fit_state=state,
    )

    assert state["fit_split"] == "train"
    assert state["validation_rows_used"] == 0
    assert state["test_rows_used"] == 0
    assert state["parameters"]["changed_factor"] == candidate.factor
    assert np.array_equal(adjusted.x_num, sample.x_num)
    if "receiver" not in changed_channels:
        assert np.array_equal(adjusted.x_cat, sample.x_cat)
    assert (
        np.array_equal(adjusted.dt_bin, sample.dt_bin)
        is ("gap" not in changed_channels)
    )


def test_v28_checkpoint_restore_preserves_device_contract_without_training():
    plan = build_evaluation_execution_plan(CONFIG)
    calls = []

    def loader(path, *, model_id, device, train):
        calls.append((path, model_id, device, train))
        return {"backend": object()}

    for operation in plan.operations:
        restored = restore_checkpoint_for_evaluation(
            plan=plan,
            operation=operation,
            device="cuda:7",
            train="fixture-train",
            loader=loader,
        )
        assert restored["restore_mode"] == operation.restore_mode
        assert restored["training_calls"] == 0
        assert restored["optimizer_updates"] == 0
        assert restored["checkpoint_writes"] == 0

    assert [call[1:] for call in calls] == [
        ("ctgan_separate_class", "cuda:7", "fixture-train"),
        ("cof_seqgen", "cuda:7", "fixture-train"),
    ]

    with pytest.raises(EvaluationOnlyContractError, match="checkpoint"):
        restore_checkpoint_for_evaluation(
            plan=plan,
            operation=replace(
                plan.operations[0],
                checkpoint_sha256="0" * 64,
            ),
            device="cuda:7",
            train="fixture-train",
            loader=loader,
        )
    assert len(calls) == 2


def test_v28_boundary_allows_only_frozen_train_validation_and_guard_contract():
    plan = build_evaluation_execution_plan(CONFIG)
    data_root = (
        REPOSITORY
        / "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00"
    )
    validate_evaluation_boundary(
        plan=plan,
        data_paths=(data_root / "train.npz", data_root / "validation.npz"),
        thresholds=plan.validation_contract["thresholds"],
        sampling_plan_sha256=plan.validation_contract[
            "sampling_plan_sha256"
        ],
        selection_requested=False,
        candidate_overrides={},
    )
    with pytest.raises(EvaluationOnlyContractError):
        validate_evaluation_boundary(
            plan=plan,
            data_paths=(data_root / "train.npz", data_root / "test.npz"),
            thresholds=plan.validation_contract["thresholds"],
            sampling_plan_sha256=plan.validation_contract[
                "sampling_plan_sha256"
            ],
            selection_requested=False,
            candidate_overrides={},
        )


def test_v28_dry_run_hashes_inputs_without_execute_side_effects():
    before = (
        tuple(REPOSITORY.glob("artifacts/benchmark_v2_8/**/*"))
        if (REPOSITORY / "artifacts/benchmark_v2_8").exists()
        else ()
    )
    report = dry_run_evaluation(build_evaluation_execution_plan(CONFIG))

    assert report["status"] == "PASS"
    assert report["counts"] == {
        "models": 2,
        "frozen_references": 3,
        "evaluation_only_candidates": 2,
        "training_trajectories": 0,
    }
    assert report["checkpoint_hashes_verified"] == {
        operation.model_id: operation.checkpoint_sha256
        for operation in build_evaluation_execution_plan(CONFIG).operations
    }
    assert report["execution_counts"] == {
        "gpu_queries": 0,
        "cuda_calls": 0,
        "model_fit_calls": 0,
        "model_sample_calls": 0,
        "optimizer_updates": 0,
        "evaluation_calls": 0,
        "selection_calls": 0,
        "test_split_reads": 0,
    }
    after = (
        tuple(REPOSITORY.glob("artifacts/benchmark_v2_8/**/*"))
        if (REPOSITORY / "artifacts/benchmark_v2_8").exists()
        else ()
    )
    assert after == before


def test_v28_artifact_store_is_append_only_and_model_owned(tmp_path):
    store = EvaluationOnlyArtifactStore(tmp_path)
    store.claim_model(
        model_id="ctgan_separate_class",
        provenance_sha256="a" * 64,
    )
    attempt = store.create_evaluation_attempt(
        model_id="ctgan_separate_class",
        candidate_id="ctgan_v28_c01_joint_gap_receiver_decoder",
        seed=2801,
    )
    store.write_json(attempt / "manifest.json", {"status": "RUNNING"})

    with pytest.raises(EvaluationOnlyContractError):
        store.claim_model(
            model_id="ctgan_separate_class",
            provenance_sha256="a" * 64,
        )
    with pytest.raises(EvaluationOnlyContractError):
        store.write_json(attempt / "manifest.json", {"status": "CHANGED"})
    with pytest.raises(EvaluationOnlyContractError):
        store.create_evaluation_attempt(
            model_id="cof_seqgen",
            candidate_id="cof_v28_c01_gap_distribution_sampler",
            seed=2801,
        )


def test_v28_cli_requires_authorization_only_for_execute():
    assert parse_args(["--mode", "plan"]).mode == "plan"
    assert parse_args(["--mode", "dry-run"]).mode == "dry-run"
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
    with pytest.raises(SystemExit):
        parse_args(
            ["--mode", "plan", "--authorization", "authorization.json"]
        )
    with pytest.raises(SystemExit):
        parse_args(["--mode", "execute", "--model", "tvae_separate_class"])


def test_v28_execute_rejects_missing_authorization_before_gpu_or_model_calls(
    tmp_path,
):
    with pytest.raises(EvaluationOnlyContractError, match="authorization"):
        execute_model_worker(
            plan=build_evaluation_execution_plan(CONFIG),
            authorization_path=tmp_path / "missing.json",
            model_id="ctgan_separate_class",
            device="cuda:0",
        )
