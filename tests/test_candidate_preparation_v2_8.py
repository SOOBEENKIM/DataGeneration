import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from eval.candidate_preparation_v2_8 import (
    V28PreparationContractError,
    build_v28_fit_state,
    load_v28_definition,
    validate_v28_io_path,
    validate_v28_definition,
    validate_v28_fit_state,
)
from experiments.candidate_preparation_runner_v2_8 import (
    build_v28_plan,
    dry_run_v28,
    v28_plan_report,
)
from scripts.prepare_candidates_v2_8 import parse_args


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = (
    REPOSITORY
    / "configs/benchmark_v2/selection_v2_8_source_amendment.yaml"
)


def _tree_inventory(root: Path):
    if not root.exists():
        return ()
    return tuple(
        (
            path.relative_to(root).as_posix(),
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def test_v28_family_is_three_frozen_controls_and_two_single_factor_candidates():
    definition = load_v28_definition(CONFIG)

    assert len(definition.candidates) == 5
    assert sum(candidate.is_control for candidate in definition.candidates) == 3
    assert [
        (candidate.model_id, candidate.factor)
        for candidate in definition.candidates
        if not candidate.is_control
    ] == [
        (
            "ctgan_separate_class",
            "joint_gap_receiver_discrete_decoder",
        ),
        ("cof_seqgen", "gap_distribution_sampler"),
    ]
    assert all(
        candidate.changed_dimensions == (candidate.factor,)
        for candidate in definition.candidates
        if not candidate.is_control
    )


def test_new_candidate_fit_state_is_train_only_and_binds_parent_provenance():
    definition = load_v28_definition(CONFIG)
    candidate = next(
        candidate
        for candidate in definition.candidates
        if candidate.candidate_id
        == "ctgan_v28_c01_joint_gap_receiver_decoder"
    )

    state = build_v28_fit_state(
        candidate=candidate,
        parameters={"joint_offsets": [[0.25, -0.25]]},
        source_commit="a" * 40,
        config_sha256=definition.config_sha256,
        train_file_sha256="b" * 64,
        train_content_sha256="c" * 64,
        sampling_plan_sha256="d" * 64,
    )

    assert state["fit_split"] == "train"
    assert state["validation_rows_used"] == 0
    assert state["test_rows_used"] == 0
    assert state["factor"] == "joint_gap_receiver_discrete_decoder"
    assert state["parent_provenance"] == {
        "candidate_id": "ctgan_v27_c01_amount_quantile_inverse",
        "checkpoint_sha256": (
            "7e9de7e609ad48e8de687426240fe4fd0a47b5d2d0abcbc70a4edb58f182d10d"
        ),
        "validation_sample_sha256": (
            "f1e0634576d4b2e65de30138f87c8d8999d18f5969a700921f06dac4f23d7a49"
        ),
        "manifest_sha256": (
            "f0be4ee4b59552ef67b08f0677f252576793ab13fd842c1304f22ed86561889f"
        ),
        "source_commit": (
            "de57f79b0b15a9086b6ae26beea25c7f55427a7d"
        ),
    }
    validate_v28_fit_state(
        state=state,
        candidate=candidate,
        source_commit="a" * 40,
        config_sha256=definition.config_sha256,
        train_file_sha256="b" * 64,
        train_content_sha256="c" * 64,
        sampling_plan_sha256="d" * 64,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update(test_split_access="ALLOWED"),
        lambda raw: raw["frozen_contract"]["thresholds"].update(
            gap_ks=0.5
        ),
        lambda raw: raw["models"]["ctgan_separate_class"][
            "candidates"
        ][1]["effective_dimensions"].update(
            amount_inverse_map="second_change"
        ),
    ],
)
def test_v28_frozen_safety_boundary_rejects_forbidden_drift(mutate):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(raw)
    mutate(tampered)

    with pytest.raises(V28PreparationContractError):
        validate_v28_definition(tampered)


def test_v28_plan_uses_existing_checkpoints_with_zero_optimizer_updates():
    report = v28_plan_report(build_v28_plan(CONFIG))

    assert len(report["relevant_source_sha256"]) == 64
    assert report["counts"] == {
        "models": 3,
        "candidates": 5,
        "frozen_parent_controls": 3,
        "future_evaluation_only": 2,
        "new_training_trajectories": 0,
    }
    assert report["execution_contract"] == {
        "checkpoint_access": "READ_ONLY",
        "existing_checkpoint_required": True,
        "optimizer_updates": 0,
        "training_calls": 0,
        "checkpoint_writes": 0,
    }
    assert all(
        operation["optimizer_updates"] == 0
        and operation["training_calls"] == 0
        and operation["checkpoint_writes"] == 0
        and operation["checkpoint_access"] == "READ_ONLY"
        for operation in report["operations"]
    )
    tvae = [
        operation
        for operation in report["operations"]
        if operation["model_id"] == "tvae_separate_class"
    ]
    assert len(tvae) == 1
    assert tvae[0]["execution_kind"] == "reuse_frozen_parent"
    assert tvae[0]["future_sampling_required"] is False


def test_v28_dry_run_verifies_parents_without_runtime_or_frozen_tree_writes():
    v27_root = (
        REPOSITORY / "artifacts/benchmark_v2_7/candidate_selection"
    )
    v28_root = (
        REPOSITORY / "artifacts/benchmark_v2_8/candidate_selection"
    )
    v27_before = _tree_inventory(v27_root)
    v28_before = _tree_inventory(v28_root)

    report = dry_run_v28(build_v28_plan(CONFIG))

    assert report["status"] == "PASS"
    assert report["parent_provenance_verified"] is True
    assert report["parents_verified"] == 3
    assert report["parent_artifacts_verified"] == 24
    assert report["authorization_created"] is False
    assert report["runtime_artifact_created"] is False
    assert report["current_execution_counts"]["gpu_queries"] == 0
    assert report["current_execution_counts"]["optimizer_updates"] == 0
    assert report["current_execution_counts"]["model_sample_calls"] == 0
    assert report["current_execution_counts"]["evaluation_calls"] == 0
    assert report["current_execution_counts"]["test_split_reads"] == 0
    assert _tree_inventory(v27_root) == v27_before
    assert _tree_inventory(v28_root) == v28_before


def test_v28_cli_exposes_only_source_only_plan_and_dry_run():
    assert parse_args(["--mode", "plan"]).mode == "plan"
    assert parse_args(["--mode", "dry-run"]).mode == "dry-run"
    with pytest.raises(SystemExit):
        parse_args(["--mode", "execute"])
    with pytest.raises(SystemExit):
        parse_args(["--mode", "plan", "--authorization", "auth.json"])
    with pytest.raises(SystemExit):
        parse_args(["--mode", "plan", "--device", "cuda:0"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["models"]["ctgan_separate_class"][
            "candidates"
        ][1]["fit_parameters"].update(additive_smoothing=1.0),
        lambda raw: raw["models"]["cof_seqgen"]["candidates"][1].update(
            calibration_components=2
        ),
        lambda raw: raw["future_evaluation_contract"].update(
            optimizer_updates=1
        ),
    ],
)
def test_v28_preregistered_factor_and_zero_update_contract_cannot_drift(
    mutate,
):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(raw)
    mutate(tampered)

    with pytest.raises(V28PreparationContractError):
        validate_v28_definition(tampered)


def test_v28_io_contract_blocks_test_and_parent_writes(tmp_path):
    repository = tmp_path / "repository"
    train = repository / "data/frozen/train.npz"
    validation = repository / "data/frozen/validation.npz"
    test = repository / "data/frozen/test.npz"
    parent = (
        repository
        / "artifacts/benchmark_v2_7/candidate_selection/parent.json"
    )
    future = (
        repository
        / "artifacts/benchmark_v2_8/candidate_selection/attempt_001"
    )
    for path in (train, validation, test, parent):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    assert validate_v28_io_path(
        repository_root=repository,
        path=train,
        access="read",
    ) == train.resolve()
    assert validate_v28_io_path(
        repository_root=repository,
        path=validation,
        access="read",
    ) == validation.resolve()
    with pytest.raises(V28PreparationContractError, match="test"):
        validate_v28_io_path(
            repository_root=repository,
            path=test,
            access="read",
        )
    with pytest.raises(V28PreparationContractError, match="read-only"):
        validate_v28_io_path(
            repository_root=repository,
            path=parent,
            access="write",
        )
    assert validate_v28_io_path(
        repository_root=repository,
        path=future,
        access="write",
    ) == future.resolve()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["fit_state_contract"].update(
            fit_split="validation"
        ),
        lambda raw: raw["models"]["ctgan_separate_class"][
            "parent"
        ].update(candidate_id="ctgan_v27_c02_categorical_logit"),
        lambda raw: raw["models"]["cof_seqgen"]["parent"].update(
            factor="gap_logit_calibration"
        ),
    ],
)
def test_v28_train_only_and_frozen_parent_lineage_cannot_drift(mutate):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(raw)
    mutate(tampered)

    with pytest.raises(V28PreparationContractError):
        validate_v28_definition(tampered)
