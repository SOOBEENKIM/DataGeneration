import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from eval.candidate_preparation_v2_7 import (
    AppendOnlyV27Store,
    V27PreparationContractError,
    build_fit_state_record,
    load_and_validate_preparation,
    validate_candidate_io_path,
    validate_fit_state_record,
    validate_preparation_definition,
)
from experiments.candidate_preparation_runner_v2_7 import (
    build_preparation_execution_plan,
    dry_run_preparation,
    preparation_plan_report,
)
from scripts.prepare_candidates_v2_7 import parse_args


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = (
    REPOSITORY
    / "configs/benchmark_v2/selection_v2_7_source_preparation.yaml"
)


def _read_only_tree_inventory(root: Path):
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


def test_plan_has_three_controls_and_six_exactly_single_factor_candidates():
    plan = load_and_validate_preparation(CONFIG)

    assert len(plan.candidates) == 9
    assert sum(candidate.is_control for candidate in plan.candidates) == 3
    for candidate in plan.candidates:
        expected = () if candidate.is_control else (candidate.factor,)
        assert candidate.changed_dimensions == expected


def test_fit_state_records_train_only_hash_and_provenance():
    plan = load_and_validate_preparation(CONFIG)
    candidate = next(
        candidate
        for candidate in plan.candidates
        if candidate.candidate_id
        == "ctgan_v27_c01_amount_quantile_inverse"
    )

    record = build_fit_state_record(
        candidate=candidate,
        parameters={"quantiles": [0.0, 0.5, 1.0]},
        source_commit="a" * 40,
        config_sha256=plan.config_sha256,
        train_file_sha256="b" * 64,
        train_content_sha256="c" * 64,
        sampling_plan_sha256=plan.sampling_plan_sha256,
        destination="candidate_artifact",
    )

    assert record["fit_split"] == "train"
    assert record["validation_rows_used"] == 0
    assert record["test_rows_used"] == 0
    assert record["candidate_id"] == candidate.candidate_id
    assert record["factor"] == "amount_inverse_map"
    assert record["state_sha256"]
    assert record["provenance"] == {
        "source_commit": "a" * 40,
        "config_sha256": plan.config_sha256,
        "train_file_sha256": "b" * 64,
        "train_content_sha256": "c" * 64,
        "sampling_plan_sha256": plan.sampling_plan_sha256,
    }


def test_test_paths_fail_closed_and_frozen_trees_are_read_only(tmp_path):
    repository = tmp_path / "repository"
    train = repository / "data/benchmark_v2_6/development/train.npz"
    validation = (
        repository / "data/benchmark_v2_6/development/validation.npz"
    )
    test = repository / "data/benchmark_v2_6/development/test.npz"
    for path in (train, validation, test):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    assert validate_candidate_io_path(
        repository_root=repository,
        path=train,
        access="read",
    ) == train.resolve()
    assert validate_candidate_io_path(
        repository_root=repository,
        path=validation,
        access="read",
    ) == validation.resolve()
    with pytest.raises(V27PreparationContractError, match="test"):
        validate_candidate_io_path(
            repository_root=repository,
            path=test,
            access="read",
        )
    with pytest.raises(V27PreparationContractError, match="read-only"):
        validate_candidate_io_path(
            repository_root=repository,
            path=repository / "artifacts/benchmark_v2_6/result.json",
            access="write",
        )


def test_worker_ownership_and_candidate_artifacts_are_exclusive(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    store = AppendOnlyV27Store(repository)

    ctgan_claim = store.claim_worker(
        model_id="ctgan_separate_class",
        provenance_sha256="a" * 64,
    )
    tvae_claim = store.claim_worker(
        model_id="tvae_separate_class",
        provenance_sha256="a" * 64,
    )
    assert ctgan_claim != tvae_claim
    with pytest.raises(V27PreparationContractError, match="owned"):
        store.claim_worker(
            model_id="ctgan_separate_class",
            provenance_sha256="a" * 64,
        )

    attempt = store.create_candidate_attempt(
        model_id="ctgan_separate_class",
        candidate_id="ctgan_v27_c01_amount_quantile_inverse",
        seed=2701,
    )
    store.write_json(attempt / "manifest.json", {"status": "PLANNED"})
    with pytest.raises(V27PreparationContractError, match="exists"):
        store.write_json(attempt / "manifest.json", {"status": "CHANGED"})


def test_plan_is_source_only_with_zero_training_and_three_frozen_controls():
    plan = build_preparation_execution_plan(CONFIG)
    report = preparation_plan_report(plan)

    assert report["counts"] == {
        "models": 3,
        "candidates": 9,
        "frozen_controls": 3,
        "future_evaluation_only": 6,
        "new_training_trajectories": 0,
    }
    assert report["current_execution_counts"] == {
        "gpu_queries": 0,
        "cuda_calls": 0,
        "candidate_training_calls": 0,
        "model_fit_calls": 0,
        "model_sample_calls": 0,
        "validation_selection_calls": 0,
        "test_split_reads": 0,
        "fresh_test_calls": 0,
        "tstr_calls": 0,
        "privacy_calls": 0,
        "five_seed_full_run_calls": 0,
    }
    assert report["authorization_created"] is False
    assert all(
        operation["future_training_required"] is False
        for operation in report["operations"]
    )


def test_dry_run_verifies_frozen_provenance_without_runtime_writes():
    plan = build_preparation_execution_plan(CONFIG)
    runtime_root = (
        REPOSITORY / "artifacts/benchmark_v2_7/candidate_selection"
    )
    before = _read_only_tree_inventory(runtime_root)

    report = dry_run_preparation(plan)

    assert report["status"] == "PASS"
    assert report["frozen_provenance_verified"] is True
    assert report["fit_state_created"] is False
    assert report["authorization_created"] is False
    assert report["current_execution_counts"]["gpu_queries"] == 0
    assert report["current_execution_counts"]["model_fit_calls"] == 0
    assert report["current_execution_counts"]["model_sample_calls"] == 0
    assert report["current_execution_counts"]["test_split_reads"] == 0
    assert _read_only_tree_inventory(runtime_root) == before


def test_preregistered_single_factor_parameters_cannot_drift():
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(raw)
    tampered["models"]["ctgan_separate_class"]["candidates"][1][
        "fit_parameters"
    ]["quantile_grid_size"] = 129

    with pytest.raises(
        V27PreparationContractError,
        match="preregistered factor parameters",
    ):
        validate_preparation_definition(tampered)


def test_cli_exposes_only_plan_and_dry_run_without_execution_options():
    assert parse_args(["--mode", "plan"]).mode == "plan"
    assert parse_args(["--mode", "dry-run"]).mode == "dry-run"
    with pytest.raises(SystemExit):
        parse_args(["--mode", "execute"])
    with pytest.raises(SystemExit):
        parse_args(["--mode", "plan", "--device", "cuda:0"])
    with pytest.raises(SystemExit):
        parse_args(["--mode", "plan", "--authorization", "anything.json"])


def test_fit_state_tampering_fails_closed():
    plan = load_and_validate_preparation(CONFIG)
    candidate = next(
        candidate
        for candidate in plan.candidates
        if candidate.candidate_id == "cof_v27_c02_gap_logit_bias"
    )
    record = build_fit_state_record(
        candidate=candidate,
        parameters={"offsets": [0.1, -0.1]},
        source_commit="a" * 40,
        config_sha256=plan.config_sha256,
        train_file_sha256="b" * 64,
        train_content_sha256="c" * 64,
        sampling_plan_sha256=plan.sampling_plan_sha256,
        destination="checkpoint",
    )
    validate_fit_state_record(
        record=record,
        candidate=candidate,
        source_commit="a" * 40,
        config_sha256=plan.config_sha256,
        train_file_sha256="b" * 64,
        train_content_sha256="c" * 64,
        sampling_plan_sha256=plan.sampling_plan_sha256,
    )
    tampered = dict(record)
    tampered["parameters"] = {"offsets": [9.0, -9.0]}
    with pytest.raises(V27PreparationContractError, match="hash"):
        validate_fit_state_record(
            record=tampered,
            candidate=candidate,
            source_commit="a" * 40,
            config_sha256=plan.config_sha256,
            train_file_sha256="b" * 64,
            train_content_sha256="c" * 64,
            sampling_plan_sha256=plan.sampling_plan_sha256,
        )


def test_frozen_endpoint_and_selection_rule_cannot_drift():
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    endpoint_tamper = copy.deepcopy(raw)
    endpoint_tamper["frozen_contract"]["endpoint"] = "changed"
    with pytest.raises(V27PreparationContractError, match="benchmark"):
        validate_preparation_definition(endpoint_tamper)

    selection_tamper = copy.deepcopy(raw)
    selection_tamper["selection_rule"]["final_tiebreaker"] = "random"
    with pytest.raises(V27PreparationContractError, match="selection"):
        validate_preparation_definition(selection_tamper)
