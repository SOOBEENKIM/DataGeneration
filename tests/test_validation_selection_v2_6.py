import ast
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from benchmarks.types import SequenceBatch, SyntheticBatch
import eval.validation_selection_v2_6 as selection
from eval.validation_selection_v2_6 import (
    DevelopmentContext,
    SelectionContractError,
    build_selection_freeze,
    canonical_sha256,
    choose_candidate,
    evaluate_candidate_artifact,
    exclusive_json,
    load_development_context,
    read_yaml_mapping,
    sha256_file,
    validate_selection_config,
)
from experiments.provenance_v2_5 import hash_batch
from generators.sampling_plan import SamplingPlan


REPOSITORY = Path(__file__).resolve().parents[1]


def sequence_fixture() -> SequenceBatch:
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
    return SequenceBatch(
        x_num=x_num,
        dt_bin=dt_bin,
        x_cat=x_cat,
        valid_mask=valid,
        y_entity=y,
        lengths=lengths,
        entity_ids=np.asarray(["a", "b", "c", "d"]),
    )


def save_batch(path: Path, batch: SequenceBatch) -> None:
    np.savez_compressed(
        path,
        **{
            field: getattr(batch, field)
            for field in SequenceBatch.__dataclass_fields__
        },
    )


def save_sample(path: Path, batch: SequenceBatch) -> SyntheticBatch:
    sample = SyntheticBatch(
        **{
            field: getattr(batch, field).copy()
            for field in SyntheticBatch.__dataclass_fields__
        }
    )
    np.savez_compressed(
        path,
        **{
            field: getattr(sample, field)
            for field in SyntheticBatch.__dataclass_fields__
        },
    )
    return sample


def test_preregistered_config_has_equal_finite_candidate_budgets():
    raw = read_yaml_mapping(
        REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
    )

    validated = validate_selection_config(raw)

    assert validated["candidate_count"] == 4
    assert validated["requested_updates"] == 20_000
    assert validated["max_gpu_wall_seconds"] == 7_200
    assert validated["selection_seed"] == 2601
    assert set(validated["definitions"]) == set(selection.MODEL_IDS)
    assert all(
        len(candidates) == 4
        for candidates in validated["definitions"].values()
    )
    assert raw["selection_budget"]["evaluation_candidates_per_model"] == 4
    assert raw["selection_budget"]["training_trajectories_per_model"] == 3
    assert raw["selection_budget"]["training_trajectories_all_models"] == 12
    assert raw["selection_budget"]["evaluation_candidates_all_models"] == 16
    assert raw["selection_budget"]["max_gpu_hours_per_model"] == 6.0
    assert raw["selection_budget"]["max_gpu_hours_all_models"] == 24.0
    assert raw["selection_budget"]["candidate_training_execution_authorized"] is False


def test_c00_c01_share_one_native_trajectory_without_resource_duplication():
    raw = read_yaml_mapping(
        REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
    )

    validated = validate_selection_config(raw)

    for model_id, candidates in validated["definitions"].items():
        by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
        c00 = by_id["c00_native_checkpoint_10000"]
        c01 = by_id["c01_native_checkpoint_20000"]
        assert c00["training_schedule"]["shared_trajectory_id"] == (
            c01["training_schedule"]["shared_trajectory_id"]
        )
        assert c00["training_schedule"]["selection_checkpoint_step"] == 10_000
        assert c01["training_schedule"]["selection_checkpoint_step"] == 20_000
        assert c00["training_schedule"]["requested_updates"] == 20_000
        assert c01["training_schedule"]["requested_updates"] == 20_000
        trajectory_ids = {
            candidate["training_schedule"]["shared_trajectory_id"]
            for candidate in candidates
        }
        assert len(trajectory_ids) == 3, model_id

    assert validated["training_trajectory_count"] == 12
    assert validated["evaluation_candidate_count"] == 16


def test_real_development_manifest_reads_train_and_validation_only(monkeypatch):
    real_load = np.load
    seen: list[Path] = []

    def traced_load(path, *args, **kwargs):
        seen.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(selection.np, "load", traced_load)
    context = load_development_context(
        repository_root=REPOSITORY,
        manifest_path=(
            REPOSITORY
            / "configs/benchmark_v2/development_data_v2_6.yaml"
        ),
    )

    assert {path.name for path in seen} == {
        "train.npz",
        "validation.npz",
    }
    assert all("test" not in {part.lower() for part in path.parts} for path in seen)
    assert context.train_content_sha256 == (
        "0c03f179930cd4d89ccc8a8b66283e620313e16d15fd53f0674c133bcb80c09d"
    )
    assert context.validation_content_sha256 == (
        "aa1576b4ccae6a2ca30d0472f0782e1231ecc61ab3d224590a772dc3448b9e66"
    )
    assert context.plan.plan_hash == (
        "862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27"
    )


def test_development_manifest_rejects_test_split_before_array_load(
    tmp_path,
    monkeypatch,
):
    manifest = {
        "schema_version": selection.DEVELOPMENT_SCHEMA,
        "status": "FROZEN_FOR_SELECTION",
        "test_split_access": "FORBIDDEN",
        "splits": {
            "train": {},
            "validation": {},
            "test": {"path": "test.npz"},
        },
    }
    path = tmp_path / "development.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    calls = 0

    def forbidden_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("array loader must not be reached")

    monkeypatch.setattr(selection.np, "load", forbidden_load)

    with pytest.raises(SelectionContractError, match="train and validation only"):
        load_development_context(
            repository_root=tmp_path,
            manifest_path=path,
        )
    assert calls == 0


def result(
    candidate_id: str,
    *,
    passed: bool,
    amount_ks: float,
    gap_ks: float,
):
    return {
        "model_id": "cof_seqgen",
        "candidate_id": candidate_id,
        "all_five_guards_pass": passed,
        "continuous_ks_max": max(amount_ks, gap_ks),
        "continuous_ks_sum": amount_ks + gap_ks,
        "candidate_definition_sha256": candidate_id.ljust(64, "0"),
        "checkpoint_sha256": candidate_id.ljust(64, "1"),
        "sampled_checkpoint_step": 20_000,
        "source_commit": "a" * 40,
        "relevant_source_sha256": "b" * 64,
    }


def test_selection_rule_requires_all_pass_then_minimizes_max_continuous_ks():
    values = [
        result("c00", passed=False, amount_ks=0.0001, gap_ks=0.0001),
        result("c01", passed=True, amount_ks=0.0050, gap_ks=0.0030),
        result("c02", passed=True, amount_ks=0.0040, gap_ks=0.0045),
        result("c03", passed=True, amount_ks=0.0045, gap_ks=0.0040),
    ]

    selected = choose_candidate(values)

    assert selected["status"] == "SELECTED"
    assert selected["selected_candidate_id"] == "c02"
    assert selected["v2_6_full_run_permitted"] is False


def test_no_all_guard_pass_candidate_prohibits_full_run():
    selected = choose_candidate(
        [
            result("c00", passed=False, amount_ks=0.01, gap_ks=0.01),
            result("c01", passed=False, amount_ks=0.02, gap_ks=0.02),
        ]
    )

    assert selected == {
        "model_id": "cof_seqgen",
        "status": "NO_PASSING_CANDIDATE",
        "selected_candidate_id": None,
        "v2_6_full_run_permitted": False,
        "reason": "no candidate passed all five validation guards",
    }


def test_primary_candidate_failure_blocks_primary_c2_selection():
    selections = {
        model_id: {"status": "SELECTED"}
        for model_id in selection.MODEL_IDS
    }
    selections["tvae_separate_class"] = {
        "status": "NO_PASSING_CANDIDATE"
    }

    readiness = selection.selection_readiness(selections)

    assert readiness["status"] == "SELECTION_FAILED_PRIMARY_NO_FULL_RUN"
    assert readiness["primary_c2_selection_ready"] is False
    assert readiness["blocking_primary_models"] == ["tvae_separate_class"]
    assert readiness["neural_secondary_comparison"] == (
        "EVALUABLE_AWAITING_AUTHORIZATION"
    )


def test_neural_only_failure_does_not_block_primary_c2_selection():
    selections = {
        model_id: {"status": "SELECTED"}
        for model_id in selection.MODEL_IDS
    }
    selections["neural_sequence"] = {"status": "NO_PASSING_CANDIDATE"}

    readiness = selection.selection_readiness(selections)

    assert readiness["status"] == (
        "PRIMARY_SELECTION_PASS_SECONDARY_NOT_EVALUABLE"
    )
    assert readiness["primary_c2_selection_ready"] is True
    assert readiness["blocking_primary_models"] == []
    assert readiness["neural_secondary_comparison"] == "NOT_EVALUABLE"


def test_primary_only_selection_marks_unexecuted_neural_not_evaluable():
    selections = {
        model_id: {"status": "SELECTED"}
        for model_id in selection.PRIMARY_MODEL_IDS
    }

    readiness = selection.selection_readiness(selections)

    assert readiness["status"] == (
        "PRIMARY_SELECTION_PASS_SECONDARY_NOT_EVALUABLE"
    )
    assert readiness["primary_c2_selection_ready"] is True
    assert readiness["blocking_primary_models"] == []
    assert readiness["neural_secondary_comparison"] == "NOT_EVALUABLE"
    assert readiness["all_models_selected"] is False


def test_real_candidate_seam_records_all_five_validation_guards(tmp_path):
    repository = tmp_path / "repository"
    candidate_root = repository / "artifacts/benchmark_v2_6/selection/candidates"
    candidate_id = "c00"
    attempt = (
        candidate_root
        / "cof_seqgen"
        / candidate_id
        / "seed_2601"
        / "attempt_001"
    )
    attempt.mkdir(parents=True)
    batch = sequence_fixture()
    sample_path = attempt / "validation_sample.npz"
    save_sample(sample_path, batch)
    checkpoint_path = attempt / "checkpoint.pt"
    checkpoint_path.write_bytes(b"fixture checkpoint")
    plan = SamplingPlan.from_batch(batch)
    context = DevelopmentContext(
        train=batch,
        validation=batch,
        plan=plan,
        tau=np.asarray([0.5, 2.0]),
        receiver_categories=3,
        manifest_sha256="d" * 64,
        train_file_sha256="e" * 64,
        validation_file_sha256="f" * 64,
        train_content_sha256=hash_batch(batch),
        validation_content_sha256=hash_batch(batch),
    )
    candidate = {
        "candidate_id": candidate_id,
        "numeric_representation": {},
        "channel_loss_weights": {},
        "sampling_rule": {},
        "training_schedule": {
            "shared_trajectory_id": "cof_native_seed_2601",
            "requested_updates": 20_000,
            "selection_checkpoint_step": 10_000,
        },
        "implementation_contract": {},
    }
    config_hash = "c" * 64
    trajectory_attempt = (
        candidate_root
        / "_trajectories/cof_seqgen/cof_native_seed_2601"
        / "seed_2601/attempt_001"
    )
    trajectory_attempt.mkdir(parents=True)
    trajectory_manifest = {
        "status": "RUNNING",
        "source_commit": "a" * 40,
        "relevant_source_sha256": "b" * 64,
        "selection_config_sha256": config_hash,
        "development_manifest_sha256": context.manifest_sha256,
        "train_file_sha256": context.train_file_sha256,
        "train_content_sha256": context.train_content_sha256,
        "validation_file_sha256": context.validation_file_sha256,
        "validation_content_sha256": context.validation_content_sha256,
        "selection_plan_sha256": plan.plan_hash,
        "model_id": "cof_seqgen",
        "shared_trajectory_id": "cof_native_seed_2601",
        "requested_updates": 20_000,
    }
    trajectory_manifest_path = trajectory_attempt / "manifest.json"
    trajectory_manifest_path.write_text(
        json.dumps(trajectory_manifest),
        encoding="utf-8",
    )
    trajectory_complete = {
        "status": "COMPLETE",
        "shared_trajectory_id": "cof_native_seed_2601",
        "requested_updates": 20_000,
        "actual_updates": 20_000,
        "completed_hard_cap_contract": True,
        "wall_cap_reached": False,
    }
    trajectory_complete_path = (
        trajectory_attempt / "TRAJECTORY_COMPLETE.json"
    )
    trajectory_complete_path.write_text(
        json.dumps(trajectory_complete),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": selection.CANDIDATE_RESULT_SCHEMA,
        "status": "COMPLETE",
        "model_id": "cof_seqgen",
        "candidate_id": candidate_id,
        "selection_seed": 2601,
        "selection_config_sha256": config_hash,
        "candidate_definition_sha256": canonical_sha256(candidate),
        "shared_trajectory_id": "cof_native_seed_2601",
        "train_file_sha256": context.train_file_sha256,
        "train_content_sha256": context.train_content_sha256,
        "validation_file_sha256": context.validation_file_sha256,
        "validation_content_sha256": context.validation_content_sha256,
        "development_manifest_sha256": context.manifest_sha256,
        "selection_plan_sha256": plan.plan_hash,
        "source_commit": "a" * 40,
        "relevant_source_sha256": "b" * 64,
        "requested_updates": 20_000,
        "actual_updates": 20_000,
        "trajectory_completed_requested_updates": True,
        "sampled_checkpoint_step": 10_000,
        "wall_cap_reached": False,
        "trajectory_manifest_path": str(
            trajectory_manifest_path.relative_to(repository)
        ),
        "trajectory_manifest_sha256": sha256_file(
            trajectory_manifest_path
        ),
        "trajectory_complete_path": str(
            trajectory_complete_path.relative_to(repository)
        ),
        "trajectory_complete_sha256": sha256_file(
            trajectory_complete_path
        ),
        "validation_sample_path": str(
            sample_path.relative_to(repository)
        ),
        "validation_sample_sha256": sha256_file(sample_path),
        "checkpoint_path": str(checkpoint_path.relative_to(repository)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    (attempt / "candidate_result.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    evaluated = evaluate_candidate_artifact(
        repository_root=repository,
        candidate_root=candidate_root,
        model_id="cof_seqgen",
        candidate=candidate,
        selection_seed=2601,
        config_sha256=config_hash,
        context=context,
        thresholds={key: 10.0 for key in selection.METRIC_KEYS},
    )

    assert evaluated["status"] == "PASS"
    assert set(evaluated["statistics"]) == set(selection.METRIC_KEYS)
    assert set(evaluated["checks"]) == set(selection.METRIC_KEYS)
    assert all(value == "PASS" for value in evaluated["checks"].values())
    assert evaluated["sample_contract"]["status"] == "PASS"


def passing_report():
    selections = {
        model_id: {
            "status": "SELECTED",
            "selected_candidate_id": "c00",
            "candidate_definition_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "sampled_checkpoint_step": 10_000,
        }
        for model_id in selection.MODEL_IDS
    }
    return {
        "status": "SELECTION_PASS",
        "all_models_selected": True,
        "primary_c2_selection_ready": True,
        "neural_secondary_comparison": (
            "EVALUABLE_AWAITING_AUTHORIZATION"
        ),
        "test_split_read": False,
        "model_selections": selections,
        "selection_config_sha256": "c" * 64,
        "candidate_source_commit": "d" * 40,
        "candidate_relevant_source_sha256": "e" * 64,
        "development_manifest_sha256": "f" * 64,
        "train_file_sha256": "1" * 64,
        "validation_file_sha256": "2" * 64,
        "train_content_sha256": "3" * 64,
        "validation_content_sha256": "4" * 64,
        "selection_plan_sha256": "5" * 64,
    }


def test_freeze_records_no_test_hash_and_still_requires_user_authorization():
    frozen = build_selection_freeze(passing_report())

    assert frozen["status"] == "FROZEN_AWAITING_SEPARATE_AUTHORIZATION"
    assert frozen["test_split_hash"] is None
    assert frozen["test_split_read"] is False
    assert frozen["fresh_test_authorized"] is False
    assert frozen["five_seed_full_experiment_authorized"] is False
    assert frozen["requires_separate_user_authorization"] is True

    failed = passing_report()
    failed["status"] = "SELECTION_FAILED_PRIMARY_NO_FULL_RUN"
    failed["all_models_selected"] = False
    failed["primary_c2_selection_ready"] = False
    with pytest.raises(SelectionContractError, match="cannot freeze"):
        build_selection_freeze(failed)


def test_freeze_allows_neural_secondary_not_evaluable_when_primary_selected():
    report = passing_report()
    report["status"] = "PRIMARY_SELECTION_PASS_SECONDARY_NOT_EVALUABLE"
    report["all_models_selected"] = False
    report["neural_secondary_comparison"] = "NOT_EVALUABLE"
    report["model_selections"]["neural_sequence"] = {
        "status": "NO_PASSING_CANDIDATE",
        "selected_candidate_id": None,
    }

    frozen = build_selection_freeze(report)

    assert frozen["primary_c2_selection_ready"] is True
    assert frozen["neural_secondary_comparison"] == "NOT_EVALUABLE"
    assert "neural_sequence" not in frozen["selected_candidates"]
    assert frozen["model_selection_status"]["neural_sequence"] == (
        "NO_PASSING_CANDIDATE"
    )


def test_selection_output_is_append_only(tmp_path):
    path = tmp_path / "selection_report.json"
    exclusive_json(path, {"status": "first"})

    with pytest.raises(SelectionContractError, match="already exists"):
        exclusive_json(path, {"status": "overwrite"})

    assert json.loads(path.read_text()) == {"status": "first"}


def test_selection_harness_has_no_training_gpu_dgp_or_model_sample_calls():
    paths = (
        REPOSITORY / "eval/validation_selection_v2_6.py",
        REPOSITORY / "scripts/select_validation_candidates_v2_6.py",
    )
    forbidden_call_names = {
        "fit",
        "sample",
        "generate_benchmark",
        "generate_fixed_binning_split",
        "cuda",
    }
    calls: list[str] = []
    imports: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

    assert not (forbidden_call_names & set(calls))
    assert "torch" not in imports
    assert not any(name.startswith("models.") for name in imports)


def test_config_preregistration_contract_and_harness_are_mutually_consistent():
    config_path = REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
    config = read_yaml_mapping(config_path)
    config_hash = sha256_file(config_path)
    preregistration = (
        REPOSITORY
        / "docs/benchmark_v2/preregistered_model_selection_v2_6.md"
    ).read_text(encoding="utf-8")
    artifact_contract = (
        REPOSITORY
        / "docs/benchmark_v2/selection_artifact_contract_v2_6.md"
    ).read_text(encoding="utf-8")

    assert config_hash == (
        "0884a0144f74f6317ae9e636c0cf5657dedac21ff12cdf545a6b6e5e29be4c4d"
    )
    assert config_hash in preregistration
    assert config["model_roles"]["primary_c2_learned"] == list(
        selection.PRIMARY_MODEL_IDS
    )
    assert config["model_roles"]["secondary"] == ["neural_sequence"]
    assert config["model_roles"]["fixed_reference"] == ["empirical_iid"]
    assert "12 training trajectories" in artifact_contract
    assert "16 evaluation candidates" in artifact_contract
    assert "c00 and c01 reference one shared native trajectory" in (
        artifact_contract
    )
    assert "PRIMARY_SELECTION_PASS_SECONDARY_NOT_EVALUABLE" in (
        artifact_contract
    )
