from pathlib import Path
import subprocess

import pytest
import yaml

from experiments.evaluation_aggregate_v2_8 import (
    EvaluationAggregateV28ContractError,
    build_v28_selection,
    build_v28_aggregate_plan,
    execute_v28_validation_aggregate,
    expected_v28_aggregate_authorization,
    inspect_v28_terminal_inventory,
    load_v28_stored_guard_results,
    validate_v28_aggregate_authorization,
    write_v28_aggregate_bundle,
)
from scripts.aggregate_evaluation_selection_v2_8 import parse_args


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = (
    REPOSITORY
    / "configs/benchmark_v2/evaluation_aggregate_v2_8.yaml"
)


def test_plan_is_stored_evidence_only_and_has_exact_primary_family():
    plan = build_v28_aggregate_plan(CONFIG)

    assert [candidate.candidate_id for candidate in plan.candidates] == [
        "ctgan_v28_c01_joint_gap_receiver_decoder",
        "cof_v28_c01_gap_distribution_sampler",
        "tvae_v27_c01_amount_inverse_decoder",
    ]
    assert [candidate.source_kind for candidate in plan.candidates] == [
        "stored_v2_8_evaluation",
        "stored_v2_8_evaluation",
        "frozen_v2_7_selected_reference",
    ]
    assert plan.forbidden_call_counts == {
        "gpu_query": 0,
        "cuda": 0,
        "training": 0,
        "sampling": 0,
        "guard_recalculation": 0,
        "candidate_reexecution": 0,
        "test_split_access": 0,
    }


def test_inventory_verifies_two_terminals_and_frozen_tvae_hashes():
    plan = build_v28_aggregate_plan(CONFIG)

    inventory = inspect_v28_terminal_inventory(
        repository_root=REPOSITORY,
        plan=plan,
    )

    assert inventory["worker_terminal_count"] == 2
    assert inventory["candidate_terminal_count"] == 2
    assert inventory["stored_evaluation_count"] == 2
    assert inventory["frozen_reference_count"] == 1
    assert set(inventory["candidate_artifacts"]) == {
        "ctgan_v28_c01_joint_gap_receiver_decoder",
        "cof_v28_c01_gap_distribution_sampler",
    }
    assert inventory["frozen_reference"]["candidate_id"] == (
        "tvae_v27_c01_amount_inverse_decoder"
    )
    assert inventory["frozen_reference"]["evaluation_sha256"] == (
        "56dc7155791440369f4fc23ef791f916cc86417603d7091730efcdd8038581b2"
    )
    assert inventory["test_split_read"] is False


def test_stored_results_select_ctgan_and_tvae_but_block_on_cof():
    plan = build_v28_aggregate_plan(CONFIG)
    inventory = inspect_v28_terminal_inventory(
        repository_root=REPOSITORY,
        plan=plan,
    )

    results, provenance = load_v28_stored_guard_results(
        repository_root=REPOSITORY,
        plan=plan,
        inventory=inventory,
    )
    report, manifest = build_v28_selection(
        candidate_results=results,
        provenance=provenance,
    )

    by_model = report["model_selections"]
    assert by_model["ctgan_separate_class"] == {
        "status": "SELECTED",
        "selected_candidate_id": (
            "ctgan_v28_c01_joint_gap_receiver_decoder"
        ),
    }
    assert by_model["tvae_separate_class"] == {
        "status": "SELECTED",
        "selected_candidate_id": (
            "tvae_v27_c01_amount_inverse_decoder"
        ),
    }
    assert by_model["cof_seqgen"] == {
        "status": "NO_PASSING_CANDIDATE",
        "selected_candidate_id": None,
    }
    assert report["primary_c2_selection_ready"] is False
    assert report["blocking_primary_models"] == ["cof_seqgen"]
    assert manifest["status"] == "FROZEN_NO_PASSING_PRIMARY_CANDIDATE"
    assert provenance["guard_recalculations_executed"] == 0
    assert provenance["validation_samples_read"] == 0
    assert provenance["test_split_read"] is False


def test_authorization_is_exact_aggregate_only_and_fail_closed():
    plan = build_v28_aggregate_plan(CONFIG)
    inventory = inspect_v28_terminal_inventory(
        repository_root=REPOSITORY,
        plan=plan,
    )
    authorization = expected_v28_aggregate_authorization(
        plan=plan,
        inventory=inventory,
        source_commit="a" * 40,
        relevant_source_sha256="b" * 64,
    )

    validate_v28_aggregate_authorization(
        plan=plan,
        inventory=inventory,
        authorization=authorization,
        source_commit="a" * 40,
        relevant_source_sha256="b" * 64,
    )
    assert authorization["scope"]["aggregate_only"] is True
    assert all(
        authorization["scope"][key] is False
        for key in (
            "gpu_query",
            "cuda",
            "training",
            "sampling",
            "guard_recalculation",
            "candidate_reexecution",
            "test_split_access",
            "fresh_test",
            "tstr",
            "privacy",
            "five_seed_full_run",
        )
    )

    tampered = dict(authorization)
    tampered["input_inventory_sha256"] = "0" * 64
    with pytest.raises(
        EvaluationAggregateV28ContractError,
        match="authorization mismatch",
    ):
        validate_v28_aggregate_authorization(
            plan=plan,
            inventory=inventory,
            authorization=tampered,
            source_commit="a" * 40,
            relevant_source_sha256="b" * 64,
        )


def test_bundle_is_append_only_and_aggregate_terminal_is_last(tmp_path):
    output = tmp_path / "aggregate_attempt_001"
    report = {
        "status": "COMPLETE",
        "primary_c2_selection_ready": False,
        "blocking_primary_models": ["cof_seqgen"],
    }
    manifest = {
        "status": "FROZEN_NO_PASSING_PRIMARY_CANDIDATE",
        "primary_c2_selection_ready": False,
    }

    terminal = write_v28_aggregate_bundle(
        output_root=output,
        authorization_path=Path("aggregate_authorization.json"),
        authorization_sha256="a" * 64,
        input_inventory={"input_inventory_sha256": "b" * 64},
        selection_report=report,
        selection_manifest=manifest,
    )

    assert terminal["status"] == "COMPLETE"
    assert terminal["primary_c2_selection_ready"] is False
    assert (output / "selection_report.json").is_file()
    assert (output / "selection_manifest.json").is_file()
    assert (output / "checksum_manifest.json").is_file()
    assert (output / "artifact_index.json").is_file()
    assert (output / "AGGREGATE_COMPLETE.json").is_file()
    with pytest.raises(
        EvaluationAggregateV28ContractError,
        match="append-only aggregate",
    ):
        write_v28_aggregate_bundle(
            output_root=output,
            authorization_path=Path("aggregate_authorization.json"),
            authorization_sha256="a" * 64,
            input_inventory={"input_inventory_sha256": "b" * 64},
            selection_report=report,
            selection_manifest=manifest,
        )


def test_plan_and_dry_run_create_no_authorization_or_runtime(tmp_path):
    output = tmp_path / "must-not-exist"
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    planned = execute_v28_validation_aggregate(
        repository_root=REPOSITORY,
        config_path=CONFIG,
        authorization_path=None,
        output_root=output,
        source_commit=source_commit,
        mode="plan",
    )
    dry_run = execute_v28_validation_aggregate(
        repository_root=REPOSITORY,
        config_path=CONFIG,
        authorization_path=None,
        output_root=output,
        source_commit=source_commit,
        mode="dry-run",
    )

    assert planned["candidate_terminal_plan"] == 2
    assert planned["frozen_reference_plan"] == 1
    assert dry_run["input_inventory_verified"] is True
    assert dry_run["selection_executed"] is False
    assert dry_run["authorization_created"] is False
    assert dry_run["runtime_artifact_created"] is False
    assert dry_run["execution_counts"] == {
        "gpu_query": 0,
        "cuda": 0,
        "training": 0,
        "sampling": 0,
        "guard_recalculation": 0,
        "candidate_reexecution": 0,
        "test_split_access": 0,
    }
    assert not output.exists()


def test_cli_requires_authorization_only_for_execute():
    planned = parse_args(["--mode", "plan"])
    assert planned.authorization is None
    with pytest.raises(SystemExit):
        parse_args(["--mode", "execute"])


def test_source_config_fails_closed_on_any_forbidden_scope(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["guard_recalculation_authorized"] = True
    changed = tmp_path / "evaluation_aggregate_v2_8.yaml"
    changed.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationAggregateV28ContractError,
        match="source-only scope",
    ):
        build_v28_aggregate_plan(changed)
