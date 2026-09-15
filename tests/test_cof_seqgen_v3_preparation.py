import hashlib
from pathlib import Path

import pytest

from experiments.cof_seqgen_v3_preparation import (
    build_v3_plan,
    dry_run_v3,
    v3_plan_report,
)
from scripts.prepare_cof_seqgen_v3 import parse_args


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = (
    REPOSITORY
    / "configs/benchmark_v2/cof_seqgen_v3_source_preparation.yaml"
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


def test_v3_plan_is_three_fixed_candidates_with_zero_execution_calls():
    report = v3_plan_report(build_v3_plan(CONFIG))

    assert report["status"] == "PASS"
    assert report["counts"] == {
        "candidates": 3,
        "frozen_hash_references": 1,
        "future_training_trajectories": 2,
        "future_validation_candidates": 2,
    }
    assert [
        operation["candidate_id"]
        for operation in report["operations"]
    ] == [
        "cof_v3_ref_v28_frozen",
        "cof_v3_c01_direct_joint",
        "cof_v3_c02_factorized_joint",
    ]
    assert report["execution_counts"] == {
        "gpu_queries": 0,
        "cuda_calls": 0,
        "model_fit_calls": 0,
        "optimizer_updates": 0,
        "checkpoint_writes": 0,
        "model_sample_calls": 0,
        "validation_execution_calls": 0,
        "selection_calls": 0,
        "test_split_reads": 0,
        "fresh_test_calls": 0,
        "tstr_calls": 0,
        "privacy_calls": 0,
        "five_seed_full_run_calls": 0,
    }
    assert report["authorization_created"] is False
    assert report["runtime_artifact_created"] is False


def test_v3_dry_run_verifies_frozen_inputs_without_runtime_or_model_calls():
    v28_root = REPOSITORY / "artifacts/benchmark_v2_8"
    v3_runtime = REPOSITORY / "artifacts/benchmark_v3"
    v28_before = _tree_inventory(v28_root)
    v3_before = _tree_inventory(v3_runtime)

    report = dry_run_v3(build_v3_plan(CONFIG))

    assert report["status"] == "PASS"
    assert report["frozen_provenance_verified"] is True
    assert report["v2_8_official_conclusion_verified"] is True
    assert report["v2_8_candidate_tree_sha256"] == (
        "3d082b1500182ccb05b4772c243a3797f72cdf4ba834e69b1e3beaeb883bcbd7"
    )
    assert report["frozen_reference_artifacts_verified"] == 5
    assert report["architecture_import_verified"] is True
    assert report["execution_counts"] == v3_plan_report(
        build_v3_plan(CONFIG)
    )["execution_counts"]
    assert report["authorization_created"] is False
    assert report["runtime_artifact_created"] is False
    assert _tree_inventory(v28_root) == v28_before
    assert _tree_inventory(v3_runtime) == v3_before


def test_v3_cli_exposes_only_source_only_plan_and_dry_run():
    assert parse_args(["--mode", "plan"]).mode == "plan"
    assert parse_args(["--mode", "dry-run"]).mode == "dry-run"
    with pytest.raises(SystemExit):
        parse_args(["--mode", "execute"])
    with pytest.raises(SystemExit):
        parse_args(["--mode", "plan", "--authorization", "auth.json"])
    with pytest.raises(SystemExit):
        parse_args(["--mode", "plan", "--device", "cuda:0"])
