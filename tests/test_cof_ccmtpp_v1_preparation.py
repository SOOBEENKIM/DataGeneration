import hashlib
from pathlib import Path

import pytest

from experiments.cof_ccmtpp_v1_preparation import (
    build_ccmtpp_plan,
    ccmtpp_plan_report,
    dry_run_ccmtpp,
)
from scripts.prepare_cof_ccmtpp_v1 import parse_args


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs/benchmark_v2/cof_ccmtpp_v1_source_only.yaml"


def _inventory(root: Path):
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


def test_plan_has_only_c0_to_c4_and_zero_execution():
    report = ccmtpp_plan_report(build_ccmtpp_plan(CONFIG))
    assert report["status"] == "PASS"
    assert report["counts"] == {
        "frozen_controls": 1,
        "implemented_candidates": 4,
        "eligible_future_candidates_now": 1,
        "gated_future_candidates": 3,
        "unimplemented_candidates": 1,
    }
    assert [item["candidate_id"] for item in report["candidates"]] == [
        "C0", "C1", "C2", "C3", "C4"
    ]
    assert report["deferred_C5"]["status"] == "LOCKED_UNIMPLEMENTED"
    assert all(value == 0 for value in report["execution_counts"].values())
    assert report["authorization_created"] is False
    assert report["runtime_artifact_created"] is False


def test_dry_run_rehashes_preserved_artifacts_without_writes_or_model_calls():
    future_runtime = REPOSITORY / "artifacts/cof_ccmtpp_v1"
    future_data = REPOSITORY / "data/cof_ccmtpp_v1"
    before_runtime = _inventory(future_runtime)
    before_data = _inventory(future_data)
    report = dry_run_ccmtpp(build_ccmtpp_plan(CONFIG))
    assert report["status"] == "PASS"
    assert report["preservation_verified"] is True
    assert report["preserved_v2_8_tree_sha256"] == (
        "3d082b1500182ccb05b4772c243a3797f72cdf4ba834e69b1e3beaeb883bcbd7"
    )
    assert report["preserved_external_bundle_sha256"] == {
        "amlsim": "cc8f4c9c6ff415ccd4f78ec193122100145e10084751b971dcc0ce5e705fa024",
        "sparkov": "62fefda6207911104a9bff0ebb41d5edf44c20a4c78dccef5875d9768be6addc",
    }
    assert all(value == 0 for value in report["execution_counts"].values())
    assert _inventory(future_runtime) == before_runtime
    assert _inventory(future_data) == before_data


def test_cli_only_allows_plan_and_dry_run_and_execute_is_fail_closed():
    assert parse_args(["--mode", "plan"]).mode == "plan"
    assert parse_args(["--mode", "dry-run"]).mode == "dry-run"
    with pytest.raises(SystemExit):
        parse_args(["--mode", "execute"])
    with pytest.raises(SystemExit):
        parse_args(["--mode", "plan", "--authorization", "auth.json"])
    with pytest.raises(SystemExit):
        parse_args(["--mode", "plan", "--device", "cuda:0"])
