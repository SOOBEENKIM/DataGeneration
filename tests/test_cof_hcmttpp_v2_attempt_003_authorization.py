import copy
from pathlib import Path

import pytest

from experiments.cof_hcmttpp_v2_execution_runner import (
    H1RunnerContractError,
    build_execution_plan,
    build_expected_authorization_claim,
    dry_run_execution,
    validate_authorization_document,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmark_v2/cof_hcmttpp_v2_execution_runner.yaml"


def test_attempt_003_is_the_only_authorizable_execution_after_both_corrections():
    plans = {
        dataset: build_execution_plan(CONFIG, dataset=dataset, candidate_id="H1")
        for dataset in ("amlsim", "sparkov")
    }

    for dataset, plan in plans.items():
        job = plan.jobs[0]
        report = dry_run_execution(plan)
        claim = build_expected_authorization_claim(
            plan,
            approval={
                "approved": True,
                "text": (
                    f"Explicit {dataset} H1 seed 4001 attempt_003 first-valid "
                    "evaluation approval after deterministic implementation corrections"
                ),
            },
        )

        assert (job.dataset, job.candidate_id, job.seed, job.attempt) == (
            dataset,
            "H1",
            4001,
            "attempt_003",
        )
        assert report["status"] == "PASS"
        assert report["implementation_correction_evidence"]["status"] == "PASS"
        assert len(report["current_readiness_boundary_sha256"]) == 64
        assert claim["current_readiness_boundary_sha256"] == report[
            "current_readiness_boundary_sha256"
        ]
        assert claim["authorized_action"] == (
            "EXTERNAL_VALIDATION_H1_FIRST_VALID_ATTEMPT_003"
        )
        assert claim["attempt"] == "attempt_003"
        assert claim["scope"]["attempt"] == "attempt_003"
        assert claim["scope"]["retry"] is False
        assert claim["scope"]["execution_class"] == (
            "FIRST_VALID_H1_EVALUATION_AFTER_TWO_DETERMINISTIC_IMPLEMENTATION_FAILURES"
        )
        evidence = claim["implementation_corrections"]
        assert evidence["authorized_attempt"] == "attempt_003"
        assert evidence["generic_retry"] is False
        assert evidence["arbitrary_attempt"] is False
        assert evidence["attempt_sweep"] is False
        assert evidence["rqs_inverse"]["report_sha256"] == (
            "e78df91540d25d53cf8153125caea676c2bdaf70cd93b4b6328489a947b040bf"
        )
        assert evidence["spawn_ownership"]["report_sha256"] == (
            "e3cef48e6b76f8dc05079307b98c142c8ffff90b91bdf89237532740f70ec3d3"
        )
        assert evidence["execution_readiness"]["passed_source_commit"] == (
            "f5125fbde533a965b0e3c71b1b3b8950c7fffc85"
        )
        assert evidence["execution_readiness"]["passed_boundary_sha256"] == (
            "77a58fd66a3352adb71c3dce5f56c764414f09afc9ef6ecb1636fa4e9ece5b9f"
        )
        assert set(evidence["failed_attempts"]) == {"attempt_001", "attempt_002"}

    amlsim_claim = build_expected_authorization_claim(
        plans["amlsim"],
        approval={"approved": True, "text": "Explicit AMLSim H1 attempt_003 approval"},
    )
    with pytest.raises(H1RunnerContractError, match="dataset-scoped authorization"):
        validate_authorization_document(plans["sparkov"], amlsim_claim)

    mutated = copy.deepcopy(amlsim_claim)
    mutated["attempt"] = "attempt_004"
    with pytest.raises(H1RunnerContractError, match="hash-bound identity"):
        validate_authorization_document(plans["amlsim"], mutated)
