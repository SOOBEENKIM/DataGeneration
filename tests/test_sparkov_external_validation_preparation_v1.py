import json
from pathlib import Path

import pytest

from scripts.run_external_validation_v1 import (
    ExternalValidationError,
    build_sparkov_validation_authorization,
    build_sparkov_validation_plan,
    evaluate_sparkov_cof_stop_rule,
    validate_external_validation_authorization,
    validate_sparkov_validation_authorization,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/benchmark_v2/external_validation_v1.yaml"


def test_sparkov_plan_is_iid_reuse_plus_three_exact_learned_jobs():
    authorization = build_sparkov_validation_authorization(
        repo_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
        approval_text="source-only Sparkov validation preparation",
    )
    validated = validate_sparkov_validation_authorization(
        repo_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
        authorization=authorization,
    )

    assert validated["status"] == "PASS"
    assert authorization["job_count"] == 3
    assert authorization["execution_authorized"] is False
    assert authorization["iid_reference"]["status"] == "REUSE_ELIGIBLE"
    assert authorization["iid_reference"]["attempt"] == "attempt_002"
    assert {
        (job["model"], job["attempt"], job["wave_id"])
        for job in authorization["jobs"]
    } == {
        ("ctgan_separate_class", "attempt_001", "sparkov_gpu_wave_1"),
        ("cof_seqgen_frozen_non_v3", "attempt_001", "sparkov_gpu_wave_1"),
        ("tvae_separate_class", "attempt_001", "sparkov_gpu_wave_2"),
    }
    by_model = {job["model"]: job for job in authorization["jobs"]}
    assert by_model["ctgan_separate_class"]["data_transformer_n_jobs"] == 1
    assert by_model["tvae_separate_class"]["data_transformer_n_jobs"] == 1
    assert by_model["cof_seqgen_frozen_non_v3"]["data_transformer_n_jobs"] == 0
    assert all(job["dataset"] == "sparkov" for job in authorization["jobs"])
    assert all(job["sparkov_fraudTest_authorized"] is False for job in authorization["jobs"])
    assert all(job["internal_test_authorized"] is False for job in authorization["jobs"])
    with pytest.raises(ExternalValidationError, match="already exists"):
        validate_external_validation_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            authorization=authorization["jobs"][0],
        )


def test_sparkov_plan_and_dry_run_are_source_only_and_execution_free():
    for mode in ("plan", "dry-run"):
        plan = build_sparkov_validation_plan(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            approval_text="source-only Sparkov validation preparation",
            mode=mode,
        )
        assert plan["status"] == "PASS"
        assert plan["job_count"] == 3
        assert plan["iid_reference"]["status"] == "REUSE_ELIGIBLE"
        assert plan["authorization_created"] is False
        assert plan["runtime_artifacts_created"] is False
        assert set(plan["execution_counts"].values()) == {0}
        assert [len(wave["jobs"]) for wave in plan["waves"]] == [2, 1]


def _evaluation(*, combined: float, fidelity: float, hard_valid: bool = True):
    return {
        "hard_validity": {
            "mask": hard_valid,
            "padding": hard_valid,
            "train_discrete_support": hard_valid,
        },
        "selection": {
            "combined_score": combined,
            "fidelity_max_ratio": fidelity,
        },
    }


def test_preregistered_cof_stop_rule_requires_strict_superiority_on_both_fields():
    passing = evaluate_sparkov_cof_stop_rule(
        cof_evaluation=_evaluation(combined=2.0, fidelity=1.5),
        tvae_evaluation=_evaluation(combined=2.1, fidelity=1.6),
        cof_terminal="COMPLETE",
        tvae_terminal="COMPLETE",
    )
    assert passing["decision"] == "CONTINUE_TO_SEPARATE_TEST_PROPOSAL"
    assert passing["test_execution_authorized"] is False

    for cof_combined, cof_fidelity in ((2.1, 1.5), (2.0, 1.6), (2.1, 1.6)):
        stopped = evaluate_sparkov_cof_stop_rule(
            cof_evaluation=_evaluation(
                combined=cof_combined, fidelity=cof_fidelity
            ),
            tvae_evaluation=_evaluation(combined=2.1, fidelity=1.6),
            cof_terminal="COMPLETE",
            tvae_terminal="COMPLETE",
        )
        assert stopped["decision"] == (
            "STOP_EXISTING_COF_FAMILY_MODEL_LEVEL_REDESIGN_REQUIRED"
        )

    invalid = evaluate_sparkov_cof_stop_rule(
        cof_evaluation=_evaluation(combined=1.0, fidelity=1.0, hard_valid=False),
        tvae_evaluation=_evaluation(combined=2.1, fidelity=1.6),
        cof_terminal="COMPLETE",
        tvae_terminal="COMPLETE",
    )
    assert invalid["decision"] == (
        "STOP_EXISTING_COF_FAMILY_MODEL_LEVEL_REDESIGN_REQUIRED"
    )


def test_sparkov_authorization_fails_closed_on_scope_or_stop_rule_change():
    authorization = dict(
        build_sparkov_validation_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            approval_text="source-only Sparkov validation preparation",
        )
    )
    authorization["stop_rule"] = {
        **authorization["stop_rule"],
        "combined_operator": "less_than_or_equal",
    }
    with pytest.raises(ExternalValidationError, match="stop rule"):
        validate_sparkov_validation_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            authorization=authorization,
        )

    authorization = dict(
        build_sparkov_validation_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            approval_text="source-only Sparkov validation preparation",
        )
    )
    authorization["jobs"] = [
        {**authorization["jobs"][0], "dataset": "amlsim"},
        *authorization["jobs"][1:],
    ]
    with pytest.raises(ExternalValidationError):
        validate_sparkov_validation_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            authorization=authorization,
        )

    assert "fraudTest" not in json.dumps(authorization["allowed_operations"])
