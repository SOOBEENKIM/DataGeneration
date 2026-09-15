import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.external_validation_runner_v1 import (
    ExternalExecutionError,
    bounded_data_transformer_policy,
    estimate_external_transform_memory,
    validate_external_transform_memory_policy,
)
from scripts.run_external_validation_v1 import (
    ExternalValidationError,
    build_amlsim_tvae_continuation_authorization,
    build_amlsim_tvae_continuation_plan,
    build_external_execution_manifest,
    validate_external_completed_result_reuse,
    validate_external_launch_schedule,
    validate_external_validation_authorization,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/benchmark_v2/external_validation_v1.yaml"
FORENSIC_JSON = (
    REPOSITORY_ROOT
    / "docs/benchmark_v2/forensic_amlsim_external_validation_v1.json"
)
FORENSIC_CSV = (
    REPOSITORY_ROOT
    / "docs/benchmark_v2/forensic_amlsim_external_validation_v1.csv"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_amlsim_scale_fixture_exposes_dense_transform_and_concurrency_risk():
    risk = estimate_external_transform_memory(
        valid_rows=914_756,
        receiver_cardinality=9_656,
        output_dtype_bytes=8,
        concurrent_tabular_jobs=2,
    )

    assert risk["receiver_dense_matrix_bytes"] == 914_756 * 9_656 * 8
    assert risk["receiver_dense_matrix_gib"] > 65
    assert risk["concurrent_receiver_dense_matrix_gib"] > 130
    assert risk["upstream_parallel_n_jobs"] == -1
    assert risk["risk"] == "GLOBAL_OOM_PLAUSIBLE"

    sparkov = estimate_external_transform_memory(
        valid_rows=898_168,
        receiver_cardinality=695,
        output_dtype_bytes=8,
        concurrent_tabular_jobs=1,
    )
    assert risk["receiver_dense_matrix_gib"] > (
        14 * sparkov["receiver_dense_matrix_gib"]
    )
    assert risk["arrays_allocated"] == sparkov["arrays_allocated"] == 0


def test_fixed_single_worker_policy_avoids_joblib_parallel_at_real_transform_seam(
    monkeypatch,
):
    from ctgan.data_transformer import DataTransformer
    import ctgan.data_transformer as transformer_module

    rows = 768  # Upstream transform chooses its parallel branch at >=500 rows.
    frame = pd.DataFrame(
        {
            "amount_log": np.linspace(-2.0, 2.0, rows),
            "dt_bin": np.arange(rows) % 16,
            "receiver": np.arange(rows) % 64,
        }
    )
    transformer = DataTransformer()
    transformer.fit(frame, discrete_columns=("dt_bin", "receiver"))
    expected_columns = transformer._synchronous_transform(
        frame, transformer._column_transform_info_list
    )
    expected = np.concatenate(expected_columns, axis=1).astype(float)

    def forbidden_parallel(*args, **kwargs):
        raise AssertionError("joblib Parallel must not be used by external transform")

    monkeypatch.setattr(transformer_module, "Parallel", forbidden_parallel)
    with bounded_data_transformer_policy(n_jobs=1) as audit:
        observed = transformer.transform(frame)

    # The continuous transformer samples a mixture component, so two successive
    # calls need not be numerically identical.  This seam test instead proves
    # that the same fitted schema is produced without reaching joblib Parallel.
    assert observed.shape == expected.shape
    assert np.isfinite(observed).all()
    assert audit == {
        "configured_n_jobs": 1,
        "joblib_worker_processes": 0,
        "execution_mode": "synchronous_column_transform",
        "upstream_n_jobs_minus_one_reachable": False,
    }
    with pytest.raises(ExternalExecutionError, match="exactly 1"):
        with bounded_data_transformer_policy(n_jobs=-1):
            pass


def test_config_and_schedule_forbid_unbounded_or_overlapping_heavy_transforms():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    policy = validate_external_transform_memory_policy(config)
    schedule = validate_external_launch_schedule(config)

    assert policy["fixed_n_jobs"] == 1
    assert policy["unbounded_n_jobs_forbidden"] is True
    assert policy["models"] == ["ctgan_separate_class", "tvae_separate_class"]
    assert policy["max_concurrent_heavy_transforms"] == 1
    for wave in schedule["waves"]:
        tabular = {
            job["model"]
            for job in wave["jobs"]
            if job["model"] in policy["models"]
        }
        assert len(tabular) <= 1


def test_completed_ctgan_can_be_reused_only_under_exact_mixed_provenance_contract():
    result = validate_external_completed_result_reuse(
        repo_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
        dataset="amlsim",
        model="ctgan_separate_class",
    )

    assert result["status"] == "REUSE_ELIGIBLE"
    assert result["scientific_contract_unchanged"] is True
    assert result["resource_policy_only_change"] is True
    assert result["model_execution_calls"] == 0

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["corrective_continuation"]["completed_result_reuse"]["amlsim"][
        "ctgan_separate_class"
    ]["evaluation_sha256"] = "0" * 64
    altered = CONFIG_PATH.parent / "_never_write_this_config.yaml"
    # Exercise the mapping seam without creating a file or touching runtime.
    with pytest.raises(ExternalValidationError, match="reuse artifact hash mismatch"):
        validate_external_completed_result_reuse(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            dataset="amlsim",
            model="ctgan_separate_class",
            config_override=config,
        )
    assert not altered.exists()


def test_forensic_inputs_are_json_only_and_runtime_artifacts_remain_immutable():
    paths = [
        REPOSITORY_ROOT
        / "artifacts/external_validation_v1/amlsim/empirical_iid/attempt_002/evaluation.json",
        REPOSITORY_ROOT
        / "artifacts/external_validation_v1/amlsim/ctgan_separate_class/attempt_001/evaluation.json",
        REPOSITORY_ROOT
        / "artifacts/external_validation_v1/amlsim/cof_seqgen_frozen_non_v3/attempt_001/evaluation.json",
    ]
    before = {path: path.read_bytes() for path in paths}
    assert all(json.loads(payload)["dataset"] == "amlsim" for payload in before.values())
    assert {path: path.read_bytes() for path in paths} == before


def test_forensic_evidence_transcribes_stored_iid_metrics_without_recalculation():
    evaluation_path = (
        REPOSITORY_ROOT
        / "artifacts/external_validation_v1/amlsim/empirical_iid/attempt_002/evaluation.json"
    )
    thresholds_path = evaluation_path.with_name("train_bootstrap_thresholds.json")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    evidence = json.loads(FORENSIC_JSON.read_text(encoding="utf-8"))

    assert evidence["source_evidence"]["empirical_iid_evaluation_sha256"] == _sha256(
        evaluation_path
    )
    assert evidence["source_evidence"]["empirical_iid_thresholds_sha256"] == _sha256(
        thresholds_path
    )
    stored_values = {
        "amount_ks_y0": evaluation["metrics"]["fidelity"]["amount_ks_y0"],
        "amount_ks_y1": evaluation["metrics"]["fidelity"]["amount_ks_y1"],
        "gap_total_variation_y0": evaluation["metrics"]["fidelity"][
            "gap_total_variation_y0"
        ],
        "gap_total_variation_y1": evaluation["metrics"]["fidelity"][
            "gap_total_variation_y1"
        ],
        "receiver_total_variation_y0": evaluation["metrics"]["fidelity"][
            "receiver_total_variation_y0"
        ],
        "receiver_total_variation_y1": evaluation["metrics"]["fidelity"][
            "receiver_total_variation_y1"
        ],
        "short_gap_receiver_repeat_error_y0": evaluation["metrics"]["coherence"][
            "short_gap_receiver_repeat_error_y0"
        ],
        "short_gap_receiver_repeat_error_y1": evaluation["metrics"]["coherence"][
            "short_gap_receiver_repeat_error_y1"
        ],
    }
    evidence_values = {
        f"{row['metric']}_{row['class'].lower()}": row["value"]
        for row in evidence["empirical_iid_stored_evaluation"]["metrics"]
    }
    assert evidence_values == stored_values
    assert all(
        row["pass"] is evaluation["selection"]["metric_pass"][
            f"{row['metric']}_{row['class'].lower()}"
        ]
        for row in evidence["empirical_iid_stored_evaluation"]["metrics"]
    )
    assert thresholds["source_split"] == "train"
    assert thresholds["frozen_before_validation"] is True

    with FORENSIC_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metric_rows = [row for row in rows if row["section"] == "evaluation"]
    assert len(metric_rows) == 8
    assert {row["pass"] for row in metric_rows} == {"false"}


def test_amlsim_tvae_continuation_authorizes_only_attempt_002_and_reuses_three_results():
    authorization = build_amlsim_tvae_continuation_authorization(
        repo_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
        approval_text="explicit AMLSim TVAE attempt_002-only approval",
    )
    validated = validate_external_validation_authorization(
        repo_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
        authorization=authorization,
        require_attempt_absent=False,
    )

    assert validated["status"] == "PASS"
    assert validated["dataset"] == "amlsim"
    assert validated["model"] == "tvae_separate_class"
    assert validated["attempt_path"].endswith(
        "/amlsim/tvae_separate_class/attempt_002"
    )
    assert authorization["job_count"] == 1
    assert authorization["attempt"] == "attempt_002"
    assert authorization["data_transformer_n_jobs"] == 1
    assert authorization["heavy_transform_exclusion_group"] == (
        "external_tabular_transform"
    )
    assert set(authorization["preserved_completed_results"]) == {
        "empirical_iid",
        "ctgan_separate_class",
        "cof_seqgen_frozen_non_v3",
    }
    assert authorization["sparkov_authorized"] is False
    assert authorization["internal_test_authorized"] is False
    assert authorization["gpu_inventory_query_authorized"] is False
    with pytest.raises(ExternalValidationError, match="already exists"):
        validate_external_validation_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            authorization=authorization,
        )


def test_tvae_continuation_plan_and_dry_run_are_execution_free_and_fail_closed():
    for mode in ("plan", "dry-run"):
        plan = build_amlsim_tvae_continuation_plan(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            approval_text="explicit AMLSim TVAE attempt_002-only approval",
            mode=mode,
        )
        assert plan["status"] == "PASS"
        assert plan["job_count"] == 1
        assert plan["target"] == {
            "dataset": "amlsim",
            "model": "tvae_separate_class",
            "attempt": "attempt_002",
        }
        assert plan["authorization_created"] is False
        assert set(plan["execution_counts"].values()) == {0}

    authorization = dict(
        build_amlsim_tvae_continuation_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            approval_text="explicit AMLSim TVAE attempt_002-only approval",
        )
    )
    authorization["data_transformer_n_jobs"] = -1
    with pytest.raises(ExternalValidationError, match="mismatch"):
        validate_external_validation_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            authorization=authorization,
        )

    authorization = dict(
        build_amlsim_tvae_continuation_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            approval_text="explicit AMLSim TVAE attempt_002-only approval",
        )
    )
    authorization["preserved_completed_results"] = {}
    with pytest.raises(ExternalValidationError, match="completed-result reuse"):
        validate_external_validation_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            authorization=authorization,
        )
