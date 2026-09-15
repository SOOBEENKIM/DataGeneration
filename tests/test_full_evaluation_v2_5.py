from pathlib import Path

import numpy as np
import yaml

from benchmarks.temporal_coupling_v2 import (
    BenchmarkConfig,
    generate_benchmark,
)
from benchmarks.types import SyntheticBatch
from eval.full_evaluation_v2_5 import (
    evaluate_full_seed,
    fit_full_evaluation_reference,
)
from eval.model_guards_v2_5 import RowGuardThresholds
from experiments.full_artifact_store_v2_5 import FullAttemptStore
from generators.sampling_plan import SamplingPlan
from scripts.run_full_experiment_v2_5 import (
    _collect_raw_records,
    _support_diagnostic_rows,
    build_job_plan,
    render_support_diagnostics_markdown,
    terminalize_evaluation_attempt,
)


def _row_guard_pass_support_fail_fixture():
    bundle = generate_benchmark(
        BenchmarkConfig(
            scenario="joint_semimarkov_v2b",
            kappa=1.0,
            n_train=160,
            n_test=80,
            min_length=8,
            max_length=8,
            fraud_rate=0.5,
            n_gap_bins=8,
            n_receiver_categories=8,
        ),
        42,
    )
    tau = np.asarray(bundle.metadata["tau"])
    plan = SamplingPlan.from_batch(bundle.test)
    synthetic = SyntheticBatch(
        x_num=bundle.test.x_num.copy(),
        dt_bin=bundle.test.dt_bin.copy(),
        x_cat=bundle.test.x_cat.copy(),
        valid_mask=bundle.test.valid_mask.copy(),
        y_entity=bundle.test.y_entity.copy(),
        lengths=bundle.test.lengths.copy(),
    )
    thresholds = RowGuardThresholds(
        amount_ks=1.0,
        gap_ks=1.0,
        amount_abs_standardized_label_effect=10.0,
        gap_abs_standardized_label_effect=10.0,
        receiver_max_abs_signed_frequency=1.0,
        calibration_trials=200,
        calibration_seed=24500,
    )
    reference = fit_full_evaluation_reference(
        bundle.train,
        bundle.test,
        tau=tau,
        window_width=4.0,
        minimum_bin_count=20,
        row_guard_thresholds=thresholds,
        v2_4_reference_contract_verified=True,
    )
    return bundle, plan, reference, synthetic, tau


def test_support_diagnostic_failure_keeps_primary_seed_valid_and_numeric():
    bundle, plan, reference, synthetic, tau = (
        _row_guard_pass_support_fail_fixture()
    )

    evaluation = evaluate_full_seed(
        synthetic,
        train=bundle.train,
        real_test=bundle.test,
        plan=plan,
        tau=tau,
        window_width=4.0,
        reference=reference,
    )

    support = evaluation["diagnostics"]["support_diagnostics"]
    confirmatory = evaluation["diagnostics"]["confirmatory_support"]
    assert evaluation["row_marginal_guard_report"]["status"] == "PASS"
    assert confirmatory["all_bins_valid"] is False
    assert evaluation["status"] == "VALID"
    assert np.isfinite(evaluation["association_recovery_error"])
    assert "confirmatory_support" not in evaluation["hard_guards"]
    assert set(support) == {"4_bin", "8_bin"}
    assert support["8_bin"]["all_bins_valid"] is False
    assert support["8_bin"]["dropped_bin_macro_gap"] is not None
    assert support["8_bin"]["occupancy_penalized_gap"] is not None
    assert support["8_bin"]["invalid_bin_count"] > 0


def test_diagnostic_failure_completes_attempt_and_enters_aggregate_seam(
    tmp_path,
):
    bundle, plan, reference, synthetic, tau = (
        _row_guard_pass_support_fail_fixture()
    )
    evaluation = evaluate_full_seed(
        synthetic,
        train=bundle.train,
        real_test=bundle.test,
        plan=plan,
        tau=tau,
        window_width=4.0,
        reference=reference,
    )
    manifest = {
        "schema_version": "benchmark-v2.5-full-attempt",
        "git_commit": "a" * 40,
        "config_hash": "b" * 64,
        "code_hash": "c" * 64,
        "scenario": "joint_semimarkov_v2b",
        "kappa": 1.0,
        "generator": "empirical_iid",
        "seed": 1,
        "sampling_plan_hash": "1" * 64,
        "data_hashes": {
            "train": "d" * 64,
            "validation": "e" * 64,
            "test": "f" * 64,
        },
        "gpu": {"id": None},
        "cuda_version": None,
        "pytorch_version": "fixture",
        "requested_training_budget": {
            "steps": 0,
            "max_wall_seconds": 10,
        },
        "actual_training_budget": {"steps": 0, "wall_seconds": 0.1},
        "baseline_definition_version": "benchmark-v2.5",
        "evaluation_version": "benchmark-v2.5-evaluation-v1",
    }
    artifact_root = tmp_path / "benchmark_v2_5"
    store, _ = FullAttemptStore.select(artifact_root, manifest)
    for relative in ("sample.npz", "checkpoints/final.pt"):
        store.write_immutable_bytes(relative, b"fixture")
    store.write_immutable_json(
        "metrics.json",
        {
            "hard_guards": evaluation["hard_guards"],
            "association_recovery_error": evaluation[
                "association_recovery_error"
            ],
            "support_diagnostics": evaluation["diagnostics"][
                "support_diagnostics"
            ],
        },
    )
    store.write_immutable_json("runtime.json", {"total_seconds": 0.1})
    store.write_immutable_json("evaluation.json", evaluation)

    status, marker = terminalize_evaluation_attempt(
        store=store,
        evaluation=evaluation,
        completion_payload={
            "association_recovery_error": evaluation[
                "association_recovery_error"
            ],
            "hard_guards": evaluation["hard_guards"],
            "sampling_plan_hash": "1" * 64,
        },
    )

    raw = yaml.safe_load(
        Path("configs/benchmark_v2/full_v2_5.yaml").read_text()
    )
    job = next(
        value
        for value in build_job_plan(raw, sampling_plan_hash="1" * 64)
        if value.generator == "empirical_iid" and value.seed == 1
    )
    records, rows = _collect_raw_records(artifact_root, [job])
    assert status == "COMPLETE"
    assert marker.name == "COMPLETE.json"
    assert not (store.path / "INVALID.json").exists()
    assert records["empirical_iid"][1]["status"] == "COMPLETE"
    assert np.isfinite(
        records["empirical_iid"][1]["association_recovery_error"]
    )
    assert rows[0]["support_8_all_bins_valid"] is False
    assert rows[0]["support_8_invalid_bin_count"] > 0

    diagnostic_rows = _support_diagnostic_rows(rows)
    report = render_support_diagnostics_markdown(diagnostic_rows)
    assert {row["bin_count"] for row in diagnostic_rows} == {4, 8}
    assert {
        "all_bins_valid",
        "dropped_bin_macro_gap",
        "occupancy_penalized_gap",
        "invalid_bin_count",
    } <= diagnostic_rows[0].keys()
    assert "4-bin" in report
    assert "8-bin" in report
    assert "Dropped-bin score" in report
    assert "Occupancy-penalized score" in report
    assert "Invalid bins" in report
