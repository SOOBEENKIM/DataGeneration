from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from eval.candidate_preparation_v2_7 import (
    FROZEN_SAMPLING_PLAN_SHA256,
    canonical_sha256,
    sha256_file,
)
from experiments.evaluation_aggregate_v2_7 import (
    AGGREGATE_SCOPE,
    EvaluationAggregateContractError,
    build_v27_selection,
    inspect_v27_terminal_inventory,
    load_v27_stored_guard_results,
    validate_v27_aggregate_authorization,
    write_v27_aggregate_bundle,
)
from experiments.evaluation_only_runner_v2_7 import (
    build_evaluation_execution_plan,
)
from scripts.aggregate_evaluation_selection_v2_7 import parse_args


MODEL_IDS = (
    "ctgan_separate_class",
    "tvae_separate_class",
    "cof_seqgen",
)
REPOSITORY = Path(__file__).resolve().parents[1]


def _candidate(
    model_id: str,
    candidate_id: str,
    *,
    passed: bool,
    amount_ks: float,
    gap_ks: float,
):
    checks = {
        "amount_ks": "PASS" if passed else "FAIL",
        "gap_ks": "PASS",
        "amount_abs_standardized_label_effect": "PASS",
        "gap_abs_standardized_label_effect": "PASS",
        "receiver_max_abs_signed_frequency": "PASS",
    }
    return {
        "model_id": model_id,
        "candidate_id": candidate_id,
        "all_five_guards_pass": passed,
        "checks": checks,
        "statistics": {
            "amount_ks": amount_ks,
            "gap_ks": gap_ks,
        },
        "continuous_ks_max": max(amount_ks, gap_ks),
        "continuous_ks_sum": amount_ks + gap_ks,
    }


def _terminal_fixture(tmp_path: Path):
    root = tmp_path / "repository"
    config_root = root / "configs/benchmark_v2"
    config_root.mkdir(parents=True)
    for name in (
        "evaluation_only_v2_7.yaml",
        "selection_v2_7_source_preparation.yaml",
    ):
        shutil.copy(
            REPOSITORY / "configs/benchmark_v2" / name,
            config_root / name,
        )
    plan = build_evaluation_execution_plan(
        config_root / "evaluation_only_v2_7.yaml"
    )
    runtime = root / "artifacts/benchmark_v2_7/candidate_selection"
    authorization = (
        root
        / "artifacts/benchmark_v2_7/authorizations/execution.json"
    )
    authorization.parent.mkdir(parents=True)
    authorization.write_text(
        '{"status":"AUTHORIZED"}\n',
        encoding="utf-8",
    )
    authorization_hash = sha256_file(authorization)
    source_commit = "d" * 40
    source_hash = "e" * 64
    operations_by_model = {
        model_id: [
            operation
            for operation in plan.operations
            if operation.model_id == model_id
        ]
        for model_id in MODEL_IDS
    }
    for model_id, operations in operations_by_model.items():
        provenance = canonical_sha256(
            {
                "source_commit": source_commit,
                "relevant_source_sha256": source_hash,
                "runner_config_sha256": plan.runner_config_sha256,
                "candidate_config_sha256": (
                    plan.candidate_plan.config_sha256
                ),
                "authorization_sha256": authorization_hash,
                "model_id": model_id,
            }
        )
        worker = runtime / "workers" / model_id
        worker.mkdir(parents=True)
        (worker / "ownership.lock").write_text(
            json.dumps(
                {
                    "schema_version": (
                        "benchmark-v2.7-evaluation-owner-v1"
                    ),
                    "model_id": model_id,
                    "provenance_sha256": provenance,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        terminals = []
        for operation in operations:
            attempt = (
                runtime
                / "evaluations"
                / model_id
                / operation.candidate_id
                / "seed_2601/attempt_001"
            )
            attempt.mkdir(parents=True)
            manifest = {
                "schema_version": (
                    "benchmark-v2.7-evaluation-attempt-manifest-v1"
                ),
                "status": "RUNNING",
                "source_commit": source_commit,
                "relevant_source_sha256": source_hash,
                "runner_config_sha256": plan.runner_config_sha256,
                "candidate_config_sha256": (
                    plan.candidate_plan.config_sha256
                ),
                "authorization_path": str(
                    authorization.relative_to(root)
                ),
                "authorization_sha256": authorization_hash,
                "model_id": model_id,
                "candidate_id": operation.candidate_id,
                "operation_definition_sha256": (
                    operation.definition_sha256
                ),
                "checkpoint_sha256": operation.checkpoint_sha256,
                "sampling_plan_sha256": FROZEN_SAMPLING_PLAN_SHA256,
                "test_split_read": False,
            }
            (attempt / "manifest.json").write_text(
                json.dumps(manifest) + "\n",
                encoding="utf-8",
            )
            if operation.is_control:
                reference = {
                    "schema_version": (
                        "benchmark-v2.7-frozen-control-reference-v1"
                    ),
                    "status": "COMPLETE",
                    "model_id": model_id,
                    "candidate_id": operation.candidate_id,
                    "source_commit": source_commit,
                    "relevant_source_sha256": source_hash,
                    "runner_config_sha256": plan.runner_config_sha256,
                    "candidate_config_sha256": (
                        plan.candidate_plan.config_sha256
                    ),
                    "operation_definition_sha256": (
                        operation.definition_sha256
                    ),
                    "frozen_control": dict(
                        next(
                            candidate
                            for candidate in plan.candidate_plan.candidates
                            if candidate.candidate_id
                            == operation.candidate_id
                        ).frozen_control
                    ),
                    "test_split_read": False,
                }
                reference_path = attempt / "control_reference.json"
                reference_path.write_text(
                    json.dumps(reference) + "\n",
                    encoding="utf-8",
                )
                complete = {
                    "status": "COMPLETE",
                    "control_reference_sha256": sha256_file(
                        reference_path
                    ),
                }
            else:
                sample = attempt / "validation_sample.npz"
                sample.write_bytes(b"fixture-validation-sample")
                evaluation = {
                    "schema_version": (
                        "benchmark-v2.7-five-guard-evaluation-v1"
                    ),
                    "status": "PASS",
                    "model_id": model_id,
                    "candidate_id": operation.candidate_id,
                    "evaluation_split": "validation",
                    "fit_split": "train",
                    "sampling_plan_sha256": (
                        FROZEN_SAMPLING_PLAN_SHA256
                    ),
                    "thresholds": dict(
                        plan.validation_contract["thresholds"]
                    ),
                    "checks": {
                        key: "PASS"
                        for key in plan.validation_contract[
                            "thresholds"
                        ]
                    },
                    "all_five_guards_pass": True,
                    "statistics": {
                        key: 0.0
                        for key in plan.validation_contract[
                            "thresholds"
                        ]
                    },
                    "test_split_read": False,
                }
                evaluation_path = attempt / "evaluation.json"
                evaluation_path.write_text(
                    json.dumps(evaluation) + "\n",
                    encoding="utf-8",
                )
                result = {
                    "status": "COMPLETE",
                    "model_id": model_id,
                    "candidate_id": operation.candidate_id,
                    "source_commit": source_commit,
                    "relevant_source_sha256": source_hash,
                    "runner_config_sha256": plan.runner_config_sha256,
                    "candidate_config_sha256": (
                        plan.candidate_plan.config_sha256
                    ),
                    "authorization_sha256": authorization_hash,
                    "sampling_plan_sha256": (
                        FROZEN_SAMPLING_PLAN_SHA256
                    ),
                    "validation_sample_sha256": sha256_file(sample),
                    "all_five_guards_pass": True,
                    "guard_status": "PASS",
                    "test_split_read": False,
                }
                result_path = attempt / "candidate_result.json"
                result_path.write_text(
                    json.dumps(result) + "\n",
                    encoding="utf-8",
                )
                fit_state = attempt / "fit_state.json"
                fit_state.write_text('{"status":"COMPLETE"}\n')
                complete = {
                    "status": "COMPLETE",
                    "fit_state_sha256": sha256_file(fit_state),
                    "validation_sample_sha256": sha256_file(sample),
                    "evaluation_sha256": sha256_file(evaluation_path),
                    "candidate_result_sha256": sha256_file(
                        result_path
                    ),
                }
            (attempt / "COMPLETE.json").write_text(
                json.dumps(complete) + "\n",
                encoding="utf-8",
            )
            terminals.append(
                {
                    "candidate_id": operation.candidate_id,
                    "status": "COMPLETE",
                    "attempt": str(attempt.relative_to(root)),
                }
            )
        (worker / "WORKER_COMPLETE.json").write_text(
            json.dumps(
                {
                    "schema_version": (
                        "benchmark-v2.7-evaluation-worker-v1"
                    ),
                    "status": "COMPLETE",
                    "model_id": model_id,
                    "provenance_sha256": provenance,
                    "terminals": terminals,
                    "test_split_read": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return root, runtime, plan


def test_selection_requires_all_five_guards_and_uses_frozen_tiebreak():
    candidates = []
    for model_id in MODEL_IDS:
        candidates.extend(
            (
                _candidate(
                    model_id,
                    f"{model_id}_failed",
                    passed=False,
                    amount_ks=0.001,
                    gap_ks=0.001,
                ),
                _candidate(
                    model_id,
                    f"{model_id}_b",
                    passed=True,
                    amount_ks=0.004,
                    gap_ks=0.003,
                ),
                _candidate(
                    model_id,
                    f"{model_id}_a",
                    passed=True,
                    amount_ks=0.004,
                    gap_ks=0.003,
                ),
            )
        )

    report, manifest = build_v27_selection(
        candidate_results=candidates,
        provenance={"test_split_read": False},
    )

    assert report["primary_c2_selection_ready"] is True
    assert all(
        value["selected_candidate_id"].endswith("_a")
        for value in report["model_selections"].values()
    )
    assert manifest["status"] == (
        "FROZEN_AWAITING_SEPARATE_FRESH_TEST_AUTHORIZATION"
    )
    assert manifest["fresh_test_authorized"] is False


def test_primary_no_pass_keeps_every_downstream_action_forbidden():
    candidates = [
        _candidate(
            model_id,
            f"{model_id}_{index}",
            passed=model_id != "cof_seqgen",
            amount_ks=0.003 + index * 0.0001,
            gap_ks=0.002,
        )
        for model_id in MODEL_IDS
        for index in range(3)
    ]

    report, manifest = build_v27_selection(
        candidate_results=candidates,
        provenance={"test_split_read": False},
    )

    assert report["primary_c2_selection_ready"] is False
    assert report["blocking_primary_models"] == ["cof_seqgen"]
    assert report["model_selections"]["cof_seqgen"]["status"] == (
        "NO_PASSING_CANDIDATE"
    )
    assert manifest["selected_candidates"] == {}
    for key in (
        "fresh_test_authorized",
        "tstr_authorized",
        "privacy_authorized",
        "five_seed_full_run_authorized",
    ):
        assert manifest[key] is False


def test_aggregate_bundle_is_append_only_and_terminal_is_last(tmp_path):
    output = tmp_path / "aggregate_attempt_001"
    terminal = write_v27_aggregate_bundle(
        output_root=output,
        authorization_path=tmp_path / "authorization.json",
        authorization_sha256="a" * 64,
        readiness={"status": "PASS", "candidate_count": 9},
        report={
            "primary_c2_selection_ready": False,
            "blocking_primary_models": ["cof_seqgen"],
        },
        selection_manifest={
            "status": "PRIMARY_SELECTION_FAILED_NO_FRESH_TEST",
            "test_split_read": False,
        },
        source_commit="b" * 40,
        relevant_source_sha256="c" * 64,
    )

    assert terminal["status"] == "COMPLETE"
    assert terminal["primary_c2_selection_ready"] is False
    index = json.loads(
        (output / "artifact_index.json").read_text(encoding="utf-8")
    )
    assert "AGGREGATE_COMPLETE.json" not in {
        row["path"] for row in index["artifacts"]
    }
    assert (output / "AGGREGATE_COMPLETE.json").is_file()
    with pytest.raises(
        EvaluationAggregateContractError,
        match="append-only",
    ):
        write_v27_aggregate_bundle(
            output_root=output,
            authorization_path=tmp_path / "authorization.json",
            authorization_sha256="a" * 64,
            readiness={"status": "PASS", "candidate_count": 9},
            report={
                "primary_c2_selection_ready": False,
                "blocking_primary_models": ["cof_seqgen"],
            },
            selection_manifest={
                "status": "PRIMARY_SELECTION_FAILED_NO_FRESH_TEST"
            },
            source_commit="b" * 40,
            relevant_source_sha256="c" * 64,
        )


def test_inventory_requires_exactly_three_workers_and_nine_candidates(
    tmp_path,
):
    root, runtime, plan = _terminal_fixture(tmp_path)

    inventory = inspect_v27_terminal_inventory(
        repository_root=root,
        runtime_root=runtime,
        plan=plan,
    )

    assert inventory["status"] == "PASS"
    assert inventory["worker_count"] == 3
    assert inventory["candidate_count"] == 9
    assert set(inventory["worker_terminals"]) == set(MODEL_IDS)
    assert len(inventory["candidate_artifacts"]) == 9

    (
        runtime
        / "workers/cof_seqgen/WORKER_FAILED.json"
    ).write_text('{"status":"FAILED"}\n', encoding="utf-8")
    with pytest.raises(
        EvaluationAggregateContractError,
        match="exactly one terminal",
    ):
        inspect_v27_terminal_inventory(
            repository_root=root,
            runtime_root=runtime,
            plan=plan,
        )


def test_stored_guard_loader_uses_no_sample_or_re_evaluation(tmp_path):
    root, runtime, plan = _terminal_fixture(tmp_path)
    inventory = inspect_v27_terminal_inventory(
        repository_root=root,
        runtime_root=runtime,
        plan=plan,
    )
    prior_root = (
        root
        / "artifacts/benchmark_v2_6/selection_single_factor/"
        "aggregate_attempt_001"
    )
    prior_root.mkdir(parents=True)
    prior_candidates = []
    for operation in plan.operations:
        if not operation.is_control:
            continue
        reference_path = (
            runtime
            / "evaluations"
            / operation.model_id
            / operation.candidate_id
            / "seed_2601/attempt_001/control_reference.json"
        )
        frozen = json.loads(
            reference_path.read_text(encoding="utf-8")
        )["frozen_control"]
        prior_candidates.append(
            {
                "model_id": operation.model_id,
                "candidate_id": f"prior_{operation.candidate_id}",
                "candidate_result_sha256": frozen[
                    "candidate_result_sha256"
                ],
                "validation_sample_sha256": frozen[
                    "validation_sample_sha256"
                ],
                "checkpoint_sha256": frozen["checkpoint_sha256"],
                "all_five_guards_pass": False,
                "checks": {
                    key: "FAIL"
                    for key in plan.validation_contract["thresholds"]
                },
                "statistics": {
                    key: float(
                        plan.validation_contract["thresholds"][key]
                    )
                    + 1.0
                    for key in plan.validation_contract["thresholds"]
                },
                "thresholds": dict(
                    plan.validation_contract["thresholds"]
                ),
            }
        )
    report_path = prior_root / "selection_report.json"
    report_path.write_text(
        json.dumps({"candidate_results": prior_candidates}) + "\n",
        encoding="utf-8",
    )
    (prior_root / "AGGREGATE_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "selection_report_sha256": sha256_file(report_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    results, provenance = load_v27_stored_guard_results(
        repository_root=root,
        runtime_root=runtime,
        plan=plan,
        inventory=inventory,
    )

    assert len(results) == 9
    assert sum(row["source_kind"] == "frozen_control" for row in results) == 3
    assert sum(row["source_kind"] == "stored_v2_7_evaluation" for row in results) == 6
    assert provenance["validation_samples_read"] == 0
    assert provenance["guard_evaluations_executed"] == 0
    assert provenance["test_split_read"] is False


def test_aggregate_authorization_binds_exact_inventory_and_forbidden_scope(
    tmp_path,
):
    root, runtime, plan = _terminal_fixture(tmp_path)
    inventory = inspect_v27_terminal_inventory(
        repository_root=root,
        runtime_root=runtime,
        plan=plan,
    )
    frozen = plan.candidate_plan.frozen_contract
    authorization = {
        "schema_version": (
            "benchmark-v2.7-aggregate-only-authorization-v1"
        ),
        "status": "AUTHORIZED",
        "approval_text": "validation-only aggregate of stored artifacts",
        "aggregate_source_commit": "f" * 40,
        "aggregate_relevant_source_sha256": "1" * 64,
        "runner_config_sha256": plan.runner_config_sha256,
        "candidate_config_sha256": (
            plan.candidate_plan.config_sha256
        ),
        "candidate_source_commit": inventory[
            "candidate_source_commit"
        ],
        "candidate_relevant_source_sha256": inventory[
            "candidate_relevant_source_sha256"
        ],
        "candidate_execution_authorization_path": inventory[
            "candidate_execution_authorization_path"
        ],
        "candidate_execution_authorization_sha256": inventory[
            "candidate_execution_authorization_sha256"
        ],
        "input_inventory_sha256": inventory["inventory_sha256"],
        "input_tree": inventory["input_tree"],
        "approved_worker_terminals": inventory["worker_terminals"],
        "approved_candidate_artifacts": inventory[
            "candidate_artifacts"
        ],
        "frozen_provenance": {
            key: frozen[key]
            for key in (
                "development_manifest_sha256",
                "train_file_sha256",
                "train_content_sha256",
                "validation_file_sha256",
                "validation_content_sha256",
                "sampling_plan_sha256",
                "v2_5_frozen_manifest_sha256",
                "v2_5_final_complete_sha256",
                "v2_6_aggregate_complete_sha256",
                "v2_6_aggregate_index_sha256",
            )
        },
        "scope": AGGREGATE_SCOPE,
    }

    validate_v27_aggregate_authorization(
        plan=plan,
        inventory=inventory,
        authorization=authorization,
        source_commit="f" * 40,
        relevant_source_sha256="1" * 64,
    )

    tampered = {**authorization, "input_inventory_sha256": "0" * 64}
    with pytest.raises(
        EvaluationAggregateContractError,
        match="authorization mismatch",
    ):
        validate_v27_aggregate_authorization(
            plan=plan,
            inventory=inventory,
            authorization=tampered,
            source_commit="f" * 40,
            relevant_source_sha256="1" * 64,
        )


def test_aggregate_cli_has_only_authorized_cpu_modes():
    args = parse_args(
        [
            "--mode",
            "plan",
            "--authorization",
            "aggregate_authorization.json",
            "--source-commit",
            "a" * 40,
        ]
    )

    assert args.mode == "plan"
    assert not hasattr(args, "device")
    assert AGGREGATE_SCOPE["gpu_query"] is False
    assert AGGREGATE_SCOPE["cuda"] is False
    assert AGGREGATE_SCOPE["model_sample"] is False
    assert AGGREGATE_SCOPE["test_split_access"] is False
    assert AGGREGATE_SCOPE["threshold_change"] is False
    with pytest.raises(SystemExit):
        parse_args(["--mode", "dry-run"])


def test_inventory_rejects_test_split_or_threshold_drift(tmp_path):
    root, runtime, plan = _terminal_fixture(tmp_path)
    attempt = (
        runtime
        / "evaluations/tvae_separate_class/"
        "tvae_v27_c01_amount_inverse_decoder/"
        "seed_2601/attempt_001"
    )
    evaluation_path = attempt / "evaluation.json"
    evaluation = json.loads(
        evaluation_path.read_text(encoding="utf-8")
    )
    evaluation["evaluation_split"] = "test"
    evaluation["thresholds"]["amount_ks"] = 1.0
    evaluation_path.write_text(
        json.dumps(evaluation) + "\n",
        encoding="utf-8",
    )
    complete_path = attempt / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["evaluation_sha256"] = sha256_file(evaluation_path)
    complete_path.write_text(
        json.dumps(complete) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationAggregateContractError,
        match="stored evaluation/result mismatch",
    ):
        inspect_v27_terminal_inventory(
            repository_root=root,
            runtime_root=runtime,
            plan=plan,
        )
