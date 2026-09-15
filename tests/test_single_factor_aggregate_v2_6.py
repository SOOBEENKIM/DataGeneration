from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_single_factor_selection_v2_6 import parse_args
from experiments.single_factor_aggregate_v2_6 import (
    EXPECTED_SCOPE,
    SingleFactorAggregateContractError,
    build_single_factor_selection,
    write_single_factor_aggregate_bundle,
)


def _candidate(
    model_id: str,
    candidate_id: str,
    *,
    passed: bool,
    amount_ks: float,
    gap_ks: float,
):
    return {
        "model_id": model_id,
        "candidate_id": candidate_id,
        "all_five_guards_pass": passed,
        "continuous_ks_max": max(amount_ks, gap_ks),
        "continuous_ks_sum": amount_ks + gap_ks,
        "checks": {
            "amount_ks": "PASS" if passed else "FAIL",
            "gap_ks": "PASS",
            "amount_abs_standardized_label_effect": "PASS",
            "gap_abs_standardized_label_effect": "PASS",
            "receiver_max_abs_signed_frequency": "PASS",
        },
    }


def test_single_factor_selection_keeps_frozen_all_pass_and_tiebreak():
    model_ids = (
        "ctgan_separate_class",
        "tvae_separate_class",
        "cof_seqgen",
    )
    results = []
    for model_id in model_ids:
        results.extend(
            (
                _candidate(
                    model_id,
                    f"{model_id}_failed",
                    passed=False,
                    amount_ks=0.0001,
                    gap_ks=0.0001,
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

    report, manifest = build_single_factor_selection(
        candidate_results=results,
        provenance={"test_split_read": False},
    )

    assert report["primary_c2_selection_ready"] is True
    assert all(
        selection["status"] == "SELECTED"
        for selection in report["model_selections"].values()
    )
    assert all(
        selection["selected_candidate_id"].endswith("_a")
        for selection in report["model_selections"].values()
    )
    assert manifest["status"] == (
        "FROZEN_AWAITING_SEPARATE_FRESH_TEST_AUTHORIZATION"
    )
    assert manifest["fresh_test_authorized"] is False
    assert manifest["test_split_read"] is False


def test_primary_no_pass_keeps_all_downstream_execution_forbidden():
    model_ids = (
        "ctgan_separate_class",
        "tvae_separate_class",
        "cof_seqgen",
    )
    results = [
        _candidate(
            model_id,
            f"{model_id}_{index}",
            passed=model_id != "tvae_separate_class",
            amount_ks=0.004 + index * 0.0001,
            gap_ks=0.003,
        )
        for model_id in model_ids
        for index in range(3)
    ]

    report, manifest = build_single_factor_selection(
        candidate_results=results,
        provenance={"test_split_read": False},
    )

    assert report["primary_c2_selection_ready"] is False
    assert report["blocking_primary_models"] == [
        "tvae_separate_class"
    ]
    assert report["model_selections"]["tvae_separate_class"][
        "status"
    ] == "NO_PASSING_CANDIDATE"
    assert manifest["status"] == (
        "PRIMARY_SELECTION_FAILED_NO_FRESH_TEST"
    )
    assert manifest["selected_candidates"] == {}
    assert manifest["fresh_test_authorized"] is False
    assert manifest["tstr_authorized"] is False
    assert manifest["privacy_authorized"] is False
    assert manifest["five_seed_full_run_authorized"] is False


def test_aggregate_bundle_is_append_only_and_terminal_written_last(
    tmp_path,
):
    output = tmp_path / "aggregate_attempt_001"
    report = {
        "status": "SELECTION_FAILED_PRIMARY_NO_FULL_RUN",
        "primary_c2_selection_ready": False,
        "blocking_primary_models": ["tvae_separate_class"],
    }
    manifest = {
        "status": "PRIMARY_SELECTION_FAILED_NO_FRESH_TEST",
        "test_split_read": False,
    }

    terminal = write_single_factor_aggregate_bundle(
        output_root=output,
        authorization_path=tmp_path / "authorization.json",
        authorization_sha256="a" * 64,
        readiness={"status": "PASS", "worker_count": 3},
        report=report,
        selection_manifest=manifest,
        source_commit="b" * 40,
        relevant_source_sha256="c" * 64,
    )

    assert terminal["status"] == "COMPLETE"
    assert terminal["primary_c2_selection_ready"] is False
    assert (output / "selection_report.json").is_file()
    assert (output / "selection_manifest.json").is_file()
    assert (output / "artifact_index.json").is_file()
    assert (output / "AGGREGATE_COMPLETE.json").is_file()
    index = json.loads(
        (output / "artifact_index.json").read_text(encoding="utf-8")
    )
    assert "AGGREGATE_COMPLETE.json" not in {
        row["path"] for row in index["artifacts"]
    }
    with pytest.raises(
        SingleFactorAggregateContractError,
        match="append-only",
    ):
        write_single_factor_aggregate_bundle(
            output_root=output,
            authorization_path=tmp_path / "authorization.json",
            authorization_sha256="a" * 64,
            readiness={"status": "PASS", "worker_count": 3},
            report=report,
            selection_manifest=manifest,
            source_commit="b" * 40,
            relevant_source_sha256="c" * 64,
        )


def test_aggregate_cli_and_scope_expose_no_execution_permission():
    args = parse_args(
        [
            "--authorization",
            "authorization.json",
            "--source-commit",
            "a" * 40,
            "--mode",
            "plan",
        ]
    )

    assert args.mode == "plan"
    assert EXPECTED_SCOPE == {
        "validation_selection": True,
        "aggregate_only": True,
        "gpu_query": False,
        "cuda": False,
        "model_fit": False,
        "model_sample": False,
        "data_generation": False,
        "test_split_access": False,
        "fresh_test": False,
        "tstr": False,
        "privacy": False,
        "five_seed_full_run": False,
    }
