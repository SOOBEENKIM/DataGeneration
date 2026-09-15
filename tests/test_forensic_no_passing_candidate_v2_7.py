from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.forensic_no_passing_candidate_v2_7 as forensic
from scripts.forensic_no_passing_candidate_v2_7 import (
    ForensicContractError,
    analyze_no_passing_candidates,
    write_forensic_outputs,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def test_stored_evaluations_match_frozen_aggregate_without_recalculation():
    evidence = analyze_no_passing_candidates(REPOSITORY)

    assert evidence["stored_evaluation_value_comparisons"] == 30
    assert evidence["stored_evaluation_value_matches"] == 30
    assert evidence["stored_guard_decision_comparisons"] == 45
    assert evidence["stored_guard_decision_matches"] == 45
    assert evidence["selection_conclusion_changed"] is False
    assert evidence["primary_c2_selection_ready"] is False
    assert evidence["model_summaries"]["ctgan_separate_class"][
        "selection_status"
    ] == "NO_PASSING_CANDIDATE"
    assert evidence["model_summaries"]["tvae_separate_class"][
        "selected_candidate_id"
    ] == "tvae_v27_c01_amount_inverse_decoder"
    assert evidence["model_summaries"]["cof_seqgen"][
        "selection_status"
    ] == "NO_PASSING_CANDIDATE"
    assert evidence["minimum_failure_components"] == {
        "ctgan_separate_class": [
            "gap_ks",
            "receiver_max_abs_signed_frequency",
        ],
        "cof_seqgen": [
            "gap_abs_standardized_label_effect",
            "gap_ks",
        ],
    }
    rows = {
        row["candidate_id"]: row
        for row in evidence["candidate_rows"]
    }
    assert rows["ctgan_v27_c01_amount_quantile_inverse"][
        "gap_ks"
    ] == pytest.approx(0.04114196583960961)
    assert rows["ctgan_v27_c01_amount_quantile_inverse"][
        "receiver_max_abs_signed_frequency"
    ] == pytest.approx(0.026672031008210265)
    assert rows["cof_v27_c01_empirical_residual"][
        "gap_abs_standardized_label_effect"
    ] == pytest.approx(0.25894890573252216)
    assert rows["cof_v27_c02_gap_logit_bias"]["gap_ks"] == pytest.approx(
        0.009416293536285036
    )
    assert rows["tvae_v27_c01_amount_inverse_decoder"][
        "all_five_guards_pass"
    ] is True
    assert evidence["hypotheses"] == {
        "aggregate_implementation_defect": {
            "verdict": "REFUTED",
            "basis": evidence["hypotheses"][
                "aggregate_implementation_defect"
            ]["basis"],
        },
        "evaluator_implementation_defect": {
            "verdict": "INCONCLUSIVE",
            "basis": evidence["hypotheses"][
                "evaluator_implementation_defect"
            ]["basis"],
        },
        "numeric_decode_sampling_path_limit": {
            "verdict": "SUPPORTED",
            "basis": evidence["hypotheses"][
                "numeric_decode_sampling_path_limit"
            ]["basis"],
        },
        "current_candidate_space_limit": {
            "verdict": "SUPPORTED",
            "basis": evidence["hypotheses"][
                "current_candidate_space_limit"
            ]["basis"],
        },
    }
    assert evidence["execution_counts"] == {
        "gpu_queries": 0,
        "cuda_calls": 0,
        "training_calls": 0,
        "model_restore_calls": 0,
        "model_sample_calls": 0,
        "candidate_reexecution_calls": 0,
        "guard_recalculation_calls": 0,
        "validation_sample_reads": 0,
        "test_split_reads": 0,
        "fresh_test_calls": 0,
        "tstr_calls": 0,
        "privacy_calls": 0,
        "five_seed_full_run_calls": 0,
    }


def test_forensic_statistical_reader_uses_only_report_and_evaluations(
    monkeypatch,
):
    original = forensic._read_json
    read_names: list[str] = []

    def traced(path):
        read_names.append(path.name)
        return original(path)

    monkeypatch.setattr(forensic, "_read_json", traced)
    evidence = analyze_no_passing_candidates(REPOSITORY)

    assert evidence["data_sources"] == {
        "selection_reports": 1,
        "stored_evaluations": 6,
        "validation_samples": 0,
        "test_splits": 0,
    }
    assert read_names.count("selection_report.json") == 1
    assert read_names.count("evaluation.json") == 6
    assert set(read_names) == {"selection_report.json", "evaluation.json"}


def test_forensic_reader_rejects_non_report_or_evaluation_inputs(tmp_path):
    report = tmp_path / "selection_report.json"
    report.write_text("{}", encoding="utf-8")
    forbidden = tmp_path / "validation_sample.npz"
    forbidden.write_bytes(b"forbidden")

    with pytest.raises(
        ForensicContractError,
        match="stored selection_report.json or evaluation.json",
    ):
        analyze_no_passing_candidates(
            REPOSITORY,
            extra_input_paths=[forbidden],
        )


def test_forensic_docs_outputs_are_append_only(tmp_path):
    evidence = analyze_no_passing_candidates(REPOSITORY)
    json_path = tmp_path / "forensic_no_passing_candidate_v2_7.json"
    csv_path = tmp_path / "forensic_no_passing_candidate_v2_7.csv"

    written = write_forensic_outputs(
        evidence=evidence,
        json_path=json_path,
        csv_path=csv_path,
    )

    assert written["json_path"] == str(json_path)
    assert written["csv_path"] == str(csv_path)
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["stored_evaluation_value_matches"] == 30
    with pytest.raises(ForensicContractError, match="append-only"):
        write_forensic_outputs(
            evidence=evidence,
            json_path=json_path,
            csv_path=csv_path,
        )
