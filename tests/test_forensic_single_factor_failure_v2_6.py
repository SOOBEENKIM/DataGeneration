from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.forensic_single_factor_failure_v2_6 import (
    ForensicContractError,
    analyze_single_factor_failure,
    write_forensic_evidence,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def test_real_stored_guard_values_match_independent_recalculation():
    evidence = analyze_single_factor_failure(REPOSITORY)

    assert evidence["guard_value_comparisons"] == 45
    assert evidence["guard_value_matches"] == 45
    assert evidence["all_stored_guard_values_match"] is True
    assert evidence["all_sample_contracts_pass"] is True
    assert {
        model: value["selection_status"]
        for model, value in evidence["model_summaries"].items()
    } == {
        "ctgan_separate_class": "NO_PASSING_CANDIDATE",
        "tvae_separate_class": "NO_PASSING_CANDIDATE",
        "cof_seqgen": "NO_PASSING_CANDIDATE",
    }
    assert evidence["hypotheses"]["A"]["verdict"] == "REFUTED"
    assert evidence["hypotheses"]["B"]["verdict"] == "SUPPORTED"
    assert evidence["hypotheses"]["C"]["verdict"] == "SUPPORTED"
    assert evidence["execution_counts"] == {
        "gpu_queries": 0,
        "cuda_calls": 0,
        "model_fit_calls": 0,
        "model_sample_calls": 0,
        "data_generation_calls": 0,
        "test_split_reads": 0,
        "fresh_test_calls": 0,
        "tstr_calls": 0,
        "privacy_calls": 0,
        "five_seed_full_run_calls": 0,
    }


def test_forensic_evidence_bundle_is_append_only(tmp_path):
    evidence = analyze_single_factor_failure(REPOSITORY)
    output = tmp_path / "forensic_attempt_001"

    terminal = write_forensic_evidence(
        output_root=output,
        evidence=evidence,
    )

    assert terminal["status"] == "COMPLETE"
    assert (output / "candidate_guard_evidence.csv").is_file()
    assert (output / "forensic_evidence.json").is_file()
    assert (output / "forensic_artifact_index.json").is_file()
    assert (output / "FORENSIC_COMPLETE.json").is_file()
    parsed = json.loads(
        (output / "forensic_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert parsed["all_stored_guard_values_match"] is True
    with pytest.raises(ForensicContractError, match="append-only"):
        write_forensic_evidence(
            output_root=output,
            evidence=evidence,
        )
