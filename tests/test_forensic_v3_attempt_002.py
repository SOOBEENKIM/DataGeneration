from pathlib import Path

import pytest

from scripts import forensic_v3_attempt_002 as forensic
from scripts.forensic_v3_attempt_002 import (
    ForensicV3Error,
    analyze_v3_attempt_002_failure,
    write_forensic_bundle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_independent_recalculation_matches_all_ten_stored_guards():
    report = analyze_v3_attempt_002_failure(REPOSITORY_ROOT)

    expected = {
        "cof_v3_c01_direct_joint": {
            "amount_ks": (0.0023712825288826345, "PASS"),
            "amount_abs_standardized_label_effect": (
                0.025977361556596704,
                "PASS",
            ),
            "gap_ks": (0.08809609370650923, "FAIL"),
            "gap_abs_standardized_label_effect": (
                0.09239624042148471,
                "FAIL",
            ),
            "receiver_max_abs_signed_frequency": (
                0.05518617076377294,
                "FAIL",
            ),
        },
        "cof_v3_c02_factorized_joint": {
            "amount_ks": (0.004754681371885838, "PASS"),
            "amount_abs_standardized_label_effect": (
                0.023221142495914473,
                "PASS",
            ),
            "gap_ks": (0.08206371872349516, "FAIL"),
            "gap_abs_standardized_label_effect": (
                0.47361343948757445,
                "FAIL",
            ),
            "receiver_max_abs_signed_frequency": (
                0.30260494412935485,
                "FAIL",
            ),
        },
    }

    for candidate_id, metrics in expected.items():
        for metric, (value, check) in metrics.items():
            evidence = report["guard_recalculation"][candidate_id][metric]
            assert evidence["independent_value"] == pytest.approx(
                value,
                abs=1e-12,
            )
            assert evidence["stored_value"] == pytest.approx(value, abs=1e-12)
            assert evidence["independent_check"] == check
            assert evidence["stored_check"] == check
            assert evidence["matches_stored"] is True

    assert all(value == 0 for value in report["execution_counts"].values())


def test_discrete_forensic_separates_contract_compliance_from_pmf_failure():
    report = analyze_v3_attempt_002_failure(REPOSITORY_ROOT)

    for candidate_id in (
        "cof_v3_c01_direct_joint",
        "cof_v3_c02_factorized_joint",
    ):
        contract = report["sample_contracts"][candidate_id]
        assert contract["sampling_plan_labels_exact"] is True
        assert contract["sampling_plan_lengths_exact"] is True
        assert contract["sampling_plan_mask_exact"] is True
        assert contract["mask_matches_lengths"] is True
        assert contract["padding_zero"] is True
        assert contract["codec_round_trip_errors"] == 0
        assert contract["joint_states_outside_train_support_mask"] == 0

        support = report["support_coverage"][candidate_id]
        assert support["train_support_mask_allowed_states"] == 1024
        assert support["train_observed_joint_states"] == 1024
        assert support["synthetic_joint_states_in_train_support_fraction"] == 1.0

        for label in ("0", "1"):
            for family in ("gap_bin", "receiver", "joint"):
                comparison = report["classwise_pmf"][candidate_id][label][family]
                assert sum(comparison["real_pmf"]) == pytest.approx(1.0)
                assert sum(comparison["synthetic_pmf"]) == pytest.approx(1.0)
                assert comparison["total_variation"] > 0.0

    direct = report["classwise_pmf"]["cof_v3_c01_direct_joint"]
    factorized = report["classwise_pmf"]["cof_v3_c02_factorized_joint"]
    assert max(direct[label]["joint"]["total_variation"] for label in ("0", "1")) > 0.05
    assert max(factorized[label]["joint"]["total_variation"] for label in ("0", "1")) > 0.20


def test_v2_8_comparison_preserves_amount_pass_and_localizes_discrete_regression():
    report = analyze_v3_attempt_002_failure(REPOSITORY_ROOT)
    comparison = report["v2_8_comparison"]

    assert comparison["v2_8_stored_evaluation_matches_independent"] is True
    assert comparison["v2_8_all_amount_guards_pass"] is True
    for candidate_id in (
        "cof_v3_c01_direct_joint",
        "cof_v3_c02_factorized_joint",
    ):
        candidate = comparison["candidates"][candidate_id]
        assert candidate["v3_all_amount_guards_pass"] is True
        assert candidate["gap_ks_delta_vs_v2_8"] > 0
        assert candidate["receiver_guard_delta_vs_v2_8"] > 0


def test_hypothesis_verdicts_and_append_only_evidence_bundle(tmp_path):
    report = analyze_v3_attempt_002_failure(REPOSITORY_ROOT)

    assert report["hypotheses"]["A_evaluator_mask_label_mapping"]["verdict"] == "REFUTED"
    assert report["hypotheses"]["B_joint_codec_support_decoder_sampling_bug"]["verdict"] == "REFUTED"
    assert report["hypotheses"]["C_v3_objective_architecture_limit"]["verdict"] == "SUPPORTED"
    assert report["code_audit"]["model_v3"]["sha256"]
    assert report["code_audit"]["candidate_sampler"]["sha256"]

    paths = {
        "markdown": tmp_path / "report.md",
        "csv": tmp_path / "evidence.csv",
        "json": tmp_path / "evidence.json",
    }
    hashes = write_forensic_bundle(report=report, **paths)
    assert set(hashes) == {"markdown", "csv", "json"}
    assert "REFUTED" in paths["markdown"].read_text(encoding="utf-8")
    assert "pmf_state" in paths["csv"].read_text(encoding="utf-8")
    assert b"\r\n" not in paths["csv"].read_bytes()
    assert paths["json"].read_text(encoding="utf-8").startswith("{\n")

    with pytest.raises(ForensicV3Error, match="append-only output exists"):
        write_forensic_bundle(report=report, **paths)


def test_numeric_extractor_loads_only_stored_samples_and_frozen_validation(
    monkeypatch,
):
    loaded = []
    original = forensic._load_npz

    def tracked(path, fields):
        loaded.append(Path(path).name)
        return original(path, fields)

    monkeypatch.setattr(forensic, "_load_npz", tracked)
    report = analyze_v3_attempt_002_failure(REPOSITORY_ROOT)

    assert loaded.count("validation.npz") == 1
    assert loaded.count("sample.npz") == 2
    assert loaded.count("validation_sample.npz") == 1
    assert "train.npz" not in loaded
    assert "test.npz" not in loaded
    assert "shared_sampling_plan.npz" not in loaded
    assert report["sampling_plan_provenance"]["frozen_train_npz_loaded"] is False
