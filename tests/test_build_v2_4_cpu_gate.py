from scripts.build_v2_4_cpu_gate import build_report


def fixtures():
    prior = {
        "gate_b_continuous_iid_and_block_ordering": "PASS",
        "gate_c_parametric_oracle": "PASS",
        "gate_a_signed_receiver_frequency": "PASS",
        "gate_a_positionwise_and_segments": "PASS",
        "gate_e_bad_generator_invalid_score_routing": "PASS",
        "gate_e_reference_and_oracle_bin_validity": "PASS",
    }
    auroc = {
        "status": "PASS",
        "clean": {
            "threshold": 0.004,
            "calibration_order_statistic_rank": 200,
            "calibration_seeds": 200,
            "validation_seeds": 200,
            "false_failures": 2,
            "false_fail_ci_high": 0.036,
        },
        "power": [{"detection_power_ci_low": 0.98}] * 12,
    }
    return prior, auroc


def test_v2_4_gate_authorizes_only_learned_smoke_when_everything_passes():
    prior, auroc = fixtures()
    report = build_report(
        prior=prior,
        fanout={"status": "PASS"},
        channel={"status": "PASS"},
        auroc=auroc,
        verification={"pytest_status": "PASS", "compileall_status": "PASS"},
        evidence_paths={},
    )
    assert report["overall_status"] == "PASS"
    assert report["learned_smoke_authorized"] is True
    assert report["full_experiment_authorized"] is False
    assert report["auroc_gate"]["power_cells"] == 12


def test_v2_4_gate_fails_closed_on_any_retained_failure():
    prior, auroc = fixtures()
    prior["gate_c_parametric_oracle"] = "FAIL"
    report = build_report(
        prior=prior,
        fanout={"status": "PASS"},
        channel={"status": "PASS"},
        auroc=auroc,
        verification={"pytest_status": "PASS", "compileall_status": "PASS"},
        evidence_paths={},
    )
    assert report["overall_status"] == "FAIL"
    assert report["learned_smoke_authorized"] is False
