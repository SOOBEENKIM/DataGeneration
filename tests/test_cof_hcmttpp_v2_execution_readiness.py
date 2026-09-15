from pathlib import Path

from experiments.cof_hcmttpp_v2_execution_readiness import (
    run_h1_attempt_003_readiness_gate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_attempt_003_spawn_ownership_and_h1_numerics_are_execution_ready(tmp_path):
    report = run_h1_attempt_003_readiness_gate(
        repository_root=ROOT,
        scratch_root=tmp_path,
    )

    assert report["status"] == "PASS"
    assert report["scope"] == {
        "candidate_id": "H1",
        "seed": 4001,
        "attempt": "attempt_003",
        "datasets": ["amlsim", "sparkov"],
    }
    for dataset in ("amlsim", "sparkov"):
        result = report["datasets"][dataset]
        assert result["spawn_payload_round_trip"] == "PASS"
        assert result["ownership_attach"] == "PASS"
        assert result["tail_state_fit_split"] == "train"
        assert result["tail_state_nontrain_rows"] == 0
        assert result["optimizer_steps"] == 1
        assert result["finite_loss"] is True
        assert result["finite_gradients"] is True
        assert result["finite_parameters"] is True
        assert result["parameters_changed"] is True
        assert result["routes"] == {
            "zero": True,
            "positive_body": True,
            "central_endpoint": True,
            "positive_tail": True,
            "y0": True,
            "y1": True,
            "padding": True,
        }
        assert result["rqs_forward_inverse_finite"] is True
        assert result["central_tail_boundary_finite"] is True
        assert result["invalid_domain_rejected"] is True

    assert report["mutation_rejections"] == {
        "ownership_id": "REJECTED",
        "dataset": "REJECTED",
        "candidate_id": "REJECTED",
        "seed": "REJECTED",
        "attempt": "REJECTED",
        "authorization_sha256": "REJECTED",
    }
    assert report["execution_counts"] == {
        "gpu_queries": 0,
        "cuda_calls": 0,
        "external_data_body_reads": 0,
        "actual_runtime_root_writes": 0,
        "model_fit_calls": 0,
        "model_sample_calls": 0,
        "evaluation_calls": 0,
        "optimizer_steps_per_dataset": 1,
    }
