from pathlib import Path

import yaml

from scripts.validate_full_experiment_v2_5_preparation import (
    validate_config,
    verify_preparation,
)


def test_frozen_v2_5_config_and_preserved_v2_4_hashes_validate_read_only():
    result = verify_preparation(repository_root=Path("."))
    assert result["status"] == "PASS"
    assert result["generator_count"] == 13
    assert result["v2_4_artifact_count"] == 1679
    assert result["v2_4_gate_report_hash_matches_index"] is True
    assert result["v2_4_full_experiment_authorized"] is False
    assert result["full_experiment_authorized"] is False


def test_config_has_exact_scope_train_only_plan_and_equal_gpu_caps():
    raw = yaml.safe_load(
        Path("configs/benchmark_v2/full_v2_5.yaml").read_text()
    )
    validate_config(raw)
    assert raw["scope"]["scenarios"] == ["joint_semimarkov_v2b"]
    assert raw["scope"]["kappas"] == [1.0]
    assert raw["scope"]["model_seeds"] == [1, 2, 3, 4, 5]
    assert raw["sampling_plan"]["test_labels_or_lengths_used"] is False
    assert raw["sampling_plan"]["seed"] == 10_001
    assert raw["sampling_plan"][
        "shared_by_all_generators_and_model_seeds"
    ] is True
    assert raw["data"]["n_train"] == 31_951
    assert raw["execution"]["full_experiment_authorized"] is False
    for name in ("ctgan_separate_class", "tvae_separate_class"):
        budget = raw["baselines"][name]
        assert budget["max_wall_seconds_total"] == 7200
        assert budget["max_wall_seconds_per_class"] == {
            "0": 3600,
            "1": 3600,
        }


def test_full_cof_config_is_128_2_50_and_separate_from_smoke():
    full = yaml.safe_load(
        Path("configs/benchmark_v2/full_v2_5.yaml").read_text()
    )
    smoke_cof = yaml.safe_load(
        Path("configs/benchmark_v2/cof.yaml").read_text()
    )
    cof = full["baselines"]["cof_seqgen"]
    assert (cof["d_model"], cof["n_layers"], cof["diffusion_steps"]) == (
        128,
        2,
        50,
    )
    assert cof["batch_size"] == 256
    assert cof["optimizer"]["lr"] == 0.001
    assert cof["sampling_chunk_size"] == 256
    assert smoke_cof["smoke"]["diffusion_steps"] == 5
    assert smoke_cof["smoke"]["training_steps"] == 100
    assert full["configuration_role"] == "full_experiment"


def test_primary_endpoint_plugin_names_and_zero_prohibited_runs():
    raw = yaml.safe_load(
        Path("configs/benchmark_v2/full_v2_5.yaml").read_text()
    )
    evaluation = raw["evaluation"]
    assert evaluation["primary_endpoint"]["name"] == (
        "continuous_association_recovery_error"
    )
    assert evaluation["primary_endpoint"]["sole_primary_endpoint"] is True
    assert evaluation["support_diagnostic_bins"] == [4, 8]
    assert "plug_in_hmm" in raw["baselines"]
    assert "plug_in_hsmm" in raw["baselines"]
    assert "hmm" not in raw["baselines"]
    assert "hsmm" not in raw["baselines"]
    assert raw["execution"]["five_seed_run_started"] is False
    assert raw["execution"]["full_experiment_authorized"] is False
    assert raw["execution"]["sweep_authorized"] is False
