import hashlib
from pathlib import Path

import yaml

from models.cof_hcmttpp_v2 import H1_CANDIDATE_ID, PERMANENT_V1_CHAIN_STATE


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmark_v2/cof_hcmttpp_v2_source_only.yaml"
C1_CONFIG = ROOT / "configs/benchmark_v2/cof_ccmtpp_v1_source_only.yaml"


def test_h1_source_config_docs_and_frozen_c1_factor_isolation_agree():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    c1 = yaml.safe_load(C1_CONFIG.read_text(encoding="utf-8"))

    assert config["implementation_status"] == "IMPLEMENTED_SOURCE_ONLY"
    assert config["candidate_count"] == 1
    assert config["h1"]["candidate_id"] == H1_CANDIDATE_ID
    assert config["h1"]["status"] == "IMPLEMENTED_SOURCE_ONLY"
    assert config["h1"]["changed_factors"] == ["gap_decoder"]
    assert config["permanent_v1_stop"]["state"] == PERMANENT_V1_CHAIN_STATE
    implementation = config["implementation"]
    assert implementation["model_path"] == "models/cof_hcmttpp_v2.py"
    assert len(implementation["model_sha256"]) == 64
    assert implementation["public_model_class"] == "CoFHCMTTPPV2H1"
    assert implementation["public_gap_decoder_class"] == "H1HurdleRQSGapDecoder"
    assert implementation["focused_tests"] == [
        "tests/test_cof_hcmttpp_v2_model.py",
        "tests/test_cof_hcmttpp_v2_contract.py",
    ]

    safety = config["safety"]
    assert safety["cpu_synthetic_fixture_tests_authorized"] is True
    for forbidden in (
        "runner_creation_authorized",
        "authorization_creation_authorized",
        "runtime_artifact_creation_authorized",
        "gpu_query_authorized",
        "cuda_authorized",
        "fit_authorized",
        "checkpoint_write_authorized",
        "sample_authorized",
        "evaluation_authorized",
        "internal_test_authorized",
        "sparkov_fraud_test_authorized",
    ):
        assert safety[forbidden] is False

    c1_model = c1["model"]
    h1_budget = config["fixed_model_and_budget"]
    for key in (
        "seed",
        "d_model",
        "n_heads",
        "n_layers",
        "max_length",
        "dropout",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "requested_updates",
        "max_wall_seconds",
        "checkpoint_interval_updates",
    ):
        assert h1_budget[key] == c1_model[key]
    assert h1_budget["receiver_path"] == "flat_no_copy"
    assert h1_budget["amount_contract"] == c1["common_contract"]["amount_contract"]

    architecture = (ROOT / "docs/benchmark_v2/cof_hcmttpp_v2_architecture_specification.md").read_text(encoding="utf-8")
    preregistration = (ROOT / "docs/benchmark_v2/preregistered_cof_hcmttpp_v2.md").read_text(encoding="utf-8")
    artifact_contract = (ROOT / "docs/benchmark_v2/cof_hcmttpp_v2_artifact_contract.md").read_text(encoding="utf-8")
    for public_name in (
        "CoFHCMTTPPV2H1",
        "H1HurdleRQSGapDecoder",
        "TrainOnlyH1TailState",
        "build_h1_checkpoint_bundle",
    ):
        assert public_name in architecture + preregistration + artifact_contract
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() in preregistration
    assert "execution runner is not implemented" in preregistration
