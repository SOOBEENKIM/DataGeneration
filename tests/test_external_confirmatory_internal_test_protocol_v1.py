import csv
import hashlib
import json
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs/benchmark_v2/external_confirmatory_internal_test_v1.yaml"
)
EVIDENCE_PATH = (
    REPOSITORY_ROOT
    / "docs/benchmark_v2/cross_dataset_external_validation_aggregate_v1.json"
)
CSV_PATH = (
    REPOSITORY_ROOT
    / "docs/benchmark_v2/cross_dataset_external_validation_aggregate_v1.csv"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cross_dataset_table_is_an_exact_join_of_frozen_aggregates():
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    inputs = evidence["input_artifacts"]
    aggregates = {
        dataset: json.loads(
            (REPOSITORY_ROOT / record["path"]).read_text(encoding="utf-8")
        )
        for dataset, record in inputs.items()
    }
    assert all(
        _sha256(REPOSITORY_ROOT / record["path"]) == record["sha256"]
        for record in inputs.values()
    )
    amlsim = {item["model"]: item for item in aggregates["amlsim"]["models"]}
    sparkov = {item["model"]: item for item in aggregates["sparkov"]["models"]}
    for item in evidence["models"]:
        model = item["model"]
        assert item["amlsim"] == {
            "fidelity_max_ratio": amlsim[model]["fidelity_max_ratio"],
            "coherence_max_ratio": amlsim[model]["coherence_max_ratio"],
            "combined_score": amlsim[model]["combined_score"],
        }
        assert item["sparkov"] == {
            "fidelity_max_ratio": sparkov[model]["selection"][
                "fidelity_max_ratio"
            ],
            "coherence_max_ratio": sparkov[model]["selection"][
                "coherence_max_ratio"
            ],
            "combined_score": sparkov[model]["selection"]["combined_score"],
        }
        assert item["descriptive_macro"]["combined_score"] == (
            item["amlsim"]["combined_score"]
            + item["sparkov"]["combined_score"]
        ) / 2
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 4


def test_confirmatory_protocol_is_exactly_eight_one_shot_internal_test_jobs():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["status"] == "source_only_not_authorized"
    assert set(config["datasets"]) == {"amlsim", "sparkov"}
    assert config["models"] == [
        "empirical_iid",
        "ctgan_separate_class",
        "tvae_separate_class",
        "cof_seqgen_frozen_non_v3",
    ]
    assert config["execution_plan"]["job_count"] == 8
    assert config["execution_plan"]["one_execution_per_dataset_model"] is True
    assert config["execution_plan"]["retries"] == "forbidden"
    assert all(
        record["test_input"].endswith("/internal_test.npz")
        for record in config["datasets"].values()
    )
    assert (
        config["datasets"]["sparkov"]["sparkov_fraudTest_access"]
        == "forbidden_before_path_resolution"
    )


def test_confirmatory_protocol_freezes_validation_state_and_authorizes_nothing():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = config["frozen_contract"]
    assert frozen["model_training_calls"] == 0
    assert frozen["transform_fit_calls"] == 0
    assert frozen["threshold_fit_calls"] == 0
    assert frozen["seed_changes_allowed"] is False
    assert frozen["model_or_hyperparameter_changes_allowed"] is False
    assert frozen["evaluator_or_formula_changes_allowed"] is False
    assert frozen["conditioning_plan"]["shared_by_all_four_models"] is True
    assert set(config["current_execution_counts"].values()) == {0}
    forbidden = set(config["authorization_scope_required"]["forbidden"])
    assert {
        "training_or_refit",
        "validation_or_test_calibration",
        "seed_change",
        "retry",
        "Sparkov_fraudTest_path_resolution_or_read",
        "TSTR",
        "privacy",
    }.issubset(forbidden)
