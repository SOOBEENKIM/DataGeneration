from copy import deepcopy
from pathlib import Path

import yaml

from scripts.calibrate_auroc_gate_v2_4 import (
    integrated_clean_t,
    task_key,
    worker,
)


def test_cell_seed_worker_uses_validation_orientation_and_fixed_operators():
    raw = yaml.safe_load(
        Path("configs/benchmark_v2/main_v2_3_candidate.yaml").read_text()
    )
    raw = deepcopy(raw)
    raw["data"]["fraud_rate"] = 0.2
    payload = {
        "benchmark_config": raw,
        "base_counts": {
            "train_entities": 2_000,
            "orientation_validation_entities": 1_000,
            "test_entities": 2_000,
        },
        "hgb_max_iter": 2,
        "leakage_magnitudes": [0.02, 0.05],
        "threads_per_worker": 1,
        "partition": "validation",
        "audit_seed": 2000,
        "validation_trial_ordinal": 0,
        "scenario": "joint_semimarkov_v2b",
        "kappa": 1.0,
        "multiplier": 1,
        "include_leakage": True,
    }
    result = worker(payload)
    assert result["task_key"] == task_key(payload)
    assert len(result["clean_rows"]) == 2
    assert len(result["leakage_rows"]) == 12
    assert {
        row["feature"] for row in result["leakage_rows"]
    } == {"amount", "gap", "receiver_category"}
    assert all(
        row["test_labels_used_for_orientation"] is False
        for row in result["clean_rows"] + result["leakage_rows"]
    )
    assert all(
        row["train_realised_probability_change"] > 0
        for row in result["leakage_rows"]
    )


def test_integrated_clean_t_requires_all_eight_dimensions():
    rows = []
    for classifier in ("hist_gradient_boosting", "logistic"):
        for scenario in ("joint_semimarkov_v2b", "markov_persistence_v2a"):
            for kappa in (0.0, 1.0):
                rows.append(
                    {
                        "partition": "validation",
                        "audit_seed": 2000,
                        "classifier": classifier,
                        "scenario": scenario,
                        "kappa": kappa,
                        "test_auroc": 0.51,
                    }
                )
    import pandas as pd

    result = integrated_clean_t(pd.DataFrame(rows), "validation")
    assert len(result) == 1
    assert abs(result.iloc[0]["T_global_8"] - 0.01) < 1e-12
