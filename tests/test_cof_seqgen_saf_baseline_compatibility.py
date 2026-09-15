from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.cof_seqgen_saf_contract import CanonicalEntitySequenceDataset, CanonicalSchema
from generators.cof_seqgen_saf_baselines import (
    BASELINE_SPECS,
    FLATTENED_CONTROLS,
    PRIMARY_SEQUENTIAL,
    BaselineCompatibilityError,
    build_shared_generation_plan,
    comparison_manifest,
    dependency_readiness,
    materialize_compatibility_view,
    validate_raw_generated_events,
)


def _dataset() -> CanonicalEntitySequenceDataset:
    ids = [f"e{i}" for i in range(10)]
    static = pd.DataFrame({"entity_id": ids, "segment": [i % 2 for i in range(10)]})
    splits = pd.DataFrame({"entity_id": ids, "split": ["train"] * 7 + ["validation"] + ["test"] * 2})
    rows = []
    for entity, length in zip(ids, range(2, 12)):
        time = 0.0
        for index in range(length):
            gap = np.nan if index == 0 else float(index % 3)
            if index:
                time += gap
            rows.append(
                {
                    "entity_id": entity,
                    "event_id": f"{entity}-{index}",
                    "event_index": index,
                    "timestamp": time,
                    "gap": gap,
                    "receiver_or_mark": f"m{index % 3}",
                    "amount_or_numeric_value": float(index),
                }
            )
    return CanonicalEntitySequenceDataset(
        CanonicalSchema("toy", "absolute", "seconds", static_context_columns=("segment",)),
        static,
        pd.DataFrame(rows),
        splits,
    )


def test_baseline_tiers_do_not_misrepresent_flat_models_or_tabpfn() -> None:
    assert set(PRIMARY_SEQUENTIAL) == {
        "tabularargn",
        "cpar",
        "realtabformer",
        "empirical_sequence_sampler",
    }
    assert BASELINE_SPECS["tabdit"].tier == "reported_only"
    assert "generation_code" in BASELINE_SPECS["tabdit"].execution_status
    assert set(FLATTENED_CONTROLS) == {"ctgan", "tvae", "gaussian_copula"}
    assert "tabpfn" not in BASELINE_SPECS
    assert comparison_manifest()["tabpfn_role"].endswith("not_a_generator")


def test_shared_plan_uses_train_entities_only_and_is_deterministic() -> None:
    dataset = _dataset()
    first = build_shared_generation_plan(dataset, n_entities=12, seed=4)
    second = build_shared_generation_plan(dataset, n_entities=12, seed=4)
    assert first.lengths == second.lengths
    assert first.source_train_entity_ids == second.source_train_entity_ids
    assert set(first.source_train_entity_ids) <= set(dataset.entity_ids_for_split("train"))
    assert first.fit_split == "train"


def test_views_preserve_parent_child_and_mark_flattened_controls() -> None:
    dataset = _dataset()
    relational = materialize_compatibility_view(dataset, "realtabformer")
    assert set(relational) == {"parent", "child"}
    flat = materialize_compatibility_view(dataset, "ctgan")
    assert "entity_id" not in flat["flat_events"]


def test_raw_output_validator_rejects_temporal_repair_need() -> None:
    dataset = _dataset()
    plan = build_shared_generation_plan(dataset, n_entities=3, seed=5)
    rows = []
    for entity, length in zip(plan.entity_ids, plan.lengths):
        time = 0.0
        for index in range(length):
            gap = np.nan if index == 0 else 1.0
            if index:
                time += gap
            rows.append({"entity_id": entity, "event_index": index, "timestamp": time, "gap": gap, "receiver_or_mark": "m", "amount_or_numeric_value": 1.0})
    generated = pd.DataFrame(rows)
    validate_raw_generated_events(generated, plan)
    broken = generated.copy()
    broken.loc[broken["event_index"] == 1, "timestamp"] += 1
    with pytest.raises(BaselineCompatibilityError, match="timestamps and gaps"):
        validate_raw_generated_events(broken, plan)


def test_dependency_probe_is_read_only_metadata() -> None:
    result = dependency_readiness()
    assert set(result) == set(BASELINE_SPECS)
    assert all(item["installation_attempted"] is False for item in result.values())
    assert result["cpar"]["version"] == "1.38.0"
    assert result["realtabformer"]["version"] == "0.2.4"
