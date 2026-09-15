from __future__ import annotations

import random
import os

import numpy as np
import pandas as pd
import pytest
import torch

from data.cof_seqgen_saf_contract import CanonicalEntitySequenceDataset, CanonicalSchema
from generators.cof_seqgen_saf_baselines import (
    BaselineCompatibilityError,
    build_shared_generation_plan,
    validate_raw_generated_events,
)
from generators.cof_seqgen_saf_external_baselines import (
    EmpiricalSequenceSampler,
    REaLTabFormerWrapper,
    SDVCPARWrapper,
    SDVFlatControlWrapper,
    TabDiTUnavailable,
)
from experiments.cof_seqgen_saf_external_validation import _seed_everything, _sha256_tree


def _dataset() -> CanonicalEntitySequenceDataset:
    ids = [f"e{i}" for i in range(12)]
    static = pd.DataFrame(
        {"entity_id": ids, "segment": ["a" if i % 2 else "b" for i in range(12)]}
    )
    splits = pd.DataFrame(
        {
            "entity_id": ids,
            "split": ["train"] * 8 + ["validation"] * 2 + ["test"] * 2,
        }
    )
    rows = []
    for entity_position, entity in enumerate(ids):
        timestamp = 0.0
        for event_index in range(3 + entity_position % 3):
            gap = np.nan if event_index == 0 else float(1 + event_index % 2)
            if event_index:
                timestamp += gap
            rows.append(
                {
                    "entity_id": entity,
                    "event_id": f"{entity}-{event_index}",
                    "event_index": event_index,
                    "timestamp": timestamp,
                    "gap": gap,
                    "receiver_or_mark": f"m{(event_index + entity_position) % 3}",
                    "amount_or_numeric_value": float(event_index + entity_position),
                }
            )
    return CanonicalEntitySequenceDataset(
        CanonicalSchema(
            "toy_external_baseline",
            "relative",
            "steps",
            static_context_columns=("segment",),
        ),
        static,
        pd.DataFrame(rows),
        splits,
    )


def test_empirical_sampler_is_exact_and_train_only() -> None:
    dataset = _dataset()
    plan = build_shared_generation_plan(dataset, n_entities=7, seed=2)
    wrapper = EmpiricalSequenceSampler().fit(dataset)
    generated = wrapper.sample(plan)
    validate_raw_generated_events(generated, plan)
    assert wrapper.fit_record.provenance.fit_split == "train"
    assert set(plan.source_train_entity_ids) <= set(dataset.entity_ids_for_split("train"))


def test_tabdit_fails_closed_when_upstream_generator_is_absent() -> None:
    with pytest.raises(BaselineCompatibilityError, match="no generator architecture"):
        TabDiTUnavailable().fit(_dataset())


def test_gaussian_copula_flat_control_executes_canonical_projection() -> None:
    dataset = _dataset()
    plan = build_shared_generation_plan(dataset, n_entities=4, seed=7)
    wrapper = SDVFlatControlWrapper("gaussian_copula").fit(dataset)
    generated = wrapper.sample(plan)
    validate_raw_generated_events(generated, plan)
    assert wrapper.fit_record.length_mode == "protocol_partition_not_learned"


def test_cpar_pinned_private_context_path_executes_on_cpu() -> None:
    dataset = _dataset()
    plan = build_shared_generation_plan(dataset, n_entities=3, seed=11)
    wrapper = SDVCPARWrapper(epochs=1, cuda=False).fit(dataset)
    generated = wrapper.sample(plan)
    validate_raw_generated_events(generated, plan)
    assert wrapper.fit_record.package_version == "1.38.0"


def test_external_runner_resets_all_rngs_and_enables_hard_determinism() -> None:
    _seed_everything(31)
    first = (random.random(), np.random.random(), torch.rand(3))
    _seed_everything(31)
    second = (random.random(), np.random.random(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_external_model_tree_hash_covers_paths_and_contents(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("one", encoding="utf-8")
    first = _sha256_tree(model)
    (model / "weights.bin").write_bytes(b"two")
    second = _sha256_tree(model)
    assert first != second
    (model / "weights.bin").write_bytes(b"changed")
    assert second != _sha256_tree(model)


@pytest.mark.skipif(
    os.environ.get("COFSEQ_RUN_EXTERNAL_SMOKE") != "1",
    reason="explicit external dependency smoke only",
)
def test_realtabformer_parent_child_cpu_smoke(tmp_path) -> None:
    dataset = _dataset()
    plan = build_shared_generation_plan(dataset, n_entities=2, seed=17)
    wrapper = REaLTabFormerWrapper(
        workspace_dir=tmp_path / "rtf",
        epochs=1,
        batch_size=4,
        seed=17,
        output_max_length=64,
    ).fit(dataset, device="cpu", fit_kwargs={"n_critic": 0})
    generated = wrapper.sample(plan, device="cpu", gen_batch=2)
    validate_raw_generated_events(generated, plan)
    assert set(wrapper.model_artifacts) == {"parent", "child"}
    assert all(path.is_dir() for path in wrapper.model_artifacts.values())
