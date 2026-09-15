import copy
from pathlib import Path

import pytest
import numpy as np
import torch
import yaml

from eval.cof_seqgen_v3_contract import (
    FROZEN_AMOUNT_CONTRACT,
    FROZEN_THRESHOLDS,
    V3ContractError,
    bind_train_only_conditioning_contract,
    build_train_joint_support_state,
    load_v3_definition,
    validate_v3_io_path,
    validate_v3_definition,
)
from generators.sampling_plan import SamplingPlan
from models.cof_seqgen_v3 import JointStateCodec


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = (
    REPOSITORY
    / "configs/benchmark_v2/cof_seqgen_v3_source_preparation.yaml"
)


def test_joint_support_state_is_train_only_hash_bound_and_padding_free():
    codec = JointStateCodec(gap_bins=2, receiver_classes=3)
    gap = torch.tensor([[0, 1, 1], [1, 0, 0]])
    receiver = torch.tensor([[0, 2, 1], [2, 1, 2]])
    valid_mask = torch.tensor(
        [[True, True, False], [True, True, False]]
    )
    provenance = {
        "train_file_sha256": "a" * 64,
        "train_content_sha256": "b" * 64,
        "sampling_plan_sha256": "c" * 64,
    }

    state = build_train_joint_support_state(
        codec=codec,
        gap=gap,
        receiver=receiver,
        valid_mask=valid_mask,
        fit_split="train",
        provenance=provenance,
    )

    assert state["fit_split"] == "train"
    assert state["validation_rows_used"] == 0
    assert state["test_rows_used"] == 0
    assert state["observed_joint_states"] == [0, 1, 5]
    assert state["support_mask"] == [True, True, False, False, False, True]
    assert len(state["state_sha256"]) == 64

    for forbidden_split in ("validation", "test", "fresh_test"):
        with pytest.raises(V3ContractError, match="train-only"):
            build_train_joint_support_state(
                codec=codec,
                gap=gap,
                receiver=receiver,
                valid_mask=valid_mask,
                fit_split=forbidden_split,
                provenance=provenance,
            )


def test_conditioning_binding_freezes_y_length_mask_plan_and_amount_contract():
    plan = SamplingPlan(
        y_entity=np.array([0, 1], dtype=np.int64),
        lengths=np.array([2, 1], dtype=np.int64),
        valid_mask=np.array(
            [[True, True, False], [True, False, False]]
        ),
        plan_hash="d" * 64,
    )

    binding = bind_train_only_conditioning_contract(
        plan=plan,
        expected_plan_sha256="d" * 64,
        amount_contract=FROZEN_AMOUNT_CONTRACT,
        fit_split="train",
    )

    assert binding["sampling_plan_sha256"] == "d" * 64
    assert binding["entity_count"] == 2
    assert binding["lengths"] == [2, 1]
    assert binding["amount_contract"] == FROZEN_AMOUNT_CONTRACT
    assert len(binding["binding_sha256"]) == 64

    with pytest.raises(V3ContractError, match="SamplingPlan"):
        bind_train_only_conditioning_contract(
            plan=plan,
            expected_plan_sha256="e" * 64,
            amount_contract=FROZEN_AMOUNT_CONTRACT,
            fit_split="train",
        )
    with pytest.raises(V3ContractError, match="amount"):
        bind_train_only_conditioning_contract(
            plan=plan,
            expected_plan_sha256="d" * 64,
            amount_contract={"algorithm": "changed"},
            fit_split="train",
        )
    with pytest.raises(V3ContractError, match="train-only"):
        bind_train_only_conditioning_contract(
            plan=plan,
            expected_plan_sha256="d" * 64,
            amount_contract=FROZEN_AMOUNT_CONTRACT,
            fit_split="validation",
        )


def test_v3_config_has_only_frozen_reference_and_two_joint_candidates():
    definition = load_v3_definition(CONFIG)

    assert [candidate.candidate_id for candidate in definition.candidates] == [
        "cof_v3_ref_v28_frozen",
        "cof_v3_c01_direct_joint",
        "cof_v3_c02_factorized_joint",
    ]
    assert [candidate.architecture for candidate in definition.candidates] == [
        "frozen_reference",
        "direct_joint",
        "factorized_joint",
    ]
    assert definition.candidates[0].training_required is False
    assert all(
        candidate.training_required
        for candidate in definition.candidates[1:]
    )
    assert definition.raw["frozen_contract"]["thresholds"] == (
        FROZEN_THRESHOLDS
    )
    assert definition.raw["frozen_contract"]["amount_contract"] == (
        FROZEN_AMOUNT_CONTRACT
    )
    assert definition.raw["model_contract"]["coherence_lambda"] == 0.0
    assert definition.raw["model_contract"]["post_hoc_calibration"] == (
        "FORBIDDEN"
    )
    assert definition.raw["model_contract"]["legacy_independent_heads"] == (
        "FORBIDDEN"
    )

    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for mutation in (
        lambda value: value["frozen_contract"]["thresholds"].update(
            receiver_max_abs_signed_frequency=0.03
        ),
        lambda value: value["model_contract"].update(
            coherence_lambda=0.1
        ),
        lambda value: value["candidates"].append(
            {"candidate_id": "unregistered"}
        ),
    ):
        tampered = copy.deepcopy(raw)
        mutation(tampered)
        with pytest.raises(V3ContractError):
            validate_v3_definition(tampered)


def test_v3_fit_io_is_train_only_and_source_phase_is_write_free(tmp_path):
    repository = tmp_path / "repository"
    train = (
        repository
        / "data/benchmark_v2_5/frozen/scenario/kappa/train.npz"
    )
    validation = train.with_name("validation.npz")
    test = train.with_name("test.npz")
    future = (
        repository
        / "artifacts/benchmark_v3/candidate_selection/attempt_001"
    )
    for path in (train, validation, test):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    assert validate_v3_io_path(
        repository_root=repository,
        path=train,
        access="read",
        purpose="fit",
        source_only=True,
    ) == train.resolve()
    with pytest.raises(V3ContractError, match="train-only"):
        validate_v3_io_path(
            repository_root=repository,
            path=validation,
            access="read",
            purpose="fit",
            source_only=True,
        )
    with pytest.raises(V3ContractError, match="test"):
        validate_v3_io_path(
            repository_root=repository,
            path=test,
            access="read",
            purpose="provenance",
            source_only=True,
        )
    with pytest.raises(V3ContractError, match="source-only"):
        validate_v3_io_path(
            repository_root=repository,
            path=future,
            access="write",
            purpose="artifact",
            source_only=True,
        )
