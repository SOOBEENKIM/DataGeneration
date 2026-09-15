from pathlib import Path
import copy

import numpy as np
import pytest
import yaml

from benchmarks.types import SyntheticBatch
from eval.single_factor_amendment_v2_6 import (
    AppendOnlyAmendmentStore,
    SingleFactorContractError,
    compute_model_diagnostics,
    load_and_validate_amendment,
    validate_amendment_definition,
    validate_amendment_io_path,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = (
    REPOSITORY
    / "configs/benchmark_v2/selection_v2_6_single_factor_amendment.yaml"
)


def _sample() -> SyntheticBatch:
    lengths = np.asarray([2, 2, 2, 2], dtype=np.int64)
    valid = np.ones((4, 2), dtype=np.bool_)
    return SyntheticBatch(
        x_num=np.asarray(
            [[[0.0], [0.2]], [[0.1], [0.3]], [[0.7], [0.9]], [[0.8], [1.0]]],
            dtype=np.float32,
        ),
        dt_bin=np.asarray(
            [[0, 1], [0, 1], [1, 1], [0, 1]],
            dtype=np.int64,
        ),
        x_cat=np.asarray(
            [[[0], [1]], [[0], [1]], [[1], [2]], [[1], [2]]],
            dtype=np.int64,
        ),
        valid_mask=valid,
        y_entity=np.asarray([0, 0, 1, 1], dtype=np.int64),
        lengths=lengths,
    )


def test_each_candidate_has_exactly_one_preregistered_intervention():
    plan = load_and_validate_amendment(CONFIG)

    assert plan.model_ids == (
        "ctgan_separate_class",
        "tvae_separate_class",
        "cof_seqgen",
    )
    assert len(plan.candidates) == 9
    assert plan.reused_control_count == 3
    assert plan.evaluation_only_count == 3
    assert plan.training_trajectory_count == 3
    for candidate in plan.candidates:
        if candidate.execution_kind == "reuse_frozen_control":
            assert candidate.intervention_dimension == "none"
            assert candidate.changed_dimensions == ()
        else:
            assert candidate.intervention_dimension != "none"
            assert candidate.changed_dimensions == (
                candidate.intervention_dimension,
            )


def test_model_diagnostics_are_channel_specific_and_preregistered():
    plan = load_and_validate_amendment(CONFIG)

    assert {
        "amount_signed_standardized_class_effect",
        "gap_signed_standardized_class_effect",
        "receiver_signed_frequency_by_category",
    } <= set(plan.diagnostics_by_model["ctgan_separate_class"])
    assert {
        "gap_raw_frequency_by_bin",
        "receiver_raw_frequency_by_category",
        "gap_frequency_hhi",
        "receiver_frequency_hhi",
    } <= set(plan.diagnostics_by_model["tvae_separate_class"])
    assert plan.diagnostics_by_model["cof_seqgen"] == (
        "generated_amount_mean",
        "generated_amount_std",
        "generated_amount_q01",
        "generated_amount_q05",
        "generated_amount_q25",
        "generated_amount_q50",
        "generated_amount_q75",
        "generated_amount_q95",
        "generated_amount_q99",
    )


def test_io_is_train_validation_only_test_fail_closed_and_append_only(
    tmp_path,
):
    repository = tmp_path / "repository"
    repository.mkdir()
    train = repository / "data/frozen/train.npz"
    validation = repository / "data/frozen/validation.npz"
    test = repository / "data/frozen/test.npz"
    unrelated_data = repository / "data/frozen/audit_latent.npz"
    for path in (train, validation, test, unrelated_data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    assert validate_amendment_io_path(
        repository_root=repository,
        path=train,
        role="train split",
        access="read",
    ) == train.resolve()
    assert validate_amendment_io_path(
        repository_root=repository,
        path=validation,
        role="validation split",
        access="read",
    ) == validation.resolve()
    with pytest.raises(SingleFactorContractError, match="test"):
        validate_amendment_io_path(
            repository_root=repository,
            path=test,
            role="split",
            access="read",
        )
    with pytest.raises(SingleFactorContractError, match="train/validation"):
        validate_amendment_io_path(
            repository_root=repository,
            path=unrelated_data,
            role="development split",
            access="read",
        )
    old_candidate = (
        repository
        / "artifacts/benchmark_v2_6/selection/candidates/ctgan/file.json"
    )
    with pytest.raises(SingleFactorContractError, match="write"):
        validate_amendment_io_path(
            repository_root=repository,
            path=old_candidate,
            role="candidate output",
            access="write",
        )

    store = AppendOnlyAmendmentStore(repository)
    first = store.create_candidate_attempt(
        model_id="ctgan_separate_class",
        candidate_id="ctgan_sf_c01_shared_transformer",
        seed=2601,
    )
    store.write_json(first / "manifest.json", {"attempt": 1})
    second = store.create_candidate_attempt(
        model_id="ctgan_separate_class",
        candidate_id="ctgan_sf_c01_shared_transformer",
        seed=2601,
    )

    assert first.name == "attempt_001"
    assert second.name == "attempt_002"
    assert (first / "manifest.json").read_text() == '{"attempt":1}\n'
    with pytest.raises(SingleFactorContractError, match="exists"):
        store.write_json(first / "manifest.json", {"attempt": 999})


def test_stored_sample_diagnostics_match_each_model_contract():
    plan = load_and_validate_amendment(CONFIG)
    sample = _sample()

    for model_id in plan.model_ids:
        diagnostics = compute_model_diagnostics(
            model_id=model_id,
            sample=sample,
            tau=np.asarray([0.5, 2.0]),
            receiver_categories=3,
        )
        assert tuple(diagnostics) == plan.diagnostics_by_model[model_id]

    tvae = compute_model_diagnostics(
        model_id="tvae_separate_class",
        sample=sample,
        tau=np.asarray([0.5, 2.0]),
        receiver_categories=3,
    )
    assert sum(tvae["gap_raw_frequency_by_bin"]) == pytest.approx(1.0)
    assert sum(tvae["receiver_raw_frequency_by_category"]) == pytest.approx(
        1.0
    )
    cof = compute_model_diagnostics(
        model_id="cof_seqgen",
        sample=sample,
        tau=np.asarray([0.5, 2.0]),
        receiver_categories=3,
    )
    assert cof["generated_amount_mean"] == pytest.approx(0.5)
    assert cof["generated_amount_q50"] == pytest.approx(0.5)


def test_threshold_update_and_residual_contract_tampering_fail_closed():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    threshold_tamper = copy.deepcopy(config)
    threshold_tamper["frozen_contract"]["thresholds"]["amount_ks"] = 1.0
    with pytest.raises(SingleFactorContractError, match="threshold"):
        validate_amendment_definition(threshold_tamper)

    update_tamper = copy.deepcopy(config)
    update_tamper["models"]["tvae_separate_class"]["candidates"][1][
        "effective_dimensions"
    ]["requested_updates"] = 40_000
    with pytest.raises(SingleFactorContractError, match="single-factor"):
        validate_amendment_definition(update_tamper)

    residual_tamper = copy.deepcopy(config)
    residual_tamper["models"]["cof_seqgen"]["candidates"][2][
        "residual_contract"
    ]["validation_refit"] = "ALLOWED"
    with pytest.raises(SingleFactorContractError, match="residual"):
        validate_amendment_definition(residual_tamper)
