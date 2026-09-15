from pathlib import Path

import numpy as np
import pytest

from benchmarks.types import SequenceBatch
from eval.validation_selection_v2_6 import read_yaml_mapping
from generators.candidate_adapters_v2_6 import (
    CandidateAdapterContractError,
    TrainOnlyZScore,
    build_candidate_adapter,
    candidate_adapter_spec,
)
from generators.candidate_model_backends_v2_6 import BACKEND_REGISTRY_V2_6
from generators.sampling_plan import SamplingPlan


REPOSITORY = Path(__file__).resolve().parents[1]


def train_fixture() -> SequenceBatch:
    lengths = np.asarray([2, 1, 2, 1], dtype=np.int64)
    valid = np.arange(3)[None, :] < lengths[:, None]
    amount = np.zeros((4, 3, 1), dtype=np.float32)
    amount[..., 0][valid] = np.asarray(
        [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
        dtype=np.float32,
    )
    gap = np.zeros((4, 3), dtype=np.int64)
    gap[valid] = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int64)
    receiver = np.zeros((4, 3, 1), dtype=np.int64)
    receiver[..., 0][valid] = np.asarray(
        [0, 1, 2, 1, 0, 2],
        dtype=np.int64,
    )
    return SequenceBatch(
        x_num=amount,
        dt_bin=gap,
        x_cat=receiver,
        valid_mask=valid,
        y_entity=np.asarray([0, 0, 1, 1], dtype=np.int64),
        lengths=lengths,
        entity_ids=np.asarray(["a", "b", "c", "d"]),
    )


def selection_config():
    return read_yaml_mapping(
        REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
    )


def test_train_only_zscore_fits_valid_train_rows_and_round_trips_exactly():
    train = train_fixture()

    transform = TrainOnlyZScore.fit(train)
    transformed = transform.transform_train(train)
    restored = transform.inverse_synthetic(transformed)

    assert transform.mean.tolist() == [6.0]
    assert np.allclose(
        transform.std,
        np.asarray([np.std([1, 3, 5, 7, 9, 11])]),
    )
    assert np.all(transformed.x_num[~train.valid_mask] == 0)
    assert np.allclose(
        restored.x_num[train.valid_mask],
        train.x_num[train.valid_mask],
        rtol=0,
        atol=1e-6,
    )
    assert transform.checkpoint_state()["fit_split"] == "train"
    assert transform.checkpoint_state()["valid_row_count"] == 6


def test_ctgan_tvae_class_pair_budget_is_exactly_half_per_class():
    config = selection_config()

    for model_id in ("ctgan_separate_class", "tvae_separate_class"):
        for candidate in config["models"][model_id]["candidates"]:
            spec = candidate_adapter_spec(
                config,
                model_id=model_id,
                candidate_id=candidate["candidate_id"],
            )
            assert spec.class_updates == {0: 10_000, 1: 10_000}
            assert spec.class_wall_seconds == {0: 3_600.0, 1: 3_600.0}
            assert sum(spec.class_updates.values()) == spec.requested_updates
            assert sum(spec.class_wall_seconds.values()) == (
                spec.max_wall_seconds
            )


def test_neural_and_cof_candidate_parameters_map_exactly_without_backend_calls():
    config = selection_config()
    backend_calls = 0

    def forbidden_backend():
        nonlocal backend_calls
        backend_calls += 1
        raise AssertionError("fixture must not instantiate a model backend")

    neural_spec = candidate_adapter_spec(
        config,
        model_id="neural_sequence",
        candidate_id="c03_zscore_weighted_tempered_checkpoint_20000",
    )
    neural = build_candidate_adapter(
        neural_spec,
        backend_factory=forbidden_backend,
    )
    assert neural.loss_weights == {
        "amount": 2.0,
        "gap": 2.0,
        "receiver": 1.0,
    }
    assert neural.sampling_rule["categorical_temperature"] == 0.75

    cof_spec = candidate_adapter_spec(
        config,
        model_id="cof_seqgen",
        candidate_id="c03_zscore_weighted_tempered_checkpoint_20000",
    )
    cof = build_candidate_adapter(
        cof_spec,
        backend_factory=forbidden_backend,
    )
    assert cof.loss_weights == {
        "amount": 2.0,
        "gap": 2.0,
        "receiver": 1.0,
        "label": 1.0,
    }
    assert cof.sampling_rule["discrete_feedback_temperature"] == 0.75
    assert cof.sampling_rule["final_categorical_decode"] == (
        "temperature_multinomial"
    )
    assert cof.sampling_rule["final_categorical_temperature"] == 0.75
    assert backend_calls == 0


def test_adapter_binds_only_the_fixed_train_fitted_sampling_plan():
    train = train_fixture()
    plan = SamplingPlan.from_train_policy(
        train,
        entity_count=5,
        seed=26001,
    )
    different_plan = SamplingPlan.from_train_policy(
        train,
        entity_count=5,
        seed=26002,
    )
    spec = candidate_adapter_spec(
        selection_config(),
        model_id="neural_sequence",
        candidate_id="c00_native_checkpoint_10000",
    )
    adapter = build_candidate_adapter(
        spec,
        backend_factory=lambda: None,
    )

    adapter.bind_train_only_sampling_plan(plan)
    adapter.assert_sampling_plan(plan)

    try:
        adapter.assert_sampling_plan(different_plan)
    except ValueError as error:
        assert "SamplingPlan hash mismatch" in str(error)
    else:
        raise AssertionError("a non-frozen SamplingPlan must be rejected")


def test_all_four_real_v26_backends_import_and_instantiate_without_model_calls():
    expected = {
        "ctgan_separate_class",
        "tvae_separate_class",
        "neural_sequence",
        "cof_seqgen",
    }

    assert set(BACKEND_REGISTRY_V2_6) == expected
    backends = {
        model_id: backend_type()
        for model_id, backend_type in BACKEND_REGISTRY_V2_6.items()
    }

    assert all(
        backend.baseline_definition_version == "benchmark-v2.6-candidate-v1"
        for backend in backends.values()
    )
    assert all(callable(backend.fit) for backend in backends.values())
    assert all(callable(backend.sample) for backend in backends.values())
    assert all(not hasattr(backend, "model") for backend in backends.values())
    assert all(not hasattr(backend, "models") for backend in backends.values())


def test_fake_backend_receives_train_only_transform_and_checkpoint_state():
    train = train_fixture()
    captured = {}

    class FakeBackend:
        def fit(self, fitted_train, *, config, seed):
            captured["train"] = fitted_train
            captured["config"] = config
            captured["seed"] = seed

        def sample(self, *args, **kwargs):
            raise AssertionError("fixture must not sample")

    spec = candidate_adapter_spec(
        selection_config(),
        model_id="neural_sequence",
        candidate_id="c03_zscore_weighted_tempered_checkpoint_20000",
    )
    adapter = build_candidate_adapter(
        spec,
        backend_factory=FakeBackend,
    )

    adapter.fit_train_only(train, base_config={}, seed=2601)
    state = adapter.checkpoint_metadata()

    assert captured["seed"] == 2601
    assert np.isclose(
        captured["train"].x_num[train.valid_mask].mean(),
        0,
        atol=1e-6,
    )
    assert captured["config"]["v2_6_channel_loss_weights"] == {
        "amount": 2.0,
        "gap": 2.0,
        "receiver": 1.0,
    }
    assert captured["config"]["v2_6_sampling_rule"][
        "categorical_temperature"
    ] == 0.75
    assert state["train_only_zscore"]["mean"] == [6.0]
    assert state["train_only_zscore"]["std"] == pytest.approx(
        [np.std([1, 3, 5, 7, 9, 11])]
    )
    assert state["train_only_zscore"]["fit_split"] == "train"


def test_adapter_rejects_validation_and_test_material_before_backend_creation():
    train = train_fixture()
    calls = 0

    def forbidden_backend():
        nonlocal calls
        calls += 1
        raise AssertionError("backend must not be constructed")

    spec = candidate_adapter_spec(
        selection_config(),
        model_id="cof_seqgen",
        candidate_id="c00_native_checkpoint_10000",
    )
    adapter = build_candidate_adapter(
        spec,
        backend_factory=forbidden_backend,
    )

    for forbidden_key in (
        "validation_batch",
        "validation_labels",
        "validation_lengths",
        "test_path",
        "fresh_test_path",
    ):
        with pytest.raises(
            CandidateAdapterContractError,
            match="validation/test",
        ):
            adapter.fit_train_only(
                train,
                base_config={forbidden_key: object()},
                seed=2601,
            )

    assert calls == 0
