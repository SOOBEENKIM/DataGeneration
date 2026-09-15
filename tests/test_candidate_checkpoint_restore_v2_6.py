from pathlib import Path

import pytest
import torch

from experiments.candidate_runner_v2_6 import (
    build_checkpoint_bundle,
    load_candidate_checkpoint_bundle,
    restore_candidate_checkpoint_for_sampling,
)
from generators.candidate_model_backends_v2_6 import (
    CheckpointableCTGANCandidateV26,
    CheckpointableTVAECandidateV26,
    ConditionalCTGANCandidateV26,
    ConditionalTVAECandidateV26,
)
from generators.sampling_plan import SamplingPlan
from tests.test_candidate_runner_v2_6 import train_fixture


@pytest.mark.parametrize(
    ("model_id", "backend_type", "synthesizer_type"),
    [
        (
            "ctgan_separate_class",
            ConditionalCTGANCandidateV26,
            CheckpointableCTGANCandidateV26,
        ),
        (
            "tvae_separate_class",
            ConditionalTVAECandidateV26,
            CheckpointableTVAECandidateV26,
        ),
    ],
)
def test_tabular_checkpoint_rng_stays_cpu_before_explicit_device_restore(
    tmp_path: Path,
    model_id,
    backend_type,
    synthesizer_type,
):
    backend = backend_type()
    backend.models = {}
    for label in (0, 1):
        synthesizer = synthesizer_type(enable_gpu=False)
        synthesizer.set_random_state(2601 + label)
        backend.models[label] = synthesizer
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        build_checkpoint_bundle(
            backend=backend,
            train_only_zscore=None,
            shared_trajectory_id=f"{model_id}_native_seed_2601",
            sampled_checkpoint_step=20_000,
            selection_plan_sha256="a" * 64,
        ),
        checkpoint,
    )

    restored = load_candidate_checkpoint_bundle(
        checkpoint,
        model_id=model_id,
        device="cuda:7",
    )

    for synthesizer in restored["backend"].models.values():
        rng_state = synthesizer.random_states[1].get_state()
        assert rng_state.dtype == torch.uint8
        assert rng_state.device.type == "cpu"
        assert synthesizer._device == torch.device("cuda:7")


@pytest.mark.parametrize("model_id", ["neural_sequence", "cof_seqgen"])
def test_sequence_checkpoint_keeps_existing_selected_device_restore(
    tmp_path: Path,
    monkeypatch,
    model_id,
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    calls = []
    expected = {"backend": object()}

    def traced_load(path, *, map_location, weights_only):
        calls.append(
            {
                "path": path,
                "map_location": map_location,
                "weights_only": weights_only,
            }
        )
        return expected

    monkeypatch.setattr(
        "experiments.candidate_runner_v2_6.torch.load",
        traced_load,
    )

    restored = load_candidate_checkpoint_bundle(
        checkpoint,
        model_id=model_id,
        device="cuda:3",
    )

    assert restored is expected
    assert calls == [
        {
            "path": checkpoint,
            "map_location": "cuda:3",
            "weights_only": False,
        }
    ]


@pytest.mark.parametrize(
    ("model_id", "backend_type", "model_config"),
    [
        (
            "ctgan_separate_class",
            ConditionalCTGANCandidateV26,
            {
                "v2_6_sampling_rule": {
                    "categorical_temperature": 0.2,
                },
            },
        ),
        (
            "tvae_separate_class",
            ConditionalTVAECandidateV26,
            {
                "lr": 0.001,
                "weight_decay": 0.00001,
                "v2_6_channel_loss_weights": {
                    "amount": 1.0,
                    "gap": 1.0,
                    "receiver": 1.0,
                },
                "v2_6_sampling_rule": {"latent_scale": 1.0},
            },
        ),
    ],
)
def test_restored_tabular_checkpoint_can_sample_without_retraining(
    tmp_path: Path,
    model_id,
    backend_type,
    model_config,
):
    train = train_fixture()
    backend = backend_type()
    backend.fit(
        train,
        config={
            "requested_steps_total": 2,
            "requested_steps_per_class": {0: 1, 1: 1},
            "max_wall_seconds_total": 60.0,
            "max_wall_seconds_per_class": {0: 30.0, 1: 30.0},
            "checkpoint_interval_steps": 1,
            "batch_size": 10,
            "cuda": False,
            "device": "cpu",
            **model_config,
        },
        seed=2601,
    )
    checkpoint = tmp_path / f"{model_id}.pt"
    torch.save(
        build_checkpoint_bundle(
            backend=backend,
            train_only_zscore=None,
            shared_trajectory_id=f"{model_id}_native_seed_2601",
            sampled_checkpoint_step=20_000,
            selection_plan_sha256="a" * 64,
        ),
        checkpoint,
    )
    plan = SamplingPlan.from_train_policy(
        train,
        entity_count=4,
        seed=26001,
    )

    restored = restore_candidate_checkpoint_for_sampling(
        checkpoint,
        model_id=model_id,
        device="cpu",
        train=train,
    )
    sample = restored["backend"].sample(plan, seed=2601)

    assert sample.valid_mask.tolist() == plan.valid_mask.tolist()
    assert sample.y_entity.tolist() == plan.y_entity.tolist()
    assert sample.lengths.tolist() == plan.lengths.tolist()
