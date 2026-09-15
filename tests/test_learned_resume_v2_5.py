import numpy as np
import pytest

from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_benchmark
from generators.cof_seqgen_adapter import CoFSeqGenAdapter
from generators.contracts_v2_5 import validate_synthetic_contract
from generators.joint_sequence_baseline import NeuralSequenceBaseline
from generators.sampling_plan import SamplingPlan


class IntentionalInterruption(RuntimeError):
    pass


def problem():
    bundle = generate_benchmark(
        BenchmarkConfig(
            scenario="joint_semimarkov_v2b",
            kappa=1.0,
            n_train=24,
            n_test=8,
            min_length=4,
            max_length=5,
            fraud_rate=0.5,
            n_gap_bins=6,
            n_receiver_categories=8,
        ),
        42,
    )
    return bundle, SamplingPlan.from_train_policy(
        bundle.train,
        entity_count=8,
        seed=10_001,
    )


def test_neural_sequence_exact_checkpoint_resume(tmp_path):
    bundle, plan = problem()
    checkpoint = tmp_path / "neural_step_1.pt"

    def interrupt(event, adapter):
        if event["step"] == 1:
            adapter.save_training_checkpoint(checkpoint)
            raise IntentionalInterruption

    first = NeuralSequenceBaseline()
    with pytest.raises(IntentionalInterruption):
        first.fit(
            bundle.train,
            config={
                "device": "cpu",
                "requested_steps": 2,
                "batch_size": 8,
                "hidden_size": 16,
                "checkpoint_interval_steps": 1,
                "checkpoint_callback": interrupt,
            },
            seed=1,
        )
    resumed = NeuralSequenceBaseline()
    resumed.fit(
        bundle.train,
        config={
            "device": "cpu",
            "requested_steps": 2,
            "batch_size": 8,
            "hidden_size": 16,
            "checkpoint_interval_steps": 1,
            "resume_checkpoint": str(checkpoint),
        },
        seed=1,
    )
    assert resumed.current_step == 2
    assert resumed.actual_training_budget["actual_steps"] == 2
    sample = resumed.sample(plan, seed=2)
    validate_synthetic_contract(sample, plan=plan, train=bundle.train)


def test_cof_exact_checkpoint_resume_and_mask_contract(tmp_path):
    bundle, plan = problem()
    checkpoint = tmp_path / "cof_step_1.pt"

    def interrupt(event, adapter):
        if event["step"] == 1:
            adapter.save_training_checkpoint(checkpoint)
            raise IntentionalInterruption

    base_config = {
        "device": "cpu",
        "requested_steps": 2,
        "batch_size": 8,
        "checkpoint_interval_steps": 1,
        "d_model": 16,
        "n_layers": 1,
        "lr": 1e-3,
        "tau": bundle.metadata["tau"],
        "window_width": 7.0,
        "temperature": 1.0,
        "coherence_lambda": 0.0,
        "diffusion_steps": 2,
    }
    first = CoFSeqGenAdapter()
    with pytest.raises(IntentionalInterruption):
        first.fit(
            bundle.train,
            config={**base_config, "checkpoint_callback": interrupt},
            seed=1,
        )
    resumed = CoFSeqGenAdapter()
    resumed.fit(
        bundle.train,
        config={**base_config, "resume_checkpoint": str(checkpoint)},
        seed=1,
    )
    assert resumed.current_step == 2
    sample = resumed.sample(plan, seed=2)
    report = validate_synthetic_contract(
        sample,
        plan=plan,
        train=bundle.train,
    )
    assert report["mask_contract"] == "PASS"
    assert np.all(sample.x_num[~sample.valid_mask] == 0)
