import numpy as np

from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_benchmark
from generators.joint_sequence_baseline import NeuralSequenceBaseline
from generators.sampling_plan import SamplingPlan


def test_neural_sequence_baseline_cpu_contract(tmp_path):
    bundle = generate_benchmark(
        BenchmarkConfig(n_train=32, n_test=8, min_length=4, max_length=6),
        42,
    )
    model = NeuralSequenceBaseline()
    checkpoint = tmp_path / "model.pt"
    model.fit(
        bundle.train,
        config={
            "device": "cpu",
            "steps": 2,
            "hidden_size": 16,
            "checkpoint_path": str(checkpoint),
        },
        seed=1,
    )
    plan = SamplingPlan.from_batch(bundle.test)
    sample = model.sample(plan, seed=2)
    assert checkpoint.is_file()
    assert np.array_equal(sample.valid_mask, plan.valid_mask)
    assert np.array_equal(sample.y_entity, plan.y_entity)
    assert np.all(sample.x_num[~sample.valid_mask] == 0)
    assert np.all(sample.dt_bin[~sample.valid_mask] == 0)
    assert np.all(sample.x_cat[~sample.valid_mask] == 0)
