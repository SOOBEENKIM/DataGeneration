import pytest

from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_benchmark
from generators.conditional_ctgan import ConditionalCTGAN
from generators.conditional_tvae import ConditionalTVAE
from generators.contracts_v2_5 import validate_synthetic_contract
from generators.sampling_plan import SamplingPlan


def problem():
    bundle = generate_benchmark(
        BenchmarkConfig(
            scenario="joint_semimarkov_v2b",
            kappa=1.0,
            n_train=48,
            n_test=8,
            min_length=4,
            max_length=5,
            fraud_rate=0.5,
            n_gap_bins=6,
            n_receiver_categories=8,
        ),
        42,
    )
    plan = SamplingPlan.from_train_policy(
        bundle.train,
        entity_count=8,
        seed=10_001,
    )
    return bundle, plan


@pytest.mark.parametrize(
    ("adapter_type", "steps"),
    [(ConditionalCTGAN, 2), (ConditionalTVAE, 2)],
)
def test_separate_class_tabular_models_share_one_budget_and_checkpoint(
    adapter_type,
    steps,
    tmp_path,
):
    bundle, plan = problem()
    checkpoints = []

    def checkpoint(event, adapter):
        path = (
            tmp_path
            / f"{adapter_type.__name__}_{event['class_label']}_{event['step']}.pt"
        )
        adapter.save_training_checkpoint(path)
        checkpoints.append(path)

    adapter = adapter_type()
    adapter.fit(
        bundle.train,
        config={
            "cuda": False,
            "batch_size": 20,
            "requested_steps_total": steps,
            "requested_steps_per_class": {"0": 1, "1": 1},
            "max_wall_seconds_total": 20.0,
            "max_wall_seconds_per_class": {"0": 10.0, "1": 10.0},
            "checkpoint_interval_steps": 1,
            "checkpoint_callback": checkpoint,
        },
        seed=1,
    )
    assert adapter.budget_allocation["max_wall_seconds_total"] == 20.0
    assert adapter.budget_allocation["max_wall_seconds_per_class"] == {
        0: 10.0,
        1: 10.0,
    }
    assert adapter.budget_allocation["requested_steps_per_class"] == {
        0: 1,
        1: 1,
    }
    assert set(adapter.models) == {0, 1}
    assert len(checkpoints) == 2
    assert all(path.is_file() for path in checkpoints)
    sample = adapter.sample(plan, seed=2)
    validate_synthetic_contract(sample, plan=plan, train=bundle.train)


def test_class_pair_cannot_double_spend_baseline_budget():
    bundle, _ = problem()
    with pytest.raises(ValueError, match="sum to one baseline budget"):
        ConditionalCTGAN().fit(
            bundle.train,
            config={
                "cuda": False,
                "batch_size": 20,
                "requested_steps_total": 2,
                "requested_steps_per_class": {"0": 1, "1": 1},
                "max_wall_seconds_total": 20.0,
                "max_wall_seconds_per_class": {"0": 20.0, "1": 20.0},
                "checkpoint_interval_steps": 1,
            },
            seed=1,
        )
