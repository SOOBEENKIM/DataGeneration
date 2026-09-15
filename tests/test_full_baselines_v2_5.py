import numpy as np

from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_benchmark
from generators.contracts_v2_5 import validate_synthetic_contract
from generators.full_registry_v2_5 import (
    baseline_registry,
    generator_for_v2_5,
)
from generators.plug_in_state_baselines_v2_5 import (
    ClassConditionalPlugInHMM,
    ClassConditionalPlugInHSMM,
)
from generators.sampling_plan import SamplingPlan


EXPECTED = [
    "empirical_iid",
    "block_2",
    "block_4",
    "block_8",
    "full_sequence_reference",
    "independent_markov",
    "joint_markov",
    "plug_in_hmm",
    "plug_in_hsmm",
    "ctgan_separate_class",
    "tvae_separate_class",
    "neural_sequence",
    "cof_seqgen",
]


def small_problem():
    bundle = generate_benchmark(
        BenchmarkConfig(
            scenario="joint_semimarkov_v2b",
            kappa=1.0,
            n_train=128,
            n_test=24,
            min_length=5,
            max_length=8,
            fraud_rate=0.5,
            n_gap_bins=8,
            n_receiver_categories=12,
        ),
        42,
    )
    plan = SamplingPlan.from_train_policy(
        bundle.train,
        entity_count=24,
        seed=10_001,
    )
    return bundle, plan


def test_registry_contains_every_preregistered_generator_cell_in_order():
    registry = baseline_registry()
    assert list(registry) == EXPECTED
    assert len(registry) == 13
    assert registry["full_sequence_reference"].full_sequence_competitor is False
    assert registry["ctgan_separate_class"].training_device == "gpu"
    assert registry["cof_seqgen"].role == "proposed"


def test_all_cpu_baselines_satisfy_shared_contract_and_are_deterministic():
    bundle, plan = small_problem()
    configs = {
        "independent_markov": {"laplace_alpha": 1.0},
        "joint_markov": {"laplace_alpha": 1.0},
        "plug_in_hmm": {
            "laplace_alpha": 1.0,
            "short_gap_quantile": 0.5,
            "max_wall_seconds": 7200,
        },
        "plug_in_hsmm": {
            "laplace_alpha": 1.0,
            "short_gap_quantile": 0.5,
            "duration_laplace_alpha": 1.0,
            "duration_max": 16,
            "max_wall_seconds": 7200,
        },
    }
    for generator_id in EXPECTED[:9]:
        adapter = generator_for_v2_5(generator_id)
        adapter.fit(
            bundle.train,
            config=configs.get(generator_id, {}),
            seed=1,
        )
        first = adapter.sample(plan, seed=2)
        second = adapter.sample(plan, seed=2)
        report = validate_synthetic_contract(
            first,
            plan=plan,
            train=bundle.train,
        )
        assert report["status"] == "PASS"
        assert np.array_equal(first.x_num, second.x_num)
        assert np.array_equal(first.dt_bin, second.dt_bin)
        assert np.array_equal(first.x_cat, second.x_cat)


def test_joint_markov_uses_gap_repeat_state_not_independent_alias():
    bundle, _ = small_problem()
    adapter = generator_for_v2_5("joint_markov")
    adapter.fit(bundle.train, config={"laplace_alpha": 1.0}, seed=1)
    assert adapter.joint_classes == 2 * adapter.gap_classes
    assert "joint" in adapter.models[0]
    assert "gap" not in adapter.models[0]


def test_hmm_and_hsmm_fit_threshold_and_duration_on_train_only():
    bundle, _ = small_problem()
    hmm = ClassConditionalPlugInHMM()
    hmm.fit(
        bundle.train,
        config={
            "short_gap_quantile": 0.5,
            "laplace_alpha": 1.0,
            "max_wall_seconds": 7200,
        },
        seed=1,
    )
    assert hmm.fit_metadata["state_fit_split"] == "train"
    assert hmm.fit_metadata["hidden_states"] == 2
    assert hmm.fit_metadata["state_inference"] == "deterministic_train_only"
    assert hmm.fit_metadata["uses_em"] is False
    assert hmm.fit_metadata["uses_latent_state_restarts"] is False
    assert hmm.fit_metadata["uses_test_data"] is False
    hsmm = ClassConditionalPlugInHSMM()
    hsmm.fit(
        bundle.train,
        config={
            "short_gap_quantile": 0.5,
            "laplace_alpha": 1.0,
            "duration_laplace_alpha": 1.0,
            "duration_max": 16,
            "max_wall_seconds": 7200,
        },
        seed=1,
    )
    assert hsmm.fit_metadata["duration_fit_split"] == "train"
    assert hsmm.fit_metadata["duration_family"] == (
        "smoothed_empirical_discrete"
    )
    assert len(hsmm.models[0]["duration"][0]) == 16
    assert hmm.actual_training_budget["max_wall_seconds"] == 7200
    assert hsmm.actual_training_budget["max_wall_seconds"] == 7200


def test_sampling_plan_is_derived_only_from_train_and_seed():
    first, _ = small_problem()
    second = generate_benchmark(
        BenchmarkConfig(
            scenario="joint_semimarkov_v2b",
            kappa=1.0,
            n_train=128,
            n_test=50,
            min_length=5,
            max_length=8,
            fraud_rate=0.5,
            n_gap_bins=8,
            n_receiver_categories=12,
        ),
        42,
    ), None
    plan_a = SamplingPlan.from_train_policy(
        first.train,
        entity_count=30,
        seed=123,
    )
    plan_b = SamplingPlan.from_train_policy(
        second[0].train,
        entity_count=30,
        seed=123,
    )
    assert np.array_equal(plan_a.y_entity, plan_b.y_entity)
    assert np.array_equal(plan_a.lengths, plan_b.lengths)
    assert plan_a.plan_id == plan_b.plan_id
    assert len(plan_a.plan_hash) == 64


def test_shared_sampling_plan_does_not_change_with_model_seed():
    bundle, _ = small_problem()
    plans = [
        SamplingPlan.from_train_policy(
            bundle.train,
            entity_count=24,
            seed=10_001,
        )
        for _model_seed in range(1, 6)
    ]
    assert len({plan.plan_hash for plan in plans}) == 1
    assert all(
        np.array_equal(plans[0].y_entity, plan.y_entity)
        and np.array_equal(plans[0].lengths, plan.lengths)
        for plan in plans[1:]
    )
