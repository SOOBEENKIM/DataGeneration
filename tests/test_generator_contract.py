from dataclasses import replace
import time
import tracemalloc

import numpy as np

from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_benchmark
from benchmarks.types import SequenceBatch
from generators.class_conditional_markov import ClassConditionalMarkov
from generators.empirical_conditional_block import EmpiricalConditionalBlock
from generators.empirical_conditional_iid import EmpiricalConditionalIID
from generators.sampling_plan import SamplingPlan


def setup():
    bundle = generate_benchmark(BenchmarkConfig(n_train=256, n_test=64), 42)
    return bundle.train, SamplingPlan.from_batch(bundle.test)


def assert_contract(sample, plan):
    assert np.array_equal(sample.y_entity, plan.y_entity)
    assert np.array_equal(sample.lengths, plan.lengths)
    assert np.array_equal(sample.valid_mask, plan.valid_mask)
    assert np.all(sample.x_num[~sample.valid_mask] == 0)
    assert np.all(sample.dt_bin[~sample.valid_mask] == 0)
    assert np.all(sample.x_cat[~sample.valid_mask] == 0)


def test_empirical_iid_contract_and_determinism():
    train, plan = setup()
    model = EmpiricalConditionalIID()
    model.fit(train, seed=1)
    first, second = model.sample(plan, seed=2), model.sample(plan, seed=2)
    assert_contract(first, plan)
    assert np.array_equal(first.x_num, second.x_num)
    assert not np.array_equal(first.x_num, model.sample(plan, seed=3).x_num)


def test_empirical_iid_amlsim_scale_sampling_has_no_dense_receiver_matrix():
    receiver_cardinality = 9_656
    receiver = np.tile(np.arange(receiver_cardinality, dtype=np.int64), 2)
    labels = np.repeat(np.array([0, 1], dtype=np.int64), receiver_cardinality)
    train = SequenceBatch(
        x_num=receiver.reshape(-1, 1, 1).astype(np.float32),
        dt_bin=(receiver % 8).reshape(-1, 1),
        x_cat=receiver.reshape(-1, 1, 1),
        valid_mask=np.ones((len(receiver), 1), dtype=np.bool_),
        y_entity=labels,
        lengths=np.ones(len(receiver), dtype=np.int64),
        entity_ids=np.arange(len(receiver)),
    )
    n_sequences, sequence_length = 6_250, 32
    plan = SamplingPlan(
        y_entity=np.arange(n_sequences, dtype=np.int64) % 2,
        lengths=np.full(n_sequences, sequence_length, dtype=np.int64),
        valid_mask=np.ones((n_sequences, sequence_length), dtype=np.bool_),
        plan_hash="synthetic-amlsim-scale-plan",
    )
    model = EmpiricalConditionalIID()
    model.fit(train, seed=31001)

    tracemalloc.start()
    started = time.perf_counter()
    sample = model.sample(plan, seed=31002)
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert_contract(sample, plan)
    valid = sample.valid_mask
    assert np.array_equal(sample.x_num[..., 0][valid], sample.x_cat[..., 0][valid])
    assert np.array_equal(
        sample.dt_bin[valid], sample.x_cat[..., 0][valid] % 8
    )
    assert elapsed < 0.75
    assert peak_bytes < 64 * 1024 * 1024


def test_block_bootstrap_contract_and_no_boundary_crossing():
    train, plan = setup()
    for block in (1, 2, 4, 8, "full"):
        model = EmpiricalConditionalBlock(block)
        model.fit(train, seed=1)
        assert_contract(model.sample(plan, seed=2), plan)


def test_markov_contract_and_determinism():
    train, plan = setup()
    model = ClassConditionalMarkov()
    model.fit(train, config={"laplace_alpha": 1}, seed=1)
    first, second = model.sample(plan, seed=2), model.sample(plan, seed=2)
    assert_contract(first, plan)
    assert np.array_equal(first.dt_bin, second.dt_bin)
