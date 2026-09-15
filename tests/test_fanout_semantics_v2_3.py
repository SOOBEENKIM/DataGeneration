import numpy as np

from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_benchmark
from eval.channel_controls_v2 import (
    channel_only_summaries,
    fixed_step_unique_receiver,
)
from scripts.audit_fanout_semantics_v2_3 import alignment_destroyed


def test_fixed_step_unique_receiver_is_gap_independent():
    categories = np.array([1, 1, 2, 3, 3])
    assert fixed_step_unique_receiver(categories, width=3) == np.mean(
        [1, 1, 2, 3, 2]
    )


def test_alignment_intervention_preserves_receiver_paths_within_strata():
    bundle = generate_benchmark(
        BenchmarkConfig(
            scenario="joint_semimarkov_v2b",
            kappa=1.0,
            n_train=100,
            n_test=2000,
            min_length=16,
            max_length=18,
        ),
        42,
    )
    changed = alignment_destroyed(bundle.test, seed=4)
    for label in (0, 1):
        for length in np.unique(bundle.test.lengths):
            selected = (
                (bundle.test.y_entity == label)
                & (bundle.test.lengths == length)
            )
            original = sorted(
                tuple(row[:length, 0]) for row in bundle.test.x_cat[selected]
            )
            permuted = sorted(
                tuple(row[:length, 0]) for row in changed.x_cat[selected]
            )
            assert original == permuted


def test_gap_velocity_is_receiver_independent():
    bundle = generate_benchmark(
        BenchmarkConfig(
            scenario="joint_semimarkov_v2b",
            kappa=1.0,
            n_train=100,
            n_test=200,
            min_length=16,
            max_length=18,
        ),
        42,
    )
    changed = alignment_destroyed(bundle.test, seed=4)
    tau = np.asarray(bundle.metadata["tau"])
    original = channel_only_summaries(bundle.test, tau=tau)
    permuted = channel_only_summaries(changed, tau=tau)
    np.testing.assert_array_equal(
        original["gap_time_window_velocity"],
        permuted["gap_time_window_velocity"],
    )
