import numpy as np

from eval.coherence_v2 import coherence_bin_diagnostics
from eval.joint_association_v2 import (
    association_delta,
    association_statistics,
    bootstrap_delta_interval,
)


def test_association_statistics_match_hand_computed_delta_and_sign():
    values = np.array([0.0, 2.0, 3.0, 5.0])
    labels = np.array([0, 0, 1, 1])
    assert association_delta(values, labels) == 3.0
    stats = association_statistics(values, labels, real_delta=2.0)
    assert stats["delta_joint"] == 3.0
    assert stats["association_recovery_error"] == 1.0
    assert stats["recovery_ratio"] == 1.5
    assert stats["sign_consistent"]
    low, high = bootstrap_delta_interval(values, labels, resamples=200, seed=7)
    assert low <= 3.0 <= high


def test_empty_synthetic_bin_is_penalized_not_rewarded():
    real = np.array([-2.0, -1.0, 1.0, 2.0])
    labels = np.array([0, 0, 1, 1])
    synth = np.zeros(4)
    report = coherence_bin_diagnostics(
        real, labels, synth, labels, np.array([-3.0, 0.0, 3.0]), minimum=1
    )
    assert report["invalid_bin_count"] == 1
    assert report["rows"][0]["excluded_from_dropped_macro"]
    assert report["occupancy_penalized_gap"] > report["dropped_bin_macro_gap"]
    assert report["worst_score_macro_gap"] > report["dropped_bin_macro_gap"]
