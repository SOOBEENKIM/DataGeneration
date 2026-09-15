import numpy as np

from benchmarks.validation import stratified_entity_bootstrap_auc


def test_cluster_bootstrap_auc_contains_null_and_detects_signal():
    labels = np.tile([0, 1], 100)
    null_scores = np.tile([0.4, 0.4], 100)
    point, low, high = stratified_entity_bootstrap_auc(
        null_scores, labels, resamples=200, seed=4
    )
    assert point == 0.5
    assert low < 0.55
    assert high > 0.45

    signal_scores = labels.astype(float)
    point, low, high = stratified_entity_bootstrap_auc(
        signal_scores, labels, resamples=200, seed=5
    )
    assert point == 1.0
    assert low > 0.99
