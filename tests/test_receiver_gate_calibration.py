import numpy as np

from scripts.calibrate_receiver_gate_v2_1 import gate


def test_signed_cluster_frequency_gate_accepts_equal_entity_distributions():
    categories = np.tile(np.arange(8), (40, 2))
    lengths = np.full(40, 16)
    labels = np.repeat([0, 1], 20)

    passed, maximum, bound = gate(
        categories, lengths, labels, n_categories=8, margin=0.02
    )

    assert passed
    assert maximum == 0
    assert bound == 0


def test_signed_cluster_frequency_gate_detects_category_leakage():
    categories = np.tile(np.arange(8), (40, 2))
    lengths = np.full(40, 16)
    labels = np.repeat([0, 1], 20)
    categories[labels == 1, :8] = 0

    passed, maximum, _ = gate(
        categories, lengths, labels, n_categories=8, margin=0.02
    )

    assert not passed
    assert maximum > 0.4
