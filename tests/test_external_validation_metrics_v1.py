import time
import tracemalloc

import numpy as np
import pytest

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.external_validation_metrics_v1 import (
    ExternalMetricError,
    compute_external_validation_metrics,
    validate_external_hard_contract,
)


def _batch(batch_type, *, receiver_shift=0):
    lengths = np.array([3, 3], dtype=np.int64)
    mask = np.ones((2, 3), dtype=np.bool_)
    values = {
        "x_num": np.array(
            [[[0.0], [1.0], [2.0]], [[2.0], [3.0], [4.0]]],
            dtype=np.float32,
        ),
        "dt_bin": np.array([[1, 1, 2], [2, 1, 2]], dtype=np.int64),
        "x_cat": (
            np.array([[[2], [2], [3]], [[4], [4], [4]]], dtype=np.int64)
            + receiver_shift
        ),
        "valid_mask": mask,
        "y_entity": np.array([0, 1], dtype=np.int64),
        "lengths": lengths,
    }
    if batch_type is SequenceBatch:
        values["entity_ids"] = np.array(["a", "b"])
    return batch_type(**values)


def test_external_metrics_are_class_conditional_and_include_coherence():
    real = _batch(SequenceBatch)
    same = _batch(SyntheticBatch)

    metrics = compute_external_validation_metrics(
        real=real,
        synthetic=same,
        gap_tau=np.array([0.0, 1.0, 3.0]),
        short_gap_threshold=1.0,
    )

    assert set(metrics["fidelity"]) == {
        "amount_ks_y0",
        "amount_ks_y1",
        "gap_total_variation_y0",
        "gap_total_variation_y1",
        "receiver_total_variation_y0",
        "receiver_total_variation_y1",
    }
    assert set(metrics["coherence"]) == {
        "short_gap_receiver_repeat_error_y0",
        "short_gap_receiver_repeat_error_y1",
    }
    assert all(value == 0.0 for value in metrics["fidelity"].values())
    assert all(value == 0.0 for value in metrics["coherence"].values())


def test_receiver_total_variation_keeps_the_small_distribution_contract():
    real = _batch(SequenceBatch)
    synthetic = _batch(SyntheticBatch)
    synthetic.x_cat[0, :, 0] = np.array([2, 3, 3])

    metrics = compute_external_validation_metrics(
        real=real,
        synthetic=synthetic,
        gap_tau=np.array([0.0, 1.0, 3.0]),
        short_gap_threshold=1.0,
    )

    assert metrics["fidelity"]["receiver_total_variation_y0"] == pytest.approx(
        1.0 / 3.0
    )
    assert metrics["fidelity"]["receiver_total_variation_y1"] == 0.0


def test_amlsim_scale_receiver_metric_avoids_n_by_k_work():
    n_sequences, sequence_length, receiver_cardinality = 6_250, 32, 9_656
    n_rows = n_sequences * sequence_length
    valid = np.ones((n_sequences, sequence_length), dtype=np.bool_)
    labels = np.arange(n_sequences, dtype=np.int64) % 2
    lengths = np.full(n_sequences, sequence_length, dtype=np.int64)
    amount = (np.arange(n_rows) % 101).reshape(n_sequences, sequence_length, 1).astype(
        np.float32
    )
    gap = (np.arange(n_rows) % 8).reshape(n_sequences, sequence_length)
    receiver = (np.arange(n_rows) % receiver_cardinality).reshape(
        n_sequences, sequence_length, 1
    )
    real = SequenceBatch(
        x_num=amount,
        dt_bin=gap,
        x_cat=receiver,
        valid_mask=valid,
        y_entity=labels,
        lengths=lengths,
        entity_ids=np.arange(n_sequences),
    )
    synthetic = SyntheticBatch(
        x_num=amount.copy(),
        dt_bin=gap.copy(),
        x_cat=np.roll(receiver, 1, axis=1),
        valid_mask=valid.copy(),
        y_entity=labels.copy(),
        lengths=lengths.copy(),
    )

    tracemalloc.start()
    started = time.perf_counter()
    metrics = compute_external_validation_metrics(
        real=real,
        synthetic=synthetic,
        gap_tau=np.arange(8, dtype=float),
        short_gap_threshold=3.0,
    )
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert metrics["fidelity"]["receiver_total_variation_y0"] == pytest.approx(0.0)
    assert elapsed < 0.75
    assert peak_bytes < 64 * 1024 * 1024


def test_external_hard_contract_rejects_out_of_vocabulary_and_bad_plan():
    real = _batch(SequenceBatch)
    synthetic = _batch(SyntheticBatch)
    passed = validate_external_hard_contract(
        real_validation=real,
        synthetic=synthetic,
        gap_cardinality=16,
        receiver_cardinality=10,
    )
    assert passed == {
        "mask": True,
        "padding": True,
        "train_discrete_support": True,
    }

    invalid = _batch(SyntheticBatch, receiver_shift=20)
    with pytest.raises(ExternalMetricError, match="receiver support"):
        validate_external_hard_contract(
            real_validation=real,
            synthetic=invalid,
            gap_cardinality=16,
            receiver_cardinality=10,
        )
