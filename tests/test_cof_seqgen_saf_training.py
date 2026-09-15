from __future__ import annotations

import numpy as np
import torch

from data.cof_seqgen_saf_tensorizer import TensorizedSequence
from experiments.cof_seqgen_saf_training import (
    SequenceWindowDataset,
    collate_windows,
)
from models.cof_seqgen_saf import CoFSeqGenSAF, SAFModelConfig, fit_train_only_gap_support


def _sequence(length: int = 7) -> TensorizedSequence:
    return TensorizedSequence(
        entity_id="e",
        gap=np.asarray([np.nan] + [float(i % 3) for i in range(1, length)], dtype=np.float32),
        receiver=np.asarray([3 + i % 3 for i in range(length)], dtype=np.int64),
        numeric_value=np.arange(length, dtype=np.float32),
        auxiliary_categorical=(
            np.asarray([3 + i % 2 for i in range(length)], dtype=np.int64),
        ),
        auxiliary_numeric=np.column_stack(
            (np.arange(length), -np.arange(length))
        ).astype(np.float32),
        static_categorical=(3,),
        static_numeric=np.asarray([0.5], dtype=np.float32),
    )


def test_windowing_has_one_warmup_and_no_duplicate_targets() -> None:
    dataset = SequenceWindowDataset((_sequence(),), context_window=4)
    assert len(dataset) == 2
    assert dataset[0]["target_mask"].tolist() == [True, True, True, True]
    assert dataset[1]["target_mask"].tolist() == [False, True, True, True]
    assert sum(item["target_mask"].sum() for item in dataset) == 7


def test_auxiliary_batch_runs_loss_and_sampling() -> None:
    batch = collate_windows([SequenceWindowDataset((_sequence(),), context_window=8)[0]])
    config = SAFModelConfig(
        candidate_id="SAF-O1",
        receiver_vocab_size=8,
        static_dim=1,
        static_categorical_vocab_sizes=(6,),
        auxiliary_categorical_vocab_sizes=(6,),
        auxiliary_numeric_dim=2,
        hidden_dim=16,
    )
    model = CoFSeqGenSAF(config, fit_train_only_gap_support([0, 1, 2]))
    losses = model.compute_loss(**batch)
    losses["loss"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    sampled = model.sample_fixed_lengths(
        [3],
        static=batch["static"],
        static_categorical=batch["static_categorical"],
    )
    assert len(sampled["auxiliary_categorical"]) == 1
    assert sampled["auxiliary_numeric"].shape == (1, 3, 2)
