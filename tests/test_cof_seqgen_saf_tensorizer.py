from __future__ import annotations

import numpy as np
import pandas as pd

from data.cof_seqgen_saf_contract import (
    CanonicalEntitySequenceDataset,
    CanonicalSchema,
    RESERVED_CODES,
)
from data.cof_seqgen_saf_tensorizer import SAFTensorizer


def _dataset() -> CanonicalEntitySequenceDataset:
    ids = [f"e{i}" for i in range(10)]
    static = pd.DataFrame(
        {
            "entity_id": ids,
            "entity_label": [0, 1] * 5,
        }
    )
    splits = pd.DataFrame(
        {
            "entity_id": ids,
            "split": ["train"] * 7 + ["validation"] + ["test"] * 2,
        }
    )
    rows = []
    for entity in ids:
        for index in range(3):
            rows.append(
                {
                    "entity_id": entity,
                    "event_id": f"{entity}-{index}",
                    "event_index": index,
                    "timestamp": float(index),
                    "gap": np.nan if index == 0 else 1.0,
                    "receiver_or_mark": "seen" if entity != "e7" else "validation-only",
                    "amount_or_numeric_value": float(index * 10),
                }
            )
    return CanonicalEntitySequenceDataset(
        CanonicalSchema(
            "controlled_coupling_joint_semimarkov_v2b_kappa_1_00",
            "absolute",
            "step",
            static_context_columns=("entity_label",),
        ),
        static,
        pd.DataFrame(rows),
        splits,
    )


def test_tensorizer_is_train_only_and_unknown_is_distinct() -> None:
    dataset = _dataset()
    tensorizer = SAFTensorizer.fit(dataset)
    validation = tensorizer.transform_split(dataset, "validation")
    assert set(validation[0].receiver) == {RESERVED_CODES.unk}
    assert tensorizer.state.fit_split == "train"
    assert tensorizer.state.model_config_kwargs()["static_categorical_vocab_sizes"]


def test_integer_static_context_survives_pandas_numpy_scalar_conversion() -> None:
    dataset = _dataset()
    tensorizer = SAFTensorizer.fit(dataset)
    train = tensorizer.transform_split(dataset, "train")
    encoded = {sequence.static_categorical[0] for sequence in train}
    assert RESERVED_CODES.unk not in encoded
    assert len(encoded) == 2


def test_numeric_transform_round_trip_and_first_gap() -> None:
    dataset = _dataset()
    tensorizer = SAFTensorizer.fit(dataset)
    train = tensorizer.transform_split(dataset, "train")
    assert np.isnan(train[0].gap[0])
    codec = tensorizer.state.event_numeric_codecs[0][1]
    original = np.asarray([-10.0, 0.0, 100.0])
    recovered = codec.decode(codec.encode(original))
    assert np.allclose(original, recovered, atol=1e-5)


def test_tensorizer_state_is_serializable_and_support_observed() -> None:
    tensorizer = SAFTensorizer.fit(_dataset())
    state = tensorizer.state.to_dict()
    assert state["schema_version"] == "cof-seqgen-saf-tensorizer-state-v2"
    assert set(state["gap_support"]["representatives"]) == {1.0}
