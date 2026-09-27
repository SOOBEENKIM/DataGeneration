"""Registered order-only control; no new state or output parameters."""
from contextlib import contextmanager

from mostlyai.engine._common import SLEN_SIDX_SDEC_COLUMN
from benchmarks.argn_state_adapter import engine_adapter, model_class
from benchmarks.argn_event_weight_control import event_weighted_adapter
from benchmarks.argn_past_state import PastState


def label_first_model_class(stats, parent=None):
    parent = parent if parent is not None else model_class(stats, enabled=False)
    codec = PastState(stats)
    label = codec.prefixes["event_is_fraud"]
    category = codec.prefixes["category"]

    class LabelFirstSequentialModel(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            first = [SLEN_SIDX_SDEC_COLUMN, label, category]
            self.column_order = first + [c for c in self.tgt_columns if c not in first]

    return LabelFirstSequentialModel


@contextmanager
def label_first_adapter(stats, weighting, mean_training_length, source_dir=None):
    if weighting not in {"customer", "event"}:
        raise ValueError(weighting)
    import mostlyai.engine._tabular.generation as generation
    adapter = (event_weighted_adapter(stats, mean_training_length, source_dir)
               if weighting == "event" else engine_adapter(stats, False, source_dir))
    with adapter as train:
        # Native loss uses isinstance against this exact registered parent.
        model = label_first_model_class(stats, parent=train.__globals__["SequentialModel"])
        saved = train.__globals__["SequentialModel"], generation.SequentialModel
        train.__globals__["SequentialModel"] = generation.SequentialModel = model
        try:
            yield train
        finally:
            train.__globals__["SequentialModel"], generation.SequentialModel = saved
