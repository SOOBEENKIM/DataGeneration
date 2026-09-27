"""Train-only finite-state label reference inside a frozen label-first ARGN.

No per-row relabeling: label probabilities are replaced before stochastic native
sampling, and the sampled label conditions all subsequent transaction fields.
These known statistical controls are not a proposed novel architecture.
"""
import ast
from contextlib import contextmanager
import numpy as np
import torch
from benchmarks.argn_label_first_control import label_first_model_class
from benchmarks.argn_past_state import PastState


def reference_table(parameters, mode):
    assert mode in ('markov', 'duration')
    pooled = parameters['previous_label']
    table = np.array([[pooled[str(p)]] * 22 for p in (-1, 0, 1)], dtype=np.float64)
    if mode == 'duration':
        for key, value in parameters['age_hazard'].items():
            previous, age = ast.literal_eval(key)
            if previous >= 0:
                table[previous + 1, age] = value
    return table


def probabilities(features, table, max_length):
    present = features[..., 0] > .5
    previous = features[..., 1] > .5
    # log_run_count is a strict-past summary of the previous observed/generated run.
    age = torch.expm1(features[..., 3].double() * np.log1p(max_length)).round().long().clamp(0, 21)
    row = torch.where(present, previous.long() + 1, torch.zeros_like(age))
    return table[row, age]


def reference_model_class(stats, parameters, mode):
    parent = label_first_model_class(stats)
    codec = PastState(stats)
    label_key = codec.prefixes['event_is_fraud'] + '__cat'
    values = reference_table(parameters, mode)

    class ReferenceModel(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Not a parameter/buffer: parent checkpoint loads strictly unchanged.
            self.reference_values = torch.tensor(values, device=self.device)
            self.predictors.register_forward_hook(self.replace_label)

        def replace_label(self, module, args, output):
            if args[1] != label_key:
                return output
            p = probabilities(self._state_features, self.reference_values, stats['seq_len']['max'])
            logits = torch.full_like(output, -1e9)
            logits[..., codec.codes['1']] = p.log().to(output.dtype)
            logits[..., codec.codes['0']] = torch.log1p(-p).to(output.dtype)
            return logits

    return ReferenceModel


@contextmanager
def reference_generation(stats, parameters, mode):
    import mostlyai.engine._tabular.generation as generation
    original = generation.SequentialModel
    generation.SequentialModel = reference_model_class(stats, parameters, mode)
    try:
        yield
    finally:
        generation.SequentialModel = original
