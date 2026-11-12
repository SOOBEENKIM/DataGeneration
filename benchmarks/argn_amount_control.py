"""Known-method amount-head controls on a frozen label-first ARGN.

Only amount regressors/predictors are optimized. Two-expert mixture and label
routing have identical parameters and initialization, separating routing from
additional capacity. The parent checkpoint is never modified.
"""
from contextlib import contextmanager
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.argn_past_state import PastState
from benchmarks.argn_transition_reference import reference_model_class


class AmountHeads(nn.Module):
    def __init__(self, model, keys, kind="shared"):
        super().__init__()
        assert kind in {"shared", "mixture", "routed"}
        self.kind, self.keys = kind, list(keys)
        self.regressors = nn.ModuleList()
        self.predictors = nn.ModuleList()
        self.dropout = nn.Dropout(model.regressors.dropout.p)
        for _ in range(1 if kind == "shared" else 2):
            self.regressors.append(nn.ModuleDict({k: deepcopy(model.regressors.get(k)) for k in keys}))
            self.predictors.append(nn.ModuleDict({k: deepcopy(model.predictors.predictors[k]) for k in keys}))
        self.requires_grad_(True)

    def expert(self, x, key, index):
        for layer in self.regressors[index][key]:
            x = layer(self.dropout(x))
        return self.predictors[index][key](F.relu(x))

    def forward(self, x, key, label):
        first = self.expert(x, key, 0)
        if self.kind == "shared":
            return first
        second = self.expert(x, key, 1)
        if self.kind == "routed":
            return torch.where(label.bool().unsqueeze(-1), second, first)
        return torch.logsumexp(torch.stack([first.log_softmax(-1), second.log_softmax(-1)]), 0) - math.log(2)


def conditional_objective(losses, labels, natural_fraud_rate, balanced):
    """Unbiased natural or macro conditional risk from a stratified 50:50 batch."""
    assert labels.eq(0).any() and labels.eq(1).any()
    normal, fraud = losses[labels.eq(0)].mean(), losses[labels.eq(1)].mean()
    p = .5 if balanced else natural_fraud_rate
    return (1 - p) * normal + p * fraud


@contextmanager
def amount_generation(stats, parameters, payload_path):
    """Use the registered duration reference and trained amount head in generation."""
    import mostlyai.engine._tabular.generation as generation
    parent = reference_model_class(stats, parameters, "duration")
    codec = PastState(stats)
    label_key = codec.prefixes["event_is_fraud"] + "__cat"

    class AmountModel(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            payload = torch.load(payload_path, map_location=self.device, weights_only=True)
            heads = AmountHeads(self, payload["keys"], payload["kind"]).to(self.device)
            heads.load_state_dict(payload["state_dict"])
            heads.eval().requires_grad_(False)
            # Unregistered: the frozen native checkpoint still loads strictly.
            object.__setattr__(self, "amount_heads", heads)
            self._amount_inputs = {}
            self.embedders.get(label_key).register_forward_pre_hook(self.capture_label)
            self.regressors.register_forward_pre_hook(self.capture_inputs)
            self.predictors.register_forward_hook(self.replace_amount)

        def capture_label(self, module, args):
            labels = args[0]
            if labels.ndim == 3:  # teacher tensors include a scalar channel and padding
                labels = labels.squeeze(-1)
            else:
                assert torch.all(labels.eq(codec.codes["0"]) | labels.eq(codec.codes["1"]))
            self._amount_labels = labels.eq(codec.codes["1"]).long()

        def capture_inputs(self, module, args):
            if args[1] in self.amount_heads.keys:
                self._amount_inputs[args[1]] = torch.cat(args[0], -1)

        def replace_amount(self, module, args, output):
            key = args[1]
            if key in self.amount_heads.keys:
                return self.amount_heads(self._amount_inputs.pop(key), key, self._amount_labels)
            return output

    original = generation.SequentialModel
    generation.SequentialModel = AmountModel
    try:
        yield
    finally:
        generation.SequentialModel = original
