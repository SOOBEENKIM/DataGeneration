"""Minimal past-state augmentation of the pinned MOSTLY AI engine 1.0.4.

The native model, heads, loss, and generator remain in use. A bias-free linear
state projection is added to the history vector read by transaction regressors.
B has the identical parameters but zero state inputs (a native-equivalent control).
"""
from contextlib import contextmanager
import inspect
import numpy as np
import torch
from torch import nn

from mostlyai.engine._common import SLEN_SIDX_SDEC_COLUMN, get_argn_name
from mostlyai.engine._tabular.argn import SequentialModel as NativeSequentialModel
from mostlyai.engine._tabular.training import BatchCollator as NativeBatchCollator

from benchmarks.argn_past_state import FEATURES, STATE_COLUMN, PastState


def model_class(stats: dict, enabled: bool):
    codec = PastState(stats)
    category = codec.prefixes["category"]

    class StateSequentialModel(NativeSequentialModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Fixed order in BOTH teacher training and free generation.
            self.column_order = [SLEN_SIDX_SDEC_COLUMN, category] + [
                c for c in self.tgt_columns if c not in [SLEN_SIDX_SDEC_COLUMN, category]
            ]
            self.state_enabled = enabled
            self.state_codec = codec
            self.history_dim = self.history_compressor.dim_output
            # Preserve the native RNG stream for initialization and data ordering.
            with torch.random.fork_rng(devices=[]):
                self.state_projection = nn.Linear(len(FEATURES), self.history_dim, bias=False)
                nn.init.normal_(self.state_projection.weight, std=0.02)
            self.state_projection.to(self.device)
            self._state_features = None
            self.regressors.register_forward_pre_hook(self._condition_regressor)

        def _condition_regressor(self, module, args):
            inputs, sub_col = args
            if sub_col.startswith(SLEN_SIDX_SDEC_COLUMN + "__"):
                return args
            first = inputs[0]
            state = self._state_features
            assert state is not None and state.shape[:2] == first.shape[:2]
            if not self.state_enabled:
                state = torch.zeros_like(state)
            projected = self.state_projection(state)
            joined = torch.cat([
                first[..., :-self.history_dim], first[..., -self.history_dim:] + projected,
            ], dim=-1)
            return ([joined] + inputs[1:], sub_col)

        def forward(self, x, mode, **kwargs):
            if mode == "trn":
                self._state_features = x[STATE_COLUMN]
                try:
                    return super().forward(x, mode, **kwargs)
                finally:
                    self._state_features = None
            history_state = kwargs.get("history_state")
            if history_state is None:
                memory = codec.empty(kwargs["batch_size"])
            else:
                assert len(history_state) == 3, "state must travel with LSTM history"
                memory = history_state[2][0].detach().cpu().numpy()
                kwargs["history_state"] = history_state[:2]
            self._state_features = torch.as_tensor(codec.features(memory), device=self.device).unsqueeze(1)
            try:
                outputs, history, recurrent = super().forward(x, mode, **kwargs)
            finally:
                self._state_features = None
            # Generator filters EVERY tensor in history_state along batch axis 1.
            # Packing summaries here makes early-ended customers safe to remove.
            event = {k: v.detach().cpu().numpy() for k, v in outputs.items()}
            memory = codec.advance(memory, event)
            packed = torch.as_tensor(memory, dtype=torch.float64, device=self.device).unsqueeze(0)
            return outputs, history, (*recurrent, packed)

    return StateSequentialModel


class FullHistoryCollator(NativeBatchCollator):
    def __init__(self, is_sequential, max_sequence_window, device):
        assert is_sequential
        # All past events reach BOTH models; no boundary reset or extra S history.
        super().__init__(is_sequential=True, max_sequence_window=None, device=device)

    def __call__(self, records):
        result = super().__call__(records)
        lengths = [len(record[STATE_COLUMN]) for record in records]
        features = np.zeros((len(records), max(lengths), len(FEATURES)), dtype=np.float32)
        for i, record in enumerate(records):
            features[i, :lengths[i]] = np.asarray(record[STATE_COLUMN], dtype=np.float32)
        result[STATE_COLUMN] = torch.as_tensor(features, device=self.device)
        return result


@contextmanager
def engine_adapter(stats, enabled, source_dir=None):
    import mostlyai.engine._tabular.training as training
    import mostlyai.engine._tabular.generation as generation
    model = model_class(stats, enabled)
    saved = (training.SequentialModel, training.BatchCollator, training.train, generation.SequentialModel)
    training.SequentialModel = generation.SequentialModel = model
    training.BatchCollator = FullHistoryCollator
    source = inspect.getsource(training)
    replacements = {
        "max_epochs_cap = math.ceil((trn_cnt + val_cnt) / 50)": "max_epochs_cap = max_epochs  # registered study cap",
        "torch.backends.cudnn.benchmark = True": "torch.backends.cudnn.benchmark = False  # reproducible paired study",
    }
    for old, new in replacements.items():
        assert source.count(old) == 1
        source = source.replace(old, new)
    if source_dir:
        (source_dir / "training_adapter_source.py").write_text(source)
    namespace = dict(training.__dict__)
    # Source imports are intentionally overwritten only for the two adapters.
    exec(compile(source, "registered_argn_training", "exec"), namespace)
    namespace["SequentialModel"] = model
    namespace["BatchCollator"] = FullHistoryCollator
    training.train = namespace["train"]
    try:
        yield training.train
    finally:
        training.SequentialModel, training.BatchCollator, training.train, generation.SequentialModel = saved
