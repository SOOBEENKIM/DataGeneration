"""Read-only one-step queries from cached native teacher histories.

The dummy recurrent states affect only the discarded *next* history. Never use
the returned history to roll out multiple steps.
"""
import torch
from mostlyai.engine._tabular.argn import _sampling_fixed_probs


def label_probability(logits, key, fraud_code, fixed_probs):
    probabilities = logits.softmax(-1)
    if key in fixed_probs:
        shape = probabilities.shape
        probabilities = _sampling_fixed_probs(
            probabilities.reshape(-1, shape[-1]), fixed_probs[key]
        ).reshape(shape)
    return probabilities[..., fraud_code]


@torch.no_grad()
def teacher_with_history(model, batch):
    captured = {}
    def save(module, args, output):
        captured['history'] = output[0].detach()
    handle = model.history_compressor.register_forward_hook(save)
    try:
        logits, _ = model(batch, mode='trn')
    finally:
        handle.remove()
    return logits, captured['history'], model.context_compressor(batch)


@torch.no_grad()
def one_step(model, history, memory, context, fixed_values, label_key, fixed_probs):
    n = history.shape[0]
    lstm = model.history_compressor.get()
    recurrent = tuple(torch.zeros(lstm.num_layers, n, lstm.hidden_size,
                                 dtype=history.dtype, device=history.device) for _ in range(2))
    packed = memory.to(dtype=torch.float64, device=history.device).unsqueeze(0)
    captured = {}
    def save(module, args, output):
        if args[1] == label_key:
            captured['logits'] = output.detach()
    handle = model.predictors.register_forward_hook(save)
    try:
        outputs, _, _ = model(None, mode='gen', batch_size=n, history=history,
                              history_state=(*recurrent, packed), context=context,
                              fixed_values=fixed_values, fixed_probs=fixed_probs)
    finally:
        handle.remove()
    assert label_key not in fixed_values, 'label must be predicted'
    return captured['logits'][:, 0], outputs
