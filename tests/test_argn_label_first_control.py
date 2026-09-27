"""Check the factorization, unchanged model capacity, and generation parity."""
import copy
import numpy as np
import torch
import pytest

from tests.test_argn_state_first import fixture
from benchmarks.argn_state_adapter import FullHistoryCollator, model_class
from benchmarks.argn_label_first_control import label_first_model_class, label_first_adapter
from benchmarks.argn_frozen_probe import teacher_with_history, one_step
from benchmarks.argn_past_state import STATE_COLUMN
from mostlyai.engine._common import SLEN_SIDX_SDEC_COLUMN
from mostlyai.engine._tabular.generation import _fix_rare_token_probs, _translate_fixed_probs


def test_only_order_changes_initial_model_and_training_adapter(fixture):
    stats, codec, records, kwargs = fixture
    torch.manual_seed(414)
    original = model_class(stats, False)(**kwargs).eval()
    torch.manual_seed(414)
    candidate = label_first_model_class(stats)(**kwargs).eval()
    assert original.state_dict().keys() == candidate.state_dict().keys()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
    assert candidate.column_order[:3] == [SLEN_SIDX_SDEC_COLUMN,
        codec.prefixes['event_is_fraud'], codec.prefixes['category']]
    assert sorted(original.column_order) == sorted(candidate.column_order)
    assert candidate.state_enabled is False
    for weighting in ['customer', 'event']:
        with label_first_adapter(stats, weighting, 5.5) as train:
            import mostlyai.engine._tabular.generation as generation
            cls = train.__globals__['SequentialModel']
            assert cls is generation.SequentialModel
            rebuilt = cls(**kwargs).eval()
            rebuilt.load_state_dict(candidate.state_dict(), strict=True)
            assert rebuilt.column_order == candidate.column_order
            loss = train.__globals__['_calculate_sample_losses'](
                rebuilt, FullHistoryCollator(True, None, torch.device('cpu'))(records)).mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert rebuilt.state_projection.weight.grad.abs().sum() == 0


def test_current_fields_cannot_leak_into_first_label_and_stepwise_parity(fixture):
    stats, codec, records, kwargs = fixture
    model = label_first_model_class(stats)(**kwargs).eval()
    batch = FullHistoryCollator(True, None, torch.device('cpu'))(records)
    key = codec.prefixes['event_is_fraud'] + '__cat'
    masks = _translate_fixed_probs(_fix_rare_token_probs(stats), stats)
    with torch.no_grad():
        teacher, histories, context = teacher_with_history(model, batch)
        altered = copy.deepcopy(batch)
        for field in altered:
            if field.startswith('tgt:') and not field.startswith(SLEN_SIDX_SDEC_COLUMN):
                altered[field][:, 3:] = 0
        changed, _ = model(altered, mode='trn')
        # Includes the current label: only previous events can affect label_t.
        torch.testing.assert_close(changed[key][:, :4], teacher[key][:, :4], rtol=0, atol=0)
        memory = codec.empty(2)
        history = recurrent = None
        for t in range(5):
            fixed = {k:v[:, t] for k,v in batch.items() if k.startswith('tgt:')}
            structural = {k:v for k,v in fixed.items() if k.startswith(SLEN_SIDX_SDEC_COLUMN)}
            # Generate all current fields, then provide them as a separate probe.
            for values in [structural, {k:v for k,v in fixed.items() if k != key}]:
                logits, _ = one_step(model, histories[:, t:t+1], torch.from_numpy(memory),
                                    context, values, key, masks)
                torch.testing.assert_close(logits, teacher[key][:, t], rtol=1e-5, atol=2e-6)
            _, history, recurrent = model(None, mode='gen', batch_size=2, context=context,
                history=history, history_state=recurrent, fixed_values=fixed)
            memory = codec.advance(memory, {k:v.numpy() for k,v in fixed.items()})
            np.testing.assert_allclose(memory, recurrent[2][0].numpy())
        # Ended customers can be removed without changing the survivor's state.
        mask = torch.tensor([False, True])
        filtered = tuple(t[:, mask, ...] for t in recurrent)
        _, _, next_state = model(None, mode='gen', batch_size=1,
            context=[[v[mask] for v in g] for g in context], history=history[mask],
            history_state=filtered, fixed_values={k:v[mask, 5] for k,v in batch.items() if k.startswith('tgt:')})
        np.testing.assert_allclose(codec.features(next_state[2][0].numpy())[0], records[1][STATE_COLUMN][6])
