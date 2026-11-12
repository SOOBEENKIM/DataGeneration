import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from mostlyai.engine._common import get_cardinalities, get_ctx_sequence_length
from mostlyai.engine._tabular.argn import SequentialModel, ModelSize, get_model_units
from mostlyai.engine._tabular.training import _calculate_sample_losses
from mostlyai.engine._encoding_types.tabular.numeric import _decode_numeric_digit
from benchmarks.argn_past_state import PastState, FEATURES, STATE_COLUMN, AMOUNT_CAP, DURATION_CAP
from benchmarks.argn_state_adapter import model_class, FullHistoryCollator

SOURCE = Path(__file__).resolve().parents[2] / "research-argn-fraud-audit/artifacts/sparkov_argn_control_v2/codec_digit_both"


@pytest.fixture(scope="module")
def fixture():
    torch.set_num_threads(2)
    ts = json.loads((SOURCE / "ModelStore/tgt-stats/stats.json").read_text())
    cs = json.loads((SOURCE / "ModelStore/ctx-stats/stats.json").read_text())
    frame = pd.read_parquet(SOURCE / "OriginalData/encoded-data/part.000000-trn.parquet")
    codec = PastState(ts)
    records = frame.iloc[:2].to_dict("records")
    for record in records:
        for key in record:
            if key.startswith("tgt:"):
                record[key] = record[key][:8]
        record[STATE_COLUMN] = codec.sequence(record)
    kwargs = dict(tgt_cardinalities=get_cardinalities(ts), ctx_cardinalities=get_cardinalities(cs),
                  tgt_seq_len_median=ts["seq_len"]["median"], tgt_seq_len_max=ts["seq_len"]["max"],
                  ctxseq_len_median=get_ctx_sequence_length(cs, key="median"), model_size=ModelSize.M,
                  column_order=None, device=torch.device("cpu"))
    return ts, codec, records, kwargs


def encoded_event(codec, label, gap=5, amount=100):
    event = {codec.prefixes["event_is_fraud"] + "__cat": [codec.codes[str(label)]]}
    for name, value in [("gap", gap), ("amount_or_numeric_value", amount)]:
        s = codec.columns[name]
        cents = int(round(value * 10 ** -s["min_decimal"]))
        for d in range(s["min_decimal"], s["max_decimal"] + 1):
            event[codec.prefixes[name] + f"__E{d}"] = [(cents // 10 ** (d - s["min_decimal"])) % 10 - s["min_digits"][f"E{d}"]]
    return event


def test_hand_worked_run_reset_and_empty_history(fixture):
    _, codec, _, _ = fixture
    m = codec.empty(1)
    assert np.count_nonzero(codec.features(m)) == 0
    for label, gap, run, elapsed in [(0, 999, 1, 0), (1, 5, 1, 0), (1, 0, 2, 0), (1, 5, 3, 5), (0, 9, 1, 0)]:
        m = codec.advance(m, encoded_event(codec, label, gap))
        assert (m[0, 2], m[0, 3]) == (run, elapsed)
    assert m[0, 0] == 5  # fraud termination did not end the customer


def test_suffix_and_current_event_cannot_change_past_features(fixture):
    _, codec, records, _ = fixture
    record = copy.deepcopy(records[0])
    original = codec.sequence(record)
    # Deliberately alter label, amount, gap at t=3 and every later row.
    for name in ["event_is_fraud", "gap", "amount_or_numeric_value"]:
        for key in record:
            if key.startswith(codec.prefixes[name] + "__"):
                record[key][3:] = 0
    altered = codec.sequence(record)
    np.testing.assert_array_equal(original[:4], altered[:4])
    assert not np.array_equal(original[4:], altered[4:])


def test_collation_never_crops_or_resets_history(fixture):
    _, _, records, _ = fixture
    batch = FullHistoryCollator(True, 2, torch.device("cpu"))(records)
    assert batch[STATE_COLUMN].shape == (2, 8, len(FEATURES))
    np.testing.assert_array_equal(batch[STATE_COLUMN][0].numpy(), records[0][STATE_COLUMN])


def test_native_equivalence_gradients_and_reload(fixture):
    ts, _, records, kwargs = fixture
    torch.manual_seed(101)
    baseline = model_class(ts, False)(**kwargs)
    torch.manual_seed(101)
    native = SequentialModel(**kwargs)
    native.column_order = baseline.column_order
    torch.manual_seed(101)
    state_model = model_class(ts, True)(**kwargs)
    batch = FullHistoryCollator(True, 2, torch.device("cpu"))(records)
    baseline.eval(); native.eval(); state_model.eval()
    with torch.no_grad():
        b, _ = baseline(batch, mode="trn")
        n, _ = native(batch, mode="trn")
        s, _ = state_model(batch, mode="trn")
    for key in n:
        torch.testing.assert_close(b[key], n[key], rtol=0, atol=0)
        torch.testing.assert_close(s[key][:, 0], n[key][:, 0], rtol=0, atol=0)
    for model, enabled in [(baseline, False), (state_model, True)]:
        model.train()
        loss = _calculate_sample_losses(model, batch).mean()
        loss.backward()
        magnitude = model.state_projection.weight.grad.abs().sum().item()
        assert (magnitude > 0) == enabled
    rebuilt_kwargs = dict(kwargs, model_size=get_model_units(state_model))
    rebuilt = model_class(ts, True)(**rebuilt_kwargs)
    rebuilt.load_state_dict(state_model.state_dict(), strict=True)


def test_stepwise_teacher_equivalence_and_batch_filtering(fixture):
    ts, codec, records, kwargs = fixture
    model = model_class(ts, True)(**kwargs).eval()
    batch = FullHistoryCollator(True, None, torch.device("cpu"))(records)
    key = codec.prefixes["event_is_fraud"] + "__cat"
    with torch.no_grad():
        teacher, _ = model(batch, mode="trn")
        context = model.context_compressor(batch)
        history = recurrent = None
        for index in range(5):
            fixed = {k: v[:, index] for k, v in batch.items() if k.startswith("tgt:")}
            captured = {}
            def save(module, args, output):
                if args[1] == key:
                    captured["logits"] = output.clone()
            handle = model.predictors.register_forward_hook(save)
            model(None, mode="gen", batch_size=2, fixed_values={k:v for k,v in fixed.items() if k != key},
                  context=context, history=history, history_state=recurrent)
            handle.remove()
            torch.testing.assert_close(captured["logits"][:, 0], teacher[key][:, index], rtol=1e-5, atol=2e-6)
            # Consume the REAL row to continue a teacher-forced prefix.
            _, history, recurrent = model(None, mode="gen", batch_size=2, fixed_values=fixed,
                                          context=context, history=history, history_state=recurrent)
            expected = np.stack([record[STATE_COLUMN][index + 1] for record in records])
            np.testing.assert_allclose(codec.features(recurrent[2][0].numpy()), expected)
        # Match the native generator's early-completed customer filtering exactly.
        mask = torch.tensor([False, True])
        filtered = tuple(t[:, mask, ...] for t in recurrent)
        filtered_context = [[t[mask, ...] for t in c] for c in context]
        fixed = {k: v[mask, 5] for k, v in batch.items() if k.startswith("tgt:")}
        _, _, next_state = model(None, mode="gen", batch_size=1, fixed_values=fixed,
                                context=filtered_context, history=history[mask], history_state=filtered)
        np.testing.assert_allclose(codec.features(next_state[2][0].numpy())[0], records[1][STATE_COLUMN][6])


def test_feature_caps_agree_with_native_decoder_tail_replacement(fixture):
    _, codec, _, _ = fixture
    for amount, gap in [(0, 0), (100, 5), (29999.99, 1999999)]:
        event = encoded_event(codec, 1, gap, amount)
        for name, cap in [("gap", DURATION_CAP), ("amount_or_numeric_value", AMOUNT_CAP)]:
            prefix = codec.prefixes[name] + "__"
            frame = pd.DataFrame({k[len(prefix):]:v for k,v in event.items() if k.startswith(prefix)})
            decoded = _decode_numeric_digit(frame, codec.columns[name]).to_numpy(dtype=float)
            np.testing.assert_allclose(np.minimum(codec.numeric(event, name), cap), np.minimum(decoded, cap))


def test_cached_one_step_matches_native_true_prefix_and_has_no_label_leakage(fixture):
    from benchmarks.argn_frozen_probe import teacher_with_history, one_step, label_probability
    from mostlyai.engine._tabular.generation import _fix_rare_token_probs, _translate_fixed_probs
    ts, codec, records, kwargs = fixture
    masks = _translate_fixed_probs(_fix_rare_token_probs(ts), ts)
    key = codec.prefixes['event_is_fraud'] + '__cat'
    for enabled in [False, True]:
        model = model_class(ts, enabled)(**kwargs).eval()
        batch = FullHistoryCollator(True, None, torch.device('cpu'))(records)
        with torch.no_grad():
            teacher, histories, context = teacher_with_history(model, batch)
            memory = codec.empty(2)
            history = recurrent = None
            for index in range(5):
                fixed = {k:v[:, index] for k,v in batch.items() if k.startswith('tgt:')}
                logits, _ = one_step(model, histories[:, index:index+1], torch.from_numpy(memory),
                                     context, {k:v for k,v in fixed.items() if k != key}, key, masks)
                torch.testing.assert_close(logits, teacher[key][:, index], rtol=1e-5, atol=2e-6)
                if history is not None:
                    torch.testing.assert_close(histories[:, index:index+1], history, rtol=1e-5, atol=2e-6)
                    np.testing.assert_allclose(memory, recurrent[2][0].numpy())
                _, history, recurrent = model(None, mode='gen', batch_size=2, fixed_values=fixed,
                                              context=context, history=history, history_state=recurrent)
                memory = codec.advance(memory, {k:v.numpy() for k,v in fixed.items()})
            altered = copy.deepcopy(batch)
            altered[key][:, 3:] = codec.codes['1']
            _, new_histories, _ = teacher_with_history(model, altered)
            torch.testing.assert_close(histories[:, :4], new_histories[:, :4], rtol=0, atol=0)
            # The native sampler suppresses the categorical unknown label.
            raw = torch.zeros(3, 3)
            p = label_probability(raw, key, codec.codes['1'], masks)
            torch.testing.assert_close(p, torch.full_like(p, .5))
