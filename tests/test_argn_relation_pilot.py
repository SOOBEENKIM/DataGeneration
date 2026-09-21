import importlib
import torch
import pandas as pd
from models.argn_relation_pilot import official_extension, RelationPath, GAP, CATEGORY, tensor_hash


def kwargs():
    return dict(tgt_cardinalities={GAP:103, 'tgt:t1/c1__cat':8, 'tgt:t2/c2__bin':12, CATEGORY:15},
                tgt_seq_len_median=8, tgt_seq_len_max=16, ctx_cardinalities={},
                ctxseq_len_median={}, model_size='S', column_order=None, device=torch.device('cpu'))


def batch():
    torch.manual_seed(123)
    return {k: torch.randint(1, n, (3, 7, 1)) for k, n in kwargs()['tgt_cardinalities'].items()}


def test_native_initial_equivalence_and_exact_capacity():
    native = importlib.import_module('mostlyai.engine._tabular.argn').SequentialModel
    x = batch()
    torch.manual_seed(7)
    original = native(**kwargs()).eval()
    expected, _ = original(x, mode='trn', column_order=original.tgt_columns)
    hashes, extras = [], []
    for arm in ('A', 'G', 'R'):
        torch.manual_seed(7)
        with official_extension(arm, on_init=lambda m,h:hashes.append(h)) as ext:
            model = ext['model_class'](**kwargs()).eval()
            actual, _ = model(x, mode='trn', column_order=model.tgt_columns)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            if arm != 'A':
                extras.append((sum(p.numel() for p in model.relation_path.parameters()),
                               tensor_hash(model.relation_path.state_dict())))
    assert len(set(hashes)) == 1
    assert extras[0] == extras[1] and extras[0][0] == 2143
    assert importlib.import_module('mostlyai.engine._tabular.argn').SequentialModel is native


def test_current_and_future_category_do_not_leak():
    x = batch()
    with official_extension('R') as ext:
        model = ext['model_class'](**kwargs()).eval()
        torch.nn.init.normal_(model.relation_path.output.weight)
        original, _ = model(x, mode='trn', column_order=model.tgt_columns)
        changed = {k:v.clone() for k,v in x.items()}
        changed[CATEGORY][:,3:] = (changed[CATEGORY][:,3:] + 1) % 15
        for k in changed:
            changed[k][:,4:] = 1
        updated, _ = model(changed, mode='trn', column_order=model.tgt_columns)
        torch.testing.assert_close(updated[CATEGORY][:,:4], original[CATEGORY][:,:4], rtol=0, atol=0)


def test_gradient_and_roundtrip():
    x = batch()
    for arm in ('G', 'R'):
        with official_extension(arm) as ext:
            model = ext['model_class'](**kwargs())
            logits, _ = model(x, mode='trn', column_order=model.tgt_columns)
            loss = torch.nn.functional.cross_entropy(logits[CATEGORY].flatten(0,1), x[CATEGORY].flatten())
            loss.backward()
            assert model.relation_path.output.weight.grad.abs().sum() > 0
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            opt.step()
            copy = ext['model_class'](**kwargs())
            copy.load_state_dict(model.state_dict(), strict=True)
            assert tensor_hash(copy.state_dict()) == tensor_hash(model.state_dict())


def test_training_and_recursive_generation_agree():
    x = batch()
    with official_extension('R') as ext:
        model = ext['model_class'](**kwargs()).eval()
        torch.nn.init.normal_(model.relation_path.output.weight, std=.1)
        teacher, _ = model(x, mode='trn', column_order=model.tgt_columns)
        history = state = None
        captured = []
        handle = model.relation_path.register_forward_hook(lambda m,args,out: captured.append((args[0].clone(),out.clone())))
        last = None
        for t in range(7):
            # Native sampling is used for category; other columns are supplied as
            # current observed values. Feed its sampled category into the next step.
            model.set_previous_generated(last, t)
            fixed = {k: v[:,t] for k,v in x.items() if k != CATEGORY}
            outputs, history, state = model(None, mode='gen', batch_size=3, fixed_values=fixed,
                history=history, history_state=state, column_order=model.tgt_columns)
            previous = captured[-1][0].flatten()
            if t == 0:
                assert previous.eq(0).all()
            else:
                torch.testing.assert_close(previous, torch.tensor(last[CATEGORY].to_numpy()))
            # Record generated values, not teacher categories.
            last = pd.DataFrame({CATEGORY: outputs[CATEGORY].flatten().numpy()})
        handle.remove()
        assert model.relation_calls == 7


def test_same_observed_prefix_training_generation_logit_match():
    x = batch()
    for arm in ('G','R'):
        with official_extension(arm) as ext:
            model = ext['model_class'](**kwargs()).eval()
            torch.nn.init.normal_(model.relation_path.output.weight, std=.1)
            reference, _ = model(x, mode='trn', column_order=model.tgt_columns)
            history = state = None
            # First run fixed complete events to build the exact observed prefix.
            for t in range(7):
                prior = None if t == 0 else pd.DataFrame({CATEGORY:x[CATEGORY][:,t-1,0].numpy()})
                model.set_previous_generated(prior, t)
                scores = []
                native_adjust = model._relation_adjust
                def capture(logits, sub_col, mode, data, outputs):
                    result = native_adjust(logits, sub_col, mode, data, outputs)
                    if sub_col == CATEGORY:
                        scores.append(result.detach())
                    return result
                model._relation_adjust = capture
                fixed = {k:v[:,t] for k,v in x.items() if k != CATEGORY}
                _, _, _ = model(None, mode='gen', batch_size=3, fixed_values=fixed,
                    history=history, history_state=state, column_order=model.tgt_columns)
                model._relation_adjust = native_adjust
                torch.testing.assert_close(scores[0][:,0], reference[CATEGORY][:,t], rtol=1e-5, atol=1e-6)
                # Replay all observed values to advance the state without sample drift.
                _, history, state = model(None, mode='gen', batch_size=3,
                    fixed_values={k:v[:,t] for k,v in x.items()}, history=history,
                    history_state=state, column_order=model.tgt_columns)


def test_generator_source_and_class_restoration_on_exception():
    argn = importlib.import_module('mostlyai.engine._tabular.argn')
    old = argn.SequentialModel
    try:
        with official_extension('R') as ext:
            assert len(ext['generation_source_sha256']) == 64
            raise RuntimeError('expected scoped abort')
    except RuntimeError:
        pass
    assert argn.SequentialModel is old


def test_workspace_copies_encoding_metadata_without_checkpoint():
    import tempfile
    from pathlib import Path
    from scripts.run_argn_relation_pilot import prepare_workspace
    with tempfile.TemporaryDirectory(prefix='argn-pilot-cpu-') as folder:
        prepare_workspace(Path(folder) / 'workspace')
