"""Read-only teacher-forcing and path interventions on frozen ARGN checkpoints."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json

from run_argn_fraud_audit import (
    ROOT, OUT, OLD, DOCS, CORE, LABEL, MERCHANT, config, load_prepared, digest, write,
    pd, np, torch,
)
from mostlyai.engine import set_random_state
from mostlyai.engine._common import get_sequence_length_stats, RIDX_SUB_COLUMN_PREFIX
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._tabular.common import get_argn_column_names
from mostlyai.engine._tabular.encoding import encode_df, pad_tgt_sequences, _enrich_positional_columns, flatten_frame
from mostlyai.engine._tabular.probability import _initialize_model
from mostlyai.engine._tabular.training import BatchCollator, _calculate_sample_losses
import torch.nn.functional as F


def encoded_prefix(raw, parents, ts, cs):
    raw = raw.sort_values(['entity_id', 'event_index'], kind='stable').copy()
    raw['gap'] = raw.gap.fillna(0)
    raw = raw[['entity_id', *ts['columns']]]
    enc, _, key = encode_df(raw, ts, tgt_context_key='entity_id', n_jobs=1)
    maximum = get_sequence_length_stats(ts)['max']
    enc = flatten_frame(_enrich_positional_columns(pad_tgt_sequences(enc, key), key, maximum), key)
    for c in enc:
        if c.startswith('tgt:'): enc[c] = enc[c].map(lambda x: x[:512])
    par = parents.loc[parents.entity_id.isin(raw.entity_id)].copy()
    ctx, pk, _ = encode_df(par, cs, ctx_primary_key='entity_id', n_jobs=1)
    merged = enc.merge(ctx, left_on=key, right_on=pk, how='left', validate='one_to_one')
    return merged.drop(columns=list(set([key, pk])))


@contextmanager
def intervention(model, mode, merchant_column):
    handle = None
    if mode == 'shuffle_current_merchant':
        def column_hook(module, args, result):
            result = dict(result)
            result[merchant_column] = result[merchant_column].roll(1, dims=0)
            return result
        handle = model.column_embedders.register_forward_hook(column_hook)
    elif mode == 'zero_history_after_first':
        def history_hook(module, args, result):
            history, state = result
            history = history.clone(); history[:, 1:] = 0
            return history, state
        handle = model.history_compressor.register_forward_hook(history_hook)
    try:
        yield
    finally:
        if handle is not None: handle.remove()


@torch.no_grad()
def probe(name):
    config(); base, _ = load_prepared(); set_random_state(20260926)
    folder = OLD/'artifacts/argn_relation_pilot_v1/A' if name == 'old_A' else OUT/name
    if name != 'old_A': assert (folder/'FIT.json').exists()
    ws = Workspace(folder/'workspace'); before = digest(ws.model_tabular_weights_path)
    model, ts, cs, *_ = _initialize_model(workspace=ws, device='cpu')
    model.eval()
    # Explicit ordering makes all three interventions use the same conditional prediction task.
    model.column_order = list(model.tgt_columns)
    frame = encoded_prefix(pd.read_parquet(base/'validation.parquet'), pd.read_parquet(base/'context.parquet'), ts, cs)
    columns = dict(zip(ts['columns'], get_argn_column_names(ts['columns'], list(ts['columns']))))
    names = {c: next(k for k in model.tgt_cardinalities if k.startswith(columns[c]+'__')) for c in ts['columns']}
    assert all(sum(k.startswith(col+'__') for k in model.tgt_cardinalities) == 1 for col in columns.values())
    collate = BatchCollator(True, None, torch.device('cpu'))
    totals = defaultdict(lambda: [0., 0, 0])
    scale = None
    for start in range(0, len(frame), 3):
        batch = collate(frame.iloc[start:start+3].to_dict('records'))
        mask = torch.zeros_like(batch[names['category']].squeeze(-1), dtype=torch.bool)
        for k in batch:
            if k.startswith(RIDX_SUB_COLUMN_PREFIX): mask |= batch[k].squeeze(-1).ne(0)
        if scale is None:
            once = float(_calculate_sample_losses(model, batch).mean())
            twice = float(_calculate_sample_losses(model, {k: torch.cat([v, v], dim=0) for k,v in batch.items()}).mean())
            scale = dict(batch_size=len(mask), native_loss=once, duplicated_batch_loss=twice, ratio=twice/once)
            assert np.isclose(twice/once, .5, atol=1e-5), 'Native loss normalization behavior changed'
        position = torch.arange(mask.shape[1])[None,:].expand_as(mask)
        groups = {'all': mask, 'first': mask & position.eq(0), 'later': mask & position.gt(0)}
        if LABEL in names:
            y = batch[names[LABEL]].squeeze(-1)
            for value in ['0','1']: groups['class_'+value] = mask & y.eq(ts['columns'][LABEL]['codes'][value])
        for mode in ['teacher', 'shuffle_current_merchant', 'zero_history_after_first']:
            with intervention(model, mode, columns[MERCHANT]):
                output, _ = model(batch, mode='trn', column_order=model.tgt_columns)
            for field, key in names.items():
                target = batch[key].squeeze(-1)
                losses = F.cross_entropy(output[key].flatten(0,1), target.flatten(), reduction='none').reshape_as(target)
                correct = output[key].argmax(-1).eq(target)
                for group, use in groups.items():
                    if field == 'gap': use = use & position.gt(0)
                    acc = totals[(mode, field, group)]
                    acc[0] += float(losses[use].double().sum()); acc[1] += int(use.sum()); acc[2] += int(correct[use].sum())
    rows = [dict(run=name, intervention=k[0], field=k[1], group=k[2], events=v[1],
                 nll=v[0]/v[1] if v[1] else None, accuracy=v[2]/v[1] if v[1] else None) for k,v in totals.items()]
    # Interventions must not change information paths upstream of the altered component.
    for field in ['gap', MERCHANT]:
        assert np.allclose(totals[('teacher',field,'all')], totals[('shuffle_current_merchant',field,'all')])
    for field in names:
        assert np.allclose(totals[('teacher',field,'first')], totals[('zero_history_after_first',field,'first')])
    assert digest(ws.model_tabular_weights_path) == before
    pd.DataFrame(rows).to_csv(DOCS/f'probe_{name}.csv', index=False)
    write(DOCS/f'probe_{name}.json', dict(checkpoint_sha256=before, protocol_sha256=digest(DOCS/'PROBE_PROTOCOL.md'),
          script_sha256=digest(__file__), customers=len(frame), prefix_limit=512, native_loss_batch_scaling=scale,
          test_outcomes_accessed=False, checkpoint_unchanged=True))
    print(pd.DataFrame(rows).query("field == 'category' or (field == @LABEL and group == 'all')").to_string(index=False), flush=True)


def lookup():
    config(); base, _ = load_prepared()
    fit = pd.read_parquet(base/'fit.parquet'); real = pd.read_parquet(base/'validation.parquet')
    table = fit.groupby(MERCHANT).category.agg(['nunique', lambda x: x.mode().iloc[0]])
    pred = real[MERCHANT].map(table.iloc[:,1])
    result = dict(fit_merchants=len(table), fit_merchants_with_multiple_categories=int(table['nunique'].gt(1).sum()),
        validation_events=len(real), merchant_coverage=float(pred.notna().mean()),
        lookup_category_accuracy=float(pred.eq(real.category).mean()),
        prefix_events=int(real.event_index.lt(512).sum()),
        prefix_lookup_category_accuracy=float(pred[real.event_index.lt(512)].eq(real.loc[real.event_index.lt(512),'category']).mean()),
        uses='fit-only modal-category lookup; diagnostic, not a sequence generator')
    write(DOCS/'merchant_lookup_diagnostic.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('name', choices=['old_A','seed_20260927','seed_20260928','lookup'])
    name = parser.parse_args().name
    lookup() if name == 'lookup' else probe(name)
