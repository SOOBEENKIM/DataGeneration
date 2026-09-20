"""Read-only, fixed-order ARGN replay with explicit sequence-control caveat.

Re-encodes complete observed/generated paths, including native remaining-length
codes. This measures fitted conditional predictions; it is not a causal isolation
of exposure bias. Future/current target invariance and prefix agreement are gates.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_cs_saf_external_audit_v1 import OUT, CONTRACT, digest, write
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import get_sequence_length_stats
from mostlyai.engine._tabular.encoding import encode_df, pad_tgt_sequences, _enrich_positional_columns, flatten_frame
from mostlyai.engine._tabular.probability import _initialize_model
from mostlyai.engine._tabular.training import BatchCollator


def encode_paths(raw, context, tgt_stats, ctx_stats):
    raw = raw.copy().reset_index(drop=True)
    ix = raw.groupby('entity_id', sort=False).cumcount()
    raw.loc[ix == 0, 'gap'] = 0.0
    encoded, _, key = encode_df(raw, tgt_stats, tgt_context_key='entity_id', n_jobs=1)
    maximum = get_sequence_length_stats(tgt_stats)['max']
    assert raw.groupby('entity_id').size().max() <= maximum
    encoded = _enrich_positional_columns(pad_tgt_sequences(encoded, key), key, maximum)
    encoded = flatten_frame(encoded, key)
    context = context.copy()
    context['entity_label'] = context.entity_label.astype(str)
    ctx, pk, _ = encode_df(context, ctx_stats, ctx_primary_key='entity_id', n_jobs=1)
    frame = encoded.merge(ctx, left_on=key, right_on=pk, how='left', validate='one_to_one')
    ids = frame[key].tolist()
    return frame.drop(columns=[key, pk]), ids, raw


@torch.no_grad()
def probabilities(model, frame, batch_size, mark_key, device, *, checks=False):
    collate = BatchCollator(is_sequential=True, max_sequence_window=None, device=torch.device(device))
    out = []
    errors = []
    for start in range(0, len(frame), batch_size):
        rows = frame.iloc[start:start + batch_size]
        batch = collate(rows.to_dict('records'))
        logits, _ = model(batch, mode='trn', column_order=model.tgt_columns)
        p = logits[mark_key].softmax(-1)
        # No training mark was pooled into RARE. Official generation suppresses code 0.
        p[..., 0] = 0
        p /= p.sum(-1, keepdim=True)
        if checks and start == 0:
            t = 5
            changed = {k: v.clone() for k, v in batch.items()}
            for k in model.tgt_cardinalities:
                changed[k][:, t+1:] = 0
            changed[mark_key][:, t] = (changed[mark_key][:, t] + 1) % model.tgt_cardinalities[mark_key]
            for k in model.tgt_cardinalities:
                if k.startswith('tgt:t2/'):
                    changed[k][:, t] = 0
            other, _ = model(changed, mode='trn', column_order=model.tgt_columns)
            errors.append(float((logits[mark_key][:, :t+1]-other[mark_key][:, :t+1]).abs().max()))
            prefix = {k: (v[:, :t+1] if k in model.tgt_cardinalities else v) for k,v in batch.items()}
            small, _ = model(prefix, mode='trn', column_order=model.tgt_columns)
            errors.append(float((logits[mark_key][:, :t+1]-small[mark_key]).abs().max()))
            assert max(errors) < 1e-5, errors
        for i, values in enumerate(rows[mark_key]):
            length = len(values)-1
            code = np.asarray(values[:length], dtype=int)
            pred = p[i, :length].cpu().numpy()
            previous = np.roll(code, 1)
            out.append((pred[np.arange(length), previous], pred[np.arange(length), code]))
    return out, errors


def audit(kappa, seed, device):
    torch.set_num_threads(1)
    c = json.loads(CONTRACT.read_text())
    folder = OUT/f'kappa_{kappa}/seed_{seed}'
    assert (folder/'DONE.json').exists()
    ws = Workspace(folder/'workspace')
    before = digest(ws.model_tabular_weights_path)
    model, ts, cs, _, _, _, _, _ = _initialize_model(workspace=ws, device=device)
    model.eval()
    mark_stats = ts['columns']['receiver_or_mark']
    assert mark_stats['no_of_rare_categories'] == 0
    mark_key = next(k for k in model.tgt_cardinalities if k.startswith('tgt:t1/'))
    inp = OUT/f'kappa_{kappa}/input'
    plan = pd.read_parquet(inp/'plan.parquet').drop(columns='source_train_entity_id')
    validation = pd.read_parquet(inp/'validation_canonical.parquet')
    vp = pd.read_parquet(inp/'validation_context.parquet')
    vp['__saf_planned_length'] = vp.entity_id.map(validation.groupby('entity_id').size())
    sources = [('validation', validation, vp)]
    sources += [(f'generated_{t}', pd.read_parquet(folder/f'generated_{t}.parquet'), plan) for t in c['generation_seeds']]
    checks = {}
    for name, raw, context in sources:
        frame, ids, raw = encode_paths(raw, context, ts, cs)
        pred, error = probabilities(model, frame, 128, mark_key, device, checks=True)
        groups = dict(tuple(raw.groupby('entity_id', sort=False)))
        output = []
        for eid, (rep, obs) in zip(ids, pred):
            part = groups[eid].copy().reset_index(drop=True)
            assert len(part) == len(rep)
            part['event_index'] = np.arange(len(part))
            part['predicted_repeat'] = rep
            part['observed_mark_probability'] = obs
            part['actual_repeat'] = part.receiver_or_mark.eq(part.receiver_or_mark.shift()).astype(int)
            output.append(part)
        pd.concat(output, ignore_index=True).to_parquet(folder/f'replay_{name}.parquet', index=False)
        checks[name] = {'max_current_target_future_or_prefix_difference': max(error), 'entities': len(ids)}
    assert before == digest(ws.model_tabular_weights_path)
    write(folder/'replay_checks.json', dict(checks=checks, parameters=sum(p.numel() for p in model.parameters()),
          weights_unchanged=True, caveat='reencoded observed paths; sequence control uses realized length; no isolated causal exposure-bias claim'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--kappa',type=int,required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--device',default='cpu')
    a=p.parse_args(); audit(a.kappa,a.seed,a.device)
