"""Official TabularARGN audit. Preparation/evaluation use the original environment.

Fit/roundtrip/sample use an isolated official 2.4.0 environment. No oracle targets,
test data, patched model, generated-content repair, or automatic scientific retry.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONTRACT = ROOT / 'configs/benchmark_v2/cs_saf_external_audit_v1.json'
OUT = ROOT / 'artifacts/cs_saf/external_audit_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(source):
    import numpy as np
    from data.cof_seqgen_saf_tensorizer import load_canonical_dataset
    from generators.cof_seqgen_saf_external_baselines import _train_frames
    c = json.loads(CONTRACT.read_text())
    for k in c['kappas']:
        dest = OUT / f'kappa_{k}' / 'input'
        dest.mkdir(parents=True, exist_ok=False)
        src = source / f'data/cs_saf/prevalence_v1/pi_0.10_kappa_{k}'
        manifest = json.loads((src / 'cs_saf_manifest.json').read_text())
        for name, expected in manifest['files'].items():
            assert digest(src / name) == expected, name
        dataset = load_canonical_dataset(src, allowed_splits=('train', 'validation'))
        parent, child = _train_frames(dataset)
        # Explicit categorical type avoids treating the group label as continuous.
        parent['entity_label'] = parent.entity_label.astype(str)
        child = child.sort_values('entity_id', kind='stable').reset_index(drop=True)
        parent.to_parquet(dest / 'train_context.parquet', index=False)
        child.to_parquet(dest / 'train_events.parquet', index=False)
        ids = set(dataset.entity_ids_for_split('validation'))
        validation = dataset.events[dataset.events.entity_id.isin(ids)].copy()
        validation.to_parquet(dest / 'validation_canonical.parquet', index=False)
        dataset.static_context[dataset.static_context.entity_id.isin(ids)].to_parquet(dest / 'validation_context.parquet', index=False)
        pos = np.random.default_rng(c['generation_plan_seed']).integers(len(parent), size=c['generation_entities'])
        plan = parent.iloc[pos].reset_index(drop=True).copy()
        plan.insert(1, 'source_train_entity_id', plan.entity_id)
        plan['entity_id'] = [f'external-audit-{i:08d}' for i in range(len(plan))]
        plan.to_parquet(dest / 'plan.parquet', index=False)
        assert set(plan.source_train_entity_id) <= set(parent.entity_id)
        assert not set(parent.entity_id) & ids
        write(dest / 'provenance.json', dict(source=str(src), manifest_sha256=digest(src / 'cs_saf_manifest.json'),
              train_entities=len(parent), train_events=len(child), validation_entities=len(ids),
              data_generation_seed=42, test_accessed=False,
              files={p.name: digest(p) for p in sorted(dest.glob('*.parquet'))}))


def fit(kappa, seed, device, smoke):
    import numpy as np
    import pandas as pd
    import torch
    from mostlyai.engine import TabularARGN, set_random_state, generate
    from mostlyai.engine._common import load_generated_data
    from mostlyai.engine._workspace import Workspace
    from mostlyai.engine._tabular.encoding import encode_df
    from mostlyai.engine._tabular.generation import _decode_df
    assert importlib.metadata.version('mostlyai-engine') == '2.4.0'
    c = json.loads(CONTRACT.read_text())
    assert kappa in c['kappas'] and seed in c['training_seeds']
    torch.set_num_threads(c['cpu_threads'])
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(c['gpu_memory_fraction'])
    else:
        assert smoke, 'scientific fits use the registered GPU budget'
    inp = OUT / f'kappa_{kappa}' / 'input'
    provenance = json.loads((inp / 'provenance.json').read_text())
    for name, expected in provenance['files'].items():
        assert digest(inp / name) == expected
    parent = pd.read_parquet(inp / 'train_context.parquet')
    child = pd.read_parquet(inp / 'train_events.parquet')
    plan = pd.read_parquet(inp / 'plan.parquet')
    folder = OUT / ('cpu_smoke' if smoke else f'kappa_{kappa}/seed_{seed}')
    folder.mkdir(parents=True, exist_ok=False)
    if smoke:
        parent = parent.groupby('entity_label', group_keys=False).head(128).copy()
        child = child[child.entity_id.isin(parent.entity_id)].copy()
        plan = plan.iloc[:16].copy()
    options = dict(model=c['model'], max_epochs=2 if smoke else c['max_epochs'],
        max_training_time=c['max_training_minutes'], batch_size=32 if smoke else c['batch_size'],
        max_sequence_window=c['max_sequence_window'], value_protection=c['value_protection'],
        enable_flexible_generation=c['enable_flexible_generation'],
        tgt_context_key='entity_id', ctx_primary_key='entity_id',
        tgt_encoding_types={'gap': 'TABULAR_NUMERIC_AUTO', 'receiver_or_mark': 'TABULAR_CATEGORICAL',
                            'amount_or_numeric_value': 'TABULAR_NUMERIC_AUTO'},
        ctx_encoding_types={'entity_label': 'TABULAR_CATEGORICAL', '__saf_planned_length': 'TABULAR_NUMERIC_AUTO'},
        ctx_data=parent, random_state=seed, device=device,
        workspace_dir=folder / 'workspace', verbose=1)
    write(folder / 'start.json', dict(contract_sha256=digest(CONTRACT), kappa=kappa, seed=seed,
        smoke=smoke, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        versions={x: importlib.metadata.version(x) for x in ['mostlyai-engine','torch','pandas','numpy']},
        train_entities=len(parent), test_accessed=False))
    started = time.monotonic()
    model = TabularARGN(**options).fit(child)
    fit_seconds = time.monotonic() - started
    ws = Workspace(folder / 'workspace')
    weights = Path(ws.model_tabular_weights_path)
    assert weights.exists()
    weights_digest = digest(weights)
    write(folder / 'fit_complete.json', dict(fit_seconds=fit_seconds, weights_sha256=weights_digest,
          sequential=ws.tgt_stats.read()['is_sequential']))
    assert ws.tgt_stats.read()['is_sequential']
    # Public generate() reloads weights from the saved workspace, as in the paper scripts.
    del model
    if not smoke:
        raw = pd.read_parquet(inp / 'validation_canonical.parquet')
        raw.loc[raw.event_index == 0, 'gap'] = 0.0
        raw = raw[['entity_id', 'gap', 'receiver_or_mark', 'amount_or_numeric_value']].reset_index(drop=True)
        stats = ws.tgt_stats.read()
        for gen_seed in c['generation_seeds']:
            set_random_state(gen_seed)
            encoded, _, key = encode_df(raw, stats, tgt_context_key='entity_id', n_jobs=1)
            decoded = _decode_df(encoded, stats, context_key=key).rename(columns={key:'entity_id'})
            assert len(decoded) == len(raw)
            assert decoded.entity_id.tolist() == raw.entity_id.tolist(), 'roundtrip entity/order changed'
            decoded.to_parquet(folder / f'roundtrip_{gen_seed}.parquet', index=False)
    for gen_seed in c['generation_seeds'][:1 if smoke else 5]:
        set_random_state(gen_seed)
        generate(ctx_data=plan.drop(columns='source_train_entity_id'), batch_size=256,
                 sampling_temperature=c['temperature'], sampling_top_p=c['top_p'], device=device,
                 workspace_dir=folder / 'workspace')
        generated = load_generated_data(folder / 'workspace')
        generated.to_parquet(folder / f'generated_{gen_seed}.parquet', index=False)
        print('SAMPLE_COMPLETE', kappa, seed, gen_seed, len(generated), flush=True)
        assert digest(weights) == weights_digest, 'sampling modified fitted weights'
    write(folder / 'DONE.json', dict(fit_seconds=fit_seconds, total_seconds=time.monotonic()-started,
          weights_sha256=weights_digest, test_accessed=False, smoke=smoke))


def evaluate():
    import numpy as np
    import pandas as pd
    from experiments.cs_saf_generation_metrics import CachedGroupMetrics
    c = json.loads(CONTRACT.read_text())
    records = []
    for k in c['kappas']:
        inp = OUT / f'kappa_{k}' / 'input'
        train = pd.read_parquet(inp / 'train_events.parquet')
        train['event_index'] = train.groupby('entity_id', sort=False).cumcount()
        train.loc[train.event_index == 0, 'gap'] = np.nan
        validation = pd.read_parquet(inp / 'validation_canonical.parquet')
        parents = pd.read_parquet(inp / 'train_context.parquet')
        val_parent = pd.read_parquet(inp / 'validation_context.parquet')
        plan = pd.read_parquet(inp / 'plan.parquet')
        groups = {}
        for group in ['pooled', 'context_0', 'context_1']:
            label = group[-1]
            tr_ids = set(parents.entity_id if group == 'pooled' else parents.loc[parents.entity_label.astype(str) == label, 'entity_id'])
            va_ids = set(val_parent.entity_id if group == 'pooled' else val_parent.loc[val_parent.entity_label.astype(str) == label, 'entity_id'])
            ge_ids = set(plan.entity_id if group == 'pooled' else plan.loc[plan.entity_label.astype(str) == label, 'entity_id'])
            groups[group] = (CachedGroupMetrics(train[train.entity_id.isin(tr_ids)], validation[validation.entity_id.isin(va_ids)]), va_ids, ge_ids)
        for seed in c['training_seeds']:
            folder = OUT / f'kappa_{k}/seed_{seed}'
            if not (folder / 'DONE.json').exists():
                continue
            for kind in ['roundtrip', 'generated']:
                for tape in c['generation_seeds']:
                    frame = pd.read_parquet(folder / f'{kind}_{tape}.parquet')
                    frame['event_index'] = frame.groupby('entity_id', sort=False).cumcount()
                    frame.loc[frame.event_index == 0, 'gap'] = np.nan
                    for group, (metric, va_ids, ge_ids) in groups.items():
                        ids = va_ids if kind == 'roundtrip' else ge_ids
                        part = frame[frame.entity_id.isin(ids)]
                        row = dict(kappa=k, seed=seed, tape=tape, kind=kind, group=group,
                              planned_entities=len(ids), generated_entities=part.entity_id.nunique(), events=len(part))
                        row.update(metric.score(part))
                        records.append(row)
    assert records, 'no completed scientific fits'
    report = ROOT / 'docs/cs_saf/external_audit_v1'
    pd.DataFrame(records).to_csv(report / 'metrics.csv', index=False)
    write(report / 'metrics.json', records)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'fit', 'smoke', 'evaluate'])
    p.add_argument('--source-root', type=Path)
    p.add_argument('--kappa', type=int, default=1)
    p.add_argument('--seed', type=int, default=20260920)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.mode == 'prepare':
        assert a.source_root is not None
        prepare(a.source_root)
    elif a.mode == 'evaluate':
        evaluate()
    else:
        fit(a.kappa, a.seed, a.device, a.mode == 'smoke')
