"""Three registered fits on isolated copies of an existing official workspace."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OLD = ROOT.parent / 'research-cs-saf-external-audit'
SOURCE = OLD / 'artifacts/cs_saf/external_port_v1'
OUT = ROOT / 'artifacts/argn_relation_pilot_v1'
CONFIG = ROOT / 'configs/argn_relation_pilot_v1.json'

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from benchmarks.cs_saf_external import evaluate, fit_state, tv
from models.argn_relation_pilot import official_extension, GAP, CATEGORY


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def original_manifest():
    paths = list((SOURCE / 'input/sparkov').glob('*'))
    paths += list((SOURCE / 'runs/sparkov/ARGN/workspace').rglob('*'))
    paths += list((OLD / 'models').glob('cs_saf*.py'))
    return {str(p.relative_to(OLD)): digest(p) for p in paths if p.is_file()}


def inputs():
    p = SOURCE / 'input/sparkov'
    pre = json.loads((p / 'preflight.json').read_text())
    for name, h in pre['files'].items():
        assert digest(p / name) == h
    events = pd.read_parquet(p / 'events.parquet')
    parents = pd.read_parquet(p / 'context.parquet')
    roles = pd.read_parquet(p / 'roles.parquet')
    plan = pd.read_parquet(p / 'plan.parquet')
    frames = {r: events.loc[events.entity_id.isin(roles.loc[roles.role.eq(r), 'entity_id'])].copy()
              for r in ('fit', 'check', 'validation')}
    return frames, parents, plan, fit_state(frames['fit'], 'sparkov')


def encoded_validation(raw, parents, ts, cs):
    from mostlyai.engine._common import get_sequence_length_stats
    from mostlyai.engine._tabular.encoding import (
        encode_df, pad_tgt_sequences, _enrich_positional_columns, flatten_frame)
    raw = raw.sort_values(['entity_id', 'event_index'], kind='stable').copy()
    raw.loc[raw.event_index.eq(0), 'gap'] = 0.
    raw = raw[['entity_id', 'gap', 'receiver_or_mark', 'amount_or_numeric_value', 'category']]
    encoded, _, key = encode_df(raw, ts, tgt_context_key='entity_id', n_jobs=1)
    maximum = get_sequence_length_stats(ts)['max']
    assert raw.groupby('entity_id').size().max() <= maximum
    encoded = flatten_frame(_enrich_positional_columns(pad_tgt_sequences(encoded, key), key, maximum), key)
    par = parents.loc[parents.entity_id.isin(raw.entity_id)].copy()
    for name, st in cs['columns'].items():
        if st['encoding_type'] == 'TABULAR_CATEGORICAL':
            par[name] = par[name].astype(str)
    ctx, pk, _ = encode_df(par, cs, ctx_primary_key='entity_id', n_jobs=1)
    data = encoded.merge(ctx, left_on=key, right_on=pk, how='left', validate='one_to_one')
    return data.drop(columns=list(set([key, pk])))


@torch.no_grad()
def prediction(model, frame, device):
    from mostlyai.engine._tabular.training import BatchCollator
    from mostlyai.engine._common import RIDX_SUB_COLUMN_PREFIX
    collate = BatchCollator(is_sequential=True, max_sequence_window=None, device=torch.device(device))
    names = {'gap': GAP, 'receiver_or_mark': 'tgt:t1/c1__cat',
             'amount_or_numeric_value': 'tgt:t2/c2__bin', 'category': CATEGORY}
    sums, counts = {k: 0. for k in names}, {k: 0 for k in names}
    model.eval()
    for start in range(0, len(frame), 2):
        batch = collate(frame.iloc[start:start+2].to_dict('records'))
        outputs, _ = model(batch, mode='trn', column_order=model.tgt_columns)
        mask = torch.zeros_like(batch[CATEGORY].squeeze(-1), dtype=torch.bool)
        for key in batch:
            if key.startswith(RIDX_SUB_COLUMN_PREFIX):
                mask |= batch[key].squeeze(-1).ne(0)
        for name, key in names.items():
            target = batch[key].squeeze(-1).long()
            use = mask.clone()
            if name == 'gap':
                use[:, 0] = False
            loss = F.cross_entropy(outputs[key].flatten(0, 1), target.flatten(), reduction='none').reshape_as(target)
            sums[name] += float(loss[use].double().sum())
            counts[name] += int(use.sum())
    return dict(nll={k: sums[k] / counts[k] for k in sums}, counts=counts)


def run(arm, device):
    from mostlyai.engine import train, set_random_state
    from mostlyai.engine._common import load_generated_data
    from mostlyai.engine._workspace import Workspace
    from mostlyai.engine._tabular.probability import _initialize_model
    cfg = json.loads(CONFIG.read_text())
    assert arm in cfg['arms'] and importlib.metadata.version('mostlyai-engine') == cfg['argn_version']
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert physical is not None and physical.isdigit(), 'One explicitly selected physical GPU required'
    admission_raw = subprocess.check_output(['nvidia-smi', f'--id={physical}',
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    used, utilization = map(int, admission_raw.split(','))
    assert used < 512 and utilization < 5, f'GPU occupied: {physical} {admission_raw}'
    assert device == 'cuda' and torch.cuda.is_available()
    torch.set_num_threads(cfg['cpu_threads'])
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    folder = OUT / arm
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    before = original_manifest()
    write(folder / 'gpu_admission.json', dict(physical_gpu=physical, memory_used_mib=used,
        utilization_percent=utilization, hostname=subprocess.check_output(['hostname'],text=True).strip(),
        pid=os.getpid(), per_process_memory_fraction=cfg['gpu_memory_fraction']))
    write(folder / 'original_manifest.json', before)
    write(folder / 'START.json', dict(config_sha256=digest(CONFIG), arm=arm,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), fit_seed=cfg['fit_seed'],
        input_preflight_sha256=digest(SOURCE / 'input/sparkov/preflight.json'),
        versions={p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'pandas', 'mostlyai-engine')},
        test_outcomes_accessed=False))
    try:
        wsdir = folder / 'workspace'
        shutil.copytree(SOURCE / 'runs/sparkov/ARGN/workspace/OriginalData', wsdir / 'OriginalData')
        models = []
        def on_init(model, base_hash):
            models.append(model)
            if not (folder / 'initial.json').exists():
                write(folder / 'initial.json', dict(base_tensor_sha256=base_hash,
                    parameters=sum(p.numel() for p in model.parameters()),
                    extra_parameters=sum(p.numel() for p in model.relation_path.parameters()) if arm != 'A' else 0,
                    extra_sha256=None if arm == 'A' else __import__('models.argn_relation_pilot', fromlist=['tensor_hash']).tensor_hash(model.relation_path.state_dict())))
        set_random_state(cfg['fit_seed'])
        with official_extension(arm, cfg['rank'], cfg['extra_seed'], on_init) as ext:
            write(folder / 'adapter_sources.json', {k: v for k, v in ext.items() if k.endswith('sha256')})
            train(model=cfg['model'], max_epochs=cfg['max_epochs'], max_training_time=cfg['max_training_minutes'],
                  batch_size=cfg['batch_size'], max_sequence_window=cfg['max_sequence_window'],
                  enable_flexible_generation=False, device=device, workspace_dir=wsdir)
            fit_seconds = time.monotonic() - start
            ws = Workspace(wsdir)
            weights_before = digest(ws.model_tabular_weights_path)
            torch.save(models[0].state_dict(), folder / 'last.pt')
            frames, parents, plan, metric_state = inputs()
            model, ts, cs, *_ = _initialize_model(workspace=ws, device=device)
            frame = encoded_validation(frames['validation'], parents, ts, cs)
            pred = prediction(model, frame, device)
            write(folder / 'prediction.json', pred)
            generations = []
            ctx = plan[parents.columns].copy()
            for name, st in cs['columns'].items():
                if st['encoding_type'] == 'TABULAR_CATEGORICAL':
                    ctx[name] = ctx[name].astype(str)
            for seed in cfg['generation_seeds']:
                set_random_state(seed)
                started = time.monotonic()
                ext['generate'](ctx_data=ctx, batch_size=64, device=device, workspace_dir=wsdir,
                                sampling_temperature=1., sampling_top_p=1.)
                raw = load_generated_data(wsdir)
                assert set(raw.entity_id) <= set(plan.entity_id)
                raw.to_parquet(folder / f'native_{seed}.parquet', index=False)
                output = raw.copy()
                output['event_index'] = output.groupby('entity_id', sort=False).cumcount()
                output.loc[output.event_index.eq(0), 'gap'] = np.nan
                output['timestamp'] = output.gap.fillna(0).groupby(output.entity_id, sort=False).cumsum()
                output.to_parquet(folder / f'generated_{seed}.parquet', index=False)
                score = evaluate(frames['validation'], output, metric_state)
                score['merchant_category_joint_tv'] = tv(frames['validation'], output, ['receiver_or_mark', 'category'])
                trace = models[-1]
                generations.append(dict(seed=seed, seconds=time.monotonic()-started, metrics=score,
                    relation_calls=trace.relation_calls, relation_events=trace.relation_sampled_events,
                    relation_trace=trace.relation_trace))
                if arm != 'A':
                    assert trace.relation_sampled_events > 0
                assert digest(ws.model_tabular_weights_path) == weights_before
                print('GENERATED', arm, seed, len(output), flush=True)
            assert original_manifest() == before
            progress = pd.read_csv(ws.model_progress_messages_path)
            progress.to_csv(folder / 'history.csv', index=False)
            best = progress.loc[progress.is_checkpoint.eq(1)].iloc[-1]
            result = dict(arm=arm, dataset='sparkov', fit_seconds=fit_seconds,
                total_seconds=time.monotonic()-start, last_epoch=float(progress.epoch.max()),
                selected_epoch=float(best.epoch), optimizer_updates=int(progress.steps.max()),
                best_check_loss=float(best.val_loss), prediction=pred, generations=generations,
                weights_sha256=weights_before, originals_unchanged=True,
                initial=json.loads((folder/'initial.json').read_text()),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                config_sha256=digest(CONFIG), test_outcomes_accessed=False,
                files={p.name: digest(p) for p in folder.iterdir() if p.is_file()})
            write(folder / 'DONE.json', result)
            print('DONE', arm, flush=True)
    except Exception as exc:
        write(folder / 'FAILED.json', dict(type=type(exc).__name__, error=str(exc), seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('arm', choices=['A', 'G', 'R'])
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    run(args.arm, args.device)
