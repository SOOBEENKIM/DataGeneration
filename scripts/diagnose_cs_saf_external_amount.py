"""Frozen D tail probabilities and paired prefix substitutions; no fitting/sampling."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import lognorm
import torch
from data.cof_seqgen_saf_tensorizer import SAFTensorizerState
from data.cs_saf_external import TargetWindows
from scripts.run_cs_saf_external_controls import build_model, digest, write, OUT as CONTROLS, OLD
from scripts.run_cs_saf_external_port import inputs

CONFIG = ROOT / 'configs/cs_saf_external_amount_diagnostic_v1.json'
OUT = ROOT / 'artifacts/cs_saf/external_amount_diagnostic_v1'
DOC = ROOT / 'docs/cs_saf/external_amount_diagnostic_v1'


def survival(parameters, thresholds):
    """Unconditional mass above positive raw-amount thresholds, by component."""
    z = (parameters['mu'][:, :, None] - np.log(thresholds)[None, None, :]) / parameters['sigma'][:, :, None]
    component = (1 - parameters['p0'][:, None, None]) * parameters['weight'][:, :, None] * ndtr(z)
    return component.sum(1), component


def decompose(rr, gr, rg, gg):
    history = .5 * ((gr - rr) + (gg - rg))
    current = .5 * ((rg - rr) + (gg - gr))
    return dict(history=history, current=current, interaction=gg-gr-rg+rr, total=gg-rr)


def hidden(model, batch):
    h = model.encoder(**{k: v for k, v in batch.items() if k != 'target_position'})
    row = torch.arange(len(h), device=h.device)
    return h[row, batch['target_position']]


def mixed_history(real, generated, generated_amount):
    # All other fields come from the opposite source; current fields cannot
    # enter a strictly shifted encoder. The query gap/mark is passed separately.
    other, amount = (real, generated) if generated_amount else (generated, real)
    return {**other, 'numeric_value': amount['numeric_value']}


def parameters(model, h, query):
    pos = query['target_position']; row = torch.arange(len(pos), device=pos.device)
    p = model.amount_parameters(h, query['gap'][row, pos], query['receiver'][row, pos])
    weight, loc, sd, zero = p
    s = model.amount_state
    return dict(weight=weight.double().exp().cpu().numpy(),
                mu=loc.double().cpu().numpy() * s['log_scale'] + s['log_mean'],
                sigma=sd.double().cpu().numpy() * s['log_scale'],
                p0=(zero.double().sigmoid().cpu().numpy() if s['zero_rate'] else np.zeros(len(zero))),
                pred_log1p=(model.amount_point(p).double().cpu().numpy() * s['codec_scale'] + s['codec_mean']))


def paired_parameters(model, real, generated):
    hr, hg = hidden(model, real), hidden(model, generated)
    ha = hidden(model, mixed_history(real, generated, True))
    ho = hidden(model, mixed_history(real, generated, False))
    return dict(RR=parameters(model, hr, real), GR=parameters(model, hg, real),
                RG=parameters(model, hr, generated), GG=parameters(model, hg, generated),
                AG=parameters(model, ha, generated), OG=parameters(model, ho, generated))


def align_raw(frame, sequences):
    groups = {eid: g.sort_values('event_index') for eid, g in frame.groupby('entity_id', sort=False)}
    arrays = []
    for seq in sequences:
        g = groups[seq.entity_id]
        np.testing.assert_array_equal(g.event_index, np.arange(seq.length))
        assert len(g) == seq.length
        arrays.append(g.amount_or_numeric_value.to_numpy(float))
    return np.concatenate(arrays)


def generated_sequences(frame, plan, real, state):
    groups = {eid: g.sort_values('event_index') for eid, g in frame.groupby('entity_id', sort=False)}
    assert set(groups) == set(plan.entity_id)
    codec = state.event_numeric_codecs[0][1]
    # Learned strings and reserved codes must survive the saved parquet roundtrip.
    for c in (state.receiver_codec, *(v for _, v in state.auxiliary_categorical_codecs)):
        codes = np.arange(1, c.vocab_size)
        np.testing.assert_array_equal(c.encode(c.decode(codes)), codes)
    result, amounts = [], []
    for row, seq in zip(plan.itertuples(index=False), real):
        g = groups[row.entity_id]
        np.testing.assert_array_equal(g.event_index, np.arange(seq.length))
        assert len(g) == row.length == seq.length
        a = g.amount_or_numeric_value.to_numpy(float)
        assert np.isfinite(a).all() and (a >= 0).all()
        result.append(replace(seq, entity_id=row.entity_id, gap=g.gap.to_numpy(np.float32),
            receiver=state.receiver_codec.encode(g.receiver_or_mark), numeric_value=codec.encode(a),
            auxiliary_categorical=tuple(c.encode(g[name]) for name, c in state.auxiliary_categorical_codecs)))
        amounts.append(a)
    return result, np.concatenate(amounts)


def boundary_audit(model, windows):
    starts = np.cumsum(np.r_[0, windows.lengths[:-1]])
    # First, context boundary, beyond boundary, and last positions of 8 entities.
    selected = np.unique(np.r_[np.arange(min(4, len(starts))),
                                np.arange(max(0, len(starts)-4), len(starts))])
    targets = np.unique(np.concatenate([starts[i] + np.minimum([0, 1, 31, 32, windows.lengths[i]-1],
                                                               windows.lengths[i]-1) for i in selected]))
    batch = windows.batch(targets)
    actual = hidden(model, batch)
    error = 0.
    for j in range(len(targets)):
        n = int(batch['target_position'][j]) + 1
        # Reconstruct an unpadded single-prefix call with generation placeholders.
        one = {}
        for k, v in batch.items():
            if k == 'target_position':
                continue
            if isinstance(v, tuple):
                one[k] = tuple(x[j:j+1, :n].clone() if x.ndim >= 2 else x[j:j+1].clone() for x in v)
            else:
                one[k] = v[j:j+1, :n].clone() if v.ndim >= 2 and k != 'static' else v[j:j+1].clone()
        one['receiver'][:, -1] = 1
        one['numeric_value'][:, -1] = 0.
        one['gap'][:, -1] = float('nan')
        for v in one['auxiliary_categorical']:
            v[:, -1] = 1
        expected = model.encoder(**one)[:, -1]
        error = max(error, float((expected - actual[j:j+1]).abs().max()))
    assert error < 2e-5, error
    return dict(targets=len(targets), max_hidden_error=error)


def save_case(folder, label, params, amounts, positions, entities, marks, thresholds):
    p, comp = survival(params, thresholds)
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1.000001
    reference = sum((1-params['p0'][:, None]) * params['weight'][:, i, None] *
                    lognorm.sf(thresholds[None, :], s=params['sigma'][:, i, None],
                               scale=np.exp(params['mu'][:, i, None])) for i in range(3))
    maxerr = float(abs(p-reference).max())
    assert maxerr < 1e-12, maxerr
    np.savez_compressed(folder/f'{label}.npz', **params, probability=p,
                        amount=amounts, position=positions, entity=entities, mark=marks)
    return dict(events=len(amounts), max_survival_error=maxerr,
                component_tail_mean=comp.mean(0).tolist())


@torch.no_grad()
def run(name, device, cfg, folder):
    inp, _, _, frames, plan, _ = inputs(name)
    data = torch.load(inp/'prepared.pt', map_location='cpu')
    state = SAFTensorizerState.from_dict(data['state'])
    source = CONTROLS/'runs'/name/cfg['model']
    done = json.loads((source/'DONE.json').read_text())
    parent_start = json.loads((source/'START.json').read_text())
    for f, expected in parent_start['scientific_source_sha256'].items():
        assert digest(ROOT/f) == expected, f
    paths = [source/'best.pt', source/'amount_state.json', inp/'prepared.pt', inp/'plan.parquet']
    paths += [source/f'generated_raw_{s}.parquet' for s in cfg['saved_generation_seeds']]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    assert digest(source/'best.pt') == done['best_sha256']
    for record in done['results']['raw']['generations']:
        assert digest(source/f"generated_raw_{record['seed']}.parquet") == record['sha256']
    ast = json.loads((source/'amount_state.json').read_text())
    model, _ = build_model(cfg['model'], state, ast, {})
    model.load_state_dict(torch.load(source/'best.pt', map_location='cpu')['state_dict'])
    model.to(device).eval(); model.calibration = None
    for p in model.parameters():
        p.requires_grad_(False)
    fit_amount = frames['fit'].amount_or_numeric_value.to_numpy(float)
    thresholds = np.r_[np.quantile(fit_amount, [.99, .999]), fit_amount.max(), 10*fit_amount.max()]
    result = dict(dataset=name, thresholds=dict(zip(cfg['thresholds'], thresholds)), files=hashes,
                  audits={}, cases={}, actual_fit_exceedance=((fit_amount[:, None] > thresholds).mean(0)).tolist())
    val = TargetWindows(data['sequences']['validation'], device=device)
    result['audits']['validation_boundary'] = boundary_audit(model, val)
    accum = []
    for start in range(0, len(val), cfg['batch_size']):
        batch = val.batch(np.arange(start, min(start+cfg['batch_size'], len(val))))
        accum.append(parameters(model, hidden(model, batch), batch))
    params = {k: np.concatenate([a[k] for a in accum]) for k in accum[0]}
    actual = align_raw(frames['validation'], data['sequences']['validation'])
    positions = np.concatenate([np.arange(n) for n in val.lengths])
    entities = np.repeat(np.arange(len(val.lengths)), val.lengths)
    result['cases']['validation'] = save_case(folder, 'validation', params, actual, positions, entities,
        val.flat['receiver'].cpu().numpy(), thresholds)
    log_mae = float(np.abs(params['pred_log1p'] - np.log1p(actual)).mean())
    previous = done['results']['raw']['prediction']['amount_log_mae']
    assert abs(log_mae-previous) < 2e-6, (log_mae, previous)
    result['audits']['validation_log_mae'] = dict(recomputed=log_mae, previous=previous, difference=log_mae-previous)
    del accum, params, val
    print(name, 'validation complete', flush=True)
    real = [data['sequences']['fit'][int(i)] for i in plan.fit_position]
    for row, seq in zip(plan.itertuples(index=False), real):
        assert row.source_entity_id == seq.entity_id and row.length == seq.length
    result['matched_plan'] = dict(plans=len(plan), unique_source_entities=int(plan.source_entity_id.nunique()),
                                   source_is_training_fit=True, events=int(plan.length.sum()))
    rw = TargetWindows(real, device=device)
    result['audits']['matched_real_boundary'] = boundary_audit(model, rw)
    actual = align_raw(frames['fit'], real)
    positions = np.concatenate([np.arange(n) for n in rw.lengths])
    entities = np.repeat(np.arange(len(rw.lengths)), rw.lengths)
    for seed in cfg['saved_generation_seeds']:
        frame = pd.read_parquet(source/f'generated_raw_{seed}.parquet')
        seq, sampled_amount = generated_sequences(frame, plan, real, state)
        gw = TargetWindows(seq, device=device)
        result['audits'][f'generated_{seed}_boundary'] = boundary_audit(model, gw)
        assert torch.equal(rw.static, gw.static)
        assert all(torch.equal(a,b) for a,b in zip(rw.static_cat, gw.static_cat))
        accum = {case: [] for case in cfg['cases']}
        for start in range(0, len(rw), cfg['batch_size']):
            ix = np.arange(start, min(start+cfg['batch_size'], len(rw)))
            outputs = paired_parameters(model, rw.batch(ix), gw.batch(ix))
            for case, output in outputs.items():
                accum[case].append(output)
        for case, arrays in accum.items():
            params = {k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
            query_real = case[-1] == 'R'
            amounts = actual if query_real else sampled_amount
            marks = (rw if query_real else gw).flat['receiver'].cpu().numpy()
            label = f'{seed}_{case}'
            result['cases'][label] = save_case(folder, label, params, amounts, positions, entities, marks, thresholds)
        del accum, params, gw
        print(name, seed, 'six paired cases complete', flush=True)
    for p, h in hashes.items():
        assert digest(ROOT/p) == h
    result['unchanged_inputs_verified'] = True
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('dataset'); p.add_argument('--device', default='cpu')
    a = p.parse_args(); cfg = json.loads(CONFIG.read_text()); assert a.dataset in cfg['datasets']
    source_files = ['scripts/diagnose_cs_saf_external_amount.py', 'configs/cs_saf_external_amount_diagnostic_v1.json',
                    'docs/cs_saf/external_amount_diagnostic_v1/preregistration.md']
    subprocess.check_call(['git','ls-files','--error-unmatch',*source_files], cwd=ROOT, stdout=subprocess.DEVNULL)
    assert not subprocess.check_output(['git','diff','HEAD','--',*source_files], cwd=ROOT)
    folder = OUT/a.dataset
    assert not folder.exists(), 'never overwrite a scientific diagnostic run'
    folder.mkdir(parents=True)
    torch.set_num_threads(1)
    if a.device == 'cuda':
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    begun = time.monotonic()
    start = dict(config=cfg, source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                 source_sha256={f:digest(ROOT/f) for f in source_files}, device=a.device,
                 physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), neural_fits=0, new_rollouts=0)
    write(folder/'START.json', start)
    try:
        result = run(a.dataset, a.device, cfg, folder)
        result.update(start=start, elapsed_seconds=time.monotonic()-begun,
                      array_sha256={f.name:digest(f) for f in sorted(folder.glob('*.npz'))})
        write(folder/'DONE.json', result)
        write(DOC/f'{a.dataset}_audit.json', result)
        print(a.dataset, 'DONE', result['elapsed_seconds'], flush=True)
    except Exception as e:
        write(folder/'FAILED.json', dict(error_type=type(e).__name__, message=str(e), elapsed_seconds=time.monotonic()-begun))
        raise


if __name__ == '__main__':
    main()
