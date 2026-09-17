"""Registered read-only checkpoint diagnosis; no optimizer or training path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from benchmarks.cs_saf_oracle import SemiMarkovCopyOracle
from benchmarks.temporal_coupling_v2 import BenchmarkConfig
from experiments.cs_saf_followup import CACHE, folder_for, load_cache, make_model, verify_artifacts
from experiments.cs_saf_pilot import ROOT, frozen_source, state_digest
from experiments.cs_saf_v3 import mark_tv
from experiments.cof_seqgen_saf_training import _seed_everything
from scripts.audit_cs_saf_oracle import sha256

CONFIG = ROOT / 'configs/benchmark_v2/cs_saf_rollout_audit_v1.json'
CONFIG_SHA = 'fd9661b92596a8146d9fd5229d55fab7dfe7e58d5d9b2ad9bd9c434942ab1bfa'
FIELDS = ('gap', 'receiver', 'numeric_value', 'valid_mask', 'lengths', 'codes')
SCORES = ('grid_mark_TV', 'grid_repeat_L1', 'grid_excess_TV', 'factual_mark_TV',
          'factual_repeat_L1', 'factual_excess_TV', 'grid_signed_repeat_bias')


def contract():
    if sha256(CONFIG) != CONFIG_SHA:
        raise ValueError('registered contract changed')
    return json.loads(CONFIG.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def select(data, ids):
    return {k: data[k][ids].clone() for k in FIELDS}


def quantize(model, gap):
    result = gap.clone()
    mask = torch.isfinite(gap)
    result[mask] = model.support.decode_tensor(model._support_code(gap[mask]) - 3)
    return result


def make_oracle(model, kappa):
    raw = yaml.safe_load((ROOT / 'configs/benchmark_v2/full_v2_5.yaml').read_text())
    cfg = BenchmarkConfig.from_mapping(raw, 'joint_semimarkov_v2b', kappa)
    return SemiMarkovCopyOracle(cfg, np.asarray(model.support.upper_bounds, dtype=np.float32).astype(float))


def oracle_arrays(oracle, data, kappa, *, binned_history=False):
    """Observable filter: predictions precede current-mark assimilation."""
    gap, mark, lengths = (data[k].numpy() for k in ('gap', 'receiver', 'lengths'))
    valid = data['valid_mask'].numpy().copy(); valid[:, 0] = False
    if not np.isnan(gap[:, 0]).all() or not np.isfinite(gap[valid]).all():
        raise ValueError('invalid gaps')
    if np.any(gap[valid] < 0) or np.any((mark[data['valid_mask']] < 3) | (mark[data['valid_mask']] > 66)):
        raise ValueError('invalid observations')
    active = (data['codes'].numpy() == 4) & bool(kappa)
    prior = oracle.advance(oracle.initial(len(gap)))
    priors, repeats = np.zeros_like(gap, dtype=float), np.zeros_like(gap, dtype=float)
    for t in range(1, gap.shape[1]):
        ids = np.flatnonzero(valid[:, t]); p = prior[ids].sum(-1)[:, 1]
        bins = np.searchsorted(oracle.upper, gap[ids, t], side='left')
        q = oracle.config.q_low + (oracle.config.q_high - oracle.config.q_low) * p
        ix = active[ids]
        q[ix] = oracle.copy_curve(p[ix], True)[np.arange(ix.sum()), bins[ix]]
        priors[ids, t] = p; repeats[ids, t] = q + (1-q)/64
        equal = mark[ids, t] == mark[ids, t-1]
        logw = np.log(np.where(equal[:, None], oracle.repeat_by_state, 1-oracle.repeat_by_state))
        if binned_history:
            logw[ix] += np.log(oracle.bin_mass[:, bins[ix]].T)
        else:
            logw[ix] += -np.log(oracle.scales) - gap[ids[ix], t, None] / oracle.scales
        posterior = prior[ids] * np.exp(logw-logw.max(1, keepdims=True))[:, :, None]
        posterior /= posterior.sum((1, 2), keepdims=True)
        prior[ids] = oracle.advance(posterior)
    return priors, repeats


def inverse_cdf(probabilities, uniforms):
    # Normalize like Categorical(probs=...), then pin the terminal CDF to one.
    p = probabilities.double(); p = p / p.sum(-1, keepdim=True)
    # Pinned torch 2.1.2 rejects CUDA cumsum under deterministic algorithms.
    # The short float64 CDF scan is deterministic on CPU; probabilities/tapes
    # and all registered sampling distributions remain unchanged.
    cumulative = p.cpu().cumsum(-1).to(p.device); cumulative[..., -1] = 1.
    return (uniforms.double()[..., None] >= cumulative).sum(-1).long()


def tapes(seed, shape, device):
    streams = np.random.SeedSequence(seed).spawn(3)
    values = (np.random.default_rng(streams[0]).random(shape),
              np.random.default_rng(streams[1]).random(shape),
              np.random.default_rng(streams[2]).standard_normal(shape))
    return tuple(torch.from_numpy(x).to(device) for x in values)


def initial_hidden(model, codes):
    e = model.encoder
    h = e.start[None].expand(len(codes), -1) + e.static_projection(e.static_categorical[0](codes))
    return h, h[None].repeat(e.config.num_layers, 1, 1).contiguous()


def stream_event(model, gap, mark, value, valid, state):
    """One existing GRU event update, equivalent to the unmodified shifted encoder."""
    e = model.encoder
    features = torch.stack((torch.log1p(torch.nan_to_num(gap).clamp_min(0)), torch.isfinite(gap).to(gap.dtype)), -1)
    event = torch.cat((e.history_gap(features), e.history_mark(mark.clamp(0, 66)),
                       torch.nan_to_num(value)[:, None]), -1)
    event = e.input_projection(event) * valid[:, None]
    output, state = e.gru(event[:, None], state)
    return output[:, 0], state


@torch.no_grad()
def rollout(model, panel, arm, seed, device):
    if arm not in ('OBS', 'QUANT', 'GAP', 'FULL'):
        raise ValueError('unregistered arm')
    x = {k: v.to(device) for k, v in panel.items()}
    gap, mark, value = (x[k].clone() for k in ('gap', 'receiver', 'numeric_value'))
    if arm == 'QUANT': gap = quantize(model, gap)
    gu, mu, vn = tapes(seed, gap.shape, device)
    repeat = torch.zeros_like(gap)
    hidden, state = initial_hidden(model, x['codes'])
    for t in range(1, gap.shape[1]):
        hidden, state = stream_event(model, gap[:, t-1], mark[:, t-1], value[:, t-1], x['valid_mask'][:, t-1], state)
        active = x['valid_mask'][:, t]
        if arm in ('GAP', 'FULL'):
            decoder = model.gap_decoder
            probabilities = decoder.probabilities(hidden) if hasattr(decoder, 'probabilities') else decoder.logits(hidden).softmax(-1)
            bins = inverse_cdf(probabilities, gu[:, t])
            emitted = model.support.decode_tensor(bins)
            gap[active, t] = emitted[active]
        context = model.context(hidden, (x['codes'],))
        logp, logrepeat, _ = model.mark_distribution(context, gap[:, t], mark[:, t-1], active, static_codes=x['codes'])
        emitted = inverse_cdf(logp.exp(), mu[:, t])
        mark[active, t] = emitted[active]
        repeat[active, t] = logrepeat.exp()[active]
        if arm == 'FULL':
            loc, scale = model._value_parameters(hidden, gap[:, t], emitted)
            emitted_value = loc + scale * vn[:, t].to(loc.dtype)
            value[active, t] = emitted_value[active]
    result = dict(x, gap=gap, receiver=mark, numeric_value=value)
    return {k: v.cpu() for k, v in result.items()}, repeat.cpu().numpy()


@torch.no_grad()
def evaluate_histories(model, data, oracle, kappa, reference, device, *, grid=False,
                       quantized_history=False, prior_cache=None):
    """Same float64 oracle/TV arithmetic as the published auditor."""
    priors, oracle_repeat = prior_cache or oracle_arrays(oracle, data, kappa)
    repeat = np.zeros(tuple(data['gap'].shape), dtype=float)
    factual_tv = np.zeros_like(repeat)
    entity_scores = np.zeros((len(repeat), len(SCORES)), dtype=float)
    for label in (0, 1):
        selected = torch.where(data['codes'] == label+3)[0]
        pi = torch.as_tensor(reference[label], device=device, dtype=torch.float64)
        for start in range(0, len(selected), 256):
            ids = selected[start:start+256]
            x = {k: data[k][ids].to(device) for k in FIELDS}
            history_gap = quantize(model, x['gap']) if quantized_history else x['gap']
            hidden = model.encoder(history_gap, x['receiver'], x['numeric_value'], x['valid_mask'], static_categorical=(x['codes'],))
            context = model.context(hidden, (x['codes'],))
            mask = x['valid_mask'].clone(); mask[:, 0] = False
            rows, cols = torch.where(mask)
            previous = x['receiver'].roll(1, 1)[mask]
            bins = model._support_code(x['gap'][mask])-3
            codes = x['codes'][:, None].expand_as(mask)[mask]
            flat = context[mask]
            oq = torch.as_tensor(oracle.copy_curve(priors[ids][:][mask.cpu().numpy()], bool(kappa*label)), device=device, dtype=torch.float64)
            local_scores = []
            for j in range(0, len(flat), 512):
                sl = slice(j, j+512); c, prev = flat[sl], previous[sl]
                q, _ = model.response_curves(c, prev, static_codes=codes[sl]); q = q.double()
                fresh = model.new_mark_head(c).clone(); fresh[:, :3] = -torch.inf
                fresh = fresh.softmax(-1)[:, 3:].double()
                r = q + (1-q) * fresh.gather(1, (prev-3)[:, None])
                op = oq[sl] + (1-oq[sl])/64
                b = bins[sl, None]
                tv = mark_tv(q, fresh, prev-3, oq[sl]) if grid else mark_tv(q.gather(1, b), fresh, prev-3, oq[sl].gather(1, b))
                ractual = r.gather(1, b)[:, 0]
                tactual = tv.gather(1, b)[:, 0] if grid else tv[:, 0]
                aactual = (r-op).abs().gather(1, b)[:, 0]
                if (tactual-aactual).min() < -2e-6:
                    raise AssertionError('TV contraction failed')
                global_rows = ids[rows[sl].cpu()].numpy(); columns = cols[sl].cpu().numpy()
                repeat[global_rows, columns] = ractual.cpu().numpy()
                factual_tv[global_rows, columns] = tactual.cpu().numpy()
                if grid:
                    grid_tv, grid_abs = (tv*pi).sum(1), ((r-op).abs()*pi).sum(1)
                    local_scores.append(torch.stack((grid_tv, grid_abs, grid_tv-grid_abs, tactual,
                                                     aactual, tactual-aactual, ((r-op)*pi).sum(1)), 1))
            if grid:
                vals = torch.cat(local_scores)
                entity = torch.zeros((len(ids), len(SCORES)), device=device, dtype=torch.float64)
                entity.index_add_(0, rows, vals/mask.sum(1)[rows, None])
                entity_scores[ids.numpy()] = entity.cpu().numpy()
    result = {'repeat': repeat, 'oracle_repeat': oracle_repeat, 'factual_tv': factual_tv}
    if grid: result['entity_scores'] = entity_scores
    return result


def transition_arrays(data, label):
    mask = data['valid_mask'].numpy().copy(); mask[:, 0] = False
    mask &= (data['codes'].numpy() == label+3)[:, None]
    rows, steps = np.where(mask)
    marks = data['receiver'].numpy()
    return rows, steps, data['gap'].numpy()[mask], (marks[rows, steps] == marks[rows, steps-1]).astype(float)


def bin_summary(gaps, values, edges):
    bins = np.searchsorted(edges, gaps, side='right')
    counts = np.bincount(bins, minlength=5)
    means = np.bincount(bins, weights=values, minlength=5)/counts.clip(min=1)
    return counts, means


def mutual_information(counts, means):
    joint = counts[:, None]*np.stack((1-means, means), -1)
    joint = joint/joint.sum()
    expected = joint.sum(1, keepdims=True)*joint.sum(0, keepdims=True)
    mask = joint > 0
    return float(np.sum(joint[mask]*np.log(joint[mask]/expected[mask])))


def reference_metrics(data, edges):
    result = {}
    for label in (0, 1):
        _, _, gaps, observed = transition_arrays(data, label)
        counts, means = bin_summary(gaps, observed, edges[str(label)])
        result[str(label)] = {'counts': counts.tolist(), 'observed_repeat': means.tolist(),
                              'mi': mutual_information(counts, means)}
    return result


def summarize_predictions(data, pred, reference, edges, *, binned_oracle=None):
    result = {}
    for label in (0, 1):
        key = str(label); rows, steps, gaps, observed = transition_arrays(data, label)
        counts, means = bin_summary(gaps, observed, edges[key])
        _, pm = bin_summary(gaps, pred['repeat'][rows, steps], edges[key])
        _, po = bin_summary(gaps, pred['oracle_repeat'][rows, steps], edges[key])
        target = np.array(reference[key]['observed_repeat'])
        weight = np.array(reference[key]['counts'], float); weight /= weight.sum()
        terms = np.stack((means-pm, pm-po, po-target))
        np.testing.assert_allclose(terms.sum(0), means-target, atol=1e-12, rtol=0)
        if np.any(counts == 0): raise ValueError('empty diagnostic gap bin')
        def step_mean(values):
            n = np.bincount(steps, minlength=32)
            avg = np.bincount(steps, weights=values, minlength=32)/n.clip(min=1)
            return [None if count == 0 else float(v) for count, v in zip(n, avg)]
        result[key] = {'entities': int((data['codes'] == label+3).sum()), 'counts': counts.tolist(),
            'observed_repeat': means.tolist(), 'model_repeat': pm.tolist(), 'oracle_repeat': po.tolist(),
            'empirical_repeat_L1': float(weight@abs(means-target)),
            'expected_repeat_L1': float(weight@abs(pm-target)),
            'local_repeat_discrepancy_L1': float(weight@abs(pm-po)),
            'oracle_composition_L1': float(weight@abs(po-target)),
            'sampling_residual_L1': float(weight@abs(means-pm)),
            'signed_decomposition': terms.tolist(), 'repeat_signed_bias': float(weight@(pm-target)),
            'mi': mutual_information(counts, means),
            'mi_error': abs(mutual_information(counts, means)-reference[key]['mi']),
            'step_counts': np.bincount(steps, minlength=32).tolist(),
            'step_model_repeat': step_mean(pred['repeat'][rows, steps]),
            'step_observed_repeat': step_mean(observed),
            'step_oracle_repeat': step_mean(pred['oracle_repeat'][rows, steps]),
            'step_factual_TV': step_mean(pred['factual_tv'][rows, steps]),
            'step_repeat_absolute_error': step_mean(abs(pred['repeat'][rows, steps]-pred['oracle_repeat'][rows, steps]))}
        if binned_oracle is not None:
            _, pb = bin_summary(gaps, binned_oracle[rows, steps], edges[key])
            result[key]['binned_history_oracle_repeat'] = pb.tolist()
            result[key]['binned_history_local_discrepancy_L1'] = float(weight@abs(pm-pb))
        if 'entity_scores' in pred:
            result[key]['conditional'] = dict(zip(SCORES, pred['entity_scores'][data['codes'].numpy() == label+3].mean(0).tolist()))
    return result


def load_cell(pi, kappa):
    payload = load_cache(CACHE / f'pi_{pi:.2f}_kappa_{kappa}.pt')
    # Reuse and verify the exact published train-fitted metric bin boundaries.
    folder = ROOT / f'artifacts/cs_saf/followup_v1/generation/pi_{pi:.2f}/trial_0/kappa_{kappa}/U'
    verify_artifacts(folder)
    old = json.loads((folder/'comparison.json').read_text())
    edges = {str(s): old['metrics'][f'context_{s}']['train_metric_state']['gap_bin_edges'] for s in (0, 1)}
    for s in (0, 1):
        train = payload['train']; mask = train['valid_mask'].numpy().copy(); mask[:, 0] = False
        mask &= (train['codes'].numpy() == s+3)[:, None]
        np.testing.assert_allclose(np.quantile(train['gap'].numpy()[mask], [.2, .4, .6, .8]), edges[str(s)], atol=1e-7, rtol=0)
    rng = np.random.default_rng(contract()['panel_seed'])
    ids = np.concatenate([rng.choice(np.flatnonzero(payload['validation']['codes'].numpy() == s+3),
                         contract()['panel_entities_per_context'], replace=False) for s in (0, 1)])
    panel = select(payload['validation'], torch.from_numpy(ids))
    panel_ids = [payload['validation']['entity_ids'][i] for i in ids]
    return payload, edges, panel, panel_ids


def job(pi, kappa, trial, candidate, device, cell, source):
    started = time.monotonic()
    payload, edges, panel, panel_ids = cell
    out = ROOT / contract()['output'] / f'pi_{pi:.2f}/kappa_{kappa}/trial_{trial}/{candidate}'
    if (out/'COMPLETE.json').exists():
        saved = json.loads((out/'COMPLETE.json').read_text())
        if saved['source_commit'] != source: raise ValueError('resume source changed')
        for name, digest in saved['artifact_sha256'].items():
            if sha256(out/name) != digest: raise ValueError('resume artifact changed')
        return
    out.mkdir(parents=True, exist_ok=False)
    original = folder_for(pi, kappa, trial, candidate); verified = verify_artifacts(original)
    old = json.loads((original/'conditional_accuracy.json').read_text())['checkpoints']['best']
    checkpoint = torch.load(original/'checkpoint_best.pt', map_location='cpu', weights_only=False)
    model = make_model(payload, candidate, device, trial)
    model.load_state_dict(checkpoint['model_state'], strict=True); model.eval()
    digest = state_digest(model)
    if digest != old['state_sha256']: raise ValueError('checkpoint tensor identity changed')
    oracle = make_oracle(model, kappa)
    reference = old['reference_probabilities']
    val = {k: payload['validation'][k] for k in FIELDS}
    reference_curves = reference_metrics(val, edges)
    pred = evaluate_histories(model, val, oracle, kappa, reference, device, grid=True)
    tf = summarize_predictions(val, pred, reference_curves, edges)
    for label in ('0', '1'):
        for metric in ('grid_mark_TV', 'grid_repeat_L1', 'factual_mark_TV'):
            target = old['splits']['validation']['groups'][label]['metrics'][metric]['mean']
            np.testing.assert_allclose(tf[label]['conditional'][metric], target, atol=2e-6, rtol=0)
    np.savez_compressed(out/'validation_scores.npz', scores=pred['entity_scores'],
                        entity_ids=np.asarray(payload['validation']['entity_ids']), codes=val['codes'].numpy())
    panel_pred = evaluate_histories(model, panel, oracle, kappa, reference, device, grid=True)
    panel_tf = summarize_predictions(panel, panel_pred, reference_curves, edges)
    quant_pred = evaluate_histories(model, panel, oracle, kappa, reference, device, grid=True, quantized_history=True)
    panel_quant = summarize_predictions(panel, quant_pred, reference_curves, edges)
    saved = torch.load(original/'generated_sample.pt', map_location='cpu', weights_only=False)
    if saved['model_state_sha256'] != digest: raise ValueError('sample/checkpoint mismatch')
    sample = saved['sample']; native = {k: sample['static_codes' if k == 'codes' else k] for k in FIELDS}
    gen_pred = evaluate_histories(model, native, oracle, kappa, reference, device)
    binned = oracle_arrays(oracle, native, kappa, binned_history=True)[1]
    native_summary = summarize_predictions(native, gen_pred, reference_curves, edges, binned_oracle=binned)
    published_folder = ROOT/f'artifacts/cs_saf/followup_v1/generation/pi_{pi:.2f}/trial_{trial}/kappa_{kappa}/{candidate}'
    verify_artifacts(published_folder)
    published = json.loads((published_folder/'comparison.json').read_text())
    for label in ('0', '1'):
        expected = published['metrics']['context_'+label]['metrics']
        for ours, theirs in (('empirical_repeat_L1', 'short_gap_repeat_curve_l1'), ('mi_error', 'gap_repeat_mi_error')):
            np.testing.assert_allclose(native_summary[label][ours], expected[theirs], atol=1e-7, rtol=0)
    np.savez_compressed(out/'native_probability_audit.npz', model_repeat=gen_pred['repeat'],
                        oracle_repeat=gen_pred['oracle_repeat'], binned_history_oracle_repeat=binned)
    arms = {}; raw_samples = {}
    for base_seed in contract()['sampling_seeds']:
        seed = base_seed + contract()['sampling_seed_trial_offset']*trial
        for arm in contract()['rollout_arms']:
            generated, during = rollout(model, panel, arm, seed, device)
            after = evaluate_histories(model, generated, oracle, kappa, reference, device)
            mask = generated['valid_mask'].numpy().copy(); mask[:, 0] = False
            np.testing.assert_allclose(during[mask], after['repeat'][mask], atol=2e-6, rtol=0)
            if not np.isfinite(generated['numeric_value'][generated['valid_mask']]).all(): raise ValueError('nonfinite numeric output')
            bins = oracle_arrays(oracle, generated, kappa, binned_history=True)[1]
            arms[f'{arm}_{seed}'] = summarize_predictions(generated, after, reference_curves, edges, binned_oracle=bins)
            raw_samples[f'{arm}_{seed}'] = {**generated, 'predicted_repeat': torch.from_numpy(during)}
    torch.save({'panel_entity_ids': panel_ids, 'samples': raw_samples}, out/'panel_rollouts.pt')
    if state_digest(model) != digest or sha256(original/'checkpoint_best.pt') != old['checkpoint_sha256']:
        raise ValueError('read-only audit mutated input')
    write_json(out/'diagnosis.json', {'pi': pi, 'kappa': kappa, 'trial': trial, 'candidate': candidate,
        'source_commit': source, 'config_sha256': CONFIG_SHA, 'checkpoint_sha256': old['checkpoint_sha256'],
        'checkpoint_state_sha256': digest, 'input_artifact_sha256': verified['artifact_sha256'],
        'cache_sha256': sha256(CACHE/f'pi_{pi:.2f}_kappa_{kappa}.pt'), 'panel_entity_ids': panel_ids,
        'metric_edges': edges, 'reference_curves': reference_curves, 'TF_all': tf, 'TF_panel': panel_tf,
        'TF_quantized_panel': panel_quant, 'native': native_summary, 'rollouts': arms,
        'seconds': time.monotonic()-started, 'test_accessed': False, 'parameters_changed': False})
    write_json(out/'COMPLETE.json', {'source_commit': source, 'config_sha256': CONFIG_SHA,
        'status': 'COMPLETE', 'artifact_sha256': {p.name: sha256(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'complete': str(out.relative_to(ROOT)), 'seconds': time.monotonic()-started}), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--device', default='cpu')
    parser.add_argument('--cpu-gate', action='store_true')
    parser.add_argument('--gpu-gate', action='store_true')
    args = parser.parse_args(); c = contract(); source = frozen_source()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    out = ROOT/c['output']; out.mkdir(parents=True, exist_ok=True)
    if args.cpu_gate:
        command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_cs_saf_rollout_audit.py']
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (out/'cpu_gate.log').write_text(run.stdout+run.stderr)
        write_json(out/'cpu_gate.json', {'PASS': run.returncode == 0, 'source_commit': source,
            'config_sha256': CONFIG_SHA, 'command': command, 'log_sha256': sha256(out/'cpu_gate.log'),
            'test_accessed': False, 'scientific_outcomes_compared': False})
        print(run.stdout+run.stderr, flush=True)
        if run.returncode: raise RuntimeError('CPU gate failed')
        return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    _seed_everything(c['panel_seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device); torch.cuda.set_per_process_memory_fraction(.25, device)
    if args.gpu_gate:
        if device.type != 'cuda': raise ValueError('GPU gate requires CUDA')
        payload, _, panel, _ = load_cell(.05, 1)
        small = select(panel, torch.tensor([0, 1, 2, 128, 129, 130]))
        errors = {}
        for candidate in c['candidates']:
            model = make_model(payload, candidate, device, 0)
            cp = torch.load(folder_for(.05, 1, 0, candidate)/'checkpoint_best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(cp['model_state']); model.eval(); before = state_digest(model)
            oracle = make_oracle(model, 1)
            reference = np.full((2, len(model.support.representatives)), 1/len(model.support.representatives))
            for arm in c['rollout_arms']:
                gen, during = rollout(model, small, arm, c['sampling_seeds'][0], device)
                after = evaluate_histories(model, gen, oracle, 1, reference, device)
                mask = gen['valid_mask'].numpy().copy(); mask[:, 0] = False
                error = float(np.max(np.abs(during[mask]-after['repeat'][mask])))
                if error > c['numerical_identity_atol']: raise ValueError('GPU probability replay mismatch')
                errors[candidate+'/'+arm] = error
            if state_digest(model) != before: raise ValueError('GPU gate mutated model')
        write_json(out/'gpu_gate.json', {'PASS': True, 'source_commit': source, 'config_sha256': CONFIG_SHA,
            'functional_sequences_per_arm': 6, 'TF32': False, 'max_probability_errors': errors})
        print(json.dumps(errors), flush=True)
        return
    if not (out/'cpu_gate.json').exists() or not json.loads((out/'cpu_gate.json').read_text())['PASS']:
        raise RuntimeError('CPU gate must pass before scientific execution')
    gate = json.loads((out/'cpu_gate.json').read_text())
    if gate['source_commit'] != source: raise RuntimeError('CPU gate source mismatch')
    if device.type == 'cuda':
        gate = json.loads((out/'gpu_gate.json').read_text())
        if not gate['PASS'] or gate['source_commit'] != source: raise RuntimeError('GPU gate source mismatch')
    write_json(out/'execution.json', {'source_commit': source, 'config_sha256': CONFIG_SHA,
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
        'device': str(device), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'TF32': False, 'status': 'RUNNING', 'new_fits': 0, 'test_accessed': False})
    for pi in c['prevalences']:
        for kappa in c['kappas']:
            cell = load_cell(pi, kappa)
            for trial in c['trials']:
                for candidate in c['candidates']:
                    job(pi, kappa, trial, candidate, device, cell, source)
    state = json.loads((out/'execution.json').read_text()); state['status'] = 'COMPLETE'
    write_json(out/'execution.json', state)


if __name__ == '__main__':
    main()
