"""Registered cross-history and coherent-oracle diagnostic, without new fitting."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

from experiments import cs_saf_rollout_audit as parent
from experiments.cs_saf_followup import folder_for, make_model, verify_artifacts
from experiments.cs_saf_pilot import ROOT, frozen_source, state_digest
from experiments.cof_seqgen_saf_training import _seed_everything
from benchmarks import cs_saf_joint_oracle as joint
from scripts.audit_cs_saf_oracle import sha256

CONFIG = ROOT/'configs/benchmark_v2/cs_saf_replay_oracle_v1.json'
CONFIG_SHA = '65ebc689fea1df2d9c7714d248f80a02ee32761299d8079fa796d4ed39b4b3ed'
PARENT = ROOT/'artifacts/cs_saf/rollout_audit_v1'
write_json = parent.write_json


def contract():
    if sha256(CONFIG) != CONFIG_SHA: raise ValueError('registered configuration changed')
    c = json.loads(CONFIG.read_text())
    if sha256(ROOT/'docs/cs_saf/rollout_audit_v1_result.json') != c['parent_result_sha256']:
        raise ValueError('frozen parent result changed')
    return c


def verify_folder(path, *, config=None):
    complete = json.loads((path/'COMPLETE.json').read_text())
    if config and complete['config_sha256'] != config: raise ValueError('manifest config mismatch')
    for name, expected in complete['artifact_sha256'].items():
        if sha256(path/name) != expected: raise ValueError('artifact changed: '+str(path/name))
    return complete


def quantized_data(data, oracle, representatives):
    result = {k: v.clone() for k, v in data.items()}
    gap = result['gap'].numpy(); mask = np.isfinite(gap)
    gap[mask] = np.asarray(representatives, np.float32)[np.searchsorted(oracle.upper, gap[mask], side='left')]
    return result


def oracle_summary(data, probability, reference, edges):
    # In an oracle control, the online predictor IS its stated oracle. The
    # zero local-discrepancy field is definitional, not a new accuracy result.
    result = parent.summarize_predictions(data, {'repeat': probability, 'oracle_repeat': probability,
        'factual_tv': np.zeros_like(probability)}, reference, edges)
    for group in result.values():
        for key in ('local_repeat_discrepancy_L1', 'step_factual_TV', 'step_repeat_absolute_error',
                    'oracle_composition_L1', 'signed_decomposition', 'oracle_repeat', 'step_oracle_repeat'):
            group.pop(key, None)
        group['conditional_TV_evaluated'] = False
    return result


def load_models(payload, pi, kappa, trial, device):
    models, provenance = {}, {}
    for candidate in contract()['models']:
        folder = folder_for(pi, kappa, trial, candidate)
        manifest = verify_artifacts(folder)
        cp = torch.load(folder/'checkpoint_best.pt', map_location='cpu', weights_only=False)
        old = json.loads((folder/'conditional_accuracy.json').read_text())['checkpoints']['best']
        model = make_model(payload, candidate, device, trial)
        model.load_state_dict(cp['model_state'], strict=True); model.eval()
        digest = state_digest(model)
        if digest != old['state_sha256']: raise ValueError('checkpoint tensor mismatch')
        models[candidate] = model
        provenance[candidate] = {'checkpoint_sha256': sha256(folder/'checkpoint_best.pt'), 'state_sha256': digest,
                                 'reference': old['reference_probabilities'], 'input_manifest': manifest}
    return models, provenance


def job(pi, kappa, trial, device, cell, source):
    c = contract(); started = time.monotonic()
    payload, edges, panel, ids = cell
    out = ROOT/c['output']/f'pi_{pi:.2f}/kappa_{kappa}/trial_{trial}'
    if (out/'COMPLETE.json').exists():
        complete = verify_folder(out, config=CONFIG_SHA)
        if complete['source_commit'] != source: raise ValueError('resume source changed')
        return
    out.mkdir(parents=True, exist_ok=False)
    models, provenance = load_models(payload, pi, kappa, trial, device)
    oracle = parent.make_oracle(models['U'], kappa)
    representatives = models['U'].support.representatives
    validation = {k: payload['validation'][k] for k in parent.FIELDS}
    reference = parent.reference_metrics(validation, edges)
    quant_reference = parent.reference_metrics(quantized_data(validation, oracle, representatives), edges)
    replay, prediction_arrays, parent_manifests = {}, {}, {}
    max_diagonal_probability_error = 0.; max_diagonal_metric_error = 0.
    for source_candidate in c['models']:
        previous = PARENT/f'pi_{pi:.2f}/kappa_{kappa}/trial_{trial}/{source_candidate}'
        manifest = verify_folder(previous, config=parent.CONFIG_SHA)
        old = json.loads((previous/'diagnosis.json').read_text())
        samples = torch.load(previous/'panel_rollouts.pt', map_location='cpu', weights_only=False)
        if samples['panel_entity_ids'] != ids or old['panel_entity_ids'] != ids: raise ValueError('panel mismatch')
        if old['checkpoint_state_sha256'] != provenance[source_candidate]['state_sha256']:
            raise ValueError('source checkpoint and parent paths mismatch')
        parent_manifests[source_candidate] = manifest
        for mode in c['replay_modes']:
            for base in c['sampling_seeds']:
                seed = base+c['sampling_seed_trial_offset']*trial
                sample = samples['samples'][f'{mode}_{seed}']
                data = {k: sample[k] for k in parent.FIELDS}
                mask = data['valid_mask'].numpy().copy(); mask[:, 0] = False
                prior_cache = parent.oracle_arrays(oracle, data, kappa)
                for target in c['models']:
                    pred = parent.evaluate_histories(models[target], data, oracle, kappa,
                        provenance[target]['reference'], device, prior_cache=prior_cache)
                    key = f'{mode}/seed_{seed}/target_{target}/source_{source_candidate}'
                    scores = parent.summarize_predictions(data, pred, reference, edges)
                    replay[key] = scores; prediction_arrays[key] = pred['repeat']
                    if target == source_candidate:
                        probability_error = float(np.max(abs(pred['repeat'][mask]-sample['predicted_repeat'].numpy()[mask])))
                        metric_error = max(abs(scores[label]['expected_repeat_L1']-old['rollouts'][f'{mode}_{seed}'][label]['expected_repeat_L1']) for label in ('0', '1'))
                        max_diagonal_probability_error = max(max_diagonal_probability_error, probability_error)
                        max_diagonal_metric_error = max(max_diagonal_metric_error, metric_error)
                        if max(probability_error, metric_error) > c['numerical_replay_atol']:
                            raise ValueError('parent diagonal reproduction failed')
    teacher = {}
    for info in ('CONT', 'BIN', 'MODELINFO'):
        data = quantized_data(panel, oracle, representatives) if info == 'BIN' else panel
        r = joint.predict(oracle, data, kappa, info)
        teacher[info] = {'raw_reference': oracle_summary(data, r, reference, edges),
                         'matched_reference': oracle_summary(data, r, quant_reference if info == 'BIN' else reference, edges)}
    controls, paths = {}, {}
    oracle_replay_max_error = 0.
    for mode in c['oracle_modes']:
        info = 'MODELINFO' if mode == 'FIX_MODELINFO' else ('BIN' if mode.endswith('BIN') else 'CONT')
        for base in c['sampling_seeds']:
            seed = base+c['sampling_seed_trial_offset']*trial
            data, during = joint.sample(oracle, panel, kappa, representatives, mode, seed)
            after = joint.predict(oracle, data, kappa, info)
            error = float(np.max(abs(after-during))); oracle_replay_max_error = max(oracle_replay_max_error, error)
            if error > c['algebra_atol']: raise ValueError('oracle replay mismatch')
            key = f'{mode}_{seed}'
            controls[key] = {'raw_reference': oracle_summary(data, during, reference, edges),
                            'matched_reference': oracle_summary(data, during, quant_reference if info == 'BIN' else reference, edges)}
            paths[key] = {**data, 'predicted_repeat': torch.from_numpy(during)}
    for candidate, model in models.items():
        if state_digest(model) != provenance[candidate]['state_sha256']: raise ValueError('model mutated')
    np.savez_compressed(out/'replay_probabilities.npz', **prediction_arrays)
    torch.save({'panel_entity_ids': ids, 'samples': paths}, out/'oracle_samples.pt')
    write_json(out/'replay.json', {'pi': pi, 'kappa': kappa, 'trial': trial, 'source_commit': source,
        'config_sha256': CONFIG_SHA, 'provenance': provenance, 'parent_manifests': parent_manifests,
        'panel_entity_ids': ids, 'metric_edges': edges, 'reference': reference, 'quantized_reference': quant_reference,
        'matrices': replay, 'max_diagonal_probability_error': max_diagonal_probability_error,
        'max_diagonal_metric_error': max_diagonal_metric_error, 'test_accessed': False, 'new_fits': 0})
    write_json(out/'oracle.json', {'pi': pi, 'kappa': kappa, 'trial': trial, 'teacher': teacher,
        'controls': controls, 'oracle_replay_max_error': oracle_replay_max_error,
        'known_DGP_control_not_fitted_baseline': True, 'numeric_values': 'observed_independent_conditioning',
        'realized_dataset_latents_used': False, 'test_accessed': False})
    write_json(out/'COMPLETE.json', {'source_commit': source, 'config_sha256': CONFIG_SHA,
        'artifact_sha256': {p.name: sha256(p) for p in out.iterdir() if p.is_file()}, 'status': 'COMPLETE'})
    print(json.dumps({'cell_complete': f'pi_{pi:.2f}/kappa_{kappa}/trial_{trial}',
                      'seconds': time.monotonic()-started}), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--device', default='cpu')
    parser.add_argument('--cpu-gate', action='store_true'); parser.add_argument('--gpu-gate', action='store_true')
    args = parser.parse_args(); c = contract(); source = frozen_source()
    root = ROOT/c['output']; root.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.cpu_gate:
        command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_cs_saf_joint_oracle.py']
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (root/'cpu_gate.log').write_text(result.stdout+result.stderr)
        write_json(root/'cpu_gate.json', {'PASS': result.returncode == 0, 'source_commit': source,
            'config_sha256': CONFIG_SHA, 'command': command, 'log_sha256': sha256(root/'cpu_gate.log')})
        print(result.stdout+result.stderr, flush=True)
        if result.returncode: raise RuntimeError('CPU gate failed')
        return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8'); _seed_everything(2026091901)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device); torch.cuda.set_per_process_memory_fraction(.25, device)
    if args.gpu_gate:
        payload, _, panel, _ = parent.load_cell(.05, 1)
        models, provenance = load_models(payload, .05, 1, 0, device)
        data = parent.select(panel, torch.tensor([0, 1, 2, 128, 129, 130]))
        oracle = parent.make_oracle(models['U'], 1); errors = {}
        for candidate, model in models.items():
            gen, during = parent.rollout(model, data, 'OBS', c['sampling_seeds'][0], device)
            pred = parent.evaluate_histories(model, gen, oracle, 1, provenance[candidate]['reference'], device)
            mask = gen['valid_mask'].numpy().copy(); mask[:, 0] = False
            error = float(np.max(abs(during[mask]-pred['repeat'][mask])))
            if error > c['numerical_replay_atol']: raise ValueError('GPU replay gate failed')
            errors[candidate] = error
        write_json(root/'gpu_gate.json', {'PASS': True, 'source_commit': source, 'config_sha256': CONFIG_SHA, 'max_errors': errors})
        print(errors, flush=True); return
    for name in ('cpu_gate.json', 'gpu_gate.json'):
        gate = json.loads((root/name).read_text())
        if not gate['PASS'] or gate['source_commit'] != source: raise ValueError('missing/mismatched gate')
    execution = {'source_commit': source, 'config_sha256': CONFIG_SHA, 'status': 'RUNNING', 'new_fits': 0,
        'device': str(device), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'TF32': False, 'torch': torch.__version__, 'numpy': np.__version__, 'test_accessed': False}
    write_json(root/'execution.json', execution)
    for pi in c['prevalences']:
        for kappa in c['kappas']:
            cell = parent.load_cell(pi, kappa)
            for trial in c['trials']: job(pi, kappa, trial, device, cell, source)
    execution['status'] = 'COMPLETE'; write_json(root/'execution.json', execution)


if __name__ == '__main__': main()
