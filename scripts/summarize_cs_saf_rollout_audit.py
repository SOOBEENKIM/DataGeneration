"""Verify the complete registered audit before computing its descriptive screens."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.cs_saf_rollout_audit import ROOT, CONFIG_SHA, contract, write_json, sha256


def statistics(values):
    a = np.asarray(values, dtype=float)
    if a.shape != (5,) or not np.isfinite(a).all(): raise ValueError('five finite paired model trials required')
    mean = float(a.mean()); half = float(2.7764451051977987*a.std(ddof=1)/np.sqrt(5))
    return {'values': a.tolist(), 'mean': mean, 'ci95_descriptive': [mean-half, mean+half],
            'negative_trials': int((a < 0).sum()), 'positive_trials': int((a > 0).sum())}


def scalar(report, domain, label, metric):
    if domain in ('TF_all', 'TF_panel', 'TF_quantized_panel', 'native'):
        item = report[domain][str(label)]
        return item['conditional'][metric] if metric in item.get('conditional', {}) else item[metric]
    items = [v[str(label)][metric] for k, v in report['rollouts'].items() if k.startswith(domain+'_')]
    if len(items) != 3: raise ValueError('three registered sampling tapes required')
    return float(np.mean(items))


def main():
    c = contract(); root = ROOT/c['output']
    execution = json.loads((root/'execution.json').read_text())
    if execution['status'] != 'COMPLETE': raise RuntimeError('scientific audit incomplete')
    reports = {}; checksums = 0; raw = []
    for pi in c['prevalences']:
        for kappa in c['kappas']:
            panel = None
            for trial in c['trials']:
                for candidate in c['candidates']:
                    path = root/f'pi_{pi:.2f}/kappa_{kappa}/trial_{trial}/{candidate}'
                    complete = json.loads((path/'COMPLETE.json').read_text())
                    if complete['config_sha256'] != CONFIG_SHA or complete['source_commit'] != execution['source_commit']:
                        raise ValueError('mixed source/config')
                    for name, digest in complete['artifact_sha256'].items():
                        if sha256(path/name) != digest: raise ValueError('changed output '+str(path/name))
                        checksums += 1
                    report = json.loads((path/'diagnosis.json').read_text())
                    if panel is not None and report['panel_entity_ids'] != panel: raise ValueError('unpaired panel')
                    panel = report['panel_entity_ids']
                    if len(report['rollouts']) != 12 or report['parameters_changed'] or report['test_accessed']:
                        raise ValueError('invalid execution scope')
                    reports[pi, kappa, trial, candidate] = report
                    raw.append(report)
    stages = [('TF_panel', 'OBS'), ('OBS', 'QUANT'), ('QUANT', 'GAP'), ('GAP', 'FULL')]
    pairs = c['diagnostic_pairs']; contrasts = {}; screen = {}; screen_passes = []
    for a, b in pairs:
        pair = f'{a}_minus_{b}'; contrasts[pair] = {}; screen[pair] = {}
        for pi in c['prevalences']:
            contrasts[pair][f'{pi:.2f}'] = {}
            for kappa in c['kappas']:
                for label in (0, 1):
                    cell = {}; prefix = f'kappa_{kappa}_context_{label}'
                    domains = {
                        'TF_all': ['grid_mark_TV', 'grid_repeat_L1', 'grid_excess_TV', 'factual_mark_TV', 'factual_repeat_L1', 'grid_signed_repeat_bias', 'expected_repeat_L1'],
                        'native': ['empirical_repeat_L1', 'expected_repeat_L1', 'mi_error', 'sampling_residual_L1', 'local_repeat_discrepancy_L1', 'oracle_composition_L1', 'binned_history_local_discrepancy_L1'],
                    }
                    for domain in ['TF_panel', 'TF_quantized_panel', 'OBS', 'QUANT', 'GAP', 'FULL']:
                        domains[domain] = ['expected_repeat_L1', 'empirical_repeat_L1', 'mi_error', 'repeat_signed_bias']
                    for domain, metrics in domains.items():
                        cell[domain] = {}
                        for metric in metrics:
                            cell[domain][metric] = statistics([
                                scalar(reports[pi, kappa, t, a], domain, label, metric)-scalar(reports[pi, kappa, t, b], domain, label, metric)
                                for t in c['trials']])
                    cell['stages'] = {}
                    for previous, following in stages:
                        before = np.asarray(cell[previous]['expected_repeat_L1']['values'])
                        after = np.asarray(cell[following]['expected_repeat_L1']['values'])
                        cell['stages'][previous+'_to_'+following] = statistics(after-before)
                    contrasts[pair][f'{pi:.2f}'][prefix] = cell
        for metric in ('grid_repeat_L1', 'factual_mark_TV'):
            details = {}
            for pi in c['prevalences']:
                active = contrasts[pair][f'{pi:.2f}']['kappa_1_context_1']['TF_all']
                tv, bad = active['grid_mark_TV'], active[metric]
                joint = int(((np.asarray(tv['values']) < 0) & (np.asarray(bad['values']) > 0)).sum())
                ok = joint >= 4 and tv['mean'] <= -.0001 and bad['mean'] >= .0001
                details[f'{pi:.2f}'] = {'joint_trials': joint, 'qualifies': ok}
            passed = sum(v['qualifies'] for v in details.values()) >= 3
            key = 'endpoint_'+metric
            screen[pair][key] = {'prevalences': details, 'PASS': passed}
            if passed: screen_passes.append(pair+'/'+key)
        for previous, following in stages:
            name = previous+'_to_'+following
            directions = {}
            for sign in (-1, 1):
                details = {}
                for pi in c['prevalences']:
                    st = contrasts[pair][f'{pi:.2f}']['kappa_1_context_1']['stages'][name]
                    count = int((np.asarray(st['values'])*sign > 0).sum())
                    details[f'{pi:.2f}'] = {'consistent_trials': count,
                                           'qualifies': count >= 4 and st['mean']*sign >= .001}
                passed = sum(v['qualifies'] for v in details.values()) >= 3
                directions[str(sign)] = {'prevalences': details, 'PASS': passed}
                if passed: screen_passes.append(pair+'/stage_'+name+'/direction_'+str(sign))
            screen[pair]['stage_'+name] = directions
    result = {'version': c['version'], 'base_commit': c['base_commit'], 'source_commit': execution['source_commit'],
        'config_sha256': CONFIG_SHA, 'new_fits': 0, 'models': len(reports), 'rollout_sequences': len(reports)*256*12,
        'verified_output_checksums': checksums, 'execution': execution,
        'cpu_gate': json.loads((root/'cpu_gate.json').read_text()), 'technical_checks': 'PASS',
        'publication_screen': 'PASS' if screen_passes else 'FAIL', 'passing_screens': screen_passes,
        'screens': screen, 'paired_contrasts': contrasts,
        'uncertainty': 'five paired model trials; three tapes averaged within trial; descriptive intervals; no multiplicity adjustment; same DGP data',
        'raw_reports': raw}
    write_json(root/'summary.json', result)
    print(json.dumps({k: result[k] for k in ('models', 'rollout_sequences', 'verified_output_checksums', 'technical_checks', 'publication_screen', 'passing_screens')}, indent=2))


if __name__ == '__main__': main()
