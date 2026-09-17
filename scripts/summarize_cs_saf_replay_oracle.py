"""Complete cross-history/joint-oracle aggregation; all directions retained."""
from pathlib import Path
import copy
import hashlib
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.cs_saf_replay_oracle import ROOT, CONFIG_SHA, contract, verify_folder, write_json
from experiments.cs_saf_followup import folder_for
from scripts.audit_cs_saf_oracle import sha256


def stats(values):
    v = np.asarray(values, float)
    if v.shape != (5,) or not np.isfinite(v).all(): raise ValueError('five finite paired trials required')
    mean = float(v.mean()); half = float(2.7764451051977987*v.std(ddof=1)/np.sqrt(5))
    return {'values': v.tolist(), 'mean': mean, 'ci95_descriptive': [mean-half, mean+half],
            'negative_trials': int((v < 0).sum()), 'positive_trials': int((v > 0).sum())}


def matrix_value(report, mode, target, source, label, metric='expected_repeat_L1'):
    vals = [v[str(label)][metric] for key, v in report['matrices'].items()
            if key.startswith(mode+'/') and key.endswith(f'/target_{target}/source_{source}')]
    if len(vals) != 3: raise ValueError('missing matrix tape')
    return float(np.mean(vals))


def oracle_value(report, mode, label, *, reference='matched_reference', metric='expected_repeat_L1'):
    vals = [v[reference][str(label)][metric] for key, v in report['controls'].items() if key.startswith(mode+'_')]
    if len(vals) != 3: raise ValueError('missing oracle tape')
    return float(np.mean(vals))


def screen(by_prevalence, c):
    output = {}
    for direction in (-1, 1):
        eligible = [pi for pi, item in by_prevalence.items()
            if item['mean']*direction >= c['descriptive_materiality'] and
            sum(v*direction > 0 for v in item['values']) >= c['descriptive_consistent_trials']]
        output[str(direction)] = {'eligible_prevalences': eligible, 'consistently_material': len(eligible) >= c['descriptive_consistent_prevalences']}
    return output


def main():
    c = contract(); root = ROOT/c['output']
    execution = json.loads((root/'execution.json').read_text())
    if execution['status'] != 'COMPLETE': raise ValueError('registered experiment incomplete')
    gates = {name: json.loads((root/(name+'.json')).read_text()) for name in ('cpu_gate', 'gpu_gate')}
    for gate in gates.values():
        if not gate['PASS'] or gate['source_commit'] != execution['source_commit'] or gate['config_sha256'] != CONFIG_SHA:
            raise ValueError('mismatched technical gate')
    if sha256(root/'cpu_gate.log') != gates['cpu_gate']['log_sha256']: raise ValueError('changed gate log')
    replays, oracles, manifests = {}, {}, {}; checked = 0; parent_checked = 0
    for pi in c['prevalences']:
        for k in c['kappas']:
            for t in c['trials']:
                path = root/f'pi_{pi:.2f}/kappa_{k}/trial_{t}'
                manifest = verify_folder(path, config=CONFIG_SHA)
                if manifest['source_commit'] != execution['source_commit']: raise ValueError('mixed source')
                checked += len(manifest['artifact_sha256']); manifests[f'{pi:.2f}/{k}/{t}'] = manifest
                report = json.loads((path/'replay.json').read_text())
                replays[pi,k,t] = report; oracles[pi,k,t] = json.loads((path/'oracle.json').read_text())
                if len(report['matrices']) != 54: raise ValueError('incomplete replay matrix')
                if len(oracles[pi,k,t]['controls']) != 15: raise ValueError('incomplete oracle controls')
                for name, previous in report['parent_manifests'].items():
                    parent_path = ROOT/f'artifacts/cs_saf/rollout_audit_v1/pi_{pi:.2f}/kappa_{k}/trial_{t}/{name}'
                    verify_folder(parent_path)
                    parent_checked += len(previous['artifact_sha256'])
    replay_summary = {}; oracle_summary = {}; identities = []; screens = {}
    for pi in c['prevalences']:
        pi_key = f'{pi:.2f}'; replay_summary[pi_key] = {}; oracle_summary[pi_key] = {}
        for k in c['kappas']:
            for label in (0,1):
                cell = f'kappa_{k}_context_{label}'; rep = {}; oracle_result = {}
                for mode in c['replay_modes']:
                    matrices = {a: {b: stats([matrix_value(replays[pi,k,t], mode, a, b, label) for t in c['trials']])
                        for b in c['models']} for a in c['models']}
                    effects = {}
                    for a,b in c['pairs']:
                        aa = np.array(matrices[a][a]['values']); ab = np.array(matrices[a][b]['values'])
                        ba = np.array(matrices[b][a]['values']); bb = np.array(matrices[b][b]['values'])
                        predictor = ((aa-ba)+(ab-bb))/2; history = ((aa-ab)+(ba-bb))/2; total = aa-bb
                        identities.append(float(np.max(abs(total-predictor-history))))
                        if identities[-1] > c['algebra_atol']: raise ValueError('matrix decomposition failed')
                        effects[a+'_minus_'+b] = {'predictor_F': stats(predictor), 'source_history_H': stats(history), 'diagonal_total': stats(total)}
                    rep[mode] = {'matrix': matrices, 'effects': effects}
                replay_summary[pi_key][cell] = rep
                for mode in c['oracle_modes']:
                    oracle_result[mode] = {reference: {metric: stats([oracle_value(oracles[pi,k,t], mode, label, reference=reference, metric=metric) for t in c['trials']])
                        for metric in ('expected_repeat_L1','empirical_repeat_L1','mi_error')} for reference in ('raw_reference','matched_reference')}
                effects = {}
                for a,b in [('FIX_CONT','JOINT_CONT'), ('FIX_BIN','JOINT_BIN')]:
                    effects[a+'_minus_'+b] = stats([oracle_value(oracles[pi,k,t], a, label)-oracle_value(oracles[pi,k,t], b, label) for t in c['trials']])
                effects['FIX_MODELINFO_minus_TF_MODELINFO'] = stats([
                    oracle_value(oracles[pi,k,t], 'FIX_MODELINFO', label)-oracles[pi,k,t]['teacher']['MODELINFO']['matched_reference'][str(label)]['expected_repeat_L1'] for t in c['trials']])
                oracle_result['effects'] = effects; oracle_summary[pi_key][cell] = oracle_result
    for mode in c['replay_modes']:
        for a,b in c['pairs']:
            for component in ('predictor_F','source_history_H'):
                key = mode+'/'+a+'_minus_'+b+'/'+component
                screens[key] = screen({p: cells['kappa_1_context_1'][mode]['effects'][a+'_minus_'+b][component] for p,cells in replay_summary.items()}, c)
    for effect in ('FIX_CONT_minus_JOINT_CONT','FIX_BIN_minus_JOINT_BIN','FIX_MODELINFO_minus_TF_MODELINFO'):
        screens['oracle/'+effect] = screen({p: cells['kappa_1_context_1']['effects'][effect] for p,cells in oracle_summary.items()}, c)
    # Retrospective tolerances: explicitly not a new or retroactive PASS rule.
    previous = json.loads((ROOT/'docs/cs_saf/rollout_audit_v1_result.json').read_text())
    raw = {(r['pi'],r['kappa'],r['trial'],r['candidate']): r for r in previous['raw_reports']}
    tradeoffs = {}
    for pi in c['prevalences']:
        result = {}
        for metric in ('empirical_repeat_L1','mi_error'):
            base = [raw[pi,1,t,'E']['native']['1'][metric] for t in c['trials']]
            cand = [raw[pi,1,t,'L003']['native']['1'][metric] for t in c['trials']]
            difference = stats(np.array(cand)-base)
            result[metric] = {'E_mean': float(np.mean(base)), 'L003_mean': float(np.mean(cand)),
                'difference': difference, 'relative_cost_percent_ratio_of_means': float(100*difference['mean']/np.mean(base)),
                'required_mean_tolerance': max(0.,difference['mean']),
                'required_descriptive_upper_tolerance': max(0.,difference['ci95_descriptive'][1]),
                'justified_acceptance_margin': None, 'retroactive_PASS': False}
        null_reductions = []
        for t in c['trials']:
            deltas = []
            for k,label in ((0,0),(0,1),(1,0)):
                values = []
                for candidate in ('L003','E'):
                    path = folder_for(pi,k,t,candidate); verify_folder(path)
                    audit = json.loads((path/'intervention_audit.json').read_text())
                    values.append(audit['responses'][str(label)]['mean_copy_range'])
                deltas.append(values[0]-values[1])
            null_reductions.append(float(np.mean(deltas)))
        result['equal_three_null_copy_range_change'] = stats(null_reductions)
        result['active_grid_TV_change'] = previous['paired_contrasts']['L003_minus_E'][f'{pi:.2f}']['kappa_1_context_1']['TF_all']['grid_mark_TV']
        tradeoffs[f'{pi:.2f}'] = result
    full = {'version': c['version'], 'base_commit': c['base_commit'], 'execution': execution, 'gates': gates,
        'new_fits': 0, 'models': 120, 'cells': 40, 'matrix_evaluations': 2160, 'replayed_entity_sequences': 552960,
        'oracle_generated_sequences': 153600, 'verified_output_checksums': checked, 'verified_parent_checksums': parent_checked,
        'maximum_decomposition_error': max(identities),
        'maximum_parent_probability_error': max(r['max_diagonal_probability_error'] for r in replays.values()),
        'maximum_parent_metric_error': max(r['max_diagonal_metric_error'] for r in replays.values()),
        'maximum_oracle_replay_error': max(r['oracle_replay_max_error'] for r in oracles.values()),
        'technical_checks': 'PASS', 'replay_summary': replay_summary, 'oracle_summary': oracle_summary,
        'descriptive_screens': screens, 'tradeoff_frontier': tradeoffs,
        'oracle_trial_role': 'sampling variation only; not learned-model or independent-data replication',
        'scientific_success_required_for_preservation': False, 'output_manifests': manifests,
        'raw_replays': list(replays.values()), 'raw_oracle_reports': list(oracles.values())}
    write_json(root/'summary.json', full)
    compact = copy.deepcopy(full)
    del compact['raw_replays']; del compact['raw_oracle_reports']
    compact['full_evidence'] = {'path': str((root/'summary.json').resolve()), 'sha256': sha256(root/'summary.json')}
    write_json(root/'compact_result.json', compact)
    print(json.dumps({k:compact[k] for k in ('technical_checks','models','cells','matrix_evaluations','oracle_generated_sequences','maximum_decomposition_error','maximum_parent_probability_error')},indent=2))


if __name__ == '__main__': main()
