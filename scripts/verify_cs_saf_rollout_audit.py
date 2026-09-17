"""Independent arithmetic and identity checks on the finished diagnostic output."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/cs_saf/rollout_audit_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((ART/'summary.json').read_text())
    raw = summary['raw_reports']; by_key = {}
    arrays = 0; decompositions = 0; max_reconstruction_error = 0.; alignment = {}
    for report in raw:
        pi, k, t, c = report['pi'], report['kappa'], report['trial'], report['candidate']
        by_key[pi, k, t, c] = report
        folder = ART/f'pi_{pi:.2f}/kappa_{k}/trial_{t}/{c}'
        with np.load(folder/'validation_scores.npz', allow_pickle=False) as archive:
            values, ids = archive['scores'], archive['entity_ids']
            if not np.isfinite(values).all() or (values[:, [2, 5]] < -2e-6).any():
                raise ValueError('invalid TV decomposition')
            np.testing.assert_allclose(values[:, 0], values[:, 1]+values[:, 2], rtol=0, atol=1e-12)
            np.testing.assert_allclose(values[:, 3], values[:, 4]+values[:, 5], rtol=0, atol=1e-12)
            key = (pi, k)
            if key in alignment: np.testing.assert_array_equal(ids, alignment[key])
            else: alignment[key] = ids.copy()
            arrays += 1
        domains = [report[key] for key in ('TF_all', 'TF_panel', 'TF_quantized_panel', 'native')]
        domains.extend(report['rollouts'].values())
        for domain in domains:
            for label, group in domain.items():
                ref = report['reference_curves'][label]
                weights = np.array(ref['counts'], dtype=float); weights /= weights.sum()
                target = np.array(ref['observed_repeat']); empirical = np.array(group['observed_repeat'])
                reconstructed = np.array(group['signed_decomposition']).sum(0)
                err = float(np.max(abs(reconstructed-(empirical-target))))
                max_reconstruction_error = max(max_reconstruction_error, err)
                assert err < 1e-12
                assert abs(weights@abs(empirical-target)-group['empirical_repeat_L1']) < 1e-12
                assert abs(weights@abs(np.array(group['model_repeat'])-target)-group['expected_repeat_L1']) < 1e-12
                assert sum(group['counts']) == sum(group['step_counts'])
                decompositions += 1
    # Reconstruct the decisive stage contrast from each tape, independent of
    # the aggregation helper and the reported screen flags.
    screen = {}
    for a, b in (('E', 'U'), ('L003', 'E')):
        pair = a+'_minus_'+b; eligible = []
        for pi in (.05, .10, .25, .50):
            differences = []
            for trial in range(5):
                effects = []
                for candidate in (a, b):
                    report = by_key[pi, 1, trial, candidate]
                    observations = [g['1']['expected_repeat_L1'] for name, g in report['rollouts'].items() if name.startswith('OBS_')]
                    assert len(observations) == 3
                    effects.append(sum(observations)/3-report['TF_panel']['1']['expected_repeat_L1'])
                differences.append(effects[0]-effects[1])
            reported = summary['paired_contrasts'][pair][f'{pi:.2f}']['kappa_1_context_1']['stages']['TF_panel_to_OBS']['values']
            np.testing.assert_allclose(differences, reported, atol=1e-15, rtol=0)
            if sum(v > 0 for v in differences) >= 4 and sum(differences)/5 >= .001: eligible.append(pi)
        screen[pair] = {'eligible_prevalences': eligible, 'PASS': len(eligible) >= 3}
        assert screen[pair]['PASS'] == summary['screens'][pair]['stage_TF_panel_to_OBS']['1']['PASS']
    evidence = ROOT/'docs/cs_saf/rollout_audit_v1_result.json'
    assert digest(evidence) == digest(ART/'compact_result.json')
    output = {'PASS': True, 'audit_source_commit': summary['source_commit'],
        'verification_source_sha256': digest(Path(__file__)), 'compact_result_sha256': digest(evidence),
        'full_summary_sha256': digest(ART/'summary.json'), 'entity_array_groups_checked': arrays,
        'signed_decompositions_and_curves_checked': decompositions,
        'maximum_signed_reconstruction_error': max_reconstruction_error,
        'independently_reconstructed_mark_feedback_screens': screen}
    (ROOT/'docs/cs_saf/rollout_audit_v1_verification.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__': main()
