"""Postprocessing audit; no fitting, sampling, or held-out event loading."""
from datetime import datetime, timezone
import json
import subprocess
import numpy as np
import pandas as pd
import torch
from run_initial import ROOT, OUT, DOCS, PRIOR, ARMS, CFG, verify, digest, write, current


def main():
    verify()
    decision = json.loads((DOCS / 'COMPLETE.json').read_text())
    status = json.loads((OUT / 'STATUS.json').read_text())
    assert not status['running'] and not status['pending']
    assert len(status['finished']) == 6
    assert all(x['exit_code'] == 0 for x in status['finished'].values())
    assert decision['new_full_generations'] == 24
    fits = json.loads((OUT / 'FIT_COMPLETE.json').read_text())
    records = []
    base = current()['state_dict']
    for arm in ARMS:
        head = OUT / arm / 'time_head.pt'
        assert digest(head) == fits['heads'][arm]
        state = torch.load(head, map_location='cpu', weights_only=True)['state_dict']
        for key, value in base.items():
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)
        for fs in CFG['fit_seeds']:
            routing = json.loads((OUT / f'worker_{fs}' / arm / 'ROUTING_CHECK.json').read_text())
            assert routing['strict_past_clock_exact']
            assert routing['reordered_compacted_batch']
            assert routing['strict_native_checkpoint']
            assert routing['first_native_logits'] == 28
            assert routing['sampled_digit_positions'] == 210
            folder = OUT / 'runs' / f'{arm}_{fs}'
            done = json.loads((folder / 'COMPLETE.json').read_text())
            assert done['generations'] == 4 and not done['test_events_read']
            for gs in CFG['generation_seeds']:
                rec = json.loads((folder / f'GENERATION_{gs}.json').read_text())
                assert rec['exact_labels_lengths']
                assert rec['head_sha256'] == digest(head)
                assert rec['sha256'] == digest(folder / f'generated_validation_{gs}.parquet')
                records.append(rec)
    checks = pd.read_csv(DOCS / 'independent_checks.csv')
    labels = pd.read_csv(DOCS / 'labels_lengths_audit.csv')
    assert len(checks) == 24 and checks.exact_labels_lengths.all()
    assert len(labels) == 56
    for _, group in labels.groupby(['fit_seed', 'generation_seed']):
        assert group.rows.nunique() == group.fraud_rate.nunique() == 1
    direction = pd.read_csv(DOCS / 'posthoc_gap_direction.csv')
    horizon = pd.read_csv(DOCS / 'horizon_diagnostic.csv')
    paired = direction.merge(horizon, on=['arm', 'fit_seed', 'generation_seed', 'start', 'end'])
    assert len(paired) == 192
    np.testing.assert_allclose(paired.gap_log_w1, paired.gap_w1, rtol=0, atol=1e-12)
    tails = pd.read_csv(DOCS / 'posthoc_tail_parity.csv')
    assert len(tails[tails.arm.isin(ARMS)]) == 24
    assert tails.tail_gap_exact.all()
    prior_hashes = json.loads((PRIOR / 'report/ARTIFACT_HASHES.json').read_text())['sha256']
    for relative, sha in prior_hashes.items():
        assert digest(PRIOR / relative) == sha, relative
    rows = sum(r['rows'] for r in records)
    assert rows == decision['generated_rows'] == int(checks.rows.sum())
    report = dict(
        completed_utc=datetime.now(timezone.utc).isoformat(),
        completed_workers=6, new_full_generations=24, generated_rows=rows,
        independently_checked_generations=len(checks),
        matched_comparison_generations=len(labels),
        prior_artifact_hashes_verified=len(prior_hashes),
        original_current_state_tensors_exact=True,
        generation_output_and_head_hashes_verified=True,
        native_checkpoint_and_clock_routing_checks_passed=6,
        posthoc_direction_matching_registered_horizon_rows=192,
        new_generations_with_exact_current_tail_gaps=24,
        labels_lengths_and_fraud_rate_exact=True,
        frozen_source_protocol_and_config_verified=True,
        new_native_argn_fits=0, initial_mixture_fits=1, calibration_fits=2,
        tracked_checkout_clean=True, test_events_read=False,
        active_model='clock_prefix', automatic_promotion=False,
        committed=False, pushed=False, experiment_running=False,
        baseline_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(DOCS / 'FINAL_AUDIT.json', report)
    # Hash the deliverable code, learned heads, raw generations and reports.
    # Copied third-party workspaces are already covered by inherited manifests.
    inventory = {}
    for path in sorted(OUT.rglob('*')):
        relative = path.relative_to(OUT)
        if not path.is_file() or '__pycache__' in relative.parts or 'workspaces' in relative.parts:
            continue
        if path == DOCS / 'ARTIFACT_HASHES.json':
            continue
        if relative.parts[0] in {'logs'}:
            continue
        inventory[str(relative)] = digest(path)
    write(DOCS / 'ARTIFACT_HASHES.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), sha256=inventory))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
