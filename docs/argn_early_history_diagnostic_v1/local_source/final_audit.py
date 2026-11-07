"""Validate frozen evidence, paired supports, completed workers and result inventory."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import diagnose as d
import diagnose_amendment2 as amended

amended.verify()
prior_hashes = json.loads((d.CURRENT / 'report/ARTIFACT_HASHES.json').read_text())['sha256']
for relative, expected in prior_hashes.items():
    assert d.digest(d.CURRENT / relative) == expected, relative

completed = json.loads((d.DOCS / 'COMPLETE.json').read_text())
assert completed['trajectories'] == 426 and completed['time_values'] == 14138474
assert completed['new_model_fits'] == completed['full_transaction_generations'] == 0
jobs = [d.DOCS / f'{s}_COMPLETE.json' for s in ['optimization', 'internal_validation', 'development']]
jobs += [d.DOCS / f'generated_path_{fs}_{gs}_COMPLETE.json' for fs in d.CFG['fit_seeds'] for gs in d.CFG['generation_seeds']]
job_data = [json.loads(p.read_text()) for p in jobs]
assert sum(x['trajectories'] for x in job_data) == 426
assert sum(x['time_values'] for x in job_data) == 14138474
data = pd.read_csv(d.DOCS / 'all_diagnostics.csv')
keys = ['split', 'path_source', 'fit_seed', 'generation_seed', 'noise_seed', 'start', 'end', 'group']
assert len(data) == 5020
assert not data.duplicated(keys + ['model', 'mode']).any()
assert np.isfinite(data.gap_w1).all() and (data.gap_w1 >= 0).all()
counts = data.groupby(keys)[['events', 'customers']].nunique()
assert counts.eq(1).all().all(), 'Unmatched comparison support'
first = data[(data.start == 1) & (data.end == 6) & data['mode'].isin(['GG', 'RR'])]
comparison = first.pivot(index=keys + ['model'], columns='mode', values='gap_w1')
np.testing.assert_array_equal(comparison.GG, comparison.RR)
status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=d.ROOT, text=True).strip()
assert not status
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=d.ROOT, text=True).strip()
result = dict(completed_utc=datetime.now(timezone.utc).isoformat(), prior_artifacts_verified=len(prior_hashes),
    input_and_amendment_hashes_verified=True, completed_jobs=len(jobs), trajectories=426,
    metric_rows=len(data), matched_supports=True, first_five_real_generated_clock_noop_exact=True,
    original_heads_and_generations_unchanged=True, git_head=commit, tracked_status_clean=True,
    native_training_jobs=0, native_generation_jobs=0, diagnostic_workers_finished=True,
    test_events_read=False, committed=False, pushed=False,
    active_model='clock_prefix', local_candidate='argn_clock_stratified_loss_v1/personal_six',
    next_initialization_training_started=False)
d.write(d.DOCS / 'FINAL_AUDIT.json', result)
paths = sorted(p for p in d.OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
               and p.name != 'ARTIFACT_HASHES.json')
d.write(d.DOCS / 'ARTIFACT_HASHES.json', dict(sha256={str(p.relative_to(d.OUT)): d.digest(p) for p in paths}))
print(json.dumps(result, indent=2), flush=True)
