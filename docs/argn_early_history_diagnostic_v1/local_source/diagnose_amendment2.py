"""Use the recorded entity-to-encoded-record map for generated-label controls."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import diagnose as d
import diagnose_amendment1 as a1

POSITIONS = d.ROOT / 'artifacts/argn_followup_diagnostics_v1/positions.parquet'
INPUTS = POSITIONS.parent / 'D1_INPUTS.json'
OriginalData = d.Data


def identity_map():
    old = json.loads(INPUTS.read_text())
    assert d.digest(POSITIONS) == old['positions_sha256']
    p = pd.read_parquet(POSITIONS, columns=['record', 'customer_id', 'event_index', 'label'])
    m = pd.read_parquet(d.META / 'metadata_development.parquet')
    assert p[['record', 'event_index']].rename(columns={'record': 'entity_id'}).equals(m[['entity_id', 'event_index']])
    np.testing.assert_array_equal(p.label, m.label)
    pairs = p[['record', 'customer_id']].drop_duplicates()
    assert len(pairs) == 147 and pairs.record.is_unique and pairs.customer_id.is_unique
    real = d.frame(d.SOURCE / 'prepared/validation.parquet').rename(columns={'entity_id': 'customer_id'})
    joined = p.merge(real, on=['customer_id', 'event_index'], validate='one_to_one', suffixes=('_metadata', '_source'), sort=False)
    assert len(joined) == len(m)
    np.testing.assert_array_equal(joined.label_metadata, joined.label_source)
    early = m.event_index.lt(d.CFG['horizon'])
    np.testing.assert_allclose(joined.gap[early], m.gap[early], rtol=0, atol=0, equal_nan=True)
    # Full source has one old numeric-clipping difference after our horizon.
    diff = np.isfinite(joined.gap) & (joined.gap.to_numpy() != m.gap.to_numpy())
    return pairs.set_index('customer_id').record.to_dict(), dict(customers=len(pairs), rows=len(m),
        labels_exact=True, early_gaps_exact=True, later_decoded_gap_differences=int(diff.sum()),
        early_decoded_gap_differences=int((diff & early).sum()))


class DataMapped(OriginalData):
    def __init__(self, split, fs=None, gs=None):
        super().__init__(split)
        if fs is None:
            return
        assert split == 'development'
        mapping, audit = identity_map()
        generated = d.frame(d.locate('refined_personal', fs, gs))
        generated = generated[generated.event_index.lt(d.CFG['horizon'])].copy()
        records = generated.entity_id.map(mapping)
        assert records.notna().all() and records.nunique() == 147
        column = {record: i for i, record in enumerate(self.ids)}
        gj = records.map(column).to_numpy(dtype=int)
        gt = generated.event_index.to_numpy()
        present = np.zeros(self.shape, bool)
        present[gt, gj] = True
        self.mask &= present
        self.phase = np.zeros(self.shape, int)
        self.phase[gt, gj] = generated.phase_num.to_numpy()
        self.tag.update(path_source='generated_labels', fit_seed=fs, generation_seed=gs)
        assert not self.mask[0].any()


def verify():
    a1.verify_amendment()
    m = json.loads((d.OUT / 'AMENDMENT2_MANIFEST.json').read_text())
    for p, sha in m['hashes'].items():
        assert d.digest(p) == sha, p


def run():
    verify()
    assert not (d.DOCS / 'COMPLETE.json').exists()
    mapping, audit = identity_map()
    d.write(d.DOCS / 'ENTITY_MAPPING_CHECK.json', audit)
    # Reuse the three already completed, correctly aligned actual-path jobs.
    results = []
    for split in ['optimization', 'internal_validation', 'development']:
        old = json.loads((d.DOCS / f'{split}_COMPLETE.json').read_text())
        results.append((split, old['trajectories'], old['time_values']))
    d.Data = DataMapped
    specs = [('development', fs, gs) for fs in d.CFG['fit_seeds'] for gs in d.CFG['generation_seeds']]
    with ProcessPoolExecutor(max_workers=d.CFG['max_cpu_jobs']) as pool:
        results += list(pool.map(d.job, specs))
    frames = [pd.read_csv(d.DOCS / f'{name}.csv') for name, _, _ in results]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(d.DOCS / 'all_diagnostics.csv', index=False)
    keys = ['split', 'path_source', 'model', 'mode', 'start', 'end', 'group']
    combined.groupby(keys).mean(numeric_only=True).to_csv(d.DOCS / 'mean_diagnostics.csv')
    verify()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=d.ROOT, text=True).strip()
    d.write(d.DOCS / 'COMPLETE.json', dict(trajectories=sum(x[1] for x in results),
        time_values=sum(x[2] for x in results), new_model_fits=0, full_transaction_generations=0,
        original_heads_and_outputs_unchanged=True, test_events_read=False, pushed=False,
        amendments=['AMENDMENT1.md', 'AMENDMENT2.md'], completed_utc=datetime.now(timezone.utc).isoformat()))
    print('DIAGNOSTICS_COMPLETE', flush=True)


if __name__ == '__main__':
    run()
