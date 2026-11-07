"""Repair only the historical audit's interval selector; retain frozen simulator."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import diagnose as d


def validate_preserved_snapshot():
    # The original attempt read all 40 files and checked their labels/lengths,
    # then wrote this table before its start-only interval selection failed.
    result = pd.read_csv(d.DOCS / 'saved_generation_positions.csv')
    previous = pd.read_csv(d.CURRENT / 'report/horizon_diagnostic.csv')
    assert len(result) == 5 * 8 * len(d.BINS)
    assert not result.duplicated(['model', 'fit_seed', 'generation_seed', 'start', 'end']).any()
    for name, oldname in [('clock_prefix', 'clock_prefix'), ('refined_personal', 'personal_six'),
                          ('refined_position', 'position_six'), ('clock_joint', 'clock_joint')]:
        a = result[result.model.eq(name) & result.start.eq(6) & result.end.eq(50)]
        b = previous[previous.arm.eq(oldname) & previous.start.eq(6) & previous.end.eq(50)]
        keys = ['fit_seed', 'generation_seed']
        paired = a.merge(b, on=keys, validate='one_to_one', suffixes=('_new', '_old'))
        assert len(a) == len(b) == len(paired) == 8
        np.testing.assert_allclose(paired.gap_w1_new, paired.gap_w1_old, rtol=0, atol=1e-12)
    d.write(d.DOCS / 'SAVED_RESULTS_CHECK.json', dict(previous_6_49_metrics_reproduced=32,
        outputs_read=40, rows_in_generated_files_unchanged=True,
        amendment='AMENDMENT1.md: select both start=6 and exclusive end=50'))
    print('SAVED_GENERATION_REPRODUCTION_PASSED_32', flush=True)


original_verify = d.verify


def verify_amendment():
    original_verify()
    m = json.loads((d.OUT / 'AMENDMENT1_MANIFEST.json').read_text())
    for p, sha in m['hashes'].items():
        assert d.digest(p) == sha, p


if __name__ == '__main__':
    d.verify = verify_amendment
    d.snapshots = validate_preserved_snapshot
    d.run()
