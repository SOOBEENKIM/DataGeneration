"""Independent pandas/dictionary reductions; never imports the audited experiment.

Post-run diagnostics of sparse table support are descriptive, not new fits/tuning.
Only development identities are read. No individual records are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def lookup(series, frame, keys):
    idx = pd.Index(frame[keys[0]]) if len(keys) == 1 else pd.MultiIndex.from_frame(frame[keys])
    return series.reindex(idx, fill_value=0).to_numpy(float)


def reconstruct(canonical, audit, cfg, name):
    splits = pd.read_parquet(canonical / 'entity_splits.parquet')
    dev = splits.loc[splits.split.isin(['train', 'validation'])]
    cols = ['entity_id', 'event_id', 'timestamp', 'receiver_or_mark', 'amount_or_numeric_value']
    secondary = 'transaction_type' if name == 'berka' else 'category'
    cols.append(secondary)
    d = pd.read_parquet(canonical / 'events.parquet', columns=cols,
                        filters=[('entity_id', 'in', dev.entity_id.tolist())])
    d = d.rename(columns={'receiver_or_mark': 'mark', 'amount_or_numeric_value': 'amount'})
    d = d.sort_values(['entity_id', 'timestamp', 'event_id'], kind='stable').reset_index(drop=True)
    d['split'] = d.entity_id.map(dev.set_index('entity_id').split)
    assert not d.split.isna().any()
    tr_ids = dev.loc[dev.split.eq('train'), 'entity_id'].tolist()
    ranked = sorted(tr_ids, key=lambda v: hashlib.sha256(f'{name}:{v}:{cfg["partition_seed"]}'.encode()).digest())
    fit_ids = set(ranked[:int(len(ranked)*cfg['fit_fraction'])])
    assert hashlib.sha256(json.dumps(sorted(fit_ids)).encode()).hexdigest() == audit['fit_ids_sha256']
    d['fit'] = d.entity_id.isin(fit_ids)
    d['gap'] = d.groupby('entity_id').timestamp.diff()
    edges = np.unique(d.loc[d.fit & d.gap.gt(0), 'gap'].quantile(cfg['positive_gap_quantiles']).to_numpy())
    np.testing.assert_array_equal(edges, audit['gap_edges'])
    d['gap_bin'] = np.digitize(d.gap, edges, right=True)+1
    d.loc[d.gap.eq(0), 'gap_bin'] = 0
    d.loc[d.gap.isna(), 'gap_bin'] = -1
    vocab = sorted(d.loc[d.fit, 'mark'].unique().tolist())
    assert hashlib.sha256(json.dumps(vocab).encode()).hexdigest() == audit['vocabulary_sha256']
    np.testing.assert_allclose(d.loc[d.fit, 'amount'].quantile(.95), audit['amount_fit_p95'], rtol=0, atol=1e-12)
    # Local per-entity recurrence, independently of the experiment's global cumsum.
    primary = np.zeros(len(d), bool)
    past_run = np.zeros(len(d), int)
    for _, rows in d.groupby('entity_id', sort=False):
        indexes = rows.index.to_numpy()
        timestamps, marks = rows.timestamp.to_numpy(), rows.mark.to_numpy()
        counts = pd.Series(timestamps).value_counts()
        single = pd.Series(timestamps).map(counts).to_numpy() == 1
        run = 1
        for t in range(1, len(rows)):
            past_run[indexes[t]] = run
            ok = bool(single[t] and single[t-1])
            primary[indexes[t]] = ok
            run = run+1 if ok and marks[t] == marks[t-1] else 1
    d['unambiguous'] = primary
    d['previous_mark'] = d.groupby('entity_id').mark.shift()
    d['repeat'] = d.mark.eq(d.previous_mark).astype(float)
    d['secondary'] = d[secondary].fillna('<MISSING>').astype(str)
    d['secondary_repeat'] = d.secondary.eq(d.groupby('entity_id').secondary.shift()).astype(float)
    d['log_amount'] = np.log1p(d.amount)
    d['run_bin'] = np.select([past_run <= 1, past_run <= 3, past_run <= 7], ['1', '2-3', '4-7'], default='8+')
    return d, vocab


def independent_scores(tr, ev, vocab, cfg):
    a = cfg['conditional_pseudocount']
    counts = tr.groupby('mark').size()
    p0 = (lookup(counts, ev, ['mark']) + cfg['marginal_pseudocount']) / (len(tr)+(len(vocab)+1)*cfg['marginal_pseudocount'])
    # Vocabulary OOV is explicitly checked below (none on these datasets).
    assert tr.mark.isin(vocab).all() and ev.mark.isin(vocab).all()
    assert tr.previous_mark.isin(vocab).all() and ev.previous_mark.isin(vocab).all()
    c1 = lookup(tr.groupby(['previous_mark', 'mark']).size(), ev, ['previous_mark', 'mark'])
    n1 = lookup(tr.groupby('previous_mark').size(), ev, ['previous_mark'])
    p1 = (c1+a*p0)/(n1+a)
    c2 = lookup(tr.groupby(['previous_mark', 'gap_bin', 'mark']).size(), ev, ['previous_mark', 'gap_bin', 'mark'])
    n2 = lookup(tr.groupby(['previous_mark', 'gap_bin']).size(), ev, ['previous_mark', 'gap_bin'])
    p2 = (c2+a*p1)/(n2+a)
    y = ev.log_amount.to_numpy()
    a0 = float(tr.log_amount.mean())
    sums1 = lookup(tr.groupby('mark').log_amount.sum(), ev, ['mark'])
    nmark = lookup(counts, ev, ['mark'])
    a1 = (sums1+a*a0)/(nmark+a)
    sums2 = lookup(tr.groupby(['mark', 'gap_bin']).log_amount.sum(), ev, ['mark', 'gap_bin'])
    nmarkgap = lookup(tr.groupby(['mark', 'gap_bin']).size(), ev, ['mark', 'gap_bin'])
    a2 = (sums2+a*a1)/(nmarkgap+a)
    vals = {'M0_mark_nll': -np.log(p0), 'M1_mark_nll': -np.log(p1), 'M2_mark_nll': -np.log(p2),
            'A0_amount_mae': abs(y-a0), 'A1_amount_mae': abs(y-a1), 'A2_amount_mae': abs(y-a2)}
    for lo, hi in [('M0_mark_nll', 'M1_mark_nll'), ('M1_mark_nll', 'M2_mark_nll'),
                   ('A0_amount_mae', 'A1_amount_mae'), ('A1_amount_mae', 'A2_amount_mae')]:
        vals[f'{hi}_minus_{lo}'] = vals[hi]-vals[lo]
    coverage = dict(evaluation_events=len(ev), unseen_previous_mark_fraction=float((n1==0).mean()),
                    unseen_previous_gap_fraction=float((n2==0).mean()),
                    unseen_previous_current_pair_fraction=float((c1==0).mean()),
                    unseen_previous_gap_current_triple_fraction=float((c2==0).mean()),
                    median_training_count_for_observed_pair=float(np.median(c1)),
                    median_training_count_for_observed_triple=float(np.median(c2)))
    return vals, coverage


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists(), 'preserve previous verification'
    cfgpath = ROOT/'configs/cs_saf_external_relations_v1.json'
    cfg = json.loads(cfgpath.read_text())
    manifest = json.loads((args.run/'run_manifest.json').read_text())
    assert sha(cfgpath) == manifest['config_sha256']
    assert sha(ROOT/'experiments/cs_saf_external_relations_v1.py') == manifest['source_sha256']
    for file, expected in manifest['files'].items():
        assert sha(args.run/file) == expected
    report = dict(passed=True, run_manifest_sha256=sha(args.run/'run_manifest.json'),
                  verifier_sha256=sha(__file__), independently_imports_experiment=False,
                  point_estimates_only=True, interval_algorithm_unit_tested_not_independently_recomputed=True,
                  test_outcome_analysis=False, new_fits=0, datasets={})
    for name in cfg['datasets']:
        audit = json.loads((args.run/f'{name}_audit.json').read_text())
        d, vocab = reconstruct(args.data_root/'canonical'/name, audit, cfg, name)
        rows_checked, max_error = 0, 0.
        profiles = pd.read_csv(args.run/f'{name}_profiles.csv', dtype={'run_bin': str})
        for r in profiles.to_dict('records'):
            sub = d.loc[d.split.eq(r['split']) & d.unambiguous]
            for key in r['grouping'].split('x'):
                sub = sub.loc[sub[key].eq(r[key])]
            assert len(sub) == r['rows'] and sub.entity_id.nunique() == r['entities']
            val, macro = sub[r['metric']].mean(), sub.groupby('entity_id')[r['metric']].mean().mean()
            np.testing.assert_allclose([val, macro], [r['value'], r['entity_mean']], rtol=0, atol=1e-11)
            max_error = max(max_error, abs(val-r['value']), abs(macro-r['entity_mean']))
            assert r['coverage'] == (len(sub)>=cfg['min_transitions'] and sub.entity_id.nunique()>=cfg['min_entities'])
            rows_checked += 1
        expected = pd.read_csv(args.run/f'{name}_empirical_scores.csv').set_index(['evaluation', 'comparison'])
        scores_checked, support = 0, {}
        for label, trmask, evmask in [('internal_check', d.fit, d.split.eq('train') & ~d.fit),
                                      ('validation', d.split.eq('train'), d.split.eq('validation'))]:
            tr, ev = d.loc[trmask & d.unambiguous], d.loc[evmask & d.unambiguous]
            vals, support[label] = independent_scores(tr, ev, vocab, cfg)
            for key, val in vals.items():
                ref = expected.loc[(label, key)]
                assert len(val) == ref['rows'] and ev.entity_id.nunique() == ref['entities']
                macro = pd.DataFrame({'id': ev.entity_id.to_numpy(), 'v': val}).groupby('id').v.mean().mean()
                np.testing.assert_allclose([val.mean(), macro], [ref['value'], ref['entity_mean']], rtol=0, atol=1e-11)
                max_error = max(max_error, abs(val.mean()-ref['value']), abs(macro-ref['entity_mean']))
                scores_checked += 1
        runs = []
        for split in ['train', 'validation']:
            for run in ['1', '2-3', '4-7', '8+']:
                sub = d.loc[d.split.eq(split) & d.unambiguous & d.run_bin.eq(run)]
                runs.append(dict(split=split, run_bin=run, transitions=len(sub), entities=int(sub.entity_id.nunique()),
                                 repeats=int(sub.repeat.sum()), estimable=bool(len(sub)>=500 and sub.entity_id.nunique()>=30)))
        extra = {}
        if name == 'sparkov':
            train, val = d.loc[d.split.eq('train')], d.loc[d.split.eq('validation')]
            counts = train.groupby('mark').secondary.nunique()
            mapping = train.drop_duplicates('mark').set_index('mark').secondary
            mismatch = val.secondary.ne(val.mark.map(mapping))
            extra = dict(train_merchant_categories_max=int(counts.max()),
                         train_merchants_with_multiple_categories=int(counts.gt(1).sum()),
                         validation_events_violating_train_merchant_category_map=int(mismatch.sum()))
        report['datasets'][name] = dict(development_rows=len(d), profile_rows_verified=rows_checked,
            score_rows_verified=scores_checked, max_absolute_error=max_error,
            postrun_empirical_support_diagnostic=support, complete_run_coverage=runs,
            postrun_category_mapping_check=extra)
        print(name, rows_checked, scores_checked, 'PASS', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
