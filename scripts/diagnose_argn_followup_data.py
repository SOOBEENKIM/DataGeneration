"""D0: source alignment and retrospective cohorts, without held-out events."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_argn_state_first import ROOT, OUT, SOURCE, digest, write, check_manifest
from benchmarks.argn_past_state import PastState
from mostlyai.engine._workspace import Workspace

DEST = ROOT / 'artifacts/argn_followup_diagnostics_v1'
DOCS = ROOT / 'docs/argn_state_first_v1/followup'
DATA = ROOT.parent / 'cof-seqgen-0707-2119-Version3-complete/data/cof_seqgen_saf'
CORE = ['gap', 'receiver_or_mark', 'amount_or_numeric_value', 'category', 'event_is_fraud']
ARMS = ['B', 'B_S', 'B_event_weighted']
SEEDS = [20260930, 20261001]


def folder(arm, fs):
    base = OUT if arm != 'B_event_weighted' else ROOT / 'artifacts/argn_event_weight_control_v1'
    return base / 'runs' / f'{arm}_{fs}'


def customer_summary(frame):
    rows = []
    for cid, g in frame.groupby('customer_id', sort=True):
        g = g.sort_values('event_index', kind='stable')
        y = pd.to_numeric(g.event_is_fraud).to_numpy(dtype=int)
        start = np.flatnonzero((y == 1) & np.r_[True, y[:-1] != 1])
        end = np.flatnonzero((y == 1) & np.r_[y[1:] != 1, True])
        rows.append(dict(customer_id=int(cid), events=len(g), fraud=int(y.sum()),
                         cohort='normal_only' if not y.any() else ('all_fraud' if y.all() else 'mixed'),
                         first_fraud=int(y[0]), previous_fraud=int(y[:-1].sum()),
                         previous_normal=int((y[:-1] == 0).sum()),
                         onset=int(((y[:-1] == 0) & (y[1:] == 1)).sum()),
                         terminate=int(((y[:-1] == 1) & (y[1:] == 0)).sum()),
                         runs=len(start), run_events=int((end-start+1).sum()),
                         right_censored_fraud=int(y[-1] == 1),
                         short=int(len(g) <= 100)))
    return pd.DataFrame(rows)


def aggregate_cohorts(table, dataset, arm='', fit_seed=0, generation_seed=0):
    rows = []
    for cohort in ['all', 'normal_only', 'mixed', 'all_fraud']:
        g = table if cohort == 'all' else table[table.cohort == cohort]
        if len(g) == 0: continue
        s = g.select_dtypes('number').sum()
        rows.append(dict(dataset=dataset, arm=arm, fit_seed=fit_seed, generation_seed=generation_seed,
                         cohort=cohort, customers=len(g), short_customers=int(g.short.sum()),
                         events=int(s.events), fraud=int(s.fraud), fraud_rate=s.fraud/s.events,
                         first_fraud_rate=s.first_fraud/len(g), min_length=g.events.min(),
                         median_length=g.events.median(), max_length=g.events.max(),
                         runs=int(s.runs), previous_fraud=int(s.previous_fraud),
                         terminations=int(s.terminate),
                         termination_rate=s.terminate/s.previous_fraud if s.previous_fraud else np.nan,
                         onset_rate=s.onset/s.previous_normal if s.previous_normal else np.nan,
                         mean_observed_fraud_run=s.run_events/s.runs if s.runs else np.nan))
    return rows


def main():
    check_manifest()
    DEST.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
    assert not (DEST/'D0_COMPLETE.json').exists()
    roles = pd.read_parquet(SOURCE/'prepared/roles.parquet')
    allowed = set(roles.entity_id.astype(str))
    raw_path = DATA/'raw/sparkov/fraudTrain.csv'
    ids = pd.read_csv(raw_path, usecols=['cc_num'], dtype={'cc_num':str}).cc_num
    skipped = set((np.flatnonzero(~ids.isin(allowed)) + 1).tolist())
    use = ['cc_num','trans_num','trans_date_trans_time','amt','merchant','category','is_fraud']
    raw = pd.read_csv(raw_path, usecols=use, dtype={'cc_num':str},
                      skiprows=lambda lineno: lineno in skipped, low_memory=False)
    assert set(raw.cc_num) == allowed
    raw['timestamp'] = pd.to_datetime(raw.trans_date_trans_time, format='%Y-%m-%d %H:%M:%S').astype('int64')/1e9
    raw = raw.rename(columns={'cc_num':'entity_id','trans_num':'event_id','amt':'amount_or_numeric_value',
                              'merchant':'receiver_or_mark','is_fraud':'event_is_fraud'})
    raw['entity_id'] = raw.entity_id.astype('int64')
    raw['receiver_or_mark'] = raw.receiver_or_mark.str.strip()
    raw = raw.sort_values(['entity_id','timestamp','event_id'], kind='stable').reset_index(drop=True)
    raw['event_index'] = raw.groupby('entity_id').cumcount()
    raw['gap'] = raw.groupby('entity_id').timestamp.diff()
    cols = ['entity_id','event_id','event_index','timestamp',*CORE]
    canonical_path = DATA/'canonical/sparkov/events.parquet'
    canonical = pd.read_parquet(canonical_path, columns=cols,
                                filters=[('entity_id','in',roles.entity_id.tolist())])
    canonical = canonical.sort_values(['entity_id','event_index']).reset_index(drop=True)
    raw = raw[cols]
    for c in ['category','receiver_or_mark','event_is_fraud']:
        raw[c] = raw[c].astype(str); canonical[c] = canonical[c].astype(str)
    pd.testing.assert_frame_equal(raw, canonical, check_dtype=False, check_exact=True)
    merged = canonical.merge(roles, on='entity_id', validate='many_to_one')
    cohort_rows = []
    prepared_hashes = {}
    for split in ['train','validation']:
        actual_path = SOURCE/'prepared'/f'{split}.parquet'
        actual = pd.read_parquet(actual_path).rename(columns={'entity_id':'customer_id'})
        expected = merged[merged.split == split][actual.columns].copy()
        actual = actual.sort_values(['customer_id','event_index']).reset_index(drop=True)
        expected = expected.sort_values(['customer_id','event_index']).reset_index(drop=True)
        for c in ['category','receiver_or_mark','event_is_fraud']:
            actual[c] = actual[c].astype(str); expected[c] = expected[c].astype(str)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
        summary = customer_summary(actual)
        summary.to_parquet(DEST/f'{split}_customers.parquet', index=False)
        cohort_rows.extend(aggregate_cohorts(summary, split))
        prepared_hashes[str(actual_path)] = digest(actual_path)
    encoded_rows = []
    ws = Workspace(OUT/'prepared/workspace'); codec = PastState(ws.tgt_stats.read())
    label = codec.prefixes['event_is_fraud']+'__cat'
    for path in sorted((OUT/'prepared/workspace/OriginalData/encoded-data').glob('*.parquet')):
        f = pd.read_parquet(path)
        for cohort in ['all','normal_only','mixed','all_fraud']:
            seqs = [np.asarray(y) == codec.codes['1'] for y in f[label]]
            if cohort != 'all':
                seqs = [y for y in seqs if ('normal_only' if not y.any() else ('all_fraud' if y.all() else 'mixed')) == cohort]
            encoded_rows.append(dict(file=path.name,cohort=cohort,customers=len(seqs),
                                     events=sum(len(y) for y in seqs),fraud=sum(int(y.sum()) for y in seqs),
                                     short_customers=sum(len(y)<=100 for y in seqs)))
    generation_hashes = {}
    for arm in ARMS:
        for fs in SEEDS:
            for gs in [20261011,20261012]:
                path = folder(arm,fs)/f'generated_validation_{gs}.parquet'
                g = pd.read_parquet(path).rename(columns={'entity_id':'customer_id'})
                cohort_rows.extend(aggregate_cohorts(customer_summary(g),'generated',arm,fs,gs))
                generation_hashes[str(path)] = digest(path)
    pd.DataFrame(cohort_rows).to_csv(DOCS/'cohorts.csv', index=False)
    pd.DataFrame(encoded_rows).to_csv(DOCS/'encoded_cohorts.csv',index=False)
    write(DEST/'D0_COMPLETE.json', dict(raw_to_canonical_exact=True, canonical_to_prepared_exact=True,
          raw_allowed_customers=len(allowed),raw_allowed_events=len(raw),
          source_raw_sha256=digest(raw_path),prepared_hashes=prepared_hashes,generation_hashes=generation_hashes,
          test_events_read=False, raw_filter='ID-only first pass; excluded row numbers skipped on attribute parse',
          script_sha256=digest(__file__),protocol_sha256=digest(ROOT/'docs/argn_state_first_v1/FOLLOWUP_DIAGNOSTIC_PROTOCOL.md')))
    print(pd.DataFrame(cohort_rows).query("dataset != 'generated'").to_string(index=False), flush=True)
    print('D0_COMPLETE', flush=True)


if __name__ == '__main__': main()
