"""Independent count-based checks of completed scientific artifacts, not quality gates."""
from collections import Counter
import json
from pathlib import Path
import runpy

from run_argn_fraud_audit import OUT, OLD, DOCS, ROOT, LABEL, MERCHANT, config, load_prepared, digest, write, pd, np


def counter_tv(a, b):
    ca, cb = Counter(a), Counter(b)
    return sum(abs(ca[k]/len(a) - cb[k]/len(b)) for k in ca.keys() | cb.keys()) / 2


def main():
    cfg = config(); base, manifest = load_prepared()
    import mostlyai.engine
    for name, expected in manifest['official_source_hashes'].items():
        assert digest(Path(mostlyai.engine.__file__).parent/name) == expected
    legacy = json.loads((DOCS/'existing_audit.json').read_text())
    for name, expected in legacy['generated_hashes'].items(): assert digest(OLD/name) == expected
    real = pd.read_parquet(base/'validation.parquet')
    fit = pd.read_parquet(base/'fit.parquet')
    supported = set(zip(fit[MERCHANT], fit.category))
    real_pairs = list(zip(real[MERCHANT], real.category))
    metrics = pd.read_csv(DOCS/'generation_metrics.csv').set_index('run')
    assert len(metrics) == len(cfg['fit_seeds']) * len(cfg['generation_seeds'])
    rows = []; curves = pd.read_csv(DOCS/'position_curves.csv')
    for seed in cfg['fit_seeds']:
        folder = OUT/f'seed_{seed}'; fitted = json.loads((folder/'FIT.json').read_text())
        from mostlyai.engine._workspace import Workspace
        ws = Workspace(folder/'workspace')
        assert digest(ws.model_tabular_weights_path) == fitted['checkpoint_sha256']
        for gs in cfg['generation_seeds']:
            path = folder/f'generated_{gs}.parquet'; record = json.loads((folder/f'generation_{gs}.json').read_text())
            assert digest(path) == record['sha256']
            assert record['checkpoint_sha256'] == fitted['checkpoint_sha256']
            raw = pd.read_parquet(path)
            assert set(raw.entity_id) <= set(real.entity_id)
            assert not raw.duplicated(['entity_id','event_index']).any()
            assert np.array_equal(raw.event_index, raw.groupby('entity_id',sort=False).cumcount())
            key = f'seed_{seed}/generated_{gs}'; m = metrics.loc[key]
            fraud = sum(str(x) == '1' for x in raw[LABEL])
            pairs = list(zip(raw[MERCHANT],raw.category))
            tv = counter_tv(real_pairs, pairs)
            assert fraud == m.generated_frauds
            assert np.isclose(fraud/len(raw), m.generated_fraud_rate, atol=1e-12, rtol=0)
            assert np.isclose(tv, m.merchant_category_tv, atol=1e-12, rtol=0)
            first_pairs = [pair for pair,idx in zip(pairs,raw.event_index) if idx == 0]
            off_support = sum(x not in supported for x in first_pairs)/len(first_pairs)
            curve = curves[curves.run.eq(key) & curves.position_band.eq(0)].iloc[0]
            assert np.isclose(off_support, curve.generated_pair_absent_from_fit, atol=1e-12, rtol=0)
            rows.append(dict(run=key, events=len(raw), customers=raw.entity_id.nunique(),
                zero_event_customers=len(set(real.entity_id)-set(raw.entity_id)),
                frauds=fraud, merchant_category_tv=tv, first_pair_absent_from_fit=off_support,
                sha256=record['sha256']))
    tests = runpy.run_path(str(ROOT/'tests/test_argn_fraud_audit.py'))
    passed = []
    for name, fn in tests.items():
        if name.startswith('test_'): fn(); passed.append(name)
    write(DOCS/'verification.json', dict(passed_tests=passed, generated_datasets=rows,
        official_sources_unchanged=True, legacy_generations_unchanged=True,
        generated_labels_counted_directly=True, joint_tv_checked_with_counter=True,
        final_test_outcomes_accessed=False, quality_or_novelty_proven=False))
    print('VERIFIED', len(rows), 'generated datasets;',len(passed),'metric tests')


if __name__ == '__main__': main()
