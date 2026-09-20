"""Verify the four registered saved runs without training or changing data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_cs_saf_external_audit_v1 import CONTRACT, OUT, digest, write


def main():
    contract = json.loads(CONTRACT.read_text())
    records = []
    inputs = []
    for kappa in contract['kappas']:
        inp = OUT / f'kappa_{kappa}/input'
        provenance = json.loads((inp / 'provenance.json').read_text())
        for name, expected in provenance['files'].items():
            assert digest(inp / name) == expected, name
        source = ROOT / provenance['source']
        manifest = json.loads((source / 'cs_saf_manifest.json').read_text())
        assert manifest['content_splits'] == ['train', 'validation']
        assert digest(source / 'cs_saf_manifest.json') == provenance['manifest_sha256']
        for name, expected in manifest['files'].items():
            assert digest(source / name) == expected, name
        canonical = pd.read_parquet(source / 'events.parquet')
        assert np.array_equal(canonical.event_index, canonical.groupby('entity_id').cumcount())
        train = pd.read_parquet(inp / 'train_events.parquet')
        parent = pd.read_parquet(inp / 'train_context.parquet')
        validation_parent = pd.read_parquet(inp / 'validation_context.parquet')
        assert not set(parent.entity_id) & set(validation_parent.entity_id)
        expected = canonical[canonical.entity_id.isin(parent.entity_id)].copy()
        expected.loc[expected.event_index == 0, 'gap'] = 0.0
        expected = expected.sort_values('entity_id', kind='stable')[train.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(train, expected)
        valid = train.groupby('entity_id').cumcount() > 0
        repeat = train.receiver_or_mark.eq(train.groupby('entity_id').receiver_or_mark.shift()).fillna(False)
        train_rate = float(repeat[valid].mean())
        labels = set(train.receiver_or_mark)
        plan = pd.read_parquet(inp / 'plan.parquet')
        assert set(plan.source_train_entity_id) <= set(parent.entity_id)
        inputs.append(dict(kappa=kappa, source=str(source.relative_to(ROOT)),
                           source_and_adapter_hashes_match=True, train_validation_disjoint=True,
                           event_order_and_values_preserved=True, train_repeat_rate=train_rate,
                           first_gap_placeholder_only=True, test_content_present=False))
        for seed in contract['training_seeds']:
            folder = OUT / f'kappa_{kappa}/seed_{seed}'
            done = json.loads((folder / 'DONE.json').read_text())
            start = json.loads((folder / 'start.json').read_text())
            checks = json.loads((folder / 'replay_checks.json').read_text())
            assert start['contract_sha256'] == digest(CONTRACT)
            assert not done['smoke'] and checks['weights_unchanged']
            weight = folder / 'workspace/ModelStore/model-data/model-weights.pt'
            if not weight.exists():
                candidates = list((folder / 'workspace/ModelStore/model-data').glob('model-weights.*'))
                assert len(candidates) == 1, candidates
                weight = candidates[0]
            assert digest(weight) == done['weights_sha256']
            count = matches = 0
            encoded_files = sorted((folder / 'workspace/OriginalData/encoded-data').glob('*.parquet'))
            assert len(encoded_files) == 2
            for file in encoded_files:
                encoded = pd.read_parquet(file, columns=['tgt:t1/c1__cat'])
                for codes in encoded.iloc[:, 0]:
                    assert codes[-1] == 0 and (codes[:-1] > 0).all()
                    codes = codes[:-1]  # structural end-of-sequence code
                    matches += int(np.sum(codes[1:] == codes[:-1]))
                    count += len(codes) - 1
            encoded_rate = matches / count
            assert abs(encoded_rate - train_rate) < 1e-15
            samples = []
            for tape in contract['generation_seeds']:
                file = folder / f'generated_{tape}.parquet'
                generated = pd.read_parquet(file)
                assert set(generated.entity_id) == set(plan.entity_id)
                assert not (set(generated.receiver_or_mark) - labels)
                nonfirst = generated.groupby('entity_id').cumcount() > 0
                gap = generated.loc[nonfirst, 'gap'].to_numpy(float)
                numeric = generated.amount_or_numeric_value.to_numpy(float)
                assert np.isfinite(gap).all() and (gap >= 0).all()
                assert np.isfinite(numeric).all()
                samples.append(dict(tape=tape, entities=generated.entity_id.nunique(),
                                    events=len(generated), sha256=digest(file)))
            assert max(v['max_current_target_future_or_prefix_difference']
                       for v in checks['checks'].values()) < 1e-5
            small_files = ['start.json', 'fit_complete.json', 'DONE.json', 'replay_checks.json']
            records.append(dict(kappa=kappa, seed=seed, path=str(folder.relative_to(ROOT)),
                                weights_sha256=digest(weight), weights_unchanged=True,
                                encoded_all_outer_train_repeat_rate=encoded_rate,
                                encoded_repeat_preserved=True, start=start, done=done,
                                generated=samples,
                                metadata_sha256={name: digest(folder / name) for name in small_files}))
    write(ROOT / 'docs/cs_saf/external_audit_v1/artifact_inventory.json',
          dict(status='VERIFIED', contract_sha256=digest(CONTRACT), inputs=inputs, runs=records,
               raw_data_and_checkpoints_location='artifacts/cs_saf/external_audit_v1 (workstation, git-ignored)',
               raw_checkpoints_uploaded_to_git=False))
    print('VERIFIED: four weights, 20 generated datasets, input identity/order, encoded repetition, replay invariance')


if __name__ == '__main__':
    main()
