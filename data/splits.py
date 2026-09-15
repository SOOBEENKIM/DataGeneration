"""
Entity-disjoint split: 70/10/20 train/val/test.

Splits on unique entity IDs (cc_num for Sparkov, SENDER_ACCOUNT_ID for AMLSim).
Oversampling is FORBIDDEN before calling this — only allowed after split.
Cached to parquet so splits are reproducible.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


ENTITY_COL = {
    'sparkov': 'cc_num',
    'amlsim': 'SENDER_ACCOUNT_ID',
}


def entity_disjoint_split(df: pd.DataFrame, entity_col: str,
                          train_frac=0.70, val_frac=0.10, seed=42):
    entities = df[entity_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(entities)

    n = len(entities)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ents = set(entities[:n_train])
    val_ents = set(entities[n_train:n_train + n_val])
    test_ents = set(entities[n_train + n_val:])

    # Verify disjoint
    assert len(train_ents & val_ents) == 0, "train/val entity overlap"
    assert len(train_ents & test_ents) == 0, "train/test entity overlap"
    assert len(val_ents & test_ents) == 0, "val/test entity overlap"
    assert len(train_ents | val_ents | test_ents) == n

    train_df = df[df[entity_col].isin(train_ents)].copy()
    val_df = df[df[entity_col].isin(val_ents)].copy()
    test_df = df[df[entity_col].isin(test_ents)].copy()

    return train_df, val_df, test_df


def main(args):
    dataset = args.dataset
    data_dir = Path(args.data_dir) / dataset
    out_dir = data_dir / 'splits'
    out_dir.mkdir(parents=True, exist_ok=True)

    entity_col = ENTITY_COL[dataset]

    df = pd.read_parquet(data_dir / 'raw.parquet')
    print(f"Loaded {len(df):,} rows, {df[entity_col].nunique()} entities")

    train_df, val_df, test_df = entity_disjoint_split(
        df, entity_col, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
    )

    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        out = out_dir / f'{split_name}.parquet'
        split_df.to_parquet(out, index=False)
        fraud_col = 'is_fraud' if dataset == 'sparkov' else 'IS_FRAUD'
        print(f"  {split_name}: {len(split_df):,} rows, "
              f"{split_df[entity_col].nunique()} entities, "
              f"fraud={split_df[fraud_col].mean():.4f}")

    # Gate 1 check: zero entity overlap
    tr_ents = set(train_df[entity_col])
    va_ents = set(val_df[entity_col])
    te_ents = set(test_df[entity_col])
    assert len(tr_ents & te_ents) == 0, "GATE FAIL: train/test entity overlap"
    assert len(tr_ents & va_ents) == 0, "GATE FAIL: train/val entity overlap"
    assert len(va_ents & te_ents) == 0, "GATE FAIL: val/test entity overlap"
    print("Gate 1 entity-disjoint check: PASS (overlap = 0)")

    meta = {
        'dataset': dataset,
        'entity_col': entity_col,
        'train_entities': len(tr_ents),
        'val_entities': len(va_ents),
        'test_entities': len(te_ents),
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
        'seed': args.seed,
    }
    json.dump(meta, open(out_dir / 'split_meta.json', 'w'), indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['sparkov', 'amlsim'])
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--train_frac', type=float, default=0.70)
    parser.add_argument('--val_frac', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
