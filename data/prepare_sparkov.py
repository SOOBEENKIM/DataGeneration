"""
Sparkov (Kaggle fraud-detection) raw CSV → cleaned parquet.

Keep: cc_num, unix_time, amt, category, is_fraud
Drop: merchant, personal info, identifiers
Output: data/sparkov/raw.parquet
"""

import argparse
import pandas as pd
from pathlib import Path

DROP_COLS = [
    'Unnamed: 0', 'trans_date_trans_time', 'merchant',
    'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
    'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num',
    'merch_lat', 'merch_long',
]
KEEP_COLS = ['cc_num', 'unix_time', 'amt', 'category', 'is_fraud']


def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for p in args.raw:
        df = pd.read_csv(p, low_memory=False)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"Combined rows: {len(df):,}  entities: {df['cc_num'].nunique()}")

    df = df[KEEP_COLS].copy()

    # Sort within entity by time
    df = df.sort_values(['cc_num', 'unix_time']).reset_index(drop=True)

    assert df['amt'].isna().sum() == 0, "NaN in amt"
    assert df['category'].isna().sum() == 0, "NaN in category"
    assert df['unix_time'].isna().sum() == 0, "NaN in unix_time"

    out = out_dir / 'raw.parquet'
    df.to_parquet(out, index=False)
    print(f"Saved {out}  shape={df.shape}")
    print(f"  fraud rate: {df['is_fraud'].mean():.4f}")
    print(f"  category unique: {df['category'].nunique()}")
    print(f"  amt range: {df['amt'].min():.2f} – {df['amt'].max():.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', nargs='+', required=True,
                        help='fraudTrain.csv and/or fraudTest.csv paths')
    parser.add_argument('--out', default='data/sparkov')
    main(parser.parse_args())
