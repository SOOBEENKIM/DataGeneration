"""
AMLSim raw CSV → cleaned parquet with graph features.

Entity: SENDER_ACCOUNT_ID, Time: TIMESTAMP (integer steps 0-199)
Keep: SENDER_ACCOUNT_ID, RECEIVER_ACCOUNT_ID, TX_AMOUNT(log1p), TIMESTAMP, IS_FRAUD
Drop: TX_TYPE (constant='TRANSFER'), TX_ID, ALERT_ID

Graph features (trailing, receiver-based, strict causal m<j):
  fan_in_recv  = # of distinct senders to RECEIVER in [t-W, t)
  fan_out_send = # of distinct receivers from SENDER in [t-W, t)
  W = 10 steps (trailing window for graph signal)

Output: data/amlsim/raw.parquet
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

GRAPH_W = 10  # trailing window for graph features


def compute_graph_features(df: pd.DataFrame, W: int) -> pd.DataFrame:
    """
    Vectorized fan_in / fan_out with strict causal window (m < j, i.e. t_m < t_j).

    TIMESTAMP is integer steps 0-199. Strategy: iterate over unique steps (200 iters),
    for each step t build a window [t-W, t) from the full df, then join.
    O(T * W * avg_tx_per_step) where T=200, W=10, avg=6616 → ~13M ops, fast.
    """
    df = df.sort_values('TIMESTAMP').reset_index(drop=True)
    unique_steps = sorted(df['TIMESTAMP'].unique())

    # Pre-index by timestamp for fast lookup
    step_groups = df.groupby('TIMESTAMP')

    fan_in_map = {}   # (receiver, t) -> distinct_senders_count
    fan_out_map = {}  # (sender, t) -> distinct_receivers_count

    for t in unique_steps:
        t_lo = t - W
        # window: strictly t_m < t  →  TIMESTAMP in [t_lo, t-1]
        window_steps = [s for s in unique_steps if t_lo <= s < t]
        if not window_steps:
            # No prior transactions → fan_in=fan_out=0 for all rows at step t
            continue

        # Gather window transactions
        window_dfs = [step_groups.get_group(s) for s in window_steps if s in step_groups.groups]
        if not window_dfs:
            continue
        w_df = pd.concat(window_dfs, ignore_index=True)

        # Causal assertion: all window timestamps strictly < t
        assert (w_df['TIMESTAMP'] < t).all(), f"causal violation at step {t}"

        # fan_in at step t: for each receiver r, #distinct senders in window
        fi = w_df.groupby('RECEIVER_ACCOUNT_ID')['SENDER_ACCOUNT_ID'].nunique()
        for recv, cnt in fi.items():
            fan_in_map[(recv, t)] = cnt

        # fan_out at step t: for each sender s, #distinct receivers in window
        fo = w_df.groupby('SENDER_ACCOUNT_ID')['RECEIVER_ACCOUNT_ID'].nunique()
        for send, cnt in fo.items():
            fan_out_map[(send, t)] = cnt

    # Merge fan_in back (vectorized, no apply)
    if fan_in_map:
        fi_df = pd.DataFrame(
            [(r, t, c) for (r, t), c in fan_in_map.items()],
            columns=['RECEIVER_ACCOUNT_ID', 'TIMESTAMP', 'fan_in']
        )
        df = df.merge(fi_df, on=['RECEIVER_ACCOUNT_ID', 'TIMESTAMP'], how='left')
    else:
        df['fan_in'] = 0.0
    df['fan_in'] = df['fan_in'].fillna(0).astype(np.float32)

    # Merge fan_out back
    if fan_out_map:
        fo_df = pd.DataFrame(
            [(s, t, c) for (s, t), c in fan_out_map.items()],
            columns=['SENDER_ACCOUNT_ID', 'TIMESTAMP', 'fan_out']
        )
        df = df.merge(fo_df, on=['SENDER_ACCOUNT_ID', 'TIMESTAMP'], how='left')
    else:
        df['fan_out'] = 0.0
    df['fan_out'] = df['fan_out'].fillna(0).astype(np.float32)

    return df


def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.raw, low_memory=False)
    print(f"Raw rows: {len(df):,}  TX_TYPE unique: {df['TX_TYPE'].unique()}")

    # TX_TYPE is constant → drop
    assert df['TX_TYPE'].nunique() == 1, "TX_TYPE is no longer constant — revisit"
    df = df.drop(columns=['TX_ID', 'TX_TYPE', 'ALERT_ID'])

    df['IS_FRAUD'] = df['IS_FRAUD'].astype(int)
    df['TX_AMOUNT_LOG'] = np.log1p(df['TX_AMOUNT'])
    df = df.drop(columns=['TX_AMOUNT'])

    print("Computing graph features (fan_in / fan_out) — may take a minute...")
    df = compute_graph_features(df, W=args.graph_w)

    df = df.sort_values(['SENDER_ACCOUNT_ID', 'TIMESTAMP']).reset_index(drop=True)

    out = out_dir / 'raw.parquet'
    df.to_parquet(out, index=False)
    print(f"Saved {out}  shape={df.shape}")
    print(f"  fraud rate: {df['IS_FRAUD'].mean():.4f}")
    print(f"  entities (senders): {df['SENDER_ACCOUNT_ID'].nunique()}")
    print(f"  TIMESTAMP range: {df['TIMESTAMP'].min()} – {df['TIMESTAMP'].max()}")
    print(f"  fan_in  range: {df['fan_in'].min():.0f} – {df['fan_in'].max():.0f}")
    print(f"  fan_out range: {df['fan_out'].min():.0f} – {df['fan_out'].max():.0f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True, help='transactions.csv path')
    parser.add_argument('--out', default='data/amlsim')
    parser.add_argument('--graph_w', type=int, default=GRAPH_W)
    main(parser.parse_args())
