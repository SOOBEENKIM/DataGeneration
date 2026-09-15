"""
Build entity-level sequences for CoF-SeqGen.

Steps:
  1. Load split parquets (train/val/test)
  2. Compute Δt (minutes for Sparkov, steps for AMLSim)
  3. Fit quantile time-bins B=16 on TRAIN Δt only
  4. Encode categorical features (label-encoding on TRAIN only)
  5. Standardize numerical features on TRAIN only
  6. Slide L-length windows over each entity's sorted sequence (non-overlapping)
     Short entities (<L): left-pad + mask
  7. Save npz arrays + metadata (encoders, bin_edges, tau_k)

Output per dataset: data/{dataset}/sequences/{split}.npz
  Arrays: x_num (W,L,d_num), x_cat_idx (W,L,d_cat), y (W,L), mask (W,L), entity_id (W,)
  d_num: [amt_std, dt_bin_float] + [fan_in, fan_out] for amlsim
  d_cat: [category_encoded] for sparkov, [] for amlsim (TX_TYPE dropped)

Metadata: data/{dataset}/sequences/meta.json + encoders.pkl
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ─── Dataset-specific config ────────────────────────────────────────────────

CONFIGS = {
    'sparkov': {
        'entity_col': 'cc_num',
        'time_col': 'unix_time',
        'time_unit': 'minutes',   # Δt = diff(unix_time) / 60
        'label_col': 'is_fraud',
        'num_cols': ['amt'],
        'cat_cols': ['category'],
        'extra_num_cols': [],     # no graph features
        'cat_topk': {},           # no top-K bucketing
        'L': 24,
        'B': 16,
    },
    'amlsim': {
        'entity_col': 'SENDER_ACCOUNT_ID',
        'time_col': 'TIMESTAMP',
        'time_unit': 'steps',     # Δt = diff(TIMESTAMP) as-is
        'label_col': 'IS_FRAUD',
        'num_cols': ['TX_AMOUNT_LOG'],
        'cat_cols': ['RECEIVER_ACCOUNT_ID'],  # fan-out proxy via soft_g rep channel
        'extra_num_cols': [],                 # fan_in/fan_out dropped; soft_g computes g
        'cat_hash': {'RECEIVER_ACCOUNT_ID': 256},  # hash(recv) % 256, preserves fan-out
        'L': 32,
        'B': 16,
    },
}


# ─── Top-K categorical encoder ───────────────────────────────────────────────

class HashEncoder:
    """
    Encode a high-cardinality categorical via hash(value) % n_buckets.

    Unlike top-K+OTHER, hash bucketing preserves diversity:
    - AML fan-out = "many distinct rare receivers" → each gets its own bucket
    - top-K+OTHER collapses ~80% of AMLSim receivers into one bucket,
      destroying the fan-out signal that soft_g's rep channel relies on.
    - Hash distributes rare values across all buckets → soft_g sees N distinct
      buckets ≈ N distinct receivers, preserving the fan-out count.

    n_buckets: vocabulary size passed to the model's embedding table.
    No fit() needed (stateless); fit() is a no-op for API compatibility.
    """

    def __init__(self, n_buckets: int):
        self.n_buckets = n_buckets

    def fit(self, series: pd.Series):
        return self   # stateless

    def transform(self, series: pd.Series) -> np.ndarray:
        return series.apply(lambda x: int(x) % self.n_buckets).values.astype(np.int32)

    @property
    def n_classes(self):
        return self.n_buckets


# ─── Core helpers ────────────────────────────────────────────────────────────

def compute_delta_t(series: pd.Series, unit: str) -> np.ndarray:
    """First transaction Δt = 0; rest = diff."""
    arr = series.values.astype(float)
    dt = np.zeros(len(arr), dtype=np.float32)
    dt[1:] = np.diff(arr)
    if unit == 'minutes':
        dt[1:] = dt[1:] / 60.0
    dt[dt < 0] = 0.0  # guard: should not happen after sort
    return dt


def fit_time_bins(dt_train_flat: np.ndarray, B: int):
    """Fit quantile bin edges on train Δt (exclude the first-tx zeros)."""
    nonzero = dt_train_flat[dt_train_flat > 0]
    quantiles = np.linspace(0, 100, B + 1)
    edges = np.percentile(nonzero, quantiles)
    edges[0] = 0.0  # include 0 in first bin
    edges[-1] = np.inf
    # τ_k = median of train Δt within each bin
    tau = np.zeros(B, dtype=np.float32)
    for k in range(B):
        mask = (dt_train_flat >= edges[k]) & (dt_train_flat < edges[k + 1])
        vals = dt_train_flat[mask]
        tau[k] = float(np.median(vals)) if len(vals) > 0 else 0.0
    return edges, tau


def apply_time_bins(dt: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Convert continuous Δt to bin index (0-indexed)."""
    bins = np.digitize(dt, edges[1:], right=False)  # 0..B-1
    bins = np.clip(bins, 0, len(edges) - 2)
    return bins.astype(np.int32)


def make_windows(entity_df: pd.DataFrame, cfg: dict,
                 scaler: StandardScaler, cat_encoders: dict,
                 bin_edges: np.ndarray, tau: np.ndarray,
                 is_train: bool):
    """
    Slide non-overlapping windows of length L over an entity's sorted sequence.
    Short sequences: left-pad with zeros + mask=False.
    Returns list of dicts with keys: x_num, x_cat, y, mask
    """
    L = cfg['L']
    num_cols = cfg['num_cols'] + cfg['extra_num_cols']
    cat_cols = cfg['cat_cols']

    # Time features
    dt = compute_delta_t(entity_df[cfg['time_col']], cfg['time_unit'])
    dt_bin = apply_time_bins(dt, bin_edges)

    # Numerical: standardize
    raw_num = entity_df[num_cols].values.astype(np.float32)
    raw_num_std = scaler.transform(raw_num)

    # Categorical: label-encoded or hash-encoded
    cat_encoded = np.zeros((len(entity_df), len(cat_cols)), dtype=np.int32)
    for i, col in enumerate(cat_cols):
        enc = cat_encoders[col]
        if isinstance(enc, HashEncoder):
            cat_encoded[:, i] = enc.transform(entity_df[col])
        else:
            cat_encoded[:, i] = enc.transform(entity_df[col].values)

    label = entity_df[cfg['label_col']].values.astype(np.int32)

    # x_num = [std_numericals..., dt_bin_as_float (for embedding lookup)]
    # We store dt_bin separately; x_num is only continuous numericals
    n = len(entity_df)

    windows = []
    # Non-overlapping: stride = L
    starts = list(range(0, n, L))
    if len(starts) == 0:
        starts = [0]

    for s in starts:
        e = s + L
        seg_len = min(e, n) - s

        # Allocate padded arrays
        x_num_w = np.zeros((L, len(num_cols)), dtype=np.float32)
        dt_bin_w = np.zeros(L, dtype=np.int32)
        x_cat_w = np.zeros((L, len(cat_cols)), dtype=np.int32)
        y_w = np.zeros(L, dtype=np.int32)
        mask_w = np.zeros(L, dtype=bool)  # True = real, False = pad

        # Fill right-aligned (pad on left)
        pad = L - seg_len
        real_slice = slice(s, s + seg_len)
        dest_slice = slice(pad, L)

        x_num_w[dest_slice] = raw_num_std[real_slice]
        dt_bin_w[dest_slice] = dt_bin[real_slice]
        if len(cat_cols) > 0:
            x_cat_w[dest_slice] = cat_encoded[real_slice]
        y_w[dest_slice] = label[real_slice]
        mask_w[dest_slice] = True

        windows.append({
            'x_num': x_num_w,      # (L, d_num)
            'dt_bin': dt_bin_w,    # (L,) int — time-bin index
            'x_cat': x_cat_w,      # (L, d_cat)
            'y': y_w,              # (L,) int
            'mask': mask_w,        # (L,) bool
        })

    return windows


def build_split(split_df: pd.DataFrame, cfg: dict, entity_col: str,
                scaler, cat_encoders, bin_edges, tau):
    all_windows = []
    entity_ids = []

    for ent, grp in split_df.groupby(entity_col, sort=False):
        grp = grp.sort_values(cfg['time_col'])
        wins = make_windows(grp, cfg, scaler, cat_encoders, bin_edges, tau, is_train=False)
        all_windows.extend(wins)
        entity_ids.extend([ent] * len(wins))

    W = len(all_windows)
    L = cfg['L']
    d_num = len(cfg['num_cols']) + len(cfg['extra_num_cols'])
    d_cat = len(cfg['cat_cols'])

    x_num = np.stack([w['x_num'] for w in all_windows])     # (W, L, d_num)
    dt_bin = np.stack([w['dt_bin'] for w in all_windows])   # (W, L)
    x_cat = np.stack([w['x_cat'] for w in all_windows])     # (W, L, d_cat)
    y = np.stack([w['y'] for w in all_windows])              # (W, L)
    mask = np.stack([w['mask'] for w in all_windows])        # (W, L) bool

    return {
        'x_num': x_num, 'dt_bin': dt_bin, 'x_cat': x_cat,
        'y': y, 'mask': mask,
        'entity_id': np.array(entity_ids),
    }


# ─── Gate 1 sanity checks ────────────────────────────────────────────────────

def sanity_checks(splits: dict, entity_col_key: str, cfg: dict, bin_edges, tau):
    tr_ents = set(splits['train']['entity_id'])
    va_ents = set(splits['val']['entity_id'])
    te_ents = set(splits['test']['entity_id'])

    assert len(tr_ents & te_ents) == 0, "GATE FAIL: train/test entity overlap"
    assert len(tr_ents & va_ents) == 0, "GATE FAIL: train/val entity overlap"
    print("Entity disjoint: PASS (overlap = 0)")

    for split_name, data in splits.items():
        seq_len = data['mask'].sum(axis=1)
        nan_check = np.isnan(data['x_num']).sum()
        inf_check = np.isinf(data['x_num']).sum()
        assert nan_check == 0, f"NaN in x_num ({split_name})"
        assert inf_check == 0, f"Inf in x_num ({split_name})"
        print(f"  {split_name}: windows={len(data['y'])}, "
              f"seq_len median={np.median(seq_len):.0f}, "
              f"fraud_rate={data['y'].mean():.4f}  NaN/Inf: 0")

    B = cfg['B']
    assert len(bin_edges) == B + 1
    assert len(tau) == B
    print(f"Δt-bin edges: {bin_edges[:4]} ... {bin_edges[-2:]}")
    print(f"τ_k (first 4): {tau[:4]}")
    print("Gate 1 sanity: PASS")


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    dataset = args.dataset
    cfg = CONFIGS[dataset]
    L, B = cfg['L'], cfg['B']
    entity_col = cfg['entity_col']
    num_cols = cfg['num_cols'] + cfg['extra_num_cols']
    cat_cols = cfg['cat_cols']

    splits_dir = Path(args.data_dir) / dataset / 'splits'
    out_dir = Path(args.data_dir) / dataset / 'sequences'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building sequences for {dataset}: L={L}, B={B}")

    # Load splits
    train_df = pd.read_parquet(splits_dir / 'train.parquet')
    val_df = pd.read_parquet(splits_dir / 'val.parquet')
    test_df = pd.read_parquet(splits_dir / 'test.parquet')

    # ── Fit encoders on TRAIN only ──────────────────────────────────────────

    # 1. Δt from train, fit time bins
    dt_train_all = []
    for _, grp in train_df.groupby(entity_col, sort=False):
        grp = grp.sort_values(cfg['time_col'])
        dt = compute_delta_t(grp[cfg['time_col']], cfg['time_unit'])
        dt_train_all.append(dt)
    dt_train_flat = np.concatenate(dt_train_all)

    bin_edges, tau = fit_time_bins(dt_train_flat, B)
    print(f"Fitted time bins on {len(dt_train_flat):,} train Δt values")

    # 2. Numerical scaler (train only)
    scaler = StandardScaler()
    scaler.fit(train_df[num_cols].values.astype(np.float32))

    # 3. Categorical encoders (train only)
    cat_hash = cfg.get('cat_hash', {})
    cat_encoders = {}
    for col in cat_cols:
        if col in cat_hash:
            enc = HashEncoder(n_buckets=cat_hash[col])
            enc.fit(train_df[col])   # no-op
            cat_encoders[col] = enc
            print(f"  {col}: hash%{cat_hash[col]} = {enc.n_classes} buckets (fan-out preserved)")
        else:
            le = LabelEncoder()
            le.fit(train_df[col].values)
            cat_encoders[col] = le
            print(f"  {col}: {len(le.classes_)} classes: {list(le.classes_[:5])}")

    # ── Build windows for all splits ────────────────────────────────────────
    split_data = {}
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"Building {split_name} windows...")
        split_data[split_name] = build_split(
            split_df, cfg, entity_col, scaler, cat_encoders, bin_edges, tau
        )
        np.savez_compressed(
            out_dir / f'{split_name}.npz',
            **split_data[split_name]
        )
        print(f"  saved: {out_dir / split_name}.npz")

    # ── Gate 1 sanity checks ────────────────────────────────────────────────
    sanity_checks(split_data, entity_col, cfg, bin_edges, tau)

    # ── Save metadata ────────────────────────────────────────────────────────
    meta = {
        'dataset': dataset,
        'L': L, 'B': B,
        'entity_col': entity_col,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'label_col': cfg['label_col'],
        'time_col': cfg['time_col'],
        'time_unit': cfg['time_unit'],
        'bin_edges': bin_edges.tolist(),
        'tau_k': tau.tolist(),
        'num_classes_cat': {
            col: int(cat_encoders[col].n_classes
                     if isinstance(cat_encoders[col], HashEncoder)
                     else len(cat_encoders[col].classes_))
            for col in cat_cols
        },
        'd_num': len(num_cols),
        'd_cat': len(cat_cols),
    }
    json.dump(meta, open(out_dir / 'meta.json', 'w'), indent=2)

    with open(out_dir / 'encoders.pkl', 'wb') as f:
        pickle.dump({'scaler': scaler, 'cat_encoders': cat_encoders,
                     'bin_edges': bin_edges, 'tau': tau}, f)

    print(f"Metadata saved to {out_dir / 'meta.json'}")
    print("Phase 1 DONE.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['sparkov', 'amlsim'])
    parser.add_argument('--data_dir', default='data')
    main(parser.parse_args())
