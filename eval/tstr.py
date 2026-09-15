"""
TSTR (Train on Synthetic, Test on Real) evaluation harness — v1.

v1 additions:
  - extract_behavioral_features(): windowed soft_g features per entity.
    Measures velocity, gap, repetition, amt_sum — the behavioral signal
    CoF-SeqGen is trained to preserve. Needed to measure coherence utility.
  - Synthetic label = model-generated (y_gen from label head), not real copy.

Protocol:
  1. Extract tabular features from sequences (real and/or synthetic)
  2. Train HistGradientBoosting (HGB) classifier on training features+labels
  3. Evaluate AUPRC on real entity-disjoint test set
  4. ΔAUPRC = AUPRC(aug) − AUPRC(scarce_real)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — basic tabular
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    x_num: np.ndarray,              # (N, L, d_num)
    dt_bin: np.ndarray,             # (N, L)
    mask: np.ndarray,               # (N, L) bool
    y: np.ndarray,                  # (N, L)
    x_cat: Optional[np.ndarray] = None,  # (N, L, d_cat) int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate sequences to per-entity tabular features.
    Numerical: [mean, std, max, min] × d_num
    Time bins: [mean, max]
    Categorical: [n_unique, entropy] × d_cat
    Sequence length
    """
    N, L, d_num = x_num.shape

    x_masked  = x_num.astype(np.float32).copy()
    dt_masked = dt_bin.astype(np.float32).copy()
    bool_mask = mask.astype(bool)
    for i in range(N):
        if not bool_mask[i].any():
            continue
        x_masked[i, ~bool_mask[i]]  = np.nan
        dt_masked[i, ~bool_mask[i]] = np.nan

    feats = []
    for c in range(d_num):
        col = x_masked[:, :, c]
        feats.append(_nanstat(col, "mean"))
        feats.append(_nanstat(col, "std"))
        feats.append(_nanstat(col, "max"))
        feats.append(_nanstat(col, "min"))

    feats.append(_nanstat(dt_masked, "mean"))
    feats.append(_nanstat(dt_masked, "max"))

    if x_cat is not None:
        d_cat = x_cat.shape[2]
        for c in range(d_cat):
            n_unique = np.zeros(N, dtype=np.float32)
            entropy  = np.zeros(N, dtype=np.float32)
            for i in range(N):
                vals = x_cat[i, bool_mask[i], c]
                if len(vals) == 0:
                    continue
                _, counts = np.unique(vals, return_counts=True)
                n_unique[i] = len(counts)
                p = counts / counts.sum()
                entropy[i]  = float(-np.sum(p * np.log(p + 1e-10)))
            feats.append(n_unique)
            feats.append(entropy)

    feats.append(mask.sum(axis=1).astype(np.float32))

    X     = np.stack(feats, axis=1)
    X     = np.nan_to_num(X, nan=0.0)
    y_seq = ((y * mask).max(axis=1) > 0).astype(np.float32)
    return X, y_seq


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — windowed behavioral (soft_g per entity)
# ─────────────────────────────────────────────────────────────────────────────

def extract_behavioral_features(
    x_num: np.ndarray,              # (N, L, d_num)
    dt_bin: np.ndarray,             # (N, L) int
    x_cat: np.ndarray,              # (N, L, d_cat) int
    mask: np.ndarray,               # (N, L) bool
    tau: np.ndarray,                # (Bbins,) float — bin median times
    W: float,
    temp: float,
    Bbins: int,
    n_cat_classes: List[int],
    batch_size: int = 512,
    device: str = "cpu",            # "cuda" for ~10× speedup with large n_cat
) -> np.ndarray:
    """
    Compute windowed behavioral features [vel, gap, rep, amt_sum] per entity.
    Uses soft_g with real dt_bin one-hot (no gradient needed).

    Returns (N, 8): [mean, max] × 4 behavioral channels.
    These measure whether CoF-generated sequences preserve behavioral structure.
    """
    from models.soft_g import soft_g as _soft_g

    tau_t = torch.tensor(tau, dtype=torch.float32).to(device)
    N, L, _ = x_num.shape
    g_list: List[np.ndarray] = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xn  = torch.from_numpy(x_num[start:end]).float().to(device)
        db  = torch.from_numpy(dt_bin[start:end]).long().to(device)
        xc  = torch.from_numpy(x_cat[start:end]).long().to(device)

        with torch.no_grad():
            bp = F.one_hot(db.clamp(0, Bbins - 1), num_classes=Bbins).float()
            if n_cat_classes and xc.shape[-1] > 0:
                K0 = n_cat_classes[0]
                cp = F.one_hot(xc[:, :, 0].clamp(0, K0 - 1), num_classes=K0).float()
            else:
                cp = None
            amt = xn[:, :, 0]
            g_b = _soft_g(bp, cp, amt, tau_t, W, temp)   # (bs, L, 4)

        g_list.append(g_b.cpu().numpy())

    g_all  = np.concatenate(g_list, axis=0)   # (N, L, 4)
    bool_m = mask.astype(bool)                  # (N, L)

    feats = []
    for ch in range(4):   # vel, gap, rep, amt_sum
        col = np.where(bool_m, g_all[:, :, ch], np.nan)
        feats.append(_nanstat(col, "mean"))
        feats.append(_nanstat(col, "max"))

    return np.stack(feats, axis=1)   # (N, 8)


def extract_features_with_behavioral(
    x_num: np.ndarray,
    dt_bin: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    x_cat: Optional[np.ndarray],
    tau: Optional[np.ndarray] = None,
    W: Optional[float] = None,
    temp: Optional[float] = None,
    Bbins: Optional[int] = None,
    n_cat_classes: Optional[List[int]] = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    extract_features() + optional windowed behavioral features concatenated.
    If soft_g params (tau, W, temp, Bbins, n_cat_classes) are provided,
    appends 8 behavioral columns (vel/gap/rep/amt_sum × mean/max).
    Pass device="cuda" to run soft_g on GPU (~10× faster for X_synth).
    """
    X_base, y_seq = extract_features(x_num, dt_bin, mask, y, x_cat)

    if tau is not None and W is not None and x_cat is not None:
        X_beh = extract_behavioral_features(
            x_num, dt_bin, x_cat, mask, tau, W, temp, Bbins, n_cat_classes,
            device=device,
        )
        X = np.concatenate([X_base, X_beh], axis=1)
    else:
        X = X_base

    return X, y_seq


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def _nanstat(arr: np.ndarray, stat: str) -> np.ndarray:
    """(N, L) → (N,) ignoring NaN."""
    fn = {"mean": np.nanmean, "std": np.nanstd,
          "max":  np.nanmax,  "min": np.nanmin}[stat]
    with np.errstate(all="ignore"):
        result = fn(arr, axis=1)
    return np.nan_to_num(result, nan=0.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TSTR evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_tstr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 0,
) -> Dict[str, float]:
    """Train HGB on (X_train, y_train), evaluate AUPRC/AUROC on test."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    clf = HistGradientBoostingClassifier(
        max_iter=200, max_leaf_nodes=31, learning_rate=0.1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    n_pos  = int(y_test.sum())

    if n_pos == 0:
        return {"auprc": 0.0, "auroc": 0.5, "n_pos_test": 0}

    return {
        "auprc":      average_precision_score(y_test, y_prob),
        "auroc":      roc_auc_score(y_test, y_prob),
        "n_pos_test": n_pos,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Coherence gap
# ─────────────────────────────────────────────────────────────────────────────

def coherence_gap(
    g_synth: np.ndarray,    # (N, 4) behavioral features of synthetic sequences
    y_synth: np.ndarray,    # (N,) fraud labels (model-generated in v1)
    g_real: np.ndarray,     # (N_real, 4)
    y_real: np.ndarray,     # (N_real,)
    n_bins: int = 2,
    feat_idx: int = 0,      # which g component (0=vel, 1=gap, 2=rep, 3=amt_sum)
) -> float:
    """
    coherence_gap = mean_k |P(fraud|g_bin=k)_synth − P(fraud|g_bin=k)_real|.
    Small gap means synthetic behavior-fraud coupling matches real data.
    """
    thresholds = np.quantile(g_real[:, feat_idx], np.linspace(0, 1, n_bins + 1)[1:-1])
    gaps = []
    for lo, hi in zip([-np.inf] + list(thresholds), list(thresholds) + [np.inf]):
        mask_s = (g_synth[:, feat_idx] > lo) & (g_synth[:, feat_idx] <= hi)
        mask_r = (g_real[:, feat_idx] > lo)  & (g_real[:, feat_idx] <= hi)
        p_s = y_synth[mask_s].mean() if mask_s.sum() > 0 else 0.0
        p_r = y_real[mask_r].mean()  if mask_r.sum() > 0 else 0.0
        gaps.append(abs(p_s - p_r))
    return float(np.mean(gaps))


# ─────────────────────────────────────────────────────────────────────────────
# Stratified scarce split
# ─────────────────────────────────────────────────────────────────────────────

def stratified_keep(
    N: int,
    y_seq: np.ndarray,
    keep: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """Stratified keep: preserves fraud prevalence in scarce split."""
    rng     = np.random.default_rng(seed)
    idx_pos = np.where(y_seq > 0)[0]
    idx_neg = np.where(y_seq == 0)[0]

    n_pos = max(1, int(len(idx_pos) * keep))
    n_neg = max(1, int(len(idx_neg) * keep))

    sel_pos = rng.choice(idx_pos, size=n_pos, replace=False)
    sel_neg = rng.choice(idx_neg, size=n_neg, replace=False)
    return np.concatenate([sel_pos, sel_neg])
