"""
Controlled κ-coupling data generator for CoF-SeqGen.

Generates synthetic financial transaction sequences where fraud is determined
by a window-based behavioral rule (velocity + fanout), with coupling strength
controlled by κ ∈ [0, 1].

  κ = 0.0: labels are random — no behavioral coupling
  κ = 1.0: all fraud entities exhibit fraud-mode behavior (burst + concentrated)
  κ ∈ (0,1): fraction κ of fraud entities follow the rule

Behavioral design:
  Fraud-mode   : burst transactions to 3 concentrated receivers
                 short gaps → high velocity, low fanout (window rule signal)
  Non-fraud    : spread-out transactions to random receivers
                 long gaps  → lower velocity, higher fanout
  Amount       : deliberately WEAK (same log-normal for both → no marginal signal)

Window rule (what CoF should learn):
  fraud = (vel > threshold) AND (fanout < threshold)
  Both signals in vel + fanout channels → discriminative even for modest κ.

Output: data/kappa_{κ:.2f}/sequences/{train,test}.npz + meta.json
  Same format as AMLSim:
    x_num  (N, L, 1)   float32 — TX_AMOUNT_LOG
    dt_bin (N, L)      int32   — timing bin index
    x_cat  (N, L, 1)   int32   — receiver category
    y      (N, L)      float32 — position-level fraud label
    mask   (N, L)      bool    — True = valid position
  meta.json: L, B, tau_k, num_classes_cat, d_num, d_cat, kappa, ...

Usage:
  python scripts/generate_kappa_data.py --kappas 0.0 0.3 0.5 0.7 1.0 --seed 42
  python scripts/generate_kappa_data.py --kappas 1.0 --verify   # sanity check g-coupling
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Dataset parameters (match AMLSim dimensions where possible) ──────────────
N_TRAIN     = 31951   # match AMLSim train size
N_TEST      = 7989    # ~20% of (N_TRAIN + N_TEST)
L           = 32      # max sequence length
BBINS       = 16      # time bins (matches AMLSim B=16)
K_CAT       = 64      # receiver categories (fewer than AMLSim 256; clean fanout signal)
FRAUD_RATE  = 0.05    # entity-level fraud prevalence
D_NUM       = 1       # one numerical feature: amount_log

# ── Behavioral parameters ─────────────────────────────────────────────────────
#
# Design goal: fraud = HIGH vel + LOW fanout (window rule)
#   Both features must discriminate in the CORRECT direction.
#   Key constraint: non-fraud must have ENOUGH transactions per window so that
#   its receiver DIVERSITY is larger than fraud's (otherwise fanout inverts).
#
# Fraud-mode: burst (short gaps) to a SINGLE receiver (cat 0)
#   Many transactions per window → high vel
#   All to cat 0 → fanout = 1 (minimum possible)
FRAUD_GAP_SCALE  = 0.5    # Exp(0.5): mean gap = 0.5 → ~14 tx per W=7 window
FRAUD_N_CAT      = 1      # EXACTLY 1 receiver category (cat 0)
FRAUD_CAT_PROBS  = [1.0]  # 100% to cat 0 → fanout = 1 within any window

# Non-fraud: moderate gaps, diverse receivers
#   Fewer transactions per window (lower vel)
#   Many distinct receivers → fanout >> 1
NONFR_GAP_SCALE  = 2.0    # Exp(2.0): mean gap = 2.0 → ~3.5 tx per W=7 window
NONFR_CAT_MIN    = 1      # receivers chosen from [1, K_CAT) → avoids cat 0 (fraud cat)
                           # Expected distinct receivers per window ≈ 3.4 → fanout >> fraud

# Amount: weak signal (same log-normal for both)
AMT_LOG_MEAN = 5.0
AMT_LOG_STD  = 1.0

# Sequence length variation
MIN_SEQ_LEN  = L // 2    # 16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kappas",     type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 1.0])
    p.add_argument("--N_train",    type=int,   default=N_TRAIN)
    p.add_argument("--N_test",     type=int,   default=N_TEST)
    p.add_argument("--fraud_rate", type=float, default=FRAUD_RATE)
    p.add_argument("--W",          type=float, default=7.0,
                   help="Window width (in time units) for g computation")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--out_base",   type=str,   default=str(ROOT / "data"))
    p.add_argument("--verify",     action="store_true",
                   help="After generation, compute g and print P(fraud|g_bin) coupling check")
    return p.parse_args()


# ── Generation ────────────────────────────────────────────────────────────────

def generate_split(N, fraud_rate, kappa, seed):
    """
    Generate N entities.

    Returns:
      x_num_arr (N, L, 1) float32  — log-amounts
      x_cat_arr (N, L, 1) int32    — receiver categories
      gap_raw   (N, L)    float32  — actual inter-tx gaps (before binning)
      mask_arr  (N, L)    bool
      y_arr     (N, L)    float32  — position-level fraud (= entity label)
      y_ent     (N,)      int32    — entity fraud label
    """
    rng = np.random.default_rng(seed)

    # Entity labels
    y_ent = rng.binomial(1, fraud_rate, N).astype(np.int32)  # (N,)

    # κ-mixing: which fraud entities follow the window rule
    follow_rule = (rng.uniform(0, 1, N) < kappa) & (y_ent == 1)  # (N,)

    # Variable sequence lengths
    seq_lens = rng.integers(MIN_SEQ_LEN, L + 1, size=N)  # (N,) in [L//2, L]

    x_num_arr = np.zeros((N, L, D_NUM), dtype=np.float32)
    x_cat_arr = np.zeros((N, L, 1),     dtype=np.int32)
    gap_raw   = np.full((N, L),  NONFR_GAP_SCALE, dtype=np.float32)  # default non-fraud gap
    mask_arr  = np.zeros((N, L), dtype=bool)
    y_arr     = np.zeros((N, L), dtype=np.float32)

    for i in range(N):
        sl = seq_lens[i]
        mask_arr[i, :sl] = True
        y_arr[i, :sl]    = float(y_ent[i])

        if follow_rule[i]:
            # ── Fraud-mode: burst to concentrated receiver ────────────────
            # Short gaps: many transactions per window → high velocity
            gaps_i = rng.exponential(FRAUD_GAP_SCALE, sl)
            # Concentrated receivers: single category (cat 0) → fanout = 1
            cat_i  = np.zeros(sl, dtype=np.int32)   # always cat 0
        else:
            # ── Non-fraud mode: spread out, diverse receivers ─────────────
            gaps_i = rng.exponential(NONFR_GAP_SCALE, sl)
            # Receivers from [1, K_CAT) → avoids cat 0 (fraud cat)
            cat_i  = rng.integers(NONFR_CAT_MIN, K_CAT, sl)

        # First transaction: gap from "account opening" = mean non-fraud gap
        gaps_i[0] = NONFR_GAP_SCALE

        # Amount: same distribution regardless of fraud label (weak signal)
        amt_i = rng.lognormal(AMT_LOG_MEAN, AMT_LOG_STD, sl)

        x_num_arr[i, :sl, 0] = np.log(np.maximum(amt_i, 1e-8))
        x_cat_arr[i, :sl, 0] = cat_i
        gap_raw[i, :sl]       = np.maximum(gaps_i, 1e-6)  # no exact zeros

    return x_num_arr, x_cat_arr, gap_raw, mask_arr, y_arr, y_ent


def compute_bins(gap_raw, mask):
    """
    Equal-frequency (quantile) binning of gaps from valid positions.
    Returns bin_edges (BBINS+1,) and tau_k (BBINS,).
    """
    valid = gap_raw[mask].astype(np.float64)
    valid = np.clip(valid, 1e-6, None)

    quantiles  = np.linspace(0, 100, BBINS + 1)
    bin_edges  = np.percentile(valid, quantiles)
    bin_edges[0]  = 0.0
    bin_edges[-1] = np.inf
    # Ensure strictly increasing (collapse duplicates slightly)
    for k in range(1, len(bin_edges) - 1):
        if bin_edges[k] <= bin_edges[k-1]:
            bin_edges[k] = bin_edges[k-1] + 1e-9

    # tau_k: median gap per bin
    bin_idx = np.digitize(valid, bin_edges[1:-1]).astype(int)  # 0-indexed
    tau_k   = np.zeros(BBINS, dtype=np.float32)
    for k in range(BBINS):
        in_bin = bin_idx == k
        if in_bin.sum() > 0:
            tau_k[k] = float(np.median(valid[in_bin]))
        elif k > 0:
            tau_k[k] = tau_k[k-1]
        else:
            tau_k[k] = float(np.median(valid))

    return bin_edges, tau_k


def apply_bins(gap_raw, mask, bin_edges):
    """Discretize gaps using precomputed bin_edges."""
    N_sp    = gap_raw.shape[0]
    dt_bin  = np.zeros((N_sp, L), dtype=np.int32)
    valid   = gap_raw[mask].astype(np.float64)
    valid   = np.clip(valid, 1e-6, None)
    indices = np.digitize(valid, bin_edges[1:-1]).astype(np.int32)
    indices = np.clip(indices, 0, BBINS - 1)
    dt_bin[mask] = indices
    return dt_bin


# ── Coupling verification ─────────────────────────────────────────────────────

def verify_coupling(x_num, dt_bin, x_cat, mask, y_ent, tau_k, W, device="cpu"):
    """
    Compute entity-level g (vel, gap, fanout, amt) and report
    P(fraud | g_bin=hi) vs P(fraud | g_bin=lo) for each channel.
    """
    import torch
    from models.teacher import compute_g_from_real

    tau_t = torch.tensor(tau_k, dtype=torch.float32).to(device)
    Bbins = len(tau_k)
    n_cat = [K_CAT]
    N     = x_num.shape[0]
    BATCH = 512

    xn_t  = torch.from_numpy(x_num).float()
    db_t  = torch.from_numpy(dt_bin).long()
    xc_t  = torch.from_numpy(x_cat).long()
    m_t   = torch.from_numpy(mask.astype(np.float32)).unsqueeze(-1)

    chunks = []
    with torch.no_grad():
        for s in range(0, N, BATCH):
            e = min(s + BATCH, N)
            g_b = compute_g_from_real(
                xn_t[s:e].to(device), db_t[s:e].to(device), xc_t[s:e].to(device),
                tau_t, W, 1.0, Bbins, n_cat,
                valid_mask=torch.from_numpy(mask[s:e]).bool().to(device),
            )
            chunks.append(g_b.cpu())
    g_all = torch.cat(chunks, dim=0)                   # (N, L, 4)
    g_ent = (g_all * m_t).sum(1) / m_t.sum(1).clamp(1) # (N, 4)
    g_np  = g_ent.numpy()

    ch_names = ["vel", "gap", "fanout", "amt"]
    print(f"\n  ── Coupling verification (g-channel P(fraud|bin)) ──")
    print(f"  {'channel':>8}  {'P_lo':>8}  {'P_hi':>8}  {'Δ':>8}  {'direction':>12}")
    for ci, name in enumerate(ch_names):
        col = g_np[:, ci]
        thr = float(np.median(col))
        lo_mask = col <= thr
        hi_mask = col >  thr
        p_lo = float(y_ent[lo_mask].mean()) if lo_mask.sum() > 0 else float("nan")
        p_hi = float(y_ent[hi_mask].mean()) if hi_mask.sum() > 0 else float("nan")
        delta = p_hi - p_lo
        direction = "fraud=hi" if p_hi > p_lo else "fraud=lo"
        print(f"  {name:>8}  {p_lo:.4f}  {p_hi:.4f}  {delta:+.4f}  {direction:>12}")


# ── Main generation ───────────────────────────────────────────────────────────

def run_kappa(kappa, N_train, N_test, fraud_rate, seed, out_base, W, verify):
    out_dir = Path(out_base) / f"kappa_{kappa:.2f}" / "sequences"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*56}")
    print(f" κ = {kappa:.2f}   →  {out_dir}")
    print(f"{'='*56}")

    seed_tr = seed
    seed_te = seed + 100_000   # disjoint seed space for test

    # Generate
    xn_tr, xc_tr, gap_tr, mask_tr, y_tr, yent_tr = generate_split(N_train, fraud_rate, kappa, seed_tr)
    xn_te, xc_te, gap_te, mask_te, y_te, yent_te = generate_split(N_test,  fraud_rate, kappa, seed_te)

    # Compute bins from training data only
    bin_edges, tau_k = compute_bins(gap_tr, mask_tr)

    dt_tr = apply_bins(gap_tr, mask_tr, bin_edges)
    dt_te = apply_bins(gap_te, mask_te, bin_edges)

    # Stats
    fraud_tr = float(yent_tr.mean())
    fraud_te = float(yent_te.mean())
    n_rule_tr = int(((yent_tr == 1) & (np.random.default_rng(seed_tr).uniform(0,1,N_train) < kappa)).sum())
    print(f"  Train: N={N_train}  fraud={fraud_tr:.4f}  ({int(yent_tr.sum())} entities)")
    print(f"  Test:  N={N_test}   fraud={fraud_te:.4f}  ({int(yent_te.sum())} entities)")
    print(f"  tau_k: [{tau_k.min():.4f} ... {tau_k.max():.4f}]")

    # Gap distribution sanity check
    fraud_mask_tr  = mask_tr & (y_tr > 0)
    nonfraud_mask  = mask_tr & (y_tr == 0)
    if fraud_mask_tr.sum() > 0 and nonfraud_mask.sum() > 0:
        med_fr  = float(np.median(gap_tr[fraud_mask_tr]))
        med_nfr = float(np.median(gap_tr[nonfraud_mask]))
        print(f"  Median gap: fraud={med_fr:.3f}  non-fraud={med_nfr:.3f}  "
              f"ratio={med_nfr/max(med_fr,1e-9):.1f}x")
    elif kappa == 0.0:
        all_gap = float(np.median(gap_tr[mask_tr]))
        print(f"  Median gap (all): {all_gap:.3f}  (κ=0 → no fraud-mode)")

    # Coupling verification using g-computation
    if verify:
        verify_coupling(xn_tr, dt_tr, xc_tr, mask_tr, yent_tr, tau_k, W)

    # Save
    np.savez_compressed(out_dir / "train.npz",
                        x_num=xn_tr, dt_bin=dt_tr, x_cat=xc_tr,
                        y=y_tr, mask=mask_tr)
    np.savez_compressed(out_dir / "test.npz",
                        x_num=xn_te, dt_bin=dt_te, x_cat=xc_te,
                        y=y_te, mask=mask_te)

    meta = {
        "dataset":     f"kappa_{kappa:.2f}",
        "kappa":       kappa,
        "L":           L,
        "B":           BBINS,
        "bin_edges":   [float(x) if x != np.inf else "Infinity" for x in bin_edges],
        "tau_k":       tau_k.tolist(),
        "num_classes_cat": {"cat_0": K_CAT},
        "d_num":       D_NUM,
        "d_cat":       1,
        "fraud_rate":  round(fraud_tr, 5),
        "W_default":   W,
        "fraud_mode":  "high_vel_low_fanout",
        "rule":        "burst(Exp(0.5) gaps, ~14 tx/window) AND cat=0 only (fanout=1)",
        "seed":        seed,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved: train.npz ({N_train}), test.npz ({N_test}), meta.json")
    return meta


def main():
    args = parse_args()
    print(f"Generating κ-controlled transaction sequences")
    print(f"  κ values     : {args.kappas}")
    print(f"  N_train      : {args.N_train}")
    print(f"  N_test       : {args.N_test}")
    print(f"  fraud_rate   : {args.fraud_rate}")
    print(f"  seed         : {args.seed}")
    print(f"  W            : {args.W}")
    print(f"  output base  : {args.out_base}")
    print(f"  verify       : {args.verify}")
    print()
    print(f"Behavioral design:")
    print(f"  Fraud-mode : Exp({FRAUD_GAP_SCALE}) gaps → mean={FRAUD_GAP_SCALE:.3f} → burst (~14 tx/W=7)")
    print(f"             : ALL receivers = cat 0 (fanout=1 per window)")
    print(f"  Non-fraud  : Exp({NONFR_GAP_SCALE}) gaps → mean={NONFR_GAP_SCALE} → spread (~3.5 tx/W=7)")
    print(f"             : receivers uniform over cat [{NONFR_CAT_MIN},{K_CAT}) → fanout≈3.4")
    print(f"  Amount     : LogNormal({AMT_LOG_MEAN},{AMT_LOG_STD}) for both → weak signal")

    for kappa in args.kappas:
        run_kappa(kappa, args.N_train, args.N_test, args.fraud_rate,
                  args.seed, args.out_base, args.W, args.verify)

    print(f"\n[Done] All κ values written to {args.out_base}/kappa_*/sequences/")


if __name__ == "__main__":
    main()
