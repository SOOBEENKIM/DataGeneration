"""
Step 0: Single-config end-to-end dry run on AMLSim.

Config: λ=1.0 · keep=0.2 · seed=1 · T_diff=50 DDIM steps

Pipeline:
  Phase 0  — scarce split (stratified, keep=0.2 of train)
  Phase 1  — compute fixed g_std from scarce train set
  Phase 2  — pretrain BehaviorTeacher (300 steps)
  Phase 3  — train CoF-SeqGen with coherence loss (N_TRAIN_STEPS)
             log: L_diff, L_coh, ||num_head.grad|| (amt-coherence activation)
  Phase 4  — DDIM generation of synthetic sequences
             sanity: distribution comparison real vs synth
  Phase 5  — TSTR evaluation
             HGB(scarce real) → AUPRC_baseline
             HGB(synthetic)   → AUPRC_synth
             ΔAUPRC = AUPRC_synth − AUPRC_baseline
  Phase 6  — coherence gap: |P(fraud|high_vel)_synth − P(fraud|high_vel)_real|

Purpose: verify evaluation harness is bug-free and numbers are sane
before running the 30+ run sweep.
"""

import sys
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.teacher import (
    BehaviorTeacher, compute_g_from_real,
    compute_g_stats_from_data, pretrain_teacher,
)
from models.seq_denoiser import SeqDenoiser
from models.cof_seqgen import CoFSeqGen
from models.sampler import ddim_sample, sample_empirical_dt_bin
from eval.tstr import (
    extract_features, run_tstr, stratified_keep, coherence_gap,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed',        type=int,   default=1)
    p.add_argument('--keep',        type=float, default=0.2)
    p.add_argument('--coh_lambda',  type=float, default=1.0)
    p.add_argument('--n_steps',     type=int,   default=2000,
                   help='CoF-SeqGen training steps')
    p.add_argument('--teacher_steps', type=int, default=300)
    p.add_argument('--T_diff',      type=int,   default=50,
                   help='DDIM reverse steps')
    p.add_argument('--batch_size',  type=int,   default=64)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--log_every',   type=int,   default=100)
    p.add_argument('--device',      type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--data_root',   type=str,
                   default=str(ROOT / 'data' / 'amlsim' / 'sequences'))
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_amlsim(data_root: str):
    root = Path(data_root)
    with open(root / 'meta.json') as f:
        meta = json.load(f)

    def _load(split):
        d = np.load(root / f'{split}.npz')
        return {k: torch.from_numpy(d[k]) for k in d}

    return meta, _load('train'), _load('test')


def seq_label(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sequence-level label: any fraud in masked positions. (N,) float."""
    return ((y.float() * mask.float()).max(dim=1).values > 0).float()


def print_section(title: str):
    print(f'\n{"─"*60}')
    print(f'  {title}')
    print(f'{"─"*60}')


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_cof(
    model: CoFSeqGen,
    x_num: torch.Tensor,    # (N, L, d_num)
    dt_bin: torch.Tensor,   # (N, L)
    x_cat: torch.Tensor,    # (N, L, d_cat)
    y: torch.Tensor,        # (N, L)
    mask: torch.Tensor,     # (N, L) bool
    n_steps: int,
    batch_size: int,
    lr: float,
    log_every: int,
    device: str,
) -> dict:
    """
    Mini-batch training for CoF-SeqGen.
    Returns dict with loss history and gradient norm history.
    """
    model.train()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    N = x_num.shape[0]
    history = {
        'L_diff': [], 'L_coh': [], 'L_label': [], 'num_grad_norm': [],
        'step': [],
    }

    t_start = time.time()
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, N, (batch_size,))
        xn = x_num[idx].to(device)
        db = dt_bin[idx].to(device)
        xc = x_cat[idx].to(device)
        yb = y[idx].to(device)
        mb = mask[idx].to(device)

        # Random diffusion time in (0, 1]
        t_frac = torch.rand(1).item() * 0.9 + 0.1

        optim.zero_grad()
        loss, info = model.compute_loss(xn, db, xc, yb, mb, t_frac)
        loss.backward()

        # Capture num_head gradient BEFORE optimizer.step()
        g = model.denoiser.num_head.weight.grad
        num_grad_norm = g.norm().item() if g is not None else 0.0

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t_start
            print(f'  step {step:5d}/{n_steps} '
                  f'| L_diff={info["L_diff"]:.4f} '
                  f'| L_coh={info["L_coh"]:.4f} '
                  f'| L_label={info["L_label"]:.4f} '
                  f'| ||num_grad||={num_grad_norm:.2e} '
                  f'| {elapsed:.0f}s')

            history['step'].append(step)
            history['L_diff'].append(info['L_diff'])
            history['L_coh'].append(info['L_coh'])
            history['L_label'].append(info['L_label'])
            history['num_grad_norm'].append(num_grad_norm)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic(
    model: CoFSeqGen,
    dt_bin_cond: torch.Tensor,   # (N, L) — real dt_bin for conditioning
    x_cat_cond: torch.Tensor,    # (N, L, d_cat)
    d_num: int,
    T_diff: int,
    batch_size: int,
    device: str,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Generate x_num_synthetic (N, L, d_num) via DDIM in batches."""
    model.eval()
    N = dt_bin_cond.shape[0]
    chunks = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_gen, _, _, _, _ = ddim_sample(
            model,
            dt_bin_cond[start:end].to(device),
            x_cat_cond[start:end].to(device),
            d_num=d_num,
            T_steps=T_diff,
            device=device,
            return_discrete=False,
            valid_mask=valid_mask[start:end].to(device),
        )
        chunks.append(x_gen.cpu())
    return torch.cat(chunks, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def print_sanity(
    x_num_real: np.ndarray,    # (N, L, d_num)
    x_num_synth: np.ndarray,
    mask: np.ndarray,
    col_names: list,
):
    print('\n[Sanity] Feature distribution comparison (masked positions):')
    for c, name in enumerate(col_names):
        r = x_num_real[:, :, c][mask.astype(bool)].flatten()
        s = x_num_synth[:, :, c][mask.astype(bool)].flatten()
        print(f'  {name:15s}: real  mean={r.mean():.3f} std={r.std():.3f} '
              f'p50={np.median(r):.3f}')
        print(f'  {" "*15}  synth mean={s.mean():.3f} std={s.std():.3f} '
              f'p50={np.median(s):.3f}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    print(f'\n{"="*60}')
    print(f'  AMLSim Dry Run | λ={args.coh_lambda} | keep={args.keep} | seed={args.seed}')
    print(f'  device={device} | n_steps={args.n_steps} | T_diff={args.T_diff}')
    print(f'{"="*60}')

    # ── Load data ─────────────────────────────────────────────────────────────
    meta, train, test = load_amlsim(args.data_root)
    tau    = torch.tensor(meta['tau_k'], dtype=torch.float32).to(device)
    Bbins  = len(meta['tau_k'])
    W      = 7.0
    temp   = 1.0
    d_num  = meta['d_num']
    L      = meta['L']
    n_cat  = []

    x_num_full   = train['x_num'].float()    # (N, L, d_num)
    dt_bin_full  = train['dt_bin'].long()
    x_cat_full   = train['x_cat'].long()
    y_full       = train['y'].float()
    mask_full    = train['mask'].bool()
    y_seq_full   = seq_label(y_full, mask_full).numpy()

    # ── Phase 0: Scarce split ─────────────────────────────────────────────────
    print_section('Phase 0 — Scarce split')
    scarce_idx = stratified_keep(
        len(y_seq_full), y_seq_full, keep=args.keep, seed=args.seed
    )
    x_num   = x_num_full[scarce_idx]
    dt_bin  = dt_bin_full[scarce_idx]
    x_cat   = x_cat_full[scarce_idx]
    y       = y_full[scarce_idx]
    mask    = mask_full[scarce_idx]
    y_seq   = y_seq_full[scarce_idx]

    print(f'  Scarce: {len(scarce_idx)} seqs '
          f'({int(y_seq.sum())} fraud, {int((1-y_seq).sum())} legit) '
          f'[fraud rate: {y_seq.mean():.3%}]')

    # ── Phase 1: g_std ────────────────────────────────────────────────────────
    print_section('Phase 1 — Compute g_std from scarce train')
    g_mean, g_std = compute_g_stats_from_data(
        x_num, dt_bin, x_cat,
        tau.cpu(), W, temp, Bbins, n_cat, batch_size=256,
        valid_mask=mask,
    )
    g_std = g_std.to(device)
    print(f'  g_std: vel={g_std[0]:.3f}  gap={g_std[1]:.3f}  '
          f'rep={g_std[2]:.3f}  amt={g_std[3]:.3f}')

    # ── Phase 2: Pretrain teacher ─────────────────────────────────────────────
    print_section(f'Phase 2 — Pretrain teacher ({args.teacher_steps} steps)')
    teacher = BehaviorTeacher(d_num, Bbins, n_cat, d_hidden=64, n_layers=2)
    teacher.to(device)

    # Mini-batch teacher pretraining
    N = x_num.shape[0]
    with torch.no_grad():
        g_real_all = compute_g_from_real(
            x_num.to(device), dt_bin.to(device), x_cat.to(device),
            tau, W, temp, Bbins, n_cat,
            valid_mask=mask.to(device),
        )
        g_target_all = g_real_all / g_std[None, None, :].clamp(min=1.0)

    teacher_optim = torch.optim.Adam(teacher.parameters(), lr=3e-3)
    teacher.train()
    for step in range(1, args.teacher_steps + 1):
        idx = torch.randint(0, N, (args.batch_size,))
        xn_ = x_num[idx].to(device)
        db_ = dt_bin[idx].to(device)
        xc_ = x_cat[idx].to(device)
        yb_ = y[idx].to(device)
        gt_ = g_target_all[idx]

        g_pred, y_logit = teacher(xn_, db_, xc_)
        t_loss = (F.binary_cross_entropy_with_logits(y_logit.float(), yb_.float())
                  + 0.1 * F.mse_loss(g_pred, gt_))
        teacher_optim.zero_grad()
        t_loss.backward()
        teacher_optim.step()

        if step % 100 == 0 or step == args.teacher_steps:
            print(f'  teacher step {step:4d}: loss={t_loss.item():.4f}')

    teacher.freeze()

    # ── Phase 3: Train CoF-SeqGen ─────────────────────────────────────────────
    print_section(f'Phase 3 — CoF-SeqGen training (λ={args.coh_lambda}, {args.n_steps} steps)')

    denoiser = SeqDenoiser(
        d_num=d_num, Bbins=Bbins, n_cat_classes=n_cat,
        d_model=128, n_heads=4, n_layers=2, L_max=64,
    )
    model = CoFSeqGen(
        denoiser=denoiser,
        tau=tau.cpu(),
        W=W, temp=temp,
        coh_lambda=args.coh_lambda,
        n_cat_classes=n_cat,
        teacher=teacher,
        g_std=g_std.cpu(),
    )

    history = train_cof(
        model, x_num, dt_bin, x_cat, y, mask,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=args.log_every,
        device=device,
    )

    # ── Phase 4: Generate synthetic sequences ─────────────────────────────────
    print_section(f'Phase 4 — DDIM generation (T={args.T_diff} steps)')
    print(f'  Generating {len(scarce_idx)} synthetic sequences...')
    t0 = time.time()

    x_num_synth = generate_synthetic(
        model, dt_bin, x_cat, d_num=d_num,
        T_diff=args.T_diff, batch_size=args.batch_size, device=device,
        valid_mask=mask,
    )
    print(f'  Done in {time.time()-t0:.1f}s')

    # Sanity: distribution check
    print_sanity(
        x_num.numpy(), x_num_synth.numpy(), mask.numpy(),
        col_names=meta['num_cols'],
    )

    # ── Phase 5: TSTR evaluation ──────────────────────────────────────────────
    print_section('Phase 5 — TSTR evaluation')

    # Test features from real entity-disjoint test set
    x_num_te  = test['x_num'].float().numpy()
    dt_bin_te = test['dt_bin'].long().numpy()
    mask_te   = test['mask'].bool().numpy()
    y_te      = test['y'].float().numpy()

    X_test, y_test = extract_features(x_num_te, dt_bin_te, mask_te, y_te)

    # Scarce-real baseline
    X_real, y_real_lab = extract_features(
        x_num.numpy(), dt_bin.numpy(), mask.numpy(), y.numpy()
    )
    result_base = run_tstr(X_real, y_real_lab, X_test, y_test, random_state=args.seed)

    # Synthetic training set (same scarce dt_bin/mask/y, generated x_num)
    X_synth, y_synth_lab = extract_features(
        x_num_synth.numpy(), dt_bin.numpy(), mask.numpy(), y.numpy()
    )
    result_synth = run_tstr(X_synth, y_synth_lab, X_test, y_test, random_state=args.seed)

    # Augmented: real scarce + synthetic
    X_aug  = np.concatenate([X_real,  X_synth], axis=0)
    y_aug  = np.concatenate([y_real_lab, y_synth_lab], axis=0)
    result_aug = run_tstr(X_aug, y_aug, X_test, y_test, random_state=args.seed)

    delta_auprc       = result_synth['auprc'] - result_base['auprc']
    delta_auprc_aug   = result_aug['auprc'] - result_base['auprc']
    prevalence_test   = y_test.mean()

    print(f'\n  Test set: {int(y_test.sum())} fraud / {len(y_test)} seqs '
          f'[prevalence {prevalence_test:.3%}]')
    print(f'  Scarce-real  AUPRC = {result_base["auprc"]:.4f}   '
          f'AUROC = {result_base["auroc"]:.4f}')
    print(f'  Synthetic    AUPRC = {result_synth["auprc"]:.4f}   '
          f'AUROC = {result_synth["auroc"]:.4f}')
    print(f'  Real+Synth   AUPRC = {result_aug["auprc"]:.4f}   '
          f'AUROC = {result_aug["auroc"]:.4f}')
    print(f'\n  ΔAUPRC (synth−base)     = {delta_auprc:+.4f}')
    print(f'  ΔAUPRC (aug−base)       = {delta_auprc_aug:+.4f}')

    # ── Phase 6: Coherence gap ────────────────────────────────────────────────
    print_section('Phase 6 — Coherence gap (behavioral-fraud coupling)')

    with torch.no_grad():
        g_real_np = compute_g_from_real(
            x_num.to(device), dt_bin.to(device), x_cat.to(device),
            tau, W, temp, Bbins, n_cat,
            valid_mask=mask.to(device),
        ).cpu().numpy()                        # (N, L, 4)
        g_synth_np = compute_g_from_real(
            x_num_synth.to(device), dt_bin.to(device), x_cat.to(device),
            tau, W, temp, Bbins, n_cat,
            valid_mask=mask.to(device),
        ).cpu().numpy()

    # Use sequence-mean g as per-entity behavior summary
    m = mask.numpy().astype(bool)
    def seq_mean_g(g, mask_):
        out = np.zeros((g.shape[0], 4))
        for i in range(g.shape[0]):
            pos = mask_[i]
            out[i] = g[i, pos].mean(axis=0) if pos.sum() > 0 else np.zeros(4)
        return out

    g_real_ent  = seq_mean_g(g_real_np,  m)
    g_synth_ent = seq_mean_g(g_synth_np, m)

    for fi, fname in enumerate(['vel', 'gap', 'rep', 'amt_sum']):
        if g_real_ent[:, fi].std() < 0.01:
            continue   # skip degenerate (rep=0 for AMLSim)
        gap_val = coherence_gap(
            g_synth_ent, y_seq,
            g_real_ent,  y_seq,
            feat_idx=fi,
        )
        print(f'  coherence_gap[{fname}] = {gap_val:.4f}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  SUMMARY')
    print(f'  g_std:    vel={g_std[0]:.2f} gap={g_std[1]:.2f} rep={g_std[2]:.2f} amt={g_std[3]:.2f}')
    print(f'  L_diff:   {history["L_diff"][0]:.3f} → {history["L_diff"][-1]:.3f}')
    print(f'  L_coh:    {history["L_coh"][0]:.3f} → {history["L_coh"][-1]:.3f}')
    print(f'  ||num_grad|| first={history["num_grad_norm"][0]:.2e} '
          f'last={history["num_grad_norm"][-1]:.2e}')
    print(f'  ΔAUPRC (synth−base) = {delta_auprc:+.4f}')
    print(f'  ΔAUPRC (aug−base)   = {delta_auprc_aug:+.4f}')
    sanity_ok = (
        history['L_diff'][-1] < history['L_diff'][0]
        and result_base['auprc'] > 0
        and result_synth['auprc'] > 0
    )
    print(f'  Harness sanity: {"✓ OK" if sanity_ok else "✗ CHECK ABOVE"}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
