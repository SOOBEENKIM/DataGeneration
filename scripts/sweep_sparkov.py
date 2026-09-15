"""
Sparkov λ sweep — full experiment grid.

Grid:
  λ (coh_lambda) ∈ {0, 0.1, 0.5, 1, 2}     (λ=0 = internal baseline w/o coherence)
  keep            ∈ {0.1, 0.2, 0.4}
  seed            ∈ {1, 2, 3}               (CI via bootstrap)

Total: 5 × 3 × 3 = 45 runs
Estimated: ~3 min/run × 45 ≈ 2.25 hours on a single GPU

Each run trains CoF-SeqGen for n_steps (default 20000),
generates synthetic data, and evaluates TSTR AUPRC.
Results are saved incrementally to results/sparkov_sweep.csv.

Usage:
  python3 scripts/sweep_sparkov.py                   # full grid
  python3 scripts/sweep_sparkov.py --lambdas 0 1     # subset
  python3 scripts/sweep_sparkov.py --keeps 0.1 --seeds 1 2  # subset

Outputs:
  results/sparkov_sweep.csv   — per-run results
  results/sparkov_sweep_agg.csv — mean±std over seeds per (λ, keep)
"""

import sys
import csv
import json
import time
import argparse
from pathlib import Path
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.teacher import (
    BehaviorTeacher, compute_g_from_real, compute_g_stats_from_data,
)
from models.seq_denoiser import SeqDenoiser
from models.cof_seqgen import CoFSeqGen
from models.sampler import ddim_sample
from eval.tstr import extract_features, run_tstr, stratified_keep


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

LAMBDA_GRID = [0.0, 0.1, 0.5, 1.0, 2.0]
KEEP_GRID   = [0.1, 0.2, 0.4]
SEED_GRID   = [1, 2, 3]

N_STEPS         = 20000
TEACHER_STEPS   = 300
T_DIFF          = 50
BATCH_SIZE      = 64
LR              = 3e-4
D_MODEL         = 128
W               = 60.0
TEMP            = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Parse args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lambdas',   type=float, nargs='+', default=LAMBDA_GRID)
    p.add_argument('--keeps',     type=float, nargs='+', default=KEEP_GRID)
    p.add_argument('--seeds',     type=int,   nargs='+', default=SEED_GRID)
    p.add_argument('--n_steps',   type=int,   default=N_STEPS)
    p.add_argument('--teacher_steps', type=int, default=TEACHER_STEPS)
    p.add_argument('--device',    type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--data_root', type=str,
                   default=str(ROOT / 'data' / 'sparkov' / 'sequences'))
    p.add_argument('--out_dir',   type=str, default=str(ROOT / 'results'))
    p.add_argument('--resume',    action='store_true',
                   help='skip already-completed configs in CSV')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (done once, shared across runs)
# ─────────────────────────────────────────────────────────────────────────────

def load_sparkov(data_root: str):
    root = Path(data_root)
    with open(root / 'meta.json') as f:
        meta = json.load(f)

    def _load(split):
        d = np.load(root / f'{split}.npz')
        return {k: torch.from_numpy(d[k]) for k in d}

    return meta, _load('train'), _load('test')


def seq_label(y, mask):
    return ((y.float() * mask.float()).max(dim=1).values > 0).float()


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_single(
    meta, train, test,
    coh_lambda: float,
    keep: float,
    seed: int,
    n_steps: int,
    teacher_steps: int,
    device: str,
) -> dict:
    """Run one (λ, keep, seed) configuration. Returns result dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    tau   = torch.tensor(meta['tau_k'], dtype=torch.float32).to(device)
    Bbins = len(meta['tau_k'])
    d_num = meta['d_num']
    n_cat = list(meta['num_classes_cat'].values())   # [14]

    x_num_full  = train['x_num'].float()
    dt_bin_full = train['dt_bin'].long()
    x_cat_full  = train['x_cat'].long()
    y_full      = train['y'].float()
    mask_full   = train['mask'].bool()
    y_seq_full  = seq_label(y_full, mask_full).numpy()

    # Scarce split (stratified)
    scarce_idx = stratified_keep(len(y_seq_full), y_seq_full, keep=keep, seed=seed)
    x_num   = x_num_full[scarce_idx]
    dt_bin  = dt_bin_full[scarce_idx]
    x_cat   = x_cat_full[scarce_idx]
    y       = y_full[scarce_idx]
    mask    = mask_full[scarce_idx]
    y_seq   = y_seq_full[scarce_idx]
    N       = x_num.shape[0]

    # Fixed g_std from scarce train
    g_mean, g_std = compute_g_stats_from_data(
        x_num, dt_bin, x_cat,
        tau.cpu(), W, TEMP, Bbins, n_cat, batch_size=256,
        valid_mask=mask,
    )
    g_std = g_std.to(device)

    # Pre-compute normalized g targets for teacher
    with torch.no_grad():
        g_real_all = compute_g_from_real(
            x_num.to(device), dt_bin.to(device), x_cat.to(device),
            tau, W, TEMP, Bbins, n_cat,
            valid_mask=mask.to(device),
        )
        g_target_all = g_real_all / g_std[None, None, :].clamp(min=1.0)

    # Pretrain teacher
    teacher = BehaviorTeacher(d_num, Bbins, n_cat, d_hidden=64, n_layers=2)
    teacher.to(device)
    t_optim = torch.optim.Adam(teacher.parameters(), lr=3e-3)
    teacher.train()
    for step in range(teacher_steps):
        idx = torch.randint(0, N, (BATCH_SIZE,))
        xn_ = x_num[idx].to(device)
        db_ = dt_bin[idx].to(device)
        xc_ = x_cat[idx].to(device)
        yb_ = y[idx].to(device)
        g_pred, y_logit = teacher(xn_, db_, xc_)
        t_loss = (F.binary_cross_entropy_with_logits(y_logit.float(), yb_.float())
                  + 0.1 * F.mse_loss(g_pred, g_target_all[idx]))
        t_optim.zero_grad()
        t_loss.backward()
        t_optim.step()
    teacher.freeze()

    # Train CoF-SeqGen
    denoiser = SeqDenoiser(
        d_num=d_num, Bbins=Bbins, n_cat_classes=n_cat,
        d_model=D_MODEL, n_heads=4, n_layers=2, L_max=64,
    )
    model = CoFSeqGen(
        denoiser=denoiser, tau=tau.cpu(), W=W, temp=TEMP,
        coh_lambda=coh_lambda, n_cat_classes=n_cat,
        teacher=teacher, g_std=g_std.cpu(),
    )
    model.train()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    loss_history = {'L_diff': [], 'L_coh': [], 'num_grad': []}
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, N, (BATCH_SIZE,))
        xn = x_num[idx].to(device)
        db = dt_bin[idx].to(device)
        xc = x_cat[idx].to(device)
        yb = y[idx].to(device)
        mb = mask[idx].to(device)
        t_frac = torch.rand(1).item() * 0.9 + 0.1

        optim.zero_grad()
        loss, info = model.compute_loss(xn, db, xc, yb, mb, t_frac)
        loss.backward()
        g_wgrad = model.denoiser.num_head.weight.grad
        num_grad_norm = g_wgrad.norm().item() if g_wgrad is not None else 0.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 2000 == 0 or step == n_steps:
            loss_history['L_diff'].append(info['L_diff'])
            loss_history['L_coh'].append(info['L_coh'])
            loss_history['num_grad'].append(num_grad_norm)

    L_diff_final = float(np.mean(loss_history['L_diff'][-3:]))
    L_coh_final  = float(np.mean(loss_history['L_coh'][-3:]))
    num_grad_final = float(np.mean(loss_history['num_grad'][-3:]))

    # Generate synthetic sequences
    model.eval()
    x_num_synth_chunks = []
    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        x_gen, _, _, _, _ = ddim_sample(
            model,
            dt_bin[start:end].to(device),
            x_cat[start:end].to(device),
            d_num=d_num, T_steps=T_DIFF, device=device, return_discrete=False,
            valid_mask=mask[start:end].to(device),
        )
        x_num_synth_chunks.append(x_gen.cpu())
    x_num_synth = torch.cat(x_num_synth_chunks, dim=0)

    # Generation quality: mean absolute shift (normalized by real std)
    bool_mask = mask.numpy().astype(bool)
    real_vals  = x_num.numpy()[:,:,0][bool_mask]
    synth_vals = x_num_synth.numpy()[:,:,0][bool_mask]
    amt_shift  = abs(real_vals.mean() - synth_vals.mean()) / (real_vals.std() + 1e-8)

    # TSTR evaluation
    x_num_te  = test['x_num'].float().numpy()
    dt_bin_te = test['dt_bin'].long().numpy()
    x_cat_te  = test['x_cat'].long().numpy()
    mask_te   = test['mask'].bool().numpy()
    y_te      = test['y'].float().numpy()

    X_test, y_test = extract_features(x_num_te, dt_bin_te, mask_te, y_te, x_cat_te)

    X_real, y_real_lab = extract_features(
        x_num.numpy(), dt_bin.numpy(), mask.numpy(), y.numpy(), x_cat.numpy()
    )
    X_synth, y_synth_lab = extract_features(
        x_num_synth.numpy(), dt_bin.numpy(), mask.numpy(), y.numpy(), x_cat.numpy()
    )
    X_aug  = np.concatenate([X_real, X_synth], axis=0)
    y_aug  = np.concatenate([y_real_lab, y_synth_lab], axis=0)

    res_base  = run_tstr(X_real,  y_real_lab,  X_test, y_test, random_state=seed)
    res_synth = run_tstr(X_synth, y_synth_lab, X_test, y_test, random_state=seed)
    res_aug   = run_tstr(X_aug,   y_aug,        X_test, y_test, random_state=seed)

    return {
        'coh_lambda':      coh_lambda,
        'keep':            keep,
        'seed':            seed,
        'n_scarce':        N,
        'n_fraud_scarce':  int(y_seq.sum()),
        'L_diff_final':    round(L_diff_final, 4),
        'L_coh_final':     round(L_coh_final, 4),
        'num_grad_final':  round(num_grad_final, 4),
        'amt_shift':       round(float(amt_shift), 4),
        'auprc_base':      round(res_base['auprc'],  4),
        'auprc_synth':     round(res_synth['auprc'], 4),
        'auprc_aug':       round(res_aug['auprc'],   4),
        'auroc_base':      round(res_base['auroc'],  4),
        'auroc_aug':       round(res_aug['auroc'],   4),
        'delta_synth':     round(res_synth['auprc'] - res_base['auprc'], 4),
        'delta_aug':       round(res_aug['auprc']   - res_base['auprc'], 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(csv_path: Path) -> None:
    """Print mean ± std over seeds grouped by (λ, keep)."""
    import csv as _csv
    rows = []
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            rows.append(row)

    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (float(r['coh_lambda']), float(r['keep']))
        groups[key].append(r)

    agg_path = csv_path.parent / (csv_path.stem + '_agg.csv')
    fieldnames = ['coh_lambda', 'keep', 'n_seeds',
                  'auprc_base_mean', 'auprc_base_std',
                  'auprc_aug_mean',  'auprc_aug_std',
                  'delta_aug_mean',  'delta_aug_std',
                  'delta_synth_mean','delta_synth_std',
                  'L_diff_final_mean', 'L_coh_final_mean', 'amt_shift_mean']

    with open(agg_path, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (lam, keep), group in sorted(groups.items()):
            def _mean(col): return float(np.mean([float(r[col]) for r in group]))
            def _std(col):  return float(np.std( [float(r[col]) for r in group]))
            w.writerow({
                'coh_lambda': lam, 'keep': keep, 'n_seeds': len(group),
                'auprc_base_mean': round(_mean('auprc_base'), 4),
                'auprc_base_std':  round(_std('auprc_base'),  4),
                'auprc_aug_mean':  round(_mean('auprc_aug'),  4),
                'auprc_aug_std':   round(_std('auprc_aug'),   4),
                'delta_aug_mean':  round(_mean('delta_aug'),  4),
                'delta_aug_std':   round(_std('delta_aug'),   4),
                'delta_synth_mean':round(_mean('delta_synth'),4),
                'delta_synth_std': round(_std('delta_synth'), 4),
                'L_diff_final_mean': round(_mean('L_diff_final'), 4),
                'L_coh_final_mean':  round(_mean('L_coh_final'),  4),
                'amt_shift_mean':    round(_mean('amt_shift'),     4),
            })

    print(f'\n[Aggregated] saved to {agg_path}')
    # Print table
    print(f'\n{"λ":>5} {"keep":>5} {"n":>3} '
          f'{"ΔAUPRC_aug":>12} {"AUPRC_base":>11} {"AUPRC_aug":>10}')
    print('─' * 55)
    with open(agg_path) as f:
        for row in _csv.DictReader(f):
            print(f'{float(row["coh_lambda"]):>5.1f} '
                  f'{float(row["keep"]):>5.2f} '
                  f'{row["n_seeds"]:>3} '
                  f'{float(row["delta_aug_mean"]):>+8.4f}±{float(row["delta_aug_std"]):.4f} '
                  f'{float(row["auprc_base_mean"]):>10.4f} '
                  f'{float(row["auprc_aug_mean"]):>10.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'sparkov_sweep.csv'

    # Load completed configs for resume
    completed = set()
    if args.resume and csv_path.exists():
        import csv as _csv
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                completed.add((float(row['coh_lambda']), float(row['keep']), int(row['seed'])))
        print(f'[Resume] {len(completed)} configs already done.')

    # Build grid
    grid = list(product(sorted(args.lambdas), sorted(args.keeps), sorted(args.seeds)))
    n_total = len(grid)

    # Load data once
    meta, train, test = load_sparkov(args.data_root)
    n_cat = list(meta['num_classes_cat'].values())

    # Precompute full-train AUPRC (upper bound, once)
    x_num_full  = train['x_num'].float()
    dt_bin_full = train['dt_bin'].long()
    x_cat_full  = train['x_cat'].long()
    y_full      = train['y'].float()
    mask_full   = train['mask'].bool()
    x_num_te    = test['x_num'].float().numpy()
    dt_bin_te   = test['dt_bin'].long().numpy()
    x_cat_te    = test['x_cat'].long().numpy()
    mask_te     = test['mask'].bool().numpy()
    y_te        = test['y'].float().numpy()

    X_test, y_test = extract_features(x_num_te, dt_bin_te, mask_te, y_te, x_cat_te)
    X_full, y_full_lab = extract_features(
        x_num_full.numpy(), dt_bin_full.numpy(), mask_full.numpy(),
        y_full.numpy(), x_cat_full.numpy()
    )
    res_full = run_tstr(X_full, y_full_lab, X_test, y_test, random_state=0)
    auprc_full = res_full['auprc']
    print(f'[Upper bound] Full-train AUPRC = {auprc_full:.4f}')

    # Open CSV
    fieldnames = [
        'coh_lambda', 'keep', 'seed', 'n_scarce', 'n_fraud_scarce',
        'L_diff_final', 'L_coh_final', 'num_grad_final', 'amt_shift',
        'auprc_base', 'auprc_synth', 'auprc_aug',
        'auroc_base', 'auroc_aug',
        'delta_synth', 'delta_aug', 'auprc_full',
    ]
    write_header = not csv_path.exists() or not args.resume
    fout = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    fout.flush()

    t_sweep_start = time.time()
    for run_idx, (lam, keep, seed) in enumerate(grid, 1):
        if (lam, keep, seed) in completed:
            print(f'[{run_idx}/{n_total}] SKIP λ={lam} keep={keep} seed={seed}')
            continue

        print(f'\n[{run_idx}/{n_total}] λ={lam} keep={keep} seed={seed} '
              f'— elapsed {(time.time()-t_sweep_start)/60:.1f}min')

        t0 = time.time()
        result = run_single(
            meta, train, test,
            coh_lambda=lam, keep=keep, seed=seed,
            n_steps=args.n_steps, teacher_steps=args.teacher_steps,
            device=args.device,
        )
        result['auprc_full'] = round(auprc_full, 4)
        elapsed = time.time() - t0

        print(f'  L_diff={result["L_diff_final"]:.4f} L_coh={result["L_coh_final"]:.4f} '
              f'||g||={result["num_grad_final"]:.2e} amt_shift={result["amt_shift"]:.3f}')
        print(f'  AUPRC: base={result["auprc_base"]:.4f} '
              f'synth={result["auprc_synth"]:.4f} '
              f'aug={result["auprc_aug"]:.4f} '
              f'→ Δ(aug)={result["delta_aug"]:+.4f}  [{elapsed:.0f}s]')

        writer.writerow(result)
        fout.flush()

    fout.close()
    print(f'\n[Sweep complete] {n_total} runs saved to {csv_path}')
    aggregate_results(csv_path)


if __name__ == '__main__':
    main()
