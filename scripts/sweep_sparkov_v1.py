"""
Sparkov λ sweep — v1 (label-behavioral KL coherence, model-generated labels).

v1 vs v0:
  - CoherenceTeacher f_φ: g→P(fraud) replaces BehaviorTeacher
  - L_coh = KL(P_model ‖ f_φ(g(x̂_0)/g_std))   [label-behavioral KL]
  - Synthetic conditioning: empirically sampled dt_bin/x_cat (removes real scaffold)
  - Synthetic labels: model label head output y_gen (Bernoulli sample)
  - TSTR features: basic + windowed behavioral (vel/gap/rep/amt_sum × mean/max)

Grid: λ∈{0,0.1,0.5,1,2} × keep∈{0.1,0.2,0.4} × seed∈{1,2,3} = 45 runs
Output: results/sparkov_sweep_v1.csv

Usage:
  python3 -u scripts/sweep_sparkov_v1.py
  python3 -u scripts/sweep_sparkov_v1.py --lambdas 0 0.5 1 --keeps 0.1
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

from models.coherence_teacher import CoherenceTeacher, pretrain_coherence_teacher
from models.teacher import compute_g_stats_from_data
from models.seq_denoiser import SeqDenoiser
from models.cof_seqgen import CoFSeqGen
from models.sampler import ddim_sample, sample_empirical_dt_bin, sample_empirical_x_cat
from eval.tstr import (
    extract_features_with_behavioral,
    run_tstr,
    stratified_keep,
)


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

LAMBDA_GRID     = [0.0, 0.1, 0.5, 1.0, 2.0]
KEEP_GRID       = [0.1, 0.2, 0.4]
SEED_GRID       = [1, 2, 3]

N_STEPS         = 20000
TEACHER_STEPS   = 2000
T_DIFF          = 50
BATCH_SIZE      = 64
LR              = 3e-4
D_MODEL         = 128
W               = 60.0
TEMP            = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambdas",   type=float, nargs="+", default=LAMBDA_GRID)
    p.add_argument("--keeps",     type=float, nargs="+", default=KEEP_GRID)
    p.add_argument("--seeds",     type=int,   nargs="+", default=SEED_GRID)
    p.add_argument("--n_steps",   type=int,   default=N_STEPS)
    p.add_argument("--teacher_steps", type=int, default=TEACHER_STEPS)
    p.add_argument("--device",    type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data_root", type=str,
                   default=str(ROOT / "data" / "sparkov" / "sequences"))
    p.add_argument("--out_dir",   type=str, default=str(ROOT / "results"))
    p.add_argument("--resume",    action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sparkov(data_root: str):
    root = Path(data_root)
    with open(root / "meta.json") as f:
        meta = json.load(f)
    def _load(split):
        d = np.load(root / f"{split}.npz")
        return {k: torch.from_numpy(d[k]) for k in d}
    return meta, _load("train"), _load("test")


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
    torch.manual_seed(seed)
    np.random.seed(seed)

    tau   = torch.tensor(meta["tau_k"], dtype=torch.float32).to(device)
    Bbins = len(meta["tau_k"])
    d_num = meta["d_num"]
    n_cat = list(meta["num_classes_cat"].values())

    x_num_full  = train["x_num"].float()
    dt_bin_full = train["dt_bin"].long()
    x_cat_full  = train["x_cat"].long()
    y_full      = train["y"].float()
    mask_full   = train["mask"].bool()
    y_seq_full  = seq_label(y_full, mask_full).numpy()

    # Scarce split
    scarce_idx = stratified_keep(len(y_seq_full), y_seq_full, keep=keep, seed=seed)
    x_num   = x_num_full[scarce_idx]
    dt_bin  = dt_bin_full[scarce_idx]
    x_cat   = x_cat_full[scarce_idx]
    y       = y_full[scarce_idx]
    mask    = mask_full[scarce_idx]
    y_seq   = y_seq_full[scarce_idx]
    N       = x_num.shape[0]
    L       = x_num.shape[1]

    # Fixed g_std from scarce train set
    g_mean, g_std = compute_g_stats_from_data(
        x_num, dt_bin, x_cat,
        tau.cpu(), W, TEMP, Bbins, n_cat, batch_size=256,
        valid_mask=mask,
    )
    g_std = g_std.to(device)

    # ── Pretrain CoherenceTeacher f_φ: g/g_std → P(fraud) ──────────────────
    f_phi = CoherenceTeacher(hidden=64).to(device)
    teacher_losses = pretrain_coherence_teacher(
        f_phi,
        x_num=x_num, dt_bin=dt_bin, x_cat=x_cat,
        y=y, mask=mask,
        tau=tau, W=W, temp=TEMP, Bbins=Bbins, n_cat_classes=n_cat,
        g_std=g_std.cpu(),
        steps=teacher_steps, lr=1e-3, batch_size=256, device=device,
    )
    teacher_final_loss = float(np.mean(teacher_losses[-50:]))

    # Gate check: print teacher BCE to verify signal
    print(f"  f_phi final BCE={teacher_final_loss:.4f}")

    # ── Train CoF-SeqGen v1 ─────────────────────────────────────────────────
    denoiser = SeqDenoiser(
        d_num=d_num, Bbins=Bbins, n_cat_classes=n_cat,
        d_model=D_MODEL, n_heads=4, n_layers=2, L_max=64,
    )
    model = CoFSeqGen(
        denoiser=denoiser,
        tau=tau.cpu(), W=W, temp=TEMP,
        coh_lambda=coh_lambda,
        n_cat_classes=n_cat,
        coherence_teacher=f_phi,
        g_std=g_std.cpu(),
    )
    model.train()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    loss_history = {"L_diff": [], "L_coh": [], "num_grad": []}
    for step in range(1, n_steps + 1):
        idx    = torch.randint(0, N, (BATCH_SIZE,))
        xn     = x_num[idx].to(device)
        db     = dt_bin[idx].to(device)
        xc     = x_cat[idx].to(device)
        yb     = y[idx].to(device)
        mb     = mask[idx].to(device)
        t_frac = torch.rand(1).item() * 0.9 + 0.1

        optim.zero_grad()
        loss, info = model.compute_loss(xn, db, xc, yb, mb, t_frac)
        loss.backward()
        g_wgrad = model.denoiser.num_head.weight.grad
        num_grad_norm = g_wgrad.norm().item() if g_wgrad is not None else 0.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 2000 == 0 or step == n_steps:
            loss_history["L_diff"].append(info["L_diff"])
            loss_history["L_coh"].append(info["L_coh"])
            loss_history["num_grad"].append(num_grad_norm)

    L_diff_final   = float(np.mean(loss_history["L_diff"][-3:]))
    L_coh_final    = float(np.mean(loss_history["L_coh"][-3:]))
    num_grad_final = float(np.mean(loss_history["num_grad"][-3:]))

    # ── Generate synthetic sequences (v1: empirical conditioning) ───────────
    model.eval()
    x_num_synth_chunks = []
    bin_pred_chunks    = []
    cat_pred_chunks    = []
    y_gen_chunks       = []
    mask_samp_chunks   = []

    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        bs  = end - start

        # Empirically sampled conditioning (removes real scaffold copy)
        dt_samp  = sample_empirical_dt_bin(dt_bin, bs, L, Bbins, device=device)
        xcat_samp = sample_empirical_x_cat(x_cat, bs, device=device)
        # Mask: use real mask from randomly sampled sequences for length distribution
        mask_idx  = torch.randint(0, N, (bs,))
        mask_samp = mask[mask_idx]

        x_gen, bin_pred, cat_preds, y_gen, _ = ddim_sample(
            model,
            dt_samp, xcat_samp,
            d_num=d_num, T_steps=T_DIFF, device=device, return_discrete=True,
            valid_mask=mask_samp.to(device),
        )
        x_num_synth_chunks.append(x_gen.cpu())
        bin_pred_chunks.append(bin_pred.cpu())
        cat_pred_chunks.append(cat_preds[0].cpu() if cat_preds else torch.zeros(bs, L).long())
        y_gen_chunks.append(y_gen.cpu())
        mask_samp_chunks.append(mask_samp)

    x_num_synth  = torch.cat(x_num_synth_chunks, dim=0)         # (N, L, d_num)
    dt_bin_synth = torch.cat(bin_pred_chunks, dim=0)             # (N, L) — model output
    x_cat_synth  = torch.cat(cat_pred_chunks, dim=0).unsqueeze(-1)  # (N, L, 1)
    y_gen_all    = torch.cat(y_gen_chunks, dim=0)                # (N, L) — model-generated
    mask_synth   = torch.cat(mask_samp_chunks, dim=0)           # (N, L)

    # Generation quality metric
    bool_mask_r  = mask.numpy().astype(bool)
    bool_mask_s  = mask_synth.numpy().astype(bool)
    real_vals    = x_num.numpy()[:, :, 0][bool_mask_r]
    synth_vals   = x_num_synth.numpy()[:, :, 0][bool_mask_s]
    amt_shift    = abs(real_vals.mean() - synth_vals.mean()) / (real_vals.std() + 1e-8)

    # Synthetic sequence-level labels (from model's label head)
    y_gen_seq = ((y_gen_all.float() * mask_synth.float()).max(dim=1).values > 0).float().numpy()
    fraud_rate_synth = y_gen_seq.mean()

    # ── TSTR evaluation ──────────────────────────────────────────────────────
    tau_np = tau.cpu().numpy()

    x_num_te  = test["x_num"].float().numpy()
    dt_bin_te = test["dt_bin"].long().numpy()
    x_cat_te  = test["x_cat"].long().numpy()
    mask_te   = test["mask"].bool().numpy()
    y_te      = test["y"].float().numpy()

    # Test features with behavioral
    X_test, y_test = extract_features_with_behavioral(
        x_num_te, dt_bin_te, mask_te, y_te, x_cat_te,
        tau=tau_np, W=W, temp=TEMP, Bbins=Bbins, n_cat_classes=n_cat,
    )

    # Real (scarce) features
    X_real, y_real_lab = extract_features_with_behavioral(
        x_num.numpy(), dt_bin.numpy(), mask.numpy(), y.numpy(), x_cat.numpy(),
        tau=tau_np, W=W, temp=TEMP, Bbins=Bbins, n_cat_classes=n_cat,
    )

    # Synthetic features — y_gen_all as positional labels (sequence-level max)
    X_synth, y_synth_lab = extract_features_with_behavioral(
        x_num_synth.numpy(),
        dt_bin_synth.numpy(),
        mask_synth.numpy(),
        y_gen_all.numpy(),   # model-generated labels (v1: NOT real labels)
        x_cat_synth.numpy(),
        tau=tau_np, W=W, temp=TEMP, Bbins=Bbins, n_cat_classes=n_cat,
    )

    X_aug = np.concatenate([X_real, X_synth], axis=0)
    y_aug = np.concatenate([y_real_lab, y_synth_lab], axis=0)

    res_base  = run_tstr(X_real,  y_real_lab,  X_test, y_test, random_state=seed)
    res_synth = run_tstr(X_synth, y_synth_lab, X_test, y_test, random_state=seed)
    res_aug   = run_tstr(X_aug,   y_aug,        X_test, y_test, random_state=seed)

    return {
        "coh_lambda":         coh_lambda,
        "keep":               keep,
        "seed":               seed,
        "n_scarce":           N,
        "n_fraud_scarce":     int(y_seq.sum()),
        "teacher_bce":        round(teacher_final_loss, 4),
        "fraud_rate_synth":   round(float(fraud_rate_synth), 4),
        "L_diff_final":       round(L_diff_final, 4),
        "L_coh_final":        round(L_coh_final, 4),
        "num_grad_final":     round(num_grad_final, 4),
        "amt_shift":          round(float(amt_shift), 4),
        "auprc_base":         round(res_base["auprc"],  4),
        "auprc_synth":        round(res_synth["auprc"], 4),
        "auprc_aug":          round(res_aug["auprc"],   4),
        "auroc_base":         round(res_base["auroc"],  4),
        "auroc_aug":          round(res_aug["auroc"],   4),
        "delta_synth":        round(res_synth["auprc"] - res_base["auprc"], 4),
        "delta_aug":          round(res_aug["auprc"]   - res_base["auprc"], 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(csv_path: Path) -> None:
    import csv as _csv
    rows = []
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            rows.append(row)

    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (float(r["coh_lambda"]), float(r["keep"]))
        groups[key].append(r)

    agg_path = csv_path.parent / (csv_path.stem + "_agg.csv")
    fieldnames = [
        "coh_lambda", "keep", "n_seeds",
        "auprc_base_mean", "auprc_base_std",
        "auprc_aug_mean",  "auprc_aug_std",
        "delta_aug_mean",  "delta_aug_std",
        "delta_synth_mean", "delta_synth_std",
        "fraud_rate_synth_mean",
        "L_diff_final_mean", "L_coh_final_mean", "amt_shift_mean",
    ]

    with open(agg_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (lam, keep), group in sorted(groups.items()):
            def _mean(col): return float(np.mean([float(r[col]) for r in group]))
            def _std(col):  return float(np.std( [float(r[col]) for r in group]))
            w.writerow({
                "coh_lambda": lam, "keep": keep, "n_seeds": len(group),
                "auprc_base_mean":       round(_mean("auprc_base"),    4),
                "auprc_base_std":        round(_std("auprc_base"),     4),
                "auprc_aug_mean":        round(_mean("auprc_aug"),     4),
                "auprc_aug_std":         round(_std("auprc_aug"),      4),
                "delta_aug_mean":        round(_mean("delta_aug"),     4),
                "delta_aug_std":         round(_std("delta_aug"),      4),
                "delta_synth_mean":      round(_mean("delta_synth"),   4),
                "delta_synth_std":       round(_std("delta_synth"),    4),
                "fraud_rate_synth_mean": round(_mean("fraud_rate_synth"), 4),
                "L_diff_final_mean":     round(_mean("L_diff_final"),  4),
                "L_coh_final_mean":      round(_mean("L_coh_final"),   4),
                "amt_shift_mean":        round(_mean("amt_shift"),     4),
            })

    print(f"\n[Aggregated v1] saved to {agg_path}")
    print(f'\n{"λ":>5} {"keep":>5} {"n":>3} '
          f'{"ΔAUPRC_aug":>12} {"fraud_synth":>12} {"AUPRC_base":>11}')
    print("─" * 60)
    with open(agg_path) as f:
        for row in _csv.DictReader(f):
            print(
                f'{float(row["coh_lambda"]):>5.1f} '
                f'{float(row["keep"]):>5.2f} '
                f'{row["n_seeds"]:>3} '
                f'{float(row["delta_aug_mean"]):>+8.4f}±{float(row["delta_aug_std"]):.4f} '
                f'{float(row["fraud_rate_synth_mean"]):>11.4f} '
                f'{float(row["auprc_base_mean"]):>10.4f}'
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sparkov_sweep_v1.csv"

    # Resume
    completed = set()
    if args.resume and csv_path.exists():
        import csv as _csv
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                completed.add((float(row["coh_lambda"]), float(row["keep"]), int(row["seed"])))
        print(f"[Resume] {len(completed)} configs already done.")

    grid    = list(product(sorted(args.lambdas), sorted(args.keeps), sorted(args.seeds)))
    n_total = len(grid)

    meta, train, test = load_sparkov(args.data_root)
    n_cat = list(meta["num_classes_cat"].values())

    # Full-train AUPRC upper bound (once, v1 features)
    tau_np = np.array(meta["tau_k"], dtype=np.float32)
    Bbins  = len(meta["tau_k"])

    x_num_te  = test["x_num"].float().numpy()
    dt_bin_te = test["dt_bin"].long().numpy()
    x_cat_te  = test["x_cat"].long().numpy()
    mask_te   = test["mask"].bool().numpy()
    y_te      = test["y"].float().numpy()
    X_test, y_test = extract_features_with_behavioral(
        x_num_te, dt_bin_te, mask_te, y_te, x_cat_te,
        tau=tau_np, W=W, temp=TEMP, Bbins=Bbins, n_cat_classes=n_cat,
    )
    X_full, y_full_lab = extract_features_with_behavioral(
        train["x_num"].float().numpy(),
        train["dt_bin"].long().numpy(),
        train["mask"].bool().numpy(),
        train["y"].float().numpy(),
        train["x_cat"].long().numpy(),
        tau=tau_np, W=W, temp=TEMP, Bbins=Bbins, n_cat_classes=n_cat,
    )
    res_full   = run_tstr(X_full, y_full_lab, X_test, y_test, random_state=0)
    auprc_full = res_full["auprc"]
    print(f"[Upper bound v1] Full-train AUPRC = {auprc_full:.4f}")

    # CSV
    fieldnames = [
        "coh_lambda", "keep", "seed", "n_scarce", "n_fraud_scarce",
        "teacher_bce", "fraud_rate_synth",
        "L_diff_final", "L_coh_final", "num_grad_final", "amt_shift",
        "auprc_base", "auprc_synth", "auprc_aug",
        "auroc_base", "auroc_aug",
        "delta_synth", "delta_aug", "auprc_full",
    ]
    write_header = not csv_path.exists() or not args.resume
    fout   = open(csv_path, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    fout.flush()

    t_sweep_start = time.time()
    for run_idx, (lam, keep, seed) in enumerate(grid, 1):
        if (lam, keep, seed) in completed:
            print(f"[{run_idx}/{n_total}] SKIP λ={lam} keep={keep} seed={seed}")
            continue

        print(f"\n[{run_idx}/{n_total}] λ={lam} keep={keep} seed={seed} "
              f"— elapsed {(time.time()-t_sweep_start)/60:.1f}min")

        t0 = time.time()
        result = run_single(
            meta, train, test,
            coh_lambda=lam, keep=keep, seed=seed,
            n_steps=args.n_steps, teacher_steps=args.teacher_steps,
            device=args.device,
        )
        result["auprc_full"] = round(auprc_full, 4)
        elapsed = time.time() - t0

        print(f"  L_diff={result['L_diff_final']:.4f} L_coh={result['L_coh_final']:.4f} "
              f"||g||={result['num_grad_final']:.2e} "
              f"fraud_synth={result['fraud_rate_synth']:.4f}")
        print(f"  AUPRC: base={result['auprc_base']:.4f} "
              f"synth={result['auprc_synth']:.4f} "
              f"aug={result['auprc_aug']:.4f} "
              f"→ Δ(aug)={result['delta_aug']:+.4f}  [{elapsed:.0f}s]")

        writer.writerow(result)
        fout.flush()

    fout.close()
    print(f"\n[Sweep v1 complete] {n_total} runs → {csv_path}")
    aggregate_results(csv_path)


if __name__ == "__main__":
    import csv
    main()
