"""
Sparkov λ sweep — v1b (calibrated label generation + coherence-gap measurement).

v1b fixes vs v1:
  [FIX-1] label BCE weight 1.0 (was 0.1) → model label head trained harder.
  [FIX-2] Post-hoc temperature calibration: find T s.t.
           mean(sigmoid(y_logit/T)) = real_prevalence on generated pool.
           → fraud_rate_synth ≈ real 1.62% (gate check).
  [MEASURE] coherence_gap[vel] and coherence_gap[amt_sum] per run.
            Primary metric: does λ↑ → coherence_gap↓?

Grid: λ∈{0,0.1,0.5,1,2} × keep∈{0.1,0.2,0.4} × seed∈{1,2,3} = 45 runs
Output: results/sparkov_sweep_v1b.csv
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
from models.teacher import compute_g_stats_from_data, compute_g_from_real
from models.seq_denoiser import SeqDenoiser
from models.cof_seqgen import CoFSeqGen
from models.sampler import (
    ddim_sample, calibrate_temperature_entity,
    sample_empirical_dt_bin, sample_empirical_x_cat,
)
from eval.tstr import (
    extract_features_with_behavioral,
    run_tstr, stratified_keep, coherence_gap,
)


# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_GRID   = [0.0, 0.1, 0.5, 1.0, 2.0]
KEEP_GRID     = [0.1, 0.2, 0.4]
SEED_GRID     = [1, 2, 3]

N_STEPS       = 20000
TEACHER_STEPS = 2000
T_DIFF        = 50
BATCH_SIZE    = 64
LR            = 3e-4
D_MODEL       = 128
W             = 60.0
TEMP          = 1.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambdas",       type=float, nargs="+", default=LAMBDA_GRID)
    p.add_argument("--keeps",         type=float, nargs="+", default=KEEP_GRID)
    p.add_argument("--seeds",         type=int,   nargs="+", default=SEED_GRID)
    p.add_argument("--n_steps",       type=int,   default=N_STEPS)
    p.add_argument("--teacher_steps", type=int,   default=TEACHER_STEPS)
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data_root",     type=str,
                   default=str(ROOT / "data" / "sparkov" / "sequences"))
    p.add_argument("--out_dir",       type=str, default=str(ROOT / "results"))
    p.add_argument("--resume",        action="store_true")
    return p.parse_args()


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

def run_single(meta, train, test, coh_lambda, keep, seed, n_steps, teacher_steps, device):
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

    scarce_idx = stratified_keep(len(y_seq_full), y_seq_full, keep=keep, seed=seed)
    x_num  = x_num_full[scarce_idx];   dt_bin = dt_bin_full[scarce_idx]
    x_cat  = x_cat_full[scarce_idx];   y      = y_full[scarce_idx]
    mask   = mask_full[scarce_idx];    y_seq  = y_seq_full[scarce_idx]
    N = x_num.shape[0];                L = x_num.shape[1]

    n_fraud   = int(y_seq.sum())
    real_prev = n_fraud / N          # entity-level prevalence

    # ── g_std ────────────────────────────────────────────────────────────────
    g_mean, g_std = compute_g_stats_from_data(
        x_num, dt_bin, x_cat, tau.cpu(), W, TEMP, Bbins, n_cat, batch_size=256,
        valid_mask=mask,
    )
    g_std = g_std.to(device)

    # ── CoherenceTeacher f_φ ─────────────────────────────────────────────────
    f_phi = CoherenceTeacher(hidden=64).to(device)
    t_losses = pretrain_coherence_teacher(
        f_phi, x_num, dt_bin, x_cat, y, mask,
        tau, W, TEMP, Bbins, n_cat, g_std.cpu(),
        steps=teacher_steps, lr=1e-3, batch_size=256, device=device,
    )
    teacher_bce = float(np.mean(t_losses[-50:]))
    print(f"  f_phi BCE={teacher_bce:.4f}  real_prev={real_prev:.4f}")

    # ── CoF-SeqGen v1b training (label BCE weight=1.0) ───────────────────────
    denoiser = SeqDenoiser(
        d_num=d_num, Bbins=Bbins, n_cat_classes=n_cat,
        d_model=D_MODEL, n_heads=4, n_layers=2, L_max=64,
    )
    model = CoFSeqGen(
        denoiser=denoiser, tau=tau.cpu(), W=W, temp=TEMP,
        coh_lambda=coh_lambda, n_cat_classes=n_cat,
        coherence_teacher=f_phi, g_std=g_std.cpu(),
    )
    model.train().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    loss_hist = {"L_diff": [], "L_coh": [], "num_grad": []}
    for step in range(1, n_steps + 1):
        idx    = torch.randint(0, N, (BATCH_SIZE,))
        t_frac = torch.rand(1).item() * 0.9 + 0.1
        optim.zero_grad()
        loss, info = model.compute_loss(
            x_num[idx].to(device), dt_bin[idx].to(device),
            x_cat[idx].to(device), y[idx].to(device),
            mask[idx].to(device), t_frac,
        )
        loss.backward()
        gn = model.denoiser.num_head.weight.grad
        num_grad_norm = gn.norm().item() if gn is not None else 0.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step % 2000 == 0 or step == n_steps:
            loss_hist["L_diff"].append(info["L_diff"])
            loss_hist["L_coh"].append(info["L_coh"])
            loss_hist["num_grad"].append(num_grad_norm)

    L_diff_final   = float(np.mean(loss_hist["L_diff"][-3:]))
    L_coh_final    = float(np.mean(loss_hist["L_coh"][-3:]))
    num_grad_final = float(np.mean(loss_hist["num_grad"][-3:]))

    # ── Generate synthetic sequences ─────────────────────────────────────────
    model.eval()
    x_num_chunks   = []
    bin_pred_chunks = []
    cat_pred_chunks = []
    y_logit_chunks  = []   # raw logits for calibration
    mask_chunks     = []

    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        bs  = end - start
        dt_samp   = sample_empirical_dt_bin(dt_bin, bs, L, Bbins, device=device)
        xcat_samp = sample_empirical_x_cat(x_cat, bs, device=device)
        mask_idx  = torch.randint(0, N, (bs,))
        mask_samp = mask[mask_idx]

        x_gen, bin_pred, cat_preds, _, y_logit = ddim_sample(
            model, dt_samp, xcat_samp,
            d_num=d_num, T_steps=T_DIFF, device=device, return_discrete=True,
            valid_mask=mask_samp.to(device),
        )
        x_num_chunks.append(x_gen.cpu())
        bin_pred_chunks.append(bin_pred.cpu())
        cat_pred_chunks.append(
            cat_preds[0].cpu() if cat_preds else torch.zeros(bs, L, dtype=torch.long)
        )
        y_logit_chunks.append(y_logit)   # already on CPU from sampler
        mask_chunks.append(mask_samp)

    x_num_synth  = torch.cat(x_num_chunks, dim=0)                        # (N, L, d_num)
    dt_bin_synth = torch.cat(bin_pred_chunks, dim=0)                      # (N, L)
    x_cat_synth  = torch.cat(cat_pred_chunks, dim=0).unsqueeze(-1)        # (N, L, 1)
    y_logit_all  = torch.cat(y_logit_chunks, dim=0)                       # (N, L)
    mask_synth   = torch.cat(mask_chunks, dim=0)                          # (N, L)

    # ── [FIX-2] Temperature calibration ──────────────────────────────────────
    # Uncalibrated entity-level fraud rate (for logging)
    y_gen_uncalib  = (torch.rand_like(y_logit_all) < torch.sigmoid(y_logit_all)).long()
    fr_uncalib_seq = ((y_gen_uncalib.float() * mask_synth.float()).max(dim=1).values > 0)
    fraud_rate_before = fr_uncalib_seq.float().mean().item()

    # Entity-level calibration: find bias b s.t.
    #   mean_entity(1 - prod_{positions}(1 - sigmoid(y_logit - b))) = real_prev
    # Bias shift is guaranteed monotone; temperature scaling is not (gets stuck
    # when entities always have at least one positive logit at T→0).
    b_calib = calibrate_temperature_entity(y_logit_all, mask_synth.bool(), real_prev)

    # Resample with calibrated bias: sigmoid(logit - b)
    y_gen_calib = (
        torch.rand(y_logit_all.shape) < torch.sigmoid(y_logit_all - b_calib)
    ).long()
    y_gen_seq = (
        (y_gen_calib.float() * mask_synth.float()).max(dim=1).values > 0
    ).float()
    fraud_rate_synth = y_gen_seq.mean().item()

    print(f"  b_calib={b_calib:.4f}  fraud_before={fraud_rate_before:.4f}  "
          f"fraud_after={fraud_rate_synth:.4f}  target={real_prev:.4f}")

    # Gate: warn if calibration still far off
    if abs(fraud_rate_synth - real_prev) > real_prev:
        print(f"  [WARN] calibration gate: fraud_rate={fraud_rate_synth:.4f} "
              f"vs target={real_prev:.4f} (>2× off)")

    # Amount quality
    bool_mask_r = mask.numpy().astype(bool)
    bool_mask_s = mask_synth.numpy().astype(bool)
    real_vals   = x_num.numpy()[:, :, 0][bool_mask_r]
    synth_vals  = x_num_synth.numpy()[:, :, 0][bool_mask_s]
    amt_shift   = abs(real_vals.mean() - synth_vals.mean()) / (real_vals.std() + 1e-8)

    # ── Coherence-gap measurement ─────────────────────────────────────────────
    # Entity-level behavioral summary (mean over sequence positions)
    tau_np = tau.cpu().numpy()
    with torch.no_grad():
        g_real_  = compute_g_from_real(
            x_num.to(device), dt_bin.to(device), x_cat.to(device),
            tau, W, TEMP, Bbins, n_cat,
            valid_mask=mask.to(device),
        ).cpu()
        g_synth_ = compute_g_from_real(
            x_num_synth.to(device), dt_bin_synth.to(device), x_cat_synth.to(device),
            tau, W, TEMP, Bbins, n_cat,
            valid_mask=mask_synth.to(device),
        ).cpu()

    # Mask out padding before computing entity-level mean
    mr = mask.unsqueeze(-1).float()
    ms = mask_synth.unsqueeze(-1).float()
    g_real_seq  = (g_real_  * mr).sum(1) / mr.sum(1).clamp(min=1)  # (N, 4)
    g_synth_seq = (g_synth_ * ms).sum(1) / ms.sum(1).clamp(min=1)  # (N, 4)

    g_real_np  = g_real_seq.numpy()
    g_synth_np = g_synth_seq.numpy()
    y_real_seq = y_seq.astype(np.float32)         # real entity labels
    y_synth_np = y_gen_seq.numpy()                # calibrated synthetic labels

    coh_gap_vel = coherence_gap(g_synth_np, y_synth_np, g_real_np, y_real_seq, feat_idx=0)
    coh_gap_amt = coherence_gap(g_synth_np, y_synth_np, g_real_np, y_real_seq, feat_idx=3)

    # ── STEP-5 diagnostic: velocity binwise fraud rate ────────────────────────
    # Hypothesis: λ↑ → synth_vel_gap→0 = teacher flattens the relationship
    #             (weak-teacher artifact). λ=0 synth closest to real pattern.
    thr = float(np.median(g_real_np[:, 0]))
    r_lo = float(y_real_seq[g_real_np[:, 0] <= thr].mean()) if (g_real_np[:, 0] <= thr).any() else 0.0
    r_hi = float(y_real_seq[g_real_np[:, 0] >  thr].mean()) if (g_real_np[:, 0] >  thr).any() else 0.0
    s_lo = float(y_synth_np[g_synth_np[:, 0] <= thr].mean()) if (g_synth_np[:, 0] <= thr).any() else 0.0
    s_hi = float(y_synth_np[g_synth_np[:, 0] >  thr].mean()) if (g_synth_np[:, 0] >  thr).any() else 0.0
    print(f"  [diag] lam={coh_lambda} keep={keep} | "
          f"REAL low={r_lo:.4f} high={r_hi:.4f} (gap {r_hi-r_lo:+.4f}) | "
          f"SYNTH low={s_lo:.4f} high={s_hi:.4f} (gap {s_hi-s_lo:+.4f})")

    # ── TSTR ─────────────────────────────────────────────────────────────────
    x_num_te  = test["x_num"].float().numpy()
    dt_bin_te = test["dt_bin"].long().numpy()
    x_cat_te  = test["x_cat"].long().numpy()
    mask_te   = test["mask"].bool().numpy()
    y_te      = test["y"].float().numpy()

    beh_kw = dict(tau=tau_np, W=W, temp=TEMP, Bbins=Bbins, n_cat_classes=n_cat)

    X_test, y_test = extract_features_with_behavioral(
        x_num_te, dt_bin_te, mask_te, y_te, x_cat_te, **beh_kw,
    )
    X_real, y_real_lab = extract_features_with_behavioral(
        x_num.numpy(), dt_bin.numpy(), mask.numpy(), y.numpy(), x_cat.numpy(), **beh_kw,
    )
    X_synth, y_synth_lab = extract_features_with_behavioral(
        x_num_synth.numpy(),
        dt_bin_synth.numpy(),
        mask_synth.numpy(),
        y_gen_calib.numpy(),   # calibrated synthetic labels
        x_cat_synth.numpy(),
        **beh_kw,
    )

    X_aug = np.concatenate([X_real, X_synth], axis=0)
    y_aug = np.concatenate([y_real_lab, y_synth_lab], axis=0)

    res_base  = run_tstr(X_real,  y_real_lab,  X_test, y_test, random_state=seed)
    res_synth = run_tstr(X_synth, y_synth_lab, X_test, y_test, random_state=seed)
    res_aug   = run_tstr(X_aug,   y_aug,        X_test, y_test, random_state=seed)

    return {
        "coh_lambda":       coh_lambda,
        "keep":             keep,
        "seed":             seed,
        "n_scarce":         N,
        "n_fraud_scarce":   n_fraud,
        "real_prevalence":  round(real_prev, 4),
        "teacher_bce":      round(teacher_bce, 4),
        "b_calib":          round(b_calib, 4),
        "fraud_rate_before":round(fraud_rate_before, 4),
        "fraud_rate_synth": round(fraud_rate_synth, 4),
        "L_diff_final":     round(L_diff_final, 4),
        "L_coh_final":      round(L_coh_final, 4),
        "num_grad_final":   round(num_grad_final, 4),
        "amt_shift":        round(float(amt_shift), 4),
        "coh_gap_vel":      round(float(coh_gap_vel), 6),
        "coh_gap_amt":      round(float(coh_gap_amt), 6),
        "synth_vel_gap":    round(float(s_hi - s_lo), 4),
        "real_vel_gap":     round(float(r_hi - r_lo), 4),
        "auprc_base":       round(res_base["auprc"],  4),
        "auprc_synth":      round(res_synth["auprc"], 4),
        "auprc_aug":        round(res_aug["auprc"],   4),
        "auroc_base":       round(res_base["auroc"],  4),
        "auroc_aug":        round(res_aug["auroc"],   4),
        "delta_synth":      round(res_synth["auprc"] - res_base["auprc"], 4),
        "delta_aug":        round(res_aug["auprc"]   - res_base["auprc"], 4),
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
        groups[(float(r["coh_lambda"]), float(r["keep"]))].append(r)

    agg_path = csv_path.parent / (csv_path.stem + "_agg.csv")
    fields = [
        "coh_lambda", "keep", "n_seeds",
        "delta_aug_mean", "delta_aug_std",
        "coh_gap_vel_mean", "coh_gap_vel_std",
        "coh_gap_amt_mean", "coh_gap_amt_std",
        "fraud_rate_synth_mean", "b_calib_mean",
        "auprc_base_mean", "auprc_aug_mean",
        "L_coh_final_mean",
    ]
    with open(agg_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (lam, keep), g in sorted(groups.items()):
            def _m(c): return float(np.mean([float(r[c]) for r in g]))
            def _s(c): return float(np.std( [float(r[c]) for r in g]))
            w.writerow({
                "coh_lambda": lam, "keep": keep, "n_seeds": len(g),
                "delta_aug_mean":       round(_m("delta_aug"),      4),
                "delta_aug_std":        round(_s("delta_aug"),      4),
                "coh_gap_vel_mean":     round(_m("coh_gap_vel"),    6),
                "coh_gap_vel_std":      round(_s("coh_gap_vel"),    6),
                "coh_gap_amt_mean":     round(_m("coh_gap_amt"),    6),
                "coh_gap_amt_std":      round(_s("coh_gap_amt"),    6),
                "fraud_rate_synth_mean":round(_m("fraud_rate_synth"), 4),
                "b_calib_mean":         round(_m("b_calib"),        4),
                "auprc_base_mean":      round(_m("auprc_base"),     4),
                "auprc_aug_mean":       round(_m("auprc_aug"),      4),
                "L_coh_final_mean":     round(_m("L_coh_final"),    5),
            })

    print(f"\n[v1b Aggregated] {agg_path}")
    print(f'\n{"λ":>5} {"keep":>5} {"n":>2}  {"ΔAUPRC_aug":>13}  '
          f'{"coh_gap_vel":>12}  {"coh_gap_amt":>12}  {"fraud_synth":>11}  {"T":>6}')
    print("─" * 80)
    with open(agg_path) as f:
        for row in _csv.DictReader(f):
            print(
                f'{float(row["coh_lambda"]):>5.1f} '
                f'{float(row["keep"]):>5.2f} '
                f'{row["n_seeds"]:>2}  '
                f'{float(row["delta_aug_mean"]):>+7.4f}±{float(row["delta_aug_std"]):.4f}  '
                f'{float(row["coh_gap_vel_mean"]):>12.6f}  '
                f'{float(row["coh_gap_amt_mean"]):>12.6f}  '
                f'{float(row["fraud_rate_synth_mean"]):>11.4f}  '
                f'{float(row["b_calib_mean"]):>6.3f}'
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sparkov_sweep_v1b.csv"

    completed = set()
    if args.resume and csv_path.exists():
        import csv as _csv
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                completed.add((float(row["coh_lambda"]), float(row["keep"]), int(row["seed"])))
        print(f"[Resume] {len(completed)} done.")

    grid    = list(product(sorted(args.lambdas), sorted(args.keeps), sorted(args.seeds)))
    n_total = len(grid)
    meta, train, test = load_sparkov(args.data_root)
    n_cat  = list(meta["num_classes_cat"].values())
    Bbins  = len(meta["tau_k"])
    tau_np = np.array(meta["tau_k"], dtype=np.float32)

    # Full-train upper bound
    beh_kw = dict(tau=tau_np, W=W, temp=TEMP, Bbins=Bbins, n_cat_classes=n_cat)
    X_test, y_test = extract_features_with_behavioral(
        test["x_num"].float().numpy(), test["dt_bin"].long().numpy(),
        test["mask"].bool().numpy(), test["y"].float().numpy(),
        test["x_cat"].long().numpy(), **beh_kw,
    )
    X_full, y_full_lab = extract_features_with_behavioral(
        train["x_num"].float().numpy(), train["dt_bin"].long().numpy(),
        train["mask"].bool().numpy(), train["y"].float().numpy(),
        train["x_cat"].long().numpy(), **beh_kw,
    )
    auprc_full = run_tstr(X_full, y_full_lab, X_test, y_test, random_state=0)["auprc"]
    print(f"[Upper bound v1b] Full-train AUPRC = {auprc_full:.4f}")

    fieldnames = [
        "coh_lambda", "keep", "seed", "n_scarce", "n_fraud_scarce",
        "real_prevalence", "teacher_bce", "b_calib",
        "fraud_rate_before", "fraud_rate_synth",
        "L_diff_final", "L_coh_final", "num_grad_final", "amt_shift",
        "coh_gap_vel", "coh_gap_amt", "synth_vel_gap", "real_vel_gap",
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

    t0_sweep = time.time()
    for run_idx, (lam, keep, seed) in enumerate(grid, 1):
        if (lam, keep, seed) in completed:
            print(f"[{run_idx}/{n_total}] SKIP λ={lam} keep={keep} seed={seed}")
            continue
        print(f"\n[{run_idx}/{n_total}] λ={lam} keep={keep} seed={seed} "
              f"— elapsed {(time.time()-t0_sweep)/60:.1f}min")
        t0 = time.time()
        result = run_single(
            meta, train, test, coh_lambda=lam, keep=keep, seed=seed,
            n_steps=args.n_steps, teacher_steps=args.teacher_steps,
            device=args.device,
        )
        result["auprc_full"] = round(auprc_full, 4)
        elapsed = time.time() - t0
        print(f"  coh_gap vel={result['coh_gap_vel']:.4f} amt={result['coh_gap_amt']:.4f}  "
              f"ΔAUPRC={result['delta_aug']:+.4f}  [{elapsed:.0f}s]")
        writer.writerow(result)
        fout.flush()

    fout.close()
    print(f"\n[v1b complete] {n_total} runs → {csv_path}")
    aggregate_results(csv_path)


if __name__ == "__main__":
    import csv
    main()
