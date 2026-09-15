"""
AMLSim λ sweep — v1b (strong teacher + coherence-gap as primary metric).

Hypothesis: AMLSim teacher is strong (AUPRC ~0.88 vs Sparkov ~0.03).
Strong teacher → λ↑ → coherence_gap↓ (label-behavioral coupling improves).
Sparkov λ↑ → gap↑ = weak-teacher artifact, NOT thesis refutation.

Differences from sweep_sparkov_v1b.py (6 changes only):
  1. Data: AMLSim sequences (RECEIVER cat, d_num=1, d_cat=1)
  2. W in step units (AMLSim time = integer steps), sweep W∈{7, 30}
  3. n_cat_classes=[256] (RECEIVER top-255+OTHER)
  4. Primary metric: coherence_gap[vel/fanout/amt], not ΔAUPRC_aug
  5. Teacher AUPRC gate: check ≥0.5, expect ~0.8+ (strong teacher signal)
  6. keep fixed at 1.0 (full data) — scarcity not the variable here

Grid: λ∈{0,0.1,0.5,1,2} × W∈{7,30} × seed∈{1,2,3} = 30 runs
Primary output: λ↑ → coh_gap_vel/fanout/amt ↓ ?

Output: results/amlsim_sweep_v1b.csv
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
from sklearn.metrics import average_precision_score

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
    run_tstr, coherence_gap,
)


# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_GRID = [0.0, 0.1, 0.5, 1.0, 2.0]
W_GRID      = [7.0, 30.0]        # step units (AMLSim TIMESTAMP is discrete steps)
SEED_GRID   = [1, 2, 3]

N_STEPS       = 20000
TEACHER_STEPS = 2000
T_DIFF        = 50
BATCH_SIZE    = 64
LR            = 3e-4
D_MODEL       = 128
TEMP          = 1.0
KEEP          = 1.0              # full data — scarcity not the variable here


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambdas",       type=float, nargs="+", default=LAMBDA_GRID)
    p.add_argument("--W_list",        type=float, nargs="+", default=W_GRID)
    p.add_argument("--seeds",         type=int,   nargs="+", default=SEED_GRID)
    p.add_argument("--n_steps",       type=int,   default=N_STEPS)
    p.add_argument("--teacher_steps", type=int,   default=TEACHER_STEPS)
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data_root",     type=str,
                   default=str(ROOT / "data" / "amlsim" / "sequences"))
    p.add_argument("--out_dir",       type=str, default=str(ROOT / "results"))
    p.add_argument("--resume",        action="store_true")
    return p.parse_args()


def load_amlsim(data_root: str):
    root = Path(data_root)
    with open(root / "meta.json") as f:
        meta = json.load(f)
    def _load(split):
        d = np.load(root / f"{split}.npz")
        return {k: torch.from_numpy(d[k]) for k in d}
    return meta, _load("train"), _load("test")


def seq_label(y, mask):
    return ((y.float() * mask.float()).max(dim=1).values > 0).float()


def teacher_auprc(f_phi, x_num, dt_bin, x_cat, y, mask, tau, W, temp, Bbins, n_cat, g_std, device,
                   batch_size=64):
    """Measure teacher AUPRC on training data (g → P(fraud)), batched."""
    chunks = []
    with torch.no_grad():
        for s in range(0, len(x_num), batch_size):
            e = min(s + batch_size, len(x_num))
            chunks.append(compute_g_from_real(
                x_num[s:e].to(device), dt_bin[s:e].to(device), x_cat[s:e].to(device),
                tau, W, temp, Bbins, n_cat,
                valid_mask=mask[s:e].to(device),
            ).cpu())
        g_real = torch.cat(chunks, dim=0)
        mask_f = mask.float()
        g_seq = (g_real * mask_f.unsqueeze(-1)).sum(1) / mask_f.sum(1, keepdim=True).clamp(min=1)
        g_norm = g_seq / g_std.cpu()
        logits = f_phi.cpu()(g_norm)
        probs  = torch.sigmoid(logits).numpy()
    y_seq = seq_label(y, mask).numpy()
    if y_seq.sum() < 2:
        return 0.0
    return float(average_precision_score(y_seq, probs))


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_single(meta, train, test, coh_lambda, W, seed, n_steps, teacher_steps, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tau   = torch.tensor(meta["tau_k"], dtype=torch.float32).to(device)
    Bbins = len(meta["tau_k"])
    d_num = meta["d_num"]           # 1 (TX_AMOUNT_LOG)
    n_cat = list(meta["num_classes_cat"].values())  # [256]
    L     = meta["L"]               # 32

    x_num_full  = train["x_num"].float()
    dt_bin_full = train["dt_bin"].long()
    x_cat_full  = train["x_cat"].long()
    y_full      = train["y"].float()
    mask_full   = train["mask"].bool()
    y_seq_full  = seq_label(y_full, mask_full).numpy()

    # Full data (no scarcity)
    x_num  = x_num_full;  dt_bin = dt_bin_full
    x_cat  = x_cat_full;  y      = y_full
    mask   = mask_full;   y_seq  = y_seq_full
    N = x_num.shape[0]

    n_fraud   = int(y_seq.sum())
    real_prev = n_fraud / N

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
    t_auprc = teacher_auprc(
        f_phi, x_num, dt_bin, x_cat, y, mask,
        tau, W, TEMP, Bbins, n_cat, g_std, device,
    )
    print(f"  f_phi BCE={teacher_bce:.4f}  teacher_auprc={t_auprc:.4f}  real_prev={real_prev:.4f}")
    if t_auprc < 0.5:
        print(f"  [WARN] teacher AUPRC={t_auprc:.4f} < 0.5 — weak teacher, results unreliable")

    # ── CoF-SeqGen v1b training ───────────────────────────────────────────────
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
    x_num_chunks    = []
    bin_pred_chunks = []
    cat_pred_chunks = []
    y_logit_chunks  = []
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
        y_logit_chunks.append(y_logit)
        mask_chunks.append(mask_samp)

    x_num_synth  = torch.cat(x_num_chunks, dim=0)
    dt_bin_synth = torch.cat(bin_pred_chunks, dim=0)
    x_cat_synth  = torch.cat(cat_pred_chunks, dim=0).unsqueeze(-1)
    y_logit_all  = torch.cat(y_logit_chunks, dim=0)
    mask_synth   = torch.cat(mask_chunks, dim=0)

    # ── Entity-level calibration ─────────────────────────────────────────────
    y_gen_uncalib  = (torch.rand_like(y_logit_all) < torch.sigmoid(y_logit_all)).long()
    fr_uncalib_seq = ((y_gen_uncalib.float() * mask_synth.float()).max(dim=1).values > 0)
    fraud_rate_before = fr_uncalib_seq.float().mean().item()

    b_calib = calibrate_temperature_entity(y_logit_all, mask_synth.bool(), real_prev)

    y_gen_calib = (
        torch.rand(y_logit_all.shape) < torch.sigmoid(y_logit_all - b_calib)
    ).long()
    y_gen_seq = (
        (y_gen_calib.float() * mask_synth.float()).max(dim=1).values > 0
    ).float()
    fraud_rate_synth = y_gen_seq.mean().item()

    print(f"  b_calib={b_calib:.4f}  fraud_before={fraud_rate_before:.4f}  "
          f"fraud_after={fraud_rate_synth:.4f}  target={real_prev:.4f}")

    if abs(fraud_rate_synth - real_prev) > real_prev:
        print(f"  [WARN] calibration: fraud_rate={fraud_rate_synth:.4f} vs target={real_prev:.4f} (>2× off)")

    # Amount quality
    bool_mask_r = mask.numpy().astype(bool)
    bool_mask_s = mask_synth.numpy().astype(bool)
    real_vals   = x_num.numpy()[:, :, 0][bool_mask_r]
    synth_vals  = x_num_synth.numpy()[:, :, 0][bool_mask_s]
    amt_shift   = abs(real_vals.mean() - synth_vals.mean()) / (real_vals.std() + 1e-8)

    # ── Coherence-gap measurement ─────────────────────────────────────────────
    # Batch g computation to avoid OOM with large n_cat (K=256)
    tau_np = tau.cpu().numpy()

    def batched_g(xn, db, xc, valid_mask, bs=BATCH_SIZE):
        chunks = []
        with torch.no_grad():
            for s in range(0, len(xn), bs):
                e = min(s + bs, len(xn))
                chunks.append(compute_g_from_real(
                    xn[s:e].to(device), db[s:e].to(device), xc[s:e].to(device),
                    tau, W, TEMP, Bbins, n_cat,
                    valid_mask=valid_mask[s:e].to(device),
                ).cpu())
        return torch.cat(chunks, dim=0)

    g_real_  = batched_g(x_num, dt_bin, x_cat, mask)
    g_synth_ = batched_g(x_num_synth, dt_bin_synth, x_cat_synth, mask_synth)

    mr = mask.unsqueeze(-1).float()
    ms = mask_synth.unsqueeze(-1).float()
    g_real_seq  = (g_real_  * mr).sum(1) / mr.sum(1).clamp(min=1)
    g_synth_seq = (g_synth_ * ms).sum(1) / ms.sum(1).clamp(min=1)

    g_real_np  = g_real_seq.numpy()
    g_synth_np = g_synth_seq.numpy()
    y_real_seq = y_seq.astype(np.float32)
    y_synth_np = y_gen_seq.numpy()

    coh_gap_vel    = coherence_gap(g_synth_np, y_synth_np, g_real_np, y_real_seq, feat_idx=0)
    coh_gap_fanout = coherence_gap(g_synth_np, y_synth_np, g_real_np, y_real_seq, feat_idx=2)
    coh_gap_amt    = coherence_gap(g_synth_np, y_synth_np, g_real_np, y_real_seq, feat_idx=3)

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
        y_gen_calib.numpy(),
        x_cat_synth.numpy(),
        **beh_kw,
    )

    X_aug = np.concatenate([X_real, X_synth], axis=0)
    y_aug = np.concatenate([y_real_lab, y_synth_lab], axis=0)

    res_base  = run_tstr(X_real,  y_real_lab,  X_test, y_test, random_state=seed)
    res_synth = run_tstr(X_synth, y_synth_lab, X_test, y_test, random_state=seed)
    res_aug   = run_tstr(X_aug,   y_aug,        X_test, y_test, random_state=seed)

    return {
        "coh_lambda":        coh_lambda,
        "W":                 W,
        "seed":              seed,
        "n_train":           N,
        "n_fraud_train":     n_fraud,
        "real_prevalence":   round(real_prev, 5),
        "teacher_bce":       round(teacher_bce, 4),
        "teacher_auprc":     round(t_auprc, 4),
        "b_calib":           round(b_calib, 4),
        "fraud_rate_before": round(fraud_rate_before, 4),
        "fraud_rate_synth":  round(fraud_rate_synth, 4),
        "L_diff_final":      round(L_diff_final, 4),
        "L_coh_final":       round(L_coh_final, 4),
        "num_grad_final":    round(num_grad_final, 4),
        "amt_shift":         round(float(amt_shift), 4),
        "coh_gap_vel":       round(float(coh_gap_vel),    6),
        "coh_gap_fanout":    round(float(coh_gap_fanout), 6),
        "coh_gap_amt":       round(float(coh_gap_amt),    6),
        "auprc_base":        round(res_base["auprc"],  4),
        "auprc_synth":       round(res_synth["auprc"], 4),
        "auprc_aug":         round(res_aug["auprc"],   4),
        "auroc_base":        round(res_base["auroc"],  4),
        "auroc_aug":         round(res_aug["auroc"],   4),
        "delta_synth":       round(res_synth["auprc"] - res_base["auprc"], 4),
        "delta_aug":         round(res_aug["auprc"]   - res_base["auprc"], 4),
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
        groups[(float(r["coh_lambda"]), float(r["W"]))].append(r)

    agg_path = csv_path.parent / (csv_path.stem + "_agg.csv")
    fields = [
        "coh_lambda", "W", "n_seeds",
        "coh_gap_vel_mean", "coh_gap_vel_std",
        "coh_gap_fanout_mean", "coh_gap_fanout_std",
        "coh_gap_amt_mean", "coh_gap_amt_std",
        "delta_synth_mean", "delta_synth_std",
        "delta_aug_mean", "delta_aug_std",
        "fraud_rate_synth_mean", "b_calib_mean",
        "teacher_auprc_mean", "auprc_base_mean", "auprc_synth_mean",
        "L_coh_final_mean",
    ]
    with open(agg_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (lam, W), g in sorted(groups.items()):
            def _m(c): return float(np.mean([float(r[c]) for r in g]))
            def _s(c): return float(np.std( [float(r[c]) for r in g]))
            w.writerow({
                "coh_lambda": lam, "W": W, "n_seeds": len(g),
                "coh_gap_vel_mean":     round(_m("coh_gap_vel"),    6),
                "coh_gap_vel_std":      round(_s("coh_gap_vel"),    6),
                "coh_gap_fanout_mean":  round(_m("coh_gap_fanout"), 6),
                "coh_gap_fanout_std":   round(_s("coh_gap_fanout"), 6),
                "coh_gap_amt_mean":     round(_m("coh_gap_amt"),    6),
                "coh_gap_amt_std":      round(_s("coh_gap_amt"),    6),
                "delta_synth_mean":     round(_m("delta_synth"),    4),
                "delta_synth_std":      round(_s("delta_synth"),    4),
                "delta_aug_mean":       round(_m("delta_aug"),      4),
                "delta_aug_std":        round(_s("delta_aug"),      4),
                "fraud_rate_synth_mean":round(_m("fraud_rate_synth"), 4),
                "b_calib_mean":         round(_m("b_calib"),        4),
                "teacher_auprc_mean":   round(_m("teacher_auprc"),  4),
                "auprc_base_mean":      round(_m("auprc_base"),     4),
                "auprc_synth_mean":     round(_m("auprc_synth"),    4),
                "L_coh_final_mean":     round(_m("L_coh_final"),    5),
            })

    print(f"\n[AMLSim v1b Aggregated] {agg_path}")
    print(f'\n{"λ":>5} {"W":>5} {"n":>2}  {"coh_gap_vel":>12}  '
          f'{"coh_gap_fanout":>14}  {"coh_gap_amt":>12}  '
          f'{"ΔAUPRC_synth":>13}  {"teacher_ap":>10}')
    print("─" * 90)
    with open(agg_path) as f:
        for row in _csv.DictReader(f):
            print(
                f'{float(row["coh_lambda"]):>5.1f} '
                f'{float(row["W"]):>5.0f} '
                f'{row["n_seeds"]:>2}  '
                f'{float(row["coh_gap_vel_mean"]):>12.6f}  '
                f'{float(row["coh_gap_fanout_mean"]):>14.6f}  '
                f'{float(row["coh_gap_amt_mean"]):>12.6f}  '
                f'{float(row["delta_synth_mean"]):>+7.4f}±{float(row["delta_synth_std"]):.4f}  '
                f'{float(row["teacher_auprc_mean"]):>10.4f}'
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "amlsim_sweep_v1b.csv"

    completed = set()
    if args.resume and csv_path.exists():
        import csv as _csv
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                completed.add((float(row["coh_lambda"]), float(row["W"]), int(row["seed"])))
        print(f"[Resume] {len(completed)} done.")

    grid    = list(product(sorted(args.lambdas), sorted(args.W_list), sorted(args.seeds)))
    n_total = len(grid)
    meta, train, test = load_amlsim(args.data_root)
    n_cat  = list(meta["num_classes_cat"].values())
    Bbins  = len(meta["tau_k"])
    tau_np = np.array(meta["tau_k"], dtype=np.float32)

    print(f"[AMLSim v1b] {n_total} runs  n_cat={n_cat}  d_num={meta['d_num']}  L={meta['L']}")

    # Full-train upper bound (TSTR reference)
    beh_kw_ref = dict(tau=tau_np, W=args.W_list[0], temp=TEMP, Bbins=Bbins, n_cat_classes=n_cat)
    X_test_ref, y_test_ref = extract_features_with_behavioral(
        test["x_num"].float().numpy(), test["dt_bin"].long().numpy(),
        test["mask"].bool().numpy(), test["y"].float().numpy(),
        test["x_cat"].long().numpy(), **beh_kw_ref,
    )
    X_full_ref, y_full_ref = extract_features_with_behavioral(
        train["x_num"].float().numpy(), train["dt_bin"].long().numpy(),
        train["mask"].bool().numpy(), train["y"].float().numpy(),
        train["x_cat"].long().numpy(), **beh_kw_ref,
    )
    auprc_full = run_tstr(X_full_ref, y_full_ref, X_test_ref, y_test_ref, random_state=0)["auprc"]
    print(f"[Upper bound] Full-train AUPRC (W={args.W_list[0]}) = {auprc_full:.4f}")

    fieldnames = [
        "coh_lambda", "W", "seed", "n_train", "n_fraud_train",
        "real_prevalence", "teacher_bce", "teacher_auprc", "b_calib",
        "fraud_rate_before", "fraud_rate_synth",
        "L_diff_final", "L_coh_final", "num_grad_final", "amt_shift",
        "coh_gap_vel", "coh_gap_fanout", "coh_gap_amt",
        "auprc_base", "auprc_synth", "auprc_aug",
        "auroc_base", "auroc_aug",
        "delta_synth", "delta_aug", "auprc_full",
    ]
    # "a" mode only when resuming; otherwise start fresh to avoid duplicate headers
    fout   = open(csv_path, "a" if args.resume else "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if not args.resume:
        writer.writeheader()
    fout.flush()

    t0_sweep = time.time()
    for run_idx, (lam, W, seed) in enumerate(grid, 1):
        if (lam, W, seed) in completed:
            print(f"[{run_idx}/{n_total}] SKIP λ={lam} W={W} seed={seed}")
            continue
        print(f"\n[{run_idx}/{n_total}] λ={lam} W={W} seed={seed} "
              f"— elapsed {(time.time()-t0_sweep)/60:.1f}min")
        t0 = time.time()
        result = run_single(
            meta, train, test, coh_lambda=lam, W=W, seed=seed,
            n_steps=args.n_steps, teacher_steps=args.teacher_steps,
            device=args.device,
        )
        result["auprc_full"] = round(auprc_full, 4)
        elapsed = time.time() - t0
        print(f"  coh_gap vel={result['coh_gap_vel']:.4f} "
              f"fanout={result['coh_gap_fanout']:.4f} "
              f"amt={result['coh_gap_amt']:.4f}  "
              f"ΔAUPRC_synth={result['delta_synth']:+.4f}  [{elapsed:.0f}s]")
        writer.writerow(result)
        fout.flush()

    fout.close()
    print(f"\n[AMLSim v1b complete] {n_total} runs → {csv_path}")
    aggregate_results(csv_path)


if __name__ == "__main__":
    import csv
    main()
