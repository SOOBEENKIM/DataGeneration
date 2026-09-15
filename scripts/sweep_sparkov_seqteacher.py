"""
Sparkov sweep with SequenceTeacher coherence (v1c) — C2b generality experiment.

Purpose:
  AMLSim Exp3 showed Δwindow=+0.174 mean (C2b supported, timing-dominant channel).
  This sweep tests whether C2b generalises to Sparkov (category-dominant channel).

Sparkov diagnostics (diag_sparkov_teacher.py):
  amount-only AUPRC=0.868 → +timing=+0.017 → +category=+0.058 → full=0.943
  Δwindow≈+0.075 at teacher level (different dominant channel vs AMLSim: category > timing)

Attribution arms:
  amount_only : use_time=False, use_recv=False  → pure amount signal
  amount_time : use_time=True,  use_recv=False  → amount + timing
  full        : use_time=True,  use_recv=True   → full window (amount+timing+category)

Attribution:
  Δtiming  = amount_time − amount_only   (timing contribution, expected ~small for Sparkov)
  Δcat     = full − amount_time          (category contribution, expected ~dominant for Sparkov)
  Δwindow  = full − amount_only          (total window, = Δtiming + Δcat)

Improvements from AMLSim Exp3 lessons:
  - seeds 3 → 10 (bootstrap CI on Δwindow needed; AMLSim std=0.23 unusable with 3 seeds)
  - λ=0 included as noise-floor control (full vs amount_only ≈0 at λ=0 → λ>0 signal is real)
  - collapse zone flagging: amount_only Δsynth < −0.3 → λ over-constrains amount coherence
  - coherence_gap vs λ recorded for all runs

Grid: λ∈{0,0.1,0.5,1,2} × teacher_type∈{amount_only,amount_time,full} × seed∈{1..10}
      = 5 × 3 × 10 = 150 runs (~8-10 hours)
Output: results/sparkov_seqteacher.csv
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

from models.coherence_teacher import SequenceTeacher, pretrain_sequence_teacher
from models.teacher import compute_g_from_real
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
LAMBDA_GRID   = [0.0, 0.1, 0.5, 1.0, 2.0]
W_FIXED       = 7.0          # same as AMLSim for consistency
SEED_GRID     = list(range(1, 11))   # 10 seeds for bootstrap CI
TEACHER_TYPES = ["amount_only", "amount_time", "full"]

N_STEPS       = 20000
TEACHER_STEPS = 2000
T_DIFF        = 50
BATCH_SIZE    = 64
LR            = 3e-4
D_MODEL       = 128
TEMP          = 1.0

# amount_only collapse threshold: flag if Δsynth < this (learning instability artifact)
COLLAPSE_THRESH = -0.3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambdas",       type=float, nargs="+", default=LAMBDA_GRID)
    p.add_argument("--W",             type=float,            default=W_FIXED)
    p.add_argument("--seeds",         type=int,   nargs="+", default=SEED_GRID)
    p.add_argument("--teacher_types", type=str,   nargs="+", default=TEACHER_TYPES)
    p.add_argument("--n_steps",       type=int,   default=N_STEPS)
    p.add_argument("--teacher_steps", type=int,   default=TEACHER_STEPS)
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data_root",     type=str,
                   default=str(ROOT / "data" / "sparkov" / "sequences"))
    p.add_argument("--out_dir",       type=str, default=str(ROOT / "results"))
    p.add_argument("--resume",        action="store_true")
    return p.parse_args()


def load_sparkov(data_root):
    root = Path(data_root)
    with open(root / "meta.json") as f:
        meta = json.load(f)
    def _load(split):
        d = np.load(root / f"{split}.npz")
        return {k: torch.from_numpy(d[k]) for k in d}
    return meta, _load("train"), _load("test")


def seq_label(y, mask):
    return ((y.float() * mask.float()).max(dim=1).values > 0).float()


def teacher_auprc_seq(f_phi_seq, x_num, dt_bin, x_cat, y_ent, mask,
                      Bbins, K_cat, device, batch_size=64):
    """Entity-level AUPRC via mean-pool over masked positions."""
    f_phi_seq.to(device).eval()
    all_probs = []
    with torch.no_grad():
        for s in range(0, len(x_num), batch_size):
            e   = min(s + batch_size, len(x_num))
            xn  = x_num[s:e].to(device)
            db  = dt_bin[s:e].to(device)
            xc  = x_cat[s:e].to(device)
            m   = mask[s:e].to(device).bool()
            bin_p = F.one_hot(db, num_classes=Bbins).float()
            cat_p = F.one_hot(xc[:, :, 0], num_classes=K_cat).float() if K_cat > 0 else None
            logit = f_phi_seq(xn, bin_p, cat_p, m)
            mf    = m.float()
            ent_logit = (logit * mf).sum(1) / mf.sum(1).clamp(min=1)
            all_probs.append(torch.sigmoid(ent_logit).cpu().numpy())
    probs = np.concatenate(all_probs)
    y_np  = y_ent.numpy()
    if y_np.sum() < 2:
        return 0.0
    return float(average_precision_score(y_np, probs))


_grad_gate_done = set()

def check_grad_gate(model, x_num, dt_bin, x_cat, y, mask, device, N, label="full"):
    """Verify all relevant heads get nonzero grad through L_coh (once per teacher_type)."""
    model.train()
    idx    = torch.randint(0, N, (BATCH_SIZE,))
    t_frac = 0.5
    model.zero_grad()
    loss, info = model.compute_loss(
        x_num[idx].to(device), dt_bin[idx].to(device),
        x_cat[idx].to(device), y[idx].to(device),
        mask[idx].to(device), t_frac,
    )
    loss.backward()
    dn = model.denoiser
    gn_num = dn.num_head.weight.grad
    gn_bin = dn.bin_head.weight.grad
    gn_cat = dn.cat_heads[0].weight.grad if dn.cat_heads else None
    gn_y   = dn.y_head.weight.grad
    n = lambda g: f"{g.norm().item():.4f}" if g is not None else "NONE"
    print(f"  [grad-gate teacher={label}] "
          f"num={n(gn_num)}  bin={n(gn_bin)}  cat={n(gn_cat)}  y={n(gn_y)}")
    assert gn_num is not None and gn_num.norm() > 0, "FAIL: num_head grad=0"
    assert gn_y   is not None and gn_y.norm()   > 0, "FAIL: y_head grad=0"
    if label in ("amount_time", "full"):
        assert gn_bin is not None and gn_bin.norm() > 0, \
            f"FAIL: bin_head grad=0 for {label}"
    if label == "full" and gn_cat is not None:
        assert gn_cat.norm() > 0, "FAIL: cat_head grad=0 (category C2b dead)"
    print(f"  [grad-gate teacher={label}] PASS")
    model.zero_grad()


# ─────────────────────────────────────────────────────────────────────────────

def run_single(meta, train, test, coh_lambda, W, seed, teacher_type,
               n_steps, teacher_steps, device,
               X_real_cache=None, X_test_cache=None):
    global _grad_gate_done

    torch.manual_seed(seed)
    np.random.seed(seed)

    tau   = torch.tensor(meta["tau_k"], dtype=torch.float32).to(device)
    Bbins = len(meta["tau_k"])
    d_num = meta["d_num"]           # 1 (amt)
    n_cat = list(meta["num_classes_cat"].values())   # [14] (category)
    K_cat = n_cat[0]                # 14
    L     = meta["L"]               # 24

    x_num_full  = train["x_num"].float()
    dt_bin_full = train["dt_bin"].long()
    x_cat_full  = train["x_cat"].long()
    y_full      = train["y"].float()
    mask_full   = train["mask"].bool()
    y_seq_full  = seq_label(y_full, mask_full).numpy()

    x_num  = x_num_full;   dt_bin = dt_bin_full
    x_cat  = x_cat_full;   y      = y_full
    mask   = mask_full;    y_seq  = y_seq_full
    N = x_num.shape[0]

    n_fraud   = int(y_seq.sum())
    real_prev = n_fraud / N
    y_ent     = torch.from_numpy(y_seq).float()

    # ── SequenceTeacher f_φ_seq ──────────────────────────────────────────────
    use_time = (teacher_type in ("amount_time", "full"))
    use_recv = (teacher_type == "full")   # Sparkov: use_recv controls category channel
    f_phi_seq = SequenceTeacher(
        d_num=d_num, Bbins=Bbins, K_cat=K_cat,
        h=64, n_layers=2,
        use_time=use_time, use_recv=use_recv,
    ).to(device)

    t_losses = pretrain_sequence_teacher(
        f_phi_seq, x_num, dt_bin, x_cat, y, mask,
        Bbins=Bbins, K_cat=K_cat,
        steps=teacher_steps, lr=1e-3, batch_size=256, device=device,
    )
    teacher_bce = float(np.mean(t_losses[-50:]))
    t_auprc = teacher_auprc_seq(
        f_phi_seq, x_num, dt_bin, x_cat, y_ent, mask,
        Bbins=Bbins, K_cat=K_cat, device=device,
    )
    print(f"  f_φ_seq[{teacher_type}] BCE={teacher_bce:.4f}  "
          f"teacher_auprc={t_auprc:.4f}  real_prev={real_prev:.4f}")

    # ── CoF-SeqGen v1c training ───────────────────────────────────────────────
    denoiser = SeqDenoiser(
        d_num=d_num, Bbins=Bbins, n_cat_classes=n_cat,
        d_model=D_MODEL, n_heads=4, n_layers=2, L_max=64,
    )
    model = CoFSeqGen(
        denoiser=denoiser, tau=tau.cpu(), W=W, temp=TEMP,
        coh_lambda=coh_lambda, n_cat_classes=n_cat,
        coherence_teacher=f_phi_seq, g_std=None,
    )
    model.train().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    if coh_lambda > 0 and teacher_type not in _grad_gate_done:
        check_grad_gate(model, x_num, dt_bin, x_cat, y, mask, device, N, teacher_type)
        _grad_gate_done.add(teacher_type)

    loss_hist = {"L_diff": [], "L_coh": [], "num_grad": [], "bin_grad": [], "cat_grad": []}
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
        dn = model.denoiser
        num_grad = dn.num_head.weight.grad
        bin_grad = dn.bin_head.weight.grad
        cat_grad = dn.cat_heads[0].weight.grad if dn.cat_heads else None
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step % 2000 == 0 or step == n_steps:
            loss_hist["L_diff"].append(info["L_diff"])
            loss_hist["L_coh"].append(info["L_coh"])
            loss_hist["num_grad"].append(num_grad.norm().item() if num_grad is not None else 0.0)
            loss_hist["bin_grad"].append(bin_grad.norm().item() if bin_grad is not None else 0.0)
            loss_hist["cat_grad"].append(cat_grad.norm().item() if cat_grad is not None else 0.0)

    L_diff_final   = float(np.mean(loss_hist["L_diff"][-3:]))
    L_coh_final    = float(np.mean(loss_hist["L_coh"][-3:]))
    num_grad_final = float(np.mean(loss_hist["num_grad"][-3:]))
    bin_grad_final = float(np.mean(loss_hist["bin_grad"][-3:]))
    cat_grad_final = float(np.mean(loss_hist["cat_grad"][-3:]))

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

    bool_mask_r = mask.numpy().astype(bool)
    bool_mask_s = mask_synth.numpy().astype(bool)
    real_vals   = x_num.numpy()[:, :, 0][bool_mask_r]
    synth_vals  = x_num_synth.numpy()[:, :, 0][bool_mask_s]
    amt_shift   = abs(real_vals.mean() - synth_vals.mean()) / (real_vals.std() + 1e-8)

    # ── Coherence-gap ────────────────────────────────────────────────────────
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
    beh_kw = dict(tau=tau_np, W=W, temp=TEMP, Bbins=Bbins, n_cat_classes=n_cat)

    if X_test_cache is not None:
        X_test, y_test = X_test_cache
    else:
        X_test, y_test = extract_features_with_behavioral(
            test["x_num"].float().numpy(), test["dt_bin"].long().numpy(),
            test["mask"].bool().numpy(), test["y"].float().numpy(),
            test["x_cat"].long().numpy(), **beh_kw,
        )

    if X_real_cache is not None:
        X_real, y_real_lab = X_real_cache
    else:
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
        device=device,
    )

    X_aug = np.concatenate([X_real, X_synth], axis=0)
    y_aug = np.concatenate([y_real_lab, y_synth_lab], axis=0)

    res_base  = run_tstr(X_real,  y_real_lab,  X_test, y_test, random_state=seed)
    res_synth = run_tstr(X_synth, y_synth_lab, X_test, y_test, random_state=seed)
    res_aug   = run_tstr(X_aug,   y_aug,        X_test, y_test, random_state=seed)

    delta_synth = round(res_synth["auprc"] - res_base["auprc"], 4)

    # collapse flag: amount_only with very negative Δsynth = L_coh over-constraining amount
    collapse = (teacher_type == "amount_only" and delta_synth < COLLAPSE_THRESH)
    if collapse:
        print(f"  [COLLAPSE] amount_only Δsynth={delta_synth:.4f} < {COLLAPSE_THRESH} "
              f"— amount-only over-constraint artifact at λ={coh_lambda}")

    return {
        "teacher_type":      teacher_type,
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
        "bin_grad_final":    round(bin_grad_final, 4),
        "cat_grad_final":    round(cat_grad_final, 4),
        "amt_shift":         round(float(amt_shift), 4),
        "coh_gap_vel":       round(float(coh_gap_vel),    6),
        "coh_gap_fanout":    round(float(coh_gap_fanout), 6),
        "coh_gap_amt":       round(float(coh_gap_amt),    6),
        "auprc_base":        round(res_base["auprc"],  4),
        "auprc_synth":       round(res_synth["auprc"], 4),
        "auprc_aug":         round(res_aug["auprc"],   4),
        "auroc_base":        round(res_base["auroc"],  4),
        "auroc_aug":         round(res_aug["auroc"],   4),
        "delta_synth":       delta_synth,
        "delta_aug":         round(res_aug["auprc"] - res_base["auprc"], 4),
        "collapse_flag":     int(collapse),
    }


# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(csv_path: Path, n_bootstrap: int = 2000) -> None:
    import csv as _csv
    from collections import defaultdict
    rows = []
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            rows.append(row)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["teacher_type"], float(r["coh_lambda"]))].append(r)

    agg_path = csv_path.parent / (csv_path.stem + "_agg.csv")
    fields = [
        "teacher_type", "coh_lambda", "n_seeds",
        "teacher_auprc_mean",
        "delta_synth_mean", "delta_synth_std",
        "delta_synth_ci_lo", "delta_synth_ci_hi",
        "coh_gap_fanout_mean", "coh_gap_amt_mean",
        "fraud_rate_synth_mean", "b_calib_mean",
        "auprc_base_mean", "auprc_synth_mean",
        "bin_grad_mean", "cat_grad_mean",
        "L_coh_final_mean", "n_collapse",
    ]
    rng = np.random.default_rng(42)

    with open(agg_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (ttype, lam), g in sorted(groups.items()):
            def _m(c): return float(np.mean([float(r[c]) for r in g]))
            def _s(c): return float(np.std( [float(r[c]) for r in g]))
            ds_vals = np.array([float(r["delta_synth"]) for r in g])
            boots   = [rng.choice(ds_vals, size=len(ds_vals), replace=True).mean()
                       for _ in range(n_bootstrap)]
            ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            n_collapse = sum(int(r["collapse_flag"]) for r in g)
            w.writerow({
                "teacher_type": ttype,
                "coh_lambda":   lam,
                "n_seeds":      len(g),
                "teacher_auprc_mean":    round(_m("teacher_auprc"),  4),
                "delta_synth_mean":      round(float(ds_vals.mean()), 4),
                "delta_synth_std":       round(float(ds_vals.std()),  4),
                "delta_synth_ci_lo":     round(ci_lo,  4),
                "delta_synth_ci_hi":     round(ci_hi,  4),
                "coh_gap_fanout_mean":   round(_m("coh_gap_fanout"), 6),
                "coh_gap_amt_mean":      round(_m("coh_gap_amt"),    6),
                "fraud_rate_synth_mean": round(_m("fraud_rate_synth"), 4),
                "b_calib_mean":          round(_m("b_calib"),        4),
                "auprc_base_mean":       round(_m("auprc_base"),     4),
                "auprc_synth_mean":      round(_m("auprc_synth"),    4),
                "bin_grad_mean":         round(_m("bin_grad_final"), 6),
                "cat_grad_mean":         round(_m("cat_grad_final"), 6),
                "L_coh_final_mean":      round(_m("L_coh_final"),    5),
                "n_collapse":            n_collapse,
            })

    print(f"\n[Sparkov SeqTeacher Aggregated] → {agg_path}")
    print(f'\n{"teacher":>12} {"λ":>5} {"n":>2}  {"Δsynth":>8}±std  {"95% CI":>18}  '
          f'{"coh_gap_f":>10}  {"teacher_ap":>10}  {"collapse":>8}')
    print("─" * 100)
    with open(agg_path) as f:
        for row in _csv.DictReader(f):
            ci = f'[{float(row["delta_synth_ci_lo"]):+.4f},{float(row["delta_synth_ci_hi"]):+.4f}]'
            print(
                f'{row["teacher_type"]:>12} '
                f'{float(row["coh_lambda"]):>5.1f} '
                f'{row["n_seeds"]:>2}  '
                f'{float(row["delta_synth_mean"]):>+7.4f}±{float(row["delta_synth_std"]):.4f}  '
                f'{ci:>18}  '
                f'{float(row["coh_gap_fanout_mean"]):>10.6f}  '
                f'{float(row["teacher_auprc_mean"]):>10.4f}  '
                f'{row["n_collapse"]:>8}'
            )

    # 3-arm attribution with bootstrap CI on Δwindow
    print("\n── 3-Arm Attribution (delta_SYNTH, Sparkov): timing · category · window ──")
    print("  Δtiming  = amount_time − amount_only   (expected ~small for Sparkov)")
    print("  Δcat     = full − amount_time           (expected ~dominant: +0.058 at teacher level)")
    print("  Δwindow  = full − amount_only           (= Δtiming + Δcat)")
    print("  [CI] bootstrap 95% on per-λ Δwindow")
    print()

    by_lam_type = {}
    with open(agg_path) as f:
        for row in _csv.DictReader(f):
            by_lam_type[(row["teacher_type"], float(row["coh_lambda"]))] = row

    # Load raw for paired bootstrap
    raw = defaultdict(list)
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            raw[(row["teacher_type"], float(row["coh_lambda"]))].append(float(row["delta_synth"]))

    lambdas = sorted(set(float(r["coh_lambda"]) for r in rows))
    print(f'  {"λ":>5}  {"full":>8}  {"amt_time":>10}  {"amt_only":>10}  '
          f'{"Δtiming":>8}  {"Δcat":>7}  {"Δwindow":>8}  {"Δwin CI":>18}  {"C2b?":>15}')
    print("  " + "─" * 100)

    w_diffs_all = []
    for lam in lambdas:
        fk = ("full",         lam)
        tk = ("amount_time",  lam)
        ak = ("amount_only",  lam)
        missing = [k[0] for k in [fk, tk, ak] if k not in by_lam_type]
        if missing:
            print(f'  {lam:>5.1f}  (missing: {missing})')
            continue
        fd = float(by_lam_type[fk]["delta_synth_mean"])
        td = float(by_lam_type[tk]["delta_synth_mean"])
        ad = float(by_lam_type[ak]["delta_synth_mean"])
        dt = td - ad
        dc = fd - td
        dw = fd - ad

        # bootstrap CI on Δwindow: resample seeds
        f_raw = np.array(raw[fk])
        a_raw = np.array(raw[ak])
        min_n = min(len(f_raw), len(a_raw))
        boots_w = []
        for _ in range(n_bootstrap):
            fi = rng.choice(len(f_raw), size=min_n, replace=True)
            ai = rng.choice(len(a_raw), size=min_n, replace=True)
            boots_w.append(f_raw[fi].mean() - a_raw[ai].mean())
        ci_lo_w = float(np.percentile(boots_w, 2.5))
        ci_hi_w = float(np.percentile(boots_w, 97.5))
        ci_str  = f'[{ci_lo_w:+.4f},{ci_hi_w:+.4f}]'

        # collapse check
        n_col_a = int(by_lam_type[ak].get("n_collapse", 0))
        col_tag = f" [col={n_col_a}]" if n_col_a > 0 else ""

        if lam > 0:
            w_diffs_all.append(dw)

        if lam == 0.0:
            c2b = "(noise floor)"
        elif ci_lo_w > 0:
            c2b = "✓ C2b (CI>0)"
        elif dw > 0.03:
            c2b = "~ C2b (CI straddles 0)"
        elif dw < -0.02:
            c2b = "✗ window<amtonly"
        else:
            c2b = "~ C2a (tied)"

        print(f'  {lam:>5.1f}  {fd:>+8.4f}  {td:>+10.4f}  {ad:>+10.4f}'
              f'{col_tag}  '
              f'{dt:>+8.4f}  {dc:>+7.4f}  {dw:>+8.4f}  {ci_str:>18}  {c2b:>15}')

    if w_diffs_all:
        mw = float(np.mean(w_diffs_all))
        print(f'\n  Mean Δwindow across λ>0: {mw:+.4f}')

        # Overall bootstrap: pool all λ>0 full and amount_only seeds
        f_all = np.concatenate([np.array(raw[("full", lam)]) for lam in lambdas if lam > 0])
        a_all = np.concatenate([np.array(raw[("amount_only", lam)]) for lam in lambdas if lam > 0])
        # Paired within λ: compute per-λ mean diff, then bootstrap across λ
        per_lam_diffs = [
            np.array(raw[("full", lam)]).mean() - np.array(raw[("amount_only", lam)]).mean()
            for lam in lambdas if lam > 0
        ]
        boots_overall = []
        for _ in range(n_bootstrap):
            sampled = rng.choice(per_lam_diffs, size=len(per_lam_diffs), replace=True)
            boots_overall.append(sampled.mean())
        oci_lo = float(np.percentile(boots_overall, 2.5))
        oci_hi = float(np.percentile(boots_overall, 97.5))
        print(f'  Overall CI [λ>0 pooled]: [{oci_lo:+.4f}, {oci_hi:+.4f}]')

        print()
        print("  ── JUDGMENT ──")
        if oci_lo > 0:
            print(f"  C2b CONFIRMED (Sparkov): Δwindow={mw:+.4f}, 95% CI [{oci_lo:+.4f},{oci_hi:+.4f}] > 0")
            print("  → C2b generality supported across AMLSim + Sparkov")
        elif mw > 0.03:
            print(f"  C2b LIKELY (Sparkov): Δwindow={mw:+.4f} but CI straddles 0 [{oci_lo:+.4f},{oci_hi:+.4f}]")
            print("  → directional support; CI ambiguous — generality tentative")
        else:
            print(f"  C2b NOT CONFIRMED (Sparkov): Δwindow={mw:+.4f}, CI [{oci_lo:+.4f},{oci_hi:+.4f}]")
            print("  → C2b may be AMLSim-specific; scope C2b claim accordingly")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sparkov_seqteacher.csv"

    completed = set()
    if args.resume and csv_path.exists():
        import csv as _csv
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                completed.add((
                    row["teacher_type"], float(row["coh_lambda"]),
                    float(row["W"]), int(row["seed"])
                ))
        print(f"[Resume] {len(completed)} done.")

    grid = list(product(
        sorted(args.teacher_types),
        sorted(args.lambdas),
        [args.W],
        sorted(args.seeds),
    ))
    n_total = len(grid)
    meta, train, test = load_sparkov(args.data_root)
    n_cat  = list(meta["num_classes_cat"].values())
    Bbins  = len(meta["tau_k"])
    tau_np = np.array(meta["tau_k"], dtype=np.float32)

    print(f"[Sparkov SeqTeacher] {n_total} runs  n_cat={n_cat}(category)  d_num={meta['d_num']}  L={meta['L']}")
    print(f"  teacher_types={args.teacher_types}  W={args.W}  seeds={len(args.seeds)}")
    print(f"  Expected dominant C2b channel: CATEGORY (diag: Δcat=+0.058 > Δtiming=+0.017)")

    # Pre-cache X_real and X_test (constant across all runs)
    beh_kw_ref = dict(tau=tau_np, W=args.W, temp=TEMP, Bbins=Bbins, n_cat_classes=n_cat)
    print("[Cache] Pre-computing X_test and X_real (constant across all runs)...")
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
    print(f"[Upper bound] Full-train AUPRC (W={args.W}) = {auprc_full:.4f}")
    print(f"[Cache] X_real and X_test cached — reused across all {n_total} runs.")

    fieldnames = [
        "teacher_type", "coh_lambda", "W", "seed",
        "n_train", "n_fraud_train", "real_prevalence",
        "teacher_bce", "teacher_auprc",
        "b_calib", "fraud_rate_before", "fraud_rate_synth",
        "L_diff_final", "L_coh_final",
        "num_grad_final", "bin_grad_final", "cat_grad_final",
        "amt_shift",
        "coh_gap_vel", "coh_gap_fanout", "coh_gap_amt",
        "auprc_base", "auprc_synth", "auprc_aug",
        "auroc_base", "auroc_aug",
        "delta_synth", "delta_aug", "auprc_full", "collapse_flag",
    ]
    fout   = open(csv_path, "a" if args.resume else "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if not args.resume:
        writer.writeheader()
    fout.flush()

    t0_sweep = time.time()
    for run_idx, (ttype, lam, W, seed) in enumerate(grid, 1):
        if (ttype, lam, W, seed) in completed:
            print(f"[{run_idx}/{n_total}] SKIP {ttype} λ={lam} seed={seed}")
            continue
        print(f"\n[{run_idx}/{n_total}] teacher={ttype} λ={lam} W={W} seed={seed} "
              f"— elapsed {(time.time()-t0_sweep)/60:.1f}min")
        t0 = time.time()
        result = run_single(
            meta, train, test,
            coh_lambda=lam, W=W, seed=seed, teacher_type=ttype,
            n_steps=args.n_steps, teacher_steps=args.teacher_steps,
            device=args.device,
            X_real_cache=(X_full_ref, y_full_ref),
            X_test_cache=(X_test_ref, y_test_ref),
        )
        result["auprc_full"] = round(auprc_full, 4)
        elapsed = time.time() - t0
        col_tag = " [COLLAPSE]" if result["collapse_flag"] else ""
        print(f"  coh_gap fanout={result['coh_gap_fanout']:.4f} "
              f"amt={result['coh_gap_amt']:.4f}  "
              f"Δsynth={result['delta_synth']:+.4f}  "
              f"Δaug={result['delta_aug']:+.4f}{col_tag}  [{elapsed:.0f}s]")
        writer.writerow(result)
        fout.flush()

    fout.close()
    print(f"\n[Sparkov SeqTeacher complete] {n_total} runs → {csv_path}")
    aggregate_results(csv_path)


if __name__ == "__main__":
    import csv
    main()
