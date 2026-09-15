"""
AMLSim sweep with SequenceTeacher coherence (v1c) — attribution experiment.

Purpose:
  Previously (v1b): CoherenceTeacher(4-dim g MLP) AUPRC=0.084 (≈ prevalence=0.036).
  Now (v1c):        SequenceTeacher(BiGRU, full window) AUPRC≈0.978 — g was the bottleneck.

Attribution test (C2b isolation):
  teacher_type="full"         : SequenceTeacher(amount+timing+receiver)
  teacher_type="amount_only"  : SequenceTeacher(amount only, timing/receiver disabled)

Judgment table (from AMLSim_seqteacher_실행지시.md):
  CoF+full > CoF+amount_only > row-baseline (significant) → C2b·C3 valid, headline confirmed.
  full ≈ amount_only (both above baseline)                → gain is C2a only (amount), C2b weak.
  Both ≤ baseline                                         → no support → archive / diagnosis paper.

TSTR classification: LR on behavioral features (extract_features_with_behavioral).
Coherence gap: g binwise P(fraud|g_bin) divergence (fanout=feat2, amt=feat3).

Grid: λ∈{0,0.1,0.5,1,2} × W=7 × teacher_type∈{full,amount_only} × seed∈{1,2,3} = 30 runs
Output: results/amlsim_seqteacher.csv

STEP 3 grad gate (first run only): after first L_coh.backward(), print grad norms for
  num_head, bin_head, cat_heads[0], y_head — all must be nonzero with full teacher.
  amount_only teacher: cat_head grad may be zero (expected — no receiver channel).
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
LAMBDA_GRID       = [0.0, 0.1, 0.5, 1.0, 2.0]
W_FIXED           = 7.0          # AMLSim: step units; W=7 preferred from v1b
SEED_GRID         = [1, 2, 3]
# 3-arm attribution:
#   amount_only : use_time=False, use_recv=False  → pure amount signal
#   amount_time : use_time=True,  use_recv=False  → amount + timing
#   full        : use_time=True,  use_recv=True   → full window
# Attribution: Δtiming = amount_time − amount_only
#              Δrecv   = full − amount_time
#              Δwindow = full − amount_only  (= Δtiming + Δrecv)
TEACHER_TYPES     = ["amount_only", "amount_time", "full"]

N_STEPS       = 20000
TEACHER_STEPS = 2000
T_DIFF        = 50
BATCH_SIZE    = 64
LR            = 3e-4
D_MODEL       = 128
TEMP          = 1.0


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
                   default=str(ROOT / "data" / "amlsim" / "sequences"))
    p.add_argument("--out_dir",       type=str, default=str(ROOT / "results"))
    p.add_argument("--resume",        action="store_true")
    return p.parse_args()


def load_amlsim(data_root):
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
    """Entity-level AUPRC for SequenceTeacher. Aggregates per-position logits via mean-pool."""
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
            logit = f_phi_seq(xn, bin_p, cat_p, m)            # (B, L)
            # Mean-pool over real positions → entity-level probability
            mf = m.float()
            ent_logit = (logit * mf).sum(1) / mf.sum(1).clamp(min=1)  # (B,)
            all_probs.append(torch.sigmoid(ent_logit).cpu().numpy())
    probs = np.concatenate(all_probs)
    y_np  = y_ent.numpy()
    if y_np.sum() < 2:
        return 0.0
    return float(average_precision_score(y_np, probs))


def check_grad_gate(model, x_num, dt_bin, x_cat, y, mask, device, N, label="full"):
    """One-off grad gate: verify all 4 heads receive nonzero grad through L_coh."""
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
    gn_num  = dn.num_head.weight.grad
    gn_bin  = dn.bin_head.weight.grad
    gn_cat  = dn.cat_heads[0].weight.grad if dn.cat_heads else None
    gn_y    = dn.y_head.weight.grad
    n  = lambda g: f"{g.norm().item():.4f}" if g is not None else "NONE"
    print(f"  [grad-gate teacher={label}] "
          f"num_head={n(gn_num)}  bin_head={n(gn_bin)}  "
          f"cat_head={n(gn_cat)}  y_head={n(gn_y)}")
    # Expected nonzero heads per arm:
    #   amount_only: num + y (no timing/recv in teacher → no L_coh grad for bin/cat)
    #   amount_time: num + bin + y (timing channel feeds into q)
    #   full:        num + bin + cat + y
    assert gn_num is not None and gn_num.norm() > 0, "FAIL: num_head grad=0"
    assert gn_y   is not None and gn_y.norm()   > 0, "FAIL: y_head grad=0"
    if label in ("amount_time", "full"):
        assert gn_bin is not None and gn_bin.norm() > 0, \
            f"FAIL: bin_head grad=0 for {label} (timing C2b dead)"
    if label == "full":
        if gn_cat is not None:
            assert gn_cat.norm() > 0, "FAIL: cat_head grad=0 (receiver C2b dead)"
    print(f"  [grad-gate teacher={label}] PASS")
    model.zero_grad()


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

_grad_gate_done = set()  # track (teacher_type) to run gate once per type

def run_single(meta, train, test, coh_lambda, W, seed, teacher_type,
               n_steps, teacher_steps, device, run_label="",
               X_real_cache=None, X_test_cache=None):
    global _grad_gate_done

    torch.manual_seed(seed)
    np.random.seed(seed)

    tau   = torch.tensor(meta["tau_k"], dtype=torch.float32).to(device)
    Bbins = len(meta["tau_k"])
    d_num = meta["d_num"]           # 1 (TX_AMOUNT_LOG)
    n_cat = list(meta["num_classes_cat"].values())   # [256]
    K_cat = n_cat[0]                # 256
    L     = meta["L"]               # 32

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
    # 3-arm: amount_only (no time/recv), amount_time (time only), full (time+recv)
    use_time = (teacher_type in ("amount_time", "full"))
    use_recv = (teacher_type == "full")
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
    if t_auprc < 0.5:
        print(f"  [WARN] teacher AUPRC={t_auprc:.4f} < 0.5 — gate failed")
    elif t_auprc < 0.8:
        print(f"  [WARN] teacher AUPRC={t_auprc:.4f} < 0.8 — weaker than expected (diag: 0.978)")

    # ── CoF-SeqGen v1c training ───────────────────────────────────────────────
    denoiser = SeqDenoiser(
        d_num=d_num, Bbins=Bbins, n_cat_classes=n_cat,
        d_model=D_MODEL, n_heads=4, n_layers=2, L_max=64,
    )
    model = CoFSeqGen(
        denoiser=denoiser, tau=tau.cpu(), W=W, temp=TEMP,
        coh_lambda=coh_lambda, n_cat_classes=n_cat,
        coherence_teacher=f_phi_seq, g_std=None,   # SequenceTeacher: g_std not used
    )
    model.train().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    # STEP 3 grad gate: verify 4 heads get grad through L_coh (once per teacher_type, λ>0)
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

    if abs(fraud_rate_synth - real_prev) > real_prev:
        print(f"  [WARN] calibration: fraud_rate={fraud_rate_synth:.4f} vs target={real_prev:.4f}")

    # Amount quality
    bool_mask_r = mask.numpy().astype(bool)
    bool_mask_s = mask_synth.numpy().astype(bool)
    real_vals   = x_num.numpy()[:, :, 0][bool_mask_r]
    synth_vals  = x_num_synth.numpy()[:, :, 0][bool_mask_s]
    amt_shift   = abs(real_vals.mean() - synth_vals.mean()) / (real_vals.std() + 1e-8)

    # ── Coherence-gap measurement (via compute_g_from_real on synth output) ──
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

    # X_real and X_test are constant across all runs (W fixed, data unchanged).
    # Use cache from main() to avoid ~5min CPU recomputation per run.
    if X_test_cache is not None:
        X_test, y_test = X_test_cache
    else:
        x_num_te  = test["x_num"].float().numpy()
        dt_bin_te = test["dt_bin"].long().numpy()
        x_cat_te  = test["x_cat"].long().numpy()
        mask_te   = test["mask"].bool().numpy()
        y_te      = test["y"].float().numpy()
        X_test, y_test = extract_features_with_behavioral(
            x_num_te, dt_bin_te, mask_te, y_te, x_cat_te, **beh_kw,
        )

    if X_real_cache is not None:
        X_real, y_real_lab = X_real_cache
    else:
        X_real, y_real_lab = extract_features_with_behavioral(
            x_num.numpy(), dt_bin.numpy(), mask.numpy(), y.numpy(), x_cat.numpy(), **beh_kw,
        )

    # X_synth changes every run — run on GPU for speedup
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
        "delta_synth":       round(res_synth["auprc"] - res_base["auprc"], 4),
        "delta_aug":         round(res_aug["auprc"]   - res_base["auprc"], 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation + attribution summary
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(csv_path: Path) -> None:
    import csv as _csv
    from collections import defaultdict
    rows = []
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            rows.append(row)

    # Group by (teacher_type, lambda)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["teacher_type"], float(r["coh_lambda"]))].append(r)

    agg_path = csv_path.parent / (csv_path.stem + "_agg.csv")
    fields = [
        "teacher_type", "coh_lambda", "n_seeds",
        "teacher_auprc_mean",
        "coh_gap_fanout_mean", "coh_gap_fanout_std",
        "coh_gap_amt_mean",    "coh_gap_amt_std",
        "delta_synth_mean",    "delta_synth_std",
        "delta_aug_mean",      "delta_aug_std",
        "fraud_rate_synth_mean", "b_calib_mean",
        "auprc_base_mean", "auprc_synth_mean",
        "bin_grad_mean", "cat_grad_mean",
        "L_coh_final_mean",
    ]
    with open(agg_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (ttype, lam), g in sorted(groups.items()):
            def _m(c): return float(np.mean([float(r[c]) for r in g]))
            def _s(c): return float(np.std( [float(r[c]) for r in g]))
            w.writerow({
                "teacher_type": ttype,
                "coh_lambda":   lam,
                "n_seeds":      len(g),
                "teacher_auprc_mean":     round(_m("teacher_auprc"),  4),
                "coh_gap_fanout_mean":    round(_m("coh_gap_fanout"), 6),
                "coh_gap_fanout_std":     round(_s("coh_gap_fanout"), 6),
                "coh_gap_amt_mean":       round(_m("coh_gap_amt"),    6),
                "coh_gap_amt_std":        round(_s("coh_gap_amt"),    6),
                "delta_synth_mean":       round(_m("delta_synth"),    4),
                "delta_synth_std":        round(_s("delta_synth"),    4),
                "delta_aug_mean":         round(_m("delta_aug"),      4),
                "delta_aug_std":          round(_s("delta_aug"),      4),
                "fraud_rate_synth_mean":  round(_m("fraud_rate_synth"), 4),
                "b_calib_mean":           round(_m("b_calib"),        4),
                "auprc_base_mean":        round(_m("auprc_base"),     4),
                "auprc_synth_mean":       round(_m("auprc_synth"),    4),
                "bin_grad_mean":          round(_m("bin_grad_final"), 6),
                "cat_grad_mean":          round(_m("cat_grad_final"), 6),
                "L_coh_final_mean":       round(_m("L_coh_final"),    5),
            })

    print(f"\n[SeqTeacher Aggregated] {agg_path}")
    print(f'\n{"teacher":>12} {"λ":>5} {"n":>2}  {"coh_gap_fanout":>14}  '
          f'{"coh_gap_amt":>11}  {"ΔAUPRC_synth":>13}  {"ΔAUPRC_aug":>11}  {"teacher_ap":>10}')
    print("─" * 100)
    with open(agg_path) as f:
        for row in _csv.DictReader(f):
            print(
                f'{row["teacher_type"]:>12} '
                f'{float(row["coh_lambda"]):>5.1f} '
                f'{row["n_seeds"]:>2}  '
                f'{float(row["coh_gap_fanout_mean"]):>14.6f}  '
                f'{float(row["coh_gap_amt_mean"]):>11.6f}  '
                f'{float(row["delta_synth_mean"]):>+7.4f}±{float(row["delta_synth_std"]):.4f}  '
                f'{float(row["delta_aug_mean"]):>+5.4f}±{float(row["delta_aug_std"]):.4f}  '
                f'{float(row["teacher_auprc_mean"]):>10.4f}'
            )

    # Attribution: 3-arm (delta_SYNTH only — delta_aug saturated at AMLSim base≈0.9995)
    # Δtiming  = amount_time − amount_only  (timing channel contribution)
    # Δrecv    = full − amount_time         (receiver channel contribution)
    # Δwindow  = full − amount_only         (total window contribution)
    print("\n── 3-Arm Attribution (delta_SYNTH): timing · receiver · window ──")
    print("  (delta_aug excluded — AMLSim base≈0.9995 saturated; judge by delta_SYNTH)")
    by_lam_type = {}
    with open(agg_path) as f:
        for row in _csv.DictReader(f):
            by_lam_type[(row["teacher_type"], float(row["coh_lambda"]))] = row

    lambdas = sorted(set(float(r["coh_lambda"]) for r in rows))
    print(f'  {"λ":>5}  {"full":>8}  {"amt_time":>10}  {"amt_only":>10}  '
          f'{"Δtiming":>9}  {"Δrecv":>8}  {"Δwindow":>9}  {"C2b?":>10}')
    print("  " + "─" * 85)
    window_diffs = []
    timing_diffs = []
    recv_diffs   = []
    for lam in lambdas:
        fk = ("full",         lam)
        tk = ("amount_time",  lam)
        ak = ("amount_only",  lam)
        f_row = by_lam_type.get(fk)
        t_row = by_lam_type.get(tk)
        a_row = by_lam_type.get(ak)
        if not (f_row and t_row and a_row):
            available = [k[0] for k in [fk,tk,ak] if by_lam_type.get(k)]
            print(f'  {lam:>5.1f}  (missing: {[k[0] for k in [fk,tk,ak] if not by_lam_type.get(k)]})')
            continue
        fd = float(f_row["delta_synth_mean"])
        td = float(t_row["delta_synth_mean"])
        ad = float(a_row["delta_synth_mean"])
        d_timing = td - ad
        d_recv   = fd - td
        d_window = fd - ad
        if lam > 0:
            window_diffs.append(d_window)
            timing_diffs.append(d_timing)
            recv_diffs.append(d_recv)
        if lam == 0.0:
            c2b = "(baseline)"
        elif d_window > 0.02:
            c2b = "✓ C2b"
        elif d_window < -0.02:
            c2b = "✗ window<amtonly"
        else:
            c2b = "~ C2a only"
        print(f'  {lam:>5.1f}  {fd:>+8.4f}  {td:>+10.4f}  {ad:>+10.4f}  '
              f'{d_timing:>+9.4f}  {d_recv:>+8.4f}  {d_window:>+9.4f}  {c2b:>10}')

    if window_diffs:
        mw = float(np.mean(window_diffs))
        mt = float(np.mean(timing_diffs))
        mr = float(np.mean(recv_diffs))
        print(f'\n  Mean across λ>0: Δtiming={mt:+.4f}  Δrecv={mr:+.4f}  Δwindow={mw:+.4f}')
        print()
        if mw > 0.02:
            if mt > 0.01 and mr > 0.01:
                print("  JUDGMENT: C2b SUPPORTED — BOTH timing and receiver add above amount")
            elif mt > 0.01:
                print("  JUDGMENT: C2b(timing) SUPPORTED — timing drives window gain; receiver marginal")
            elif mr > 0.01:
                print("  JUDGMENT: C2b(receiver) SUPPORTED — receiver drives window gain; timing marginal")
            else:
                print("  JUDGMENT: C2b (unknown channel mix) — check individual Δ")
        elif abs(mw) <= 0.02:
            print("  JUDGMENT: C2a ONLY — window(timing+recv) ≤ 0.02 above amount-only")
            print("    → headline narrows to 'amount coherence only'")
        else:
            print("  JUDGMENT: UNEXPECTED — window HURTS vs amount-only; investigate")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "amlsim_seqteacher.csv"

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

    # Grid: teacher_type × lambda × seed (W fixed)
    grid = list(product(
        sorted(args.teacher_types),
        sorted(args.lambdas),
        [args.W],
        sorted(args.seeds),
    ))
    n_total = len(grid)
    meta, train, test = load_amlsim(args.data_root)
    n_cat  = list(meta["num_classes_cat"].values())
    Bbins  = len(meta["tau_k"])
    tau_np = np.array(meta["tau_k"], dtype=np.float32)

    print(f"[AMLSim SeqTeacher] {n_total} runs  n_cat={n_cat}  d_num={meta['d_num']}  L={meta['L']}")
    print(f"  teacher_types={args.teacher_types}  W={args.W}")

    # Full-train upper bound — also serves as X_real_cache and X_test_cache for all runs.
    # W is fixed across all runs so these features are identical every time.
    # Pre-computing once here eliminates 2 of 3 extract_behavioral_features calls per run.
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
    print("[Cache] X_real and X_test cached — will be reused across all 45 runs.")

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
        "delta_synth", "delta_aug", "auprc_full",
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
        print(f"  coh_gap fanout={result['coh_gap_fanout']:.4f} "
              f"amt={result['coh_gap_amt']:.4f}  "
              f"ΔAUPRC_synth={result['delta_synth']:+.4f}  "
              f"ΔAUPRC_aug={result['delta_aug']:+.4f}  [{elapsed:.0f}s]")
        writer.writerow(result)
        fout.flush()

    fout.close()
    print(f"\n[AMLSim SeqTeacher complete] {n_total} runs → {csv_path}")
    aggregate_results(csv_path)


if __name__ == "__main__":
    import csv
    main()
