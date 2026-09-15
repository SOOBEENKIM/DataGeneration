"""
Coupling inversion localisation for CoF-SeqGen.

Diagnostic:
  1. Train CoF at λ=0 (pure diffusion, no coherence loss).
  2. Generate N synthetic sequences.
  3. Compute g_synth per entity (vel, gap, fanout, amt).
  4. Run the pretrained SequenceTeacher on synthetic sequences
     → teacher_score_on_synth (N,) = P(fraud | synthetic behaviour)
  5. corr(teacher_score_on_synth, y_gen_seq):
       < 0  →  라벨 head 반전:  y_head 부호/조건이 잘못됨 (수정 가능)
       > 0  →  denoiser 특징 반전:  denoiser가 fraud 행동 자체를 반대로 생성 (더 깊은 결함)
  6. Per-channel P(fraud|g_bin) real vs synth → coupling direction check.

Usage:
  python scripts/diag_cof_coupling.py --lambdas 0.0 --seeds 1
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.coherence_teacher import SequenceTeacher, pretrain_sequence_teacher
from models.teacher import compute_g_from_real
from models.seq_denoiser import SeqDenoiser
from models.cof_seqgen import CoFSeqGen
from models.sampler import (
    ddim_sample,
    sample_empirical_dt_bin, sample_empirical_x_cat,
)


# ─── hyperparams (match sweep) ───────────────────────────────────────────────
N_STEPS       = 20000
TEACHER_STEPS = 2000
T_DIFF        = 50
BATCH_SIZE    = 64
LR            = 3e-4
D_MODEL       = 128
TEMP          = 1.0
W_FIXED       = 7.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambdas",   type=float, nargs="+", default=[0.0])
    p.add_argument("--seeds",     type=int,   nargs="+", default=[1])
    p.add_argument("--n_steps",   type=int,   default=N_STEPS)
    p.add_argument("--teacher_steps", type=int, default=TEACHER_STEPS)
    p.add_argument("--W",         type=float, default=W_FIXED)
    p.add_argument("--device",    type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data_root", type=str,
                   default=str(ROOT / "data" / "amlsim" / "sequences"))
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


def teacher_probs_on_synth(f_phi_seq, x_num, dt_bin, x_cat, mask,
                            Bbins, K_cat, device, batch_size=64):
    """Entity-level teacher probability on synthetic sequences (N,)."""
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
            logit = f_phi_seq(xn, bin_p, cat_p, m)   # (B, L)
            mf = m.float()
            ent_logit = (logit * mf).sum(1) / mf.sum(1).clamp(min=1)
            all_probs.append(torch.sigmoid(ent_logit).cpu().numpy())
    return np.concatenate(all_probs)


def p_fraud_by_bin(g_vec, y_vec, lo_thr, hi_thr):
    """P(fraud | g ≤ lo_thr) and P(fraud | g > lo_thr) using quantile thresholds."""
    lo_mask = g_vec <= lo_thr
    hi_mask = g_vec >  lo_thr
    p_lo = float(y_vec[lo_mask].mean()) if lo_mask.sum() > 0 else float("nan")
    p_hi = float(y_vec[hi_mask].mean()) if hi_mask.sum() > 0 else float("nan")
    return p_lo, p_hi


def run_diag(lam, seed, args, meta, train):
    print(f"\n{'='*60}")
    print(f" λ={lam}  seed={seed}")
    print(f"{'='*60}")

    device = args.device
    torch.manual_seed(seed)
    np.random.seed(seed)

    tau   = torch.tensor(meta["tau_k"], dtype=torch.float32).to(device)
    Bbins = len(meta["tau_k"])
    d_num = meta["d_num"]
    n_cat = list(meta["num_classes_cat"].values())
    K_cat = n_cat[0]
    L     = meta["L"]
    W     = args.W

    x_num  = train["x_num"].float()
    dt_bin = train["dt_bin"].long()
    x_cat  = train["x_cat"].long()
    y      = train["y"].float()
    mask   = train["mask"].bool()
    N      = x_num.shape[0]

    y_seq     = seq_label(y, mask).numpy()
    real_prev = float(y_seq.mean())
    y_ent     = torch.from_numpy(y_seq).float()

    # ── 1. Pretrain SequenceTeacher on real data ─────────────────────────────
    print(f"\n[1] Pretraining SequenceTeacher (steps={args.teacher_steps}) ...")
    f_phi_seq = SequenceTeacher(
        d_num=d_num, Bbins=Bbins, K_cat=K_cat,
        h=64, n_layers=2, use_time=True, use_recv=True,
    ).to(device)
    t_losses = pretrain_sequence_teacher(
        f_phi_seq, x_num, dt_bin, x_cat, y, mask,
        Bbins=Bbins, K_cat=K_cat,
        steps=args.teacher_steps, lr=1e-3, batch_size=256, device=device,
    )
    teacher_bce = float(np.mean(t_losses[-50:]))

    # Quick AUPRC on real data to verify teacher quality
    t_probs_real = teacher_probs_on_synth(
        f_phi_seq, x_num, dt_bin, x_cat, mask, Bbins, K_cat, device,
    )
    t_auprc_real = float(average_precision_score(y_seq, t_probs_real)) \
        if y_seq.sum() >= 2 else 0.0
    print(f"    Teacher BCE={teacher_bce:.4f}  AUPRC_real={t_auprc_real:.4f}  "
          f"prev={real_prev:.4f}")
    if t_auprc_real < 0.8:
        print(f"    [WARN] Teacher AUPRC={t_auprc_real:.4f} < 0.8 — weak teacher signal")

    # ── 2. Train CoF at λ ─────────────────────────────────────────────────────
    print(f"\n[2] Training CoF λ={lam} for {args.n_steps} steps ...")
    denoiser = SeqDenoiser(
        d_num=d_num, Bbins=Bbins, n_cat_classes=n_cat,
        d_model=D_MODEL, n_heads=4, n_layers=2, L_max=64,
    )
    model = CoFSeqGen(
        denoiser=denoiser, tau=tau.cpu(), W=W, temp=TEMP,
        coh_lambda=lam, n_cat_classes=n_cat,
        coherence_teacher=f_phi_seq,
    )
    model.train().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    for step in range(1, args.n_steps + 1):
        idx    = torch.randint(0, N, (BATCH_SIZE,))
        t_frac = torch.rand(1).item() * 0.9 + 0.1
        optim.zero_grad()
        loss, _ = model.compute_loss(
            x_num[idx].to(device), dt_bin[idx].to(device),
            x_cat[idx].to(device), y[idx].to(device),
            mask[idx].to(device), t_frac,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step % 5000 == 0:
            print(f"    step {step}/{args.n_steps}  loss={loss.item():.4f}")

    # ── 3. Generate synthetic sequences (class-conditional) ──────────────────
    print(f"\n[3] Generating {N} synthetic sequences (class-conditional CFG) ...")
    model.eval()
    x_num_chunks, bin_chunks, cat_chunks, ylogit_chunks, mask_chunks = [], [], [], [], []
    ycond_chunks = []
    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        bs  = end - start
        dt_samp   = sample_empirical_dt_bin(dt_bin, bs, L, Bbins, device=device)
        xcat_samp = sample_empirical_x_cat(x_cat, bs, device=device)
        mask_samp = mask[torch.randint(0, N, (bs,))]
        # Pre-sample entity fraud labels at real prevalence → X|y generation
        y_cond_batch = torch.bernoulli(torch.full((bs,), real_prev)).long().to(device)
        x_gen, bin_pred, cat_preds, _, y_logit = ddim_sample(
            model, dt_samp, xcat_samp,
            d_num=d_num, T_steps=T_DIFF, device=device, return_discrete=True,
            y_cond=y_cond_batch,
            valid_mask=mask_samp.to(device),
        )
        x_num_chunks.append(x_gen.cpu())
        bin_chunks.append(bin_pred.cpu())
        cat_chunks.append(cat_preds[0].cpu() if cat_preds
                          else torch.zeros(bs, L, dtype=torch.long))
        ylogit_chunks.append(y_logit)
        mask_chunks.append(mask_samp)
        ycond_chunks.append(y_cond_batch.cpu())

    x_num_synth  = torch.cat(x_num_chunks, dim=0)
    dt_bin_synth = torch.cat(bin_chunks, dim=0)
    x_cat_synth  = torch.cat(cat_chunks, dim=0).unsqueeze(-1)
    y_logit_all  = torch.cat(ylogit_chunks, dim=0)
    mask_synth   = torch.cat(mask_chunks, dim=0)
    # y_gen_seq: entity fraud labels from pre-sampled y_cond (not from y_logit)
    # With CFG, the label is the input to generation, not an output to calibrate.
    y_gen_seq   = torch.cat(ycond_chunks, dim=0).float()
    fraud_synth = float(y_gen_seq.mean())
    print(f"    fraud_synth={fraud_synth:.4f}  target={real_prev:.4f}  (Bernoulli pre-sample, no calibration needed)")

    # ── 4. Teacher score on synthetic sequences ───────────────────────────────
    print(f"\n[4] Running teacher on synthetic sequences ...")
    t_score_synth = teacher_probs_on_synth(
        f_phi_seq, x_num_synth, dt_bin_synth, x_cat_synth, mask_synth,
        Bbins, K_cat, device,
    )
    y_gen_np = y_gen_seq.numpy().astype(float)

    pearson_r, pearson_p = pearsonr(t_score_synth, y_gen_np)
    spearman_r, spearman_p = spearmanr(t_score_synth, y_gen_np)
    print(f"    Pearson  r={pearson_r:+.4f}  p={pearson_p:.4g}")
    print(f"    Spearman r={spearman_r:+.4f}  p={spearman_p:.4g}")

    # ── 5. Per-channel P(fraud|g_bin) coupling direction ─────────────────────
    print(f"\n[5] Per-channel coupling direction ...")

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

    g_raw_real  = batched_g(x_num, dt_bin, x_cat, mask)
    g_raw_synth = batched_g(x_num_synth, dt_bin_synth, x_cat_synth, mask_synth)

    mr = mask.unsqueeze(-1).float()
    ms = mask_synth.unsqueeze(-1).float()
    g_real_ent  = ((g_raw_real  * mr).sum(1) / mr.sum(1).clamp(min=1)).numpy()
    g_synth_ent = ((g_raw_synth * ms).sum(1) / ms.sum(1).clamp(min=1)).numpy()

    ch_names = ["vel(0)", "gap(1)", "fanout(2)", "amt(3)"]
    ch_idx   = [0, 1, 2, 3]

    print(f"    {'ch':>8}  {'real P(lo)':>10}  {'real P(hi)':>10}  "
          f"{'real dir':>9}  {'synth P(lo)':>11}  {'synth P(hi)':>11}  "
          f"{'synth dir':>10}  {'match?':>6}")
    print(f"    {'-'*90}")

    coupling_correct = 0
    coupling_total   = 0
    for ci, name in zip(ch_idx, ch_names):
        g_r = g_real_ent[:, ci]
        g_s = g_synth_ent[:, ci]
        # Use real-data quantile thresholds for both
        q50_r = float(np.median(g_r))
        p_r_lo, p_r_hi = p_fraud_by_bin(g_r, y_seq, q50_r, None)
        p_s_lo, p_s_hi = p_fraud_by_bin(g_s, y_gen_np, q50_r, None)

        real_dir  = "fraud=hi" if p_r_hi > p_r_lo else "fraud=lo"
        synth_dir = "fraud=hi" if (not np.isnan(p_s_hi) and not np.isnan(p_s_lo)
                                   and p_s_hi > p_s_lo) else "fraud=lo"
        match = "✓" if real_dir == synth_dir else "✗ INV"
        coupling_total += 1
        if real_dir == synth_dir:
            coupling_correct += 1

        p_r_lo_s  = f"{p_r_lo:.4f}"  if not np.isnan(p_r_lo)  else "  NaN"
        p_r_hi_s  = f"{p_r_hi:.4f}"  if not np.isnan(p_r_hi)  else "  NaN"
        p_s_lo_s  = f"{p_s_lo:.4f}"  if not np.isnan(p_s_lo)  else "  NaN"
        p_s_hi_s  = f"{p_s_hi:.4f}"  if not np.isnan(p_s_hi)  else "  NaN"
        print(f"    {name:>8}  {p_r_lo_s:>10}  {p_r_hi_s:>10}  "
              f"{real_dir:>9}  {p_s_lo_s:>11}  {p_s_hi_s:>11}  "
              f"{synth_dir:>10}  {match:>6}")

    # ── 6. 종합 판정 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" 종합 판정  (λ={lam}, seed={seed})")
    print(f"{'='*60}")
    print(f"  Teacher AUPRC on real   : {t_auprc_real:.4f}")
    print(f"  Teacher-vs-CoFLabel     : Pearson r = {pearson_r:+.4f}  (p={pearson_p:.3g})")
    print(f"                          : Spearman r= {spearman_r:+.4f}  (p={spearman_p:.3g})")
    print(f"  채널별 방향 일치        : {coupling_correct}/{coupling_total}")
    print()

    # Decision rule
    # Phase 1 (pre-CFG): r<0 = label head inverted; r>0 = denoiser inverted
    # Phase 2 (post-CFG): coupling_correct is primary; r>0 is expected (labels are inputs now)
    majority_correct = coupling_correct >= (coupling_total + 1) // 2
    all_correct = coupling_correct == coupling_total

    if all_correct:
        verdict = "CFG 수정 성공 — 반전 해소 (4/4 방향 일치)"
        detail  = (f"class-conditional generation 적용 후 모든 채널 coupling 방향이 "
                   f"real data와 일치.\n"
                   f"  vel·fanout 반전 완전 해소: fraud 엔티티가 올바른 행동 모드(low-fanout, "
                   f"low-vel) 생성.\n"
                   f"  → 게이트 1 통과. 다음: controlled κ-데이터에서 CoF vs CTGAN/TVAE.")
    elif majority_correct:
        verdict = f"CFG 부분 수정 ({coupling_correct}/{coupling_total} 방향 일치)"
        detail  = (f"다수 채널이 올바른 방향. 나머지 채널 확인 필요.\n"
                   f"  → 추가 학습 스텝 또는 guidance_scale > 1.0 시도.")
    elif abs(pearson_r) < 0.02 and abs(spearman_r) < 0.02:
        verdict = "판단 불가 (|r| < 0.02, 방향 불일치)"
        detail  = "Teacher와 CoF 라벨 간 상관 없음 + 방향 불일치. 아키텍처 점검 필요."
    elif pearson_r < 0:
        verdict = "라벨 head 반전 — 수정 가능 (fixable)"
        detail  = (f"Teacher(real 기반)는 synthetic 시퀀스에서 '비-fraud 행동' 패턴에 "
                   f"높은 점수를 주지만, CoF 라벨은 이를 fraud로.\n"
                   f"  → y_head 부호/조건 수정으로 해결 가능.")
    else:
        verdict = "Denoiser 특징 반전 — CFG 미적용 또는 더 깊은 결함"
        detail  = (f"Teacher와 CoF 라벨이 동의하지만 실제 방향과 반대.\n"
                   f"  → class-conditional 생성(CFG) 적용 확인, "
                   f"또는 guidance_scale > 1.0 시도.")

    print(f"  ▶ {verdict}")
    print(f"    {detail}")
    print()
    return {
        "lam": lam, "seed": seed,
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
        "t_auprc_real": t_auprc_real,
        "coupling_correct": coupling_correct, "coupling_total": coupling_total,
        "verdict": verdict,
    }


def main():
    args = parse_args()
    print(f"Device: {args.device}")
    print(f"Data: {args.data_root}")
    meta, train, _ = load_amlsim(args.data_root)
    print(f"Loaded AMLSim: N={train['x_num'].shape[0]}  L={meta['L']}  "
          f"Bbins={len(meta['tau_k'])}  d_num={meta['d_num']}")

    results = []
    for lam in args.lambdas:
        for seed in args.seeds:
            r = run_diag(lam, seed, args, meta, train)
            results.append(r)

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(" 전체 요약")
        print(f"{'='*60}")
        for r in results:
            print(f"  λ={r['lam']} seed={r['seed']:2d}  "
                  f"Pearson r={r['pearson_r']:+.4f}  dir={r['coupling_correct']}/{r['coupling_total']}  "
                  f"→ {r['verdict']}")


if __name__ == "__main__":
    main()
