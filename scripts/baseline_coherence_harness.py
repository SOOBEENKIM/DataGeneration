"""
Baseline coherence harness — C3 verification.

Tests: coherence_gap(CoF) < coherence_gap(row-level baselines)
CoF-SeqGen is sequence-aware; CTGAN/TVAE/row-shuffle treat each row independently
and therefore cannot preserve the within-entity behavior→fraud coupling.

Generators:
  cof          — native: reuse existing amlsim_seqteacher.csv (no retraining)
  row_shuffle  — flatten real rows, random-chunk into pseudo-entities (fast, marginal-preserving)
  ctgan        — CTGAN trained on flat rows (no entity structure)
  tvae         — TVAE trained on flat rows (no entity structure)

Controls:
  C0 (real_split) — real train 80/20 entity split → noise floor
  C1 (label_shuf) — real sequences, entity fraud-labels randomly shuffled → upper bound

Assembly for row-level generators:
  P2 (random): random permutation of rows, then sequential chunking into L-length entities
  P1 (seq)   : sequential ordering (no entity column → equivalent to P2)

Bootstrap CI: 2000 resamples over entities (entity-level variance)

Output: results/baseline_coherence.csv
"""

import sys
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.teacher import compute_g_from_real
from eval.tstr import coherence_gap

# ── Constants ────────────────────────────────────────────────────────────────
TEMP      = 1.0
W_DEFAULT = 7.0
N_BOOT    = 2000
BATCH_SZ  = 512
FEAT_NAMES = ["vel", "gap", "fanout", "amt"]
FEAT_IDX   = [0, 1, 2, 3]   # all 4 g-channels

# CFG-CoF training constants (match sweep_amlsim_seqteacher.py)
COF_N_STEPS       = 20000
COF_TEACHER_STEPS = 2000
COF_T_DIFF        = 50
COF_BATCH_SIZE    = 64
COF_LR            = 3e-4
COF_D_MODEL       = 128


# ── Data loading ─────────────────────────────────────────────────────────────

def load_amlsim(split: str = "train"):
    path = ROOT / "data" / "amlsim" / "sequences" / f"{split}.npz"
    meta_path = ROOT / "data" / "amlsim" / "sequences" / "meta.json"
    d = np.load(path)
    with open(meta_path) as f:
        meta = json.load(f)
    return d, meta


def make_tau_tensor(meta, device):
    tau = torch.tensor(meta["tau_k"], dtype=torch.float32).to(device)
    return tau


# ── g computation ─────────────────────────────────────────────────────────────

def compute_g_seq_np(
    x_num: np.ndarray,   # (N, L, 1)
    dt_bin: np.ndarray,  # (N, L) int
    x_cat: np.ndarray,   # (N, L, 1) int
    mask: np.ndarray,    # (N, L) bool
    tau: torch.Tensor,   # (Bbins,) on device
    W: float,
    Bbins: int,
    n_cat: list,
    device: str,
) -> np.ndarray:
    """Return g_seq (N, 4): masked mean of soft_g over valid positions."""
    N = x_num.shape[0]
    chunks = []
    with torch.no_grad():
        for s in range(0, N, BATCH_SZ):
            e = min(s + BATCH_SZ, N)
            xn = torch.from_numpy(x_num[s:e]).float().to(device)
            db = torch.from_numpy(dt_bin[s:e]).long().to(device)
            xc = torch.from_numpy(x_cat[s:e]).long().to(device)
            g_b = compute_g_from_real(
                xn, db, xc, tau, W, TEMP, Bbins, n_cat,
                valid_mask=torch.from_numpy(mask[s:e]).bool().to(device),
            )
            chunks.append(g_b.cpu())
    g_all = torch.cat(chunks, dim=0)   # (N, L, 4)
    m = torch.from_numpy(mask.astype(np.float32)).unsqueeze(-1)   # (N, L, 1)
    g_seq = (g_all * m).sum(1) / m.sum(1).clamp(min=1)            # (N, 4)
    return g_seq.numpy()


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_coh_gap(
    g_synth: np.ndarray,
    y_synth: np.ndarray,
    g_real: np.ndarray,
    y_real: np.ndarray,
    feat_idx: int,
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> tuple:
    """Bootstrap CI (entity-level resampling) on coherence_gap."""
    rng = np.random.default_rng(seed)
    N = len(g_synth)
    gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        gaps.append(coherence_gap(g_synth[idx], y_synth[idx], g_real, y_real, feat_idx=feat_idx))
    return float(np.mean(gaps)), float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


# ── Assembly: flat rows → sequences ──────────────────────────────────────────

def assemble_rows(
    rows_num: np.ndarray,    # (M,) amount_log
    rows_dtbin: np.ndarray,  # (M,) int
    rows_cat: np.ndarray,    # (M,) int
    rows_fraud: np.ndarray,  # (M,) int (0/1)
    N_entities: int,
    L: int,
    mode: str,               # "P2" (random) or "P1" (sequential)
    seed: int,
) -> tuple:
    """
    Assemble M rows into (N_entities, L) sequences.
    Mode P2: randomly permute all rows then chunk.
    Mode P1: sequential assignment (equivalent to P2 for row-level generators).
    Returns: x_num (N,L,1), dt_bin (N,L), x_cat (N,L,1), mask (N,L), y_seq (N,)
    """
    rng = np.random.default_rng(seed)
    n_rows = len(rows_num)
    target = N_entities * L

    if mode in ("P2", "P1"):
        idx = rng.permutation(n_rows)
        rows_num   = rows_num[idx]
        rows_dtbin = rows_dtbin[idx]
        rows_cat   = rows_cat[idx]
        rows_fraud = rows_fraud[idx]

    # Tile to reach target if needed
    if n_rows < target:
        repeats = target // n_rows + 1
        rows_num   = np.tile(rows_num,   repeats)[:target]
        rows_dtbin = np.tile(rows_dtbin, repeats)[:target]
        rows_cat   = np.tile(rows_cat,   repeats)[:target]
        rows_fraud = np.tile(rows_fraud, repeats)[:target]
    else:
        rows_num   = rows_num[:target]
        rows_dtbin = rows_dtbin[:target]
        rows_cat   = rows_cat[:target]
        rows_fraud = rows_fraud[:target]

    x_num_s  = rows_num.reshape(N_entities, L, 1).astype(np.float32)
    dt_bin_s = rows_dtbin.reshape(N_entities, L).astype(np.int32)
    x_cat_s  = rows_cat.reshape(N_entities, L, 1).astype(np.int32)
    mask_s   = np.ones((N_entities, L), dtype=bool)
    y_seq_s  = (rows_fraud.reshape(N_entities, L).max(1) > 0).astype(np.float32)

    return x_num_s, dt_bin_s, x_cat_s, mask_s, y_seq_s


# ── Flat table ────────────────────────────────────────────────────────────────

def make_flat_table(d) -> tuple:
    """
    Flatten (N, L) sequences to per-row table using only valid (mask=True) positions.
    Returns DataFrame and raw arrays for fast use.
    Uses position-level is_fraud: ~0.13% fraud rate per row → ~4% entity-level after assembly.
    """
    mask = d["mask"].astype(bool)   # (N, L)
    x_num  = d["x_num"][:, :, 0][mask]          # (M,)
    dt_bin = d["dt_bin"][mask].astype(np.int32)  # (M,)
    x_cat  = d["x_cat"][:, :, 0][mask].astype(np.int32)  # (M,)
    y_row  = d["y"][mask].astype(np.int32)       # (M,) position-level fraud

    df = pd.DataFrame({
        "amount_log": x_num.astype(np.float32),
        "dt_bin":     dt_bin,
        "category":   x_cat,
        "is_fraud":   y_row,
    })
    return df, x_num, dt_bin, x_cat, y_row


# ── Result recording ──────────────────────────────────────────────────────────

def make_result_row(
    generator: str,
    assembly: str,
    seed: int,
    g_synth: np.ndarray,
    y_synth: np.ndarray,
    g_real: np.ndarray,
    y_real: np.ndarray,
    notes: str = "",
    n_synth: int = 0,
    delta_aug_scarce: float = float("nan"),
) -> dict:
    """Compute all coherence_gap metrics and bootstrap CIs for a single generator run."""
    gaps = {}
    for fi, fname in zip(FEAT_IDX, FEAT_NAMES):
        gaps[f"coh_gap_{fname}"] = round(
            coherence_gap(g_synth, y_synth, g_real, y_real, feat_idx=fi), 6
        )

    gaps["coh_gap_mean"] = round(float(np.mean([gaps[f"coh_gap_{f}"] for f in FEAT_NAMES])), 6)

    # Bootstrap CI on mean coherence_gap (averaged across 4 features)
    rng = np.random.default_rng(42)
    N = len(g_synth)
    boot_means = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, N, size=N)
        val = float(np.mean([
            coherence_gap(g_synth[idx], y_synth[idx], g_real, y_real, feat_idx=fi)
            for fi in FEAT_IDX
        ]))
        boot_means.append(val)
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    fraud_rate = float(y_synth.mean())

    return {
        "generator":         generator,
        "assembly":          assembly,
        "seed":              seed,
        "n_synth":           n_synth if n_synth else len(g_synth),
        "fraud_rate_synth":  round(fraud_rate, 5),
        "coh_gap_vel":       gaps["coh_gap_vel"],
        "coh_gap_gap":       gaps["coh_gap_gap"],
        "coh_gap_fanout":    gaps["coh_gap_fanout"],
        "coh_gap_amt":       gaps["coh_gap_amt"],
        "coh_gap_mean":      gaps["coh_gap_mean"],
        "coh_gap_mean_ci_lo": round(ci_lo, 6),
        "coh_gap_mean_ci_hi": round(ci_hi, 6),
        "delta_aug_scarce":  round(delta_aug_scarce, 6) if not np.isnan(delta_aug_scarce) else "",
        "notes":             notes,
    }


# ── C0: real-vs-real floor ────────────────────────────────────────────────────

def run_C0_real_split(d, meta, tau, W, Bbins, n_cat, device, seed=42):
    """80/20 entity split: coherence_gap(held-out real, reference real) = noise floor."""
    N = d["x_num"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_ref = int(N * 0.8)
    ref_idx  = idx[:n_ref]
    held_idx = idx[n_ref:]

    y_seq = (d["y"] * d["mask"]).max(1).astype(np.float32)

    g_ref  = compute_g_seq_np(d["x_num"][ref_idx],  d["dt_bin"][ref_idx],
                               d["x_cat"][ref_idx],  d["mask"][ref_idx],
                               tau, W, Bbins, n_cat, device)
    g_held = compute_g_seq_np(d["x_num"][held_idx], d["dt_bin"][held_idx],
                               d["x_cat"][held_idx], d["mask"][held_idx],
                               tau, W, Bbins, n_cat, device)

    y_ref  = y_seq[ref_idx]
    y_held = y_seq[held_idx]

    return make_result_row(
        "C0_real_split", "native", seed,
        g_held, y_held, g_ref, y_ref,
        notes="real-vs-real floor; held-out 20% vs reference 80%",
        n_synth=len(held_idx),
    )


# ── C1: label shuffle upper bound ────────────────────────────────────────────

def run_C1_label_shuffle(d, meta, tau, W, Bbins, n_cat, device, g_real, y_real, seed=42):
    """Shuffle entity fraud labels; keeps g intact but destroys g→y coupling."""
    rng = np.random.default_rng(seed)
    y_seq = (d["y"] * d["mask"]).max(1).astype(np.float32)
    y_shuf = rng.permutation(y_seq)

    g_all = compute_g_seq_np(d["x_num"], d["dt_bin"], d["x_cat"], d["mask"],
                              tau, W, Bbins, n_cat, device)

    return make_result_row(
        "C1_label_shuf", "native", seed,
        g_all, y_shuf.astype(np.float32), g_real, y_real,
        notes="real sequences; entity fraud labels randomly permuted → upper bound",
        n_synth=len(y_shuf),
    )


# ── CoF: reuse sweep CSV ──────────────────────────────────────────────────────

def run_cof_from_csv(g_real, y_real, cof_csv_path, lam=None):
    """
    Extract CoF coherence_gap from existing sweep CSV.
    Uses teacher_type='full'; picks lam with lowest coh_gap_mean if lam not specified.
    Bootstrap CI is seed-level (10 seeds).
    """
    rows = list(csv.DictReader(open(cof_csv_path)))
    full_rows = [r for r in rows if r["teacher_type"] == "full"]
    if not full_rows:
        print("[CoF] No 'full' rows in CSV, skipping.")
        return []

    # Group by lambda
    by_lam = defaultdict(list)
    for r in full_rows:
        by_lam[float(r["coh_lambda"])].append(r)

    if lam is None:
        # Pick lambda with lowest mean(coh_gap_vel + coh_gap_fanout + coh_gap_amt)
        best_lam, best_mean = None, float("inf")
        for l, rs in by_lam.items():
            mean3 = np.mean([
                np.mean([float(r["coh_gap_vel"]), float(r["coh_gap_fanout"]), float(r["coh_gap_amt"])])
                for r in rs
            ])
            if mean3 < best_mean:
                best_mean, best_lam = mean3, l
        lam = best_lam

    results = []
    for rs in [by_lam[lam]]:
        vals_vel    = np.array([float(r["coh_gap_vel"])    for r in rs])
        vals_fanout = np.array([float(r["coh_gap_fanout"]) for r in rs])
        vals_amt    = np.array([float(r["coh_gap_amt"])    for r in rs])
        # gap (feat_idx=1) not in sweep CSV — use NaN
        mean_vel    = float(vals_vel.mean())
        mean_fanout = float(vals_fanout.mean())
        mean_amt    = float(vals_amt.mean())
        mean3       = float(np.mean([mean_vel, mean_fanout, mean_amt]))

        # Seed-level bootstrap CI on mean (vel+fanout+amt)/3
        rng = np.random.default_rng(42)
        n_s = len(rs)
        all_seed_means = (vals_vel + vals_fanout + vals_amt) / 3.0
        boots = [rng.choice(all_seed_means, size=n_s, replace=True).mean()
                 for _ in range(N_BOOT)]
        ci_lo = float(np.percentile(boots, 2.5))
        ci_hi = float(np.percentile(boots, 97.5))

        fraud_rate = float(np.mean([float(r["fraud_rate_synth"]) for r in rs]))
        n_synth    = int(rs[0]["n_train"]) if rs else 0

        results.append({
            "generator":          "cof",
            "assembly":           "native",
            "seed":               f"mean({n_s}seeds)",
            "n_synth":            n_synth,
            "fraud_rate_synth":   round(fraud_rate, 5),
            "coh_gap_vel":        round(mean_vel, 6),
            "coh_gap_gap":        "",          # not in sweep CSV
            "coh_gap_fanout":     round(mean_fanout, 6),
            "coh_gap_amt":        round(mean_amt, 6),
            "coh_gap_mean":       round(mean3, 6),
            "coh_gap_mean_ci_lo": round(ci_lo, 6),
            "coh_gap_mean_ci_hi": round(ci_hi, 6),
            "delta_aug_scarce":   "",
            "notes":              f"from sweep CSV; teacher=full λ={lam:.1f}; CI=seed-level({n_s}seeds)",
        })
    return results


# ── CFG-CoF: train class-conditional CoF, generate X|y ───────────────────────

def run_cof_cfg(
    d, meta, tau, W, Bbins, n_cat, device,
    g_real, y_real,
    lam: float = 2.0,
    seed: int = 1,
    n_steps: int = COF_N_STEPS,
    teacher_steps: int = COF_TEACHER_STEPS,
    guidance_scale: float = 1.0,
):
    """
    Train CFG-CoF (class-conditional SeqDenoiser + SequenceTeacher coherence)
    and measure coherence_gap vs real data.

    Key difference vs old CoF: SeqDenoiser.forward receives y_cond (entity fraud
    label) as input → fraud entities generate correct behavioral modes.
    """
    from models.coherence_teacher import SequenceTeacher, pretrain_sequence_teacher
    from models.seq_denoiser import SeqDenoiser
    from models.cof_seqgen import CoFSeqGen
    from models.sampler import ddim_sample, sample_empirical_dt_bin, sample_empirical_x_cat

    torch.manual_seed(seed)
    np.random.seed(seed)

    d_num      = meta["d_num"]
    n_cat_list = list(meta["num_classes_cat"].values())
    K_cat      = n_cat_list[0]
    L          = meta["L"]
    N          = d["x_num"].shape[0]

    x_num_t  = torch.from_numpy(d["x_num"].astype(np.float32))
    dt_bin_t = torch.from_numpy(d["dt_bin"].astype(np.int64))
    x_cat_t  = torch.from_numpy(d["x_cat"].astype(np.int64))
    y_t      = torch.from_numpy(d["y"].astype(np.float32))
    mask_t   = torch.from_numpy(d["mask"].astype(bool))

    y_seq     = ((y_t * mask_t.float()).max(dim=1).values > 0).float().numpy()
    real_prev = float(y_seq.mean())

    # ── 1. Pretrain SequenceTeacher ──────────────────────────────────────────
    print(f"  [CFG-CoF λ={lam} seed={seed}] Pretraining teacher ({teacher_steps} steps)...")
    f_phi_seq = SequenceTeacher(
        d_num=d_num, Bbins=Bbins, K_cat=K_cat, h=64, n_layers=2,
        use_time=True, use_recv=True,
    ).to(device)
    pretrain_sequence_teacher(
        f_phi_seq, x_num_t, dt_bin_t, x_cat_t, y_t, mask_t,
        Bbins=Bbins, K_cat=K_cat,
        steps=teacher_steps, lr=1e-3, batch_size=256, device=device,
    )

    # ── 2. Train CFG-CoF ─────────────────────────────────────────────────────
    print(f"  [CFG-CoF λ={lam} seed={seed}] Training CoF ({n_steps} steps)...")
    denoiser = SeqDenoiser(
        d_num=d_num, Bbins=Bbins, n_cat_classes=n_cat_list,
        d_model=COF_D_MODEL, n_heads=4, n_layers=2, L_max=64,
    )
    model = CoFSeqGen(
        denoiser=denoiser, tau=tau.cpu(), W=W, temp=TEMP,
        coh_lambda=lam, n_cat_classes=n_cat_list,
        coherence_teacher=f_phi_seq, cfg_dropout=0.15,
    )
    model.train().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=COF_LR)

    for step in range(1, n_steps + 1):
        idx    = torch.randint(0, N, (COF_BATCH_SIZE,))
        t_frac = torch.rand(1).item() * 0.9 + 0.1
        optim.zero_grad()
        loss, _ = model.compute_loss(
            x_num_t[idx].to(device), dt_bin_t[idx].to(device),
            x_cat_t[idx].to(device), y_t[idx].to(device),
            mask_t[idx].to(device), t_frac,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step % 5000 == 0:
            print(f"    step {step}/{n_steps}  loss={loss.item():.4f}")

    # ── 3. Generate with y_cond (class-conditional) ──────────────────────────
    print(f"  [CFG-CoF λ={lam} seed={seed}] Generating {N} entities...")
    model.eval()
    x_num_chunks, bin_chunks, cat_chunks, mask_chunks, ycond_chunks = [], [], [], [], []

    for start in range(0, N, COF_BATCH_SIZE):
        end = min(start + COF_BATCH_SIZE, N)
        bs  = end - start
        dt_samp   = sample_empirical_dt_bin(dt_bin_t, bs, L, Bbins, device=device)
        xcat_samp = sample_empirical_x_cat(x_cat_t, bs, device=device)
        mask_samp = mask_t[torch.randint(0, N, (bs,))]
        # Pre-sample entity fraud labels at real prevalence → X|y generation
        y_cond_b = torch.bernoulli(torch.full((bs,), real_prev)).long().to(device)
        x_gen, bin_pred, cat_preds, _, _ = ddim_sample(
            model, dt_samp, xcat_samp,
            d_num=d_num, T_steps=COF_T_DIFF, device=device, return_discrete=True,
            y_cond=y_cond_b, guidance_scale=guidance_scale,
            feedback_discrete=True, feedback_after=0.3, discrete_temp=1.0,
            start_from_mask=True,
            valid_mask=mask_samp.to(device),
        )
        x_num_chunks.append(x_gen.cpu().numpy())
        bin_chunks.append(bin_pred.cpu().numpy())
        cat_chunks.append(cat_preds[0].cpu().numpy() if cat_preds
                          else np.zeros((bs, L), dtype=np.int32))
        mask_chunks.append(mask_samp.numpy())
        ycond_chunks.append(y_cond_b.cpu().numpy())

    x_num_synth  = np.concatenate(x_num_chunks, axis=0)            # (N, L, 1)
    dt_bin_synth = np.concatenate(bin_chunks, axis=0)               # (N, L)
    x_cat_synth  = np.concatenate(cat_chunks, axis=0)[:, :, None]   # (N, L, 1)
    mask_synth   = np.concatenate(mask_chunks, axis=0)              # (N, L)
    y_gen_seq    = np.concatenate(ycond_chunks, axis=0).astype(np.float32)  # (N,)

    fraud_synth = float(y_gen_seq.mean())
    print(f"    fraud_synth={fraud_synth:.4f}  target={real_prev:.4f}")

    # ── 4. Coherence gap ─────────────────────────────────────────────────────
    g_synth = compute_g_seq_np(x_num_synth, dt_bin_synth, x_cat_synth, mask_synth,
                                tau, W, Bbins, n_cat, device)

    return make_result_row(
        "cof_cfg", "native", seed,
        g_synth, y_gen_seq, g_real, y_real,
        notes=(f"CFG-CoF λ={lam} seed={seed} guidance={guidance_scale:.1f}; "
               f"y_cond=Bernoulli({real_prev:.4f})"),
        n_synth=N,
    )


# ── Row-shuffle ───────────────────────────────────────────────────────────────

def run_row_shuffle(d, meta, tau, W, Bbins, n_cat, device, g_real, y_real, seed=42):
    """
    Flatten all valid rows, randomly assign to N entities of length L.
    Preserves marginal distributions but destroys within-entity coherence.
    Entity fraud labels: Bernoulli(real_prev) — independent of features.
    (Using max(row_fraud) fails when row-level fraud rate >> position-level rate,
    e.g. in κ-data where all positions of a fraud entity are labeled fraud.)
    """
    N, L = d["mask"].shape
    real_prev = float(y_real.mean())
    _, rows_num, rows_dtbin, rows_cat, rows_fraud = make_flat_table(d)

    rows_fraud_dummy = np.zeros(len(rows_num), dtype=np.int32)
    x_num_s, dt_bin_s, x_cat_s, mask_s, _ = assemble_rows(
        rows_num, rows_dtbin, rows_cat, rows_fraud_dummy, N, L, mode="P2", seed=seed
    )
    # Override entity labels with Bernoulli(real_prev) — consistent with CTGAN/TVAE
    rng = np.random.default_rng(seed)
    y_seq_s = rng.binomial(1, real_prev, size=N).astype(np.float32)

    g_synth = compute_g_seq_np(x_num_s, dt_bin_s, x_cat_s, mask_s,
                                tau, W, Bbins, n_cat, device)

    return make_result_row(
        "row_shuffle", "P2", seed,
        g_synth, y_seq_s, g_real, y_real,
        notes=(f"real rows randomly reassigned to entities; marginals preserved, coherence destroyed; "
               f"entity labels=Bernoulli({real_prev:.4f})"),
        n_synth=N,
    )


# ── CTGAN / TVAE (calibrated) ─────────────────────────────────────────────────
# Train on features ONLY (no is_fraud column) to avoid CTGAN's conditional
# over-sampling of the minority class (0.13% row fraud → 98% entity fraud).
# Entity fraud labels are assigned independently: Bernoulli(real_prevalence).
# This is the cleanest row-level baseline: features from model, labels from prior.

def _run_row_generator(
    gen_name: str,
    model_cls,
    model_kwargs: dict,
    d, meta, tau, W, Bbins, n_cat, device, g_real, y_real,
    epochs: int = 30, seed: int = 1,
):
    N, L = d["mask"].shape
    flat_df, _, _, _, _ = make_flat_table(d)
    # Drop is_fraud: prevent conditional over-sampling on minority class
    feat_df = flat_df.drop(columns=["is_fraud"])
    n_sample = N * L
    K_cat = n_cat[0] if n_cat else 1
    real_prev = float(y_real.mean())

    print(f"  [{gen_name.upper()}] Training on {len(feat_df)} rows (no is_fraud), "
          f"epochs={epochs}, seed={seed}...")
    model = model_cls(**model_kwargs, epochs=epochs, verbose=True, batch_size=500)
    model.fit(feat_df, discrete_columns=["dt_bin", "category"])

    print(f"  [{gen_name.upper()}] Sampling {n_sample} rows...")
    synth_df = model.sample(n_sample)

    rows_num   = synth_df["amount_log"].values.astype(np.float32)
    rows_dtbin = synth_df["dt_bin"].values.astype(np.int32).clip(0, Bbins - 1)
    rows_cat   = synth_df["category"].values.astype(np.int32).clip(0, K_cat - 1)
    # Assign entity labels independently: Bernoulli(real prevalence)
    # This decouples feature generation from fraud labeling — correct for row-level test.
    rng_seed = np.random.default_rng(seed)
    rows_fraud_dummy = np.zeros(n_sample, dtype=np.int32)  # placeholder, overridden below

    x_num_s, dt_bin_s, x_cat_s, mask_s, _ = assemble_rows(
        rows_num, rows_dtbin, rows_cat, rows_fraud_dummy, N, L, mode="P2", seed=seed
    )
    # Override entity labels with Bernoulli(real_prev) — independent of features
    y_seq_s = rng_seed.binomial(1, real_prev, size=N).astype(np.float32)

    g_synth = compute_g_seq_np(x_num_s, dt_bin_s, x_cat_s, mask_s,
                                tau, W, Bbins, n_cat, device)

    return make_result_row(
        gen_name, "P2", seed,
        g_synth, y_seq_s, g_real, y_real,
        notes=(f"{gen_name.upper()} epochs={epochs}; features only (no is_fraud); "
               f"entity labels=Bernoulli({real_prev:.4f})"),
        n_synth=N,
    )


def run_ctgan(d, meta, tau, W, Bbins, n_cat, device, g_real, y_real,
              epochs: int = 30, seed: int = 1):
    from ctgan import CTGAN
    return _run_row_generator(
        "ctgan", CTGAN, {"cuda": (device != "cpu")},
        d, meta, tau, W, Bbins, n_cat, device, g_real, y_real, epochs, seed,
    )


def run_tvae(d, meta, tau, W, Bbins, n_cat, device, g_real, y_real,
             epochs: int = 30, seed: int = 1):
    from ctgan import TVAE
    return _run_row_generator(
        "tvae", TVAE, {"cuda": (device != "cpu")},
        d, meta, tau, W, Bbins, n_cat, device, g_real, y_real, epochs, seed,
    )


# ── CSV output ────────────────────────────────────────────────────────────────

FIELDS = [
    "generator", "assembly", "seed", "n_synth", "fraud_rate_synth",
    "coh_gap_vel", "coh_gap_gap", "coh_gap_fanout", "coh_gap_amt",
    "coh_gap_mean", "coh_gap_mean_ci_lo", "coh_gap_mean_ci_hi",
    "delta_aug_scarce", "notes",
]


def append_results(rows: list, out_path: Path):
    exists = out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)
    for row in rows:
        print(f"  [SAVED] {row['generator']:>15} {row['assembly']:>6}  "
              f"coh_gap_mean={row['coh_gap_mean']:.5f} "
              f"CI=[{row['coh_gap_mean_ci_lo']:.5f},{row['coh_gap_mean_ci_hi']:.5f}]  "
              f"fraud={row['fraud_rate_synth']:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generators", default="C0,C1,cof,row_shuffle,ctgan,tvae",
                   help="comma-separated list; add 'cof_cfg' for CFG-CoF")
    p.add_argument("--ctgan_epochs",       type=int,   default=100)
    p.add_argument("--tvae_epochs",        type=int,   default=100)
    p.add_argument("--seeds",              type=int,   default=1,
                   help="seeds for row_shuffle/ctgan/tvae (C0/C1/cof use fixed seed)")
    p.add_argument("--cof_lam",            type=float, default=None,
                   help="Old CoF lambda (default: auto-pick)")
    # CFG-CoF options
    p.add_argument("--cof_cfg_lam",        type=float, default=2.0,
                   help="CFG-CoF coherence lambda (default 2.0 = best from old sweep)")
    p.add_argument("--cof_cfg_seeds",      type=int,   default=3,
                   help="number of seeds for CFG-CoF (1..N)")
    p.add_argument("--cof_cfg_steps",      type=int,   default=COF_N_STEPS)
    p.add_argument("--cof_cfg_teacher_steps", type=int, default=COF_TEACHER_STEPS)
    p.add_argument("--cof_cfg_guidance",   type=float, default=1.0,
                   help="CFG guidance scale (1.0=conditional only, >1=full CFG)")
    p.add_argument("--W",                  type=float, default=W_DEFAULT)
    p.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out",                default=str(ROOT / "results" / "baseline_coherence.csv"))
    p.add_argument("--data_root",          default=None,
                   help="Path to sequences/ dir (overrides default AMLSim path)")
    p.add_argument("--ref_split",          default="train",
                   help="Split for g_real/y_real reference (train or test). "
                        "Generators always train on train.npz; only the coherence reference changes.")
    return p.parse_args()


def main():
    args      = parse_args()
    out_path  = Path(args.out)
    gens      = [g.strip() for g in args.generators.split(",")]
    device    = args.device

    print(f"Device: {device}")
    print(f"Generators: {gens}")
    print(f"Output: {out_path}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.data_root:
        data_seq_dir = Path(args.data_root)
        print(f"[Step 1] Loading train data from {data_seq_dir}...")
        d_raw = np.load(data_seq_dir / "train.npz")
        with open(data_seq_dir / "meta.json") as f:
            meta = json.load(f)
        d = d_raw
        # Reference split (for g_real/y_real): generators train on train, reference may differ
        if args.ref_split != "train":
            ref_path = data_seq_dir / f"{args.ref_split}.npz"
            print(f"  [ref_split={args.ref_split}] Loading reference from {ref_path}...")
            d_ref = np.load(ref_path)
        else:
            d_ref = d
    else:
        print("[Step 1] Loading AMLSim train data...")
        d, meta = load_amlsim("train")
        if args.ref_split != "train":
            print(f"  [ref_split={args.ref_split}] Loading AMLSim reference from {args.ref_split}...")
            d_ref, _ = load_amlsim(args.ref_split)
        else:
            d_ref = d
    N, L    = d["mask"].shape
    Bbins   = meta["B"]
    n_cat   = list(meta["num_classes_cat"].values())  # [256]

    tau = make_tau_tensor(meta, device)

    # ── Compute g_real, y_real from reference split ───────────────────────────
    print(f"[Step 2] Computing g_real and y_real from {args.ref_split} set...")
    y_seq = (d_ref["y"] * d_ref["mask"]).max(1).astype(np.float32)
    g_real = compute_g_seq_np(
        d_ref["x_num"], d_ref["dt_bin"], d_ref["x_cat"], d_ref["mask"],
        tau, args.W, Bbins, n_cat, device
    )
    y_real = y_seq
    print(f"  g_real shape: {g_real.shape}  y_real fraud rate: {y_real.mean():.4f}")

    # ── Run generators ────────────────────────────────────────────────────────
    if "C0" in gens:
        print("\n[C0] Real-vs-real floor (80/20 split)...")
        row = run_C0_real_split(d, meta, tau, args.W, Bbins, n_cat, device)
        append_results([row], out_path)

    if "C1" in gens:
        print("\n[C1] Label shuffle upper bound...")
        row = run_C1_label_shuffle(d, meta, tau, args.W, Bbins, n_cat, device,
                                   g_real, y_real)
        append_results([row], out_path)

    if "cof" in gens:
        print("\n[CoF] Loading from sweep CSV...")
        cof_csv = ROOT / "results" / "amlsim_seqteacher.csv"
        if cof_csv.exists():
            rows = run_cof_from_csv(g_real, y_real, cof_csv, lam=args.cof_lam)
            append_results(rows, out_path)
        else:
            print(f"  [CoF] CSV not found: {cof_csv}")

    if "cof_cfg" in gens:
        print(f"\n[CFG-CoF] λ={args.cof_cfg_lam}, {args.cof_cfg_seeds} seed(s), "
              f"n_steps={args.cof_cfg_steps}, guidance={args.cof_cfg_guidance}...")
        for seed in range(1, args.cof_cfg_seeds + 1):
            print(f"  seed={seed}...")
            row = run_cof_cfg(
                d, meta, tau, args.W, Bbins, n_cat, device,
                g_real, y_real,
                lam=args.cof_cfg_lam,
                seed=seed,
                n_steps=args.cof_cfg_steps,
                teacher_steps=args.cof_cfg_teacher_steps,
                guidance_scale=args.cof_cfg_guidance,
            )
            append_results([row], out_path)

    if "row_shuffle" in gens:
        print(f"\n[Row-shuffle] Running {args.seeds} seed(s)...")
        for seed in range(1, args.seeds + 1):
            print(f"  seed={seed}...")
            row = run_row_shuffle(d, meta, tau, args.W, Bbins, n_cat, device,
                                  g_real, y_real, seed=seed)
            append_results([row], out_path)

    if "ctgan" in gens:
        print(f"\n[CTGAN] epochs={args.ctgan_epochs}, {args.seeds} seed(s)...")
        for seed in range(1, args.seeds + 1):
            print(f"  seed={seed}...")
            row = run_ctgan(d, meta, tau, args.W, Bbins, n_cat, device,
                            g_real, y_real,
                            epochs=args.ctgan_epochs, seed=seed)
            append_results([row], out_path)

    if "tvae" in gens:
        print(f"\n[TVAE] epochs={args.tvae_epochs}, {args.seeds} seed(s)...")
        for seed in range(1, args.seeds + 1):
            print(f"  seed={seed}...")
            row = run_tvae(d, meta, tau, args.W, Bbins, n_cat, device,
                           g_real, y_real,
                           epochs=args.tvae_epochs, seed=seed)
            append_results([row], out_path)

    print(f"\n[Done] Results written to {out_path}")
    print(f"  Total rows: {sum(1 for _ in open(out_path)) - 1}")


if __name__ == "__main__":
    main()
