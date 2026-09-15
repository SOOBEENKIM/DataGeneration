"""
Fidelity + memorization analysis at κ=1.0 (seed=1).

Computes:
  1. TVD: total variation distance on marginal distributions
     (amount_log, dt_bin, category) for each generator.
  2. CovErr: mean behavioral feature distribution error
     |mean(g_synth) - mean(g_real)| / std(g_real) per channel.
  3. Privacy (NN ratio): distance to nearest real entity in g-feature space,
     normalized by real-vs-real NN distance. Ratio > 1 → no memorization.

Usage:
  python scripts/compute_fidelity.py --kappa 1.00 [--guidance 2.0]

Output: results/fidelity_table.csv, figs/fidelity_table.tex
"""

import sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.baseline_coherence_harness import (
    make_flat_table, assemble_rows, compute_g_seq_np,
    run_C0_real_split,
    COF_N_STEPS, COF_TEACHER_STEPS, COF_D_MODEL, COF_LR,
    COF_BATCH_SIZE, COF_T_DIFF, TEMP,
)
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kappa",    type=float, default=1.0)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--seed",     type=int,   default=1)
    p.add_argument("--ctgan_epochs", type=int, default=30,
                   help="Epochs for CTGAN/TVAE (fewer than full run, for speed)")
    return p.parse_args()


def tvd(a: np.ndarray, b: np.ndarray, n_bins: int = 30, lo=None, hi=None) -> float:
    """Total variation distance between two 1-D distributions."""
    lo = min(a.min(), b.min()) if lo is None else lo
    hi = max(a.max(), b.max()) if hi is None else hi
    bins = np.linspace(lo, hi, n_bins + 1)
    pa, _ = np.histogram(a, bins=bins, density=True)
    pb, _ = np.histogram(b, bins=bins, density=True)
    pa /= pa.sum() + 1e-12
    pb /= pb.sum() + 1e-12
    return 0.5 * float(np.abs(pa - pb).sum())


def tvd_discrete(a: np.ndarray, b: np.ndarray) -> float:
    """TVD for discrete (categorical) arrays."""
    vals = sorted(set(a.tolist()) | set(b.tolist()))
    n = len(vals)
    v2i = {v: i for i, v in enumerate(vals)}
    pa = np.zeros(n)
    pb = np.zeros(n)
    for v in a: pa[v2i[v]] += 1
    for v in b: pb[v2i[v]] += 1
    pa /= pa.sum() + 1e-12
    pb /= pb.sum() + 1e-12
    return 0.5 * float(np.abs(pa - pb).sum())


def nn_ratio(g_synth: np.ndarray, g_real: np.ndarray, subsample: int = 2000) -> float:
    """
    Nearest-neighbor privacy ratio:
      d_SR / d_RR
    where d_SR = avg NN distance (synth → real) and d_RR = avg NN distance (real → real, leave-one-out).
    Ratio > 1 means synthetic entities are farther from real than real entities are from each other.
    """
    rng = np.random.default_rng(42)
    N = min(len(g_synth), len(g_real), subsample)
    s_idx = rng.choice(len(g_synth), N, replace=False)
    r_idx = rng.choice(len(g_real),  N, replace=False)
    gs = g_synth[s_idx]
    gr = g_real[r_idx]

    # SR: for each synth, find nearest real
    from sklearn.metrics.pairwise import euclidean_distances
    D_sr = euclidean_distances(gs, gr)
    np.fill_diagonal(D_sr, np.inf)
    d_sr = D_sr.min(axis=1).mean()

    # RR: leave-one-out
    D_rr = euclidean_distances(gr, gr)
    np.fill_diagonal(D_rr, np.inf)
    d_rr = D_rr.min(axis=1).mean()

    return float(d_sr / (d_rr + 1e-12))


def extract_marginals(x_num, dt_bin, x_cat, mask):
    """Extract flat valid-position arrays for TVD computation."""
    m = mask.astype(bool)
    amt  = x_num[m, 0] if x_num.ndim == 3 else x_num[m]
    dbin = dt_bin[m]
    cat  = x_cat[m, 0] if x_cat.ndim == 3 else x_cat[m]
    return amt, dbin.astype(int), cat.astype(int)


def main():
    args = parse_args()
    tag  = f"{args.kappa:.2f}"
    data_root = ROOT / f"data/kappa_{tag}/sequences"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fidelity] κ={tag}, seed={args.seed}, guidance={args.guidance}, device={device}")

    # ── Load real data ──────────────────────────────────────────────────────────
    d    = {k: v for k, v in np.load(data_root / "train.npz").items()}
    meta = json.load(open(data_root / "meta.json"))
    tau  = torch.tensor(meta["tau_k"], dtype=torch.float32)
    W    = meta.get("W_default", 7.0)
    Bbins = meta["B"]
    n_cat = list(meta["num_classes_cat"].values())

    # Compute real behavioral features (entity-level, shape N×4)
    N = d["x_num"].shape[0]
    g_real = compute_g_seq_np(
        d["x_num"], d["dt_bin"],
        d["x_cat"] if d["x_cat"].ndim == 3 else d["x_cat"][:, :, None],
        d["mask"], tau, W, Bbins, n_cat, device
    )
    mask_r = d["mask"].astype(bool)
    y_real = (d["y"].astype(bool).max(axis=1) if d["y"].ndim == 2 else d["y"].astype(bool)).astype(float)

    real_amt, real_dbin, real_cat = extract_marginals(d["x_num"], d["dt_bin"], d["x_cat"], d["mask"])

    results = []

    # ── Helper: compute metrics from synthetic data ─────────────────────────────
    def eval_synth(name, x_num_s, dt_bin_s, x_cat_s, mask_s, y_s):
        synth_amt, synth_dbin, synth_cat = extract_marginals(x_num_s, dt_bin_s, x_cat_s, mask_s)
        tvd_amt  = tvd(real_amt,  synth_amt,  lo=real_amt.min(),  hi=real_amt.max())
        tvd_dbin = tvd_discrete(real_dbin, synth_dbin)
        tvd_cat  = tvd_discrete(real_cat,  synth_cat)
        tvd_mean = (tvd_amt + tvd_dbin + tvd_cat) / 3.0

        g_s = compute_g_seq_np(x_num_s, dt_bin_s,
                                x_cat_s if x_cat_s.ndim==3 else x_cat_s[:,:,None],
                                mask_s, tau, W, Bbins, n_cat, device)
        cov_err = float(np.mean(np.abs(g_s.mean(0) - g_real.mean(0)) / (g_real.std(0) + 1e-8)))
        nn_r = nn_ratio(g_s, g_real)
        fraud = float(y_s.mean())
        print(f"  {name:<15} TVD_amt={tvd_amt:.4f} TVD_dbin={tvd_dbin:.4f} "
              f"TVD_cat={tvd_cat:.4f} CovErr={cov_err:.4f} NN_ratio={nn_r:.3f} fraud={fraud:.4f}")
        results.append({
            "generator": name, "seed": args.seed,
            "tvd_amt": round(tvd_amt, 4), "tvd_dbin": round(tvd_dbin, 4),
            "tvd_cat": round(tvd_cat, 4), "tvd_mean": round(tvd_mean, 4),
            "cov_err": round(cov_err, 4), "nn_ratio": round(nn_r, 4),
            "fraud_rate": round(fraud, 4),
        })

    # ── Row-shuffle ─────────────────────────────────────────────────────────────
    print("\n[Row-shuffle]")
    flat_df, rows_num, rows_dtbin, rows_cat, rows_fraud = make_flat_table(d)
    x_rs, dt_rs, xcat_rs, mask_rs, y_rs = assemble_rows(
        rows_num, rows_dtbin, rows_cat, rows_fraud, N, meta["L"], "P2", args.seed
    )
    eval_synth("row_shuffle", x_rs, dt_rs, xcat_rs, mask_rs, y_rs)

    # ── CTGAN ───────────────────────────────────────────────────────────────────
    print(f"\n[CTGAN] (epochs={args.ctgan_epochs})")
    try:
        from ctgan import CTGAN
        feat_df = flat_df.drop(columns=["is_fraud"])
        model_ct = CTGAN(cuda=(device.type != "cpu"), epochs=args.ctgan_epochs, verbose=False, batch_size=500)
        model_ct.fit(feat_df, discrete_columns=["dt_bin", "category"])
        synth_ct = model_ct.sample(N * meta["L"])
        rows_ct_n = synth_ct["amount_log"].values.astype(np.float32)
        rows_ct_d = synth_ct["dt_bin"].values.astype(np.int32).clip(0, Bbins-1)
        rows_ct_c = synth_ct["category"].values.astype(np.int32).clip(0, n_cat[0]-1) if n_cat else np.zeros(len(synth_ct), np.int32)
        x_ct, dt_ct, xcat_ct, mask_ct, y_ct = assemble_rows(
            rows_ct_n, rows_ct_d, rows_ct_c, np.zeros(len(rows_ct_n), np.int32),
            N, meta["L"], "P2", args.seed
        )
        rng = np.random.default_rng(args.seed)
        y_ct = rng.binomial(1, float(y_real.mean()), size=N).astype(np.float32)
        eval_synth("ctgan", x_ct, dt_ct, xcat_ct, mask_ct, y_ct)
    except Exception as e:
        print(f"  CTGAN failed: {e}")

    # ── TVAE ────────────────────────────────────────────────────────────────────
    print(f"\n[TVAE] (epochs={args.ctgan_epochs})")
    try:
        from ctgan import TVAE
        feat_df2 = flat_df.drop(columns=["is_fraud"])
        model_tv = TVAE(cuda=(device.type != "cpu"), epochs=args.ctgan_epochs, batch_size=500)
        model_tv.fit(feat_df2, discrete_columns=["dt_bin", "category"])
        synth_tv = model_tv.sample(N * meta["L"])
        rows_tv_n = synth_tv["amount_log"].values.astype(np.float32)
        rows_tv_d = synth_tv["dt_bin"].values.astype(np.int32).clip(0, Bbins-1)
        rows_tv_c = synth_tv["category"].values.astype(np.int32).clip(0, n_cat[0]-1) if n_cat else np.zeros(len(synth_tv), np.int32)
        x_tv, dt_tv, xcat_tv, mask_tv, y_tv = assemble_rows(
            rows_tv_n, rows_tv_d, rows_tv_c, np.zeros(len(rows_tv_n), np.int32),
            N, meta["L"], "P2", args.seed
        )
        rng = np.random.default_rng(args.seed + 100)
        y_tv = rng.binomial(1, float(y_real.mean()), size=N).astype(np.float32)
        eval_synth("tvae", x_tv, dt_tv, xcat_tv, mask_tv, y_tv)
    except Exception as e:
        print(f"  TVAE failed: {e}")

    # ── CFG-CoF ─────────────────────────────────────────────────────────────────
    print(f"\n[CFG-CoF] (guidance={args.guidance})")
    from models.coherence_teacher import SequenceTeacher, pretrain_sequence_teacher
    from models.seq_denoiser import SeqDenoiser
    from models.cof_seqgen import CoFSeqGen
    from models.sampler import ddim_sample, sample_empirical_dt_bin, sample_empirical_x_cat

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d_num = meta["d_num"]
    n_cat_list = list(meta["num_classes_cat"].values())
    K_cat = n_cat_list[0]
    L = meta["L"]
    real_prev = float(y_real.mean())

    x_num_t  = torch.from_numpy(d["x_num"].astype(np.float32))
    dt_bin_t = torch.from_numpy(d["dt_bin"].astype(np.int64))
    x_cat_t  = torch.from_numpy(d["x_cat"].astype(np.int64))
    y_t      = torch.from_numpy(d["y"].astype(np.float32))
    mask_t   = torch.from_numpy(d["mask"].astype(bool))

    f_phi_seq = SequenceTeacher(d_num=d_num, Bbins=Bbins, K_cat=K_cat, h=64, n_layers=2,
                                 use_time=True, use_recv=True).to(device)
    pretrain_sequence_teacher(f_phi_seq, x_num_t, dt_bin_t, x_cat_t, y_t, mask_t,
                               Bbins=Bbins, K_cat=K_cat, steps=COF_TEACHER_STEPS,
                               lr=1e-3, batch_size=256, device=device)

    denoiser = SeqDenoiser(d_num=d_num, Bbins=Bbins, n_cat_classes=n_cat_list,
                            d_model=COF_D_MODEL, n_heads=4, n_layers=2, L_max=64)
    model = CoFSeqGen(denoiser=denoiser, tau=tau.cpu(), W=W, temp=TEMP,
                       coh_lambda=0.0, n_cat_classes=n_cat_list,
                       coherence_teacher=f_phi_seq, cfg_dropout=0.15)
    model.train().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=COF_LR)
    for step in range(1, COF_N_STEPS + 1):
        idx    = torch.randint(0, N, (COF_BATCH_SIZE,))
        t_frac = torch.rand(1).item() * 0.9 + 0.1
        optim.zero_grad()
        loss, _ = model.compute_loss(x_num_t[idx].to(device), dt_bin_t[idx].to(device),
                                      x_cat_t[idx].to(device), y_t[idx].to(device),
                                      mask_t[idx].to(device), t_frac)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step % 5000 == 0:
            print(f"    step {step}/{COF_N_STEPS}  loss={loss.item():.4f}")

    model.eval()
    x_num_chunks, bin_chunks, cat_chunks, mask_chunks, yc_chunks = [], [], [], [], []
    for start in range(0, N, COF_BATCH_SIZE):
        end = min(start + COF_BATCH_SIZE, N)
        bs  = end - start
        dt_samp   = sample_empirical_dt_bin(dt_bin_t, bs, L, Bbins, device=device)
        xcat_samp = sample_empirical_x_cat(x_cat_t, bs, device=device)
        mask_samp = mask_t[torch.randint(0, N, (bs,))]
        y_cond_b  = torch.bernoulli(torch.full((bs,), real_prev)).long().to(device)
        x_gen, bin_pred, cat_preds, _, _ = ddim_sample(
            model, dt_samp, xcat_samp, d_num=d_num, T_steps=COF_T_DIFF, device=device,
            return_discrete=True, y_cond=y_cond_b, guidance_scale=args.guidance,
            feedback_discrete=True, feedback_after=0.3, discrete_temp=1.0, start_from_mask=True,
            valid_mask=mask_samp.to(device),
        )
        x_num_chunks.append(x_gen.cpu().numpy())
        bin_chunks.append(bin_pred.cpu().numpy())
        cat_chunks.append(cat_preds[0].cpu().numpy() if cat_preds else np.zeros((bs, L), np.int32))
        mask_chunks.append(mask_samp.numpy())
        yc_chunks.append(y_cond_b.cpu().numpy())

    x_cof = np.concatenate(x_num_chunks)
    dt_cof = np.concatenate(bin_chunks)
    xcat_cof = np.concatenate(cat_chunks)[:, :, None]
    mask_cof = np.concatenate(mask_chunks)
    y_cof   = np.concatenate(yc_chunks).astype(np.float32)
    eval_synth("cof_cfg", x_cof, dt_cof, xcat_cof, mask_cof, y_cof)

    # ── Save ────────────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out = ROOT / "results" / "fidelity_table.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # LaTeX
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fidelity and privacy metrics at $\kappa=1.0$ (seed=1). "
        r"TVD = total variation distance (lower = better fidelity). "
        r"NN ratio $> 1$ indicates no memorization.}",
        r"\label{tab:fidelity}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Generator} & TVD$_{\rm amt}$ & TVD$_{\rm dbin}$ & TVD$_{\rm cat}$ & TVD$_{\rm mean}$ & CovErr & NN ratio \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        star = r"$\star$" if r["generator"] == "cof_cfg" else ""
        tex_lines.append(
            rf"  {r['generator']}{star} & {r['tvd_amt']:.4f} & {r['tvd_dbin']:.4f} "
            rf"& {r['tvd_cat']:.4f} & \textbf{{{r['tvd_mean']:.4f}}} "
            rf"& {r['cov_err']:.4f} & {r['nn_ratio']:.3f} \\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_out = ROOT / "figs" / "fidelity_table.tex"
    tex_out.parent.mkdir(exist_ok=True)
    tex_out.write_text("\n".join(tex_lines) + "\n")
    print(f"LaTeX: {tex_out}")

    print("\n[Summary]")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
