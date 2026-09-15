"""
Plot the κ-curve: coherence_gap vs κ for each generator.

Shows:
  - CFG-CoF (λ=2.0, 3 seeds, with CI)
  - row_shuffle / CTGAN / TVAE (single seed)
  - C0 floor (dashed)
  - C1 upper bound (dashed)

Usage:
  python scripts/plot_kappa_curve.py --kappas 0.0 0.3 0.5 0.7 1.0 --out figs/kappa_curve.pdf
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent


def load_csv(csv_path):
    if not Path(csv_path).exists():
        return []
    return list(csv.DictReader(open(csv_path)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kappas",    type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 1.0])
    p.add_argument("--results_root", default=str(ROOT / "results"))
    p.add_argument("--out",       default=str(ROOT / "figs" / "kappa_curve.pdf"))
    p.add_argument("--csv_name",  default="kappa_curve.csv",
                   help="CSV filename to load per κ (e.g. kappa_curve_g2.csv)")
    return p.parse_args()


def get_stat(rows, gen, asm="P2"):
    """Return mean ± CI for a generator across seeds.

    Multi-seed: CI = mean ± 1.96·SEM (across-seed variability).
    Single-seed: CI = within-seed bootstrap CI from the harness.
    """
    rs = [r for r in rows if r["generator"] == gen and r["assembly"] == asm]
    if not rs:
        rs = [r for r in rows if r["generator"] == gen]  # any assembly
    if not rs:
        return None
    means = [float(r["coh_gap_mean"]) for r in rs]
    m = float(np.mean(means))
    n = len(rs)
    if n > 1:
        sem = float(np.std(means, ddof=1) / np.sqrt(n))
        ci_lo = m - 1.96 * sem
        ci_hi = m + 1.96 * sem
    else:
        ci_lo = float(rs[0]["coh_gap_mean_ci_lo"])
        ci_hi = float(rs[0]["coh_gap_mean_ci_hi"])
    return {"mean": m, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}


def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    g_label = "2.0" if "g2" in args.csv_name else "3.0"
    GEN_STYLES = {
        "cof_cfg":    dict(color="#1f77b4", label=f"CFG-CoF (ours, g={g_label})", lw=2.2, marker="o", zorder=5),
        "row_shuffle": dict(color="#d62728", label="Row-shuffle",    lw=1.5, marker="s", ls="--"),
        "ctgan":       dict(color="#ff7f0e", label="CTGAN",          lw=1.5, marker="^", ls="-.", zorder=3),
        "tvae":        dict(color="#9467bd", label="TVAE",           lw=1.5, marker="D", ls=":",  zorder=3),
    }
    CTRL_STYLES = {
        "C0_real_split": dict(color="gray",  label="C0 (real vs real)",  ls="--", lw=1.0, alpha=0.7),
        "C1_label_shuf": dict(color="black", label="C1 (label shuffle)", ls=":",  lw=1.0, alpha=0.7),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    kappas = args.kappas

    # Collect stats per κ
    data = {gen: {"x": [], "mean": [], "ci_lo": [], "ci_hi": []} for gen in list(GEN_STYLES) + list(CTRL_STYLES)}

    for kappa in kappas:
        tag = f"{kappa:.2f}"
        csv_path = Path(args.results_root) / f"kappa_{tag}" / args.csv_name
        if not csv_path.exists():
            csv_path = Path(args.results_root) / f"kappa_{tag}" / "kappa_curve.csv"
        if not csv_path.exists():
            csv_path = Path(args.results_root) / f"kappa_{tag}" / "baseline_coherence.csv"
        rows = load_csv(csv_path)
        if not rows:
            print(f"  [κ={kappa}] No CSV found at {csv_path}, skipping.")
            continue
        for gen in GEN_STYLES:
            asm = "native" if gen == "cof_cfg" else "P2"
            stat = get_stat(rows, gen, asm)
            if stat:
                data[gen]["x"].append(kappa)
                data[gen]["mean"].append(stat["mean"])
                data[gen]["ci_lo"].append(stat["ci_lo"])
                data[gen]["ci_hi"].append(stat["ci_hi"])
        for gen in CTRL_STYLES:
            stat = get_stat(rows, gen, "native")
            if stat:
                data[gen]["x"].append(kappa)
                data[gen]["mean"].append(stat["mean"])
                data[gen]["ci_lo"].append(stat["ci_lo"])
                data[gen]["ci_hi"].append(stat["ci_hi"])

    # Plot controls as horizontal-ish lines (C0, C1)
    for gen, style in CTRL_STYLES.items():
        d = data[gen]
        if not d["x"]:
            continue
        ax.plot(d["x"], d["mean"], **{k: v for k, v in style.items() if k not in ("alpha",)},
                alpha=style.get("alpha", 1.0))

    # Plot generators
    for gen, style in GEN_STYLES.items():
        d = data[gen]
        if not d["x"]:
            continue
        x = np.array(d["x"])
        m = np.array(d["mean"])
        lo = np.clip(d["ci_lo"], 0, None)
        hi = np.array(d["ci_hi"])
        ax.fill_between(x, lo, hi, alpha=0.15, color=style["color"])
        ax.plot(x, m, **{k: v for k, v in style.items() if k not in ("zorder",)},
                zorder=style.get("zorder", 3), markersize=5)

    # ── Effective-κ overlay for real datasets ────────────────────────────────
    eff_k_path = Path(args.results_root) / "effective_kappa.csv"
    if eff_k_path.exists():
        eff_data = list(csv.DictReader(open(eff_k_path)))
        eff_colors = {"AMLSim": "#2ca02c", "Sparkov": "#8c564b"}
        for row in eff_data:
            keff = float(row["keff"])
            name = row["dataset"]
            color = eff_colors.get(name, "green")
            ax.axvline(keff, color=color, ls=":", lw=1.2, alpha=0.8,
                       label=f"{name} (κ_eff≈{keff:.2f})")

    ax.set_xlabel("κ  (coupling strength)", fontsize=11)
    ax.set_ylabel("Coherence gap  (↓ better)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(bottom=0)
    ax.grid(axis="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")
    plt.close()


if __name__ == "__main__":
    main()
