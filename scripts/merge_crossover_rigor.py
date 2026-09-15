"""
Merge κ=0.5 crossover rigor (5-seed baselines) into kappa_curve_g2.csv
and regenerate the κ-curve figure.

Run after scripts/run_kappa05_baseline_multiseed.sh completes.

Input:
  results/kappa_0.50/kappa_curve_g2.csv  (5-seed CoF + 1-seed baselines)
  results/kappa_0.50/baseline_5seed.csv  (row_shuffle + ctgan + tvae ×5, 100 epochs — consistent protocol)
Output:
  results/kappa_0.50/kappa_curve_g2_5seed.csv  (5-seed CoF + 5-seed baselines)
  figs/kappa_curve_g2_5seed.png
"""

from pathlib import Path
import pandas as pd
import subprocess, sys

ROOT = Path(__file__).parent.parent

# ── Merge κ=0.5 ────────────────────────────────────────────────────────────────
g2 = pd.read_csv(ROOT / "results/kappa_0.50/kappa_curve_g2.csv")
# All baselines ×5 seeds, 100 epochs — same protocol throughout the κ-curve
b5 = pd.read_csv(ROOT / "results/kappa_0.50/baseline_5seed.csv")

# Keep C0/C1/cof_cfg from g2; replace row_shuffle/ctgan/tvae with 5-seed versions
keep = g2[g2.generator.isin(["C0_real_split", "C1_label_shuf", "cof_cfg"])]
merged = pd.concat([keep, b5], ignore_index=True)
merged = merged.drop_duplicates(subset=["generator", "seed"], keep="last")

out = ROOT / "results/kappa_0.50/kappa_curve_g2_5seed.csv"
merged.to_csv(out, index=False)
print(f"Saved {out}: {len(merged)} rows")
for gen in ["C0_real_split","C1_label_shuf","cof_cfg","row_shuffle","ctgan","tvae"]:
    sub = merged[merged.generator == gen]
    if len(sub):
        print(f"  {gen}: n={len(sub)}, mean={sub.coh_gap_mean.mean():.4f}")

# ── Also copy to kappa_curve_g2.csv for κ=0.5 ─────────────────────────────────
merged.to_csv(ROOT / "results/kappa_0.50/kappa_curve_g2.csv", index=False)
print("\nUpdated results/kappa_0.50/kappa_curve_g2.csv with 5-seed baselines")

# ── Regenerate figure ──────────────────────────────────────────────────────────
PYTHON = sys.executable
cmd = [PYTHON, "scripts/plot_kappa_curve.py",
       "--kappas", "0.0", "0.3", "0.5", "0.7", "1.0",
       "--csv_name", "kappa_curve_g2.csv",
       "--out", "figs/kappa_curve_g2.png"]
print(f"\nRegenerating: {' '.join(cmd)}")
subprocess.run(cmd, cwd=ROOT, check=True)
print("Done.")
