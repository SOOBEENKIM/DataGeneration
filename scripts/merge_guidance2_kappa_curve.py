"""
Merge guidance=2.0 cof_cfg results into kappa_curve CSVs.

For each κ:
  1. Load kappa_curve.csv (g=3.0 baseline/C0/C1/row_shuffle/ctgan/tvae rows + old cof_cfg)
  2. Load guidance2_sweep.csv (g=2.0 cof_cfg rows)
  3. Replace old cof_cfg rows with new g=2.0 rows
  4. Save as kappa_curve_g2.csv

Also regenerates the κ-curve figure with g=2.0 data.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
KAPPAS = [0.00, 0.30, 0.50, 0.70, 1.00]

print("=== Merging guidance=2.0 cof_cfg into κ-curve CSVs ===\n")

for k in KAPPAS:
    tag = f"{k:.2f}"
    main_path = ROOT / "results" / f"kappa_{tag}" / "kappa_curve.csv"
    g2_path   = ROOT / "results" / f"kappa_{tag}" / "guidance2_sweep.csv"
    out_path  = ROOT / "results" / f"kappa_{tag}" / "kappa_curve_g2.csv"

    if not main_path.exists():
        print(f"κ={tag}: SKIP (no kappa_curve.csv)")
        continue
    if not g2_path.exists():
        print(f"κ={tag}: SKIP (no guidance2_sweep.csv)")
        continue

    main = pd.read_csv(main_path)
    g2   = pd.read_csv(g2_path)

    # Keep non-cof_cfg rows from main; use g2 for cof_cfg
    non_cof = main[main.generator != "cof_cfg"].copy()
    cof_g2  = g2[g2.generator == "cof_cfg"].copy()

    merged = pd.concat([non_cof, cof_g2], ignore_index=True)
    merged.to_csv(out_path, index=False)

    # Summary
    cof_g3 = main[main.generator == "cof_cfg"]
    print(f"κ={tag}:")
    print(f"  cof_cfg g=3.0 mean: {cof_g3.coh_gap_mean.mean():.4f} ({len(cof_g3)} seeds)")
    print(f"  cof_cfg g=2.0 mean: {cof_g2.coh_gap_mean.mean():.4f} ({len(cof_g2)} seeds)")
    for gen in ["row_shuffle", "ctgan", "tvae", "C0_real_split", "C1_label_shuf"]:
        sub = merged[merged.generator == gen]
        if len(sub):
            print(f"  {gen}: {sub.coh_gap_mean.mean():.4f} ({len(sub)} seeds)")
    print(f"  Saved → {out_path.name}\n")

print("=== All merges complete ===")
print("\nNow regenerate figure with --results_root-specific paths...")
print("Run: python scripts/plot_kappa_curve.py --kappas 0.0 0.3 0.5 0.7 1.0 --out figs/kappa_curve_g2.png --use_g2")
