#!/bin/bash
# Run row_shuffle, ctgan, tvae seeds 1-5 at κ=0.50
# Existing kappa_curve.csv has seed=1 only → this adds seeds 2-5 (re-runs seed 1 too)
# Goal: CI bands for crossover rigor (Tier 1)

set -e
PYTHON="${COFSEQ_PYTHON:-python3}"
mkdir -p logs results/kappa_0.50

OUT="results/kappa_0.50/baseline_5seed.csv"
LOG="logs/kappa050_baseline_5seed.log"

echo "================================================================"
echo "[κ=0.50] row_shuffle + ctgan + tvae, 5 seeds → ${LOG}"
echo "================================================================"

$PYTHON -u scripts/baseline_coherence_harness.py \
    --generators row_shuffle,ctgan,tvae \
    --data_root data/kappa_0.50/sequences \
    --out "$OUT" \
    --seeds 5 \
    2>&1 | tee "$LOG"

# Merge with existing kappa_curve.csv (replace single-seed baselines with 5-seed)
python3 -c "
import pandas as pd

main = pd.read_csv('results/kappa_0.50/kappa_curve.csv')
extra = pd.read_csv('results/kappa_0.50/baseline_5seed.csv')

# Keep C0/C1/cof_cfg from main; use extra for row_shuffle/ctgan/tvae
keep_from_main = main[main.generator.isin(['C0_real_split','C1_label_shuf','cof_cfg'])]
merged = pd.concat([keep_from_main, extra], ignore_index=True)
merged = merged.drop_duplicates(subset=['generator','seed'], keep='last')
merged.to_csv('results/kappa_0.50/kappa_curve_5seed.csv', index=False)
print(f'Saved kappa_curve_5seed.csv: {len(merged)} rows')
for gen in ['C0_real_split','C1_label_shuf','cof_cfg','row_shuffle','ctgan','tvae']:
    sub = merged[merged.generator==gen]
    if len(sub) > 0:
        print(f'  {gen}: {len(sub)} rows, mean={sub.coh_gap_mean.mean():.4f}')
"
echo "[κ=0.50] Done."
