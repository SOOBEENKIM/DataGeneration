#!/bin/bash
# Resume κ-curve from where it was killed.
# κ=0.30: add ctgan+tvae only (cof_cfg+row_shuffle already in CSV)
# κ=0.50, 0.70, 1.00: full run
set -e

PYTHON="${COFSEQ_PYTHON:-python3}"
LAM=0.0
GUIDANCE=3.0
mkdir -p logs

# ── κ=0.30: ctgan + tvae only (append to existing CSV) ───────────────────────
echo "================================================================"
echo "[κ=0.30] Adding ctgan+tvae (cof_cfg+row_shuffle already done)"
echo "================================================================"
$PYTHON -u scripts/baseline_coherence_harness.py \
    --generators ctgan,tvae \
    --data_root data/kappa_0.30/sequences \
    --out results/kappa_0.30/kappa_ctgan_tvae.csv \
    --cof_cfg_lam $LAM \
    --cof_cfg_seeds 5 \
    --cof_cfg_guidance $GUIDANCE \
    2>&1 | tee logs/kappa030_ctgan_tvae.log
echo "[κ=0.30] ctgan+tvae done"

# Merge: append ctgan+tvae rows to existing kappa_curve.csv
python3 -c "
import pandas as pd
a = pd.read_csv('results/kappa_0.30/kappa_curve.csv')
b = pd.read_csv('results/kappa_0.30/kappa_ctgan_tvae.csv')
pd.concat([a, b], ignore_index=True).to_csv('results/kappa_0.30/kappa_curve.csv', index=False)
print(f'Merged kappa_0.30: {len(a)+len(b)} rows total')
"

# ── κ=0.50, 0.70, 1.00: full run ─────────────────────────────────────────────
for KAPPA in 0.50 0.70 1.00; do
    DATA_DIR="data/kappa_${KAPPA}/sequences"
    OUT_DIR="results/kappa_${KAPPA}"
    OUT_CSV="${OUT_DIR}/kappa_curve.csv"
    LOG="logs/kappa${KAPPA/./}_curve.log"

    mkdir -p "$OUT_DIR"
    echo ""
    echo "================================================================"
    echo "[κ=${KAPPA}] All generators → ${LOG}"
    echo "================================================================"

    $PYTHON -u scripts/baseline_coherence_harness.py \
        --generators C0,C1,cof_cfg,row_shuffle,ctgan,tvae \
        --data_root "${DATA_DIR}" \
        --out "${OUT_CSV}" \
        --cof_cfg_lam $LAM \
        --cof_cfg_seeds 5 \
        --cof_cfg_guidance $GUIDANCE \
        2>&1 | tee "$LOG"

    echo "[κ=${KAPPA}] Done → ${OUT_CSV}"
done

echo ""
echo "================================================================"
echo "RESUME COMPLETE."
echo "  κ=0.00: results/kappa_0.00/kappa_curve.csv (already done)"
echo "  κ=0.30: results/kappa_0.30/kappa_curve.csv (merged)"
echo "  κ=0.50: results/kappa_0.50/kappa_curve.csv"
echo "  κ=0.70: results/kappa_0.70/kappa_curve.csv"
echo "  κ=1.00: results/kappa_1.00/kappa_curve.csv"
echo "================================================================"
