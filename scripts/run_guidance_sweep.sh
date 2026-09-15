#!/bin/bash
# guidance=2.0 sweep at κ=0.00 and κ=1.00
# Goal: check if lower guidance removes spurious coupling at κ=0 without hurting κ=1.0
# Runs cof_cfg only (5 seeds); does NOT rerun ctgan/tvae/C0/C1

set -e
PYTHON="${COFSEQ_PYTHON:-python3}"
mkdir -p logs results/kappa_0.00 results/kappa_1.00

for KAPPA in 0.00 1.00; do
    LOG="logs/guidance_sweep_kappa${KAPPA/./}.log"
    OUT="results/kappa_${KAPPA}/guidance2_sweep.csv"
    DATA="data/kappa_${KAPPA}/sequences"

    echo "================================================================"
    echo "[κ=${KAPPA}] cof_cfg guidance=2.0, 5 seeds → ${LOG}"
    echo "================================================================"

    $PYTHON -u scripts/baseline_coherence_harness.py \
        --generators cof_cfg \
        --data_root "${DATA}" \
        --out "${OUT}" \
        --cof_cfg_lam 0.0 \
        --cof_cfg_seeds 5 \
        --cof_cfg_guidance 2.0 \
        2>&1 | tee "$LOG"

    echo "[κ=${KAPPA}] done → ${OUT}"
done

echo ""
echo "=== guidance=2.0 sweep COMPLETE ==="
echo "Compare:"
python3 -c "
import pandas as pd, numpy as np
for k in ['0.00','1.00']:
    try:
        g3 = pd.read_csv(f'results/kappa_{k}/kappa_curve.csv')
        g2 = pd.read_csv(f'results/kappa_{k}/guidance2_sweep.csv')
        g3c = g3[g3.generator=='cof_cfg']
        g2c = g2[g2.generator=='cof_cfg']
        print(f'κ={k}: guidance=3.0 mean={g3c.coh_gap_mean.mean():.4f} | guidance=2.0 mean={g2c.coh_gap_mean.mean():.4f}')
    except Exception as e:
        print(f'κ={k}: error reading results — {e}')
"
