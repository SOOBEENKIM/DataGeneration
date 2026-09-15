#!/bin/bash
# Run cof_cfg with guidance=2.0 at κ=0.30, 0.50, 0.70
# κ=0.00 and κ=1.00 already done in guidance2_sweep.csv

set -e
PYTHON="${COFSEQ_PYTHON:-python3}"
mkdir -p logs

for KAPPA in 0.30 0.50 0.70; do
    DATA_DIR="data/kappa_${KAPPA}/sequences"
    OUT="results/kappa_${KAPPA}/guidance2_sweep.csv"
    LOG="logs/guidance2_kappa${KAPPA/./}.log"

    echo "================================================================"
    echo "[κ=${KAPPA}] cof_cfg guidance=2.0, 5 seeds → ${LOG}"
    echo "================================================================"

    $PYTHON -u scripts/baseline_coherence_harness.py \
        --generators cof_cfg \
        --data_root "${DATA_DIR}" \
        --out "${OUT}" \
        --cof_cfg_lam 0.0 \
        --cof_cfg_seeds 5 \
        --cof_cfg_guidance 2.0 \
        2>&1 | tee "$LOG"

    echo "[κ=${KAPPA}] Done → ${OUT}"
done

echo ""
echo "=== guidance=2.0 COMPLETE for κ=0.30, 0.50, 0.70 ==="
python3 -c "
import pandas as pd, numpy as np

print('=== guidance=2.0 full κ-curve (cof_cfg only) ===')
kappas = [0.00, 0.30, 0.50, 0.70, 1.00]
for k in kappas:
    tag = f'{k:.2f}'
    try:
        df = pd.read_csv(f'results/kappa_{tag}/guidance2_sweep.csv')
        cof = df[df.generator=='cof_cfg']
        if len(cof):
            print(f'  κ={k:.2f}: mean={cof.coh_gap_mean.mean():.4f} seeds={cof.coh_gap_mean.values.round(4)}')
    except FileNotFoundError:
        print(f'  κ={k:.2f}: (missing)')
"
