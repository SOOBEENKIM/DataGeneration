#!/bin/bash
# Resume: κ=0.70 (full re-run) + κ=1.00 (full run)
set -e

PYTHON="${COFSEQ_PYTHON:-python3}"
LAM=0.0
GUIDANCE=3.0
mkdir -p logs

for KAPPA in 0.70 1.00; do
    DATA_DIR="data/kappa_${KAPPA}/sequences"
    OUT_DIR="results/kappa_${KAPPA}"
    OUT_CSV="${OUT_DIR}/kappa_curve.csv"
    LOG="logs/kappa${KAPPA/./}_curve.log"

    mkdir -p "$OUT_DIR"
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

    echo "[κ=${KAPPA}] Done."
done

echo "ALL DONE."
