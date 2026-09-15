#!/bin/bash
# Scale-up runner: 10-seed validation on κ=1.0, then full κ-curve
# Step 1: 10-seed κ=1.0 CI check
# Step 2: κ-curve (5 κ values, 5 seeds each, all generators)
set -e

PYTHON="${COFSEQ_PYTHON:-python3}"
LAM=0.0
GUIDANCE=3.0
mkdir -p logs results/kappa_{0.00,0.30,0.50,0.70,1.00}

# ── Step 1: 10-seed κ=1.0 validation ─────────────────────────────────────────
echo "================================================================"
echo "[Step 1] 10-seed κ=1.0 validation (lam=${LAM}, guidance=${GUIDANCE})"
echo "================================================================"
$PYTHON -u scripts/baseline_coherence_harness.py \
    --generators cof_cfg \
    --data_root data/kappa_1.00/sequences \
    --out results/kappa_1.00/validation_10seed.csv \
    --cof_cfg_lam $LAM \
    --cof_cfg_seeds 10 \
    --cof_cfg_guidance $GUIDANCE \
    2>&1 | tee logs/kappa100_10seed.log

echo ""
echo "[Step 1 done] Results in results/kappa_1.00/validation_10seed.csv"
echo ""

# ── Step 2: κ-curve (all 5 kappa values, 5 seeds, all generators) ────────────
echo "================================================================"
echo "[Step 2] κ-curve sweep (5 κ values × 5 seeds × all generators)"
echo "================================================================"

for KAPPA in 0.00 0.30 0.50 0.70 1.00; do
    DATA_DIR="data/kappa_${KAPPA}/sequences"
    OUT_DIR="results/kappa_${KAPPA}"
    OUT_CSV="${OUT_DIR}/kappa_curve.csv"
    LOG="logs/kappa${KAPPA/./}_curve.log"

    mkdir -p "$OUT_DIR"
    echo ""
    echo "[κ=${KAPPA}] → ${LOG}"

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
echo "ALL DONE. Summary:"
echo "  10-seed CI:   results/kappa_1.00/validation_10seed.csv"
for KAPPA in 0.00 0.30 0.50 0.70 1.00; do
    echo "  κ=${KAPPA}:    results/kappa_${KAPPA}/kappa_curve.csv"
done
echo "================================================================"
