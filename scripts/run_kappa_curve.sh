#!/bin/bash
# κ-curve: run CFG-CoF + row-baselines on each κ dataset
# Usage: bash scripts/run_kappa_curve.sh [--kappas "0.0 0.3 0.5 0.7 1.0"]
#
# For each κ:
#   - C0, C1 (controls)
#   - CFG-CoF (3 seeds)
#   - row_shuffle, ctgan, tvae (1 seed each — fast)
# Output: results/kappa_<k>/baseline_coherence.csv

PYTHON="${COFSEQ_PYTHON:-python3}"
KAPPAS="${KAPPAS:-0.0 0.3 0.5 0.7 1.0}"
SEEDS="${CFG_SEEDS:-3}"
LAM="${LAM:-2.0}"

for KAPPA in $KAPPAS; do
    TAG=$(printf "%.2f" $KAPPA)
    DATA_DIR="data/kappa_${TAG}/sequences"
    OUT_DIR="results/kappa_${TAG}"
    LOG="logs/kappa${TAG/./}_harness.log"

    # Skip if already complete
    if [ -f "${OUT_DIR}/baseline_coherence.csv" ]; then
        N=$(tail -n +2 "${OUT_DIR}/baseline_coherence.csv" | wc -l)
        if [ "$N" -ge 7 ]; then
            echo "[κ=${TAG}] Already complete (${N} rows). Skipping."
            continue
        fi
    fi

    mkdir -p "$OUT_DIR"
    echo "[κ=${TAG}] Starting harness → ${LOG}"

    $PYTHON -u scripts/baseline_coherence_harness.py \
        --generators C0,C1,cof_cfg,row_shuffle,ctgan,tvae \
        --data_root "${DATA_DIR}" \
        --out "${OUT_DIR}/baseline_coherence.csv" \
        --cof_cfg_lam "$LAM" \
        --cof_cfg_seeds "$SEEDS" \
        > "$LOG" 2>&1

    echo "[κ=${TAG}] Done."
done

echo ""
echo "=== κ-curve complete. Run analyze on each: ==="
for KAPPA in $KAPPAS; do
    TAG=$(printf "%.2f" $KAPPA)
    echo "  κ=${TAG}: python scripts/analyze_baseline_coherence.py --csv results/kappa_${TAG}/baseline_coherence.csv"
done
