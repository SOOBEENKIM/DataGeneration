#!/bin/bash
# ibm_normal_rows_v1: the evaluations after `apply` and the stable utility (the first measure run stalled with all cores; rerun with a thread limit).
set -u
W=/home/ssd-990/soobeenkim/DataGeneration/workspace/research-argn-state-first
cd /home/ssd-990/soobeenkim/DataGeneration || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
LOG="$W/logs/normal_all.log"; echo "RESTART $(date)" >> "$LOG"
./run python "$W/scripts/sim_normal_rows.py" measure > "$W/logs/normal_measure.log" 2>&1
echo "MEASURED $(date)" >> "$LOG"
./run python "$W/scripts/run_fraud_structure.py" --dataset ibm_normal > "$W/logs/normal_fs.log" 2>&1
./run python "$W/scripts/run_rare_state_persistence.py" --dataset ibm_normal > "$W/logs/normal_st.log" 2>&1
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_normal --detector logistic > "$W/logs/normal_u_lg.log" 2>&1
echo "NORMAL_DONE $(date)" >> "$LOG"
