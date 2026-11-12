#!/bin/bash
# ibm_normal_rows_v1: CPU post-processing arms on the adopted model's dev generations, then the standard evaluations. No GPU.
set -u
W=/home/ssd-990/soobeenkim/DataGeneration/workspace/research-argn-state-first
cd /home/ssd-990/soobeenkim/DataGeneration || exit 1
export CUDA_VISIBLE_DEVICES=
LOG="$W/logs/normal_all.log"; echo "START $(date)" >> "$LOG"
./run python "$W/scripts/sim_normal_rows.py" apply > "$W/logs/normal_apply.log" 2>&1 || { echo "APPLY_FAILED $(date)" >> "$LOG"; exit 1; }
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_normal --detector stable > "$W/logs/normal_u_stable.log" 2>&1
./run python "$W/scripts/sim_normal_rows.py" measure > "$W/logs/normal_measure.log" 2>&1
./run python "$W/scripts/run_fraud_structure.py" --dataset ibm_normal > "$W/logs/normal_fs.log" 2>&1
./run python "$W/scripts/run_rare_state_persistence.py" --dataset ibm_normal > "$W/logs/normal_st.log" 2>&1
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_normal --detector logistic > "$W/logs/normal_u_lg.log" 2>&1
echo "NORMAL_DONE $(date)" >> "$LOG"
