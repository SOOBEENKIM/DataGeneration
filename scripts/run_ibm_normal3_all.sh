#!/bin/bash
# ibm_normal_rows_v1 amendment 3: fit3 -> apply 3 -> utility -> measure 3 -> structure -> stage 1 -> logistic utility. CPU only, thread-limited.
set -u
W=/home/ssd-990/soobeenkim/DataGeneration/workspace/research-argn-state-first
cd /home/ssd-990/soobeenkim/DataGeneration || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
LOG="$W/logs/normal3_all.log"; echo "START $(date)" >> "$LOG"
./run python "$W/scripts/sim_normal_rows_v2.py" fit3 > "$W/logs/normal3_fit.log" 2>&1 || { echo "FIT_FAILED $(date)" >> "$LOG"; exit 1; }
./run python "$W/scripts/sim_normal_rows_v2.py" apply 3 > "$W/logs/normal3_apply.log" 2>&1 || { echo "APPLY_FAILED $(date)" >> "$LOG"; exit 1; }
echo "APPLIED $(date)" >> "$LOG"
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_normal3 --detector stable > "$W/logs/normal3_u_stable.log" 2>&1
./run python "$W/scripts/sim_normal_rows_v2.py" measure 3 > "$W/logs/normal3_measure.log" 2>&1
echo "MEASURED $(date)" >> "$LOG"
./run python "$W/scripts/run_fraud_structure.py" --dataset ibm_normal3 > "$W/logs/normal3_fs.log" 2>&1
./run python "$W/scripts/run_rare_state_persistence.py" --dataset ibm_normal3 > "$W/logs/normal3_st.log" 2>&1
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_normal3 --detector logistic > "$W/logs/normal3_u_lg.log" 2>&1
echo "NORMAL3_DONE $(date)" >> "$LOG"
