#!/bin/bash
# ibm_normal_rows_v1 amendment 4: replicate draws (utility) -> structure and stage 1 for ALL4 (draw 0). CPU only, thread-limited.
set -u
W=/home/ssd-990/soobeenkim/DataGeneration/workspace/research-argn-state-first
cd /home/ssd-990/soobeenkim/DataGeneration || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
LOG="$W/logs/normal4_all.log"; echo "START $(date)" >> "$LOG"
./run python "$W/scripts/sim_normal_rows_rep.py" > "$W/logs/normal4_rep.log" 2>&1 || { echo "REP_FAILED $(date)" >> "$LOG"; exit 1; }
echo "REPLICATED $(date)" >> "$LOG"
./run python "$W/scripts/run_fraud_structure.py" --dataset ibm_normal4 > "$W/logs/normal4_fs.log" 2>&1
./run python "$W/scripts/run_rare_state_persistence.py" --dataset ibm_normal4 > "$W/logs/normal4_st.log" 2>&1
echo "NORMAL4_DONE $(date)" >> "$LOG"
