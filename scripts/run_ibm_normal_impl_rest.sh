#!/bin/bash
# ibm_normal_impl_v1: decode stage onward (the first decode stopped on an over-strict format assertion; generations were complete).
set -u
W=/home/ssd-990/soobeenkim/DataGeneration/workspace/research-argn-state-first
cd /home/ssd-990/soobeenkim/DataGeneration || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
LOG="$W/logs/impl_all.log"; echo "RESTART $(date)" >> "$LOG"
./run python "$W/scripts/run_ibm_normal_impl.py" decode > "$W/logs/impl_decode.log" 2>&1 || { echo "DECODE_FAILED_AGAIN $(date)" >> "$LOG"; exit 1; }
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_impl --detector stable > "$W/logs/impl_u_stable.log" 2>&1
echo "UTILITY $(date)" >> "$LOG"
./run python "$W/scripts/run_ibm_normal_impl.py" measure > "$W/logs/impl_measure.log" 2>&1
./run python "$W/scripts/run_fraud_structure.py" --dataset ibm_impl > "$W/logs/impl_fs.log" 2>&1
./run python "$W/scripts/run_rare_state_persistence.py" --dataset ibm_impl > "$W/logs/impl_st.log" 2>&1
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_impl --detector logistic > "$W/logs/impl_u_lg.log" 2>&1
echo "IMPL_DONE $(date)" >> "$LOG"
