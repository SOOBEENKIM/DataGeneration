#!/bin/bash
# ibm_normal_impl_v1: 8 independent dev generations for arms B and R (CPU; other users' GPU jobs are never touched), the amount decoding
# stage (BA, RA), then the standard evaluations. One workspace per fit, so the two arms of a fit run one after the other.
set -u
W=/home/ssd-990/soobeenkim/DataGeneration/workspace/research-argn-state-first
cd /home/ssd-990/soobeenkim/DataGeneration || exit 1
export CUDA_VISIBLE_DEVICES=
LOG="$W/logs/impl_all.log"; echo "START $(date)" >> "$LOG"
while tmux has-session -t impltest 2>/dev/null; do sleep 20; done      # the first check run (arm R, fit 20261108) uses that fit's workspace
./run python "$W/scripts/run_ibm_normal_impl.py" generate --fit 20261108 > "$W/logs/impl_gen_20261108.log" 2>&1 &
./run python "$W/scripts/run_ibm_normal_impl.py" generate --fit 20261109 > "$W/logs/impl_gen_20261109.log" 2>&1 &
wait
echo "GENERATED $(date)" >> "$LOG"
export OMP_NUM_THREADS=8
./run python "$W/scripts/run_ibm_normal_impl.py" decode > "$W/logs/impl_decode.log" 2>&1 || { echo "DECODE_FAILED $(date)" >> "$LOG"; exit 1; }
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_impl --detector stable > "$W/logs/impl_u_stable.log" 2>&1
echo "UTILITY $(date)" >> "$LOG"
./run python "$W/scripts/run_ibm_normal_impl.py" measure > "$W/logs/impl_measure.log" 2>&1
./run python "$W/scripts/run_fraud_structure.py" --dataset ibm_impl > "$W/logs/impl_fs.log" 2>&1
./run python "$W/scripts/run_rare_state_persistence.py" --dataset ibm_impl > "$W/logs/impl_st.log" 2>&1
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_impl --detector logistic > "$W/logs/impl_u_lg.log" 2>&1
echo "IMPL_DONE $(date)" >> "$LOG"
