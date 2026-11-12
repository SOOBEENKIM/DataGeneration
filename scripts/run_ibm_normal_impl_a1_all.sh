#!/bin/bash
# ibm_normal_impl_v1 amendment 1 (seed extension): 16 more independent dev generations for arms B and R (CPU, as the first 8), the amount
# decoding stage for them (BA, RA), then the evaluations over all 24 per arm into docs/ibm_normal_impl_v1/a1/. The first 8's outputs stay.
set -u
W=/home/ssd-990/soobeenkim/DataGeneration/workspace/research-argn-state-first
cd /home/ssd-990/soobeenkim/DataGeneration || exit 1
export CUDA_VISIBLE_DEVICES=
LOG="$W/logs/impl_a1_all.log"; echo "START $(date)" >> "$LOG"
./run python "$W/scripts/run_ibm_normal_impl.py" generate --set a1 --fit 20261108 > "$W/logs/impl_a1_gen_20261108.log" 2>&1 &
./run python "$W/scripts/run_ibm_normal_impl.py" generate --set a1 --fit 20261109 > "$W/logs/impl_a1_gen_20261109.log" 2>&1 &
wait
N=$(ls "$W"/artifacts/ibm_normal_impl_v1/runs/[BR]_2026110[89]/generated_validation_20261[23]0[0-7].parquet 2>/dev/null | wc -l)
echo "GENERATED $N/32 $(date)" >> "$LOG"
[ "$N" -eq 32 ] || { echo "GENERATE_INCOMPLETE $(date)" >> "$LOG"; exit 1; }
export OMP_NUM_THREADS=8
./run python "$W/scripts/run_ibm_normal_impl.py" decode --set a1 > "$W/logs/impl_a1_decode.log" 2>&1 || { echo "DECODE_FAILED $(date)" >> "$LOG"; exit 1; }
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_impl_a1 --detector stable > "$W/logs/impl_a1_u_stable.log" 2>&1
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_impl_a1 --detector logistic > "$W/logs/impl_a1_u_lg.log" 2>&1
./run python "$W/scripts/run_ibm_normal_impl.py" summary --set a1 > "$W/logs/impl_a1_summary.log" 2>&1
echo "UTILITY $(date)" >> "$LOG"
./run python "$W/scripts/run_ibm_normal_impl.py" measure --set a1 > "$W/logs/impl_a1_measure.log" 2>&1 &
./run python "$W/scripts/run_fraud_structure.py" --dataset ibm_impl_a1 > "$W/logs/impl_a1_fs.log" 2>&1 &
./run python "$W/scripts/run_rare_state_persistence.py" --dataset ibm_impl_a1 > "$W/logs/impl_a1_st.log" 2>&1 &
wait
echo "IMPL_A1_DONE $(date)" >> "$LOG"
