#!/bin/bash
# ibm_sealed_test_v1: wait until two GPUs are free (other users' jobs are never touched), then run the one-shot pipeline:
# test windows -> ARGN ladder + RTF arms for the test context -> structure / stage-1 / utility on the sealed test.
set -u
W=/home/ssd-990/soobeenkim/DataGeneration/workspace/research-argn-state-first
cd /home/ssd-990/soobeenkim/DataGeneration || exit 1
LOG="$W/logs/sealed_test.log"
RTF=environments/realtabformer/bin/python
if [ -f "$W/logs/sealed_test.started" ]; then echo "already started once; refusing to run again" >> "$LOG"; exit 1; fi
pick() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 4000 {print $1}' | head -n 2 | tr '\n' ' '; }
echo "WAITING $(date)" >> "$LOG"
while true; do G=($(pick)); if [ "${#G[@]}" -ge 2 ]; then sleep 60; G2=($(pick)); if [ "${#G2[@]}" -ge 2 ]; then G=("${G2[@]}"); break; fi; fi; sleep 120; done
date > "$W/logs/sealed_test.started"
echo "START gpus ${G[*]} $(date)" >> "$LOG"
$RTF "$W/scripts/run_realtabformer_ibm.py" prepare_test >> "$LOG" 2>&1
one() {  # $1 gpu, $2 fit
  CUDA_VISIBLE_DEVICES="$1" ./run python "$W/scripts/run_ibm_sealed_test.py" --fit "$2" >> "$W/logs/sealed_argn_$2.log" 2>&1
  CUDA_VISIBLE_DEVICES="$1" $RTF "$W/scripts/run_realtabformer_ibm.py" generate --fit "$2" --arm rtf_table --split test >> "$W/logs/sealed_rtf_$2.log" 2>&1
  CUDA_VISIBLE_DEVICES="$1" $RTF "$W/scripts/run_realtabformer_ibm.py" generate --fit "$2" --arm rtf_native --split test >> "$W/logs/sealed_rtf_$2.log" 2>&1
}
one "${G[0]}" 20261108 &
one "${G[1]}" 20261109 &
wait
echo "GENERATED $(date)" >> "$LOG"
./run python "$W/scripts/run_fraud_structure.py" --dataset ibm_test > "$W/logs/sealed_fs.log" 2>&1
./run python "$W/scripts/run_rare_state_persistence.py" --dataset ibm_test > "$W/logs/sealed_st.log" 2>&1
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_test --detector stable > "$W/logs/sealed_u_stable.log" 2>&1
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_test --detector stable --features no_merchant > "$W/logs/sealed_u_nm.log" 2>&1
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_test --detector stable --features no_channel > "$W/logs/sealed_u_nc.log" 2>&1
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_test --detector logistic > "$W/logs/sealed_u_lg.log" 2>&1
echo "SEALED_DONE $(date)" >> "$LOG"
