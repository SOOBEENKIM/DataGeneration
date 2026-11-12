#!/bin/bash
# ibm_episode_state_v1 stage 2: generate the episode arm for the 2013 dev context (both fits in parallel) and evaluate it
# next to the adopted model and RTF + table. $1 = "cpu" (default; other users' GPU jobs are never touched) or two GPU ids.
set -u
W=/home/ssd-990/soobeenkim/DataGeneration/workspace/research-argn-state-first
cd /home/ssd-990/soobeenkim/DataGeneration || exit 1
LOG="$W/logs/episode_all.log"; D0="${1:-cpu}"; D1="${2:-cpu}"
dev() { if [ "$1" = "cpu" ]; then echo ""; else echo "$1"; fi; }
echo "START devices $D0 $D1 $(date)" >> "$LOG"
CUDA_VISIBLE_DEVICES="$(dev "$D0")" ./run python "$W/scripts/run_ibm_episode_state.py" generate --fit 20261108 > "$W/logs/episode_gen_20261108.log" 2>&1 &
CUDA_VISIBLE_DEVICES="$(dev "$D1")" ./run python "$W/scripts/run_ibm_episode_state.py" generate --fit 20261109 > "$W/logs/episode_gen_20261109.log" 2>&1 &
wait
echo "GENERATED $(date)" >> "$LOG"
./run python "$W/scripts/run_fraud_structure.py" --dataset ibm_episode > "$W/logs/episode_fs.log" 2>&1
./run python "$W/scripts/run_rare_state_persistence.py" --dataset ibm_episode > "$W/logs/episode_st.log" 2>&1
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_episode --detector stable > "$W/logs/episode_u_stable.log" 2>&1
./run python "$W/scripts/run_fraud_utility.py" --dataset ibm_episode --detector logistic > "$W/logs/episode_u_lg.log" 2>&1
echo "EPISODE_DONE $(date)" >> "$LOG"
