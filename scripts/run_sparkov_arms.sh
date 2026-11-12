#!/bin/bash
# sparkov_transfer_v1 amendment 1: the four part arms for one fit on one GPU.  usage: run_sparkov_arms.sh <gpu> <fit>
set -u
W=/home/ssd-990/soobeenkim/DataGeneration/workspace/research-argn-state-first
cd /home/ssd-990/soobeenkim/DataGeneration || exit 1
export CUDA_VISIBLE_DEVICES="$1"
FIT="$2"
LOG="$W/logs/spk_arms2_${FIT}.log"
./run python "$W/scripts/run_sparkov_transfer.py" generate --fit "$FIT" --arm sf_gap --parts gap >> "$LOG" 2>&1
./run python "$W/scripts/run_sparkov_transfer.py" generate --fit "$FIT" --arm sf_attr --parts attr >> "$LOG" 2>&1
./run python "$W/scripts/run_sparkov_transfer.py" generate --fit "$FIT" --arm sf_full --parts gap attr >> "$LOG" 2>&1
./run python "$W/scripts/run_sparkov_transfer.py" generate --fit "$FIT" --arm sf_attr_all --parts attr_all >> "$LOG" 2>&1
echo ARMS_DONE >> "$LOG"
