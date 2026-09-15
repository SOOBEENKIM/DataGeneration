# External validation v1 performance-corrective continuation launch plan

Status: **SUPERSEDED; DO NOT EXECUTE.** This historical three-wave plan allowed
CTGAN and TVAE heavy transforms to overlap and is preserved only for provenance.
Its authorization history remains append-only. No command in this document is
authorized under the memory-corrective source.

## Frozen scope

Only launch scheduling changed. The receiver total-variation definition,
seeds, attempts, conditioning, train-only bootstrap, validation selection
rule, thresholds, data, model configurations, and wall caps remain unchanged.

| Wave | Dataset | Model | Seed | Attempt | Slot | Whole-job cap |
|---|---|---|---:|---|---|---:|
| CPU | AMLSim | empirical_iid | 31001 | attempt_002 | CPU A | 7,200 s |
| CPU | Sparkov | empirical_iid | 32001 | attempt_002 | CPU B | 7,200 s |
| GPU 1 | AMLSim | ctgan_separate_class | 31001 | attempt_001 | physical GPU A | 10,800 s |
| GPU 1 | AMLSim | tvae_separate_class | 31001 | attempt_001 | physical GPU B | 10,800 s |
| GPU 1 | AMLSim | cof_seqgen_frozen_non_v3 | 31001 | attempt_001 | physical GPU C | 10,800 s |
| GPU 2 | Sparkov | ctgan_separate_class | 32001 | attempt_001 | physical GPU A | 10,800 s |
| GPU 2 | Sparkov | tvae_separate_class | 32001 | attempt_001 | physical GPU B | 10,800 s |
| GPU 2 | Sparkov | cof_seqgen_frozen_non_v3 | 32001 | attempt_001 | physical GPU C | 10,800 s |

The two CPU jobs run concurrently. Each GPU wave runs three jobs concurrently
on three distinct physical GPUs. A wave may start only after every job in the
preceding wave has a terminal `COMPLETE.json`, `INVALID.json`, or `FAILED.json`.
Each GPU worker sees exactly one physical GPU through `CUDA_VISIBLE_DEVICES`
and therefore always receives `--device cuda:0`. No runner performs GPU
inventory discovery.

The learned jobs retain their 7,200-second training caps inside the 10,800
second whole-job watchdog. The conservative three-wave wall-time upper bound
is 2 + 3 + 3 = 8 hours. No retry beyond the named attempts is authorized.

## SSH preamble

The user chooses three idle, distinct physical GPU identifiers in the SSH
shell. They may be indices or GPU UUIDs accepted by `CUDA_VISIBLE_DEVICES`.

```bash
export EXTV1_REPO=<REPO_ROOT>
export EXTV1_PY=<COFSEQ_PYTHON>
export EXTV1_EXPECTED_HEAD='<scheduled-head>'
export EXTV1_AUTH="$EXTV1_REPO/artifacts/external_validation_v1/wave_scheduled_continuation_authorization_history/authorization_${EXTV1_EXPECTED_HEAD}_attempt_001.json"
export EXTV1_GPU_A='<idle-physical-gpu-a>'
export EXTV1_GPU_B='<idle-physical-gpu-b>'
export EXTV1_GPU_C='<idle-physical-gpu-c>'
cd "$EXTV1_REPO"
test "$(git rev-parse HEAD)" = "$EXTV1_EXPECTED_HEAD"
test -f "$EXTV1_AUTH"
test -n "$EXTV1_GPU_A"
test -n "$EXTV1_GPU_B"
test -n "$EXTV1_GPU_C"
test "$EXTV1_GPU_A" != "$EXTV1_GPU_B"
test "$EXTV1_GPU_A" != "$EXTV1_GPU_C"
test "$EXTV1_GPU_B" != "$EXTV1_GPU_C"
mkdir -p artifacts/external_validation_v1/runner_logs

extv1_terminal() {
  test -f "$1/COMPLETE.json" || test -f "$1/INVALID.json" || test -f "$1/FAILED.json"
}
```

## Wave 0: two CPU jobs in parallel

All precondition checks must pass before either launch.

```bash
test ! -e artifacts/external_validation_v1/amlsim/empirical_iid/attempt_002
test ! -e artifacts/external_validation_v1/sparkov/empirical_iid/attempt_002
test ! -e artifacts/external_validation_v1/runner_logs/extv1_sched_amlsim_iid_attempt_002.log
test ! -e artifacts/external_validation_v1/runner_logs/extv1_sched_sparkov_iid_attempt_002.log
! tmux has-session -t extv1_sched_amlsim_iid 2>/dev/null
! tmux has-session -t extv1_sched_sparkov_iid 2>/dev/null

tmux new-session -d -s extv1_sched_amlsim_iid "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_sched_amlsim_iid_attempt_002.log 2>&1; exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset amlsim --model empirical_iid --device cpu'"
tmux new-session -d -s extv1_sched_sparkov_iid "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_sched_sparkov_iid_attempt_002.log 2>&1; exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset sparkov --model empirical_iid --device cpu'"
```

Mandatory CPU terminal barrier:

```bash
while ! extv1_terminal artifacts/external_validation_v1/amlsim/empirical_iid/attempt_002 || ! extv1_terminal artifacts/external_validation_v1/sparkov/empirical_iid/attempt_002; do sleep 30; done
```

## Wave 1: three AMLSim GPU jobs in parallel

All precondition checks must pass after the CPU barrier and before any wave-1
launch.

```bash
test ! -e artifacts/external_validation_v1/amlsim/ctgan_separate_class/attempt_001
test ! -e artifacts/external_validation_v1/amlsim/tvae_separate_class/attempt_001
test ! -e artifacts/external_validation_v1/amlsim/cof_seqgen_frozen_non_v3/attempt_001
test ! -e artifacts/external_validation_v1/runner_logs/extv1_sched_amlsim_ctgan_attempt_001.log
test ! -e artifacts/external_validation_v1/runner_logs/extv1_sched_amlsim_tvae_attempt_001.log
test ! -e artifacts/external_validation_v1/runner_logs/extv1_sched_amlsim_cof_attempt_001.log
! tmux has-session -t extv1_sched_amlsim_ctgan 2>/dev/null
! tmux has-session -t extv1_sched_amlsim_tvae 2>/dev/null
! tmux has-session -t extv1_sched_amlsim_cof 2>/dev/null

tmux new-session -d -s extv1_sched_amlsim_ctgan "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_sched_amlsim_ctgan_attempt_001.log 2>&1; CUDA_VISIBLE_DEVICES=\"$EXTV1_GPU_A\" exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset amlsim --model ctgan_separate_class --device cuda:0'"
tmux new-session -d -s extv1_sched_amlsim_tvae "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_sched_amlsim_tvae_attempt_001.log 2>&1; CUDA_VISIBLE_DEVICES=\"$EXTV1_GPU_B\" exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset amlsim --model tvae_separate_class --device cuda:0'"
tmux new-session -d -s extv1_sched_amlsim_cof "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_sched_amlsim_cof_attempt_001.log 2>&1; CUDA_VISIBLE_DEVICES=\"$EXTV1_GPU_C\" exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset amlsim --model cof_seqgen_frozen_non_v3 --device cuda:0'"
```

Mandatory AMLSim GPU terminal barrier:

```bash
while ! extv1_terminal artifacts/external_validation_v1/amlsim/ctgan_separate_class/attempt_001 || ! extv1_terminal artifacts/external_validation_v1/amlsim/tvae_separate_class/attempt_001 || ! extv1_terminal artifacts/external_validation_v1/amlsim/cof_seqgen_frozen_non_v3/attempt_001; do sleep 30; done
```

## Wave 2: three Sparkov GPU jobs in parallel

All precondition checks must pass after the wave-1 barrier and before any
wave-2 launch.

```bash
test ! -e artifacts/external_validation_v1/sparkov/ctgan_separate_class/attempt_001
test ! -e artifacts/external_validation_v1/sparkov/tvae_separate_class/attempt_001
test ! -e artifacts/external_validation_v1/sparkov/cof_seqgen_frozen_non_v3/attempt_001
test ! -e artifacts/external_validation_v1/runner_logs/extv1_sched_sparkov_ctgan_attempt_001.log
test ! -e artifacts/external_validation_v1/runner_logs/extv1_sched_sparkov_tvae_attempt_001.log
test ! -e artifacts/external_validation_v1/runner_logs/extv1_sched_sparkov_cof_attempt_001.log
! tmux has-session -t extv1_sched_sparkov_ctgan 2>/dev/null
! tmux has-session -t extv1_sched_sparkov_tvae 2>/dev/null
! tmux has-session -t extv1_sched_sparkov_cof 2>/dev/null

tmux new-session -d -s extv1_sched_sparkov_ctgan "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_sched_sparkov_ctgan_attempt_001.log 2>&1; CUDA_VISIBLE_DEVICES=\"$EXTV1_GPU_A\" exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset sparkov --model ctgan_separate_class --device cuda:0'"
tmux new-session -d -s extv1_sched_sparkov_tvae "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_sched_sparkov_tvae_attempt_001.log 2>&1; CUDA_VISIBLE_DEVICES=\"$EXTV1_GPU_B\" exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset sparkov --model tvae_separate_class --device cuda:0'"
tmux new-session -d -s extv1_sched_sparkov_cof "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_sched_sparkov_cof_attempt_001.log 2>&1; CUDA_VISIBLE_DEVICES=\"$EXTV1_GPU_C\" exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset sparkov --model cof_seqgen_frozen_non_v3 --device cuda:0'"
```

Mandatory final terminal barrier:

```bash
while ! extv1_terminal artifacts/external_validation_v1/sparkov/ctgan_separate_class/attempt_001 || ! extv1_terminal artifacts/external_validation_v1/sparkov/tvae_separate_class/attempt_001 || ! extv1_terminal artifacts/external_validation_v1/sparkov/cof_seqgen_frozen_non_v3/attempt_001; do sleep 30; done
```

Read-only status checks:

```bash
tmux ls | grep '^extv1_sched_'
find artifacts/external_validation_v1/amlsim artifacts/external_validation_v1/sparkov -maxdepth 4 -type f \( -name COMPLETE.json -o -name INVALID.json -o -name FAILED.json \) -print | sort
tail -n 50 artifacts/external_validation_v1/runner_logs/extv1_sched_*_attempt_*.log
```

No aggregate, selection, internal test, Sparkov fraudTest, TSTR, privacy,
raw-data, materialization, GPU inventory, retry, sweep, or threshold-changing
command is included.
