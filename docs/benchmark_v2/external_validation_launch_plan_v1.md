# External validation v1 authorization and SSH launch plan

Status: **SUPERSEDED — DO NOT EXECUTE**. The original all-`attempt_001` plan is
preserved for history only. After the empirical receiver-TV performance
correction, only `external_validation_performance_continuation_launch_plan_v1.md`
and its separately generated continuation authorization may be used.

## Exact authorized matrix

| Batch | Dataset | Model | Seed | Device | Requested updates | Training cap | Whole-job cap | GPU-hours upper |
|---|---|---|---:|---|---:|---:|---:|---:|
| CPU | AMLSim | empirical_iid | 31001 | CPU | 0 | 0 | 7,200 s | 0 |
| CPU | Sparkov | empirical_iid | 32001 | CPU | 0 | 0 | 7,200 s | 0 |
| GPU wave 1 | AMLSim | ctgan_separate_class | 31001 | GPU A | 10,000 | 7,200 s | 10,800 s | 2.0 |
| GPU wave 1 | AMLSim | tvae_separate_class | 31001 | GPU B | 20,000 | 7,200 s | 10,800 s | 2.0 |
| GPU wave 1 | AMLSim | cof_seqgen_frozen_non_v3 | 31001 | GPU C | 20,000 | 7,200 s | 10,800 s | 2.0 |
| GPU wave 2 | Sparkov | ctgan_separate_class | 32001 | GPU A | 10,000 | 7,200 s | 10,800 s | 2.0 |
| GPU wave 2 | Sparkov | tvae_separate_class | 32001 | GPU B | 20,000 | 7,200 s | 10,800 s | 2.0 |
| GPU wave 2 | Sparkov | cof_seqgen_frozen_non_v3 | 32001 | GPU C | 20,000 | 7,200 s | 10,800 s | 2.0 |

The learned-model training budget remains the frozen v2.5 two-hour cap. The
three-hour whole-job cap additionally covers train-only bootstrap, checkpoint
and sample writes, validation sampling, metrics, and terminal finalization. The
empirical whole-job cap is two hours. A cap failure is terminal and cannot be
retried under this authorization.

The maximum authorized learned compute is 12 GPU-hours. With three distinct
idle physical GPUs, the two GPU waves have a six-hour whole-job upper bound.
Running the two CPU controls together first contributes at most two hours, so
the conservative end-to-end upper bound is eight hours. This is a scheduling
bound, not an instruction to extend any model budget.

## Authorization

The authorization is stored exclusively under:

```text
artifacts/external_validation_v1/authorization_history/
  authorization_<source-head>_attempt_001.json
```

It contains exactly eight child grants and forbids retries, early stopping,
sweeps, aggregation, GPU inventory queries, test/TSTR/privacy/full execution,
raw CSV access, materialization, and existing artifact mutation. Every child
binds source/config/bundle/train/validation hashes, fixed seed, model budget,
attempt path, and the validation conditioning rule. The runner rejects an
authorization missing any job or containing an extra/duplicate job.

## SSH environment and duplicate-prevention preamble

Run from the same project SSH shell. The user selects three idle physical GPU
indices outside this runner; the commands below do not query GPU inventory.

```bash
export EXTV1_REPO=<REPO_ROOT>
export EXTV1_PY=<COFSEQ_PYTHON>
export EXTV1_AUTH="$EXTV1_REPO/artifacts/external_validation_v1/authorization_history/authorization_<source-head>_attempt_001.json"
export EXTV1_GPU_A='<idle-physical-index-A>'
export EXTV1_GPU_B='<idle-physical-index-B>'
export EXTV1_GPU_C='<idle-physical-index-C>'
test "$EXTV1_GPU_A" != "$EXTV1_GPU_B"
test "$EXTV1_GPU_A" != "$EXTV1_GPU_C"
test "$EXTV1_GPU_B" != "$EXTV1_GPU_C"
cd "$EXTV1_REPO"
test "$(git rev-parse HEAD)" = '<source-head>'
test -f "$EXTV1_AUTH"
```

Before every launch, both the tmux session and target attempt/log must be
absent. The command uses shell `noclobber` for the runner log. The runner itself
uses exclusive attempt ownership, so a race still fails closed.

## Batch 0: CPU controls

These use no GPU and may run together. Wait until both terminal markers exist
before GPU wave 1 to avoid train-bootstrap CPU contention.

```bash
test ! -e artifacts/external_validation_v1/amlsim/empirical_iid/attempt_001
test ! -e artifacts/external_validation_v1/runner_logs/extv1_amlsim_iid_attempt_001.log
! tmux has-session -t extv1_amlsim_iid 2>/dev/null
tmux new-session -d -s extv1_amlsim_iid "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_amlsim_iid_attempt_001.log 2>&1; exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset amlsim --model empirical_iid --device cpu'"

test ! -e artifacts/external_validation_v1/sparkov/empirical_iid/attempt_001
test ! -e artifacts/external_validation_v1/runner_logs/extv1_sparkov_iid_attempt_001.log
! tmux has-session -t extv1_sparkov_iid 2>/dev/null
tmux new-session -d -s extv1_sparkov_iid "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_sparkov_iid_attempt_001.log 2>&1; exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset sparkov --model empirical_iid --device cpu'"
```

Create the log directory once, before the first launch, with
`mkdir -p artifacts/external_validation_v1/runner_logs`. This is permitted only
at execution time and is not performed during authorization preparation.

## GPU wave 1: AMLSim

```bash
test ! -e artifacts/external_validation_v1/amlsim/ctgan_separate_class/attempt_001
test ! -e artifacts/external_validation_v1/amlsim/tvae_separate_class/attempt_001
test ! -e artifacts/external_validation_v1/amlsim/cof_seqgen_frozen_non_v3/attempt_001
test ! -e artifacts/external_validation_v1/runner_logs/extv1_amlsim_ctgan_attempt_001.log
test ! -e artifacts/external_validation_v1/runner_logs/extv1_amlsim_tvae_attempt_001.log
test ! -e artifacts/external_validation_v1/runner_logs/extv1_amlsim_cof_attempt_001.log
! tmux has-session -t extv1_amlsim_ctgan 2>/dev/null
! tmux has-session -t extv1_amlsim_tvae 2>/dev/null
! tmux has-session -t extv1_amlsim_cof 2>/dev/null
tmux new-session -d -s extv1_amlsim_ctgan "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_amlsim_ctgan_attempt_001.log 2>&1; CUDA_VISIBLE_DEVICES=\"$EXTV1_GPU_A\" exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset amlsim --model ctgan_separate_class --device cuda:0'"
tmux new-session -d -s extv1_amlsim_tvae "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_amlsim_tvae_attempt_001.log 2>&1; CUDA_VISIBLE_DEVICES=\"$EXTV1_GPU_B\" exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset amlsim --model tvae_separate_class --device cuda:0'"
tmux new-session -d -s extv1_amlsim_cof "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_amlsim_cof_attempt_001.log 2>&1; CUDA_VISIBLE_DEVICES=\"$EXTV1_GPU_C\" exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset amlsim --model cof_seqgen_frozen_non_v3 --device cuda:0'"
```

## GPU wave 2: Sparkov

Launch only after all three wave-1 attempts have a terminal marker.
For the frozen CoF job, `--model` and `cof_seqgen_frozen_non_v3` are two
separate argv tokens; the merged spelling `--modelcof_seqgen_frozen_non_v3`
is invalid and must never be used.

```bash
test ! -e artifacts/external_validation_v1/sparkov/ctgan_separate_class/attempt_001
test ! -e artifacts/external_validation_v1/sparkov/tvae_separate_class/attempt_001
test ! -e artifacts/external_validation_v1/sparkov/cof_seqgen_frozen_non_v3/attempt_001
test ! -e artifacts/external_validation_v1/runner_logs/extv1_sparkov_ctgan_attempt_001.log
test ! -e artifacts/external_validation_v1/runner_logs/extv1_sparkov_tvae_attempt_001.log
test ! -e artifacts/external_validation_v1/runner_logs/extv1_sparkov_cof_attempt_001.log
! tmux has-session -t extv1_sparkov_ctgan 2>/dev/null
! tmux has-session -t extv1_sparkov_tvae 2>/dev/null
! tmux has-session -t extv1_sparkov_cof 2>/dev/null
tmux new-session -d -s extv1_sparkov_ctgan "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_sparkov_ctgan_attempt_001.log 2>&1; CUDA_VISIBLE_DEVICES=\"$EXTV1_GPU_A\" exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset sparkov --model ctgan_separate_class --device cuda:0'"
tmux new-session -d -s extv1_sparkov_tvae "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_sparkov_tvae_attempt_001.log 2>&1; CUDA_VISIBLE_DEVICES=\"$EXTV1_GPU_B\" exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset sparkov --model tvae_separate_class --device cuda:0'"
tmux new-session -d -s extv1_sparkov_cof "bash -lc 'cd \"$EXTV1_REPO\"; set -C; exec >artifacts/external_validation_v1/runner_logs/extv1_sparkov_cof_attempt_001.log 2>&1; CUDA_VISIBLE_DEVICES=\"$EXTV1_GPU_C\" exec \"$EXTV1_PY\" -m scripts.run_external_validation_v1 --repo-root . --config configs/benchmark_v2/external_validation_v1.yaml --mode execute --authorization \"$EXTV1_AUTH\" --dataset sparkov --model cof_seqgen_frozen_non_v3 --device cuda:0'"
```

## Read-only status checks

```bash
tmux ls | grep '^extv1_'
find artifacts/external_validation_v1/amlsim artifacts/external_validation_v1/sparkov -maxdepth 4 -type f \( -name COMPLETE.json -o -name INVALID.json -o -name FAILED.json \) -print | sort
tail -n 50 artifacts/external_validation_v1/runner_logs/extv1_*_attempt_001.log
```

Do not launch wave 2 until the three AMLSim terminal paths exist. A terminal
failure is reported as-is; it does not permit retry, tuning, threshold changes,
or replacement of an attempt.
