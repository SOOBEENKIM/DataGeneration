# Benchmark v2.5 frozen full runner

## Scope and freeze

`scripts/run_full_experiment_v2_5.py` is the only v2.5 full-experiment
orchestrator. It does not change the v2.4 smoke-only scope of
`scripts/run_benchmark_v2.py`. It reads the frozen
`configs/benchmark_v2/full_v2_5.yaml`; the config remains
`full_experiment_authorized: false`.

The runner supports exactly three modes:

- `plan`: verifies the clean execution-source/config worktree, every frozen
  split file and content hash, the shared SamplingPlan file and content hash,
  metadata, row-guard calibration, and the exact scenario/kappa/seed/generator
  scope. It also imports, instantiates, and maps the frozen config for all 13
  real registry adapters. It emits the 65-job plan without fit, sample, DGP,
  CUDA API, data preparation, or evaluation calls.
- `dry-run`: creates an isolated metadata-only fixture and verifies external
  authorization, the 65-job plan, attempt allocation, exact resume, mismatch
  routing to a new attempt, INVALID routing, and fake-GPU scheduling. It does
  not import the DGP, generator registry, model adapters, or CUDA runtime.
- `full`: the real frozen experiment path. It is fail-closed and must not be
  invoked without a separate explicit user authorization.

The immutable experimental scope is:

- scenario `joint_semimarkov_v2b`;
- kappa `1.0`;
- model seeds `1, 2, 3, 4, 5`;
- the 13 generators in `full_v2_5.yaml`, in their registered order;
- the shared SamplingPlan full SHA-256 for every job.

## External full authorization

Full mode requires:

`artifacts/benchmark_v2_5/full/authorization/authorization.json`

The manifest schema is `benchmark-v2.5-full-authorization-v1`. It must pin:

- the current source commit and frozen config SHA-256;
- the v2.4 gate-pass tag and commit;
- the v2.4 artifact-index and gate-report SHA-256 values;
- the capacity-preflight artifact-index and correction-note SHA-256 values;
- the frozen data-manifest, split-content, and SamplingPlan SHA-256 values;
- the exact scenario, kappa, seed, generator, and analysis scope;
- the explicit approval text and its SHA-256.

The approval text may be embedded as `approval_text`, or supplied by
`approval_text_path` relative to the authorization manifest. The runner hashes
the actual bytes and compares them with `approval_text_sha256`; an unanchored
hash is rejected.

If frozen data is absent, data preparation is allowed only when the
authorization additionally contains
`frozen_data.allow_prepare_if_absent: true` and already pins the expected
manifest, split, and SamplingPlan hashes. Preparation uses the exclusive-write
contract in `scripts/prepare_full_data_v2_5.py`. Existing partial data or
calibration output is never overwritten. After preparation, all authorized
hashes are checked before a baseline can run.

Any source, config, v2.4, capacity, data, SamplingPlan, scope, or approval
mismatch stops before model execution.

The one authorized scheduler-continuation exception adds a `continuation`
object. It pins the previous/current source and relevant-code hashes, the
exact committed changed-path list, the unchanged config/data/SamplingPlan
hashes, both preserved attempt_002 tree hashes, the 48 terminal and 17
pending job counts, and the fixed transition `attempt_002 -> attempt_003`.
Only the runner, full-attempt allocator, their tests, and benchmark-v2 docs
may differ. A change to config, DGP/benchmarks, evaluator, generator/model,
data, SamplingPlan, endpoint, hard guards, or statistics rejects the
authorization.

## Plan and execution phases

Each plan entry records its device class, seed, scenario, kappa, requested
steps, wall cap, checkpoint interval, SamplingPlan hash, attempt template,
dependencies, execution order, and expected artifacts. The fixed path is:

```text
artifacts/benchmark_v2_5/full/
  joint_semimarkov_v2b/kappa_1.00/
    <generator>/seed_<1..5>/attempt_<NNN>/
```

Full mode has two ordered phases:

1. Phase A runs the nine CPU generators under their frozen per-job caps.
2. Phase B reads live GPU inventory before assignment and before every wave.
   A GPU is eligible only when it has no compute process, memory use is at most
   1,024 MiB, and utilization is at most 5%. Both `nvidia-smi -L` and PyTorch
   CUDA availability/device count must succeed in the actual full-run shell.
   A busy GPU is excluded individually; it cannot stop a wave while another
   eligible GPU exists. Effective wave size is
   `min(configured maximum concurrency=4, current idle GPU count)`, so the
   runner continues at concurrency three, two, or one as availability
   changes. One runner-owned spawned process receives one heavy job and an
   explicit physical `CUDA_VISIBLE_DEVICES` value. CTGAN/TVAE class models
   retain their shared per-baseline budget from the frozen config.

If the live idle count is zero, the runner creates a new append-only
`gpu_wait_attempt_NNN/WAITING_FOR_GPU.json`, appends a heartbeat on every
poll, and rereads inventory every 30 seconds. An idle device causes automatic
`RESUMED.json` and scheduling continues. Only exhaustion of the fixed
1,800-second wait budget creates `WAIT_BUDGET_EXHAUSTED.json` and mandatory
STOPPED state. The selector never terminates an external process.

The runner passes the frozen adapter parameters unchanged. In particular,
CoF uses `d_model=128`, `n_layers=2`, `diffusion_steps=50`, and
`sampling_chunk_size=256`.

Every CPU and GPU cell-seed runs in a runner-owned child process. The parent
sets a monotonic deadline from `job.max_wall_seconds`; the cap covers child
startup, training, sampling, and evaluation. At the deadline the parent sends
SIGTERM only to that owned child, waits a bounded grace period, and sends
SIGKILL only if the child remains alive. A wall-cap result is permanently
recorded as `FAILED` with `failure_class=wall_cap`, actual elapsed time, peak
recorded GPU memory, and the last atomic checkpoint pointer. There is no
unbounded GPU-worker join, and no external user or GPU process is targeted.

## Attempt and resume contract

An attempt begins with an exclusive immutable `manifest.json`,
`RUNNING.json`, and append-only logs. It records source/config/code/data/plan
hashes, GPU and software versions, and requested/actual budgets.

During execution it maintains:

- atomic `heartbeat.json` and `partial_metrics.json`;
- append-only `stdout.log`, `stderr.log`, `progress.jsonl`, and
  `resource.jsonl`;
- atomic hashed `checkpoints/latest` and immutable step checkpoints;
- a final sample, checkpoint, raw metrics, runtime, and evaluation;
- append-only SHA-256 entries in `artifact_index.jsonl`;
- exactly one terminal marker: `COMPLETE.json`, `FAILED.json`,
  `INVALID.json`, `UNAVAILABLE.json`, `CANCELLED.json`, or
  `INTERRUPTED.json`.

Only a byte-identical manifest resumes the latest checkpoint. Any
source/config/code/data/SamplingPlan/GPU manifest change allocates the next
immutable attempt. An INVALID attempt is not silently retried. A third
consecutive identical infrastructure failure sets the mandatory-stop marker.
KeyboardInterrupt, SystemExit, and SIGTERM are written as operator
interruptions with the latest checkpoint, propagated to the parent, and never
misclassified as a model/code `FAILED`.

For the provenance-locked scheduler correction only, preserved terminal
attempt_002 jobs are replayed read-only into the new run state. They are not
allocated, fit, sampled, or evaluated again. Exactly the 17 jobs without an
attempt_002 terminal marker are pending; their allocator floor is
`attempt_003`, so no synthetic attempt_001/attempt_002 placeholders are
created. This set includes CTGAN seeds 4 and 5, TVAE seeds 1–5, neural
sequence seeds 1–5, and CoF seeds 1–5. If any attempt_002 tree, terminal
marker, manifest provenance, config/data/SamplingPlan value, or authorized
changed path differs, continuation fails closed.

Hard-guard failures are INVALID. OOM, non-finite training failures, and wall
cap failures are preserved as FAILED. INVALID seeds are never removed as NaN
before aggregation.

## Run-level failure policy

`full/run_state.jsonl` is append-only. Its events include generator, seed,
attempt, status, failure class, timestamp, and manifest hash. Atomic current
markers are `RUNNING.json`, `STOPPED.json`, and `COMPLETE.json`.

- Frozen-data, authorization, manifest, hard global contract, GPU-runtime, or
  artifact-corruption failures stop the whole run.
- A secondary generator's FAILED/INVALID/UNAVAILABLE result terminally
  cancels that generator's remaining queued seeds, but does not block other
  independent generators or the C2-primary generators.
- If empirical i.i.d., CTGAN, TVAE, or CoF is invalid, C2 is
  `NOT_EVALUABLE`; other already authorized independent jobs may still produce
  terminal artifacts.
- One GPU job failure does not suppress unrelated later GPU jobs.
- Three consecutive identical infrastructure failures create mandatory
  `STOPPED.json`; no later queued job starts.
- Finalization begins only when all 65 jobs are COMPLETE, FAILED, INVALID,
  UNAVAILABLE, or explicitly CANCELLED.

Continuation writes its new run-state markers below
`full/continuation_attempt_003/`; the preserved attempt_002 top-level
RUNNING/STOPPED evidence is not overwritten. The 48 replayed terminal events
and 17 attempt_003 outcomes must together cover all 65 planned jobs before
finalization. Preserved CTGAN INVALID outcomes keep C2 `NOT_EVALUABLE`;
CTGAN seeds 4–5 still run because their five-seed hard-guard pattern remains
part of the frozen plan.

## Frozen aggregation and final artifacts

No aggregate, p-value, or C2 conclusion is produced while any of the 65 jobs
lacks a terminal marker. A generator is valid for aggregation only with all
five COMPLETE seeds and PASS hard guards.

The C2 primary Holm family is exactly CoF versus:

- `ctgan_separate_class`;
- `tvae_separate_class`;
- `empirical_iid`.

The secondary family is separately CoF versus `neural_sequence`,
`independent_markov`, `joint_markov`, `plug_in_hmm`, and `plug_in_hsmm`.
Secondary results cannot change C2. If CoF, CTGAN, TVAE, or empirical i.i.d.
does not have five valid seeds, no C2 aggregate is computed. A failed C2 uses
the exact conclusion:

`C2 was not supported in this preregistered experiment.`

After all jobs are terminal, full mode creates the final payloads under an
append-only staging directory such as
`full/finalization_attempt_001/`:

- `full_experiment_report.md`;
- `raw_results.csv`;
- `aggregate_statistics.csv`;
- `effect_sizes_and_holm.csv`;
- `per_seed_runtime.csv`;
- `support_diagnostics.csv` with separate 4-bin and 8-bin all-bin validity,
  dropped-bin score, occupancy-penalized score, and invalid-bin count;
- `artifact_index.json`;
- `checksum_manifest_report.json`.

The same support-diagnostic fields are rendered in
`full_experiment_report.md`. They are non-decision diagnostics and cannot
invalidate a seed whose true hard validity guards pass.

Each payload is completed and hashed before the index and checksum manifest
are created. Only after every staged hash verifies does the runner atomically
write top-level `full/FINAL_COMPLETE.json`; the run-level COMPLETE marker is
later still. If finalization is interrupted, a later invocation audits and
preserves the partial staging attempt, creates
`finalization_attempt_002` (or the next number), and never overwrites partial
staged or legacy top-level files.

For continuation, `checksum_manifest_report.json` also identifies the
preserved attempt number, new attempt number, both attempt_002 tree hashes,
and the 48/17 reused/pending counts. Valid secondary diagnostics and every
failure artifact remain in the 65-job collection even when C2 is
`NOT_EVALUABLE`.

Runtime data and artifacts under `artifacts/benchmark_v2_5/` and
`data/benchmark_v2_5/` are ignored by Git.

The read-only shell diagnosis is recorded at
`artifacts/benchmark_v2_5/runner_preflight/gpu_environment_diagnosis.md`.
At preparation time the current shell had no `/dev/nvidia*`,
`nvidia-smi -L` failed, and PyTorch reported CUDA unavailable with zero
devices. This differs from the preserved capacity artifact's visible RTX 3090
UUID. The runner therefore cannot enter full mode from this shell. The
capacity preflight is not rerun: its frozen timing measurements and projection
remain valid provenance, while GPU shell visibility is a separate
environment-access precondition.

## Safe commands for this preparation stage

The following commands do not run real data generation or training:

```bash
python3 -m scripts.run_full_experiment_v2_5 \
  --mode plan \
  --data-manifest <existing-read-only-manifest>

python3 -m scripts.run_full_experiment_v2_5 \
  --mode dry-run \
  --work-root <new-temporary-directory>
```

The command below is deliberately documented but was not invoked during this
runner-hardening work:

```bash
python3 -m scripts.run_full_experiment_v2_5 --mode full
```

Full experiment, five-seed execution, sweeps, frozen-data generation, CPU
baseline execution, and GPU training remain prohibited for this preparation
stage.
