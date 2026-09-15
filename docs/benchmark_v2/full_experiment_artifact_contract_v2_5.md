# Benchmark v2.5 full-experiment artifact contract

Version: `benchmark-v2.5-full-attempt`.

This contract applies to every CPU baseline, learned baseline, and CoF seed.
Runtime artifacts live only below `artifacts/benchmark_v2_5/` and are not
committed to Git.

## Run identity and attempt allocation

The only valid path shape is:

```text
artifacts/benchmark_v2_5/full/
  joint_semimarkov_v2b/
    kappa_1.00/
      <generator>/
        seed_<1..5>/
          attempt_<NNN>/
```

Attempts are append-only. An existing attempt is never cleared, reused with a
different manifest, or overwritten. A manifest mismatch, terminal failure,
OOM, or code error allocates the next numbered attempt. Conflicting terminal
markers, a missing manifest, a broken latest pointer, or a checkpoint hash
mismatch is artifact corruption and stops the experiment.

## Immutable start files

Before model work, create with exclusive file creation:

- `manifest.json`;
- `RUNNING.json`;
- `stdout.log`;
- `stderr.log`;
- `checkpoints/`.

The manifest records:

- exact Git commit;
- resolved config SHA-256 and relevant-code SHA-256;
- scenario, κ, generator ID, and seed;
- the one shared train-derived SamplingPlan full SHA-256;
- SHA-256 for frozen train, validation, and test data;
- physical GPU ID/name and CUDA/PyTorch versions;
- requested and actual training-budget fields;
- baseline definition version;
- evaluation version.

The manifest is canonical JSON and immutable. `actual_training_budget` begins
with zero updates/time and is finalized in `runtime.json`; the immutable
manifest preserves the requested starting state.

All generator and model-seed manifests must carry one identical SamplingPlan
hash. A missing or different value refuses attempt allocation, resume, and
aggregation.

## Progress and checkpoints

The fixed learned checkpoint interval is 100 updates, which is denser than
10% of each requested schedule. At every interval, atomically save:

- `checkpoints/step_XXXXXXXX.pt` or the adapter-equivalent checkpoint;
- `checkpoints/latest`, a replace-atomic pointer containing step, relative
  path, and SHA-256;
- one append-only `progress.jsonl` record with step, loss, fixed validation
  metric, elapsed seconds, and peak GPU memory;
- replace-atomic `partial_metrics.json`;
- appended stdout/stderr.

Checkpoint filenames are exclusive. Rewriting `step_*.pt` is an error.
Progress writes are one fsynced append per canonical JSON line. Validation
metrics are monitoring only and cannot select a checkpoint or stop training.

CTGAN/TVAE checkpoints contain both class models, the active class, optimizer
state, update counters, and RNG state. Their total elapsed budget is tracked
at baseline level and each class has a separate 3,600-second ceiling. Neural
and CoF checkpoints contain model, optimizer, next step, RNG state, loss
history, and fixed training metadata.

## Exact resume

An interrupted nonterminal attempt may resume only if:

1. its full canonical manifest equals the requested manifest;
2. Git, config, code, and all data hashes therefore match;
3. `checkpoints/latest` parses;
4. the referenced checkpoint exists;
5. its SHA-256 matches the pointer.

Resume starts from that checkpoint and appends progress/logs. A same-manifest
attempt interrupted before its first checkpoint may restart at step zero
inside that same empty attempt. A completed attempt is never rerun. A
different manifest creates `attempt_002`; it never modifies `attempt_001`.

## Scheduler-only attempt_002 continuation

The attempt_002 scheduler-stop correction is a narrow provenance exception,
not a model retry. Its external authorization must verify:

- the immutable attempt_002 full-tree and terminal-marker-tree SHA-256;
- 48 singly terminal attempt_002 jobs and exactly 17 pending jobs;
- previous/current source and relevant-code SHA-256 values;
- unchanged config, split-content, SamplingPlan, baseline-definition, and
  evaluation versions;
- a commit diff limited to the runner, full-attempt allocator, tests, and
  benchmark-v2 documentation.

The 48 terminal jobs are referenced read-only and never fit, sampled,
evaluated, cancelled, or terminalized again. The other 17 jobs allocate at
least `attempt_003`, even when no lower attempt directory exists. Any partial
lower attempt remains in place. Later invocation under the same continuation
authorization may resume or reuse attempt_003 under the normal exact-manifest
rules.

The continuation run state lives at
`full/continuation_attempt_003/`. This prevents its RUNNING, WAITING, STOPPED,
and COMPLETE markers from overwriting the preserved attempt_002 run-level
evidence. Finalization may begin only after the 48 references plus 17 new
terminal outcomes cover the frozen 65-job plan.

## Required terminal artifacts

Immediately after a valid seed finishes, save immutably:

- `sample.npz`;
- `checkpoints/final.pt` (or an adapter-equivalent final checkpoint mirrored
  to this contract name);
- `metrics.json`;
- `runtime.json`;
- `evaluation.json`;
- `COMPLETE.json`.

`metrics.json` contains the seed-level association-recovery error, every
hard-guard result, and the separate 4-bin/8-bin support diagnostics.
`evaluation.json` keeps hard guards under `hard_guards` and non-decision
support results under `diagnostics.support_diagnostics`.
`runtime.json` records requested/actual updates, requested/actual wall time,
cap status, GPU identity, CUDA/PyTorch versions, and peak GPU memory.
`COMPLETE.json` is permitted only when every hard guard is `PASS` and includes
SHA-256 values for all required terminal artifacts.

The evaluation provenance also identifies the train-only rank-maximum
row-guard calibration artifact (200 trials, seed 24,500), short-gap threshold,
coherence reference, and frozen v2.4 gate hash. These evaluation inputs are
created before learned execution and are immutable.

A train-discrete-support, canonical-mask, C0/C1-reference, or direct
row-marginal hard-guard failure does not create a successful primary score.
It writes failure evidence and routes the generator to `INVALID`. A 4-bin or
8-bin confirmatory-support diagnostic failure remains fully reported but does
not null the primary score, block `COMPLETE.json`, or change the primary
decision.

## Failure and unavailable markers

`FAILED.json` records:

- exception type/message and traceback reference;
- last checkpoint;
- peak GPU memory;
- `stdout.log` and `stderr.log` paths;
- the affected attempt and failure class.

`UNAVAILABLE.json` records an import, implementation, environment, or contract
reason and `full_experiment_must_stop=true`. Neither marker is overwritten.
The same infrastructure failure on three consecutive new attempts is a
mandatory stop. Performance is never a failure class.

## Aggregation boundary

Aggregate code reads only terminal `COMPLETE.json` attempts with verified
hashes, the shared SamplingPlan hash, and all five required seeds. A missing,
different-plan, failed, unavailable, corrupt, non-finite,
train-discrete-support-failed, row-marginal-failed, or mask-failed seed makes
that generator `INCOMPLETE` or `INVALID`; it is not removed from an average.
A confirmatory 4-bin/8-bin diagnostic failure alone does not exclude a
COMPLETE seed. Partial metrics are excluded from every comparison, stopping
choice, and hyperparameter decision.
