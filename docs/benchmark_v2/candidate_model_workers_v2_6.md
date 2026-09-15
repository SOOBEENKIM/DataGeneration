# v2.6 candidate model-worker execution contract

## Scope

This source amendment adds orchestration only. It does not change
`selection_v2_6.yaml`, candidate definitions, training/sampling behavior,
hard caps, train-only transforms or SamplingPlan, validation guards, test
exclusion, or primary/secondary selection policy.

The only worker IDs are:

1. `ctgan_separate_class`;
2. `tvae_separate_class`;
3. `neural_sequence`;
4. `cof_seqgen`.

Each worker owns exactly three training trajectories and four evaluation
candidates. The combined scope remains 12 trajectories and 16 candidates.
For every model, c00/c01 share the native 20,000-update trajectory and use
its fixed 10,000 and 20,000 checkpoints.

## Ownership and terminal state

The runner requires `--model`; an omitted or unknown ID fails closed.
Exclusive creation of
`candidates/_workers/<model_id>/ownership.lock` is the concurrency boundary.
The lock, immutable worker manifest, and RUNNING marker record source,
relevant-code, config, development-manifest, SamplingPlan, authorization,
model, trajectory, candidate, and selection-seed provenance.

A second process cannot claim an already-owned model. Different model locks
do not interact. One model's failure neither changes nor cancels another
model's state.

`WORKER_COMPLETE.json` requires:

- three COMPLETE trajectory results;
- all four fixed candidate result and COMPLETE artifacts;
- immutable hashes for every indexed candidate artifact;
- `selection_aggregate_created=false`.

Any incomplete trajectory or missing candidate produces
`WORKER_FAILED.json`. Neither terminal state is overwritten.

## Aggregate-only boundary

Workers never calculate validation guards, choose a candidate, or write
selection/freeze artifacts. The aggregate-only CLI checks all four immutable
worker completions and all 16 candidate artifact hashes before creating its
append-only output attempt. It then applies the unchanged five-guard
validation selection rule. Direct invocation of the legacy selection entry
point delegates to the same readiness gate.

The aggregate command rejects test/fresh-test paths. It grants no fresh-test,
five-seed, full-run, sweep, retry, threshold, or tuning authority.

## Corrective CTGAN/TVAE continuation

The append-only continuation contract is specified in
`candidate_continuation_v2_6.md`. It does not alter the original attempt_001
worker contract. A matching continuation authorization may create only:

- CTGAN/TVAE model worker `attempt_002`;
- evaluation-only c00/c01 artifacts from the verified attempt_001 native
  checkpoints, with no native retraining;
- the c02 and c03 training trajectory `attempt_002` paths.

CoF attempt_001 is reused and never rerun. Neural uses its first normal
worker attempt. Aggregate readiness resolves the exact model terminal
attempts named by a later aggregate authorization; it no longer assumes
that every terminal is attempt_001.

## Capacity

Each model owns three trajectories capped at two GPU-hours each:

- total maximum: 24 GPU-hours;
- four-GPU hard-cap wall-time ceiling: approximately 6 hours;
- three-GPU two-wave hard-cap wall-time ceiling: approximately 8 hours.

These are cap-based projections, not new measurements.

## Preparation-only verification

The amendment is tested with fake filesystem fixtures and read-only
planning. It does not query GPU inventory or CUDA and does not call DGP,
model fit, model sample, candidate training, or validation selection.
