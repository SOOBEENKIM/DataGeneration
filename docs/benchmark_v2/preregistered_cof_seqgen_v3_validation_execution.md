# Preregistered CoF-SeqGen v3 validation candidate execution

## Status and present boundary

This commit implements the execution runner but does not authorize or run it.
No authorization manifest, runtime root, launch command, GPU inventory query,
CUDA call, optimizer update, checkpoint, sample, validation result, selection,
test, TSTR, privacy analysis, or full experiment is created in this phase.

The frozen source-preparation commit is
`165644a06309e3f18cfd34c8dd9ca1728ade192a`. All v2.5–v2.8 data,
runtime, checkpoints, authorizations, aggregates, and forensic evidence remain
read-only.

## Finite execution plan

Exactly two independently owned candidate operations are permitted:

| Candidate | Architecture | Seed | Updates | Whole-operation cap |
|---|---|---:|---:|---:|
| `cof_v3_c01_direct_joint` | one Cartesian joint state/head | 3001 | 20,000 | 7,200 seconds |
| `cof_v3_c02_factorized_joint` | gap then receiver conditioned on sampled gap | 3001 | 20,000 | 7,200 seconds |

Both use Adam (`lr=0.001`, weight decay `0`), batch size 256, checkpoint
interval 1,000 updates, 50 diffusion steps, CFG scale 2.0, categorical
temperature 1.0, paired masking, `coherence_lambda=0.0`, and the frozen
train-fitted centered empirical-residual amount contract. There is no sweep,
early stopping, update increase, validation-dependent fit, independent legacy
gap/receiver head, or post-hoc discrete calibration.

The runner requires an explicit future authorization that binds the then
current source commit and relevant source hash, both v3 config hashes, the
development manifest, frozen train and validation file/content hashes, the
SamplingPlan hash, and exactly these two candidate IDs. The authorization may
allow CUDA, fit, fixed optimizer updates, checkpoint writes, sampling, and the
validation five-guard evaluation. It must deny GPU inventory query, aggregate
selection, test/fresh-test, TSTR, privacy, sweep, early stopping, and
five-seed/full execution.

## Split and runtime path contract

The SamplingPlan is reconstructed from the frozen train-policy record
(`entity_count=7989`, seed 26001) and must hash to
`862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27`.
Training, paired joint-support fitting, and the centered empirical amount
fit-state use train only. The validation file is not opened until fitting and
synthetic generation finish. Validation is then used only for the fixed five
guards. Paths containing a test or fresh-test split are not part of the
runner's input contract and authorization cannot enable them.

At runtime the adapter verifies that c01 owns `DirectJointDiscretePath` and
c02 owns `FactorizedJointDiscretePath`. Direct sampling chooses one supported
joint state and decodes it once. Factorized sampling chooses gap first and
feeds that exact sampled gap into the receiver head. Detection of a legacy
`bin_head`/`cat_heads`, a wrong path, a nonzero coherence weight, or any
post-hoc hook fails closed.

## Child watchdog and append-only output

Each candidate has an exclusive ownership record and an append-only numbered
attempt. A parent-owned child contains fit, train-only amount fit, validation
sample generation, and five-guard evaluation. The parent monitors a monotonic
7,200-second deadline, sends SIGTERM only to that child, waits a bounded grace
period, and then sends SIGKILL only to that same child if needed. A wall-cap
termination is permanently `FAILED`; it is not early stopping.

The candidate terminal bundle includes immutable manifest, runtime,
checkpoint, checkpoint provenance, joint-support state, amount fit-state,
conditioning binding, sample, metrics, evaluation, periodic checkpoint and
progress evidence, and exactly one `COMPLETE`, `INVALID`, or `FAILED` marker.
All required artifact hashes are embedded in the terminal marker.

`COMPLETE` requires a valid all-five PASS evaluation. A completed candidate
that fails one or more frozen guards is `INVALID`, with all measurements
preserved. Infrastructure/code failure is `FAILED`. No missing or invalid
candidate can be silently omitted.

## Deferred aggregate

This commit only defines a fail-closed future aggregate-input schema. It
requires terminal, hash-bound evidence for both candidates and marks as
eligible only `COMPLETE` plus all-five PASS. It does not execute selection.
Aggregate authorization and execution, and every downstream test operation,
remain separate future decisions.

## Source-only plan and dry-run

`scripts.run_cof_seqgen_v3_validation plan` reports the exact two operations.
`dry-run` verifies source-preparation provenance, the development manifest,
train content, validation file hash without loading validation rows, the
train-derived SamplingPlan, v2.5/v2.8 frozen evidence, architecture imports,
and absence of the v3 runtime root.

Both commands keep GPU query, CUDA, fit, sample, validation evaluation,
selection, test, TSTR, privacy, and full execution counts at zero and create
neither authorization nor runtime artifacts.
