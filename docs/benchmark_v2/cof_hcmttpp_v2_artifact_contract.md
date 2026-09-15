# CoF-HCMTTPP-v2 future artifact contract

## Signed tail-scale amendment

The checkpoint must use the signed residual contract
`log beta=log beta_base+r_beta`, with zero-initialized residual weights and
bias. The earlier additive positive residual is superseded. This amendment
changes no candidate, threshold, tail gate, central RQS, gate, budget, or
frozen reference.

## Tail-conditionality amendment

The artifact schema records a train-only fixed threshold and baseline tail
statistics, not a fixed class-level runtime tail probability or scale.
Event-level tail mass and scale are conditional checkpoint outputs. This
amendment changes no candidate, gate, budget, or frozen reference.

## Status and purpose

This is a schema specification for a possible later H1 validation execution.
The source-only core `CoFHCMTTPPV2H1` and `H1HurdleRQSGapDecoder` now exist.
`TrainOnlyH1TailState`, `build_h1_checkpoint_bundle`, and
`load_h1_checkpoint_bundle` implement the train-state and strict in-memory
CPU checkpoint seams described below. No directory described here is created
by this implementation. There is no runner, authorization, persisted
checkpoint, sample, evaluation, or runtime artifact.

The v1 artifact tree is immutable and remains stopped at
`STOP_CCMTPP_V1_CHAIN_C1_GATE_FAIL`. A v2 process may reference frozen C0/C1
hashes but may never write under `artifacts/cof_ccmtpp_v1`,
`artifacts/external_validation_v1`, or the frozen external data roots.

## Future append-only layout

If separately implemented and authorized later, the only allowed shape is:

```text
artifacts/cof_hcmttpp_v2/external_validation/
  workers/<dataset>/H1/ownership.lock
  workers/<dataset>/H1/attempt_NNN/
    WORKER_COMPLETE.json | WORKER_FAILED.json
  <dataset>/H1/seed_4001/attempt_NNN/
    manifest.json
    frozen_parent_references.json
    gap_hurdle_state.json
    positive_gap_spline_state.json
    positive_gap_tail_state.json
    amount_transform_reference.json
    receiver_vocabulary_reference.json
    conditioning_plan.json
    progress.jsonl
    checkpoints/step_*.pt
    checkpoints/latest
    checkpoints/final.pt
    checkpoint_provenance.json
    validation_sample.npz
    diagnostics.json
    metrics.json
    evaluation.json
    gate_decision.json
    runtime.json
    artifact_index.json
    checksum_manifest.json
    COMPLETE.json | INVALID.json | FAILED.json
```

All files use exclusive-create and atomic-finalize semantics. Nothing is
overwritten, deleted, moved, or retried. Checksum/index completion precedes
the single candidate terminal marker; the verified candidate terminal
precedes the worker terminal marker.

## Immutable manifest

`manifest.json` must bind:

- source commit, relevant model/runner source hashes, config hash, and
  authorization hash;
- dataset, H1, seed 4001, attempt, requested updates, wall cap, optimizer,
  batch size, and checkpoint interval;
- frozen train/validation manifest, transform, receiver vocabulary,
  SamplingPlan, metric, and threshold hashes;
- forensic commit and Markdown/JSON/CSV hashes;
- frozen C0 sample/evaluation/terminal hashes and frozen C1
  sample/gate/metrics/evaluation/terminal hashes;
- exact C1 backbone, amount, receiver, loss-aggregation, and budget
  fingerprints reused by H1;
- a factor diff containing exactly `gap_decoder` and no other factor;
- zero validation/internal-test/fraudTest fit rows and every forbidden-access
  counter.

Any mismatch fails before data-body, model, device, or runtime access.

## Train-only gap state

`gap_hurdle_state.json` records per-Y train event counts, exact-zero counts and
rates, positive counts, fit split, source hashes, and zero rows from every
non-train split. The Bernoulli head parameters belong to the checkpoint, not
to validation calibration.

`positive_gap_spline_state.json` records 16 bins, fixed probability-knot rule,
minimum height/derivative constants, conditional parameter schema, inverse
and log-Jacobian versions, and the train/source/config hashes. Knot count or
rule cannot vary by dataset or observed result.

`positive_gap_tail_state.json` records for each Y: `n_positive`, `m_y`, the
strict-exceedance ordered-value hash, `u_tail`, exceedance count,
`p_tail_base`, `beta_base`, support `(0,+infinity)`, and the absence of an
upper clip. It also records the zero-residual gate initialization, the signed
log-scale formula, zero residual-weight/bias initialization, exact baseline
scale equality at initialization, their hashes, and explicit
validation/internal-test/fraudTest fit counts of zero.
These baselines are initialization anchors only. A fixed class-level runtime
tail mass or scale is forbidden. The runtime contract must admit both
`beta<beta_base` and `beta>beta_base` for finite signed residuals while
requiring a strictly positive scale. An unavailable or invalid tail state is
a terminal failure, not a fallback.

## Checkpoint and sample provenance

Every checkpoint repeats all manifest and train-only state hashes, actual
update, elapsed time, RNG state, exact module fingerprints, and conditional
tail-gate/signed-log-scale residual parameters. Reload is strict. A
provenance mismatch cannot resume or sample.

`validation_sample.npz` contains the unchanged amount/receiver/mask/Y/length
schema plus raw continuous gap and the frozen evaluator-bin mapping used only
for diagnostics. It records sample and SamplingPlan hashes. It may contain no
internal-test or fraudTest row.

## Required diagnostics

H1 must preserve all C1 diagnostics and add only gap-decoder diagnostics:

- classwise exact-zero target/sample rate and absolute error;
- classwise overall and positive-only gap KS;
- positive-gap q10/q25/q50/q75/q90/q95/q99 differences;
- RQS inverse and density finite counts;
- central versus tail target/sample counts and tail mass error by class and
  preregistered history strata;
- conditional tail-gate probability and conditional tail-scale summaries by
  class and those same strata;
- signed `r_beta`, `beta/beta_base`, below-baseline, equal-at-initialization,
  and above-baseline scale diagnostics;
- per-event positive CDF total-mass error, threshold left/right CDF
  continuity error, and finite zero/central/tail NLL counts;
- non-finite/overflow scale counts, which must be zero and may not be clipped
  or redrawn;
- minimum/maximum sampled gap, lower-bound rate, non-finite count, and exact
  upper-clip count, which must be zero;
- frozen-bin mass and raw-continuous versus mapped KS difference;
- classwise coherence, classwise/full receiver TV, and head/tail/UNK receiver
  diagnostics;
- hard validity and all forbidden-access counters.

No invalid or failed value is omitted from the artifact. Aggregation may not
drop a failed dataset or class.

## Gate and terminal state

`gate_decision.json` implements exactly the five conjunctive preregistered
requirements. It includes unrounded H1 values, frozen C0/C1 comparison values,
individual booleans, provenance hashes, and one overall decision.

If any requirement fails, the candidate terminal is valid evidence but the
family state is `STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL`. No subsequent
candidate or retry path exists. A PASS marker still does not authorize any
test or downstream execution.

Interruption, wall cap, non-finite density/sample, forbidden access, checksum
damage, or provenance mismatch records a terminal FAILED/INVALID artifact.
No process outside a future runner-owned child may be signalled.
