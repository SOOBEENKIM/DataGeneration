# CoF-ZDH-v3 D1 future artifact contract

## Status

This is a schema-only contract. No artifact path described here is created by
this package. D1 is not implemented, and there is no runner or authorization.
The H1 family remains read-only and permanently stopped at
`STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL`.

## Future append-only layout

A separately implemented and authorized validation runner would be limited
to this conceptual layout:

```text
artifacts/cof_zdh_v3/external_validation/
  workers/<dataset>/D1/ownership_attempt_001.lock
  workers/<dataset>/D1/attempt_001/
    WORKER_COMPLETE.json | WORKER_INVALID.json | WORKER_FAILED.json
  <dataset>/D1/seed_4001/attempt_001/
    manifest.json
    frozen_parent_references.json
    train_only_gap_support_state.json
    amount_transform_reference.json
    receiver_vocabulary_reference.json
    progress.jsonl
    numerical_health.jsonl
    numerical_failure.json
    checkpoints/step_*.pt
    checkpoints/latest
    checkpoints/final.pt
    checkpoint_provenance.json
    conditioning_plan.json
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

All writes are exclusive-create, atomic, and terminal-last. A failed or
invalid attempt remains immutable. No retry or alternate attempt is
preregistered.

## Manifest binding

`manifest.json` must bind before data-body/model/device access:

- source commit and hashes of model, runner, source config, runner config,
  authorization, and evaluator;
- family `cof_zdh_v3`, candidate D1, dataset, seed 4001, attempt_001, and the
  frozen optimizer/budget;
- frozen train/validation manifests, train transform, receiver vocabulary,
  amount contract, SamplingPlan, threshold, and metric hashes;
- frozen C0 and C1 manifest/sample/evaluation/gate/terminal hashes;
- H1 forensic commit and the three forensic file hashes;
- permanent H1 state `STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL`;
- byte-level fingerprints of the reused C1 causal backbone, conditioning,
  mask, amount, receiver, non-gap loss, optimizer, and budget contracts;
- a factor diff containing exactly `gap_decoder`;
- zero validation/internal-test/fraudTest fit rows and forbidden-access
  counters.

Any mismatch fails before runtime ownership is created. Existing C0/C1/H1
trees are references only and cannot be written, moved, deleted, or linked as
mutable output.

## Train-only gap support state

`train_only_gap_support_state.json` must record for each Y class:

- valid/zero/positive counts and ordered positive-`u` SHA-256;
- fixed 32-bin count and construction-version hash;
- all 33 numeric edges from `e_j=j*u_tail/32`;
- strict-exceedance `m_y`, threshold, exceedance count, and fixed mean excess
  `beta_tail`;
- zero/body/tail support convention and endpoint ownership;
- source, config, train manifest, transform, and construction-code hashes;
- validation, internal-test, and fraudTest rows used, all exactly zero;
- explicit statement that evaluator bins/support/thresholds were not read.

The state is invalid if any edge is non-finite or non-increasing, the scale
is non-finite/non-positive, strict exceedances are insufficient, or a
non-train dependency is present. There is no fallback.

## Checkpoint provenance

Each checkpoint repeats every manifest/support hash, actual update, elapsed
time, RNG state, exact module fingerprints, and optimizer-state schema. It
identifies the bounded-logit constant, hazard implementation version,
within-bin density, tail density, and fixed tail scale. Reload is strict.

It must prove that C1 non-gap state is unchanged and that no RQS state,
conditional exponential residual, pointer, hierarchy, structural loss, or
post-hoc calibration parameter exists.

## Per-step finite-state diagnostics

The runner must inspect every update even if normal progress is persisted less
frequently. `numerical_health.jsonl` records immutable summaries at least at
step 1, every progress/checkpoint step, and immediately on any failure:

- update and phase (`pre_forward`, `post_forward`, `post_loss`,
  `post_backward_pre_step`, `post_optimizer_step`);
- total/gap/amount/receiver loss values and finite booleans;
- backbone hidden-state finite count, non-finite count, min/max/abs-max;
- raw and bounded zero/hazard logit finite summaries;
- hazard, log-survival, body/tail route mass, mass-sum error, and fixed
  `beta_tail` summaries;
- within-bin width, log density, NLL, sampled `u`, and raw-gap summaries;
- global/per-module gradient norm, maximum absolute gradient, and named
  offending gradient tensors;
- parameter and optimizer-state finite/non-finite counts, maxima, and named
  offending tensors before and after optimizer step;
- most recent verified checkpoint path and SHA-256.

The checks themselves run every step. If a first invalid value appears,
`numerical_failure.json` is atomically exclusive-created with the exact step,
phase, tensor name, module, dtype, shape, finite/non-finite counts, available
range, loss state, gradient state, parameter state, prior checkpoint pointer,
and traceback. An invalid pre-step state prevents the optimizer step. An
invalid post-step state is attributed to that exact optimizer step. The
record cannot be overwritten or replaced by a generic exception.

## Validation and required diagnostics

Only after final-checkpoint completion may validation be opened for the fixed
SamplingPlan. The candidate artifact must preserve:

- classwise overall and positive-only gap KS;
- classwise exact-zero rates;
- classwise 32-bin and tail masses for real and synthetic values;
- hazard survival and body/tail route summaries by class and preregistered
  history stratum;
- within-bin conditional quantile differences and tail-excess summaries;
- route-mass normalization error, numerical-invalid count, and first-invalid
  pointer (null only when the count is zero);
- classwise coherence, classwise/full receiver TV, head/tail/UNK receiver
  diagnostics, amount diagnostics, and hard validity;
- validation/internal-test/fraudTest access and fit counters.

No invalid event, dataset, or class may be dropped or converted to NaN for an
aggregate.

## Gate and terminal semantics

`gate_decision.json` contains unrounded D1 values, frozen C0/C1 values, every
dataset/class boolean, numerical-invalid count, provenance hashes, and the
single conjunctive result.

- any failed condition: `STOP_COF_ZDH_V3_FAMILY_D1_GATE_FAIL`;
- all conditions pass:
  `COF_ZDH_V3_D1_VALIDATION_GATE_PASS_NO_TEST_AUTHORIZATION`.

COMPLETE means the authorized validation cell and artifact chain completed;
it does not imply gate PASS. PASS does not unlock internal test or any other
execution.
