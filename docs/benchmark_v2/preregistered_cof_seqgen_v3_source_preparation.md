# Preregistered CoF-SeqGen v3 source preparation

## Status

This document preregisters source-only implementation. It does not authorize
GPU inventory queries, CUDA, model fit, optimizer updates, checkpoint writes,
sampling, validation execution, selection, fresh test, TSTR, privacy analysis,
or five-seed/full execution. No execution authorization or launch command is
created in this phase.

The v2.8 result remains frozen: CTGAN and TVAE pass all five guards, CoF fails
receiver signed frequency at `0.021126555312304892 > 0.02`, and
`primary_c2_selection_ready=false`.

## Frozen inputs

The implementation is bound to:

- scenario `joint_semimarkov_v2b`, kappa `1.0`;
- the v2.5 frozen train and validation files;
- SamplingPlan SHA-256
  `862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27`;
- the unchanged v2.8 five-guard thresholds and selection rule;
- the frozen train-only centered empirical-residual amount contract with a
  257-point quantile grid;
- `coherence_lambda=0.0`;
- training seed `3001`, 20,000 requested updates, batch size 256, Adam
  learning rate `0.001`, and a 7,200-second hard wall cap; and
- the v2.8 CoF candidate as a read-only failed reference.

Validation and test cannot contribute to transform, support, SamplingPlan, or
model fitting. Test and fresh-test paths fail closed.

## Finite candidate family

Exactly three candidate records exist:

1. `cof_v3_ref_v28_frozen` is a hash reference to the frozen v2.8 artifact.
   It performs no training, sampling, or validation.
2. `cof_v3_c01_direct_joint` encodes
   `z = gap_bin * K_receiver + receiver_code`, uses one joint embedding, one
   joint MASK token, and one Cartesian joint head.
3. `cof_v3_c02_factorized_joint` represents
   `p(gap, receiver | h, Y)` as
   `p(gap | h, Y) p(receiver | gap, h, Y)`. Training conditions the receiver
   head on true gap; sampling conditions it on the gap sampled immediately
   before receiver.

Both trainable candidates use one paired corruption mask. Neither may retain
the v2 independent `bin_head` and receiver `cat_head`, and neither may invoke
gap/receiver transport, marginal-logit bias, temperature selection, or any
other post-sampling calibration.

## Retained architecture and objective

Both trainable candidates retain the non-causal Transformer sequence
backbone, learned position embedding, diffusion-time embedding, entity-label
CFG embedding, train-derived target lengths, valid-prefix/padding mask, VP
cosine amount corruption, clean-amount MSE, and fixed amount residual
contract.

The direct candidate uses joint cross-entropy. The factorized candidate uses
gap cross-entropy plus conditional receiver cross-entropy. Amount,
discrete, and label loss weights are fixed to `1.0`. The joint support mask
is fit from valid train rows only and its canonical hash is checkpoint and
artifact provenance.

## Sampling and support

Future sampling, if separately authorized, starts each valid discrete
position at one joint MASK. Direct-joint sampling makes one Cartesian-state
choice and decodes it. Factorized sampling chooses gap and then receiver
conditioned on that exact gap. Temperature is fixed at `1.0`. Padding is
canonical zero and never contributes to loss. A global train-observed
joint-support mask is applied without class-, validation-, or test-specific
adjustment.

## Validation eligibility and selection

The five hard guards and thresholds remain:

| Guard | Threshold |
|---|---:|
| amount KS | `0.006081138155655141` |
| gap KS | `0.006387882975686154` |
| amount absolute standardized label effect | `0.0363693454591819` |
| gap absolute standardized label effect | `0.051540527275560376` |
| receiver maximum absolute signed frequency | `0.02` |

Eligibility requires all five PASS. Among eligible candidates, the unchanged
v2.8 tie-break is:

1. lower `max(amount KS, gap KS)`;
2. lower `amount KS + gap KS`; and
3. lexicographic candidate ID.

Joint total variation, conditional receiver total variation,
`joint_alignment`, fanout, and development association recovery are recorded
diagnostics only. They do not change eligibility or tie-breaking.

If neither trainable candidate passes all five guards, fresh test and all
later model execution remain forbidden. No new candidate, threshold, budget,
or calibration is generated from that failure.

## Source-only runner

`scripts.prepare_cof_seqgen_v3` exposes only `plan` and `dry-run`. It has no
execute, authorization, or device option. The plan reports one frozen
reference and two future trajectories. Dry-run verifies the specification,
config, frozen v2.5 files, v2.8 configs and aggregate, the frozen CoF
reference, and the preserved v2.8 candidate tree.

Both modes report zero calls for GPU query, CUDA, fit, optimizer update,
checkpoint write, sample, validation, selection, test, TSTR, privacy, and
five-seed/full execution. Neither mode creates authorization or runtime
artifacts.

## Prohibited amendments

Threshold relaxation, validation/test-based fitting, class-specific
post-hoc transport, additional sampler/decoder calibration, update increase,
early stopping, architecture sweep, result-dependent candidate creation, and
history or frozen-artifact modification are forbidden.
