# Preregistered CoF-ZDH-v3 D1 validation study

## Frozen status

This document preregisters one future validation-only candidate. It does not
implement or authorize it. Parent source is
`a72cf11d13428908e4ec7de6aacbeb3055b7d085`.

The prior H1 family is permanently stopped at
`STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL`. Its attempt_003 forensic report
classifies AMLSim as a training-time H1 numerical-stability/architecture
limitation with insufficient evidence to identify the first internal tensor;
Sparkov completed and passed hard validity but failed the H1 scientific gate.
No H1 attempt_004 or H1 retry is allowed.

## Question and sole candidate

Can replacing only C1's gap decoder with an exact-zero category plus a
train-bounded, conditional discrete-hazard distribution over positive
log-gaps meet the frozen two-dataset gap gate without losing C1 coherence or
receiver fidelity?

There is exactly one candidate:

- D1: `cof_zdh_v3_d1`.

There is no D2, fallback density, alternate bin count, alternate tail rule,
loss-weight variant, seed variant, or hyperparameter sweep. C0 and C1 are
frozen references, not candidates.

## Factor isolation

D1 changes exactly `gap_decoder` relative to frozen CCMTPP-v1 C1. The causal
backbone, Y/L conditioning, masks, amount and flat receiver paths, loss
aggregation outside the gap term, optimizer, budget, seed, and SamplingPlan
are unchanged. Any additional difference makes D1 unauthorized and INVALID.

The D1 gap decoder is fixed before results:

- exact-zero categorical route;
- `u=log1p(gap)` for strictly positive gaps;
- 32 Y-specific equal-width body intervals from zero to a train-only
  strict-exceedance threshold;
- bounded conditional hazard logits `12*tanh(raw/12)`;
- uniform within-bin log-gap density;
- conditional tail mass from hazard survival;
- fixed train-only Y-specific exponential excess scale;
- no RQS, inverse root, learned exponential scale residual, clipping,
  fallback, redraw, or evaluator-derived interval.

## Frozen data-use boundary

AMLSim and Sparkov use their existing external protocol v1 frozen train and
validation bundles. Only valid train gaps may build D1 edges, threshold,
tail scale, support state, or initialize/fix any gap state. Train alone fits
model weights.

Validation is opened only after a final checkpoint to generate from the
already frozen dataset-specific Y/length SamplingPlan and calculate the
already frozen external metrics. It contributes zero rows to model fitting,
edge/support/tail construction, threshold calibration, vocabulary,
transforms, or candidate selection. Internal test and Sparkov fraudTest are
blocked before path resolution.

The validation evaluator's bin edges, support diagnostics, and thresholds
must not be imported by the D1 support builder. Controlled-benchmark
five-guard thresholds are not used.

## Frozen references

All comparisons are lower-is-better and use unrounded stored values.

| Dataset | Gate metric | Y=0 reference | Y=1 reference |
|---|---|---:|---:|
| AMLSim | C0 overall gap KS | 0.227623598 | 0.466188368 |
| AMLSim | C1 positive-only gap KS | 0.773860396 | not a D1 gate term |
| AMLSim | C1 coherence error | 0.096367415 | 0.046202532 |
| AMLSim | C1 receiver TV | 0.567449747 | 0.812395054 |
| Sparkov | C0 overall gap KS | 0.042308456 | 0.300651956 |
| Sparkov | C1 positive-only gap KS | 0.069522435 | not a D1 gate term |
| Sparkov | C1 coherence error | 0.000296870 | 0.000258866 |
| Sparkov | C1 receiver TV | 0.062882329 | 0.242978937 |

C1 full receiver TV is `0.562529727` for AMLSim and `0.062437900` for
Sparkov. These values and the C0/C1 artifact hashes are frozen in the source
config; they cannot be replaced after D1 is observed.

## Conjunctive D1 gate

Every condition below must pass for both AMLSim and Sparkov. Missing,
non-finite, provenance-mismatched, INVALID, or unavailable evidence is FAIL.

1. Hard validity PASS: Y/length/mask/padding, train receiver support, amount
   contract, train-only gap support, and all forbidden-access counters pass.
2. Numerical-invalid count is zero across train-state construction,
   training, checkpoint reload, sampling, and validation metrics.
3. For Y=0 and Y=1 separately, D1 overall gap KS is less than or equal to the
   corresponding frozen C0 value.
4. For Y=0, D1 positive-only gap KS is strictly less than the corresponding
   frozen C1 value.
5. For Y=0 and Y=1 separately, D1 coherence error is less than or equal to
   C1.
6. For Y=0 and Y=1 separately, and for full receiver TV, D1 receiver TV is
   less than or equal to C1.

The overall gate is the logical AND of every dataset/class/check. Partial
passing, dropped classes, rounding, averaging across datasets, or NaN
exclusion is forbidden.

Any failure freezes the new family at
`STOP_COF_ZDH_V3_FAMILY_D1_GATE_FAIL`. A PASS records
`COF_ZDH_V3_D1_VALIDATION_GATE_PASS_NO_TEST_AUTHORIZATION`; it does **not**
authorize internal test, Sparkov fraudTest, TSTR, privacy, a full run, or a
claim against IID/CTGAN/TVAE. Any such step requires a separate source-frozen
protocol and explicit authorization.

## Frozen budget

- datasets: AMLSim and Sparkov, independently;
- candidate: D1 only;
- seed: 4001;
- requested updates: 20,000;
- whole-cell wall cap: 7,200 seconds;
- batch size: 128;
- optimizer: AdamW;
- learning rate: 0.001;
- weight decay: 0.0001;
- checkpoint interval: 1,000 updates;
- early stopping, retry, sweep, update extension, checkpoint selection, and
  result-based change: forbidden.

## Current authorization state

Model implementation, execution runner, authorization, launch command,
runtime artifact, GPU/CUDA, data-body access, fit, sample, evaluation,
internal test, and Sparkov fraudTest are all unauthorized by this package.
