# v2.7 source-only candidate preregistration

## Status and scope

This document freezes the v2.7 validation-only candidate definitions before
any v2.7 candidate result exists. The analysis parent is forensic commit
`906cfc7ae31fd244c5c5057afb3ed554e3e53fc8`.

This preparation does not authorize or perform GPU inventory queries, CUDA
access, candidate training, model fitting, model sampling, validation
selection, data generation, fresh-test work, TSTR, privacy analysis, or a
five-seed/full run. It creates no execution authorization. A later execution
requires a separate preregistered authorization and is outside this commit.

The executable definition is
`configs/benchmark_v2/selection_v2_7_source_preparation.yaml`, SHA-256
`2737723a05ce8628b7c9b8e1afbe7edae323a06971fbaf1eb00465224e0d654b`.

## Frozen evidence and conclusions

The following conclusions cannot be reinterpreted by v2.7:

- evaluator or selection-runner defect: **REFUTED**;
- numeric/sampling/decoder path limitation: **SUPPORTED**;
- current objective and candidate-space limitation: **SUPPORTED**.

The five guard thresholds, train-only SamplingPlan, endpoint, DGP, and C2
rule remain unchanged. Threshold relaxation, test-based calibration, and
result-contingent replacement or extension of this candidate family are
forbidden.

Principal frozen inputs are:

| Input | SHA-256 |
|---|---|
| v2.5 config | `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3` |
| v2.5 `FINAL_COMPLETE` | `e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a` |
| v2.5 frozen manifest | `b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05` |
| v2.6 single-factor config | `707ecdbd37bc55f4a3b9ece603b70d26c8274015aa9cb97d5d376aab3d9ecb7c` |
| v2.6 candidate authorization | `30b01268d0a94006bde3235b4c09f7a2fe44ec0b2e144b7b0577e6e45bba5adb` |
| v2.6 aggregate authorization | `35ef35ac3e6cfe92a9357438eafc8fc41fb08141fe04c2c0d2a87c0d51d72013` |
| v2.6 aggregate terminal | `404c820d79cb3d17e21d617f5c255c0d1b97ed6fb7c94b9fdcd23682bdc5bf35` |
| v2.6 forensic terminal | `0c2e89a4124d35ccfa565ab504fbedf8819bed2dfe649370f811efe51a408703` |

The candidate, worker, trajectory, evaluation, aggregate, and forensic tree
record hashes are pinned in the executable config. Plan and dry-run verify
them read-only.

## Frozen validation contract

| Item | Frozen value |
|---|---|
| scenario | `joint_semimarkov_v2b` |
| kappa | 1.0 |
| selection seed | 2601 |
| train file | `c67a6fce4593317e11e1b6f36397bfb38f9607916bf8f5b32329d8fa99b700a8` |
| validation file | `68c15c05c55742bfce0f22560ffe8378d032a7a9fd9a1072e80ec8da5340fba5` |
| SamplingPlan | `862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27` |
| amount KS | 0.006081138155655141 |
| gap KS | 0.006387882975686154 |
| amount effect | 0.0363693454591819 |
| gap effect | 0.051540527275560376 |
| receiver effect | 0.02 |

Eligibility and tie-breaking remain:

1. all five guards must PASS;
2. minimize `max(amount_ks, gap_ks)`;
3. minimize `amount_ks + gap_ks`;
4. lexicographic candidate ID.

No passing candidate for any primary model prohibits the v2.7 primary C2
full run. The test split cannot be used to fit, calibrate, select, or replace
a candidate.

## Finite single-factor candidate family

There are exactly nine evaluation candidates: one immutable control and two
single-factor candidates per model. There are zero new training
trajectories. A future non-control operation may evaluate a frozen checkpoint
only after separate authorization.

### CTGAN

Separate `y=0` and `y=1` generators and the frozen native 20k checkpoint
remain fixed.

| Candidate | Sole factor |
|---|---|
| `ctgan_v27_c00_frozen_standard` | none; reuse frozen standard result/sample |
| `ctgan_v27_c01_amount_quantile_inverse` | amount inverse map only |
| `ctgan_v27_c02_categorical_logit` | categorical marginal-logit calibration only |

The amount candidate fits a class-conditional 257-knot monotone empirical
quantile inverse map using valid train rows and the train-only SamplingPlan.
It clamps tails to train extrema. Categorical logits, temperature, class
handling, and model parameters remain unchanged.

The categorical candidate fits class/channel marginal log-probability
offsets from train and frozen train-plan output, with additive smoothing 0.5
and offsets clipped to `[-2, 2]`. It does not alter amount decoding.

### TVAE

The control is the preserved temperature-multinomial categorical decoder at
temperature 0.75 with native amount inverse decoder, 1/1/1 channel weights,
and the frozen native 20k checkpoint.

| Candidate | Sole factor |
|---|---|
| `tvae_v27_c00_frozen_temperature_0_75` | none; reuse frozen 0.75 result/sample |
| `tvae_v27_c01_amount_inverse_decoder` | amount inverse decoder only |
| `tvae_v27_c02_bounded_temperature` | categorical temperature rule only |

The amount candidate uses the same class-conditional 257-knot train-fitted
monotone empirical-quantile contract as CTGAN. Temperature, categorical
decoder, weights, class handling, and model parameters remain unchanged.

The categorical candidate chooses only temperature from the fixed grid
`[0.5, 0.625, 0.75, 0.875, 1.0]` using train-only gap/receiver marginal
frequency error. Ties choose the value closest to 0.75, then the lower value.
It does not change the amount decoder or channel weights.

### CoF-SeqGen

The control is the preserved variance-deficit residual sampler with the
frozen clean-x0 objective, architecture, receiver path, and native 20k
checkpoint.

| Candidate | Sole factor |
|---|---|
| `cof_v27_c00_frozen_variance_residual` | none; reuse frozen variance-residual result/sample |
| `cof_v27_c01_empirical_residual` | amount residual sampler only |
| `cof_v27_c02_gap_logit_bias` | gap-logit calibration only |

The amount candidate replaces only the Gaussian variance-deficit residual
with a class-conditional centered 257-knot empirical-quantile residual
sampler fitted from train. It uses deterministic inverse-ECDF resampling and
preserves train residual variance. Gap, receiver, architecture, and objective
remain fixed.

The gap candidate fits class-conditional gap marginal log-probability
offsets with smoothing 0.5 and clipping to `[-2, 2]`. The amount residual
sampler, receiver path, architecture, and objective remain fixed.

## Train-only fit-state contract

Every non-control candidate must store the following in its checkpoint or
candidate artifact:

- model, candidate ID, and sole factor;
- canonical parameter payload and its SHA-256;
- source commit and candidate config SHA-256;
- train file/content and SamplingPlan SHA-256;
- `fit_split=train`;
- zero validation and test rows used for fitting;
- destination (`checkpoint` or `candidate_artifact`).

The canonical state hash is recomputed before use. Any changed parameter,
source/config/data/plan provenance, test access, or hash mismatch fails
closed. Frozen controls create no fit state and perform zero training and
zero sampling.

## Future budget and stop condition

If separately authorized later, the six non-control evaluation operations
have a 1,800-second maximum each, for at most 3.0 GPU-hours total. This is a
ceiling, not an authorization or performance result. There are zero new
training trajectories.

Candidate extension, extra updates, threshold changes, test calibration, and
result-dependent replacement are prohibited. Selection and downstream
fresh-test/full work require later, separate authorization.

## Preparation CLI

`scripts.prepare_candidates_v2_7` exposes only `plan` and `dry-run`.
There is no execute, authorization, device, GPU, fit, sample, or selection
mode. Both modes write no runtime artifact. Dry-run rehashes frozen inputs,
reconstructs the train-only SamplingPlan, and reports all forbidden execution
counts as zero.
