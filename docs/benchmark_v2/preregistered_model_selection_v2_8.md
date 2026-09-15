# Preregistered v2.8 single-factor validation amendment

## Status and scope

This amendment is source-only preparation. Its frozen parent is forensic
commit `de7548d4b22610203be0d320cd25f9491b25aee6`.

The amendment does not authorize or perform a GPU/CUDA query, training,
checkpoint update, model restore, sampling, evaluation, selection, fresh
test, TSTR, privacy analysis, or five-seed/full run. No v2.8 runtime root or
authorization exists at this stage.

The frozen v2.7 conclusion remains:

- TVAE `tvae_v27_c01_amount_inverse_decoder`: all-five-guard PASS and the
  immutable selected control;
- CTGAN `ctgan_v27_c01_amount_quantile_inverse`: both amount guards PASS,
  while gap KS and receiver frequency FAIL;
- CoF `cof_v27_c01_empirical_residual`: both amount guards and receiver
  frequency PASS, while gap KS and gap class effect FAIL;
- `primary_c2_selection_ready=false`.

The versioned amendment config is
`configs/benchmark_v2/selection_v2_8_source_amendment.yaml`, SHA-256
`9e8ecaeab15f28ec05cee27150f4c6f76142dce661536094f41a798a83a7ae98`.
The relevant source hash is
`8bea7546389a43271cec595dcdccceaf57d2e68bf5fe801eba54f6dfbf12897e`.

## Frozen validation contract

| Item | Frozen value |
|---|---|
| Scenario | `joint_semimarkov_v2b` |
| Kappa | 1.0 |
| Future selection seed | 2801 |
| Train file | `c67a6fce4593317e11e1b6f36397bfb38f9607916bf8f5b32329d8fa99b700a8` |
| Validation file | `68c15c05c55742bfce0f22560ffe8378d032a7a9fd9a1072e80ec8da5340fba5` |
| Train-only SamplingPlan | `862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27` |
| Endpoint | `continuous_association_recovery_error` |
| DGP | `benchmark_v2_joint_semimarkov_v2b_kappa_1_00_frozen` |
| C2 rule | unchanged v2.7 three-primary-model rule |

The five guard thresholds remain:

| Guard | Threshold |
|---|---:|
| Amount KS | 0.006081138155655141 |
| Gap KS | 0.006387882975686154 |
| Amount effect | 0.0363693454591819 |
| Gap effect | 0.051540527275560376 |
| Receiver effect | 0.02 |

Test access, threshold relaxation, result-based tuning, and combining more
than one changed factor are fail-closed.

## Finite candidate family

There are exactly five evaluation entries: three immutable parent references
and two future evaluation-only single-factor candidates. There are zero new
training trajectories.

### CTGAN

The parent is `ctgan_v27_c01_amount_quantile_inverse`. Its frozen native
20,000-update checkpoint and train-fitted amount inverse remain unchanged.

| Candidate | Execution | Sole changed factor |
|---|---|---|
| `ctgan_v28_c00_frozen_amount_inverse` | reference only | none |
| `ctgan_v28_c01_joint_gap_receiver_decoder` | future evaluation-only | `joint_gap_receiver_discrete_decoder` |

The new candidate adds one train-only class-conditional joint
`(gap_bin, receiver)` logit-residual decoder calibration. The target is the
valid train rows; the base distribution is output from the frozen parent
checkpoint on the fixed train-only SamplingPlan. It uses additive smoothing
0.5, logit offsets clipped to `[-2, 2]`, and only joint pairs observed in
train support.

This is one discrete-decoder factor. The amount inverse map, categorical
temperature, separate-class handling, checkpoint, and every other path are
fixed.

### CoF-SeqGen

The parent is `cof_v27_c01_empirical_residual`. Its frozen native
20,000-update checkpoint, empirical amount-residual sampler, architecture,
and clean-x0/DDIM objective remain unchanged.

| Candidate | Execution | Sole changed factor |
|---|---|---|
| `cof_v28_c00_frozen_empirical_residual` | reference only | none |
| `cof_v28_c01_gap_distribution_sampler` | future evaluation-only | `gap_distribution_sampler` |

The new candidate replaces only the gap sampling map with a train-only
class-conditional monotone gap-distribution transport. Its base distribution
is output from the frozen parent checkpoint on the fixed train-only
SamplingPlan; its target is valid train rows. It uses frozen train gap
support, discrete monotone quantile transport, and lower-gap-bin tie
breaking.

### TVAE

`tvae_v27_c01_amount_inverse_decoder` is referenced as
`tvae_v28_c00_frozen_selected_amount_inverse`. It is not trained, restored,
resampled, reevaluated, or replaced. No new TVAE candidate exists.

## Train-only fit-state rule

Each of the two new candidates must eventually record:

- model, candidate, parent candidate, and sole factor;
- the exact fit parameters and canonical state hash;
- fit split `train`;
- validation rows used for fit: 0;
- test rows used: 0;
- source commit and config hash;
- train file/content and SamplingPlan hashes;
- parent manifest, checkpoint, validation sample, and source hashes.

Validation is used only after future candidate generation to calculate the
unchanged five guards. It cannot fit or revise calibration state.

## Future evaluation-only rule

A later separately authorized runner may:

1. read the existing frozen 20,000-update checkpoint;
2. fit the one declared calibration state from train only;
3. generate a validation sample;
4. evaluate the five frozen guards.

It must record:

- checkpoint access `READ_ONLY`;
- optimizer updates: 0;
- training calls: 0;
- checkpoint writes: 0.

Any mismatch in parent checkpoint, sample, manifest, COMPLETE marker,
candidate result, evaluation, fit state, source, config, authorization, train
data, or SamplingPlan fails closed. No current command performs these future
steps.

## Future selection rule

The all-five-guard requirement and existing tie-break remain unchanged.
TVAE is the immutable selected control. CTGAN and CoF each have exactly one
new candidate. If either new primary candidate fails any guard, v2.8 primary
C2 work remains prohibited. Fresh test and all downstream work always require
a separate later amendment and authorization.

## Source-only CLI

Only the following modes exist:

```text
python -m scripts.prepare_candidates_v2_8 --mode plan
python -m scripts.prepare_candidates_v2_8 --mode dry-run
```

There is no execute, authorization, device, GPU, training, sampling,
evaluation, selection, or test option.

## Source-preparation verification

- source-only plan: PASS;
- source-only dry-run: PASS, 3 parents and 24 parent artifacts verified;
- focused v2.8/v2.7 contract tests: 50 passed;
- repository-wide tests: 347 passed with 23 existing third-party warnings;
- `compileall`: PASS;
- `git diff --check`: PASS;
- authorization and runtime artifact creation: 0.
