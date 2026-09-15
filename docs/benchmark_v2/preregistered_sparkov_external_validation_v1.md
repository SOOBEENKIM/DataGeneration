# Preregistered Sparkov validation continuation v1

Status: source-only preparation; execution is not authorized.

This amendment preserves the frozen AMLSim aggregate conclusion at commit
`c9516e55ed5169b95be853cf4055112bcd4e15f6`. In AMLSim, frozen non-v3 CoF
did not support a general superiority claim: it ranked behind empirical IID
and TVAE on the stored combined score and had the worst fidelity max-ratio.
Nothing in this Sparkov comparison changes or reinterprets that conclusion.

## Frozen scientific scope

Sparkov uses the already-materialized `fraudTrain` internal validation bundle.
The completed empirical-IID `attempt_002` is reused only after its manifest,
evaluation, terminal, source, config, frozen-bundle, train, validation,
SamplingPlan, threshold, metric-source, and seed hashes all match the pinned
record. The remaining comparison consists of exactly three new validation
jobs, all at seed 32001:

| Wave | Model | Attempt | Transform workers | Training cap | Whole-job cap |
|---|---|---|---:|---:|---:|
| 1 | CTGAN separate-class | `attempt_001` | 1 | 7,200 s | 10,800 s |
| 1 | frozen non-v3 CoF-SeqGen | `attempt_001` | 0 | 7,200 s | 10,800 s |
| 2 | TVAE separate-class | `attempt_001` | 1 | 7,200 s | 10,800 s |

CTGAN and CoF may run concurrently on distinct physical GPUs. Wave 2 cannot
begin until both wave-1 jobs have exactly one terminal marker. TVAE runs alone,
so CTGAN and TVAE cannot perform their heavy `DataTransformer` work
concurrently. CTGAN and TVAE use fixed `n_jobs=1`.

The model architectures, hyperparameters, seeds, conditioning, train-only
bootstrap metrics, selection formula, thresholds, frozen bundle, and non-v3
CoF fingerprint are unchanged. No result may be used to tune any of them.
Only `train.npz` may be used for fitting and only `validation.npz` may be used
for the preregistered validation evaluation. `internal_test.npz`, Sparkov
`fraudTest`, raw CSV, TSTR, privacy, full/five-seed execution, and GPU inventory
queries are outside this scope.

## Preregistered CoF-family stopping rule

All scores are lower-is-better. After all four Sparkov validation results have
terminal, hard-valid artifacts, continuing the existing CoF family requires
both strict inequalities:

1. `CoF selection.combined_score < TVAE selection.combined_score`
2. `CoF selection.fidelity_max_ratio < TVAE selection.fidelity_max_ratio`

A tie is not superiority. If either inequality fails, or either result is not
COMPLETE and hard-valid, the decision is
`STOP_EXISTING_COF_FAMILY_MODEL_LEVEL_REDESIGN_REQUIRED`. No additional
sampler/decoder calibration, threshold relaxation, post-result candidate, or
training extension is permitted. Work may resume only under a separately
preregistered model-level redesign.

If both inequalities hold, the decision is only
`CONTINUE_TO_SEPARATE_TEST_PROPOSAL`. It does not authorize internal test,
fraudTest, TSTR, privacy, or any other execution. A separate preregistration
and explicit user authorization would still be required.

## Source-only authorization state

The preparation code builds and validates a deterministic authorization
template, but deliberately does not write it. The template has
`execution_authorized=false`, binds the current source/config/runner hashes,
pins the completed IID evidence, contains exactly the three learned jobs above,
and carries the stopping rule verbatim. This commit creates no runtime or
authorization artifact and supplies no executable GPU launch command.
