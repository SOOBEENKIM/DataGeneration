# AMLSim TVAE attempt_002 authorization plan

Status: source preparation for one append-only continuation. This document does
not authorize Sparkov or any test-stage operation.

## Exact scope

The authorization preparation CLI creates exactly one grant:

- dataset `amlsim`;
- model `tvae_separate_class`;
- seed 31001;
- append-only `attempt_002`;
- 20,000 requested total updates under the unchanged separate-class budget;
- 7,200-second training cap and 10,800-second whole-job cap;
- one user-selected physical GPU exposed through `CUDA_VISIBLE_DEVICES` and
  runner device `cuda:0`;
- fixed `DataTransformer` `n_jobs=1` synchronous in-process transform; and
- train-only fit/bootstrap followed by validation generation and the frozen
  external fidelity/coherence evaluation.

The grant explicitly forbids Sparkov, internal test, Sparkov fraudTest, raw CSV,
GPU inventory discovery, aggregate selection, TSTR, privacy, full/five-seed
execution, threshold or selection-rule changes, architecture/hyperparameter
changes, automatic retry, sweep, and mutation of any existing artifact.

## Bound read-only evidence

The authorization includes exact manifest/evaluation/COMPLETE hashes for
AMLSim IID `attempt_002`, CTGAN `attempt_001`, and frozen non-v3 CoF
`attempt_001`. It also includes the complete tree, manifest, FAILED marker, and
runner-log hashes of TVAE `attempt_001`. Every hash is recomputed by the
validator before `attempt_002` can be claimed. A mismatch, existing
`attempt_002`, changed source/config/runner, or broadened permission fails
closed.

The future attempt manifest carries these reuse and failure-preservation
records, the authorization hash, source/config/execution hashes, frozen
bundle/train/validation hashes, metric source, memory policy, seed, budget, and
append-only target path.

## Preparation and launch boundary

`scripts.prepare_amlsim_tvae_continuation_v1` provides `plan`, `dry-run`, and
exclusive `create`. Plan and dry-run perform no NPZ body load, GPU query, CUDA,
model import, fit, sample, validation calculation, or artifact write. Create
writes only one authorization JSON beneath:

```text
artifacts/external_validation_v1/amlsim_tvae_attempt_002_authorization_history/
```

The source commit must be completed before create so that the authorization is
bound to the actual executable HEAD and hashes. The user launch is a single
tmux process on one selected idle GPU; this preparation never selects or
queries that GPU itself.
