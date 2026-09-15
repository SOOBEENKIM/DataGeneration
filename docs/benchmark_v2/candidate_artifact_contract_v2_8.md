# v2.8 evaluation-only candidate artifact contract

## Current status

This document defines a future contract; it does not create runtime
artifacts or authorize execution. The intended future root is
`artifacts/benchmark_v2_8/candidate_selection/`.

All v2.5, v2.6, and v2.7 runtime, frozen data, configs, authorizations,
checkpoints, samples, aggregate bundles, and forensic evidence are immutable
read-only inputs.

## Frozen parent references

| Model | Frozen parent | Checkpoint SHA-256 | Validation sample SHA-256 |
|---|---|---|---|
| CTGAN | `ctgan_v27_c01_amount_quantile_inverse` | `7e9de7e609ad48e8de687426240fe4fd0a47b5d2d0abcbc70a4edb58f182d10d` | `f1e0634576d4b2e65de30138f87c8d8999d18f5969a700921f06dac4f23d7a49` |
| TVAE | `tvae_v27_c01_amount_inverse_decoder` | `6f5263a5a40d9513342b33a823a02769bab65f6e1a5bd38c814b87d771a2947c` | `57ed7595d6dc9f83a8c9679606cb18d592ceac935f7d1aef2c0356cd1339c90e` |
| CoF | `cof_v27_c01_empirical_residual` | `7ce44721cadffef21ade3bf1884421c304f73ea9eff1e9bfb3ca2d9e63cc0c29` | `364b67ce239b9f692183af00e5f496002d73c9e97d5679d06a7ec168bd8cb2c0` |

Dry-run verifies, for each parent, the immutable manifest, COMPLETE marker,
candidate result, evaluation, fit state, validation sample, checkpoint, and
execution authorization. It also verifies zero optimizer updates/training,
train-only parent fit state, validation-only evaluation, unchanged
SamplingPlan, and no test read.

## Future path shape

If separately authorized later, each new candidate must use:

```text
artifacts/benchmark_v2_8/candidate_selection/
  evaluations/
    <model_id>/
      <candidate_id>/
        seed_2801/
          attempt_001/
```

An existing attempt is never overwritten. A provenance mismatch or failed
attempt requires `attempt_002`; the earlier tree remains intact. Model-level
ownership must be exclusive, and writes outside the v2.8 candidate-selection
root are rejected.

The three frozen control entries are references. They create no validation
sample, fit state, checkpoint, or copied runtime tree.

## Immutable future attempt manifest

Before a future evaluation-only operation, `manifest.json` must bind:

- source commit and relevant source hash;
- v2.8 config path and SHA-256;
- scenario, kappa, seed, model, candidate, and sole factor;
- frozen parent candidate and all eight verified parent artifact hashes;
- frozen train file/content and SamplingPlan hashes;
- frozen validation file/content hash;
- five thresholds and evaluation version;
- checkpoint access `READ_ONLY`;
- optimizer updates, training calls, and checkpoint writes all equal to 0;
- test access false.

Manifest creation must be exclusive and append-only.

## Train-only fit state

The two new candidates require `fit_state.json` with schema
`benchmark-v2.8-train-fit-state-v1`. It records:

- candidate, model, and sole changed factor;
- exact parameter payload and canonical SHA-256;
- fit split `train`;
- validation and test rows used: 0;
- source/config/train-content/SamplingPlan provenance;
- parent candidate, checkpoint, validation-sample, manifest, and source
  provenance;
- canonical full-state SHA-256.

The state is stored with the candidate artifact, not in the frozen
checkpoint. A control cannot create fit state.

## Future output ordering

Only a later authorization may permit the following append-only files:

1. immutable `manifest.json`;
2. train-only `fit_state.json`;
3. generated `validation_sample.npz`;
4. `evaluation.json`;
5. `candidate_result.json`;
6. `COMPLETE.json` last.

`COMPLETE.json` must bind the SHA-256 of every preceding artifact.
`FAILED.json` is required for a failed future attempt and must preserve the
last completed state. No marker-less exit is allowed.

TVAE is a frozen selected reference and cannot enter this output sequence.

## Fail-closed conditions

The future operation is rejected if any of the following occurs:

- test or fresh-test path access;
- threshold or SamplingPlan change;
- changed parent checkpoint/sample/provenance;
- optimizer update, training call, or checkpoint write;
- more than one changed factor;
- CTGAN amount-inverse or other non-discrete-path drift;
- CoF amount-residual, receiver, architecture, or objective drift;
- TVAE restore, sampling, evaluation, or new-candidate request;
- write outside the append-only v2.8 subtree;
- source/config/data/plan/parent state hash mismatch.

Aggregation, fresh test, TSTR, privacy, and five-seed/full work are outside
this contract and require later, separate authorization.
