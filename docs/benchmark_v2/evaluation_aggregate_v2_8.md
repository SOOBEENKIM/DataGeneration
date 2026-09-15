# v2.8 stored-evidence aggregate contract

## Status

The aggregate implementation that existed at source commit
`ea868cb0165efa0eb434dafdf99e071e31e21a9e` was specific to the v2.7
runtime root, three-worker/nine-candidate family, and v2.7 artifact schemas.
It could not safely aggregate the v2.8 two-worker/two-candidate family.
Consequently, the v2.8 aggregate authorization and aggregate runtime bundle
were not created at that commit. The source-only implementation in this
change must be committed and separately authorized before `execute` is
permitted.

## Frozen inputs

The input family is exactly:

- the stored CTGAN evaluation
  `ctgan_v28_c01_joint_gap_receiver_decoder`;
- the stored CoF-SeqGen evaluation
  `cof_v28_c01_gap_distribution_sampler`; and
- the hash-bound v2.7 selected TVAE reference
  `tvae_v27_c01_amount_inverse_decoder`.

The plan requires exactly two v2.8 candidate `COMPLETE` markers and exactly
two v2.8 worker terminal markers. It verifies each `COMPLETE` hash link,
manifest/config/checkpoint/SamplingPlan lineage, worker ownership lineage,
the candidate execution authorization, and the frozen TVAE candidate plus
the v2.7 aggregate evidence.

## Stored-decision boundary

The aggregate reads `evaluation.json` and `candidate_result.json`. It checks
that their stored status, five stored check decisions, and artifact hashes
are internally consistent. It does not open the validation samples and does
not recalculate any guard statistic or threshold comparison.

Eligibility remains all five stored row-marginal guard decisions equal to
`PASS`. The frozen tie-break is continuous KS maximum, continuous KS sum,
then candidate ID. The primary family is CTGAN, TVAE, and CoF-SeqGen.

## Authorization and append-only output

`plan` and `dry-run` need no authorization and create no runtime artifact.
`execute` requires an exact, source/hash/inventory-bound,
aggregate-only authorization under
`artifacts/benchmark_v2_8/authorizations/`. No code in this change creates
that authorization.

An authorized future execution may create only
`artifacts/benchmark_v2_8/candidate_selection/aggregate_attempt_001/`.
It writes the selection report, selection manifest, checksum manifest, and
artifact index using exclusive creation, then writes
`AGGREGATE_COMPLETE.json` last. An existing or partial output root is
fail-closed and is never overwritten.

GPU queries, CUDA, training, sampling, checkpoint changes, candidate
re-execution, guard recalculation, test access, fresh test, TSTR, privacy,
and five-seed/full runs are all outside the authorization schema and remain
forbidden.

## Source-only validation

The safe commands for this source-only phase are:

```bash
<COFSEQ_PYTHON> \
  -m scripts.aggregate_evaluation_selection_v2_8 --mode plan

<COFSEQ_PYTHON> \
  -m scripts.aggregate_evaluation_selection_v2_8 --mode dry-run
```

Neither command creates an authorization, aggregate attempt, or any model
or evaluation artifact. `execute` must not be used without a later explicit
authorization.
