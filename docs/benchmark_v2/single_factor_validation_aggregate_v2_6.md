# v2.6 single-factor validation-only aggregate

## Scope

This aggregate consumes only the nine immutable single-factor candidate
artifacts produced for CTGAN, TVAE, and CoF-SeqGen. It does not query a GPU,
initialize CUDA, fit or sample a model, generate data, read a test split, or
authorize fresh-test, TSTR, privacy, five-seed, or full execution.

The candidate source state, execution authorization, worker terminals,
whole-model tree digests, config, frozen train/validation data, train-fitted
SamplingPlan, and v2.5/v2.6 provenance are hash-bound by a separate
append-only aggregate authorization.

## Plan, dry-run, and execute

The aggregate-only CLI is:

```text
python -m scripts.aggregate_single_factor_selection_v2_6
```

- `--mode plan` verifies the authorization, three exact
  `attempt_001/WORKER_COMPLETE.json` files, nine candidate terminals, and all
  source/config/artifact hashes. It does not load candidate samples or write
  artifacts.
- `--mode dry-run` additionally loads only frozen train/validation and the
  nine stored validation samples, recomputes the five guards, and applies the
  preregistered selection rule. It writes nothing.
- `--mode execute` repeats the same checks and exclusive-creates one new
  `aggregate_attempt_NNN`. `AGGREGATE_COMPLETE.json` is written last.

The CTGAN finalization correction does not authorize or require a recovery.
Its original `attempt_001/WORKER_COMPLETE.json` is the approved terminal.
No `attempt_002`, retraining, or resampling is permitted.

## Frozen selection rule

A candidate is eligible only when all five frozen guards pass:

- amount KS;
- gap KS;
- absolute standardized amount class effect;
- absolute standardized gap class effect;
- maximum absolute signed receiver frequency.

Eligible candidates are ordered by:

1. minimum `max(amount_ks, gap_ks)`;
2. minimum `amount_ks + gap_ks`;
3. lexicographic candidate ID.

No threshold, validation split, SamplingPlan, DGP, candidate result, or
candidate sample can be changed by the aggregate.

Each model is recorded as `SELECTED` or `NO_PASSING_CANDIDATE`. If any of the
three primary models has no passing candidate, the selection manifest remains
`PRIMARY_SELECTION_FAILED_NO_FRESH_TEST`, with fresh-test, TSTR, privacy,
five-seed, and full execution all false. If all three are selected, the
manifest is only `FROZEN_AWAITING_SEPARATE_FRESH_TEST_AUTHORIZATION`; it does
not itself authorize a fresh test.

## Append-only outputs

```text
artifacts/benchmark_v2_6/selection_single_factor/
  aggregate_attempt_NNN/
    aggregate_readiness.json
    selection_report.json
    selection_manifest.json
    checksum_manifest.json
    artifact_index.json
    AGGREGATE_COMPLETE.json
```

All files use exclusive create. A pre-existing aggregate attempt is rejected.
The artifact index covers the readiness, report, manifest, and checksum
manifest; the final terminal binds their hashes and is created last.

## Source validation

- focused aggregate/runner/selection tests: 83 passed;
- repository-wide tests: 290 passed, 23 warnings;
- `compileall`: PASS;
- `git diff --check`: PASS.

The warnings are the existing third-party CTGAN deprecation and PyTorch
transformer warnings; no aggregate test invokes a production GPU, CUDA,
candidate fit, candidate sample, data-generation, test, or full-run path.
