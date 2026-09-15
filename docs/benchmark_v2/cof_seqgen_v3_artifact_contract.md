# CoF-SeqGen v3 future artifact contract

## Source-preparation phase

No v3 runtime root, authorization, checkpoint, sample, evaluation, or
terminal artifact may be created in this phase. This document defines the
contract a later, separately authorized validation-only runner must satisfy.

The later runner source is now implemented, but this remains a source-only
commit: its `execute` mode is unreachable without a separate exact,
hash-bound authorization. Plan and dry-run create no runtime artifact.

All v2.5–v2.8 data, runtime, checkpoints, authorizations, aggregates, and
forensic artifacts are immutable read-only inputs. The frozen reference
`cof_v3_ref_v28_frozen` remains a hash reference and is never copied,
retrained, or resampled.

## Future append-only layout

```text
artifacts/benchmark_v3/candidate_selection/
  workers/
    <candidate_id>/
      ownership.lock
  candidates/
    <candidate_id>/
      seed_3001/
        attempt_001/
```

The implemented runner uses one candidate attempt containing the trajectory
and its validation evaluation so the parent wall cap covers both. Every
attempt is exclusive-create and append-only. A hash mismatch or failed
attempt requires the next attempt number; no prior file is overwritten,
moved, or deleted.

## Immutable trajectory manifest

Before a future training child starts, `manifest.json` must bind:

- source commit and relevant source SHA-256;
- v3 config path and SHA-256;
- candidate ID and architecture;
- scenario, kappa, seed, requested updates, optimizer, batch size, and wall
  cap;
- frozen train file/content SHA-256;
- SamplingPlan SHA-256 and Y/length/valid-mask binding SHA-256;
- train-only joint-support state and SHA-256;
- frozen amount-contract payload and SHA-256;
- five thresholds and unchanged selection rule;
- authorization path and SHA-256; and
- validation/test rows used for fitting, both fixed to zero.

The checkpoint schema repeats source, config, train, plan, support, and
amount-contract hashes. It also records candidate architecture, requested
and actual updates, elapsed wall time, optimizer state, model state, random
states, and the paired-mask contract. A factorized checkpoint explicitly
records true-gap teacher conditioning.

## Training artifacts

A later authorized trajectory writes atomically:

1. immutable `manifest.json`;
2. `joint_support_state.json`;
3. `conditioning_binding.json`;
4. `progress.jsonl`;
5. periodic `checkpoints/step_*.pt` and atomic `checkpoints/latest`;
6. stdout and stderr logs;
7. `runtime.json`;
8. final checkpoint and its SHA-256; and
9. `COMPLETE.json` or `FAILED.json` last.

The hard wall cap covers the entire candidate operation. Any missing,
non-finite, corrupt, or provenance-mismatched state fails closed. A terminal
marker is never inferred from process exit alone.

## Validation artifacts

Only a later validation-only authorization may permit:

1. immutable evaluation `manifest.json`;
2. read-only restore of the matching v3 checkpoint;
3. validation sample generation under the frozen SamplingPlan;
4. `validation_sample.npz`;
5. five-guard `evaluation.json`;
6. joint/coherence diagnostics;
7. `candidate_result.json`; and
8. `COMPLETE.json` or `FAILED.json` last.

The manifest and terminal marker bind checkpoint, sample, evaluation,
diagnostic, config, source, train, validation, SamplingPlan, support, amount
contract, and authorization hashes. Test access is false. Post-sample gap or
receiver mutation is forbidden.

`COMPLETE` is reserved for all-five PASS. A fully evaluated candidate with
one or more guard failures is `INVALID`, not silently excluded. A wall-cap,
code, or infrastructure failure is `FAILED`.

## Selection finalization

Selection is a separate append-only operation and cannot run until both
trainable candidates are terminal. It uses only stored validation evidence.
Eligibility requires all five guards PASS, followed by the frozen v2.8
continuous-KS tie-break. INVALID or missing candidates are not silently
dropped.

No fresh-test, TSTR, privacy, or full-run authorization can be derived merely
from artifact completion. It requires a passing selected candidate and a
separate explicit user authorization.

## Fail-closed conditions

The future runner must reject:

- source/config/train/content/plan/support/amount hash mismatch;
- validation or test participation in any fit state;
- test or fresh-test path access;
- an independent gap/receiver corruption or decode;
- a v2 `bin_head`/receiver `cat_head` in a v3 candidate;
- gap/receiver marginal calibration or post-sample transport;
- nonzero coherence weight;
- changed thresholds, selection rule, seed, budget, or update count;
- writes outside the v3 append-only root; and
- any attempt to train, sample, or copy the frozen v2.8 reference.
