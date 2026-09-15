# `cof_ccmtpp_v1` future artifact contract

This contract describes a possible later authorized execution. The execution
runner is implemented source-only; no runtime directory or authorization is
created by its preparation, plan, or dry-run.

## Append-only layout

```text
artifacts/cof_ccmtpp_v1/external_validation/
  workers/<dataset>/<candidate>/ownership.lock
  workers/<dataset>/<candidate>/attempt_NNN/WORKER_COMPLETE.json | WORKER_FAILED.json
  <dataset>/<candidate>/seed_4001/attempt_NNN/
  manifest.json
  receiver_vocabulary_state.json
  receiver_hierarchy_state.json
  gap_support_state.json
  amount_transform_state.json
  conditioning_plan.json
  progress.jsonl
  checkpoints/step_*.pt
  checkpoints/latest_step_*.json
  checkpoints/final.pt
  checkpoint_provenance.json
  validation_sample.npz
  diagnostics.json
  metrics.json
  evaluation.json
  runtime.json
  gate_decision.json
  artifact_index.json
  checksum_manifest.json
  COMPLETE.json | INVALID.json | FAILED.json
```

An attempt directory and every immutable file are created with exclusive
create semantics and are never overwritten, deleted, or moved.
Each process is scoped to exactly one `--dataset` and one candidate cell.
Authorization identity includes that dataset scope and only its input/C0
evidence. The AMLSim and Sparkov ownership and attempt paths are disjoint;
neither process can claim, resolve, or finalize the other dataset's subtree.
`manifest.json` is written before data/model/device access. The artifact index
and checksum manifest are completed before the single terminal marker, which
is itself exclusive-created last. The worker terminal is written only after
the candidate terminal is verified. Existing ownership or attempts are not
reused. A new attempt requires a new matching authorization. C0 contains only
hash-bound references to the frozen external non-v3 CoF evidence.

## Immutable manifest

The manifest records schema/family/candidate/parent, dataset, seed, source
commit and relevant-source hash, config hash, authorization hash, train and
validation manifest hashes, train-only transform/vocabulary/hierarchy hashes,
SamplingPlan and amount-contract hashes, requested updates, wall cap,
optimizer, candidate factor changes, parent terminal/evaluation hashes, and
all forbidden-access counters. Entity identifiers are absent.

Checkpoint provenance repeats source, config, train, transform, vocabulary,
hierarchy, SamplingPlan, and amount-contract hashes. It includes actual update
and elapsed time. A checkpoint with any mismatch is not resumable.

## Candidate diagnostics

Every candidate records the same schema:

- gap: total/Y=0/Y=1 NLL, finite-density count, sampled minimum/maximum;
- receiver: overall/head/tail/UNK/repeat/new NLL and all six counts;
- amount: normalized MSE and frozen decode-contract hash;
- fidelity: classwise gap KS, full/head/tail receiver TV, UNK-rate error,
  and short-gap × repeat error;
- runtime: requested/actual updates, elapsed seconds, and peak memory bytes.

Values are never dropped because a candidate fails. Non-finite, incomplete,
or provenance-inconsistent results are INVALID/FAILED, not silently omitted.

## Sequential finalization

A candidate terminal artifact is eligible for its stop criterion only when
all expected files have verified checksums. The next candidate may be
authorized only from a PASS terminal artifact of its immediate parent. A C1,
C2, C3, or C4 failure ends the chain. C5 does not have an executable schema
in this version. Finalization cannot change metrics, thresholds, vocabulary,
hierarchy, sampling, loss, seed, or budgets.

## Failure and interruption

Wall cap, exception, invalid support/mask, authorization mismatch, checkpoint
damage, or forbidden path access writes a terminal record. User interruption
is classified separately from a model exception. Startup and progress have
independent watchdogs, queue events are consumed before child-exit inference,
and child cleanup is bounded. No process outside the runner-owned child may be
signalled. Validation/internal-test/fraudTest reads during fit, hierarchy,
vocabulary, or threshold construction are contract failures.
