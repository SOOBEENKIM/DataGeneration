# `cof_ccmtpp_v1` authorization-bound execution runner

## Frozen source-only status

This commit implements the future execution path but does not authorize or
run it. No authorization file, runtime root, checkpoint, sample, evaluation,
GPU inventory query, CUDA call, fit, optimizer update, internal-test read, or
Sparkov `fraudTest` path resolution is created by plan, dry-run, tests, or this
commit. The machine contract is
`configs/benchmark_v2/cof_ccmtpp_v1_execution_runner.yaml`.

The CLI has exactly three modes:

```text
python -m scripts.run_cof_ccmtpp_v1 --mode plan --candidate C1 --dataset amlsim
python -m scripts.run_cof_ccmtpp_v1 --mode dry-run --candidate C1 --dataset sparkov
python -m scripts.run_cof_ccmtpp_v1 --mode execute --candidate C1 --dataset <dataset> \
  --authorization <future-append-only-authorization.json>
```

The first two commands do not import the model or execution backend and do
not open a train or validation array body. The third validates the
authorization path, exact document identity, source/config hashes, scope, C0
evidence, and explicit candidate approval before lazy-importing the backend or
touching data, a model, or CUDA. This document intentionally provides no
authorization and no launch command.

`--dataset {amlsim,sparkov}` is an explicit worker scope in all three modes
and is mandatory for execute. One process owns exactly one dataset/candidate
cell. The plan, authorization claim, C0 evidence, input hashes, ownership
lock, attempt, and terminal records therefore contain exactly that dataset.
An AMLSim authorization cannot validate a Sparkov plan, and conversely. A
scoped process rejects any request to resolve the other dataset before a
split path is opened.

## Candidate unlock state machine

C1 is the only initial candidate. A direct C2, C3, or C4 request cannot execute
without a separate authorization containing checksum-verified `COMPLETE` and
gate `PASS` evidence from its immediate parent. The runner checks the parent
terminal, artifact index, checksum manifest, gate decision, checkpoint
provenance, source, runner/source configs, train split, train transform,
train-only threshold, and fixed validation SamplingPlan hashes.

```text
C0 frozen reference -> C1 PASS -> C2 PASS -> C3 PASS -> C4
```

A valid scientific candidate whose preregistered stop criterion fails remains
`COMPLETE` with `gate_decision.status=FAIL`; it cannot unlock the next
candidate. A hard mask/support failure is `INVALID`. An exception, watchdog,
wall cap, interruption, corrupt provenance, or forbidden access is `FAILED`.
There is no retry, sweep, early stopping, threshold change, or result-based
configuration change.

Factor isolation is checked again in the spawned execution process:

- C1: causal decoder, continuous log-gap density, and flat receiver decoder;
- C2: C1 plus sampled-gap-conditioned pointer copy only;
- C3: C2 with only its new-receiver branch replaced by the deterministic
  train-only head/tail/UNK hierarchy;
- C4: C3 plus equal-class conditional-likelihood aggregation only.

All other architecture, optimizer, loss, seed, update, batch, checkpoint, wall
cap, amount, hierarchy, vocabulary, threshold, and sampling settings must equal
the preregistration.

## Split and model boundary

For each dataset, the worker first verifies and opens only `train.npz`. The
receiver vocabulary, hierarchy, decoded continuous-gap support, and amount
transform binding contain zero validation, internal-test, and fraud-test rows.
The model has no entity-ID input. The frozen validation SamplingPlan and
`validation.npz` are opened only after training and final checkpoint creation,
for generation and evaluation. `internal_test`, `fresh_test`, `fraudTest`, and
equivalent path tokens are rejected before path resolution.

AMLSim and Sparkov frozen bundles store the train-only gap representation as
`dt_bin` plus the immutable train-fitted `gap_tau` inverse representatives.
The continuous head is trained on those nonnegative decoded gap values; it is
not a gap-bin classifier. Its upper support is the maximum valid decoded train
gap. The generated continuous gap is retained in the sample artifact and is
mapped through the unchanged train-only `gap_edges` only for the existing
external fidelity/coherence evaluator.

The validation evaluator reuses the existing train-bootstrap threshold and
fixed SamplingPlan artifacts by hash. It preserves classwise gap KS, full and
head/tail/UNK receiver fidelity, short-gap × repeat coherence, and exact
receiver overall/head/tail/UNK/repeat/new likelihood strata. It does not copy
the controlled-benchmark five-guard threshold and does not use validation to
fit state.

## Process and failure safety

Every dataset/candidate cell is owned by one parent-created append-only lock
and attempt. AMLSim and Sparkov use disjoint
`workers/<dataset>/...` and `<dataset>/.../attempt_001` namespaces, so two
separately authorized processes may safely run with different single-device
`CUDA_VISIBLE_DEVICES` mappings while each runner sees only `cuda:0`. The
runner itself does not query GPU inventory. The parent uses multiprocessing
`spawn`. The child first emits a
bootstrap handshake; setup and progress then have separate monotonic
watchdogs. Training emits progress at step 1 and every 100 updates, while
checkpoints are fixed at every 1,000 updates. The whole train, sample, and
validation-evaluation cell is bounded by 7,200 seconds.

The parent consumes the queue before inspecting child exit. On timeout it
sends `SIGTERM` only to its own child, waits 10 seconds, and uses `SIGKILL`
only for that same child followed by a five-second bounded join. Child
exceptions and interruptions are propagated as structured events. Every
claimed attempt ends in `COMPLETE`, `INVALID`, or `FAILED`, followed by a
worker terminal; no markerless process exit is accepted.

## Inputs and cost required for a future C1 approval

A later C1 authorization must be append-only, authorize exactly one dataset,
and bind:

- the then-current source commit and relevant-source hash;
- both CCMTPP configs and this exact runner config;
- that dataset's train/validation files, train transform, provenance and
  summary files;
- that dataset's frozen C0 manifest, sample, evaluation, terminal, validation
  SamplingPlan, and train-only threshold evidence;
- candidate C1, seed 4001, attempt number, exactly 20,000 updates, AdamW
  settings, batch 128, and the 7,200-second whole-job cap;
- an explicit externally isolated `cuda:0`; GPU inventory querying remains
  outside the runner authorization.

There are two C1 cells, one per dataset, requiring two separate
dataset-specific authorizations. The hard-cap upper bound is four GPU hours
total. With two separately selected idle GPUs they can finish within an
approximately two-hour wall-cap envelope; with one GPU the sequential envelope
is approximately four hours. These are caps, not performance measurements.
A future approval and launch plan are required before either cell may start.
