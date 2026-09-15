# Preregistered `cof_ccmtpp_v1` finite candidate family

## Status and immutable boundary

This document freezes a source-only architecture and authorization-bound
execution-runner implementation. It does not authorize an
experiment, GPU inventory query, CUDA initialization, fit, checkpoint write,
dataset sample, validation evaluation, selection, internal-test access,
Sparkov `fraudTest` access, TSTR, privacy analysis, or full run. The machine
contract is
`configs/benchmark_v2/cof_ccmtpp_v1_source_only.yaml` (source-preparation
SHA-256 `2f3efe9fb2cc62027a12d7380e340604c6aeaece9b39eab91350e141e7088aa6`).
Any later run requires a separate authorization bound to the eventual source,
config, train data, train-only transform, vocabulary, hierarchy, SamplingPlan,
and frozen-control hashes.

The runner machine contract is
`configs/benchmark_v2/cof_ccmtpp_v1_execution_runner.yaml`. Its plan and
dry-run do not import a model or execution backend and do not open split array
bodies. Execution validates a separately supplied append-only authorization
before all data-body, model, device, and runtime access. No such authorization
is part of this source commit.

The official worker scope is one explicit
`--dataset {amlsim,sparkov}` times one candidate per process. A C1 execution
authorization binds only AMLSim or only Sparkov, never both. Dataset scope is
part of the canonical authorization hash, and cross-dataset validation or
path resolution fails closed. Dataset-specific ownership/attempt subtrees let
two later, separately authorized workers use different physical GPUs without
sharing an artifact writer. This preparation still creates neither
authorization nor launch command.

This is a finite mechanistic study, not tuning until a baseline is beaten.
Candidate results may not change the loss, thresholds, vocabulary cutoff,
cluster rule, seed, sampling rule, update count, wall cap, or stop criteria.
Every authorized attempt, including failure, must remain append-only.

## Scientific factorization

For causal history state `h_t`, sequence label `Y`, and target length `L`, the
new family uses

```text
p(delta_t, receiver_t, amount_t | h_t, Y)
= p(delta_t | h_t, Y)
  p(copy_t | sampled_delta_t, h_t, Y)
  p(receiver_t | copy_t, sampled_delta_t, h_t, Y)
  p(amount_t | receiver_t, sampled_delta_t, h_t, Y).
```

`h_t` is produced from shifted event inputs and a strict causal attention
mask; event `t` cannot see event `t` or any future event. Inputs are amount,
continuous `log1p(gap)`, receiver, position, `L`, and `Y`. Entity identifiers
are forbidden. Valid positions form a contiguous prefix; padding is excluded
from attention, pointer support, likelihood, and diagnostics.

The gap channel has no bin-classification head. It models `u=log1p(gap)` with
a five-component logistic mixture and uses the exact `expm1` inverse within
the fixed nonnegative support. Receiver conditioning receives a
`SampledGap` produced by this head, never the true current gap. Its upper
support is the maximum valid train gap, fixed in the train-state artifact
before validation; it is not selected from candidate results. The amount
channel retains the frozen non-v3 train-only encode/inverse-decode contract;
amount is not redesigned in this candidate family.

## Train-only receiver construction

PAD is code 0 and UNK is code 1. Receiver vocabulary and hierarchy are fit
from valid train events only. Codes are ordered by decreasing train count and
then numeric code. At most 512 codes with count at least 32 are heads. All
remaining train-vocabulary codes are assigned by frequency rank round-robin
to at most 64 nonempty tail clusters. The state, counts, rule, vocabulary,
train manifest, and their hashes are checkpoint provenance. Validation,
internal test, and Sparkov `fraudTest` contribute zero rows.

The copy branch normalizes a pointer distribution over prior valid positions
only. The new branch routes among UNK, head codes, and tail clusters, then
normalizes within the selected tail cluster. PAD is never generated. Copy and
new mass are combined by one gate and must sum to one. Diagnostics preserve
overall, head, tail, UNK, repeat, and new NLL plus their counts.

## Finite candidates and sequential gates

- C0: frozen non-v3 CoF control. It is a hash reference only and is neither
  retrained nor resampled.
- C1: causal decoder, continuous-gap density, and flat receiver decoder.
- C2: C1 plus sampled-gap receiver conditioning and causal pointer copy.
- C3: C2 plus the fixed train-only head/tail new-receiver hierarchy.
- C4: C3 plus equal weighting of the mean conditional likelihood for `Y=0`
  and `Y=1`.

C1 is the only first candidate that a future authorization may unlock. C2 is
locked until C1 passes; C3 is locked until C2 passes; C4 is locked until C3
passes. A parent failure ends the family at that point and all stored evidence
remains.

The gates are fixed before results:

- C1: in every dataset and Y class, gap KS must be no larger than C0; the
  short-gap × receiver-repeat error must be no larger than C0.
- C2: short-gap × repeat error must be strictly smaller than C1 and full
  receiver TV must be no larger.
- C3: full receiver TV must be strictly smaller than C2. Head, tail, and UNK
  decomposition errors must each be no larger and at least one must be
  strictly smaller.
- C4: the preregistered `max(gap KS, receiver TV, short-gap-repeat error)`
  composite for `Y=1` must be strictly smaller than C3. The corresponding
  `Y=0` composite may be at most 1.05 times C3; if the parent value is zero,
  it must remain zero.

All multi-dataset/class requirements are conjunctive. Missing, non-finite, or
provenance-mismatched evidence fails closed.

## Deferred structural loss

C1–C4 have no structure loss. C5 is deliberately
`LOCKED_UNIMPLEMENTED`. Only after a formally passing C4 may a new amendment
select exactly one of:

1. class-conditional `(gap quantile, receiver frequency-bin)` pair loss; or
2. class-conditional short-gap × repeat coherence loss.

The two may never be combined in C5. This source commit implements neither.

## Selection and data-use rules

Train data alone may fit transforms, vocabulary, hierarchy, density state,
or calibration. Validation may only evaluate a generated candidate after a
separately authorized run. Internal test and Sparkov `fraudTest` remain
forbidden. Thresholds cannot be changed after observing a candidate. There is
no sweep, early stopping, automatic retry, or result-based update extension.
