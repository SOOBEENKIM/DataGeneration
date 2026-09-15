# `cof_ccmtpp_v1` architecture

## Causal history state

`CausalEventDecoder` embeds normalized amount, continuous `log1p(gap)`, and
receiver code, then adds position, fixed target-length, and binary `Y`
conditioning. Event tensors are shifted right with a dedicated receiver BOS.
An upper-triangular Transformer attention mask is used in addition to the
padding mask. Consequently position `t` is a function of events `<t`, `Y`,
and `L`; later values and padding cannot affect a prefix. The public interface
has no entity-ID argument.

## Continuous gap density

`MixtureLogisticGapHead` emits mixture logits, locations, and strictly
positive scales for `u=log1p(delta)`. It evaluates a stable logistic-mixture
log density and includes the inverse-transform Jacobian for gap-space NLL.
Inverse-CDF sampling is clipped only to the fixed `[0, max_gap]` support and
returns a typed `SampledGap`. Receiver and amount heads reject an ordinary
tensor where this typed model sample is required. No gap-bin classifier is
present.

## Receiver paths

C1 has a flat new-receiver softmax with PAD masked out. C2 adds a copy gate
and pointer distribution. Pointer keys are previous receiver embeddings;
the query includes the sampled-gap representation. The support is exactly
`j<t` and prior-valid positions.

C3/C4 replace only the flat new route with a structured distribution. One
route softmax covers UNK, deterministic head codes, and deterministic tail
clusters. A separate within-cluster softmax is evaluated after a tail route;
there is no single high-cardinality flat output head in this path. Copy
probability and structured-new probability form a normalized mixture. Target
probability and sampling can be computed from the structured representation
without constructing a dense event-by-vocabulary matrix.

## Amount and likelihood

`ConditionalAmountHead` operates in the already normalized amount space and
is conditioned on sampled gap and receiver. Its metadata names the frozen
non-v3 train-only encode/inverse-decode contract. It does not fit or alter an
amount transform. C1–C3 average the event likelihood over all valid events;
C4 averages the two class-specific valid-event means equally. No candidate
contains a pair or coherence structure loss.

## Checkpoint reload contract

An in-memory checkpoint bundle includes model configuration, train-only
hierarchy, CPU state dict, state hash, and exact hashes for source, config,
train manifest, transform, SamplingPlan, amount contract, and hierarchy.
Validation/internal-test/fraud-test fit row counts must be zero. Loading is
strict and rejects any provenance or state hash mismatch. The source-only
phase does not write checkpoints.

## TDD slices

The implementation was developed in vertical RED→GREEN slices:

1. shifted causal decoder, no-future/padding invariance, continuous density,
   finite NLL, range, and inverse;
2. train-only hierarchy, PAD/UNK, sampled-gap conditioning, causal pointer,
   normalization, head/tail support, and NLL decomposition;
3. finite C1–C4 composition, Y balancing, strict checkpoint reload, stop
   gates, preservation rehash, and source-only plan/dry-run.

Existing non-v3 CoF, CoF v3, CTGAN, and TVAE modules are not imported as
mutable parents and were not modified.
