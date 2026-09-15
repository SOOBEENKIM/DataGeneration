# CoF-ZDH-v3 D1 synthetic-only RED/GREEN test plan

## Current status

This is a future implementation plan, not executable test code. No model or
runner is implemented by the current package. Every future fixture described
below must be synthetic and CPU-only; it may not read a frozen train,
validation, internal-test, fraudTest, C0/C1 sample, or H1 runtime body.

Each slice must be implemented in RED then GREEN order, with the smallest
production change that satisfies that slice before proceeding.

## Slice 1: frozen lineage and factor isolation

RED: a candidate differing from C1 in any factor other than `gap_decoder`, or
one that identifies itself as H1/retry, is accepted.

GREEN: accept only family `cof_zdh_v3`, candidate D1, and an exact frozen C1
non-gap fingerprint. Assert that H1 remains
`STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL` and has no retry path.

## Slice 2: exact-zero route

RED: zero is assigned continuous positive density, epsilon relabeling is
possible, or finite extreme raw logits produce exact probability zero/one.

GREEN: exact raw zero uses only the zero category; positive values use only
the positive route. Verify stable finite NLL/gradients and the smooth bounded
logit range `[-12,12]` without clipping.

## Slice 3: train-only support construction

RED: validation changes an edge, tail threshold, tail scale, or support hash;
duplicate/disordered/non-finite state is silently repaired.

GREEN: synthetic train values deterministically produce 32 strictly
increasing equal-width edges, the strict-exceedance threshold and fixed
positive scale. Validation/internal-test/fraudTest access fails before path
resolution. Insufficient or invalid train state fails without fallback.

## Slice 4: discrete-hazard normalization

RED: route mass depends on an evaluator bin, can be negative/non-finite, or
does not sum to one under extreme finite raw scores.

GREEN: bounded logits and stable log-survival produce 32 body masses plus one
tail mass whose algebraic and numeric sum is one within a frozen tolerance.
Verify causal-history and Y responsiveness and finite gradients.

## Slice 5: body and tail likelihood

RED: body inversion uses RQS/root finding; tail scale contains a learned
exponential residual; an endpoint is double-counted; non-finite values are
clipped or redrawn.

GREEN: verify left-closed/right-open body ownership, final-body threshold
ownership, strict tail ownership, affine within-bin inverse, fixed-scale tail
log density, raw-gap Jacobian, CDF normalization, threshold continuity,
finite NLL/gradients, and no RQS or `exp(r_beta)` state.

## Slice 6: sampling contract

RED: a closed-interval variate hits an ambiguous endpoint, invalid output is
redrawn, or a hard upper cap is applied.

GREEN: injected open-interval variates deterministically select zero, each
body interval, and tail; sampled values obey support and masks. Non-finite
`u`/gap is INVALID with zero clip/fallback/redraw counts.

## Slice 7: first-invalid telemetry

RED: injected non-finite loss, gradient, parameter, optimizer state, hidden
state, hazard logit, mass, or density becomes only a generic terminal error or
is first noticed at a later progress interval.

GREEN: every phase check attributes the exact first step/tensor/module.
Pre-step invalid state prevents optimizer execution; post-step invalid state
is attributed to that optimizer step. The first-invalid artifact is
append-only and includes the last verified checkpoint pointer.

## Slice 8: checkpoint and provenance

RED: a checkpoint reloads after changing an edge, C1 fingerprint, train hash,
SamplingPlan, threshold, source/config hash, or finite-state schema.

GREEN: exact CPU round-trip succeeds; every mismatch fails closed. Prove no
validation/internal-test/fraudTest fit rows and no H1/RQS state.

## Slice 9: gate truth table

RED: averaging, rounding, missing evidence, one passing dataset, or a NaN can
produce PASS.

GREEN: only the full conjunction of hard validity, zero numerical-invalid
count, C0 classwise gap noninferiority, strict C1 Y0 positive-gap
improvement, and C1 coherence/receiver noninferiority across both datasets
passes. PASS still returns no internal-test authorization.

## Slice 10: source-only plan/dry-run boundary

RED: plan/dry-run imports a model, reads a split body, queries CUDA, creates a
runtime path, or accepts an authorization/launch operation.

GREEN: static config/provenance/factor/gate inspection reports zero model,
data-body, GPU/CUDA, fit, sample, evaluation, runtime-write, internal-test,
and fraudTest calls. Execute remains absent until a separate user request.
