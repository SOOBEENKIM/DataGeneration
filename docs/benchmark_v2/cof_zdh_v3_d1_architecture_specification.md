# CoF-ZDH-v3 D1 architecture specification

## Status and separation from H1

`CoF-ZDH-v3 D1` is a new source-only research family. It is not H1, an H1
retry, or a continuation of `cof_hcmttpp_v2`. Its parent source commit is
`a72cf11d13428908e4ec7de6aacbeb3055b7d085`. The H1 family remains
permanently frozen at:

> `STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL`

The frozen H1 forensic evidence is read-only. D1 does not authorize H1
attempt_004, any H1 retry, internal test, Sparkov fraudTest, model execution,
or a runtime artifact.

This package specifies one candidate and no executable model:

- family: `cof_zdh_v3`
- candidate: `D1`
- model ID: `cof_zdh_v3_d1`
- display name: `CoF-ZDH-v3 D1`
- expansion: Zero-inflated Discrete-Hazard conditional gap decoder
- implementation state: `SPECIFIED_NOT_IMPLEMENTED`

## Frozen parent and sole changed factor

D1 preserves the stored CCMTPP-v1 C1 implementation contract for every
non-gap factor:

- causal event decoder and shifted history inputs;
- no-future attention mask and valid/padding mask;
- binary sequence-level Y and target-length conditioning;
- positional and length embeddings;
- frozen train-only amount transform, amount head, inverse/decode contract,
  loss, and loss weight;
- flat, no-copy receiver head, receiver vocabulary, loss, and loss weight;
- ordinary valid-event likelihood aggregation with no structural loss;
- seed 4001, AdamW, learning rate 0.001, weight decay 0.0001, batch size 128,
  20,000 requested updates, 7,200-second whole-cell cap, and 1,000-update
  checkpoint interval;
- each dataset's frozen train-only transform and validation SamplingPlan.

The only changed factor is `gap_decoder`. D1 contains no pointer receiver,
receiver hierarchy, Y-balanced loss, coherence loss, amount redesign,
additional update, early stopping, sweep, retry, or result-based selection.

## Conditional distribution

For causal history state `h_t`, label `Y`, and raw nonnegative gap `g_t`, D1
uses a separate exact-zero route and a positive route:

```text
p(g_t | h_t,Y)
  = p0(h_t,Y) * delta_0
    + (1-p0(h_t,Y)) * p_positive(g_t | h_t,Y).
```

The zero route is an actual category. A target is zero if and only if its
stored raw gap equals zero. Epsilon relabeling is forbidden.

All event-level zero and hazard logits use the same bounded map:

```text
bounded_logit(a) = 12 * tanh(a / 12)
p0 = sigmoid(bounded_logit(a0)).
```

`12` is a frozen architecture constant. It is not selected from validation.
The map is smooth and prevents an event probability from becoming exactly
zero or one in ordinary finite arithmetic. Stable log-sigmoid identities,
not probability clipping, define the NLL.

## Train-only interval and tail state

For each dataset and Y class, use valid, strictly positive train gaps only and
set `u=log1p(g)`. No validation, internal-test, or fraudTest row may enter
this state.

Let `n_y` be the number of positive train values and
`m_y=max(128,ceil(0.005*n_y))`. Choose `u_tail(Y)` as the largest unique
observed `u` having at least `m_y` strict exceedances. Define the fixed tail
scale

```text
beta_tail(Y) = mean(u-u_tail(Y) | u>u_tail(Y)).
```

Fewer than 128 strict exceedances, a non-finite/non-positive threshold or
scale, or an empty class fails closed. There is no pooled or hand-adjusted
fallback.

The positive central support `(0,u_tail(Y)]` has exactly 32 equal-width
intervals:

```text
e_j(Y) = j * u_tail(Y) / 32,  j=0,...,32.
```

The body route uses `[e_(j-1),e_j)` for `j<32` and
`[e_31,e_32]` for the last interval. The tail route is the strict region
`(u_tail(Y),+infinity)`. Thus the number of intervals is source-frozen while
every numeric edge, tail boundary, fixed scale, and support decomposition is
derived only from train. Ordered positive-input hashes, counts, edges,
thresholds, scales, and construction-version hashes must be checkpointed.

These are D1 model intervals. They must not import, reuse, optimize against,
or be aligned to validation evaluator bins, support diagnostics, or bootstrap
thresholds. Accidental numerical equality with an evaluator edge does not
change provenance: the dependency graph must remain train-only.

## Conditional discrete hazard

The gap head emits 32 raw hazard scores from `h_t,Y`. Each is bounded before
conversion to a hazard:

```text
ell_j = 12 * tanh(a_j(h_t,Y) / 12)
q_j = sigmoid(ell_j),  j=1,...,32.
```

Conditional positive-route masses are

```text
w_j = q_j * product(k<j)(1-q_k)
w_tail = product(k=1..32)(1-q_k).
```

The 33 route masses are nonnegative and sum to exactly one algebraically.
Log masses are computed with bounded logits and stable `logsigmoid` /
`logsigmoid(-ell)` sums. A non-finite logit, log mass, probability, or
normalization error is INVALID; clipping, renormalization fallback, route
dropping, or redraw is forbidden.

The discrete hazard is conditional on causal history and Y. Only its
train-likelihood gradient fits its parameters. The numeric edges and tail
shape remain the frozen train-only state above.

## Within-interval and tail density

Conditional on body interval `j`, D1 uses the uniform density in log-gap
space:

```text
f_j(u|Y) = 1 / (e_j(Y)-e_(j-1)(Y)).
```

Sampling is the affine map of an open-interval uniform variate into the
selected interval. It requires no inverse spline, root finder, learned scale,
or fallback. Every width is positive because `u_tail(Y)>0` and the equal
width construction is exact from one finite threshold.

Conditional on the tail route, D1 uses a shifted exponential in log-gap
space with the fixed train-only `beta_tail(Y)`:

```text
f_tail(u|Y) = exp(-(u-u_tail(Y))/beta_tail(Y)) / beta_tail(Y),
u > u_tail(Y).
```

The conditional tail **mass** changes with `h_t,Y` through `w_tail`; the tail
scale has no learned residual. In particular, D1 has no `exp(r_beta)`,
learned rate, or unbounded scale parameterization. Tail log density is
calculated as `-log(beta_tail)-excess/beta_tail`. An open-interval uniform
variate and `-beta_tail*log1p(-v)` sample the excess.

The positive density in raw-gap coordinates includes the exact Jacobian
`du/dg=1/(1+g)`. Body and tail CDF mass is normalized by construction and is
continuous at `u_tail`, although density continuity is neither required nor
claimed.

The tail support is unbounded, but its parameters are finite train-only
constants and its event mass comes from bounded logits. A non-finite
`log1p`, log density, NLL, sampled `u`, or `expm1(u)` is terminal INVALID.
Hard upper clipping, epsilon clipping, fallback, silent omission, and redraw
are forbidden.

## Why this avoids the observed H1 failure modes by design

D1 removes both H1 numerical mechanisms implicated by the frozen history:

1. There is no RQS and therefore no RQS forward/inverse, quadratic root,
   spline Jacobian, learned height, or derivative state.
2. There is no conditional `beta_base*exp(r_beta)` path. The tail scale is a
   positive finite train-only constant and cannot explode through a learned
   exponential residual.
3. Zero and body/tail route probabilities are based on smoothly bounded
   logits and stable log-hazard sums.
4. Within-bin inversion is affine and the tail log density avoids exponent
   evaluation in the likelihood.

This is a structural numerical argument, not a performance claim. D1 may
still fail its fidelity/coherence gate or become invalid through another
non-finite module. Such a result terminates D1; it does not authorize tuning.

## Future runtime finite-state contract

Any later runner must check every update, not only progress intervals.
Required phases are `pre_forward`, `post_forward`, `post_loss`,
`post_backward_pre_step`, and `post_optimizer_step`. It must preserve:

- total and per-term loss finite state before backward;
- global and per-module gradient norm, maximum absolute gradient, and every
  non-finite tensor name before optimizer step;
- parameter and optimizer-state finite counts, maxima, and offending tensor
  names before and after optimizer step;
- backbone hidden-state, zero-logit, bounded hazard-logit, hazard,
  log-survival, route-mass, fixed tail-scale, body-width, log-density, and NLL
  finite summaries;
- route-mass normalization error and minimum/maximum route probability;
- the exact first invalid update, phase, tensor, module, dtype, shape,
  finite/non-finite counts, min/max where defined, prior checkpoint pointer,
  and traceback.

An invalid loss or gradient prevents the optimizer step. A newly invalid
parameter or optimizer state after a step records that exact step and stops.
The first-invalid record is append-only and terminalized; it may not be
reclassified as model performance or hidden by a later checkpoint.

## Source-only boundary

This specification creates no model class, runner, CLI, authorization,
launch command, checkpoint, runtime directory, data-body read, fit, sample,
evaluation, or test access. Those remain separate, explicitly unauthorized
future decisions.
