# CoF-HCMTTPP-v2 H1 architecture specification

## Source-only implementation status

The H1 core is implemented in `models/cof_hcmttpp_v2.py`. Its public mapping
is `CoFHCMTTPPV2H1` (sole model), `H1HurdleRQSGapDecoder` (gap decoder),
`TrainOnlyH1TailState` plus `fit_train_only_h1_tail_state` (train-only tail
state), and `build_h1_checkpoint_bundle`/`load_h1_checkpoint_bundle`
(in-memory CPU provenance round-trip). The implementation creates no runner,
authorization, launch command, runtime artifact, or data access path.

## Signed tail-scale amendment

This amendment supersedes only the one-sided conditional tail-scale formula
in commit `19ef5ad6a17d2068d9db3b36d97c4c8faf6072ea`. The additive positive
residual forced every event-level scale above `beta_base(Y)`. H1 instead uses
a signed log-scale residual so its scale may contract or expand around the
strict-exceedance train-only baseline while remaining positive. The tail
threshold, conditional tail gate, central RQS, sole H1 candidate, sole
`gap_decoder` changed factor, frozen references, gate, seed, and budget are
unchanged.

## Tail-conditionality amendment

This amendment supersedes only the fixed class-level tail mass and scale in
source-only specification commit
`49130970a360a63f495afa01fae42a01859e778c`. A fixed `p_tail(Y)` cannot express
the history-conditional occurrence of a positive-gap tail event and therefore
does not match H1's positive-gap conditional-calibration hypothesis. H1
remains the sole candidate, and `gap_decoder` remains its sole changed factor.
The frozen forensic inputs, C1-v1 permanent stop, seed, training budget, and
conjunctive H1 gate are unchanged.

## Scope and frozen parent

`CoF-HCMTTPP-v2` is a new, finite model-level family. It does not reopen the
stopped `cof_ccmtpp_v1` chain. The permanent v1 state is
`STOP_CCMTPP_V1_CHAIN_C1_GATE_FAIL`; C1 retry and C2–C4 authorization,
execution, or tuning remain forbidden.

H1 is the only v2 architecture. Its parent is the stored CCMTPP-v1 C1 source
and validation evidence. H1 preserves C1's causal event decoder, shifted
history inputs, causal/padding masks, position and target-length embeddings,
binary Y conditioning, amount encode/decode and conditional amount head,
flat receiver head, likelihood aggregation outside the gap term, optimizer,
seed, update count, wall cap, and fixed validation SamplingPlan. The only
architectural replacement is the gap decoder.

The source implementation reuses the C1 non-gap module classes directly. No
execution runner is part of the implementation.

## Fixed forensic basis

The design is based on commit
`8957c50f939b79eed27c60a108f230afb4b5e5ae` and the frozen
`forensic_cof_ccmtpp_v1_c1_gap_failure.*` evidence. Lower is better for every
metric below.

| Dataset | Y | C0 gap KS | C1 gap KS | C0 positive-only KS | C1 positive-only KS | C0 coherence | C1 coherence | C0 receiver TV | C1 receiver TV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AMLSim | 0 | 0.227623598 | 0.235709090 | 0.547556451 | 0.773860396 | 0.213971814 | 0.096367415 | 0.739997802 | 0.567449747 |
| AMLSim | 1 | 0.466188368 | 0.203175088 | 0.277135055 | 0.585089141 | 0.108702532 | 0.046202532 | 0.933597924 | 0.812395054 |
| Sparkov | 0 | 0.042308456 | 0.069522435 | 0.042308456 | 0.069522435 | 0.000501981 | 0.000296870 | 0.067917307 | 0.062882329 |
| Sparkov | 1 | 0.300651956 | 0.152206620 | 0.300651956 | 0.152206620 | 0.000517732 | 0.000258866 | 0.327983952 | 0.242978937 |

AMLSim validation Y=0 has zero mass 0.653988058, while C0 and C1 have
0.881611656 and 0.418278968. Sparkov has zero mass 0 for both classes and all
three sources, yet C1 Y=0 still fails. The train-to-validation Y=0 KS is only
0.027660457 for AMLSim and 0.012767441 for Sparkov. Mapping continuous C1
gaps back to the frozen bins reduces Y=0 KS by 0.021247311 and 0.018085140.
Thus positive-gap conditional calibration is the cross-dataset cause;
train/validation shift and evaluator bin mapping are not the primary cause.

## H1 hurdle factorization

For the unchanged causal state `h_t` and sequence label `Y`, H1 defines

```text
pi0_t = P(gap_t = 0 | h_t, Y)

p(gap_t | h_t, Y)
  = pi0_t * delta_0
    + (1 - pi0_t) * p_positive(gap_t | h_t, Y).
```

The zero atom is a dedicated Bernoulli hurdle logit. It is not approximated
by a continuous density at zero. For a valid target gap `g`, the gap loss is

```text
if g = 0:
  L_gap = -log(pi0)
else:
  L_gap = -log(1 - pi0) - log p_positive(g).
```

The target and comparison use exact stored gap values. No epsilon-based
relabeling of positive observations as zero is allowed.

## Positive-gap monotone quantile density

For `g>0`, let `u=log1p(g)>0`. The C1 logistic mixture is removed completely.
H1 represents a conditional quantile function
`u = Q_theta(q | h_t, Y)`, `q in (0,1)`, with one monotone
rational-quadratic spline (RQS) central body and one analytic unbounded tail.

The central RQS has exactly 16 bins. Conditional on taking the central route,
its probability knots are a deterministic uniform grid on `[0,1]`. The gap
head emits only positive normalized height increments and positive
derivatives; the probability widths are fixed. The central endpoints are
`(0,0)` and `(1,u_tail(Y))`. Minimum height fraction `1e-4 / 16` and minimum
derivative `1e-3` are architecture constants, not tunable candidate
parameters. The inverse RQS and its analytic Jacobian define the normalized
central density. A non-monotone, non-invertible, or non-finite state is a hard
failure.

The density is history- and Y-conditional through the unchanged C1 hidden
state and the RQS height/derivative outputs. There is no mixture-logistic head,
gap-bin classifier, validation calibration, or post-sampling transport.

## Train-only threshold and conditional tail

Tail state is fit separately for each Y class from valid, positive train gaps
only. Given `n_y` positive train values in log-gap space, define
`m_y=max(128, ceil(0.005*n_y))`. Among observed unique values, choose the
largest `u_tail(Y)` for which at least `m_y` observations are strictly larger.
Let `p_tail_base(Y)` be that strict exceedance fraction and

```text
beta_base(Y) = mean(u - u_tail(Y) | u > u_tail(Y)).
```

All counts, the ordered input hash, `u_tail`, `p_tail_base`, and `beta_base`
are frozen in the train-only gap state. They define the threshold, baseline
tail statistics, and deterministic initialization only; they are not fixed
event-level tail probability or scale. Fewer than 128 strict exceedances,
`beta_base<=0`, or any non-finite state fails closed; there is no pooled,
validation-derived, or hand-adjusted fallback.

Within the positive route, the conditional tail gate is

```text
pi_tail(h_t,Y) = P(u > u_tail(Y) | u>0,h_t,Y)

logit pi_tail(h_t,Y)
  = logit p_tail_base(Y) + r_tail(h_t,Y).
```

The residual-head weights and bias are initialized to zero, so the initial
gate equals the train-only baseline probability. Training may make the gate
history- and Y-conditional using train rows only.

The exponential tail scale is conditional through a signed log-scale
residual:

```text
log beta(h_t,Y) = log beta_base(Y) + r_beta(h_t,Y)
beta(h_t,Y) = beta_base(Y) * exp(r_beta(h_t,Y)).
```

`r_beta` is conditioned only on the causal history state and Y. Its weights
and bias are initialized to exactly zero, so the initial event-level scale is
exactly `beta_base(Y)`. A negative residual yields a scale below the baseline;
a positive residual yields a scale above it; and the exponential mapping
keeps every finite scale strictly positive. `beta_base(Y)` is still computed
only as the mean strict excess above the train-only `u_tail(Y)` threshold.
Overflow, an underflowed/non-positive scale, or a non-finite residual,
exponential, scale, density, NLL, or sample makes the candidate INVALID. None
may be clipped, redrawn, replaced, or silently dropped.

Let `F_c` and `f_c` be the normalized central RQS CDF and density on
`(0,u_tail(Y)]`, and let
`F_tail(u)=1-exp(-(u-u_tail(Y))/beta(h_t,Y))`. The positive-gap distribution
is

```text
0 < u <= u_tail(Y):
  F_positive(u|h_t,Y) = (1-pi_tail(h_t,Y)) * F_c(u|h_t,Y)
  f_positive(u|h_t,Y) = (1-pi_tail(h_t,Y)) * f_c(u|h_t,Y)

u > u_tail(Y):
  F_positive(u|h_t,Y) = 1-pi_tail(h_t,Y)
                        + pi_tail(h_t,Y) * F_tail(u|h_t,Y)
  f_positive(u|h_t,Y) = pi_tail(h_t,Y) * f_tail(u|h_t,Y).
```

Consequently the central body owns event-specific mass `1-pi_tail(h_t,Y)`,
the tail owns `pi_tail(h_t,Y)`, total positive mass is one, and the CDF is
continuous at `u_tail(Y)`. Positive-gap NLL includes the appropriate
`-log(1-pi_tail)` or `-log(pi_tail)` branch term plus the normalized central
or tail density term.

Sampling first draws the conditional tail gate. A central draw uses an
open-interval uniform variate through the normalized RQS; a tail draw uses an
exponential excess with conditional `beta(h_t,Y)`. The tail has support
`(u_tail,+infinity)`. Numerical protection may move a uniform variate to the
nearest representable value inside `(0,1)`, but it may not clip `u` or `gap`
to the observed train maximum or any validation-derived maximum.
`gap=expm1(u)` is emitted directly. Overflow or a non-finite sample makes the
candidate INVALID; it is never clipped, redrawn, or silently dropped.

Validation and internal test contribute zero rows to hurdle statistics,
knots, tail threshold, baseline tail statistics, gate/scale initialization,
or support construction. They also contribute zero rows to conditional gate
or scale learning.

## Frozen non-gap paths

- Causal backbone: byte-for-byte implementation reuse of the C1 causal event
  decoder, including right shift, no-future mask, valid-length mask, Y and L
  conditioning, width, depth, attention heads, dropout, and embeddings.
- Receiver: C1 `flat_no_copy` path. No pointer, hierarchy, calibration, or
  sampled-gap conditioning is added.
- Amount: the C1 frozen non-v3 train-only encode/inverse-decode contract and
  amount head. H1 changes neither amount loss nor its weight.
- Training: seed 4001, AdamW, learning rate 0.001, weight decay 0.0001, batch
  size 128, 20,000 requested updates, 7,200-second whole-cell cap, checkpoint
  interval 1,000, no early stopping, retry, sweep, or update extension.
- Conditioning/evaluation: the same dataset-specific frozen Y/length
  SamplingPlan, train-only transform, validation split, and metric formulas.

The new sampled gap naturally enters the already existing C1 amount path.
That dataflow does not authorize a new amount component or loss.

## Implemented RED-to-GREEN coverage

The CPU synthetic focused tests in `tests/test_cof_hcmttpp_v2_model.py` and
`tests/test_cof_hcmttpp_v2_contract.py` cover these vertical slices, with one
failing behavior test preceding each minimal implementation:

1. Hurdle Bernoulli: exact zero/positive likelihood split, finite gradients,
   exact zero samples, and no epsilon relabeling.
2. Central RQS: monotonicity, bijective forward/inverse, normalized density,
   analytic log-Jacobian, fixed 16-bin grid, and conditional response to
   `h_t,Y`.
3. Conditional tail: deterministic train-only threshold/baseline state and
   strict-exceedance rule; exact `r_beta=0` baseline equality; demonstrated
   `beta<beta_base` for a negative residual and `beta>beta_base` for a
   positive residual; per-event CDF mass normalization; left/right CDF
   continuity at `u_tail`; finite zero, central, and tail NLL; conditional
   tail-gate and signed log-scale responsiveness to distinct `h_t` values at
   fixed Y; unbounded samples; and zero hard upper-bound mass.
4. Factor isolation: serialized C1 backbone/receiver/amount/optimizer/budget
   configuration equality and proof that only the gap-decoder module differs.
5. Split boundary: validation/internal-test/fraudTest access during any fit
   immediately fails; fixed SamplingPlan is evaluation-only.
6. Artifact contract: append-only train-state/checkpoint/sample/diagnostic
   provenance, terminal-last finalization, and failed-gate family stop.
7. Regression: v1 chain remains stopped; no C1/C2/C3/C4 runner or
   authorization path is reopened.

## Future validation-only boundary

Even after source implementation, H1 may not run without a new explicit
authorization bound to the implementation commit, config hash, frozen train
and validation manifests, transform, SamplingPlan, C0/C1 reference evidence,
and train-only hurdle/spline/tail state. Only AMLSim H1 and Sparkov H1, one
fixed seed each, may be contemplated. Internal test, Sparkov fraudTest, TSTR,
privacy, full runs, and all result-based changes remain forbidden.
