# Preregistered CoF-HCMTTPP-v2 H1 study

## RQS inverse corrective amendment

The AMLSim and Sparkov H1 `attempt_001` runs under source commit
`70481e2e4b612002fe7d3d074b9b67294869a52b` both terminated before a
training update with `InvalidH1GapStateError: RQS inverse root is invalid`.
This is an implementation failure, not an H1 gate or performance result. The
corrective source pins the central spline cumulative endpoints exactly to
`0` and `u_tail`, derives bin heights from those pinned knots, and uses the
exact analytic roots `theta=0` and `theta=1` at exact bin endpoints. RQS
geometry, forward, and inverse arithmetic for float16/bfloat16/float32 model
parameters is evaluated in float64 so valid minimum-height bins do not lose
their inverse to quantization; the interior quadratic formula itself is
unchanged. The H1 architecture, data, threshold, seed, budget, gate, and
frozen C0/C1 references are unchanged. Epsilon clipping, silent fallback,
redraw, and reuse or retry of `attempt_001` remain forbidden.

## Signed tail-scale amendment

The one-sided scale residual in source-only specification commit
`19ef5ad6a17d2068d9db3b36d97c4c8faf6072ea` is superseded. H1 now fixes
`log beta(h_t,Y)=log beta_base(Y)+r_beta(h_t,Y)`, equivalently
`beta(h_t,Y)=beta_base(Y)*exp(r_beta(h_t,Y))`. The signed residual depends
only on causal history and Y; zero-initialized weights and bias make the
initial scale exactly equal to the strict-exceedance train-only baseline.
This is a correction within the same gap decoder, not a new candidate or
factor. The tail threshold, tail gate, central RQS, gate, budget, and frozen
references are unchanged.

## Tail-conditionality amendment

The original source-only specification fixed the positive-route tail mass as
`p_tail(Y)`. That is superseded because a class statistic cannot represent
`P(u>u_tail(Y) | u>0,h_t,Y)`. The strict-exceedance threshold and baseline
tail statistics remain train-only and fixed, but H1 now uses a conditional
tail gate and a train-anchored signed log-scale residual. This
does not create another candidate or changed factor. H1 is still the only
candidate, and its only difference from C1 is `gap_decoder`.

## Status

This document freezes a source-only implemented family. The sole H1 core is
`CoFHCMTTPPV2H1`, its gap decoder is `H1HurdleRQSGapDecoder`, its frozen
train-state type is `TrainOnlyH1TailState`, and its in-memory checkpoint seam
is `build_h1_checkpoint_bundle`. The execution runner is not implemented;
there is no authorization, launch command, runtime artifact, persisted
checkpoint, or result. The machine contract is
`configs/benchmark_v2/cof_hcmttpp_v2_source_only.yaml` with SHA-256
`f623c2c6db5843a02404da4ab3824fa1464cbc1380659dbd717e512315aee12e`.

The prior family remains permanently stopped at
`STOP_CCMTPP_V1_CHAIN_C1_GATE_FAIL`. Commit
`8957c50f939b79eed27c60a108f230afb4b5e5ae` and the three
`forensic_cof_ccmtpp_v1_c1_gap_failure` evidence files are immutable design
inputs. They are not reinterpreted as authorization to retry C1 or execute
C2–C4.

## Scientific question

Can replacing only C1's gap decoder with a zero hurdle plus an unbounded,
conditional monotone positive-gap quantile density fix the cross-dataset Y=0
gap failure while preserving C1's coherence, receiver, and Y=1 gap gains?

This is a single hypothesis test, not a tuning sequence. There is exactly one
candidate:

- H1: `cof_hcmttpp_v2_h1_hurdle_rqs`.

C0 and C1 are immutable references, not v2 candidates. No H2, fallback,
sweep, alternate knot count, alternate tail rule, or loss-weight candidate is
preregistered.

## Frozen factor isolation

Relative to stored CCMTPP-v1 C1, H1 changes exactly one factor:

```text
gap decoder:
  mixture-logistic with hard [0, train max] support
  -> Bernoulli zero hurdle + conditional 16-bin RQS quantile body
     + train-thresholded conditional tail gate and exponential tail scale.
```

Everything else is fixed to C1: causal decoder, shifted history, Y/L
conditioning, masks, amount architecture and contract, flat receiver path,
ordinary valid-event likelihood aggregation, optimizer, learning rate,
weight decay, seed, batch size, updates, wall cap, checkpoint interval, and
SamplingPlan. Any additional difference makes H1 unauthorized and INVALID.

## Frozen data use

AMLSim and Sparkov use their existing external protocol v1 frozen train and
validation bundles. Train may fit only:

- the unchanged amount transform and receiver vocabulary already frozen by
  the external protocol;
- hurdle model parameters through training loss;
- the fixed RQS head parameters through training loss;
- per-Y positive-gap `u_tail`, baseline strict-exceedance probability, and
  baseline excess scale using the exact train-only rule in the architecture
  specification;
- the conditional tail-gate and signed log-scale residual parameters through
  train likelihood only, initialized from those train-only baselines.

Validation is used only after generation to calculate the frozen metrics and
gate. It may not fit zero rates, knot positions/count, tail threshold,
baseline tail statistics, conditional gate/scale parameters or
initialization, support, thresholds, vocabulary, transforms, or SamplingPlan.
Internal test and Sparkov fraudTest are forbidden before path/body access.

## Frozen references and observed values

The comparison values are frozen before H1 exists:

| Dataset | Metric | Y=0 C0 | Y=0 C1 | Y=1 C0 | Y=1 C1 |
|---|---|---:|---:|---:|---:|
| AMLSim | gap KS | 0.227623598 | 0.235709090 | 0.466188368 | 0.203175088 |
| AMLSim | positive-only gap KS | 0.547556451 | 0.773860396 | 0.277135055 | 0.585089141 |
| AMLSim | coherence error | 0.213971814 | 0.096367415 | 0.108702532 | 0.046202532 |
| AMLSim | receiver TV | 0.739997802 | 0.567449747 | 0.933597924 | 0.812395054 |
| Sparkov | gap KS | 0.042308456 | 0.069522435 | 0.300651956 | 0.152206620 |
| Sparkov | positive-only gap KS | 0.042308456 | 0.069522435 | 0.300651956 | 0.152206620 |
| Sparkov | coherence error | 0.000501981 | 0.000296870 | 0.000517732 | 0.000258866 |
| Sparkov | receiver TV | 0.067917307 | 0.062882329 | 0.327983952 | 0.242978937 |

C0 full receiver TV is 0.734645861 for AMLSim and 0.065505762 for Sparkov;
C1 values are 0.562529727 and 0.062437900. The evidence also fixes the
AMLSim zero-atom mismatch, Sparkov upper-tail saturation, train/validation
shift, and bin-mapping results; no result may be selected or omitted after
H1 is observed.

## Conjunctive H1 gate

Every condition must pass independently in both AMLSim and Sparkov. Missing,
non-finite, provenance-mismatched, or invalid evidence is FAIL.

1. Hard validity passes: fixed Y/length/mask, padding, train receiver support,
   finite gap density/sample, exact split boundary, and forbidden-access
   counters all valid.
2. For each Y class, H1 overall gap KS is no larger than C0 overall gap KS.
3. For Y=0, H1 positive-only gap KS is strictly smaller than C1 positive-only
   gap KS.
4. For each Y class, H1 short-gap × receiver-repeat coherence error is no
   larger than C1.
5. For each Y class, H1 receiver TV is no larger than C1, and H1 full
   receiver TV is no larger than C1.

All comparisons use unrounded stored values. Equality is permitted only for
the explicitly noninferiority comparisons; positive-only Y=0 improvement is
strict. Head/tail/UNK receiver diagnostics must be reported but are not added
post hoc to the gate.

If any condition fails, the terminal state is
`STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL`. The v2 family ends permanently;
there is no H1 retry, H2, parameter adjustment, alternate tail, or automatic
test unlock. Passing H1 would mean only that this validation gate passed. It
would not authorize internal test, fraudTest, TSTR, privacy, a full run, or a
claim against IID/CTGAN/TVAE.

## Fixed training budget for a possible future run

- datasets: AMLSim and Sparkov, evaluated separately;
- candidate: H1 only;
- seed: 4001;
- requested updates: 20,000;
- whole-cell wall cap: 7,200 seconds;
- batch size: 128;
- optimizer: AdamW;
- learning rate: 0.001;
- weight decay: 0.0001;
- checkpoint interval: 1,000 updates;
- early stopping, retry, sweep, update extension, and result-based checkpoint
  selection: forbidden.

## Source-only implementation and execution boundary

The architecture document fixes the completed RED-to-GREEN source-only
coverage. CPU synthetic tests may import and exercise the mathematical model
core, including deterministic sampling from injected uniforms; they do not
fit a model or access frozen data. A future execution runner and authorization
require later, separate requests. No plan may read split bodies, initialize
CUDA, or create runtime artifacts. Any future execution must be
validation-only and append-only.

This commit authorizes none of those later steps.
