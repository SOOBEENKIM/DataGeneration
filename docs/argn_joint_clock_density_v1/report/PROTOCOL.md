# Fixed init_only residual time-density 2×2 — registered 2026-09-29

## Decision and closest prior

Implement the user's authorized comparison, preserving init_only and active
clock_prefix. Read the Sep29 research reassessment and literature-to-design
ledger first. IFL-TPP/mark-dependent lognormal mixtures already cover
history/mark-conditioned mixture outputs. SigTPP2026 already trains generated
path distributions. MTDPP2026 addresses duration marginals and dependence.
LBDTPP2026 is being fit concurrently as a current timing/mark baseline. GMR,
phase conditioning, residual heads, sliced W1 and implicit sampling gradients
are known components, not established contributions of this experiment.
The local question is whether a distribution-shape residual plus conditional
clock-gap loss improves the timing tradeoff left by init_only.

New implementation source: Figurnov et al., NeurIPS2018, §3 equation8 and §4
mixture distributions, https://papers.neurips.cc/paper/7326-implicit-reparameterization-gradients.pdf.
The mixture sample uses its ordinary forward sampler and implicit CDF gradient.
Numerical inverse-CDF finite differences agree to 5.1e-11 before quantization.
Integer-second rounding uses an explicit straight-through surrogate. This is
NOT an unbiased-gradient or convergence/stability theorem for our rollout.

## Exact 2×2 and implementation refinement before any fit

| arm | output change | objective |
| --- | --- | --- |
| mean_marginal | three bounded component-location residuals | marginal/clock preservation |
| mean_joint | identical output | same + conditional clock-gap and worst-bin guard |
| mixture_marginal | logits, locations and log-scales | marginal/clock preservation |
| mixture_joint | identical output | same + conditional clock-gap and worst-bin guard |

All use frozen init_only law plus a 1329→64→9 SiLU residual. Mean-only masks
the logits/scale outputs. All have the same stored parameter shapes; active
degrees of freedom differ and this is not proof excluding a capacity effect.
Bounds: component-location .35log-seconds, logits .5, log-scale .20 via tanh.
All first1–5 interval laws stay exact. Original GMR, six correction coefficients,
initial mixture, transition, amount and merchant heads are not refitted.
Inputs: native pre-gap context/history1322, current transition phase, strict
past20-gap median clock, clock valid flag, log-position. No future/current gap
or later-generated field enters the native head input. Zero residual exactly
recovers init_only, verified independently.

This implements the reassessment's limited mean-vs-full-law question, but
**A is not a rerun of the old six-coefficient Powell optimizer**. A/C use its
marginal preservation idea adapted to a neural head and stochastic windows:
equal-position-bin averaging and a 128-quantile weighted W1 approximation.
This explicit refinement is registered before fitting. The prior coefficient
experiments remain the exact historical controls; do not call A their exact
algorithm reproduction. All four new arms use this same adaptation.

## Data, recursion and weighting

Optimization619 customers/816283 events; internal69/100226. Parent20260930.
No development events select checkpoints or hyperparameters; no held-out test
events are read. Cached ARGN features were computed from real strict-past
prefixes and exact record/event/label matching. Their decoded gaps match the
metadata exactly in both splits. Test no-future behavior via native routing.

Training clock replays generate every interval from the start through the
full available sequence (up to3106 intervals). Their fraud phases and cached
native representations stay real. Thus this is **conditional recursive time
training**, NOT free joint ARGN rollout training. After fitting, native full
generation updates the ARGN representation from all generated fields; that
is the required final efficacy check.

Backpropagate through64 successive generated intervals in16 chunks/update.
Prefixes are generated with the current head, detached at each chunk start,
and refreshed every80 updates. The initialization baseline has its own prefix
with matched random noise. Eight chunks rotate through position strata;
eight contain a sampled onset. No actual clock is supplied to a generated
chunk. This truncated/stale-prefix gradient is an explicit computational
approximation; do not claim full-path backpropagation.

Common warm-up400 steps of quantized conditional likelihood; then160 rollout
updates. AdamW lr3e-4,weight decay1e-4,gradient clip1; fit seed20261330.
Likelihood is .5 natural-event-weighted + .5 phase-macro, implemented with
stratified draws and explicit phase population weights. It is NOT pure
natural-distribution MLE. Rollout anchor weight .05.

Marginal objective = .5(customer-normal W1+mean normal-position W1)+.25clock W1;
penalties4×positive excess over matched baseline for personal/onset(.01)
and fraud gap(.02). Position bins6–49,50–99,100–199,200–499,500–1023,
1024–2047,2048+. W1 uses128 fixed midquantiles with inverse within-group
customer counts, not exact OT. Training relative coordinate uses
log(1+(1+gap)/(1+median_gap)); native reporting keeps the prior raw-ratio metric.

Joint objective adds .25 times phase×position clock/log-gap projected W1,
directions(1,0),(0,1),(1,1)/sqrt2,(1,-1)/sqrt2, and4 times the worst excess
over baseline: .02 for any position bin or .01 for event-weight normal gap.
Groups with fewer than4 sampled events omit their joint term. These are
registered engineering settings, not literature-derived optimal constants.
Sampling is deliberately stratified; do not interpret this objective as the
unbiased headline event-population distance. Independent native metrics show
both event and customer weighting and subgroup support.

## Selection, generation, comparison and interpretation

Select on each arm's own registered internal objective averaged over two
fixed full-length time-replay noises20261401/20261402 at identity, warm-up,
and every40 rollout updates. Identity remains eligible. Save complete learning
curves, identity selection, failures and unchanged original parameter checks.
Different selected objectives cannot be compared as the same numeric loss.

Four fits, one frozen parent, two native draws20261011/20261012 per arm. This
is a screening experiment, not4 independent seeds for one method. Native
generation must pass routing/first-event/compacted-batch tests, unchanged
backbone and labels/length checks. A historical imported COMPLETE.json field
hardcodes4generations; NATIVE_COMPLETE.json records the correct2 and the
evaluator counts actual files. Never report the legacy count as performed work.

Compare against saved matched init_only, general history, GMR, current,
clock_prefix; ARGN/CPAR are separate whole-generator comparisons with their
support/training-budget caveats. LBDTPP has its own frozen protocol and only
time/mark common task. Report all costs: early/middle/tail normal gap,
onset personal timing, whole/customer normal gap, fraud timing, normal amount,
fraud relative amount, merchant diversity. No automatic model replacement.
Screen promising means either >=10% onset-relative improvement while normal
gap is within.01, or >=10% normal-gap improvement while onset-relative is within
.01, with both draws improving the target. Explicitly list costs>.02 in any
position bin, >.01 amount/merchant TV or >5% diversity loss. Passing this screen
only justifies independent refits and the same-objective general-history
control; it does not establish novelty. No commit/push for failed or uncertain
models, and preserve all previous artifacts byte-for-byte.
