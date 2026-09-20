# Frozen repeat-probability calibration: exploratory follow-up v1

Registered 2026-09-20 before implementation or fitting. Parent: `cfbe365`.
The user authorized this bounded follow-up after correcting the September 18
Notion report. Historical ER FAIL and every prior result remain unchanged.

## Question and fixed comparison

Does a train-only adjustment of the copy probability improve **joint free
generation**, and does residual regularization retain any incremental benefit?
E and L003 already use history. U remains an additional reference. Reuse their
best saved checkpoints; no base-model retraining or checkpoint reselection.
Four prevalences (.05, .10, .25, .50), both kappa settings, five existing paired
training trials: 80 new four-scalar calibration fits, 120 reused reference models.
All cells are completed regardless of the direction of scientific results.

For observed context s, let z be the frozen copy logit and f the frozen fresh
mark distribution. Fit a_s and b_s using

    q' = sigmoid(a_s + b_s z), b_s > 0
    R' = q' + (1-q') f(previous mark | history,s)
    p'(mark) = q' * point_mass(previous mark) + (1-q') * f(mark).

First events use f unchanged. Gap and value heads and the entire history
encoder stay frozen. Generated marks feed the subsequent history normally;
later gap/value distributions can therefore change indirectly. No post-generation
label reassignment, oracle targets, active-context mask, or true latent state is
used to fit calibration. s is the ordinary observed context, not an active label.

## Fit and data boundary

Only **observed training trajectories** supply features and equality targets.
The fit does not obtain true next marks for counterfactual generated histories.
This is a deliberately limited test; it need not resolve history-distribution shift.
Use all nonfirst valid train transitions, separately by observed context, and
minimize the observable Bernoulli NLL of equality using the full R', not q'.
With frozen f, its parameter-dependent part is also the mark NLL's part.

Use deterministic float64 SciPy L-BFGS-B, analytic gradient, identity initialization
(a=0,b=1), bounds a in [-8,8], b in [.05,20], maxiter=500, maxfun=2000,
maxls=40, ftol=1e-12, gtol=1e-8. One start, no coefficient sweep or validation
selection. Bounds are numerical restrictions on this candidate, not tolerances
chosen from outcomes. Report bound hits. Require finite converged fits and
nonincreasing train NLL within 1e-10 numerical tolerance; otherwise preserve the
failure and stop new dispatch for a documented technical review.

The frozen models have already seen all these training entities and used the
existing validation split for original checkpoint selection. This is in-sample
recalibration and exploratory reuse, **not independent validation**. Test data are
never loaded. New data/seed confirmation requires a subsequent separate protocol.

## Evaluation and prespecified interpretation

Primary: existing native **empirical active repeat-curve L1**, same train-derived
bin definitions, validation reference, 2,048-entity train-derived length/context
plan and paired sampling seed per trial. Reuse saved U/E/L003 reference metrics
and samples; preserve the original generation batch size and numeric settings.
Generate all gap/mark/value trajectories from calibrated models, not hybrid
fixed-gap trajectories. Save samples and confirm plan and output validity.

Secondary: active gap-repeat MI error, transition-conditioned mark TV, the complete
existing metric suite, validation conditional grid/factual mark TV, and both copy
and actual repeat response ranges in all three null contexts. Preserve per-context
results; null averages weight the three null cells equally. Same-history controls
do not establish a unique causal training mechanism.

Prespecified paired contrasts: Ecal-E, L003cal-L003, L003cal-Ecal,
L003-E, Ecal-U, L003cal-U. Report the factorial interaction
(L003cal-L003)-(Ecal-E). Report all five trial values, means, SD, direction counts,
and descriptive paired 95% t intervals (df=4), with no multiplicity-adjusted or
independent-data significance claim. Reusing a trial across prevalences does not
make 20 independent confirmations.

For a **consistent primary improvement signal** (not method superiority), require
negative mean primary differences and at least 4/5 negative paired differences
in at least 3/4 prevalences, for each specified contrast separately. Report
MI/conditional/null costs alongside that signal rather than hiding adverse cells.
For a **joint-improvement lead** over a given comparator, additionally require
nonpositive mean MI change and nonpositive mean three-null copy/repeat-range
changes in the same qualifying prevalences. These are conservative diagnostic
labels, not an application-grounded noninferiority margin. Mixed directions are
reported as a tradeoff; no post-hoc relaxation, summed score, or claim of equivalent
quality. U contrasts determine whether any apparent repair still trails baseline.

If Ecal succeeds but L003cal has no incremental benefit, revise the regularization
claim. If both fail, retain that bounded negative result; do not automatically
increase capacity, change bounds, or add more coefficients. Even a promising result
requires fresh data/seed confirmation and stronger fair external comparisons
before method superiority or real fraud-detection utility can be claimed.

## Verification and execution

CPU tests must cover identity and normalized mark probabilities, first-event and
reserved-code behavior, fresh-channel repeat mass, analytic gradients checked by
finite differences, positive slopes, deterministic fitting and save/reload,
unchanged base weights, train-only fitting inputs, causality, and valid recursive
generation. Same-source CPU and GPU smoke gates precede scientific fits. Compare
identity calibrated and original saved models, including short generated paths.

Commit registration first, then reviewed implementation/tests before execution.
Record source/config/data/checkpoint SHA256, fit inputs and parameters, device,
seeds, optimizer diagnostics, and output manifests. Historical files are immutable.
Run at most two workers, one per available GPU; inspect utilization and compute
clients at every launch, ignore only an idle MPS service, and never terminate others.
Scientific negative results do not stop the grid. Technical failures preserve
outputs and stop new dispatch. Large evidence remains on the workstation;
Git stores implementation, protocol, checked summaries and figures.

## Report correction

The September 18 Notion page was corrected on September 20 to distinguish the
additional history head from basic history use; to include E's native-generation
cost versus U before regularization; and to limit claims to the controlled
synthetic experiment. Existing result tables, figures and oracle caveats remained.
