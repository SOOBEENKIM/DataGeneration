# Evaluation implementation record

2026-09-27. Written after the four fits/eight generations finished and before
opening their quality measurements. The first-study protocol remains unchanged.
These are development diagnostics, not newly invented confirmatory acceptance
thresholds. Completion/event-count metadata was already inspected.

All eight outputs use the same 147 development contexts and native synthetic
lengths. Compare against the frozen development transactions; read outer-training
transactions only for inherited metric bins and a real-data reference. Do not
access test events. Verify generated files and checkpoints against recorded
hashes before evaluating; never modify outputs or fitted models.

## Operational definitions

- A fraud run is a maximal consecutive block of fraud-labelled transactions for
  one customer. Report its observed count and first-to-last fraud span in seconds.
  That span is not the exact time at which fraud started/ended between events.
- A run starting on the first observed customer row is left-boundary/unknown-start.
  A run ending at the last row is right-boundary/unknown-end. Report both counts;
  completed known-start run distributions require neither boundary. All-observed
  and completed-only distances are reported separately, with support counts.
- Termination at run age j uses current fraud rows with an observed binary next
  label; numerator is a next normal row. Exclude sequence end, not count it as a
  fraud termination. For run-age hazards also exclude left-boundary runs. Report
  continue and end probabilities and denominators for bands 1, 2–5, 6–10, 11–20,
  21–50, 51+. These observed-next probabilities are not an unbiased latent survival
  estimator when sequence censoring depends on fraud behaviour.
- Onset is P(next fraud | current normal, next row observed). Overall continuation
  is P(next fraud | current fraud, next row observed), with boundaries reported.
- Personal amount ratio is current raw amount / median of the preceding 20 amounts,
  requiring at least five positive-history observations. No current amount enters
  the denominator. Compare log1p distributions by label and fraud-run age. Missing
  ratios are excluded and their denominators are disclosed. Age-standardization
  reweights generated fraud-age bands to development proportions; if a reference
  band is absent in generation, report incomplete coverage, not a fabricated score.
- Retain prior training-derived amount/gap bins and direct relationship TV checks.
  Report normal/fraud gap, category, merchant and amount relations; also raw amount
  tails, prevalence, customer lengths, observed merchant diversity and invalid data.
  Sequence-length strata are retrospective diagnostics, never generation inputs.

## Uncertainty and comparisons

Use 500 bootstrap samples of customer IDs (seed 20260927), carrying each customer's
entire event history. For real-data variability, compare resampled development
customers to the fixed original development reference. The resulting real-real
distance interval is a plug-in variability reference, not a universal pass threshold.
Report scalar prevalence/onset/continuation intervals and real-real observed-run
log1p-Wasserstein distances. Personal-ratio real-real distances are also reported.

For paired B versus B+S error differences, resample the same 147 static-context
customer IDs in the reference and both generated outputs. These bootstrap intervals
are conditional on the trained model and generation draw. They do not estimate
training-seed uncertainty. Keep four draw-pair results and average the two generation
draws *within each of the two training seeds* before describing consistency. No
p-value, multiple-metric best-case claim or four-independent-model claim is made.

Evaluation report must include negative/mixed effects and trade-offs. Do not select
one favourable checkpoint, output seed or fraud-age subgroup. No amount-head change
or new large training grid is justified solely by a lower training loss.
