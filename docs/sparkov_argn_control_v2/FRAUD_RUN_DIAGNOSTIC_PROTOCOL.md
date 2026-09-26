# Exploratory fraud-run composition diagnostic

Registered after all ARGN generation and its raw-scale evaluation, before this
diagnostic computation. CPAR training is still running. Both-numeric DIGIT moves
fraud amount means closer to validation while increasing fraud continuation in
some draws and worsening amount/past-median relations. Locate that residual error
without fitting or changing any generator.

For real training/validation and every primary generated dataset, measure lengths
of observed consecutive fraud runs within each customer. Report runs touching the
first or last observed transaction: these lengths are observation-window lengths,
not uncensored fraud-episode duration estimates. Separate consecutive-fraud age
1, 2–5, 6–10, 11–20, 21+; measure event shares and current amount/prior-20 median
ratios with the same strict-past/minimum-five-history rule as the main evaluation.

Compare log1p amount-ratio distributions within those age bands and after
descriptively reweighting the generated bands to the real validation band shares.
If a real band has no generated support, do not report a fully standardized score;
report support coverage instead. This reweighting is an analysis only, not a new
synthetic dataset or a correction of model output. It does not establish a causal
effect of changing labels: within-band composition and numeric learning can also
differ. No model architecture, objective, sampling probabilities or labels change.
