# Exploratory frozen-generation-order control

Registered after the initial native/cap-control fits and before these outputs.
The first diagnostics show a large merchant/category error and residual
fraud-conditional error. Test sensitivity to ARGN's flexible within-row order;
this is an adaptive diagnostic, not a confirmatory model-selection experiment.

Use both completed cap-relaxed checkpoints, the same 147 validation static
contexts, and generation seeds 20261011/20261012. Keep the positional/length
column first. In separate controls move (a) category or (b) fraud label to the
next position; retain the relative order of all other columns. Change no weights,
temperatures, class probabilities, length targets, value protection or outputs.
No true validation labels or future lengths are supplied. Copy workspaces so
generation calls do not compete for a shared output directory. Log source and
checkpoint hashes and restore the function after each call.

Primary descriptive comparisons are merchant-category TV for category-first,
and fraud-conditional category/amount TV for fraud-first. Report all prespecified
metrics and both draws/seeds. Order sensitivity indicates imperfect consistency
between learned conditional factorizations. It does not by itself prove why
the conditionals were learned imperfectly, architectural impossibility, or a
new generation method. These are configurations of the existing baseline.
