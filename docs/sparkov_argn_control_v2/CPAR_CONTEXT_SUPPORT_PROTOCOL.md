# CPAR categorical-context support amendment

Registered while the original CPAR 128-epoch fit is running, before its generation
and before inspecting generated CPAR quality. A source-level preflight found that
validation includes four customers in states DE, HI or RI, absent from the outer
training context. SDV's PAR categorical transformer is None; DeepEcho 0.8.1 indexes
the observed category dictionary directly. Its native context encoder raises
KeyError for each of these three unseen levels.

Preserve the original 147-customer ARGN evaluation and every stored output.
Do not replace states, add validation categories to the fitted model, or change
training. CPAR's native coverage limitation is itself an implementation finding.
After the completed fit, check every context against its actual native encoder
and record coverage. Generate the 143 supported validation static contexts with
both registered seeds; the fit and its numeric-loss convention remain unchanged.
Use the verified recurrent cache/conversion accelerator to finish generation.
The original fit process may enter native generation; after model.pkl and FIT.json
exist, stop that process before resuming from the saved model, without concurrent
writers. Keep any completed original output or failed attempt as evidence.

Create a separately named common-support comparison: restrict ALL other primary
generated outputs, real validation and roundtrip validation to these same 143 IDs.
The restriction uses training/validation static category support only, not labels
or model quality. Recompute direct relations, raw numeric metrics and customer
bootstrap intervals on that population. Recompute the training-customer reference
using 143 draws. Report this cohort separately; do not combine a 143-customer CPAR
score with 147-customer ARGN scores in a ranking or figure. Common-support coverage
does not establish that CPAR can generate all requested unseen-customer contexts.
