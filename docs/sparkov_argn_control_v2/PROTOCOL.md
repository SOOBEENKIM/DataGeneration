# Sparkov labelled sequence diagnostic, version 2

Registered before fitting. Scope is baseline measurement and cause discrimination,
not a proposed generator. The earlier engine-2.4 audit remains a separate study.

## Data and task

Use the canonical outer training customers (688, 916567 events) and development
validation customers (147, 177997 events). Never load test events. Validation was
used previously and is not an untouched confirmatory set. Learn gap, merchant,
amount, category and categorical fraud label jointly. The initial gap is encoded
as zero and excluded from gap evaluation. Context is gender, state, birth year,
city population; exclude future-derived entity_any_fraud. Preserve event order.
This gap-based task does not model absolute clock hour or calendar dates.

## ARGN and controls

Use the unmodified engine 1.0.4 runtime verified in the Berka replication, native
analyze/encode, Medium model, flexible generation, window 100, batch heuristic,
optimizer and early stopping. Prepare native customer-level internal split once;
all paired child fits share it and its codec. Fit seeds 20260928/20260929.

Native max_epochs=100 is internally capped at ceil(688/50)=14. The paired
cap-relaxed control changes only that assignment in an isolated source copy;
installed library and native arm remain unchanged. Allow at most 100 epochs or
120 minutes, retain native early stopping and best-checkpoint selection. Record
source diff, hashes, progress and paired prefix equivalence. A setting effect is
not an architectural defect or a novel method.

For each fit generate two independent draws for the 147 validation static
contexts, without supplying actual future labels, transactions or lengths. This
is the primary conditional-generation comparison. Also generate with a common
147-row synthetic parent sample from a separate native parent fit; this measures
sensitivity to context population and is not a matched-person causal comparison.
No output filtering, balancing or repair. Preserve native outputs and all invalid
values. Generated IDs are linking keys, never modeled numerical attributes.

Audit numeric and categorical encode/decode roundtrips before learning. Compare
native vs relaxed fit and initial vs late generated positions. If those controls
do not identify a cause, explicitly leave it unresolved; do not automatically
attribute residual errors to history representation or error accumulation.

## Other comparisons

CPAR: SDV 1.38.0 / DeepEcho 0.8.1, seed 20260928, 128 epochs, CPU, full sequences,
no segmentation, sample_size=1, same modeled fields and outer training customers,
native variable-length generation conditioned on the same validation statics.
It uses all outer training customers; ARGN reserves its native internal validation
set, so the effective optimization data differ and must be disclosed. Existing
source-pinned equivalent PAR loss accelerator may be used after value/gradient
checks; never silently correct native loss conventions. Any sampling acceleration
must preserve the recurrent computation and have an explicit numerical check.
One CPAR fit is an exploratory comparator, not a conclusive model ranking.

Add row resampling and first-order category/label transition resampling as
diagnostic controls. Draw lengths from training customer lengths; current-row
fields are sampled together. These controls do not establish privacy.

## Measurements and interpretation

Report fraud prevalence relative to both training and validation, class-conditional
gap/amount/category relationships, merchant-category compatibility, past merchant
reuse, prior-20-transaction amount ratio, previous-label/current-label transitions,
and positional curves. Fit bins only on training data. Include train-versus-validation
differences, customer-level bootstrap intervals and support counts. Cells with fewer
than 10 customers or 20 frauds are descriptive, not strong rare-condition evidence.
In particular validation has only 45 positive gaps <=5 seconds and one fraud there.
Low-dimensional joint error can conceal minority-class errors; report fraud
conditional errors separately. Keep absolute-hour fidelity, fraud-detector utility,
privacy, other datasets, and a new architecture outside this stage.
