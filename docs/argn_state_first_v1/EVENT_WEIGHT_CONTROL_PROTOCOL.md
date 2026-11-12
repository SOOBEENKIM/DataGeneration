# Adaptive event-loss weighting control

Registered after initial B/B+S evaluation, before this control's fitting.

The initial full-history B/B+S run produced severely excessive first-event fraud.
In the 619 optimization customers, 49/619 first events are fraud (7.916%), but
the native transaction loss assigns fraud 90.398% of first-event loss mass:
each row's weight is 1/L_i inside a customer, and short histories are often all
fraud. Across all events, fraud is 0.581% by event count but 8.572% by native loss
mass. This is an identified property of the data/objective combination. It is a
candidate explanation, not yet proof of the generation failure's cause.

Train **B_event_weighted**, with state features disabled, from scratch for the
same two fit seeds. Reuse the exact codec, 619/69 customer split, full encoded
histories, column order, initialization, optimizer, batch/accumulation sizes,
60-epoch/180-minute budgets and native generation settings from B. Keep all
original unsuccessful fits and output files. Do not change A or the state module.

For transaction output column c and customer i with L_i retained events,
native loss is mean_t CE(i,t,c). Replace it by

`(L_i / mean_train_length) * mean_t CE(i,t,c)`.

The mean retained optimization length is computed once from frozen encoded
optimization records. This gives every observed transaction the same loss weight
in expectation under native uniform-customer minibatching, without label-based
reweighting, oversampling or a target prevalence. Padding is masked. Native
sequence-length loss is unchanged, and masked positional losses remain zero.
The length-head *loss* is unchanged; its shared upstream parameters may still
change through the altered transaction gradients. Do not claim a frozen length
distribution. AdamW/early stopping are unchanged, but their selected checkpoint
can differ under the modified objective. Do not compare the two objectives'
numerical validation losses as if they had the same meaning.

Generate two datasets per control fit, with the original static contexts and
native synthetic lengths. Then use the same episode/relationship/guardrail
metrics as the first comparison, including first-event fraud and full prevalence.
No prevalence matching or maximum fraud length is imposed during generation.

If this changes the failure, it supports an objective-related explanation under
this Sparkov setting. It does not demonstrate a general architectural limitation
of ARGN, or establish a new contribution. Further partial/absent improvements
must be reported. Only after this baseline issue is characterized should the
same objective be considered for a new B versus B+S pair.
