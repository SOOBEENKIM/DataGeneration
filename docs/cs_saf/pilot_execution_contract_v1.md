# CS-SAF controlled pilot execution contract v1

Fixed before materialization or training on 2026-09-16. This executes the
model/data/CPU/pilot stage requested after the oracle PASS. The earlier exact
architecture, original seed sets and stop thresholds remain in force.

## Candidate implementation and fair comparison

`models/cs_saf.py` implements C0/U0/U1/B0/B1. U0/U1/B0/B1 have identical
parameter keys, shapes and seeded initialization. The gap route and balanced
loss are the only differences. The inherited scalar gate is removed. Direct
static embedding has width 8, history 128, bilinear rank 32. The controlled
vocabulary excludes PAD/UNK/MISSING from both training and generation.

Repeat auxiliary supervision uses observed mark equality with the model's
complete repeat probability q+(1-q)*p_new(previous), not the latent copy coin.
With B uniformly sampled full entities from N train entities, the auxiliary
estimator is

    (N/B) * sum_(entity i in batch) sum_(t>0 in i) BCE_it / (2*T_label(i)).

T_label is the fixed train transition count. This is unbiased for the specified
macro-context transition loss despite varying lengths or single-context
batches. No local batch prevalence normalization is used. All sequences fit
one 32-event window; multiwindow input is rejected. Base loss keeps the
inherited per-component valid-event means. Checkpoint selection uses global
validation sums/counts for gap, mark and value, excluding auxiliary loss.

## Prevalence materialization without changing the DGP law

Keep the original full entity split assignment exactly. Only train/validation
event/static content is read and written; test membership is metadata only.
No test events or labels are read or materialized in the new grid.

Reconstruct the original production label uniforms with seed 42, skipping
draws for unrequested entity IDs. Verify the original label equals U<0.05.
Set each new label to U<pi for pi=0.05/0.10/0.25/0.50, producing nested groups.
There is no label-dependent resplitting or rare-context oversampling.

Preserve every original gap, amount, timestamp, length and entity identity.
At kappa=0 labels do not affect marks, so every mark also stays unchanged.
At kappa=1 only newly promoted active entities receive new mark paths driven
by their original gap regime, with the production q_low/q_high and uniform
fresh-mark law. Existing active paths are unchanged. A fixed per-entity random
stream (namespace `cs-saf-prevalence-v1`, seed 20260930) gives each newly active
entity the same path wherever it is active in the grid. This changes random
draw realization, not the production conditional law.

Realized gap regimes are loaded with train/validation row predicates **only
inside the DGP materializer**. No oracle fields are written to model data or
training caches. The pi=0.05 development content is checked to equal the
original exactly. New views have separate directories and manifests; originals
are immutable. Each resulting train cell must pass the previously frozen
observable-information oracle tests before training. Models never see the
regimes or oracle predictions.

## CPU prerequisite

All five candidate contracts must pass finite loss/backpropagation, support,
vocabulary, matched-capacity, bilinear-gradient, strict-past and balanced-loss
tests. Two fresh CS-B1 CPU initializations use the same 64-entity train subset
(32/context) and 32-entity validation subset (16/context), 25 epochs, batch 16,
lr=.003, patience=25. These are implementation gates, not scientific evidence.
Training histories and best-state tensors must match exactly, train objective
must decrease at least 10%, and checkpoint reload/generation/zero-gap checks
must pass. Production dimensions remain unchanged.

## Registered single-seed pilot

Model seed is 20260930. For each prevalence, train CS-U1 (ordinary objective)
and CS-B1 (balanced objective) at kappa=0 and 1: four jobs, on four distinct
available GPUs. Use float32, AdamW lr=.001, weight decay=1e-5, batch=512,
gradient clip=1, maximum 50 epochs, early-stopping patience=5, no scheduler.
Each pair has identical seeded initial tensors and shuffled entity order;
realized epochs can differ because of early stopping.

Process prevalences in increasing order. After completing the four jobs at a
prevalence, evaluate **CS-B1** against the original pilot criteria, on both
latent-copy and observable-repeat response. CS-U1 is a descriptive comparator,
not an additional post-hoc selection opportunity. C0/U0/B0 are implemented and
CPU-tested; their multi-seed performance comparison belongs to the later
confirmatory family, not this pilot.

Audit all validation nonfirst histories against every train-fitted gap bin.
Average histories within entity and then entities within context. Require:

- kappa=1/context=1 mean response >=0.05;
- all three null-cell means <=0.05;
- active response >=2 times the largest null response;
- same-state-dictionary zero-gap control max response <=1e-8;
- zero gap-support violations in generated samples.

Repeat those response checks on the observable repeat probability so latent
mixture reparameterization alone cannot pass. Inspect finite values and valid
marks as implementation guards. Generate 2,048 entities using a shared
train-only empirical parent/length plan, seed 20260930. Validation identities,
labels and lengths never set this generation plan.

If CS-B1 fails at any prevalence, record FAIL and stop all later-prevalence
training. Maximum workload is 16 model fits; first-stage failure stops at four.
Do not retry with different losses, seeds, epochs or thresholds. A technical
execution failure is recorded separately and can be corrected without turning
a scientific failure into a retry.

## Boundary and provenance

The code/config commit must precede materialization and execution. Output
directories are immutable; progress, best checkpoint, input/config hashes and
terminal reports are retained. The pilot supervisor verifies the CPU gate's
source commit and config and the data oracle gates before launching jobs.

This stage does not run five-seed confirmation, real-data models or held-out
evaluation. Even a pilot PASS cannot establish superiority, learned dilution,
or noninferiority. The previously recorded noninferiority-calibration problem
and exact confirmatory endpoint aggregation remain prerequisites for that
separate next stage.
