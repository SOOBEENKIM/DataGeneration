# Preregistered v2.6 train/validation model selection

## Status

This document freezes **preparation and runner implementation only**. It does
not authorize candidate training, the validation-selection command,
fresh-test generation or evaluation, a five-seed run, or a full experiment.
It is a result-blind amendment to preparation commit
`6e59ef2d11442395f8e314ca126d780d9e4cb62a`.

The completed v2.5 state remains read-only:

- FINAL_COMPLETE and forensic source commit
  `5502f84974133c035eb16ea72487e4a1dddd206b`;
- v2.5 config SHA-256
  `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3`;
- v2.5 endpoint, thresholds, DGP, test split, runtime artifacts, and
  `C2=NOT EVALUABLE`.

The v2.5 test result motivates the mechanism classes in the candidate list
but must not select among candidates. Selection uses train and validation
only.

## Development data

The development manifest contains exactly two splits:

| Split | File SHA-256 | Content SHA-256 |
|---|---|---|
| train | `c67a6fce4593317e11e1b6f36397bfb38f9607916bf8f5b32329d8fa99b700a8` | `0c03f179930cd4d89ccc8a8b66283e620313e16d15fd53f0674c133bcb80c09d` |
| validation | `68c15c05c55742bfce0f22560ffe8378d032a7a9fd9a1072e80ec8da5340fba5` | `aa1576b4ccae6a2ca30d0472f0782e1231ecc61ab3d224590a772dc3448b9e66` |

The common selection SamplingPlan is regenerated deterministically from
train only:

- entity count: 7,989, equal to validation entity count;
- seed: 26001;
- plan hash:
  `862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27`.

Validation labels and lengths are not copied into the SamplingPlan. Test data
has no path or hash in the development manifest and must not be opened.

## Candidate family, roles, and equal budget

The candidate-selection roles are fixed before candidate execution:

- primary C2 learned models: `ctgan_separate_class`,
  `tvae_separate_class`, and `cof_seqgen`;
- fixed nonlearned C2 reference: `empirical_iid`, which is not a candidate
  training target;
- secondary learned model: `neural_sequence`.

Each learned model has four evaluation candidates but only three training
trajectories. The common resource contract is:

- one development seed: 2601;
- 20,000 baseline-level requested updates per trajectory;
- checkpoints every 100 updates;
- eligible evaluation checkpoints fixed at 10,000 and 20,000;
- maximum 7,200 GPU seconds per trajectory;
- three trajectories and at most 6 GPU-hours per model;
- 12 trajectories and at most 24 GPU-hours across all four model families;
- four evaluation candidates per model and 16 evaluation candidates total;
- no early stopping, budget extension, replacement candidate, or partial
  candidate selection.

For each model, c00 and c01 have identical native representation, loss,
sampling, seed, update ceiling, and wall ceiling. They are two checkpoint
views of one native 20,000-update trajectory: c00 samples checkpoint 10,000
and c01 samples checkpoint 20,000. They are not trained twice. c02 and c03
each use one separate 20,000-update trajectory.

For CTGAN and TVAE, 20,000 baseline updates means 10,000 for each of the
`y=0` and `y=1` class models. Each class receives half the wall cap. Training
continues to the common 20,000-update maximum even when the candidate's
preregistered evaluation checkpoint is 10,000; this prevents a checkpoint
candidate from receiving a different resource ceiling. A baseline-level
10,000 checkpoint is a balanced bundle of the two class-model checkpoints at
5,000 updates each; the 20,000 checkpoint bundles 10,000 updates per class.

The evaluation-candidate count and trajectory budget are therefore identical
by model. Actual runtime is reported without compensating or extending a
faster or slower model. Sharing c00/c01 removes duplicate training without
changing either candidate ID or the deterministic selection rule.

## Finite candidate definitions

The full executable mappings are in
`configs/benchmark_v2/selection_v2_6.yaml`. The compact definitions are:

### CTGAN

| ID | Amount representation | Loss weighting | Categorical rule | Selected checkpoint |
|---|---|---|---|---:|
| c00 | native + train GMM | native joint adversarial | Gumbel temperature 0.2 | 10,000 |
| c01 | native + train GMM | native joint adversarial | Gumbel temperature 0.2 | 20,000 |
| c02 | train z-score + train GMM | native joint adversarial | Gumbel temperature 0.2 | 20,000 |
| c03 | train z-score + train GMM | native joint adversarial | Gumbel temperature 0.5 | 20,000 |

### TVAE

| ID | Amount representation | Loss weighting | Sample rule | Selected checkpoint |
|---|---|---|---|---:|
| c00 | native + train GMM | native 1/1/1 | latent scale 1, upstream inverse | 10,000 |
| c01 | native + train GMM | native 1/1/1 | latent scale 1, upstream inverse | 20,000 |
| c02 | train z-score + train GMM | native 1/1/1 | latent scale 1, upstream inverse | 20,000 |
| c03 | train z-score + train GMM | amount/gap/receiver 2/2/1 | latent scale 0.75, categorical temperature 0.75 | 20,000 |

### Neural sequence

| ID | Amount representation | Loss weighting | Sample rule | Selected checkpoint |
|---|---|---|---|---:|
| c00 | native | 1/1/1 | multinomial temperature 1 | 10,000 |
| c01 | native | 1/1/1 | multinomial temperature 1 | 20,000 |
| c02 | train z-score | 1/1/1 | multinomial temperature 1 | 20,000 |
| c03 | train z-score | 2/2/1 | multinomial temperature 0.75 | 20,000 |

### CoF-SeqGen

| ID | Amount representation | Loss weighting | Sample rule | Selected checkpoint |
|---|---|---|---|---:|
| c00 | native diffusion | amount/gap/receiver/label 1/1/1/1 | DDIM 50, final argmax | 10,000 |
| c01 | native diffusion | 1/1/1/1 | DDIM 50, final argmax | 20,000 |
| c02 | train z-score diffusion | 1/1/1/1 | DDIM 50, final argmax | 20,000 |
| c03 | train z-score diffusion | 2/2/1/1 | feedback/final categorical temperature 0.75 | 20,000 |

Train-only z-score uses valid train rows, records mean and standard deviation
in the checkpoint, and applies the exact inverse before validation. Zero
variance makes the candidate INVALID. It never refits on validation.

The candidate adapters are versioned v2.6 implementations in new source
files. They do not edit the frozen v2.5 adapter files. They implement:

- valid-train-row-only z-score fitting and checkpointed reversible state;
- CTGAN/TVAE class-pair half-budget enforcement;
- exact preregistered neural/CoF loss weights and categorical temperatures;
- CoF c03 feedback and final categorical temperature;
- a fixed train-fitted SamplingPlan assertion before validation sampling.

The bounded candidate runner is also implementation-only. Construction,
planning, and fixture tests do not instantiate a model, call fit/sample,
query CUDA, or invoke the DGP. Actual execution requires a separate matching
authorization manifest.

### Model-level parallel worker amendment

The 12 frozen trajectories are partitioned by the already preregistered
model boundary, without changing a candidate definition or resource:

- CTGAN: three trajectories and four evaluation candidates;
- TVAE: three trajectories and four evaluation candidates;
- Neural: three trajectories and four evaluation candidates;
- CoF: three trajectories and four evaluation candidates.

Every worker requires an explicit `--model` from this closed set. Each model
has an exclusive, append-only ownership lock and worker manifest. A duplicate
worker for the same model and provenance is rejected whether the first worker
is RUNNING or terminal; different models remain independent. Each native
c00/c01 pair still shares one 20,000-update trajectory and samples only its
fixed 10,000 and 20,000 checkpoints.

Model workers never perform validation selection or finalization. A separate
aggregate-only command is permitted to begin only after all four model
workers are terminal COMPLETE and all 16 candidate artifacts and hashes are
present. Any missing, FAILED, mismatched, or corrupt worker/candidate state
fails closed without producing a selection report. The all-five-guard
eligibility rule, primary/secondary policy, thresholds, train-only
SamplingPlan, z-score contract, and test exclusion remain unchanged.

This amendment changes orchestration only. It adds no candidate, update,
checkpoint, budget, validation rule, test access, or result-dependent
choice. Four simultaneously available GPUs give a hard-cap ceiling of about
6 hours; three GPUs give a two-wave ceiling of about 8 hours.

No other values may be searched:

- no architecture width/depth candidate;
- no optimizer or learning-rate candidate;
- no threshold candidate;
- no validation-derived additional candidate;
- no test-derived transform, temperature, loss weight, or checkpoint.

## Validation metrics

Each completed candidate must generate one sample using the common train-fit
SamplingPlan. The selection harness compares that sample with validation and
records:

1. amount KS;
2. gap KS after the frozen `tau[dt_bin]` map;
3. absolute standardized amount label effect;
4. absolute standardized gap label effect;
5. maximum entity-balanced absolute signed receiver frequency.

The train-calibrated thresholds are copied unchanged:

| Metric | Threshold |
|---|---:|
| amount KS | 0.006081138155655141 |
| gap KS | 0.006387882975686154 |
| amount effect | 0.0363693454591819 |
| gap effect | 0.051540527275560376 |
| receiver frequency | 0.02 |

This use does not alter or recalibrate the v2.5 thresholds. The source
calibration artifact SHA-256 is
`a241491deee47a4b0fc6dc22e561c8fbf304086ed04dd4dca1ace8440583b27b`.

## Selection rule

The rule is fixed before observing any candidate result:

1. A candidate is eligible only if **all five** validation guards PASS.
2. Among eligible candidates, minimize
   `max(amount_ks, gap_ks)`.
3. If tied, minimize `amount_ks + gap_ks`.
4. If still tied, choose lexicographically smaller candidate ID.

If a primary learned model has no eligible candidate:

- its state is `NO_PASSING_CANDIDATE`;
- the v2.6 primary C2 full run is prohibited;
- no alternate candidate is added;
- no threshold, weight, temperature, update count, or selection rule changes;
- the cause is reported and work stops.

If only `neural_sequence` has no eligible candidate, its state is
`NO_PASSING_CANDIDATE` and the neural secondary comparison is
`NOT_EVALUABLE`. This does not block a later, separately authorized primary
C2 full run when all three primary learned models selected eligible
candidates.

Partial candidates or partial seeds cannot select a configuration.

## Test-access exclusion

The harness accepts only the dedicated development manifest, whose split keys
must be exactly `train` and `validation`. It rejects:

- any additional split key;
- a path with a test-split path component;
- candidate samples outside the v2.6 candidate root;
- missing or mismatched train/validation file/content hashes;
- a SamplingPlan not derived from the frozen train policy;
- candidate artifacts with mismatched source/config/data/plan hashes.

It does not import the full evaluator, start a model, allocate CUDA, generate
data, or call `fit`/`sample`.

## Freeze and later authorization

Selection freeze requires all three primary learned models to select a
candidate. A neural selection is recorded when available; otherwise the
freeze records neural `NO_PASSING_CANDIDATE` and secondary
`NOT_EVALUABLE`. The freeze records:

- selected candidate definition and checkpoint hashes;
- selection config SHA-256;
- candidate implementation commit and relevant source hash;
- development manifest SHA-256;
- train/validation file and content hashes;
- selection SamplingPlan hash.

The freeze deliberately records:

- `test_split_hash=null`;
- `test_split_read=false`;
- `fresh_test_authorized=false`;
- `five_seed_full_experiment_authorized=false`.

A later, explicit user authorization is required to define/generate a fresh
v2.6 test split and run five seeds. The test split cannot be defined before
selection freeze.

## Frozen preparation hashes

At this preparation state:

- selection config SHA-256:
  `0884a0144f74f6317ae9e636c0cf5657dedac21ff12cdf545a6b6e5e29be4c4d`;
- development manifest SHA-256:
  `31ddf45dcf7b4f982ed990ec94281c3e6bad0454ec3182e2434e42358006ade5`.

These hashes identify the preregistered candidate family. Candidate execution
remains unauthorized.

## Amendment verification

The implementation commit is accepted only after focused tests,
repository-wide pytest, compileall, and `git diff --check` pass. No candidate
result may be produced or inspected while performing those checks.

Preparation execution counters:

- candidate training: 0;
- validation selection command: 0;
- fresh-test definition/generation/evaluation: 0;
- five-seed GPU run: 0;
- full experiment, sweep, and retry: 0.
