# Benchmark v2.5 full-experiment preregistration

Status: **prepared and frozen; execution is not authorized**.

Amendment 1,
`preregistered_full_experiment_v2_5_amendment_1.md`, supersedes the C2
multiplicity families, shared SamplingPlan identity, plug-in state baseline
names, CoF engineering configuration, and prospective CoF runtime estimate
below. All unaffected provisions remain frozen.

This document fixes the v2.5 full-experiment design before any v2.5 full,
five-seed, learned-model, or sweep run. The v2.4 CPU gate and four one-seed GPU
smokes are historical evidence only. No model result may change the v2 DGP,
κ, primary endpoint, threshold, AUROC operator, gate rule, model budget, or
v2.5 analysis below.

## Freeze and provenance

The preparation preflight observed:

- branch `benchmark-v2-redesign`;
- v2.4 completion HEAD
  `35ec654061f99e8e5ccd54381d40329546ed2eb7`;
- lightweight tag `benchmark-v2.4-gate-pass` at that commit;
- only the three pre-existing unstaged image deletions in the worktree;
- v2.4 artifact index SHA-256
  `c99c79b501ba264b01f1ef54f1bb7fa1c40446c865d95b6078fad6fd47fec08a`;
- exactly 1,679 records in that index;
- indexed and actual SHA-256 for
  `artifacts/benchmark_v2_4/gates/gate_report.json` both equal
  `53fc93c3650e5e6d03f9a085c74a5ad86ef2862f7a918668bce817500319ee14`;
- the preserved v2.4 gate report retains
  `full_experiment_authorized=false`.

The v2.4 index, artifacts, gate report, preregistration, source commit, and
smoke artifacts are read-only inputs. v2.5 runtime output may use only:

```text
artifacts/benchmark_v2_5/
data/benchmark_v2_5/
```

Runtime data and artifacts are ignored by Git. Only source, configuration,
tests, and documentation belong in the v2.5 preparation commit.

## Confirmatory scope

The primary experiment is exactly:

```text
scenario = joint_semimarkov_v2b
kappa = 1.0
model/sampling seeds = 1, 2, 3, 4, 5
```

κ=0, a κ-curve, v2a, additional seeds, and any sweep are outside this
experiment. Observing the κ=1 results does not authorize an extension. Any
additional κ experiment requires a separate v2.6 preregistration made before
that experiment.

The DGP parameters are unchanged from v2.2/v2.4. The frozen dataset has 31,951
train entities, 7,989 validation entities, and 7,989 test entities. All model
seeds use the same train/validation/test data hashes. Validation is a separate
DGP split used only to report a fixed monitoring metric; it cannot control
training or select a checkpoint. The test split is used only for evaluation.

Bin edges, representative gap values (`tau`), the short-gap threshold,
coherence/bin references, direct row-marginal references, label prevalence,
and the label-conditional sequence-length policy are fitted using train only.
One shared sampling plan draws 7,989 labels from train prevalence and lengths
from the train label-conditional empirical length distribution with fixed
seed 10,001. Every generator and all model seeds use its exact full SHA-256.
Test labels and test lengths do not define the plan.

## Primary endpoint

For each entity, `joint_alignment` is the covariance-like alignment between
the train-thresholded short-gap indicator and receiver-repeat indicator over
valid transitions. Let

```text
delta_joint(batch)
  = mean(joint_alignment | y=1)
    - mean(joint_alignment | y=0).
```

The sole primary endpoint is the continuous association-recovery error:

```text
abs(delta_joint(real test) - delta_joint(synthetic)).
```

Lower is better. The real-test delta is a reference value, not a fitted
threshold. No binned metric replaces or modifies this endpoint.

All models receive identical C0/C1 reference handling, train-fitted
thresholds, support checks, invalid-generator routing, direct row-marginal
hard guards, canonical prefix mask, zero-padding contract, label plan, entity
count, and length plan. A support or mask failure makes the generator
`INVALID`; its `NaN` is never silently removed from an aggregate.

Direct generator row-marginal cutoffs are distinct from, and do not alter, the
preserved v2.4 AUROC threshold. Before model execution, 200 train-only
full-sequence empirical resampling pairs of 7,989 entities calibrate amount/gap
two-sample KS, absolute standardized label effects, and maximum entity-balanced
signed receiver-frequency statistics. Each cutoff is the observed rank
200/200 maximum without interpolation; receiver frequency also retains the
pre-existing practical margin 0.02. Calibration seed is 24,500. No learned
sample contributes to these cutoffs.

## Frozen generator set

The complete generator registry is:

1. label-conditional empirical i.i.d. row bootstrap;
2. contiguous block bootstrap at lengths 2, 4, and 8;
3. full-sequence block empirical reference;
4. class-conditional independent Markov;
5. class-conditional joint observed Markov;
6. class-conditional plug-in HMM (`plug_in_hmm`);
7. class-conditional plug-in HSMM (`plug_in_hsmm`);
8. separate-class-model CTGAN;
9. separate-class-model TVAE;
10. class-conditional neural sequence model;
11. CoF-SeqGen.

Block lengths 2/4/8 are separately materialized generator cells. The
full-sequence block is an empirical oracle/reference, not a fair learned-model
competitor. It is labeled as such in every table.

CTGAN and TVAE are not called “single conditional models.” Each baseline fits
one y=0 model and one y=1 model. Generation fixes an entity label and length
first, samples i.i.d. rows from the corresponding class model, and assembles
the sequence. The two class models share one baseline-level compute budget.
The precise definitions are frozen in
`baseline_definitions_v2_5.md`.

An adapter that cannot run or cannot satisfy the common contract is not
omitted. It writes `FAILED.json` or `UNAVAILABLE.json` with the reason, and the
full experiment stops.

## Training budget

Every learned generator has the same maximum of 2.0 GPU-hours per seed at the
baseline level:

| Generator | Requested updates | Batch | Optimizer | Maximum |
|---|---:|---:|---|---:|
| CTGAN class pair | 5,000 per class | 500 | Adam, GAN defaults fixed in config | 1 h y=0 + 1 h y=1 |
| TVAE class pair | 10,000 per class | 500 | Adam, fixed in config | 1 h y=0 + 1 h y=1 |
| neural sequence | 20,000 total | 256 entities | Adam, lr 0.001 | 2 h total |
| CoF-SeqGen (128/2/50) | 20,000 total | 256 entities | Adam, lr 0.001 | 2 h total |

Only train data updates model weights. Validation metrics are logged at fixed
intervals but do not stop, extend, rank, or select training. Early stopping,
result-dependent hyperparameter changes, model-specific extra training, and
checkpoint selection by validation outcome are forbidden. Training stops at
the requested updates or the wall-clock cap, whichever occurs first. Actual
updates, wall time, and peak GPU memory are reported unchanged.

The checkpoint interval is every 100 updates. This is at least as frequent as
both 100 updates and 10% of every requested schedule. CPU baselines have no GPU
budget.

### Superseded prospective runtime estimate

These estimates use the already-preserved v2.4 bounded-smoke throughput only
for scheduling, with conservative overhead for full-data transforms,
checkpointing, evaluation, and process startup. They do not alter any
scientific rule.

| Learned generator | Expected GPU-h / seed | Five-seed expected | Five-seed hard cap |
|---|---:|---:|---:|
| CTGAN class pair | 0.60 | 3.00 | 10.00 |
| TVAE class pair | 0.60 | 3.00 | 10.00 |
| neural sequence | 0.20 | 1.00 | 10.00 |
| CoF-SeqGen | superseded; capacity preflight pending | pending | 10.00 |
| **Total** | **pending** | **pending** | **40.00** |

| Concurrent GPUs used | Expected learned wall time | Worst-case cap wall time |
|---:|---:|---:|
| 1 | pending capacity preflight | 40 h |
| 2 | pending capacity preflight | 20 h |
| 3 | pending capacity preflight | 14 h |

The discrete worst-case calculation treats the twenty model/seed runs as
2-hour jobs; three GPUs therefore require seven waves on the most-loaded GPU.
These are planning estimates, not stopping thresholds. The requested updates
are expected to be meaningful within 2 hours; if a future preflight disproves
that before launch, the cap is not increased and the full experiment remains
stopped.

## Attempt, failure, and resume policy

Each run is isolated at:

```text
artifacts/benchmark_v2_5/full/
  joint_semimarkov_v2b/
    kappa_1.00/
      <generator>/
        seed_<1..5>/
          attempt_001/
```

The immutable manifest binds source, config, code, all three data hashes,
generator, seed, GPU/software environment, requested budget, baseline
definition version, and evaluation version. A running attempt has immutable
`manifest.json` and `RUNNING.json`, atomic checkpoints, an atomic
`checkpoints/latest` pointer, append-only progress/logs, and replace-atomic
partial metrics.

An interrupted attempt resumes only when the entire manifest matches and the
latest checkpoint hash verifies. A mismatch, OOM, code error, or prior
terminal failure allocates `attempt_002` and preserves `attempt_001`.
Artifact damage and ambiguous terminal state fail closed. The full contract is
in `full_experiment_artifact_contract_v2_5.md`.

## Statistical analysis

No stochastic baseline or learned model is aggregated before all seeds 1–5
have valid terminal artifacts. Partial results are only progress information.
They cannot select, stop, extend, or rank models.

For each valid comparison:

```text
effect = mean primary error of baseline
         - mean primary error of CoF.
```

A positive value means CoF has lower error. Report the raw five errors, mean,
sample standard deviation, Hedges g in the same direction, an unpaired
bootstrap 95% interval for the mean effect, and a two-sided Welch t-test.

Amendment 1 defines the exact C2 primary Holm family as CTGAN, TVAE, and
empirical i.i.d., and a separate secondary Holm family for neural sequence,
independent Markov, joint Markov, `plug_in_hmm`, and `plug_in_hsmm`. I.i.d.
and block bootstraps remain mandatory structural references. The full-sequence
reference is always marked oracle/reference-only.

## Permitted stopping

Partial performance, whether favorable or unfavorable, is never a stopping
reason. Only these conditions stop execution:

1. hard data, mask, support, or generator contract failure;
2. the fixed GPU-hour cap;
3. the same infrastructure error on three consecutive attempts;
4. manifest-hash mismatch or artifact corruption.

A code defect must be reported with cause, affected runs, and edit scope before
the fix. After a source commit, every affected seed restarts in a new attempt.
No fix may change the DGP, endpoint, threshold, baseline definition, or
training budget.

## Authorization boundary

This preparation commit does not authorize or launch the experiment.
`full_experiment_authorized=false` remains true in the preserved v2.4 gate
report and is independently represented as false in the v2.5 config. A later
explicit instruction is required before any full, five-seed, sweep, or
`FULL_EXPERIMENT` command may run.
