# v2.6 single-factor candidate artifact contract

## Root and immutability

All future output is restricted to:

```text
artifacts/benchmark_v2_6/selection_single_factor/
```

The existing `artifacts/benchmark_v2_6/selection/` tree is input-only.
v2.5 runtime and frozen data are input-only. No old attempt, authorization,
checkpoint, candidate sample, selection report or terminal marker may be
deleted, moved, relinked, overwritten or terminalized again.

The source store uses exclusive create for files. A repeated model,
candidate and seed allocates the next `attempt_NNN`; it never reuses an
existing attempt.

## Layout

```text
selection_single_factor/
  authorization_history/
  trajectories/
    <model>/<trajectory>/seed_2601/attempt_NNN/
      manifest.json
      RUNNING.json
      checkpoints/
      progress.jsonl
      COMPLETE.json | FAILED.json
  candidates/
    <model>/<candidate>/seed_2601/attempt_NNN/
      manifest.json
      validation_sample.npz
      diagnostics.json
      five_guards.json
      COMPLETE.json | INVALID.json | FAILED.json
  aggregate_attempt_NNN/
    selection_report.json
    artifact_index.json
    checksum_manifest.json
    AGGREGATE_COMPLETE.json
```

No directory above is created by this preparation commit.

## Immutable candidate manifest

Before any future work, `manifest.json` must record:

- source commit and relevant source hash;
- single-factor config path and SHA-256;
- parent forensic commit `fc503ac9159668cde342f55aa76b40d61d3e0dc0`;
- model, candidate, seed, execution kind and sole intervention dimension;
- canonical baseline/effective dimension hashes and exact changed-key list;
- frozen train/validation file and content hashes;
- SamplingPlan hash and selection seed;
- threshold and selection-rule hashes;
- authorization path and SHA-256;
- requested updates, hard cap and expected terminal artifacts;
- `test_split_read=false`, `fresh_test_authorized=false` and
  `five_seed_full_run_authorized=false`.

The changed-key list must be empty for a reused control and contain exactly
the preregistered intervention dimension for every other candidate.

## Execution-kind contracts

### Reused frozen control

The manifest references the exact old result, checkpoint and validation
sample hashes. It performs zero training and zero sampling. A new reference
manifest may be written under the new root, but old bytes remain untouched.

### Evaluation-only candidate

The manifest binds the immutable old checkpoint and creates a new sample in
a new attempt. `training_calls=0` and the old checkpoint hash must remain
identical. CTGAN temperature, TVAE categorical decode, or CoF residual
sampling is the only permitted intervention.

The CoF residual-fit state records train file/content hash, base train-plan
sample hash, train/base variance, resulting residual variance and zero-mean
seed. Validation statistics cannot be used to fit or revise that state.

### New trajectory

The trajectory has a 20,000-update and 7,200-second hard cap. It writes
checkpoints and progress only below its own append-only attempt. CTGAN/TVAE
class updates and wall budget are exactly half each. A candidate sample is
valid only after trajectory `COMPLETE.json` and checkpoint hashes verify.

## Validation-only reads

Model fitting, SamplingPlan construction, residual fitting and transform
fitting may read frozen train only. Candidate scoring may additionally read
frozen validation. Data-path validation accepts only `train.npz` and
`validation.npz`; any test, audit-latent, fresh-test or arbitrary data path
fails closed.

Old candidate checkpoints/results/samples are readable only as hash-bound
frozen-control inputs. Writes to the old candidate root, v2.5 runtime, frozen
data, configs or authorizations are rejected.

## Diagnostics and guards

`diagnostics.json` contains the exact model-specific fields in the amendment
config. Diagnostics are descriptive and cannot change eligibility.

`five_guards.json` contains the unchanged five statistics, thresholds and
PASS/FAIL values. A candidate is eligible only when all five pass. An
INVALID/FAILED candidate cannot be silently omitted or replaced.

## Aggregate and authorization

No worker may aggregate. A separately authorized aggregate-only operation
must verify all nine candidate terminal artifacts, hashes and provenance
before creating a new aggregate attempt. `AGGREGATE_COMPLETE.json` is
written last.

Missing, mismatched, non-single-factor, test-accessing or corrupt artifacts
fail closed without producing a selection result. A no-passing-candidate
primary model keeps the v2.6 primary full run prohibited.

The original preparation commit created no authorization. The later
execution-preparation runner keeps authorization and execution separate:

- `plan` validates the immutable nine-candidate mapping and writes only an
  append-only preflight record;
- `dry-run` rehashes source, config, train, validation, the train-fitted
  SamplingPlan, frozen controls, v2.5 provenance and the prior v2.6 candidate
  tree without importing a model backend or querying CUDA;
- `execute` requires one explicit model worker, one CUDA device and a matching
  append-only authorization;
- a worker creates no validation-selection aggregate.

The three model workers own disjoint subtrees. Each worker has exactly one
frozen reference, one checkpoint-only evaluation and one new 20,000-update
trajectory. The frozen and evaluation-only candidates have zero training
calls. A worker fails closed on any source, config, data, SamplingPlan,
checkpoint, frozen-control or preserved-tree hash mismatch.

The execution implementation is isolated in:

- `experiments/single_factor_runner_v2_6.py`;
- `generators/single_factor_backends_v2_6.py`;
- `models/single_factor_components_v2_6.py`;
- `scripts/run_single_factor_candidates_v2_6.py`.

It does not alter the amendment config, threshold, DGP, guard calculation,
selection rule or any old artifact. Validation selection remains a separately
authorized later operation.

## Fixed worker operations

| Worker | Frozen reference | Checkpoint-only evaluation | New trajectory |
|---|---|---|---|
| CTGAN | existing native c01 | categorical temperature 0.5 | shared all-train-fitted transformer |
| TVAE | existing native c01 | categorical decode temperature 0.75 | amount/gap/receiver weights 2/2/1 |
| CoF | existing native c01 | train-fitted variance-deficit amount residual | epsilon-prediction amount objective |

The CTGAN new trajectory still trains two separate class generators with
10,000 updates and 3,600 seconds each; both use deep copies of one transformer
fitted exactly once on all valid train rows. The TVAE new trajectory changes
only loss weights. The CoF new trajectory keeps the frozen sequence denoiser
and discrete path while pairing the epsilon objective with its deterministic
VP reverse equation. The residual candidate derives its variance from valid
train amount variance and the fixed train-fitted SamplingPlan sample; it
never refits from validation.

## Runtime budget and parallelism

Each new trajectory retains the 7,200-second whole-operation hard cap and
each checkpoint-only evaluation retains the 1,800-second cap. The maximum is
7.5 GPU-hours. With the CTGAN, TVAE and CoF workers on three independent idle
GPUs, the hard-cap wall-time ceiling is approximately 2.5 hours. These are
preregistered ceilings, not measured results.
