# v2.6 validation-selection artifact contract

## Purpose

This contract governs future train/validation candidate artifacts. It does
not authorize their production. All paths are under new v2.6 roots; no v2.5
artifact may be moved, overwritten, linked as an output, or terminalized
again.

## Shared-trajectory and candidate layout

```text
artifacts/benchmark_v2_6/selection/candidates/
  _workers/
    <model_id>/
      ownership.lock
      attempt_001/
        worker_manifest.json
        RUNNING.json
        WORKER_COMPLETE.json | WORKER_FAILED.json
  _trajectories/
    <model_id>/
      <shared_trajectory_id>/
        seed_2601/
          attempt_001/
            manifest.json
            progress.jsonl
            checkpoints/
              step_10000.pt
              step_20000.pt
            TRAJECTORY_COMPLETE.json
            COMPLETE.json | FAILED.json
  <model_id>/
    <candidate_id>/
      seed_2601/
        attempt_001/
          candidate_result.json
          validation_sample.npz
          COMPLETE.json
```

All writes are exclusive or append-only. `attempt_001` is never overwritten;
a later authorized infrastructure retry must claim `attempt_002`. The
selection harness reads the preregistered fixed attempt. Changing that read
policy requires a result-blind amendment before candidate execution. A
failed scientific candidate is not retried or replaced.

There are exactly 12 training trajectories and 16 evaluation candidates.
Within every model, c00 and c01 reference one shared native trajectory. Its
10,000 and 20,000 checkpoints are sampled as the two candidates. c02 and c03
each reference their own trajectory.

## Model-worker ownership

Candidate execution is split into exactly four model workers:

| Worker model | Trajectories | Evaluation candidates | Maximum GPU-hours |
|---|---:|---:|---:|
| `ctgan_separate_class` | 3 | 4 | 6 |
| `tvae_separate_class` | 3 | 4 | 6 |
| `neural_sequence` | 3 | 4 | 6 |
| `cof_seqgen` | 3 | 4 | 6 |

`--model` is mandatory and accepts only these four IDs. Omitting it or
supplying any other value fails before authorization or model code runs.
Each worker may write only its own model paths below `_workers/`,
`_trajectories/`, and the model's candidate subtree.

The model worker first claims `_workers/<model_id>/ownership.lock` using
exclusive creation. The immutable lock and `worker_manifest.json` contain
the source/config/development/SamplingPlan/authorization provenance and the
exact three-trajectory/four-candidate scope. If the same model is already
owned, whether RUNNING or terminal, another worker fails closed. A different
model has a disjoint lock and remains independently runnable.

A worker emits `WORKER_COMPLETE.json` only after all three trajectories are
COMPLETE and all four `candidate_result.json` and candidate `COMPLETE.json`
files exist with recorded hashes. Otherwise it emits
`WORKER_FAILED.json`. A worker never creates a selection report, changes
another model's artifacts, or cancels another worker.

For CTGAN/TVAE the shared checkpoint is a balanced bundle:

- baseline step 10,000 contains class checkpoints 5,000/5,000;
- baseline step 20,000 contains class checkpoints 10,000/10,000.

Each checkpoint bundle stores the backend, shared trajectory ID, sampled
checkpoint step, SamplingPlan hash, and the train-only z-score object and
state when applicable. Thus mean/std and their reversible schema travel with
the exact sampled checkpoint.

## Immutable trajectory provenance

`manifest.json` is written before training and records:

- source commit and relevant source SHA-256;
- selection config and development manifest SHA-256;
- train and validation file/content hashes;
- fixed train-fit SamplingPlan hash;
- model, trajectory ID, candidate IDs, seed, requested updates, wall cap, and
  device;
- `test_split_read=false` and
  `validation_labels_or_lengths_used_for_fit=false`.

`TRAJECTORY_COMPLETE.json` is written only after the full 20,000-update
trajectory completes. It records requested/actual updates, actual training
time, `wall_cap_reached=false`, the hard-cap completion state, and every
candidate checkpoint path/SHA-256. A candidate result cannot be emitted
without both immutable trajectory artifacts and their verified hashes.

## `candidate_result.json`

Required provenance:

- schema `benchmark-v2.6-candidate-result-v1`;
- status `COMPLETE`;
- model ID, candidate ID, selection seed;
- shared trajectory ID;
- selection config SHA-256;
- canonical candidate-definition SHA-256;
- candidate source commit and relevant source SHA-256;
- development manifest SHA-256;
- train and validation file/content hashes;
- selection SamplingPlan hash;
- requested and actual updates;
- sampled checkpoint step;
- wall-cap and full-trajectory completion status;
- shared trajectory manifest and completion paths/SHA-256;
- validation sample path/SHA-256;
- checkpoint path/SHA-256.

All candidate paths must resolve below the v2.6 candidate root. A path with a
test-split component is rejected.

The result is valid only when:

- requested and actual updates both equal 20,000;
- the sampled checkpoint is the candidate's preregistered 10,000 or 20,000
  step;
- wall cap is not reached;
- the shared trajectory completed all 20,000 updates within its hard cap;
- source/config/development/train/validation/plan hashes match exactly;
- the candidate checkpoint is the declared step from the declared shared
  trajectory;
- the sample satisfies labels, lengths, mask, padding, finite numeric, and
  train-support contracts.

## Bounded training runner

`scripts/run_candidate_training_v2_6.py` is implemented but not authorized
or invoked by this amendment. Actual execution requires a separate immutable
authorization manifest matching source commit, relevant source hash,
selection config hash, and development manifest hash. The authorization must
keep test access, fresh test, and five-seed/full execution false.

Each model worker:

- opens only frozen `train.npz`; it reads validation hashes/entity count as
  metadata but never opens validation arrays during fitting or SamplingPlan
  construction;
- derives the single shared SamplingPlan from train policy;
- runs each trajectory in a runner-owned child process;
- applies the 7,200-second monotonic deadline to training and candidate
  sampling together;
- sends SIGTERM only to its child, waits a bounded grace period, then sends
  SIGKILL only to that same child if necessary;
- records wall cap, invalid candidate, contract, and infrastructure/code
  failures as append-only `FAILED.json`;
- stops its own model fail-closed after a non-COMPLETE trajectory;
- leaves every other model worker untouched;
- emits a model-level terminal manifest and never aggregates selection.

Writes are accepted only below
`artifacts/benchmark_v2_6/selection/candidates/`. The runner rejects test or
fresh-test components, `artifacts/benchmark_v2_5/`, and any v2.6 full-run
path immediately. It never writes a v2.5 runtime artifact.

## Aggregate-only selection report

Direct selection without model-worker terminals is disabled. The only
selection entry point first verifies all four `WORKER_COMPLETE.json` files,
their frozen provenance, and all 16 candidate artifact hashes. Missing,
FAILED, or corrupted worker/candidate state fails before an aggregate output
directory is created.

For the separately authorized primary-only selection amendment, the same
entry point instead verifies the three completed primary terminals and all
12 primary candidate artifacts. Neural is then explicitly recorded as
secondary `NOT_EVALUABLE`; only a preregistered secondary model may be
unavailable. A missing primary worker remains a hard failure. See
`primary_validation_selection_v2_6.md`.

The future command is:

```bash
<COFSEQ_PYTHON> \
  -m scripts.aggregate_candidate_selection_v2_6 \
  --repository-root . \
  --config configs/benchmark_v2/selection_v2_6.yaml \
  --development-manifest configs/benchmark_v2/development_data_v2_6.yaml \
  --candidate-root artifacts/benchmark_v2_6/selection/candidates \
  --output-root artifacts/benchmark_v2_6/selection/aggregate_attempt_001 \
  --authorization <matching-append-only-authorization.json> \
  --source-commit <authorized-source-commit>
```

This command is documented for a later user-owned launch and requires a
matching append-only authorization. It is **not executed** by this
preparation task.

### Append-only continuation exception

The corrective CTGAN/TVAE continuation never replaces attempt_001. Its
authorization binds the original ownership lock, failed worker terminal,
native trajectory manifest/completion, native checkpoints, and preservation
tree by SHA-256.

The approved layout is:

```text
candidates/
  _workers/<ctgan-or-tvae>/attempt_002/
  _evaluations/<ctgan-or-tvae>/<native-trajectory>/seed_2601/attempt_002/
  _trajectories/<ctgan-or-tvae>/<c02-or-c03-trajectory>/
    seed_2601/attempt_002/
  <ctgan-or-tvae>/<candidate>/seed_2601/attempt_001/
```

The candidate attempt is `attempt_001` because the failed original workers
created no CTGAN/TVAE candidate artifact. Exclusive creation makes this
append-only; any unexpected pre-existing file fails closed.

The continuation worker terminal indexes one reused native trajectory, two
newly trained trajectories, and four candidate artifacts. It records the
native training source separately from the corrected evaluation source and
must state `native_trajectory_retrained=false`.

Aggregate readiness consumes an authorization-provided model-to-terminal
map and verifies each terminal SHA-256. This supports CoF 001, CTGAN 002,
TVAE 002, and Neural 001 without weakening provenance checks. The aggregate
itself remains separately unauthorized until all exact terminal hashes can
be frozen.

Output writes use exclusive create. Existing files are never overwritten.
The report contains all five metrics, thresholds and PASS/FAIL checks for
every authorized candidate, the deterministic model-level decisions, and
the full source, config, data, plan, sample, and checkpoint provenance.
Primary-only execution writes 12 candidate results and marks Neural
`NOT_EVALUABLE`.

The selection bundle writes `selection_report.json`,
`selection_manifest.json`, `checksum_manifest.json`, and
`artifact_index.json` before writing `AGGREGATE_COMPLETE.json` last.

The compatibility entry point `scripts/select_validation_candidates_v2_6.py`
delegates to this same aggregate-only gate; it cannot bypass worker-terminal
readiness. Workers cannot invoke aggregate/finalization themselves.

With four GPUs, one model worker per GPU has a hard-cap wall-time ceiling of
approximately 6 hours. With three GPUs, one worker runs in the second wave,
for a hard-cap ceiling of approximately 8 hours, excluding small
orchestration and artifact-verification overhead.

## Failure behavior

- Missing or mismatched candidate artifact: fail closed; no selection.
- Contract or support failure: fail closed; no selection.
- Candidate guard failure: preserve the metrics and exclude it from the
  eligible set.
- No all-guard-PASS candidate for CTGAN, TVAE, or CoF:
  `SELECTION_FAILED_PRIMARY_NO_FULL_RUN`; no primary C2 freeze or run.
- No all-guard-PASS candidate only for neural:
  `PRIMARY_SELECTION_PASS_SECONDARY_NOT_EVALUABLE`; primary selection may be
  frozen, neural is `NOT_EVALUABLE`, and later execution still requires
  separate authorization.
- Mixed candidate source states: fail closed; no freeze.
- Existing output path: fail rather than overwrite.

## Freeze behavior

If and only if all three primary learned models select an eligible candidate,
the harness creates an append-only selection freeze. A selected neural
candidate is included; otherwise its secondary status is recorded as
`NOT_EVALUABLE`. The freeze contains no test path or test hash and grants no
execution authorization.

After the freeze, the following remain mandatory before a fresh test or
five-seed run:

1. verify selected v2.6 config and candidate implementation source hashes;
2. verify train/validation manifest and selection report hashes;
3. obtain separate explicit user authorization;
4. create a new v2.6 fresh-test/data protocol without modifying v2.5;
5. preregister the five-seed evaluation artifact and statistical contract.

## Current execution state

At preparation completion:

- candidate training executions: 0;
- validation selection executions: 0;
- fresh-test definitions/generations: 0;
- fresh-test evaluations: 0;
- five-seed GPU runs: 0;
- full experiments, sweeps, and retries: 0.

The implemented runner and all contract tests use fake/no-op fixtures for
watchdog behavior. Actual model fit/sample, CUDA, DGP, candidate generation,
and validation selection calls remain zero.
