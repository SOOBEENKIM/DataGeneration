# v2.8 Evaluation-Only Runner

## Status and boundary

This source-only runner implements the future validation-only execution
contract preregistered in
`selection_v2_8_source_amendment.yaml`. This implementation commit does not
authorize execution and does not create runtime artifacts.

The executable family is exactly:

- `ctgan_separate_class` /
  `ctgan_v28_c01_joint_gap_receiver_decoder`
- `cof_seqgen` / `cof_v28_c01_gap_distribution_sampler`

The frozen CTGAN and CoF parents and the selected TVAE reference are verified
by their existing artifact hashes. They are references only. TVAE is not an
executable v2.8 worker.

Both executable candidates restore the preregistered existing 20,000-update
checkpoint. CTGAN retains CPU-first deserialization followed by explicit
selected-device movement; CoF retains selected-device restore. Optimizer
updates, generator fitting, checkpoint writes, training trajectories, and
result-contingent candidate changes are zero.

## Data and intervention boundary

The train split and fixed train-only SamplingPlan are used to generate the
calibration sample and fit the sole candidate factor. Validation is opened
only after fit-state construction and synthetic generation, and is used only
for the unchanged five-guard evaluation. Test and fresh-test paths are
fail-closed.

The CTGAN candidate changes only the class-conditional joint
`(gap_bin, receiver)` decoded sampling distribution. Its frozen amount
quantile inverse remains the parent intervention. The CoF candidate changes
only the class-conditional gap distribution. Its frozen empirical amount
residual intervention remains the parent intervention. Each fit-state binds
the source, candidate config, train file/content, SamplingPlan, checkpoint,
and frozen-parent provenance hashes.

The five thresholds and SamplingPlan hash are copied exactly from the frozen
v2.8 amendment. Candidate selection is not part of a model worker.

## Authorization and artifacts

Execute mode requires a separate append-only authorization whose exact
contents are validated against the source HEAD, relevant-source hash, runner
config, candidate config, checkpoint hashes, frozen v2.5/v2.6/v2.7
provenance, and all three frozen parent/reference artifacts.
The user supplies an explicit logical CUDA device to each worker; the runner
does not perform GPU inventory selection or query.

Each model owns only its subtree under:

```text
artifacts/benchmark_v2_8/candidate_selection/
  workers/<model>/
  evaluations/<model>/<candidate>/seed_2801/attempt_001/
```

Ownership and artifact writes use exclusive creation. A second worker for the
same model or any overwrite attempt fails closed. Candidate attempts record
the immutable manifest, train-only fit-state, validation sample, evaluation,
candidate result, and terminal marker. The model worker records exactly one
`WORKER_COMPLETE.json` or `WORKER_FAILED.json`.

The per-candidate wall cap is 1,800 seconds. The parent owns and bounds the
child process; the runner never terminates an external process.

## Source-only verification

Only the following commands are permitted before a later execution
authorization:

```bash
<COFSEQ_PYTHON> \
  -m scripts.run_evaluation_only_v2_8 --mode plan

<COFSEQ_PYTHON> \
  -m scripts.run_evaluation_only_v2_8 --mode dry-run
```

Plan and dry-run do not create authorization or runtime artifacts and do not
query GPU/CUDA, restore a model, sample, evaluate, select, or access test.
