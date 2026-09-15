# External validation v1 artifact contract

Status: source-only correction. Existing runtime is preserved read-only; this
commit authorizes no continuation and creates no runtime artifact.

## Ownership and paths

Every dataset/model pair owns an independent immutable path:

```text
artifacts/external_validation_v1/
  <amlsim|sparkov>/
    <empirical_iid|ctgan_separate_class|tvae_separate_class|cof_seqgen_frozen_non_v3>/
      attempt_001/
```

The historical performance-continuation authorization is one append-only batch document
containing exactly eight single-job grants: two datasets by four fixed models.
Each IID child covers `attempt_002`; each learned-model child covers
`attempt_001`. The CLI selects exactly one child and binds its parent
authorization hash into the attempt manifest.
The batch binds current source/runner/execution hashes, config hash, v2.5
provenance, frozen CoF fingerprints, both bundle trees, and train/validation
hashes. It permits only train bootstrap, train model fit, and validation
generation/evaluation. The attempt is claimed with exclusive creation; a
pre-existing path, ownership marker, artifact, provenance mismatch, duplicate
job, or changed scope is an error. AMLSim and Sparkov never share a writable
root.

At authorization creation, all eight target attempts must be absent. During a
later wave, completed/terminal attempts from earlier authorized jobs are valid
batch history and do not invalidate the remaining grants; only the newly
selected target must still be absent. That authorization bound a scheduling-only
three-wave plan. It is now superseded because AMLSim CTGAN and TVAE transformed
the high-cardinality input concurrently and exceeded host memory; it must not
be reused.

The corrective source fixes external CTGAN/TVAE `DataTransformer` execution at
one synchronous in-process worker and forbids `n_jobs=-1`. A global
`external_tabular_transform` exclusion group permits at most one CTGAN/TVAE
heavy transform at a time. The source-level five-wave schedule is CPU IID;
AMLSim CTGAN+CoF; AMLSim TVAE alone; Sparkov CTGAN+CoF; Sparkov TVAE alone,
with a terminal barrier after each wave. This is a contract template only: the
corrective commit creates no replacement authorization and authorizes no retry.

Completed AMLSim IID, CTGAN, and CoF artifacts may be referenced by a later
continuation only through the exact hash-bound resource-policy reuse validator.
It verifies original source/config, bundle/train/validation, seed, metric,
SamplingPlan, threshold, manifest, evaluation, and COMPLETE hashes. The next
possible target is TVAE `attempt_002`, subject to separate approval.

The dedicated TVAE continuation authorization is a one-job document under
`amlsim_tvae_attempt_002_authorization_history`. It is distinct from the
superseded eight-job batch authorization. Its validator binds the three reused
AMLSim COMPLETE results, failed TVAE `attempt_001`, current source/config,
frozen data, metric source, fixed memory policy, and the absent append-only
`attempt_002` path. No other dataset/model child grant exists in this document.

## Future authorized attempt

The append-only attempt contains, in order:

- `OWNERSHIP.json`, `RUNNING.json`, and immutable `manifest.json`;
- `train_bootstrap_thresholds.json`;
- `validation_sampling_plan.npz`;
- `progress.jsonl` and learned-model `checkpoints/step_*.pt` plus `final.pt`;
- `sample.npz`, `metrics.json`, `evaluation.json`, and `runtime.json`;
- `artifact_index.json`; and
- exactly one terminal `COMPLETE.json`, `INVALID.json`, or `FAILED.json`.

Files are exclusive-created or atomically linked from a same-filesystem
temporary file. Existing attempts and terminal artifacts are not overwritten.
An exception records `FAILED.json`; a structural validity failure records an
invalid terminal and is never silently omitted from selection. Test execution
is always recorded as unauthorized.

The immutable manifest contains only train and validation data hashes. It also
records source/config/authorization/bundle provenance, dataset, model, seed,
split roles, frozen CoF source and config, and append-only status. Dynamic
receiver cardinality is recorded as an input-vocabulary binding, never an
architecture override.

## Plan and dry-run

`scripts.run_external_validation_v1` provides `plan`, `dry-run`, and a gated
future `execute` mode. Plan resolves the eight paths without reading frozen
arrays. Dry-run verifies COMPLETE/index/checksum chains and pinned tree hashes
from manifests, split/leakage manifests, current and frozen adapter
fingerprints, dynamic PAD/UNK/cardinality compatibility, and frozen budgets.
It checks frozen file presence but performs zero NPZ body reads; full byte
rehash is a separate read-only provenance audit. Both modes
reject authorization and execution-only arguments and report zero NPZ loads,
raw CSV reads, GPU/CUDA calls, model imports/fits/samples, validation metrics,
test access, TSTR, privacy, and full runs.

Execute requires the batch authorization file, explicit dataset/model/device,
an absent target path, and exact authorization provenance. A runner-owned child
enforces the authorized whole-job wall cap (7,200 seconds for empirical i.i.d.;
10,800 seconds for each learned job) in addition to the unchanged 7,200-second
learned-model training cap. Deadline termination records append-only
`FAILED.json` with `failure_class=wall_cap`; it never terminates an external
process.
