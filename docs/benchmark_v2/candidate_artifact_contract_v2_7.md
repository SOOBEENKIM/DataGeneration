# v2.7 candidate artifact and ownership contract

## Preparation boundary

This commit is source-only. It creates no v2.7 runtime root and no execution
authorization. `plan` and `dry-run` return JSON on standard output and do not
write repository artifacts.

If execution is separately authorized later, all writes are restricted to:

```text
artifacts/benchmark_v2_7/candidate_selection/
```

v2.5/v2.6 runtime, frozen data, config, authorization, checkpoints,
candidate outputs, aggregate, and forensic artifacts remain read-only.
Deletion, movement, replacement, relinking, and overwrite are forbidden.

## Future append-only layout

```text
candidate_selection/
  authorization_history/
  workers/
    <model>/
      ownership.lock
      attempt_NNN/
        manifest.json
        RUNNING.json
        WORKER_COMPLETE.json | WORKER_FAILED.json
  candidates/
    <model>/<candidate>/seed_2701/attempt_NNN/
      manifest.json
      fit_state.json
      validation_sample.npz
      diagnostics.json
      five_guards.json
      COMPLETE.json | INVALID.json | FAILED.json
```

Files use exclusive create. A repeated candidate allocates the next attempt;
an existing byte is never overwritten. A model worker must atomically claim
its model-level ownership lock before it may allocate an attempt. A second
worker for the same model fails closed. Different model workers own disjoint
subtrees.

No worker may create an aggregate or selection report. Those operations
require a later aggregate-only contract and authorization.

## Immutable operation manifest

Before a future operation, `manifest.json` must record:

- source commit and relevant-source hash;
- v2.7 config path and SHA-256;
- model, candidate, selection seed, execution kind, and sole factor;
- canonical baseline/effective dimensions and exact changed-key list;
- frozen checkpoint/result/sample paths and SHA-256 values;
- train and validation file/content hashes;
- train-only SamplingPlan hash;
- unchanged threshold and selection-rule hashes;
- authorization path and SHA-256;
- expected fit-state destination and terminal files;
- requested hard cap;
- all forbidden access/execution counters initialized to zero.

Controls require an empty changed-key list. Each non-control requires exactly
the one preregistered changed key. Any extra changed key fails before artifact
allocation.

## Train-only fitted state

Each non-control candidate must persist a canonical fitted-state payload in
its checkpoint or candidate artifact. The payload contains:

- schema version, model, candidate, and sole factor;
- factor parameters;
- `fit_split=train`;
- `fit_input=valid_train_rows_and_train_only_sampling_plan`;
- `validation_rows_used=0` and `test_rows_used=0`;
- source commit;
- config, train file, train content, and SamplingPlan hashes;
- destination and canonical state SHA-256.

The state is validated before any future candidate evaluation. Tampering or
provenance drift fails closed. Validation and test rows cannot refit or alter
state.

## Frozen controls

The three controls are references to existing verified v2.6 bytes:

| Model | Control result SHA-256 | Sample SHA-256 | Checkpoint SHA-256 |
|---|---|---|---|
| CTGAN | `bccb0143ca6bd3411284443c3b5cb85d8bdcb9cf81d844ec479305fcf7d1dc51` | `59ea09140e8b85e0e4b4eed64f7aba31f52c58ea6b0890e3a680d47c310a161b` | `7e9de7e609ad48e8de687426240fe4fd0a47b5d2d0abcbc70a4edb58f182d10d` |
| TVAE | `3146d3470a0398acae39e3f0bbc559b52bb2f36d86876ed5bbe11f7d0f907a56` | `9a16aca4bf20c738a7225c77dc2079458001d5cbcd5207f8e59c4730096f32ad` | `6f5263a5a40d9513342b33a823a02769bab65f6e1a5bd38c814b87d771a2947c` |
| CoF | `d68a78c228520a3edaffae3dfe076c7d59e446575c576905d21f7f53bc639fac` | `c91b386473bcbbabd0f7f2d7f9b3659b760d282f70061a449254f1a0d03f34a3` | `7ce44721cadffef21ade3bf1884421c304f73ea9eff1e9bfb3ca2d9e63cc0c29` |

A control performs zero training, fitting, and sampling. A future reference
manifest may point at these hashes but cannot rewrite or copy over the
source bytes.

## Validation and test boundary

Candidate state fitting and SamplingPlan fitting may read train only. A
future five-guard evaluation may additionally read the frozen validation
split. Any test, fresh-test, audit-latent, arbitrary data path, or v2.5/v2.6
write attempt fails closed.

The thresholds and SamplingPlan are byte-pinned in the config. Diagnostics
cannot override a failed guard. No invalid candidate can be silently omitted
or replaced.

## Authorization boundary

This source state contains no execution authorization. The preparation CLI
does not accept an authorization, device, or execute option. A future
authorization must bind the then-current source/config hash, every frozen
input hash, the exact nine-candidate mapping, six evaluation-only operations,
zero training trajectories, model ownership, and all prohibited downstream
actions.
