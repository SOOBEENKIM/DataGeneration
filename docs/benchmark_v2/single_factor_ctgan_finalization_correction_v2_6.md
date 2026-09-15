# v2.6 single-factor CTGAN worker finalization correction

## Scope and preservation

This correction starts from source commit
`ad19d7334a6e134645d954082ab42116dd9ee8d9`. It changes only worker
orchestration, terminal-artifact verification, and finalization-only recovery
support. It does not change or rerun a candidate, checkpoint, sample,
diagnostic, validation guard, config, SamplingPlan, DGP, selection, fresh
test, or full experiment.

The original CTGAN, TVAE, and CoF runtime trees remain append-only and
read-only. No GPU query, CUDA call, fit, sample, selection, or data generation
was used in this correction.

## Read-only CTGAN inventory

At the start of diagnosis, the CTGAN model tree contained 27 files and
56,660,198 bytes. Two deterministic record digests were captured:

- repository-relative record digest:
  `fa4b8135f6983055c9ddd6d9f1a8909ed5cf25e4846ab39624a17f05c8b4d286`;
- single-factor-runtime-relative record digest:
  `6522125871e9be16c11ea02ab2863f3a24cd7c070e05f5fd133c2952413acf95`.

### Candidate terminal artifacts

| Candidate | Artifact | SHA-256 |
|---|---|---|
| `ctgan_sf_c00_frozen_control` | `COMPLETE.json` | `f15decfd40d15a1ce3620cf3baa885cc6cd82c0a02c559f346a8028574c48a98` |
| `ctgan_sf_c00_frozen_control` | `reference_manifest.json` | `4318f95baf75176b10b350c3446fbc4123b3358faafaa0beeedb1db0977b73e8` |
| `ctgan_sf_c01_shared_transformer` | `COMPLETE.json` | `d80dba67a55d66df569f193cdc33d900883958b198671a5a6525918a9313a7c6` |
| `ctgan_sf_c01_shared_transformer` | `candidate_result.json` | `ef59000a571fd0cd3d8fa42936e9b8e788e2387c6f127f838ead71498e352bb5` |
| `ctgan_sf_c01_shared_transformer` | `diagnostics.json` | `f9a723e5b585edb8f5e60eec9cc107cd47299324c0cfef48154aef63132052f8` |
| `ctgan_sf_c01_shared_transformer` | `validation_sample.npz` | `8bf02733ecb20629e9afc798342ceb9ec3d6ddf8143035b3b144f569c09b5435` |
| `ctgan_sf_c02_temperature_only` | `COMPLETE.json` | `e80ea5e1811e3a23f7b224c3308189a6d4b60aeadbc58bca4382315181c7b10b` |
| `ctgan_sf_c02_temperature_only` | `candidate_result.json` | `e6040d34f8b3128ae4a3f1ee67d65e2da26449d3394ae5c5c2c1310d93ee8d10` |
| `ctgan_sf_c02_temperature_only` | `diagnostics.json` | `7aa222f6553e1c1b2ece0f5327b5b0c01e8c3f0c87ae144fcecec96b08f4f6c3` |
| `ctgan_sf_c02_temperature_only` | `validation_sample.npz` | `aeca771e032fdcb56cc96acfc7dcfb7cc7c025c1f798753b5c7d8045fc582db6` |

The frozen control intentionally has a reference manifest instead of a new
candidate result or diagnostic; it performs zero training and zero sampling.

The independent verifier checked all three candidate paths, terminal hashes,
candidate-result provenance, source/config/authorization hashes, diagnostic
hashes, sample hashes, `test_split_read=false`, and
`validation_selection_executed=false`. The result was PASS.

## Reproduced defect and cause

The worker manifest was created at 12:27:02 KST. The final c02 candidate
terminal was created at 12:39:21, but the worker terminal appeared only at
12:44:20. The worker terminal reports 321.90 seconds for c02 even though the
candidate files had already been completed about five minutes earlier.

The deterministic regression fixture reproduces this with a child whose
target returns while a non-daemon cleanup resource remains alive. The old
bounded runner called `process.join(max_wall_seconds)` before reading the
child result queue. The target-complete message therefore sat unread until
process-level joblib/resource-tracker cleanup ended. Worker finalization was
correct but observably delayed and could become markerless if cleanup exceeded
the operation deadline.

Confirmed root cause: parent orchestration waited for full OS-process cleanup
instead of acting on the already-emitted target-complete event.

## Corrective contract

The bounded runner now has an opt-in target-complete callback. Default callers
retain their prior behavior. The single-factor worker uses the callback only
for its final planned operation:

1. consume the child target-complete event;
2. verify all three candidate artifacts and their provenance;
3. exclusive-create `WORKER_COMPLETE.json`;
4. allow a short cleanup grace;
5. if cleanup still lingers, terminate only the runner-owned child and record
   that cleanup was forced.

Candidate success can no longer be overturned by a resource-tracker warning
after the verified target has completed. The callback cannot finalize if any
candidate path, hash, source, config, authorization, sample, diagnostic, or
terminal differs.

Every exception after a worker attempt is claimed now produces an append-only
`WORKER_FAILED.json` unless a valid complete terminal already exists.
Interrupts remain classified as interruptions; abnormal child exit and
missing child metadata retain their explicit failure class and exception.

The same artifact-verification and terminal schema is exercised for CTGAN,
TVAE, and CoF, preserving the normal TVAE/CoF worker contract.

## Finalization-only recovery

A recovery API is implemented for a genuinely markerless worker. It requires
a separate authorization that binds:

- corrected source commit and relevant source hash;
- original candidate source commit/hash;
- unchanged single-factor config hash;
- original execution authorization hash;
- prior worker path and whole-model tree file/byte/digest inventory;
- exact three candidate result records;
- a terminal-only scope with GPU query, training, sampling, selection, test,
  fresh-test, and full-run permissions all false.

Recovery re-verifies the original candidate artifacts, creates a new
append-only worker attempt, writes `recovery_manifest.json`, and writes only a
worker terminal. It performs zero training and zero sampling.

No recovery was executed for the preserved real CTGAN run. During read-only
diagnosis, its original `attempt_001/WORKER_COMPLETE.json` was found with
SHA-256
`1b3973cd551a83dbad8b3db73687d0b4fe84e235946450a414cdf480db4d60cd`.
Its timestamp proves that it was delayed, not permanently absent. A recovery
attempt against an already terminal worker is rejected, so creating
`attempt_002` would be redundant and is forbidden.

If a future worker has three verified candidate terminals but no worker
terminal, a finalization-only recovery authorization is required before the
terminal-only append-only recovery is run.

## Verification

- focused runner/checkpoint/adapter tests: 64 passed;
- repository-wide tests: 286 passed, 23 warnings;
- `compileall`: PASS;
- `git diff --check`: PASS.

The post-test CTGAN inventories were unchanged: 27 files, 56,660,198 bytes,
repository-relative digest
`fa4b8135f6983055c9ddd6d9f1a8909ed5cf25e4846ab39624a17f05c8b4d286`,
and runtime-relative digest
`6522125871e9be16c11ea02ab2863f3a24cd7c070e05f5fd133c2952413acf95`.
