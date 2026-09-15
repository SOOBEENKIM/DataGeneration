# v2.6 primary validation-only selection amendment

## Scope

This amendment permits validation-only selection from the completed primary
candidate artifacts. It does not authorize model fitting, sampling, GPU or
CUDA access, data regeneration, test or fresh-test access, or a five-seed
full experiment.

The evaluated models are exactly:

- `ctgan_separate_class`;
- `tvae_separate_class`;
- `cof_seqgen`.

`neural_sequence` was outside the completed primary candidate execution
scope. It is recorded as secondary `NOT_EVALUABLE` and does not block
primary selection. No Neural candidate path is read.

## Read-only input freeze

The approved worker terminals are:

| Model | Attempt | `WORKER_COMPLETE.json` SHA-256 |
| --- | --- | --- |
| CoF-SeqGen | `attempt_001` | `b5aaa3b6f3373e9eca0d32bc743df005539aebbfdea28725f660f4166fa5c16d` |
| CTGAN | `attempt_002` | `af3c2a54e40f353b23b65d95b27383e41aa05ca4c95b40497c3bb79cfd6aa127` |
| TVAE | `attempt_002` | `200892218a2a242e136dcaf7e441758a3e7b9281e2ea3cc7ab1c3cbf5ff759cc` |

All 12 indexed candidate results, candidate completion markers, validation
samples, checkpoints, trajectory manifests, and trajectory completion
markers must match the hashes referenced by these terminals and candidate
manifests.

Before this amendment, the complete v2.6 candidate runtime tree contained
122 files and 381,722,799 bytes. Its sorted repository-relative
`sha256sum`-record digest was
`b202a75b3f3e1e46b19bf6cfcc06735c725dfed71754c1ed01d073f9db8e06b8`.
Selection writes only outside `candidates/`; this digest must remain
unchanged.

The frozen selection config SHA-256 remains
`0884a0144f74f6317ae9e636c0cf5657dedac21ff12cdf545a6b6e5e29be4c4d`.
The development manifest and train-only SamplingPlan remain frozen.

## Selection rule

For every primary model, the harness evaluates the four existing candidate
validation samples against the five preregistered guards:

1. amount KS;
2. gap KS;
3. amount absolute standardized label effect;
4. gap absolute standardized label effect;
5. receiver maximum absolute signed frequency.

Only a candidate passing all five guards is eligible. Among eligible
candidates, selection minimizes:

1. `max(amount_ks, gap_ks)`;
2. `amount_ks + gap_ks`;
3. lexicographic candidate ID.

If any primary model has no eligible candidate,
`primary_c2_selection_ready=false` and no later full run is authorized.
Thresholds and rules are not changed after observing results.

## Authorization and modes

A selection-only authorization must bind the corrective source commit,
relevant-source hash, config/development/SamplingPlan hashes, the three
terminal paths and SHA-256 values, and the exact model scope. It must set:

- `selection_model_ids` to the three primary models;
- `unavailable_model_ids` to `["neural_sequence"]`;
- validation selection and aggregate-only authorization to true;
- test, fresh-test, five-seed/full, training, sampling, and data generation
  authorization to false.

The aggregate CLI has three modes:

- `plan`: validate source, authorization, train-only provenance, terminal
  scope, and indexed hashes without candidate evaluation or writes;
- `dry-run`: calculate the primary validation selection without writes;
- `execute`: create one append-only selection bundle.

The execute bundle contains:

- `aggregate_readiness.json`;
- `selection_report.json`;
- `selection_manifest.json`;
- `checksum_manifest.json`;
- `artifact_index.json`;
- `AGGREGATE_COMPLETE.json`, written last.

Every output is exclusive-create. A pre-existing aggregate attempt fails
closed.
