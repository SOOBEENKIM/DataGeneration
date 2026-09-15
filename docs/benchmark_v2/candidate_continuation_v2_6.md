# v2.6 CTGAN/TVAE candidate continuation amendment

## Status and immutable baseline

This amendment is source-only preparation. It authorizes no execution by
itself. GPU inventory queries, CUDA allocation, candidate training,
candidate sampling, validation selection, fresh-test generation, and
five-seed/full execution were not run while preparing it.

The immutable source baseline is commit
`941c3605121bad7fa64769d390fdfb99d61c8efd`. Existing v2.6 runtime
artifacts and authorizations remain append-only. The selection config remains
byte-identical with SHA-256
`0884a0144f74f6317ae9e636c0cf5657dedac21ff12cdf545a6b6e5e29be4c4d`.

The preserved attempt_001 records are:

| Model | Preservation-tree SHA-256 | Ownership lock | Worker terminal |
| --- | --- | --- | --- |
| CTGAN | `c7265d4d58587b680b566b44cc4545ef8e3c88844c6d8a6410fbf9cf34dbc840` | `18ebe84ef7de326873cb01e7aae51cc0127efaeaecd5af4bdcac0ae1cb9a7d40` | FAILED `d36872a9198e0ad9398276fb2748659deace2dff04068b2c95a9bb479074c149` |
| TVAE | `c0f01590e124cd79c57c2672a1248b2a0ff8a797768f454ece944f2391a25be2` | `994c20e104faebe68ede4c33a7d1963e72b496c87123f0f01eea4f1bf1af7c41` | FAILED `e2f6d87922faffc771865a2ecf22fb8acd0aab4a21cc8a70fd0d2eb50b811afd` |

The preservation digest is SHA-256 over the sorted GNU `sha256sum` records
for the original ownership lock, model worker attempt_001, and native
trajectory attempt_001, with repository-relative paths. Continuation files
are outside this frozen set.

CoF remains complete at worker attempt_001. Its terminal SHA-256 is
`b5aaa3b6f3373e9eca0d32bc743df005539aebbfdea28725f660f4166fa5c16d`.
It must not be rerun.

## Native checkpoint reuse

The CTGAN and TVAE native trajectories completed all 20,000 requested
updates under the original hard cap. Their evaluation failed after training
because the pre-correction restore path mapped a third-party CPU RNG state
onto CUDA. The checkpoints are immutable:

| Model | Step | Checkpoint SHA-256 |
| --- | ---: | --- |
| CTGAN | 10,000 | `0cd3a940b49299bf64f69cc4ce7bc5be3d91b52586b49cf0553708f2dbdd9fea` |
| CTGAN | 20,000 | `7e9de7e609ad48e8de687426240fe4fd0a47b5d2d0abcbc70a4edb58f182d10d` |
| TVAE | 10,000 | `38930c471a4fa17feb31e079486ae5d0b3e1f3db61bcdb8cc57dd8881758b979` |
| TVAE | 20,000 | `6f5263a5a40d9513342b33a823a02769bab65f6e1a5bd38c814b87d771a2947c` |

The native training source is
`68679cce85baf9cd53ef2d74c81ba6c8cd143449`, with relevant-source
SHA-256
`217d016d47ee3a9a7e4be396313d8898788662f699e595f33cff428ad90e90ac`.
The native trajectory manifest/completion SHA-256 values are respectively
`44a7cf63c29481fc7dfa48066a33af2e4641c925f5f8cf2d8226f416f976af46`
and
`8ef8ef249a4067ed3301649ab95f65ebe1c195ad94d26efbee7e0e1d6c655ce9`
for CTGAN, and
`7e7c770bb6513d21b080786290d31bb044d3ae655adaed960fa28c9071998b6a`
and
`cb4041a67bccf9a3a2afd79a20982449df41178b60248508b640643ff42e6fd1`
for TVAE.

Candidate c00 and c01 are evaluation-only continuation operations. They use
the corrected CPU-first deserialize path and explicitly move only the
backend models to the selected device. Their native trajectory fit count is
zero.

Candidate c02 and c03 are the only continuation training trajectories.
Each writes exclusively to its own trajectory `attempt_002`. The native
trajectory is never planned or trained. Candidate artifacts remain at their
first unused append-only candidate attempt, `attempt_001`, which preserves
the frozen validation-selection reader contract and does not overwrite any
pre-existing CTGAN/TVAE candidate artifact.

## Ownership and authorization

`--continuation` is accepted only with `--model ctgan_separate_class` or
`--model tvae_separate_class`. The continuation validator requires:

- the corrected source commit and relevant-source hash;
- unchanged selection config, development manifest, and train-only
  SamplingPlan hash;
- the exact original ownership lock, FAILED terminal, and attempt_001
  preservation-tree hash;
- exact native manifest, completion, and checkpoint paths and SHA-256;
- the original native training source and the corrected evaluation source;
- `native_trajectory_retraining_authorized=false`;
- `test_access_authorized=false`,
  `fresh_test_authorized=false`, and
  `five_seed_full_experiment_authorized=false`.

Only a matching append-only authorization may create the model worker
`attempt_002`. The original ownership lock is read and verified, never
modified. A pre-existing worker/evaluation/training attempt_002 fails
closed.

## Mixed-attempt aggregate contract

Workers never aggregate or select. A later, separate aggregate authorization
must name and hash the exact terminal for every model:

- CoF: attempt_001;
- CTGAN: approved continuation attempt_002;
- TVAE: approved continuation attempt_002;
- Neural: attempt_001.

The aggregate readiness gate verifies each authorized terminal path,
terminal SHA-256, schema, source provenance, config/development/SamplingPlan
hashes, and all indexed candidate artifacts. A continuation terminal must
state `native_trajectory_retrained=false`.

Candidate result provenance separates training source from evaluation
source for reused checkpoints. Mixed, explicitly authorized source states
are retained in the selection report rather than collapsed into a false
single-source claim. Validation metrics, five-guard eligibility, model
roles, thresholds, and selection rules are unchanged.

No validation selection or aggregate is authorized by the continuation
authorization. After all four exact terminals exist, an aggregate-only
authorization containing their final SHA-256 values is still required.
