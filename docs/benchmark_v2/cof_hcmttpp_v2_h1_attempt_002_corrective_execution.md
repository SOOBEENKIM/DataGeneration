# CoF-HCMTTPP-v2 H1 corrective attempt_002 contract

## Scope

This amendment permits preparation of exactly one append-only numerical
corrective execution per dataset:

- AMLSim / H1 / seed 4001 / `attempt_002`;
- Sparkov / H1 / seed 4001 / `attempt_002`.

Each cell retains 20,000 requested updates and a 7,200-second whole-cell wall
cap. It is not an architecture retry, tuning run, generic retry, or attempt
sweep. Its sole purpose is to execute the frozen H1 endpoint after the RQS
inverse numerical correction in source commit
`c809c069b5562e764bfc6e43f4f69e28b34a3404`.

## Immutable failed evidence

Both `attempt_001` trees remain append-only FAILED evidence and are required by
every dataset-scoped authorization, even though a worker reads only its own
dataset body after authorization.

| Dataset | Preserved tree SHA-256 | `FAILED.json` SHA-256 |
|---|---|---|
| AMLSim | `07bca9894bf6b619a4b4d6359773b2da2ac40a7112bebe70096712360671b389` | `5a65749584bb9f4a584bd75ed74eb7d10e0547c66c7ffd3c3f9da00431712042` |
| Sparkov | `309fbd56bc6c522b818e842b8c6082df5a0114b7f840537a0361164c744faf28` | `94af493b9dd3f3ddb2847f3cc7d6375ef744d713caf0ad83276d386ba53ed8a3` |

The correction report is
`docs/benchmark_v2/cof_hcmttpp_v2_h1_rqs_inverse_correction.md`, SHA-256
`e78df91540d25d53cf8153125caea676c2bdaf70cd93b4b6328489a947b040bf`.
The authorization validator rehashes that report and both failed trees before
any data-body, model, or CUDA dependency can be reached.

## Fail-closed execution identity

The CLI does not expose an attempt-number argument. The frozen runner config
contains exactly `attempt_002`; config validation rejects any other value.
Each dataset has an independent authorization and an independent
`ownership_attempt_002.lock`. The new attempt path is
`<dataset>/H1/seed_4001/attempt_002`, while the original `ownership.lock`,
worker terminal, and `attempt_001` tree remain untouched.

Every authorization binds the corrected source HEAD, execution-relevant source
hash, runner/source/model config hashes, correction evidence, both failed tree
hashes, one dataset's frozen bundle and train/validation artifacts, fixed
SamplingPlan, thresholds, C0/C1 reference evidence, H1 factor definition, seed,
optimizer, requested updates, batch size, and wall cap. Cross-dataset use is
rejected before dependency construction.

Generic retry, arbitrary attempt numbers, attempt sweeps, C1-v1 retry, C2-C4,
H2, tuning, threshold changes, internal test, and Sparkov fraudTest remain
forbidden. The fixed C0/C1 references, H1 gate, optimizer, seed, budget,
frozen bundle, and SamplingPlan are unchanged.

## Preparation boundary

Source-only plan and dry-run may inspect file existence and immutable hashes;
they do not open NPZ bodies, import or build the H1 model, query a GPU, call
CUDA, fit, sample, evaluate, or create runtime attempts. Actual execution is
possible only with a separately generated exact dataset-scoped authorization.
