# CoF-HCMTTPP-v2 H1 attempt_002 spawn ownership correction

## Scope and preservation

This is a source-only runner-contract correction. It performed no GPU/CUDA
query or call, external data-body read, model fit, sample, evaluation, gate,
retry, authorization creation, or launch-plan creation. The existing
AMLSim/Sparkov H1 `attempt_001` and `attempt_002` trees, authorizations, and
launch logs remain append-only evidence.

| Evidence | SHA-256 / canonical tree record |
|---|---|
| AMLSim `attempt_001` tree | `07bca9894bf6b619a4b4d6359773b2da2ac40a7112bebe70096712360671b389` (13 files, 106696 bytes) |
| Sparkov `attempt_001` tree | `309fbd56bc6c522b818e842b8c6082df5a0114b7f840537a0361164c744faf28` (13 files, 17706 bytes) |
| AMLSim `attempt_002` tree | `91a4670b6b8bebd47f289f9124d100d4d0f52e704ad02dac73ec23c70a767095` (7 files, 7006 bytes) |
| Sparkov `attempt_002` tree | `b4a900aa1e88fde24299d613fbdf90b67ba648493d904beaf9734ff348fce6a7` (7 files, 7018 bytes) |
| AMLSim `attempt_002/FAILED.json` | `eca6161ad16e2b3de3010d35a318fa4a7e05d0138fe8175c9b293ad704a8f5e9` |
| Sparkov `attempt_002/FAILED.json` | `c98002596dc43e22866555d280df88eefa18d20a0e19f3bd35deb172d8663ff6` |
| AMLSim `attempt_002` authorization / launch log | `fbc1052612989679f33b3331887d30091d6a8d51923f9175fd0328d86575eba3` / `f14230f01adf1e4c1ef09f4a8806024127529b4ef570066dcd7e99527ba2a832` |
| Sparkov `attempt_002` authorization / launch log | `b8d567ce854859d97bcfebb30d2801a194b385f0931e533290fb2824014052e5` / `8d1b81ac312df9cdc72b1bb454634fe08fbd000302d8ab25ad5997c123f7ecaf` |

Both failed terminals report `failure_class=child_exception` and
`H1RunnerContractError: spawn child attempt ownership mismatch`. They failed
before a checkpoint, fit update, validation sample, evaluation, or H1 gate.

## Hypothesis audit

### 1. Parent ownership identity fields disagree — REFUTED

For each dataset, the parent-created ownership record and manifest agree on
dataset, candidate `H1`, seed `4001`, `attempt_002`, canonical authorization
identity, and provenance identity. The authorization has the same dataset,
candidate, seed, attempt, and authorization identity. AMLSim uses
authorization identity
`9df789d4c9909b551f5a09718a73cc8666e69148bd7d2d8593ae2c40e89805c0`
and provenance identity
`e5d790be4da9ae8bc2b7490ec67d4b56e3f3b2d94ab49792807c9d621eccd549`.
Sparkov uses
`e6da161689b9a6e2678a422f36b9efb3af8257b1f0643fa88f512f20ef9e06af`
and
`636dfe975ab94ecc517a27ff7353dd7f236f68e784868c03a281b156fd7178f5`.

### 2. Backend retains the previous ownership convention — SUPPORTED

The parent correctly created and serialized
`workers/<dataset>/H1/ownership_attempt_002.lock`. The spawn payload contained
that exact absolute path. `_authorized_dependency_child` did not forward the
field to `dependencies.run_job`, however. Independently, the backend rebuilt
`workers/<dataset>/H1/ownership.lock`, the `attempt_001` convention, and passed
that stale path to `H1AttemptStore.attach_existing()`. The attach contract
correctly rejected it because the expected attempt_002 lock path and supplied
path differed. This identical source path explains both dataset failures.

### 3. Authorization or manifest canonicalization differs — REFUTED

The stored authorization, ownership record, and manifest canonical identities
match within each dataset. The child failed on path identity before any
authorization or manifest recomputation. There is no evidence of different
parent/child canonicalization.

## Regression and minimal correction

The regression fixture exercises the real boundary on CPU:

1. parent `H1AttemptStore.claim()` creates an attempt_002 ownership record;
2. parent allocates the append-only attempt and serializes plan, job,
   ownership path, and attempt path into a spawn-compatible payload;
3. the real `_authorized_dependency_child` entrypoint crosses a spawned
   process boundary;
4. a no-data/no-model/no-CUDA backend fixture attaches through
   `H1AttemptStore.attach_existing()` and completes the artifact contract.

Before the correction, AMLSim and Sparkov both failed because the backend
callable did not receive `ownership_path`. The minimal correction forwards the
exact parent-created `ownership_path` through `_authorized_dependency_child`
and makes `run_authorized_job` use it instead of reconstructing a path.

The attach contract remains fail-closed. In addition to exact resolved path,
attempt path, lock existence, and manifest existence, it now strictly decodes
the ownership record and verifies schema, dataset, candidate, seed, attempt,
and the presence of 64-character authorization/provenance identities. A
synthetic cross-identity ownership record remains rejected; no fallback,
relaxation, old-path lookup, or error suppression was added.

The H1 model and gap decoder, optimizer, seed, training budget, C0/C1 gate,
frozen bundle, SamplingPlan, threshold, and scientific endpoint are unchanged.
No `attempt_003` authorization or launch plan is part of this correction.
