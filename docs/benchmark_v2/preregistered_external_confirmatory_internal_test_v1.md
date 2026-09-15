# Preregistered external confirmatory internal-test protocol v1

Status: source-only design; no execution authorization exists.

Implementation base: `c33eeb11cde9efee8267f04ea5604eaffa9628a8`.

## Objective and immutable scope

The confirmatory evaluation covers exactly two frozen internal-test datasets
and the four unchanged validation models:

- AMLSim `internal_test.npz`
- Sparkov fraudTrain-derived `internal_test.npz`
- empirical IID, CTGAN separate-class, TVAE separate-class, and frozen non-v3
  CoF-SeqGen

Sparkov public `fraudTest` remains excluded before path resolution. Raw CSV,
TSTR, privacy, model training, checkpoint updates, refitting, candidate
selection, tuning, sweeps, and retries are outside the protocol.

## Frozen state carried from validation

For each dataset/model pair, the future authorization must hash-bind the exact
COMPLETE validation terminal, artifact index, generator checkpoint or
empirical fit-state, train-only transform state, train-only threshold state,
model source/config, evaluator source, and dataset seed. AMLSim keeps seed
31001 and Sparkov keeps seed 32001.

The learned-model state is the validation attempt's indexed
`checkpoints/final.pt`. The empirical IID implementation did not serialize a
separate object: its complete fit state is exactly the immutable train bundle's
class-conditional row pool. The execution runner restores that pool directly
from the hash-bound `train.npz` and never calls `fit`; this preserves the
empirical bootstrap definition without refitting or estimating a new state.

No transform, receiver vocabulary, amount normalization, gap boundaries,
short-gap threshold, fidelity threshold, coherence threshold, model parameter,
sampling setting, or evaluation formula may be re-estimated from validation or
test. The stored validation selections do not choose a different test model:
all four models are evaluated exactly once on each dataset.

The internal-test conditioning plan is the order-preserving `(Y, valid length)`
sequence extracted deterministically from the frozen internal-test bundle. It
has no fitted parameter. It must be hashed once before the first dataset job
and reused identically by all four models for that dataset.

## Exact eight-job plan

| Dataset | Model | Validation state reused | Test attempt |
|---|---|---|---|
| AMLSim | empirical IID | `attempt_002` | `attempt_001` |
| AMLSim | CTGAN | `attempt_001` | `attempt_001` |
| AMLSim | TVAE | `attempt_002` | `attempt_001` |
| AMLSim | frozen non-v3 CoF | `attempt_001` | `attempt_001` |
| Sparkov | empirical IID | `attempt_002` | `attempt_001` |
| Sparkov | CTGAN | `attempt_001` | `attempt_001` |
| Sparkov | TVAE | `attempt_001` | `attempt_001` |
| Sparkov | frozen non-v3 CoF | `attempt_001` | `attempt_001` |

Each job restores the validation-frozen state, generates one synthetic set
against the shared internal-test conditioning plan, and evaluates it once.
There is no test-time selection. A failed or invalid job remains FAILED or
INVALID and cannot be silently omitted or retried. Final aggregation requires
all eight explicit terminal artifacts.

The single sampling call uses the frozen validation sampling seed (`dataset
seed + 1`). The conditioning plan changes only to the exact internal-test
labels and valid lengths; its order-preserving derivation has no fitted
parameter and is shared by all four models in a dataset.
The later authorization must record the SHA-256 of each derived plan; every
job recomputes the deterministic plan and refuses to sample if that hash
differs.

## Evaluation contract

The fidelity metrics remain class-conditional amount KS, gap total variation,
and receiver total variation. Coherence remains the class-conditional
short-gap × receiver-repeat error. The same train-only thresholds and the same
ratio calculation are reused. The frozen combined score remains:

`0.5 × fidelity_max_ratio + 0.5 × coherence_max_ratio`

Lower is better. Test results report all four models and all components. They
cannot be used to change a model, threshold, seed, formula, or retry decision.
The cross-dataset validation macro table is descriptive and is not used as a
test selection rule.

## Required future authorization

A later explicit authorization must be limited to the eight jobs above and
must bind:

1. source commit, relevant model/adapter/runner/evaluator hashes, and this
   protocol config hash;
2. both frozen bundle manifests and exact `internal_test.npz` file hashes;
3. both validation aggregate input hashes;
4. train-only transform and threshold hashes;
5. each validation checkpoint/fit-state, terminal, and artifact-index hash;
6. AMLSim seed 31001, Sparkov seed 32001, and each derived test conditioning
   plan hash;
7. append-only target paths under
   `artifacts/external_confirmatory_internal_test_v1/`.

It may allow only checkpoint/fit-state restore, one internal-test NPZ load per
dataset/model job, deterministic conditioning-plan extraction, one generation
and frozen evaluation per dataset/model, and append-only artifact writes.

It must explicitly forbid authorizationless execution, training/refitting,
validation/test calibration, model or seed changes, retry, sweep, test-time
selection, raw CSV, Sparkov `fraudTest`, TSTR, privacy, and mutation of any
existing artifact. No authorization is created by this source-only amendment.

## Source-only execution implementation

`scripts.run_external_confirmatory_internal_test_v1` provides `plan`,
`dry-run`, and authorization-gated `execute` modes. Plan and dry-run inspect
manifests and indexed hashes only: they do not load NPZ bodies, import model
code, query CUDA, or create artifacts. The execution-only module is imported
only after exact authorization validation and owns one append-only
`attempt_001` path.

Every authorized job performs exactly one state restore, one sample, and one
evaluation. CTGAN and TVAE deserialize CPU-first and move only their sampling
backends to the selected device. CTGAN's checkpoint intentionally omitted its
reconstructible `DataSampler`; restore rebuilds only that sampler with the
already-fitted transformer and exact hash-bound train rows under the fixed
single-worker memory policy. It does not call transformer or model fit. CoF
reconstructs the frozen non-v3
architecture and loads its indexed state dict. No execution path calls model
fit, writes a checkpoint, recomputes thresholds, performs test-time selection,
or exposes a retry attempt. Sparkov `fraudTest` strings are rejected before
`Path` construction or resolution.

## Current execution state

Authorization artifacts, GPU/CUDA calls, model calls, internal-test reads,
Sparkov `fraudTest` accesses, TSTR, and privacy calls are all zero.
