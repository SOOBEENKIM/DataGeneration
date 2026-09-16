# CS-SAF recoverable research state

Updated 2026-09-16. Branch: `research/cs-saf`.

New work: [three-objective loss diagnostic](loss_control_v1_preregistration.md)
registered in `d4069c9` before implementation/execution. Its implementation now
passes 82 relevant tests, including historical U/B gradient equivalence and
the global A auxiliary formula. Same-source CPU gates and six pi=.05 GPU fits
with separate train/validation null diagnostics are pending. Previous v2 FAIL
is unchanged; this is a separate exploratory study.

**Current state: v2 implemented, 71 tests and CPU gate PASS; v2 pilot FAIL at
pi=0.05 in the kappa=0 rare context. V1's pi=0.25 FAIL is also preserved.**

V2 implementation preserves the registered 133,549 parameters and fresh-v1
common initialization. Direct bank-gradient isolation, known-label permutation,
strict-past behavior, likelihood/audit agreement, support, checkpoint roundtrip
and fixed auxiliary denominators passed. The revision-aware runner pins the
original v2 YAML hash and inherited v1 budget, requires the same-source CPU gate,
and reuses the immutable data/oracle index. V2 outputs use a separate namespace.
The v2 YAML retains its historical preregistration-state metadata; this file
records current execution state.

Latest v2 execution (2026-09-16):

- Source: `c77d2f9558ca019d14ccf4b833cbc6628728ff44` (committed before CPU/GPU).
- Immutable registration: `d8302e3`; model seed 20260930; 133,549 parameters.
- CPU: two identical histories/best states, train objective drop **48.33%**;
  support, vocabulary, finite-value and zero-gap checks PASS.
- GPU: **4 fits**, CS2-U1/B1 x kappa 0/1, at pi=.05 only.
- Primary CS2-B1 fails kappa=0/label=1: copy **0.081949**, repeat **0.080610**,
  both above .05. The same v1 cell was below .05 (copy 0.045730).
- At kappa=1/label=0, B1 copy response falls from v1 0.046383 to v2 **0.033930**.
  Active kappa=1/label=1 copy/repeat responses remain **0.343783/0.338277**.
- CS2-U1 also fails kappa=0/label=1 (copy **0.055189**, repeat **0.054313**).
  Thus the new failure is not attributable only to the balanced auxiliary.
- All four jobs have exact zero-gap invariance and valid generation. Total
  8,192 entities / 189,324 nonfirst gaps; support violations and reserved marks 0.
- Pi=.10/.25/.50 training stopped by the original rule. No fallback, five-seed,
  real-data or held-out run. Parent exit 2 is scientific FAIL; no worker error.
- Saved checkpoint/sample identities and 21 artifact checksums rechecked.
- The proposed three-objective loss control is **not yet preregistered or run**.

Prior v1 checkpoint forensics (2026-09-16):

- Forensic source: `d96e1b4966f9fec018aee472a26be579ab9a4d1c`.
- All 12 saved best checkpoints analyzed on full train histories. New fits and
  optimizer steps: zero. No new validation/test content or realized latents read.
- Failed-cell train null range: 0.051240, close to existing validation 0.051167.
  Median entity range 0.050248; 51.36% of null entities exceed .05 individually.
- Central observed-gap mass retains 88.49% of the range. The reaction is not
  confined to a few entities, early histories or only tail bins.
- The variation resides in the bilinear copy route. The fresh-mark term only
  attenuates it. Shared direct-route gradients transmit local cross-context
  influence, but null self-descent also raises the response. Strong gradient
  conflict or a unique historical cause is **not** established.
- Three new diagnostic tests; 16 focused tests passed. Checkpoint state hashes
  unchanged, and 24 checkpoint/array checksums verified after analysis.
- V2 registered: replace one shared rank-32 route with two observed-context
  rank-16 routes; preserve 5,440 route / 133,549 total parameters, objectives,
  data and pilot gates. No known active-label mask. Per-context capacity falls;
  shared-feature influence and within-context spurious responses remain possible.

V1 execution retained below:

- Historical base: GitHub `main` at `de8fa70`; repaired SAF v6 failed its pilot.
- Original protocol: `a191392`; exact architecture/oracle: `34cdf66`.
- Model, prevalence data and pilot contract: `924131b`; API fix: `43ceef8`.
- Exact CPU/GPU source: `0d3be638e180695bf62379af8444398bb0c9708e`.
- C0/U0/U1/B0/B1 implemented. The four aligned candidates have matching
  parameter keys/shapes/initialization. U1 and B1 have 133,549 parameters each.
- All eight pi x kappa train/validation views created. Original entity splits
  retained; train-only oracles PASS at every pi. No test bodies materialized.
- Relevant tests: **54 passed**. Two CPU runs have identical histories and
  best-state tensors; train objective decreased **48.16%**.
- GPU: **12 fits**, U1/B1 x kappa 0/1 x pi 0.05/0.10/0.25, seed 20260930.
- CS-B1: pi=0.05 PASS, pi=0.10 PASS, **pi=0.25 FAIL**. At pi=0.25,
  kappa=1/label=0 copy/repeat ranges are **0.051167 / 0.050313**, exceeding 0.05.
  Active ranges are 0.367779 / 0.361659; absent active signal is not the failure.
- Pi=0.50 **data/oracle completed; training not run** under the stop rule.
- All 12 runs: exact zero-gap invariance, no support violations or reserved
  marks, finite generated values. 24,576 generated entities retained locally.
- Validation used for checkpoint selection and predictive-response audit;
  test content not accessed. No five-seed or real-data study started.

Read these files in order when recovering from a missing conversation:

1. [Latest v2 pilot result, limits and next diagnostic](v2_pilot_v1_report_2026_09_16.md).
2. [Machine-readable v2 evidence](v2_pilot_v1_result.json).
3. [V2 preregistration and implementation contract](revision_v2_preregistration.md).
4. [Prior checkpoint analysis](checkpoint_forensics_v1_report_2026_09_16.md)
   and [evidence](checkpoint_forensics_v1_result.json).
5. [V1 pilot report](pilot_v1_report_2026_09_16.md) and [evidence](pilot_v1_result.json).
6. [Frozen v1 pilot contract](pilot_execution_contract_v1.md).
7. [Research protocol](research_protocol_v1.md).
8. [Exact architecture and oracle contract](architecture_and_oracle_v1.md).
9. [Earlier oracle-only result](oracle_audit_v1_report_2026_09_16.md).

**Next work: separately design/register a loss-weighting diagnostic, preserving
both completed failures.** A proposed extra control adds globally averaged
repeat BCE to the same v2 base objective. Comparing it with B1 keeps the global
direct-route loss coefficient equal while changing the allocation by context.
It must also account for U1's rare-null failure; neither imbalance nor shared
route interference alone has been shown to explain all results. This control
is not implemented or run. Do not resume stopped v1/v2 prevalences or move to
five-seed confirmation. Later confirmation still requires train-only margin
calibration and exact intervention-error/transition-TV aggregation. No
superiority, learned-dilution or final method-success claim is established.

Runtime roots relative to this CS-SAF worktree:

- `data/cs_saf/prevalence_v1/`
- `artifacts/cs_saf/prepared_v1/`
- `artifacts/cs_saf/cpu_gate_v1/`
- `artifacts/cs_saf/pilot_v1/`
- `artifacts/cs_saf/forensics_v1/` (entity diagnostics and gradient arrays)

V2 output namespace: `artifacts/cs_saf/revision_v2/`, containing `cpu_gate_v1/`
and `pilot_v1/` with terminal records.

Each trained job retains its best checkpoint, history/report, intervention
audit and generated sample. Compact evidence records absolute locations,
source/config/data hashes and checked artifact checksums (100 for v1,
24 checkpoint/array checks in forensics, 21 for v2).

Store code/config commits before execution and report commits afterwards.
Preserve failures as well as successes. Runtime data/checkpoints remain on the
workstation; the GitHub repository is not their full backup.
