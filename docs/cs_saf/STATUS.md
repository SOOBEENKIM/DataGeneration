# CS-SAF recoverable research state

Updated 2026-09-16. Branch: `research/cs-saf`.

**Current state: v1 remains FAIL at pi=0.25; train-only checkpoint forensics
complete; one v2 revision preregistered, not implemented or trained.**

Latest work (2026-09-16):

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

1. [Latest checkpoint analysis and limits](checkpoint_forensics_v1_report_2026_09_16.md).
2. [V2 preregistration and implementation contract](revision_v2_preregistration.md).
3. [Machine-readable forensic evidence](checkpoint_forensics_v1_result.json).
4. [V1 pilot report](pilot_v1_report_2026_09_16.md) and [evidence](pilot_v1_result.json).
5. [Frozen v1 pilot contract](pilot_execution_contract_v1.md).
6. [Research protocol](research_protocol_v1.md).
7. [Exact architecture and oracle contract](architecture_and_oracle_v1.md).
8. [Earlier oracle-only result](oracle_audit_v1_report_2026_09_16.md).

**Next work: implement the registered v2 route, verify exact capacity and
cross-bank gradient isolation, pass the CPU gate, then execute the separate v2
pilot with unchanged thresholds and stop rule.** Do not resume v1 pi=.50 or
five-seed confirmation. V2 success has not been observed. Before later
confirmation, resolve train-only noninferiority calibration and freeze exact
intervention-error/transition-TV aggregation. No superiority or learned-dilution
claim is established. The v2 pilot still reuses development data and is exploratory.

Runtime roots relative to this CS-SAF worktree:

- `data/cs_saf/prevalence_v1/`
- `artifacts/cs_saf/prepared_v1/`
- `artifacts/cs_saf/cpu_gate_v1/`
- `artifacts/cs_saf/pilot_v1/`
- `artifacts/cs_saf/forensics_v1/` (entity diagnostics and gradient arrays)

Future v2 output namespace: `artifacts/cs_saf/revision_v2/` (not executed).

Each trained job retains its best checkpoint, history/report, intervention
audit and generated sample. Compact evidence records absolute locations,
source/config/data hashes and 100 checked artifact checksums.

Store code/config commits before execution and report commits afterwards.
Preserve failures as well as successes. Runtime data/checkpoints remain on the
workstation; the GitHub repository is not their full backup.
