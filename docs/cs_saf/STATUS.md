# CS-SAF recoverable research state

Updated 2026-09-16. Branch: `research/cs-saf`.

**Current state: implementation and CPU checks complete; pilot FAIL at pi=0.25
because CS-B1 exceeds the frozen null-response ceiling.**

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

1. [Latest pilot report, interpretation and next work](pilot_v1_report_2026_09_16.md).
2. [Machine-readable pilot evidence](pilot_v1_result.json).
3. [Frozen pilot execution contract](pilot_execution_contract_v1.md).
4. [Research protocol](research_protocol_v1.md).
5. [Exact architecture and oracle contract](architecture_and_oracle_v1.md).
6. [Earlier oracle-only result](oracle_audit_v1_report_2026_09_16.md).

**Next work is failure analysis and a separately preregistered revision, not
five-seed confirmation of v1.** Inspect saved checkpoints on train histories to
locate the null-context response and compare U1/B1. The present one-seed result
does not isolate the cause. Do not tune thresholds or continue pi=0.50 as if the
pilot passed. Any new candidate needs a new contract and the same oracle/CPU/
pilot gates. Before later confirmation, resolve train-only noninferiority
calibration and freeze exact intervention-error/transition-TV aggregation.
No superiority or learned-dilution claim is established.

Runtime roots relative to this CS-SAF worktree:

- `data/cs_saf/prevalence_v1/`
- `artifacts/cs_saf/prepared_v1/`
- `artifacts/cs_saf/cpu_gate_v1/`
- `artifacts/cs_saf/pilot_v1/`

Each trained job retains its best checkpoint, history/report, intervention
audit and generated sample. Compact evidence records absolute locations,
source/config/data hashes and 100 checked artifact checksums.

Store code/config commits before execution and report commits afterwards.
Preserve failures as well as successes. Runtime data/checkpoints remain on the
workstation; the GitHub repository is not their full backup.
