# CS-SAF recoverable research state

Updated 2026-09-16. Branch: `research/cs-saf`.

**Current state: design frozen, train-only oracle PASS, learned model pending.**

- Historical base: `main` at `de8fa70`; repaired SAF v6 failed its pilot.
- Original CS-SAF protocol: `a191392`.
- Exact architecture plus oracle execution source: `34cdf66`.
- Oracle: both original kappa cells, train only, two byte-identical runs.
- Active-context mean response 0.413742; every null response 0.
- Relevant tests: 37 passed.
- Validation/test content accessed in this stage: none.
- New model training, pi-grid materialization, real-data comparison: not run.

Read these files in order when recovering from a missing conversation:

1. [Research protocol](research_protocol_v1.md).
2. [Exact architecture and oracle contract](architecture_and_oracle_v1.md).
3. [Oracle result and limitations](oracle_audit_v1_report_2026_09_16.md).
4. [Machine-readable evidence](oracle_audit_v1_result.json).

Next implementation: rank-32 bilinear gap/context route, observable-repeat
balanced auxiliary loss, parameter-matched controls, pi-grid materializer with
unchanged entity split membership, and CPU gates. Then run the registered
pilot, stopping on a failed frozen gate. The old failed noninferiority
calibration must be addressed using train-only evidence before confirmation.

Store code/config commits before execution and report commits afterwards.
Preserve failures as well as successes. Runtime data/checkpoints remain on the
workstation; the GitHub repository is not their full backup.
