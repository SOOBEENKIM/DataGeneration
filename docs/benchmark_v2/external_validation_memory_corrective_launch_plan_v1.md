# External validation v1 memory-corrective launch decision

Status: plan only; not authorized and not executable from this document. No authorization manifest or runtime artifact was created.

## Preferred continuation

1. Preserve AMLSim IID `attempt_002`, CTGAN `attempt_001`, frozen CoF `attempt_001`, and failed TVAE `attempt_001`.
2. After separate explicit authorization bound to the corrective source/config and all preserved hashes, run only AMLSim TVAE as append-only `attempt_002`.
3. Expose one user-selected GPU to the process and keep the unchanged training and whole-job caps. The source-bound contract forces `DataTransformer` to one synchronous in-process worker.
4. Do not launch any CTGAN transform concurrently. Wait for TVAE `COMPLETE`, `INVALID`, or `FAILED` before any later wave decision.

Existing CTGAN is reusable because the scientific contract is unchanged and its old manifest/evaluation/terminal plus data, metric, SamplingPlan, threshold, seed, source, and config are exactly hash-bound. This is an explicit mixed-provenance record, not an implicit equivalence claim.

## Conservative alternative

If a later governance decision rejects the exact mixed-provenance exception, authorize new AMLSim CTGAN `attempt_002` and TVAE `attempt_002`. Run CTGAN to a terminal state first, then TVAE. Never run the two heavy transforms concurrently. The current task neither chooses this alternative nor authorizes either attempt.

## Sparkov timing

Do not start Sparkov now. A separate authorization may be considered only after the AMLSim TVAE continuation is terminal and its memory-policy provenance is verified. The future safe order is:

1. Sparkov CTGAN and frozen CoF may run in parallel on distinct GPUs.
2. After both are terminal, Sparkov TVAE runs alone.

This plan keeps all seeds, models, metrics, thresholds, conditioning, wall caps, attempts, and external protocol definitions unchanged. It includes no SSH command because execution remains unauthorized.
