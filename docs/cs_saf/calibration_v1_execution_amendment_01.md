# Execution amendment 01: cuDNN TF32 response verification

2026-09-20. Registration `54811a1`, initial implementation `b66c2f1`.
23/80 cells completed. The 24th (pi=.10, kappa=1, trial=0, Ecal)
converged and saved calibration/generation/conditional evidence, but the duplicate
response-aggregation check stopped dispatch before final generation scoring.

The active copy-range difference was -1.0022077467e-6 (limit 1e-6).
The **parent E** already had -9.6125163784e-7 under the same two audit paths.
This is not a failed optimizer or a reason to change the scientific screen.

## Controlled reproduction

The same saved calibrated state was evaluated with mixed-context vs context-grouped
GRU batches. The model state, inputs and thresholds were held fixed.

| Probe | Active mean copy-range delta | Maximum entity delta (copy) |
|---|---:|---:|
| cuDNN TF32 on; different history batching | -1.0022751793e-6 | 1.3246238232e-5 |
| cuDNN TF32 off; different history batching | -1.0408956004e-9 | 1.8160790205e-8 |
| Same mixed histories, only response chunk differs, TF32 on | -1.4266002333e-9 | 2.2848447134e-8 |

The discrepancy is explained by batch-dependent reduced-precision GRU inference,
not the entity reduction or copy-head chunk size. Observable repeat ranges agree
with the same diagnosis. Full probe evidence is retained in
`artifacts/cs_saf/calibration_v1/rounding_probe.json`.

## Bounded correction and preservation

The scientific fit, saved parent states, probability transform, native generation,
original CUDA numeric settings, saved reference scores, oracle evaluation and
reported endpoints stay unchanged. No scientific margin or 1e-6 verification
tolerance is relaxed. Instead, a separate **full-precision verification** compares
the two history-batching and reduction paths with cuDNN/matmul TF32 disabled,
records the original discrepancy, and restores numeric flags afterwards. This
distinguishes an implementation inconsistency from TF32 approximation. The small
recorded TF32 discrepancies remain disclosed; full precision is not silently
substituted into only one side of the scientific comparison.

All subsequent cells use this verification. The 23 completed original cells retain
their passing original check and their original manifests/source `b66c2f1`; this
specific source is permitted for resume. Their scientific evidence is not rewritten.
The failed cell, its log, first gate records and scheduler snapshot are archived
before rerun. Its fitted parameters and saved scientific arrays must match the
archived attempt; otherwise stop and investigate. Report 80 completed calibration
fits plus one repeated technical attempt, not 80 independent new model trainings.

Before resumption: repeat CPU tests and same-source CPU/GPU gates; a GPU regression
reproduces the original failing saved state and verifies the unchanged 1e-6 check
passes using the full-precision control. Complete the remaining fixed grid without
changing models, data, fit bounds, seeds, outcomes or scientific criteria.

The first CPU smoke launch had also been blocked before fitting because the source
commit had not finished. No output cell was created; the existing clean-source
guard worked. The subsequent same-source CPU/GPU gates passed before any grid fit.
