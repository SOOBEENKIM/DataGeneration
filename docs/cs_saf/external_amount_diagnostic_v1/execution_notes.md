# Execution notes

- Parent scientific results: `c4d4c7`; diagnostic registration/source: `76f8230`.
  Registration and source were pushed before model evaluation. Reporting source
  only aggregates the registered arrays; no candidate/threshold changes.
- The CPU test collection initially used an incorrect package import for the
  existing engineering fixture. It was corrected before registration/execution;
  all three tests passed. No trained checkpoint or scientific result was involved.
- Both dataset diagnostic runs completed on their first execution. Zero failed
  or retried scientific runs, zero optimizer updates, zero new sampling runs.
- Input/code/checkpoint hashes, fit-only thresholds, exact plan source/lengths,
  static equality, reserved categorical roundtrips and strict-past boundary
  behavior were checked. Real/generated hybrid histories are deliberately
  descriptive sensitivity probes, not coherent alternate rollouts.
- Saved float64 raw amounts are re-encoded into the original float32 history
  codec for replay. The original transient float32 sampler state was not saved;
  roundoff at re-encoding can differ by about a float32 unit. Prefix/batch replay
  is numerically checked, not claimed to be a bit-exact original sampling trace.
- Four generated datasets are reused. Validation has 335,658 events; the paired
  generated plans have 394,676 events across both datasets and seeds. Six cases
  per tape plus two complete validation arrays make 26 stored arrays. Duplicated
  plan source entities and two draws from fixed weights are not independent fits.
- No empirical tail cap, zero replacement, threshold adjustment, model selection,
  or follow-up head training was applied after inspecting results.
