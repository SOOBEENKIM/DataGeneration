# Preregistered benchmark-v2 analysis

Primary scenario is `joint_semimarkov_v2b` at kappa 1.0. The primary endpoint
is the 8-bin macro coherence gap on `joint_alignment`; 4-bin is confirmatory
and 2-bin descriptive. The primary learned comparisons are CoF against each
conditional row baseline. Full runs use model seeds 1–5, separate sampling
seeds +1000, bootstrap seeds +2000, entity bootstrap with 2,000 resamples,
Welch tests, Hedges' g, bootstrap difference confidence intervals, and Holm
correction. Runs with missing samples/manifests or bins below the preregistered
minimum count are invalid, never zero-filled.

## Benchmark-calibration amendment: v2.1 candidate

Receiver-only diagnostics completed before any learned-model run showed raw TVD
has a 100% false-fail rate at the current N under clustered sequences and
decays at N^-1/2. The v2.1 candidate therefore preregisters:

- 4x train/test N; no other DGP parameter or seed change;
- raw receiver TVD remains descriptive;
- receiver equivalence uses both entity-bootstrap single-row AUROC entirely
  inside [0.48, 0.52] and a simultaneous signed per-category cluster CI within
  +/-0.02;
- 2,000 entity bootstrap resamples;
- a fixed 5% category-0 fraud-row contamination stress test with >=90%
  detection power and <=5% null false-fail;
- all original Gate B/C/D/E requirements remain unchanged and fail closed.

The amendment is identified as `benchmark_v2.1-candidate`; it does not overwrite
v2.0 configs, data, or artifacts.
