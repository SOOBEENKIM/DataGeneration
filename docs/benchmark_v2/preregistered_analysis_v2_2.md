# Preregistered benchmark-v2.2 candidate analysis

This amendment was fixed using no-model v2.0 forensics before any learned
baseline or CoF result. It does not modify or overwrite v2.0/v2.1 artifacts.

## Data and scope

- schema: `benchmark_v2.2-candidate`;
- training entities remain 31,951;
- evaluation/audit entities are 31,956;
- DGP parameters, prevalence, categories, lengths, seed, kappas, and the
  v2b scientific target are unchanged;
- all runtime output uses new `data/benchmark_v2_2` and
  `artifacts/benchmark_v2/v2_2_gate` paths.

## Primary v2b endpoint

The primary metric is continuous association recovery:

```text
delta_joint(data) =
    mean(joint_alignment | y=1) - mean(joint_alignment | y=0)

association_recovery_error(model) =
    abs(delta_joint(real_test) - delta_joint(model))
```

Report delta, standardized delta, Hedges' g, label-conditional Wasserstein
distance, 2,000-resample entity-bootstrap 95% CI, recovery ratio, and sign
consistency.

Required no-model checks:

- absolute real delta at kappa 0 <=0.01 and delta is nondecreasing within
  tolerance 0.005;
- oracle mean recovery error <= C0 sampling-floor error +0.005;
- absolute difference between i.i.d. and C1 mean recovery errors <=0.005;
- absolute difference between block-1 and C1 errors <=0.005;
- for kappa >=0.3, block 2 improves over block 1 and block 2/4/8/full errors
  move toward C0, allowing 0.005 Monte Carlo tolerance;
- v2b velocity/gap/fanout/amount standardized deltas and their kappa dynamic
  ranges are negative-control diagnostics.

Eight-bin macro coherence is confirmatory. Four/two-bin tables are
descriptive. A real or oracle reference lacking required bin count is a
benchmark-validity failure. A deliberately bad generator lacking support is
an invalid model score; its missing bin is never excluded in a way that
improves the score.

## Receiver gate

Raw TVD is descriptive. Required receiver evidence combines:

- logistic and histogram-gradient single-row AUROC equivalence, with a
  preregistered one-row-per-entity cluster bootstrap interval wholly inside
  [0.48, 0.52];
- simultaneous signed entity-frequency bounds within +/-0.02;
- 100 clean seeds for v2a/v2b and kappa 0/1;
- injected leakage in categories 0, 1, 31, and 63 at 2% and 5%;
- exact binomial 95% upper bound for false-fail <=0.05;
- exact binomial 95% lower detection-power bound >=0.80 at 2% and >=0.90 at
  5%.

Only evaluation/audit N is enlarged. Training N is not multiplied.

## Positionwise stationarity

The rule was fixed from analytical power, not observed position outcomes:

- stationary burst probability 0.30;
- positions 0–15 (`t < minimum sequence length`) are hard;
- any later position with at least 1,500 at-risk entities in each label is
  also hard; other later positions are descriptive;
- position equivalence margin +/-0.04;
- early/middle/late entity-aggregated margin +/-0.025;
- no position or segment may have the same drift direction in more than 80%
  of 30 seeds;
- equilibrium-residual analytical/unit tests are hard requirements.

For 16 simultaneous comparisons, Bonferroni z=2.955 and margin 0.04 imply a
conservative minimum of 1,147 positive entities. The preregistered 1,500
threshold adds reserve.

## Promotion and learned smoke

The candidate is promoted only if revised Gate A, Gate B, Gate C, Gate E,
continuous endpoint checks, receiver calibration, position/segment checks,
full pytest, and compileall all pass. This file must be committed before any
GPU smoke.

If promoted, allowed learned work is limited to one-seed small smoke runs for
conditional CTGAN, conditional TVAE, neural sequence baseline, and CoF.
Five-seed experiments, sweeps, and any metric/threshold change based on smoke
results remain prohibited.
