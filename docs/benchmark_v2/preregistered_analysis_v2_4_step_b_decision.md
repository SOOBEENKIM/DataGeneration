# Benchmark v2.4 Step-B frozen decision

This decision is committed after Step A and before any Step-B smoke or full
calibration. After Step B begins, the multiplier, threshold definition,
leakage operators, operator schedule, and hard/descriptive classifications
cannot change.

## Audit size and seeds

The audit multiplier is 34 for all three independent splits:

| Split | Base entities | Audit entities |
|---|---:|---:|
| train | 31,951 | 1,086,334 |
| orientation-validation | 7,989 | 271,626 |
| test | 31,956 | 1,086,504 |

Calibration clean seeds are exactly 1000--1199. Independent validation clean
seeds are exactly 2000--2199. Audit data are generated from separate seed
streams and provenance records; they are not learned-generator training data.

The enlarged audit uses a dedicated vectorized stationary-row generator.
Temporal persistence/alignment is not materialized because it cannot alter the
preregistered row marginals: label prevalence is 0.05, gap is the fixed
stationary burst/normal exponential mixture, log-amount is the fixed Normal,
and receiver category is uniform. Scenario and kappa retain disjoint seed
streams. This changes no DGP parameter and avoids generating 16--32 unused
sequence events for every audit entity. Regression tests fix these population
contracts and the continuous-gap feature schema.

## Statistic, threshold, and criteria

Each seed has one `T`: the maximum absolute test-AUROC departure over four
scenario/kappa cells and two classifiers. The Step-B threshold is the maximum
of the 200 calibration `T` values, strict detection is `T > t_hat`, and no
interpolation is used.

Validation clean passes only if the Clopper-Pearson two-sided exact 95% upper
bound is at most 0.05. Feature/classifier 2% power must have exact lower bound
at least 0.80 and 5% power must have exact lower bound at least 0.90. For a
classifier-specific power row, its injected four cells replace its clean
cells, the other classifier's four cells remain clean, and the same global
maximum and threshold are used.

No 2% feature is descriptive. Amount, continuous gap, and receiver category
are hard AUROC power requirements. Direct hard guards remain:

- gap: KS, standardized effect size, emission contract;
- receiver: signed cluster-frequency;
- amount: KS, standardized effect size, emission contract.

## Fixed leakage schedule

The same operator is applied to audit train, orientation-validation, and test.
Every validation seed evaluates each feature at 2% and 5%. Operator choice is
a fixed cycle based on zero-based validation trial ordinal, independent of the
data and numeric seed value:

- amount alternates target labels 1 and 0;
- continuous gap alternates target labels 1 and 0;
- receiver cycles `(1,0)`, `(1,31)`, `(0,1)`, `(0,63)`, where each pair is
  `(target label, target category)`.

For continuous features, exactly the nearest feasible target-class count is
replaced by the fixed population anchor: log-amount mean plus 3 SD, or the
normal-gap population 99th percentile. For receiver, exactly the nearest
feasible number of non-target-category rows is changed so the realized
class-conditional category-probability increase is recorded. Selected row
indices use deterministic split-specific injection streams.

Power is aggregated over 200 validation seeds for each
feature/classifier/magnitude. The Step-A `gap_bin` effect is only the
preregistered proxy; Step B injects and evaluates continuous raw gap.

## Smoke and engineering

The one-seed smoke runs before the full calibration and records per-task
runtime, projected full runtime, peak RSS, and checkpoint/resume behavior.
It also records base-N and selected-N nested clean deviations against the
`1/sqrt(N)` projection, explicitly as a variance-floor risk signal rather
than a one-seed variance estimate.

Completed cell-seed records append immediately and are skipped on resume.
Worker BLAS/OpenMP thread counts are `floor(total logical CPUs / workers)`.
Bootstrap, method comparison, and 1% leakage are excluded.
