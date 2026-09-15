# Benchmark v2.4 global-maximum AUROC amendment

This amendment is committed before the superseding Step A. It replaces the
classifier-specific statistic and every rank-49/50 or rank-96/100 rule in the
earlier pre-Step-A amendment. The immutable input CSV hashes and unit-test
source are retained. Any output produced under the earlier ranks is preserved
under `artifacts/benchmark_v2_4/pre_amendment_step_a_rank49/`, is marked
superseded, and cannot inform the final v2.4 audit multiplier or gate.

## One integrated familywise statistic

For each seed `s`, define exactly one statistic:

```text
T(s) = max over 2 scenarios x 2 kappa x 2 classifiers
       |test_AUROC(s, cell, classifier) - 0.5|.
```

All clean false-fail and injected-leakage detection events are the strict event
`T(s) > t_hat`. There are no classifier-specific thresholds and no subsequent
combination of classifier-level PASS/FAIL decisions.

Feature/classifier power rows remain reportable. For a target classifier row,
its four injected cell AUROCs replace its four clean cell AUROCs and the other
classifier contributes its four clean cell AUROCs. The resulting eight values
produce the same integrated `T(s)` and use the same `t_hat`. This isolates
classifier-specific sensitivity without changing the gate statistic.

The earlier mean-of-two-classifiers variance-reduction option is superseded
because it would change this statistic. If a variance-reduction alternative is
required, only repeated-cross-fit averaging within each classifier is allowed;
the final statistic remains the maximum of all eight classifier-cell values.

## Step-B calibration and validation

- Calibration uses clean seeds 1000 through 1199.
- Independent clean validation uses seeds 2000 through 2199.
- `t_hat` is the maximum of the 200 calibration `T` values: non-interpolated
  order statistic rank 200/200.
- Clean validation passes only when the Clopper-Pearson two-sided exact 95%
  upper bound for `T(s) > t_hat` is at most 0.05.
- Feature/classifier 2% power uses the same event and must have exact 95% lower
  bound at least 0.80 unless that feature is downgraded before Step B.
- Every feature/classifier 5% power uses the same event and must have exact 95%
  lower bound at least 0.90.
- Test labels do not select orientation.

The maximum calibration threshold targets a clean exceedance probability near
`1/(200+1)`, rather than nominal 5%, so independent 200-seed validation can
realistically establish an exact upper bound of 0.05.

## Step-A rank-200 threshold proxy

The immutable v2.3 source has only 100 clean seeds. Step A therefore does not
claim to observe a rank-200 order statistic. At current audit N, it uses the
larger of:

1. the observed maximum of the 100 integrated `T` values; and
2. a preregistered Gaussian-Bonferroni upper prediction bound.

For each of the eight cell-classifier signed deviations, the prediction bound
is

```text
abs(mean) upper uncertainty
+ normal_quantile(1 - 1/(2 * 8 * 201))
  * one-sided 95% chi-square upper bound for SD.
```

The first term is `abs(sample mean) + t_0.975,99 * SD/sqrt(100)`. Step A takes
the maximum bound over the eight dimensions. This is an explicitly
conservative proxy for the expected Step-B maximum; it is not a calibrated
threshold.

Power holds the observed cell-mean leakage shift fixed, divides all eight
clean deviations and the proxy threshold by `sqrt(common audit multiplier)`,
and finds the smallest integer multiplier meeting the fixed target. Because
classifier fitting and validation-orientation noise may not scale exactly,
this remains an optimistic lower bound. The largest required multiplier is
multiplied by the already fixed 1.25 safety factor.

## Scope and variance-floor risk

The chosen multiplier applies equally to AUROC-audit train,
orientation-validation, and test entity counts. Generator training remains
31,951 entities and the learned-model evaluation plan is unchanged.

Before the full Step-B run, the one-seed smoke records runtime, projected
runtime, and peak memory. It also evaluates nested base-N and selected-N clean
deviations and compares the selected-N result with the `1/sqrt(N)` projection.
This is only a risk signal: one seed cannot estimate a variance floor. The
limitation and any adverse signal are recorded before the full run.

All leakage operators, direct row-marginal hard guards, identical application
to audit train/orientation-validation/test, and the ban on post-Step-B changes
remain as committed in the earlier amendment.
