# AUROC power analysis v2.4

Status: Step A complete under the superseding integrated-global-maximum
amendment.

The earlier classifier-specific rank-49 output is preserved at
`artifacts/benchmark_v2_4/pre_amendment_step_a_rank49/` and is not used here.

## Immutable inputs and statistic

The analysis used method-A point estimates from the immutable v2.3 CSVs:

- clean SHA-256:
  `cd9a9c01781a358bed9119dd97e5d52ef0bc759b7ce29a001ef6fd96a9f7d628`
- injected SHA-256:
  `15c5ec5c9823d588c358d01773840d0772229c6fd2e183afda9392f7126c7bbd`
- preregistered source commit:
  `3510e6134b52dd67539133d86ed8cf155b1edc65`
- resolved candidate-config SHA-256:
  `5a55571495e388fce0d28d2dcd9dc8d70733b0341d571e97356755aff661365a`
- Step-A code SHA-256:
  `b752a1d45da00bbe85a5796bf055b75da10425920f6df0742c8e1e42735e1481`

For each seed, `T` is the maximum absolute AUROC departure from 0.5 over all
four scenario/kappa cells and both classifiers. Classifier-specific power uses
the same eight-value maximum: the target classifier contributes injected
values, while the other classifier contributes clean values.

## Conservative rank-200 proxy

The 100-source-seed observed global maximum was `0.0238882960`. The
preregistered Gaussian-Bonferroni upper prediction was `0.0326936518`, so the
larger latter value is the current-N threshold proxy. It uses a one-sided 95%
upper SD bound, 95% uncertainty for the absolute mean, and tail
`1/(2*8*201)`. This is only a conservative Step-A proxy; the Step-B threshold
will be the observed maximum of 200 new calibration `T` values.

The required integer audit multipliers were:

| Feature | Classifier | 2% multiplier / modeled power | 5% multiplier / modeled power |
|---|---|---:|---:|
| amount | HGB | 9 / 0.80 | 2 / 0.97 |
| amount | logistic | 8 / 0.83 | 2 / 1.00 |
| gap-bin proxy | HGB | 25 / 0.83 | 3 / 1.00 |
| gap-bin proxy | logistic | 27 / 0.82 | 3 / 1.00 |
| receiver | HGB | 11 / 0.82 | 2 / 0.98 |
| receiver | logistic | 8 / 0.82 | 2 / 0.98 |

The largest requirement is 27x. Applying the fixed 1.25 safety factor gives
`ceil(27 * 1.25) = 34x`.

## Frozen Step-B decision

The audit-only entity counts are:

- train: 1,086,334;
- orientation-validation: 271,626;
- test: 1,086,504.

All three splits use the same 34x multiplier. Learned-generator training
remains 31,951 entities and its evaluation plan is unchanged.

No variance-reduction alternative is triggered because the prespecified
unrealistic-N check applies to the pre-safety requirement, 27x, which is not
greater than 32x. No feature's 2% AUROC requirement is downgraded. Amount,
continuous gap, and receiver category remain hard at both 2% and 5%, and their
direct row-marginal guards also remain hard.

The `1/sqrt(N)` model is an optimistic lower bound. It may understate
classifier-fit or orientation variance that does not scale ideally. The
Step-B one-seed smoke must record this variance-floor risk alongside runtime
and peak memory; it cannot estimate a variance floor from one seed.

Detailed machine-readable evidence is under
`artifacts/benchmark_v2_4/step_a/`.
