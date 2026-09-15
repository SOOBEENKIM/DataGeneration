# Benchmark v2.4 pre-Step-A preregistration amendment

Commit this document before running Step A. It does not modify or replace any
v2.0, v2.2, or v2.3 config, preregistration, data, or artifact.

## Scope

V2.4 changes only the single-row AUROC measurement gate. DGP parameters,
continuous association recovery, generator training N=31,951, learned-model
evaluation, and every retained v2.2/v2.3 PASS check remain unchanged. No
learned generator result may inform this amendment.

`N_audit` applies only to the AUROC audit. The audit train, orientation
validation, and test entity counts are all multiplied by the same Step-A
multiplier. These data use separate seeds, paths, and provenance from learned
generator training.

## Fixed statistic and null calibration

For seed `s` and classifier `c`,

```text
T(s,c) = max over scenario x {kappa 0, kappa 1}
         |test_AUROC(s,c,cell) - 0.5|.
```

Step A uses method-A point estimates from the immutable v2.3 raw artifact.
Even-indexed source seeds form the 50-seed calibration half and odd-indexed
source seeds form the 50-seed diagnostic half. Quantiles never interpolate:
rank is `ceil((n+1)*0.95)`, hence rank 49 for Step A and rank 96 for the
100-seed Step-B calibration.

Step-B acceptance is fixed:

- validation clean familywise false-fail exact 95% upper bound <= 0.05;
- feature/classifier 2% leakage power exact 95% lower bound >= 0.80, except a
  feature explicitly downgraded before Step B under the rule below;
- every feature/classifier 5% leakage power exact 95% lower bound >= 0.90;
- orientation uses only the audit orientation-validation split;
- test labels never choose orientation.

## Power analysis and N decision

The Step-A `1/sqrt(N)` model is explicitly an optimistic lower bound because
the observed AUROC variance can contain classifier-fitting and orientation
noise. Step B scales audit train, orientation-validation, and test together
to make this approximation as plausible as possible.

For each feature/classifier/magnitude, Step A estimates the observed AUROC
shift distribution and searches the minimum common audit multiplier meeting
the fixed power requirement under the calibrated familywise threshold. The
largest required multiplier receives a 1.25 safety factor and determines
`N_audit`.

If a feature needs more than 32x, Step A evaluates, in this fixed order:

1. mean of the two classifier AUROCs;
2. repeated-cross-fit averaging.

If it still needs more than 32x, only that feature's 2% AUROC requirement may
be made descriptive. That decision must be documented and committed before
Step B. Its direct row-marginal guard remains a hard gate.

## Fixed leakage operators

The same operator is applied to audit train, orientation-validation, and test,
so the classifier learns the injected pattern and is evaluated on the same
population perturbation. Magnitudes are 2% and 5%; 1% is excluded.

- Amount: replace the exact feasible target-class fraction with the fixed
  population anchor `log-amount mean + 3 SD`.
- Continuous gap: replace the exact feasible target-class fraction with the
  fixed population normal-gap 99th percentile. Step A necessarily uses the
  existing v2.3 `gap_bin` injection as a conservative proxy.
- Receiver category: increase the selected class-conditional target-category
  probability by the exact feasible magnitude. The fixed operator list is
  `(y=1,q=0)`, `(y=1,q=31)`, `(y=0,q=1)`, `(y=0,q=63)` for every seed.

The realized integer count and realized probability change must be recorded.
Direction/category lists do not rotate with seed.

## Direct marginal hard guards

These remain hard even if a 2% AUROC requirement becomes descriptive:

- gap: KS, standardized effect size, emission contract;
- receiver category: signed cluster-frequency gate;
- amount: KS, standardized effect size, emission contract.

Failure of AUROC power never authorizes row leakage.

## Engineering and provenance contract

Step B removes bootstrap resampling, A/B/C comparisons, and 1% injections. It
sets worker `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to
`floor(total cores/workers)`, appends each completed cell-seed immediately,
and skips completed rows on resume. A one-seed smoke records task time and
projected total runtime before the only full Step-B run.

Artifacts record calibration/validation seed ranges, runtime source commit,
resolved-config SHA-256, and relevant-code SHA-256. After Step B begins, N,
threshold definition, leakage operators, and hard/descriptive classifications
cannot change.
