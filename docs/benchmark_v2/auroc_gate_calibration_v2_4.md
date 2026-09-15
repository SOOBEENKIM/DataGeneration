# AUROC gate calibration v2.4

Status: PASS.

This calibration replaces only the single-row AUROC component. It does not
change the DGP, continuous primary endpoint, learned-generator training N, or
any retained v2.2/v2.3 evidence.

## Design

- Integrated statistic: maximum absolute AUROC departure across four
  scenario/kappa cells and two classifiers.
- Audit multiplier: 34x for train/orientation-validation/test.
- Calibration: seeds 1000--1199, maximum rank 200/200, no interpolation.
- Validation: independent seeds 2000--2199.
- Detection event: strict `T > t_hat`.
- Leakage: amount, continuous gap, and receiver category at 2% and 5%.
- Orientation: orientation-validation labels only; test labels never choose
  sign.

## Result

The calibrated threshold is `0.0040435352475451936`.

Clean validation produced 2/200 false failures:

```text
rate = 0.01
exact 95% CI = [0.0012133490, 0.0356546684]
required upper <= 0.05
result = PASS
```

Every one of the 12 feature/classifier/magnitude cells detected 200/200
injections:

```text
power = 1.0
common exact 95% lower = 0.9817246596
2% required lower >= 0.80
5% required lower >= 0.90
result = PASS
```

All four fixed receiver target-label/category operators detected 50/50 per
classifier and magnitude. Each amount and gap target-label direction detected
100/100.

The full run took 2,382.11 seconds with eight workers. It produced 1,600
cell-seed checkpoints, 3,200 clean cell rows, 9,600 leakage cell rows, 400
clean global-T rows, and 2,400 leakage global-T rows. Maximum worker RSS was
414,404 KiB.

Machine-readable evidence is under
`artifacts/benchmark_v2_4/auroc_calibration/`.
