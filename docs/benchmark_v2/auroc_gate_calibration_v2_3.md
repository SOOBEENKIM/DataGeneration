# AUROC gate calibration v2.3

## Fixed design

This calibration was run before any learned baseline or CoF result was used.
The v2.2 DGP parameters, continuous primary endpoint, and `[0.48, 0.52]`
equivalence interval were not changed.

Each of 100 clean seeds generated disjoint train, validation, and test
entities through the production DGP path. Logistic regression and histogram
gradient boosting were evaluated in both scenarios at kappa 0 and 1.
Orientation for methods B and C was fixed using validation entities. Test
labels were never used to choose orientation.

The compared rules were:

- A: two-sided test CI wholly inside `[0.48, 0.52]`;
- B: validation-oriented usable-AUROC one-sided upper gate;
- C: repeated three-fold cross-fit usable-AUROC upper gate.

Eligibility required the exact 95% upper bound for clean familywise
false-failure to be at most 0.05, the exact lower power bound at 2% leakage to
be at least 0.80, and the exact lower power bound at 5% leakage to be at least
0.90 for every classifier/feature subgroup.

## Execution and completeness

```text
python -m scripts.calibrate_auroc_gate_v2_3 --seeds 100 --workers 8
COMPLETE under tmux (26,621.88 s; 26,622.64 s wall)

python -m scripts.summarize_auroc_calibration_v2_3
COMPLETE (1.36 s wall)
```

The raw result contracts are complete:

- 2,400 clean rows = 2 scenarios x 2 kappas x 100 seeds x 2 classifiers x
  3 methods;
- 7,200 leakage rows = 400 cells x 3 magnitudes x 2 classifiers x 3 methods;
- seeds 42 through 141 are present in every clean cell;
- each leakage magnitude has 2,400 rows.

## Results

| Method | Familywise failures | Rate | Exact 95% CI | Clean criterion | Power criterion |
|---|---:|---:|---:|---|---|
| A | 99/100 | 0.99 | [0.9455, 0.9997] | FAIL | FAIL |
| B | 90/100 | 0.90 | [0.8238, 0.9510] | FAIL | FAIL |
| C | 89/100 | 0.89 | [0.8117, 0.9438] | FAIL | FAIL |

Power is also far below the preregistered requirement. At 2% leakage, the
best observed feature/classifier/method exact lower bound is only `0.0592`
(logistic A, receiver category). At 5%, even the best lower bound is `0.7083`
(logistic A, amount), below 0.90. Gap and receiver-category power are lower.
Detailed receiver-category/direction cells are preserved separately.

## Decision

No method is eligible. No margin was widened and no method was selected based
on the observed clean sample. The v2.3 candidate therefore fails the AUROC
calibration gate and cannot be promoted. Learned CTGAN, TVAE, neural sequence,
and CoF smoke are prohibited.

Artifacts:

- `artifacts/benchmark_v2_3/auroc_calibration/clean_seed_results.csv`
- `artifacts/benchmark_v2_3/auroc_calibration/clean_cell_summary.csv`
- `artifacts/benchmark_v2_3/auroc_calibration/injected_leakage_results.csv`
- `artifacts/benchmark_v2_3/auroc_calibration/leakage_power_by_feature.csv`
- `artifacts/benchmark_v2_3/auroc_calibration/receiver_category_direction_power.csv`
- `artifacts/benchmark_v2_3/auroc_calibration/calibration_manifest.json`
