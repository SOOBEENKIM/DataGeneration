# Gate B forensic diagnosis

## Scope

This diagnosis used only preserved v2.0 full-data artifacts and no-model
generators. No learned baseline, CoF result, or GPU job was run. The command
used v2b at all five kappa values, ten sampling/oracle seeds, 4/8-bin
decompositions, block lengths 1/2/4/8/full, and 2,000 entity-bootstrap
resamples:

```text
python -m scripts.gate_b_forensic_v2 --seeds 10 --bootstrap-resamples 2000
COMPLETE (1,234.95 s; 1,235.35 s wall)
```

## Finding: support-collapse bias

At v2b kappa 1, real/C1 `joint_alignment` standard deviation is 0.05349.
Conditional i.i.d. and block-1 standard deviations collapse to 0.01291 and
0.01290. Their label deltas are also approximately zero:

| Generator | Mean delta_joint | Mean 8-bin invalid bins | Dropped-bin gap | Occupancy-penalized gap |
|---|---:|---:|---:|---:|
| C1 | 0.00017 | 0 | 0.05219 | 0.05219 |
| conditional i.i.d. | 0.00013 | 2 | 0.03242 | 0.28242 |
| block 1 | 0.00012 | 2 | 0.03358 | 0.28358 |
| block 2 | 0.03885 | 0 | 0.01749 | 0.01749 |
| block 4 | 0.05664 | 0 | 0.00792 | 0.00792 |
| block 8 | 0.06280 | 0 | 0.00627 | 0.00627 |
| block full | 0.06846 | 0 | 0.00547 | 0.00547 |
| oracle | 0.07007 | 0 | 0.00466 | 0.00466 |

The current evaluator converts a low-count synthetic bin to `NaN` and averages
only the remaining bins. Thus i.i.d. loses two 8-bin tails and appears better
than C1. With either an explicit occupancy penalty or worst-score treatment,
that apparent advantage reverses.

Across kappas, C1, oracle, and real references have no invalid 4/8-bin cells.
I.i.d. and block 1 lose as many as two 8-bin cells. This is generator support
collapse, not benchmark-reference failure.

## Continuous association endpoint

For v2b:

```text
delta_joint(data) =
    mean(joint_alignment | y=1) - mean(joint_alignment | y=0)

association_recovery_error(model) =
    abs(delta_joint(real_test) - delta_joint(model))
```

Mean recovery errors show the intended ordering without dropped-bin behavior:

| kappa | C0 floor | oracle | block full | block 8 | block 4 | block 2 | block 1 | i.i.d. | C1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | .00393 | .00484 | .00700 | .00615 | .00622 | .00685 | .00591 | .00569 | .00529 |
| 0.3 | .00460 | .00277 | .00268 | .00250 | .00492 | .00842 | .01937 | .01928 | .01857 |
| 0.5 | .00418 | .00317 | .00228 | .00459 | .00538 | .01341 | .03331 | .03361 | .03431 |
| 0.7 | .00609 | .00335 | .00246 | .00510 | .01066 | .02203 | .05118 | .05091 | .05090 |
| 1.0 | .00436 | .00241 | .00429 | .00954 | .01570 | .03349 | .07222 | .07221 | .07217 |

The artifact also reports standardized delta, Hedges' g,
label-conditional Wasserstein distance, recovery ratio, sign consistency, and
entity-bootstrap 95% delta intervals. At kappa 1, block-2/4/8/full deltas are
0.03885/0.05664/0.06280/0.06846, approaching the oracle delta 0.07007;
i.i.d. and block 1 stay near zero.

## Hypothesis adjudication

- **H-B1 supported.** I.i.d. and block-1 support collapses around zero, their
  variance and tail occupancy shrink, and missing-bin exclusion creates the
  artificial 8-bin advantage. Penalty/worst-score treatment removes it.
- **H-B2 rejected.** Conditional i.i.d. does not retain the label-dependent
  continuous joint association. Its delta and block-1 delta remain near zero,
  and their recovery errors match C1.
- **H-B3 partially supported as an evaluator bug, not a C1 construction
  error.** A hand-computed regression test proves that an empty synthetic bin
  is excluded by the old macro estimator. C1 itself retains full support and
  continuous C1/i.i.d./block-1 ordering is correct.

## Endpoint alternatives

- **A — continuous primary:** preserves all entities, has no support-dependent
  bin dropping, and directly recovers the intended oracle/C0 → sequential
  blocks → i.i.d./C1 ordering.
- **B — 8-bin plus occupancy penalty:** detects collapse, but mixes prevalence
  error with an arbitrary additive missing-bin scale (i.i.d. 0.282 at kappa 1).
- **C — 8-bin plus worst score:** also detects collapse but is dominated by an
  arbitrary worst-bin constant (i.i.d. 0.274 at kappa 1).

Alternative A is selected for the v2.2 candidate. Eight-bin results remain
confirmatory; invalid generator support is an invalid model score and cannot
improve it. Four/two-bin tables remain descriptive.

## Artifacts

Artifacts are under `artifacts/benchmark_v2/gate_b_forensics/`:

- `bin_decomposition.csv`
- `joint_alignment_distributions.csv`
- `continuous_association.csv`
- `endpoint_comparison.csv`
- `endpoint_comparison_summary.csv`
- `forensic_manifest.json`
