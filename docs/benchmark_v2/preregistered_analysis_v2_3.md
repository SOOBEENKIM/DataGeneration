# Benchmark v2.3 candidate amendment

Status: **fail-closed; not promoted**

This amendment is separate from and does not modify the v2.2 config,
preregistration, data, or artifacts. No learned baseline or CoF result was
available or used.

## Unchanged contract

- DGP parameters are exactly those of v2.2.
- Training N is 31,951 and test N is 31,956.
- Continuous association recovery error remains the primary endpoint.
- iid, block-1, and C1 must be approximately equal.
- block-2/4/8/full must recover monotonically.
- oracle must be approximately C0.
- Signed receiver-frequency calibration remains mandatory.
- Powered prefix positions and early/middle/late segments remain mandatory.
- Invalid reference/oracle support and invalid-generator score routing remain
  mandatory.
- Full pytest and compileall must pass.

The unchanged evidence may be reused because neither the DGP nor the primary
endpoint changed. Its immutable source is
`artifacts/benchmark_v2/v2_2_gate/gate_report.json`.

## Fanout amendment

For cumulative event time `T_it = sum_{u<=t} gap_iu`, fanout is

```text
mean_t |{receiver_is : s<t and T_is >= T_it-W}|.
```

It combines a gap-derived window with receiver categories. It is therefore a
secondary cross-channel positive control, not a channel-wise negative
control. `joint_alignment` remains the primary positive control.

The alignment-destroying path intervention and restoration rules are fixed in
`fanout_semantic_audit.md`.

## Hard channel-only negative controls

Only these summaries are hard negative controls, with the fixed maximum
absolute standardized class delta of 0.10:

- raw gap marginal;
- gap-only lag autocorrelation/velocity;
- raw receiver-category frequency;
- receiver-only repeat probability;
- receiver-only run-length distribution;
- fixed-step unique receiver count;
- raw amount marginal.

No gap-derived time-window fanout or other cross-channel window summary is
eligible as a negative control.

## Single-row AUROC calibration rule

The v2.3 config fixed the sample sizes, 100 clean seeds, classifiers,
validation-only orientation, three candidate gates, leakage features,
magnitudes, directions/categories, and exact-binomial criteria before the
calibration run.

A method is eligible only if:

- clean familywise false-fail exact upper bound is at most 0.05;
- every classifier/feature 2% leakage power exact lower bound is at least
  0.80;
- every classifier/feature 5% leakage power exact lower bound is at least
  0.90;
- test labels are never used for orientation.

Strictness order is A, then B, then C. The first eligible method would be
selected. The completed calibration found no eligible method, so
`selected_method` is `null`. This is a mandatory v2.3 CPU gate failure, not a
reason to widen the interval or alter the DGP.

## Conditional learned smoke

Learned smoke is authorized only if every retained and revised CPU check is
PASS. With no eligible AUROC gate, authorization is false. Therefore:

- `scripts.run_benchmark_v2` and its learned-smoke contract are not invoked;
- GPU selection is not performed;
- CTGAN, TVAE, neural baseline, and CoF smoke are not run;
- no five-seed experiment or sweep is run.
