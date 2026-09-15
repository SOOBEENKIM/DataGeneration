# Benchmark v2.4 preregistered analysis

Status: CPU gate PASS; conditional learned smoke authorized.

This final document is committed before any learned-generator smoke. No CTGAN,
TVAE, neural baseline, or CoF result was available or used to choose any DGP
parameter, endpoint, metric, threshold, audit N, leakage operator, or
hard/descriptive classification.

## Unchanged scientific contract

- DGP parameters remain the v2.2 parameters.
- Learned-generator training N remains 31,951 in the model evaluation plan.
- Continuous association recovery is the primary endpoint.
- iid, block-1, and C1 remain approximately equivalent.
- block-2/4/8/full retain monotonic recovery.
- oracle remains approximately C0.
- Signed receiver-frequency, position/segment stationarity, reference/oracle
  support, and invalid-generator routing remain hard.
- `joint_alignment` is the primary joint positive control.
- Gap-derived time-window fanout is a secondary joint positive control.
- Only true channel-wise summaries are hard negative controls.

The retained evidence is immutable and referenced from
`artifacts/benchmark_v2/v2_2_gate/gate_report.json`,
`artifacts/benchmark_v2_3/fanout_semantic_audit/`, and
`artifacts/benchmark_v2_3/channel_controls/`.

## Final single-row AUROC gate

For each audit seed:

```text
T = max over 2 scenarios x 2 kappa x 2 classifiers
    |validation-oriented test AUROC - 0.5|.
```

Orientation is fixed solely by the audit orientation-validation split. Test
labels never choose orientation.

Audit train, orientation-validation, and test use the common 34x multiplier:
1,086,334 / 271,626 / 1,086,504 entities. These are audit-only datasets with
seeds and provenance separate from learned-generator training.

Calibration uses seeds 1000--1199. The non-interpolated rank-200/200 maximum is

```text
t_hat = 0.0040435352475451936.
```

Independent clean validation seeds 2000--2199 produced 2/200 strict
`T > t_hat` failures. The Clopper-Pearson exact 95% upper bound is
0.035654668397239776, below 0.05.

For classifier-specific power, the target classifier contributes four
injected cell AUROCs and the other classifier contributes four clean cell
AUROCs to the same integrated maximum. All 12
feature/classifier/magnitude cells detected 200/200 trials. Their exact 95%
lower bound is 0.9817246596448638, above 0.80 at 2% and 0.90 at 5%.

Amount, continuous gap, and receiver category remain hard AUROC requirements
at 2% and 5%. Direct marginal hard guards also remain: gap and amount
KS/effect-size/emission contracts, and receiver signed cluster-frequency.

The fixed leakage schedule and exact split-wise operator definitions are those
in `preregistered_analysis_v2_4_step_b_decision.md`; they are not changed by
the observed results.

## CPU gate decision

`artifacts/benchmark_v2_4/gates/gate_report.json` records:

- every retained DGP/endpoint check PASS;
- fanout semantic reclassification PASS;
- true channel-only controls PASS;
- calibrated global-maximum AUROC PASS;
- full pytest PASS: 112 tests;
- compileall PASS;
- `learned_smoke_authorized=true`;
- `full_experiment_authorized=false`.

## Conditional learned smoke

Only the following are authorized:

- one conditional CTGAN smoke;
- one conditional TVAE smoke;
- one neural sequence baseline smoke;
- one CoF smoke;
- primary challenge scenario `joint_semimarkov_v2b`;
- kappa 1.0 only;
- seed 1 only;
- at most 512 train entities;
- 100 requested training steps;
- one independent process per selected GPU when DDP is unavailable.

Each run must use `scripts.run_benchmark_v2` and save sample, checkpoint,
metrics, runtime, peak GPU memory, and manifest artifacts. The runner rejects
multiple seeds, multiple kappas, multiple generators per invocation, train
N above 512, and non-smoke mode.

Smoke results cannot alter any v2.4 rule. No five-seed run, sweep, or
`FULL_EXPERIMENT` is authorized.
