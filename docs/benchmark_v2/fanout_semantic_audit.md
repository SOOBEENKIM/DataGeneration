# Fanout semantic audit

## Definition

For entity `i`, cumulative event time is

```text
T_it = sum_{u<=t} gap_iu
```

and the implemented fanout is

```text
fanout_i =
  mean_t |{receiver_is : s<t and T_is >= T_it - W}|
```

The code constructs the window from gap-derived cumulative time and counts
unique receiver categories inside it. Fanout is therefore a cross-channel
summary. It is not a receiver-only or gap-only negative control.

## No-model intervention

The intervention permutes each intact receiver path among entities with the
same label and length. Gap, amount, label, length, and mask stay fixed. Since
whole paths are exchanged, receiver category frequencies, repeat rates, run
lengths, and fixed-step unique-receiver counts are preserved exactly as
label-stratified multisets. The original mapping restores shared alignment.

```text
python -m scripts.audit_fanout_semantics_v2_3 --permutations 10
PASS (490.91 s; 491.32 s wall, final run including gap-only velocity)
```

At v2b kappa 1:

| Summary | Original delta | Alignment-destroyed delta | Absolute ratio |
|---|---:|---:|---:|
| joint_alignment | 0.06959 | 0.00022 | 0.00322 |
| time-window fanout | -0.34947 | -0.04086 | 0.11693 |

Restoring the original mapping restores both deltas exactly. The maximum
change across true channel-only control deltas is `1.33e-15`:

- raw gap mean: exact;
- gap lag-1 autocorrelation: exact;
- gap-only time-window velocity: exact;
- receiver repeat rate: within `1.11e-16`;
- receiver mean run length: within `4.44e-16`;
- fixed-step unique receiver: within `1.33e-15`;
- raw amount mean: exact;
- raw receiver category counts: exact by assertion.

At kappa 0, both joint signals are sampling noise and the intervention has no
scientific positive-control expectation.

## Decision

For benchmark-v2.3:

- `joint_alignment` remains the primary joint positive control;
- gap-derived time-window fanout becomes a secondary joint positive control;
- fanout is removed from channel-only negative controls;
- fixed-step unique receiver count is added as the receiver-only diversity
  negative control.

The regression tests verify the formula's gap independence for fixed-step
diversity and exact path-multiset preservation by the intervention.

Artifacts are under
`artifacts/benchmark_v2_3/fanout_semantic_audit/`.
