# Receiver gate diagnosis

## Scope and preserved state

This is a CPU-only, pre-model diagnosis. No CoF, CTGAN, TVAE, neural, or other
learned generator result was used. Benchmark v2.0 data, configs, and gate
artifacts remain unchanged. The original raw receiver TVD tolerance was not
altered.

## Minimal reproduction

The receiver-only CLI processed all 20 scenario/kappa/split datasets and 1,000
entity-label permutations per dataset in 22.48 seconds. Entities, not rows, are
the permutation unit.

At v2b kappa 1 test, pooled TVD was 0.05564 versus permutation median 0.05356
(excess 0.00208, tail probability 0.318). At v2a kappa 1 test, pooled TVD was
0.10670 versus median 0.04061 (excess 0.06609, tail probability 0.001).
These tails calibrate a finite-sample null only; they are not equivalence tests.

## No-model Monte Carlo

The Monte Carlo uses the exact production receiver transition and equilibrium
functions. It ran 30 DGP seeds for N, 2N, and 4N, both scenarios, and kappa
0/1: 360 total replicates in 2,221.76 seconds.

Key results:

| Scenario | kappa | N-scale | Mean pooled TVD | q95 | Raw 0.02 false-fail |
|---|---:|---:|---:|---:|---:|
| v2a | 0 | 1 | 0.03394 | 0.04146 | 1.00 |
| v2a | 0 | 4 | 0.01653 | 0.01857 | 0.00 |
| v2a | 1 | 1 | 0.10990 | 0.12824 | 1.00 |
| v2a | 1 | 4 | 0.05634 | 0.06452 | 1.00 |
| v2b | 0 | 1 | 0.05555 | 0.06314 | 1.00 |
| v2b | 0 | 4 | 0.02726 | 0.03042 | 1.00 |
| v2b | 1 | 1 | 0.05502 | 0.06298 | 1.00 |
| v2b | 1 | 4 | 0.02775 | 0.03166 | 1.00 |

Log–log TVD-versus-N slopes are −0.48 to −0.52, matching sampling error
decaying as effective sample size to the power −1/2. For v2a kappa 1, the
fraud/non-fraud repeat-rate difference is 0.886 and mean-run difference is
about 6.37, explaining its much smaller effective number of receiver draws.
No category-specific population direction persists across seeds. v2b kappa 0
and 1 have nearly identical TVD curves.

The current raw threshold would need roughly 3N for v2a kappa 0, 8N for v2b,
and about 30N for v2a kappa 1 based on mean-square scaling.

## H1–H4 decisions

| Hypothesis | Evidence | Counter-evidence | Decision |
|---|---|---|---|
| H1: long runs/effective N | N^-1/2 decay; v2a kappa 1 run/repeat contrast; no fixed category direction | v2b also has high TVD despite equal label dynamics, because only about 425 fraud entities exist | Strongly supported |
| H2: transition/equilibrium bug | None: receiver-only path matches production exactly; v2b kappa 0/1 curves match; repeat means match construction | A persistent bias would not decay at N^-1/2 | Rejected by current evidence |
| H3: length/mask/seed difference | Test fraud mean length is 24.43 versus 23.99, but streams are identical across scenario/kappa | Entity-balanced TVD is not materially reduced; train length means differ by only 0.03; N scaling explains TVD | Not supported as primary cause |
| H4: raw TVD inappropriate for clustered sequences | 100% false-fail at current N; entity classifier AUROCs remain about 0.5; null TVD changes strongly with persistence/effective N | Naive entity-balanced plug-in TVD remains biased and is not itself a fix | Supported |

The v2a permutation excess is not evidence of marginal leakage: entity-label
permutation is not exchangeable when label intentionally changes receiver
persistence. The nonlinear plug-in TVD has different finite-sample bias under
the two dependence structures.

## Gate alternatives

The comparison uses a preregistered 5-percentage-point category-0 fraud-row
leakage stress test. The 4N candidate was calibrated over 30 seeds per
scenario/kappa cell using simultaneous Bonferroni intervals for signed
entity-cluster category frequencies. Permutation p-values alone are never
accepted as equivalence.

| Alternative | False-fail evidence | Leakage power | Required N | CPU/storage | Interpretation and trade-off |
|---|---|---|---:|---|---|
| A. Keep raw TVD <=0.02 | 100% at current N; still 100% for v2a k1 and v2b at 4N | High for large leakage but unusable specificity | up to ~30N | roughly 1.5 CPU-hours generation and 2.8 GiB at 30N | Simple but ignores clustering and is prohibitively conservative |
| B. Entity-balanced TVD + cluster CI | 0/30 false failures in each of four 4N cells | 30/30 detections in each cell | 4N | 1,203 s for 120 receiver-only replicates; about 0.37 GiB for full data | Statistically interpretable signed per-category equivalence; familywise CI is conservative |
| C. Permutation-calibrated excess TVD | Near zero for most v2b cells, but falsely flags v2a because labels are not exchangeable under label-dependent persistence | High under exchangeability only | current N | 22.48 s, negligible storage | Invalid as the sole v2a gate; retained as descriptive null calibration |
| D. AUROC equivalence + cluster-aware signed receiver diagnostic | Signed-frequency component: 0/30 false failures per cell; existing v2.0 AUROC points 0.4909–0.5105 | Signed-frequency component: 30/30 detections per cell; 4N AUROC bootstrap still required before adoption | 4N candidate | receiver calibration 1,203 s plus classifier cost; about 0.37 GiB full data | Best alignment with “no usable row signal,” while retaining interpretable category leakage checks |

## Selected amendment

Alternative D, with B's simultaneous signed per-category entity-cluster
confidence interval, is the statistically preferred v2.1 candidate. Raw TVD
remains reported but is no longer a hard equivalence statistic. The candidate
uses 4N to improve power and also address the v2b 8-bin low-count failures.

This is a benchmark-calibration amendment, not a result-driven change:

- schema/config are explicitly versioned `benchmark_v2.1-candidate`;
- v2.0 artifacts are preserved;
- no DGP rho, K, length, prevalence, seed, kappa, or primary endpoint changes;
- the amendment was selected without learned-model results.

The signed-frequency calibration passes its preregistered thresholds in all
four cells: false-fail rate 0.00 (required <=0.05) and leakage detection power
1.00 (required >=0.90). The largest clean simultaneous bound in the raw
replicates remains well below the +/-0.02 practical margin, while contaminated
category-0 bounds exceed it.

This does not promote v2.1 to an accepted benchmark. The 4N AUROC
entity-bootstrap and a full-data Gate A–E rerun have not been executed. More
importantly, v2.0 Gate B shows a substantive i.i.d.-versus-C1 improvement
(v2b kappa 1, 8-bin upper bound 0.02364 versus margin 0.00787); increasing N
would narrow uncertainty around that failure, not repair it. A 4N full-data
rerun is therefore not an appropriate next CPU action until Gate B's
preregistered scientific expectation is reviewed without learned-model
results. The candidate remains fail-closed.

## Artifacts

See `artifacts/benchmark_v2/receiver_diagnosis/` for counts, position/segment
TVDs, permutation nulls, run statistics, all 360 Monte Carlo replicates,
summaries, effective-sample-size estimates, and `tvd_vs_n.png`. Candidate
calibration outputs are under
`artifacts/benchmark_v2/receiver_diagnosis/v2_1_candidate/`.
