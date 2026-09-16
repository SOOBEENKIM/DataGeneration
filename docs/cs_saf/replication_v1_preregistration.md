# Fixed U/E/ER: five-trial and prevalence replication v1

Registered 2026-09-16 before new implementation or outcomes. The user authorized
sequential execution after the v4 exploratory PASS. [Immutable contract](../../configs/benchmark_v2/cs_saf_replication_v1.yaml).
All prior failures and discovery seed 20260930 remain separate.

Execution annotation: registered `8f5eaef`; CPU/GPU source `e35c6c2`.
All 30 stage-1 fits completed. ER response PASS 5/5 and null TV vs E improves 5/5,
but ER-E mean active TV +.000047795 fails the exact registered directional gate.
The conditional 90-fit expansion was not started. [Result](replication_v1_report_2026_09_16.md).
Original YAML and decision rules below remain unchanged.

## Fixed methods and paired randomness

U is ordinary-loss v2 (133,549 parameters); E is raw b+h+r with base loss;
ER is exactly E forward with the already fixed .01 smooth centered-residual RMS
penalty (E/ER each 133,581 parameters, epsilon 1e-4). No architecture, loss,
coefficient, optimizer, budget or checkpoint-selection search is performed.

Five new paired trials use model/order seeds 20261001..20261005, route-bank seeds
20261101..20261105, and generation seeds 20261201..20261205. Original v2 used a
separate constant bank seed 20261010: changing only the global seed would not vary
this part. This study explicitly reinitializes the same bank tensors with their
exact original uniform fan-in distributions and the registered bank seed.
This is randomness replication, not a new model. Route interaction/history
coefficients start at zero. Common tensors match across U/E/ER; E and ER have
entirely identical initial states within each cell. All weights are fresh;
paired entity orders and generation plans are checked.

Use immutable pi=.05/.10/.25/.50, kappa=0/1 prepared train/validation views.
The same synthetic entities are reused; proportions are not independent datasets,
and changing prevalence changes minority counts. This study cannot separate
relative prevalence from absolute sample count or prove new-DGP generalization.
No test, real-data, external-baseline or realized-latent training targets are used.

Only U's zero-gap audit evaluates its constant sigmoid once then broadcasts, as
already required for v3's numerical control: expanded identical logits can show
an ulp across CPU vector/tail kernels. This does not alter U training, ordinary
forward, generation or learned states. The 1e-8 zero-gap threshold is unchanged.

## Budget and stage decision

Stage 1 finishes all 30 fits: three candidates x two kappas x five new trials,
pi=.05. Do not inspect a partial seed set and decide to add/drop seeds.
Advance only if BOTH conditions hold:

1. ER passes the original copy/repeat active >=.05, each null <=.05,
   selectivity >=2, zero-gap <=1e-8 and valid-generation gates in all five trials.
2. For ER-E and ER-U separately, best-validation TV differences averaged over
   the five paired trials have a strictly negative equal-three-null mean and a
   nonpositive active-cell mean. No accuracy-worsening allowance is added.

Report every trial's accuracy direction and combined PASS count, not just means.
This is an exploration/expansion rule, not a significance/noninferiority test.
Five trials do not establish statistical power for tiny effects. On stage-1
failure, do not run the other 90 fits; report the bounded replication failure
without changing seeds, penalty, thresholds or model to rescue it.

If stage 1 passes, finish all 90 additional fits at pi=.10/.25/.50. Do not truncate
the remaining prevalence curve on a scientific failure. Apply the same criterion
per prevalence and report every individual cell. All-four-prevalence PASS means
robustness within this existing synthetic dataset, not external superiority,
independent confirmation or publication.

## Evaluation and uncertainty

Use the unchanged v3 oracle/full-64-mark TV auditor, full train and validation
observed histories, training-context marginal gap-bin weights, and entity-first
aggregation. Also report factual TV, repeat BCE/error, responses and residual RMS.
Save best and epoch 9 if actually reached. If inherited early stopping ends before
epoch 9, report missing rather than extend training or substitute a checkpoint.
Best validation NLL remains the sole checkpoint-selection rule. Fixed epoch 9 is
a mandatory secondary analysis: do not hide a reversal or select the favorable
checkpoint afterward. This remains adaptive research on reused validation data.

For ER-E, ER-U and E-U report all five paired seed effects, mean, sample SD, SE,
and descriptive two-sided 95% Student-t intervals (df=4, critical 2.7764451051977987).
State small-n/normality assumptions and that intervals are not multiplicity adjusted
or independent-data uncertainty. Entity SE is separate; do not pool entities from
five trials as independent training replicates. Report snapshot coverage; missing
epoch-9 pairs are never treated as zero or imputed. With incomplete snapshot
coverage report available values/mean/SD/SE but no five-trial interval.

## Verification and execution

Before training: frozen-function/objective/gradient equivalence, seed/bank variation,
common initialization, selection/aggregation/boundary tests. Then two identical
tiny CPU fits per candidate using trial 0 at pi=.05/kappa=1, with reload, loss-drop,
support/validity and zero-gap checks. Commit tested source before CPU/GPU; both
must match source/config hashes. Check index/hash/oracle for all eight caches.

At most four simultaneous fits, one per free GPU; check no compute process before
every launch and preserve device/PID/log evidence. Never stop another job.
Technical failures preserve artifacts and stop new dispatch while in-flight jobs
finish. Distinguish technical failures from scientific FAIL. Each fit saves
checkpoints, training history, sample, response/accuracy audits, entity arrays and
checksums. Save and push the report regardless of result.
