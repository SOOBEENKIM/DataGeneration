# CS-SAF v4: raw E prediction with centered residual regularization

Registered 2026-09-16 before implementation or training. This is the missing
cell of the exploratory 2 x 2 forward-parameterization / penalty comparison,
following v3's response PASS but active accuracy FAIL against U. Two fresh fits
only: pi=.05, kappa 0/1, seed 20260930. This is adaptive exploratory development
on reused validation data, not independent confirmation.

[Immutable contract](../../configs/benchmark_v2/cs_saf_revision_v4.yaml).
[Parent evidence](v3_pilot_v1_result.json). Original failures and contracts remain.

Execution annotation: registered `642b878`; CPU/GPU source `6356649`. Both fresh
fits completed; [result](v4_pilot_v1_report_2026_09_16.md). Response and registered
best-validation accuracy screens PASS. Fixed epoch 9 active TV is worse than E;
no robust E accuracy superiority, later-stage run or final method-success claim.
The original YAML and criteria below remain unchanged.

## Exact intervention

Let h_s(c)=alpha_s dot u_s(c)/4, r_s(c,k)=(w_s*u_s(c)) dot v_s(k)/4,
and delta_s(c,k)=r_s(c,k)-sum_j pi_s(j)r_s(c,j), with the same train-fitted pi.
CS4-ER1 predicts copy logits **b(c)+h_s(c)+r_s(c,k)**, exactly as E.
Only its penalty uses delta: L=L_base+.01*mean_transition[
sqrt(sum_k pi_s(k)delta_s(c,k)^2+1e-8)-1e-4].

| Forward | No added penalty | Same .01 centered-residual penalty |
|---|---|---|
| Raw b+h+r | saved E | new ER |
| Centered b+h+delta | saved C | saved R |

All four models have the same 133,581 parameters and initial tensors. There are
no new parameters, optimizer changes, auxiliary BCE, active-label masks, oracle
training targets, warm starts or lambda search. Shared features mean the penalty
can indirectly affect history predictions despite zero direct gradients on
alpha and copy_base. This experiment does not prove that history is unaffected.
E/C have the same function class; their training parameterizations differ.

Reuse existing E/C/R and historical U best/epoch-9 audits only after verifying
parent evidence, artifact and checkpoint hashes, exact data, initialization,
training order prefix, reference probabilities and sampling plan. The auditor
and parent model sources are hash-pinned and remain unchanged. New source is
committed after tests and before deterministic CPU and GPU execution.

## Comparisons and decisions fixed before new outcomes

Primary ER-E isolates adding the penalty to E. ER-U checks the ordinary-loss
baseline. Mandatory ER-R checks forward parameterization under regularization;
C-E and R-C complete the table. Report the paired entity contrast
(ER-E)-(R-C), with descriptive entity SE, for every metric/split/checkpoint/group.
It is not a seed-level interaction estimate or a calibrated significance test.

The original response criteria remain: copy/repeat active >=.05, every null
<=.05, selectivity >=2, zero-gap <=1e-8, valid generated support/marks/values.
Additionally, at best validation ER-E and ER-U must each have strictly negative
mean TV change across the three equally weighted null cells and nonpositive
TV change in the active cell. Use the identical full-mark conditional TV auditor,
all observed histories and train context-marginal gap-bin weighting. Report
individual null cells, factual TV/BCE, train and fixed epoch 9 even on failure.
This directional screen is exploratory, not a noninferiority test. No thresholds
are relaxed after results. All two fits finish even on scientific failure.

Success means eligibility to design a separately registered broader study.
It does not establish multi-seed, external-baseline, held-out, real-data,
free-running behavioral-fidelity, novelty or publication success. Failure means
no automatic next architecture or lambda sweep: reevaluate this bounded
hypothesis family using the complete factorial evidence. It does not prove every
possible model impossible. Preserve and push all results, including failure.
