# Fixed-checkpoint route decomposition — preregistration

Registered 2026-09-16 before implementation/execution, after the six-fit
[loss diagnostic](loss_control_v1_report_2026_09_16.md). The [immutable YAML](../../configs/benchmark_v2/cs_saf_route_decomposition_v1.yaml)
pins its parent evidence and all 12 saved best/epoch-9 snapshots. This is an
exploratory analysis of reused train/validation data, with **zero new fits,
optimizer steps, generated samples or held-out test access**.

## Precisely defined question

Whole-route zeroing worsened factual repeat BCE even where the DGP has no
current-gap predictive effect. Does the trained route contain a useful
gap-independent correction plus an unnecessary gap-varying residual?
This is not yet a causal explanation of training or a demonstrated new model.

For observed context s, history/static feature c and saved support bin k:

```text
b(c) = copy_base(c)
r_s(c,k) = dot(w_s * tanh(Wc_s*c+bc_s), tanh(Wg_s*E_gap(k))) / sqrt(16)
pi_s(k) = count_train_nonfirst_transitions(s,k) / count_train_nonfirst_transitions(s)
mu_s(c) = sum_k pi_s(k) * r_s(c,k)
delta_s(c,k) = r_s(c,k) - mu_s(c)
```

The count measure is fitted once to each saved train view, using the exact
float32 bin boundary convention, and reused unchanged for validation and every
checkpoint/candidate. It is a **context-marginal** distribution, not p(gap|history).
It defines a one-way centering; it is not a full functional-ANOVA purification
or the uniquely correct population decomposition. Center **logits**, not
probabilities: sigmoid(E[logit]) is generally not E[sigmoid(logit)].

Four fixed-weight arms, with the same history features and fresh-mark head:

| Arm | Copy logit | Purpose |
|---|---|---|
| F | b + mu + delta | Original full model, reconstructed exactly |
| M | b + mu | Retain train-reference mean, remove gap variation |
| R | b + delta | Remove the mean, retain centered variation |
| Z | b | Existing whole-route removal |

For each arm, q=sigmoid(logit), p_repeat=q+(1-q)*p_fresh(previous).
The mark distribution stays q*one_hot(previous)+(1-q)*p_fresh. Current mark
is a likelihood target only; histories remain strictly past. Active-label
knowledge does not enter any arm or reference measure. Both observed groups
receive the identical transformation.

M/Z have zero current-gap response **by construction**, including in active
cells. This is a mechanical check, never a learned null-safety success. Keeping
F in known-active cells and M in known-null cells would use oracle knowledge;
it is prohibited as a claimed deployable solution.

## Measurements and frozen interpretations

Analyze all train/validation nonfirst histories, pi=.05, both kappas, all U/A/B,
best and fixed epoch 9: 12 checkpoint records. Save entity identities and arrays.
For F/M/R/Z report factual repeat BCE, factual full-mark NLL, full-bin copy and
repeat range; report mu, |mu| and train-weighted RMS(delta) on the logit scale.
Aggregate histories within entity and then entities within context.

Primary paired contrasts: M-F (remove residual), M-Z (retain mean), R-F
(remove mean), F-Z (historical whole-route comparison). At fixed fresh heads,
pairwise full-mark NLL differences must equal repeat-BCE differences; check
this numerically instead of conflating absolute mark and repeat losses.
Report mean, median, p90, entity standard error. Entity variation does not
represent seed uncertainty. No p-value, adaptive shrinkage coefficient or
new success threshold is chosen from the result.

Primary explanation concerns **U, kappa=0, rare context**. Directional support
requires the stated sign in both train and validation, at best and epoch 9:

1. M-Z BCE <0: the retained reference-mean component is useful.
2. M-F BCE <0: removing residual improves the primary null cell.
3. In U's active kappa=1 rare context, M-F BCE >0: the residual is useful there.

These are separately reported directional observations, not confirmatory
tests. A/B and all other cells are mandatory controls, not alternative primary
selections. Mixed signs or uncertainty must be retained. Even all three signs
do not prove that a retrained centered/regularized model will succeed.

## Verification, preprocessing audit and execution

Check independent original forward/grid agreement, exact reconstruction and
weighted-zero residual, nonlinear link placement, mark/repeat difference
identity, zero M/Z ranges, train-only reference frequencies, entity alignment,
and unchanged checkpoint hashes. Tolerances are pinned in YAML. Verify caches
against the original prepared manifest and checkpoints against the previous
result, including source and model-state identity. Confirm train/validation
entity disjointness, first-gap missingness, reserved-code exclusions, valid
lengths, support representative/bin round trips and train-fitted tensorizers.

Commit the tested implementation before analysis. Preserve all old artifacts;
use `artifacts/cs_saf/route_decomposition_v1/`. This protocol authorizes no model
retraining, seed/rank/penalty sweep, baseline fit or expansion to later pi.
Literature/theory/code audit is recorded separately from empirical diagnostics.

## Relation to existing theory

Main/interaction decomposition and its dependence on a chosen measure have
prior art: [Lengerich et al., AISTATS 2020](https://proceedings.mlr.press/v108/lengerich20a.html).
Our one-way checkpoint analysis does not reproduce their full purification
algorithm and is not claimed as a new mathematical decomposition. Its value
must come from what it reveals about this generator and from a later, tested
method for retaining active conditional dependence while avoiding null response.
