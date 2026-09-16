# CS-SAF v2 three-objective diagnostic — preregistration

Registered 2026-09-16, before this diagnostic's implementation or fits, after
the [observed v2 failure](v2_pilot_v1_report_2026_09_16.md). This is exploratory
reuse of development data, not an independent confirmation or a resumed v2 pilot.
The immutable [contract](../../configs/benchmark_v2/cs_saf_loss_control_v1.yaml)
freezes the following scope. Preserve all prior failures and artifacts.

Execution update: registration commit `d4069c9`; source `f4bed4b` passed 82
tests and all three CPU gates. Six GPU fits are complete; all three objectives
fail the rare-null response gate. See the [result](loss_control_v1_report_2026_09_16.md).
The scope below and the original YAML remain unchanged.

## Question and objectives

At pi=.05, U1 and B1 both have unwanted response in the kappa=0 rare context.
The U1/B1 contrast mixes added auxiliary weight with its allocation by context.
Use identical v2 architecture, seed, initialization, train-only tensorizer,
entity order, optimizer, budget, and checkpoint-selection rule for:

```text
U (CS2-U1): L_base
A (CS2-A1): L_base + sum_i repeat_BCE_i / T
B (CS2-B1): L_base + (1/2) sum_y sum_{i in y} repeat_BCE_i / T_y
```

Here each i is a nonfirst train transition; N is the train entity count,
T_y its context transition count, T=sum_y T_y, and E the total train event count.
For an entity-uniform minibatch of size b, A's auxiliary is
`(N/T) * mean_batch(sum_transitions_in_entity BCE)`; B retains the existing
`mean_batch(N/(2*T_y) * sum_transitions_in_entity BCE)` exactly. Fixed global
train denominators handle variable lengths and single-context batches.
No batch label normalization, oversampling, loss coefficient sweep, oracle
target, gate mask or architecture change is allowed.

The full-data direct-route gradient coefficients by context are:

```text
U: T_y/E
A: T_y/E + T_y/T
B: T_y/E + 1/2
sum over contexts: U=T/E; A=B=T/E+1
```

The full mark likelihood's route-dependent derivative equals the observable
repeat BCE derivative; its nonrepeat fresh-mark component has no direct route
parameters. This statement holds for direct route parameters at fixed shared
features. A/B match the total scalar coefficient, not their gradient norm or
AdamW updates. Base loss retains its historical minibatch valid-event means,
so the global expression does not assert exact finite-minibatch unbiasedness
for the entire base objective. Shared feature effects remain part of the fit.

Primary descriptive contrasts: **A-U** (adding an unbalanced auxiliary),
**B-A** (changing its context allocation at matched total coefficient).
U/B alone cannot separate those changes.

## Frozen execution

Train all three candidates freshly at pi=.05, kappa=0/1: six scientific fits,
seed 20260930, identical initialized tensors/epoch permutation stream. Repeat
U/B under the same source as A and compare their initial/best state identities,
selected epoch, base NLL and response with the saved v2 runs. Initial identity
is mandatory; response/NLL differences above 1e-6 must be investigated before
interpreting contrasts. GPU bitwise reproduction is measured, not assumed.

Keep v1's AdamW .001, decay 1e-5, clip 1, batch 512, FP32, max 50 epochs,
patience 5, no scheduler, selection by global validation base NLL. All six
fits complete regardless of response failure; stop for technical invalidity.
No promotion of another primary, extra seed, fallback or adaptive training.

Before GPU fits, test objective values and gradients against direct formulas,
partition invariance with variable lengths/single-context batches, matched
initialization and old U/B semantics, and diagnostic array alignment. Run the
inherited two-run CPU determinism/loss-decrease/reload/generation gate for each
of U/A/B on this same committed source (six tiny CPU fits).

## Diagnose U's null response separately

At each best checkpoint and fixed zero-based epoch 9, evaluate **all train and
validation nonfirst histories**, all 31 bins. Save epoch 9 without changing
selection or patience; if early stopping precedes it, record absence without
substitution. Fixed-epoch contrasts help distinguish checkpoint-selection
effects from different training trajectories; they are not causal proof.

For each context and entity, average history-level full copy/repeat ranges,
central copy/repeat ranges, factual observable-repeat BCE, and BCE after
zeroing only the current-gap route at the same weights and history features.
Fit central 5th–95th percentile bin intervals from **train context gap counts**
only and reuse them for validation; do not replace the full-support gate.
Report mean, entity standard error, median, p90 and fraction of entity copy
ranges above .05. Preserve aligned entity arrays and A-U/B-A paired contrasts.
Any 95% normal intervals are descriptive entity variation conditional on the
fitted model, not multi-seed uncertainty or confirmatory hypothesis tests.

Train/validation similarity and broad central responses would argue against
an explanation limited to validation-only outliers or extreme gap bins. A
positive `route_BCE - zero_gap_BCE` means that disabling the current-gap route
improves that diagnostic loss for the fixed model. It does not establish that
a retrained zero-route model is better, or explain historical gradient causes.
History information and shared representations remain present in both arms.

## Decisions and continuation

Report the inherited full-support response gates separately for U/A/B at .05:
active >=.05, each null <=.05, active >=2*max null, both copy and repeat;
zero-gap invariance <=1e-8; support/marks/finite values valid. Generate 2,048
entities per best checkpoint using the same train-only parent/length plan.
These per-candidate preflights are not a full prevalence pilot or a proof of
distributional accuracy. Diagnostic completion is distinct from model success.

If a candidate passes this local preflight, the evidence can motivate a
separately registered candidate and multi-prevalence pilot. If all fail, preserve
the result and use the diagnostics to formulate a new structural/regularization
hypothesis, without running it as a fallback. Before a confirmatory study, exact
conditional-distribution error aggregation and train-only noninferiority
calibration still need freezing, then multiple seeds. No real-data or held-out
test access is authorized by this experimental contract.

Runtime namespace: `artifacts/cs_saf/loss_control_v1/`. Commit registration,
then tested implementation before CPU/GPU execution, then compact results.
