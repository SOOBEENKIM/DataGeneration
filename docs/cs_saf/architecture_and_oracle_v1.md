# CS-SAF architecture and oracle contract v1

Frozen before the first CS-SAF oracle run, 2026-09-16. This implements the
design/oracle stage of `research_protocol_v1.md`. It does not claim a trained
CS-SAF model, a prevalence experiment, or a successful dependency route.
Machine-readable contracts are `cs_saf_architecture_v1.yaml` and
`cs_saf_oracle_audit_v1.yaml` under `configs/benchmark_v2/`.

## Scope and inherited evidence

Use the repaired static codec and retain all previous failed SAF experiments.
Do not restart v6's stopped five-seed run. The new candidate family has a new
namespace. Original data, checkpoints, and reports remain immutable.

The oracle stage reads only the **existing canonical train content** at the
original approximately 5% context prevalence. It verifies identifiability of
the conditional rule before a new prevalence grid or learned model is built.
Reweighting conditional oracle summaries at pi=0.05/0.10/0.25/0.50 is an
analytic diagnostic, not four new datasets and not evidence of learned
mechanism dilution. The conditional law does not depend on pi; the later
training experiment must measure the effect of minority sample availability.

## Two distinctions required by the actual production DGP

The inherited DGP has a semi-Markov gap regime Zg and an independent receiver
regime Zr. At kappa=1 and label=1, the latent copy coin uses Zg; otherwise it
uses Zr. The current gap is an exponential emission from Zg.

Thus current gap and copying share a hidden parent. Altering the model's gap
input measures its **conditional predictive response**, not the causal effect
of do(gap). A physical intervention on gap alone would not alter the copy coin
in this DGP. This terminology is fixed before new model evaluation.

Also, a new mark drawn uniformly from K categories may equal the previous
mark. If q is the latent copy probability, observable repeat probability is

    p_repeat = q + (1-q)/K                 [DGP oracle]
    p_repeat = q + (1-q)*p_new(previous)   [learned mixture]

An observed equality is not a latent copy label. Training must not use realized
oracle regimes/coins, nor train q directly against observed equality as if it
were the true coin. Mixture components can be nonidentifiable, so q sensitivity
alone is insufficient evidence: observable repeat response and full mark
transition distributions remain necessary endpoints.

## Exact candidate architecture

The chronological factorization and numerical value decoder are inherited:

    p(gap_t | history, static)
    p(mark_t | history, static, gap_t)
    p(value_t | history, static, gap_t, mark_t)

History is a one-layer shifted GRU, width 128, using at most 32 past events
(the controlled sequences have at most 32 events). Current observations never
enter their own history. The repaired static label embedding has width 8 and
conditions both GRU initialization and the direct mark context.

For c=concat(history_128, static_8), gap embedding e of width 32:

    u = tanh(Wc*c + bc)                    [32 components]
    v = tanh(Wg*e)                        [32 components, no bias]
    copy_logit = wb*c + bb + dot(w, u*v)/sqrt(32)

Initialize w=0 and other projections normally. This starts from the no-gap
route without killing the initial gradient of w. The fresh-mark distribution
is softmax(Wnew*c+bnew), independent of current gap; PAD/UNK/MISSING outputs
are masked in the controlled closed-vocabulary task. The full mark distribution
is q*one_hot(previous)+(1-q)*p_new, with p_new alone for the first event.

Every routed/unrouted model contains all the same tensors. Controls replace e
by an exact zero vector before the bias-free Wg. The same state dictionary can
be used to inspect the zero-gap control. No learned scalar gate is used.

The primary gap head is unordered categorical over at most 31 observed
positive training representatives. An exact zero atom is used only if train
contains it. The C0 comparator retains the inherited continuous gap head.
Ordered hazard is excluded from this first CS-SAF candidate family.

**Support qualification:** the production DGP uses continuous exponential
gaps, not truly discrete physical time. Its empirical training-atom restriction
is a quantization/inductive-bias choice. Zero empirical-support violations do
not prove exact recovery of the population distribution or an irreducible
error lower bound for continuous models. The earlier C0/U0 contrast also
changes the gap distribution family; only the route/objective contrasts below
have exactly identical parameter budgets.

## Objective and matched ablations

Let r_t = 1[mark_t=mark_(t-1)] at valid nonfirst events. Use

    L = L_base + lambda * (1/2) * sum_y mean_(transitions in y) BCE(p_repeat,r)

L_base is the inherited gap+mark+value NLL with its existing valid-event
reductions. lambda=1 for the balanced candidates and 0 otherwise. Context
weights are computed once from train transition counts, not validation or
each batch. A uniformly sampled transition estimator uses 1/(2*pi_transition_y)
without self-normalizing observed batch labels. A concrete window-batch runner
must implement the corresponding fixed global denominator; unequal window
length must not silently change the intended transition objective.

| Candidate | Gap support | Current-gap route | Balanced repeat auxiliary |
|---|---|---|---|
| CS-C0 | continuous control | off | off |
| CS-U0 | train-aligned | off | off |
| CS-U1 | train-aligned | bilinear | off |
| CS-B0 | train-aligned | off | on |
| CS-B1 | train-aligned | bilinear | on |

U0/U1/B0/B1 must have identical keys, tensor shapes and parameter counts.
C0 differs only in the gap decoder family; report this parameter-count
exception instead of manufacturing unused parameters to claim equivalence.

Use the same data, sampling plan, minibatch order, seeds and maximum training
budget. AdamW lr=0.001, weight decay=1e-5, batch=512, gradient clip=1,
50 epochs maximum, patience=5. Select checkpoints using the common validation
base NLL, excluding auxiliary loss. Report realized epochs/updates; maximum
budget matching does not imply identical early-stopped compute. No search is
included. Seeds remain the original protocol's pilot 20260930 and confirmation
20261001--20261005.

## Oracle mathematics and information boundary

Track a normalized posterior over (regime, residual duration), using the
production duration PMFs and equilibrium residual initialization. Residuals
decrease by one; at one remaining event, switch regime and draw a new duration.
This preserves the full semi-Markov memory, rather than replacing it with a
two-state Markov approximation or exposing the realized latent state.

For the active cell, infer the state from strictly past continuous gaps and
past observed repeats. For null cells, track the receiver regime from past
repeats alone. No current mark, future event, realized regime, amount, absolute
time origin, or first missing gap is used.

The current model gap route sees a train-fitted **bin**, not exact continuous
gap or latent state. For bin [a,b] and scale s_z, integrate

    P(bin | z) = exp(-a/s_z) - exp(-b/s_z).

Update the strict-past prior with this mass, then average q_low/q_high over
the posterior. Using the density at a representative would answer a different
question and could falsely suggest the model sees more information than it
does. After recording the prediction, update the history filter using the
actual continuous gap and observed repeat, matching SAF's past inputs.

The response range covers all 31 fitted bins at the same history. Average
histories within entity, then average entities within context. For empirical
calibration compare predicted observable repeat with actual equality; report
the mean entity residual and entity-cluster standard error. Transitions are
not treated as independent samples.

## Frozen oracle checks

- Active mean latent-copy response range >=0.05.
- Every null mean range <=0.05, active response >=2 times the largest null.
- No-current-gap oracle maximum range <=1e-8.
- At least 100 train entities per context.
- Each cell's absolute observable-repeat calibration residual <=
  max(0.02, 5 entity-cluster SE).
- Paired original split assignments and train static records match across kappa.
- All fitted support representatives are observed train atoms; static codec
  distinguishes the two labels with no reserved code.

For the later model pilot, retain all original copy-response criteria and
also check observable-repeat response; a latent mixture reparameterization
alone cannot count as success. These oracle checks do not test a learned
zero-gap model or synthetic gap violations, and do not unlock confirmation.

## Execution, persistence and remaining gates

The runner refuses an uncommitted source tree and an existing output directory.
It records source commit, source/config hashes, runtime versions, source
manifests, filtered train-content hashes and the original split hash. Full
event files are not rehashed because they contain sealed rows. Parquet row
predicates exclude validation/test bodies; split membership is structural
metadata. The runner never opens `oracle_latents.parquet`.

Before a model pilot: implement the candidate, validate gradients/capacity,
materialize the pi grid preserving original entity split membership (no
label-dependent resplitting), freeze data-generation streams, and pass CPU
end-to-end/determinism checks. Preserve the original pi=0.05 data; additional
prevalences must use new namespaces and retain the same DGP law. The exact
new materializer is not implemented or silently authorized by an oracle pass.

Before five-seed confirmation: freeze intervention-error and context-specific
transition-TV aggregation, and resolve the failed legacy noninferiority
calibration using train-only evidence. Old failed margins remain reported;
do not adjust them retrospectively using candidate outcomes. No real-data
comparison or held-out access is part of this design/oracle stage.

Persist both successful and failed outcomes. Commit execution source/config
before running; commit compact reports afterwards. Large data/checkpoints stay
on the remote workstation and need a separate backup strategy. Git is the
source/protocol/report history, not a full runtime-artifact backup.
