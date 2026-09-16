# CS-SAF v3: explicit history correction and centered residual

Registered 2026-09-16 before implementation/training. This is a bounded exploratory
pi=.05 pilot: eight fresh GPU fits, four candidates x kappa 0/1. It follows the
[mixed fixed-checkpoint result](route_decomposition_v1_report_2026_09_16.md), not a
successful v2 model. [Immutable contract](../../configs/benchmark_v2/cs_saf_revision_v3.yaml).
Old failures, thresholds, splits and checkpoints remain unchanged.

Execution annotation: registered at `9813573`, final CPU/GPU source `44a2bd3`.
Eight GPU fits completed; [result](v3_pilot_v1_report_2026_09_16.md).
Primary R response gate PASS, accuracy screen vs historical U FAIL; no promotion.
The original YAML and criteria below remain unchanged.

## Architecture and comparisons

Keep v2's 128-dimensional strict-past GRU, observed static embedding, support,
fresh/value heads, and two rank-16 route banks. Let `u_s(c)=tanh(Wc_s*c+bc_s)`.
Add **32 coefficients**, alpha[2,16], initially zero, giving the nonlinear
history-only correction `h_s(c)=dot(alpha_s,u_s(c))/sqrt(16)`. Define the old raw
route r and its centered residual `delta=r-E_(train gap|s)[r]`. Reference bin
frequencies are fitted to the full train view before CPU subsampling, and fixed.

| Candidate | Copy logit | Objective | Comparison purpose |
|---|---|---|---|
| historical U | b+r | base | Existing saved ordinary-loss baseline, no new fit |
| CS3-H1 | b+h | base | Nonlinear history-only, no current-gap pathway |
| CS3-E1 | b+h+r | base | E−U: effect of explicit history coefficients |
| CS3-C1 | b+h+delta | base | C−E: centering/parameterization with same capacity |
| CS3-R1 | b+h+delta | base + residual penalty | R−C: regularization at identical architecture |

All new candidates have 133,581 stored parameters; E/C/R have the same active
parameter count. H has 1,056 dormant gap-bank parameters, explicitly excluded
from its active capacity: nominal parameter matching does not make it equally
expressive. H is a negative control, not a proposed successful active model.
The shared u/GRU still allow indirect feature interactions. The history and
residual **coefficients** are separate; their feature extractor is not independent.

For fixed route features, E and C map via `alpha_C=alpha_E+w*E_pi[tanh(Wg*E)]`.
Thus centering alone is not additional expressivity or a new theorem. Fresh
initial states are identical (zero alpha and w), but optimized trajectories may
differ. No checkpoint is used to initialize training. A test must establish this
function-preserving map at nonzero weights, and proper penalty gradients.

## One fixed regularizer, no search

Only R uses `lambda=.01` times the global train-transition mean of
`sqrt(E_pi[delta(c,k)^2]+epsilon^2)-epsilon`, with `epsilon=1e-4`.
This smooth RMS penalizes the function's gap-varying part, not the retained
history term. Its approximately linear large-residual growth avoids a quadratic
penalty increasing quadratically on a useful large active response. This is a
known norm regularization choice, not a null-safety theorem or novelty claim.
Lambda is fixed now as a weak logit-scale penalty (unit RMS costs about .01
objective units per transition), not selected by a validation sweep. This one
choice can fail and does not establish the best possible regularization.

The same rule is applied to both observed groups; no active/null label roles,
oracle targets, balancing, oversampling or auxiliary BCE enter training.
The minibatch estimator uses fixed full-train transition counts, with entity
sampling correction, excluding first/padding events. Direct penalty gradients
must not enter alpha/copy_base. Shared feature updates can still change history
predictions indirectly. All optimizers use inherited weight decay uniformly.

## Training and verification

Inherit seed 20260930, fresh-v2 common tensors/bank seed, AdamW lr .001, batch 512,
50 epochs maximum, patience 5, clipping 1, weight decay 1e-5. Select by unchanged
global validation base NLL, never by penalty or the post-training oracle.
Preserve best and zero-based epoch-9 states. All eight runs finish to make the
ablation interpretable; failure authorizes no fallback/lambda/seed/prevalence sweep.

Before GPU: algebraic equivalence, weighted centering, factual/grid likelihood,
strict-past invariance, context permutation, support/validity, nonfirst/fixed
penalty denominators, penalty gradient isolation, checkpoint reload, parameter
counts, paired initialization/order, and two deterministic CPU fits per candidate.
Source must be committed and CPU PASS must match the GPU source. Only currently
available GPUs are used, with output and errors persisted on the workstation.

## Conditional accuracy, beyond response amplitude

Run the existing observable-information semi-Markov oracle **only in evaluation**.
It filters strictly past observed gaps/repeats; it reads no realized DGP state or
held-out content. Oracle configuration/hash and saved bin boundaries are fixed.
It defines `p*(mark)=q*one_hot(previous)+(1-q*)/64`; learned fresh probabilities
need not be uniform, so evaluating just q is insufficient.

Primary error per observed history is full 64-mark TV, integrated over all saved
bins using the context-marginal train bin frequencies. This measure is not
p(gap|history), a causal intervention distribution, or free-running synthesis.
Report factual-bin TV, grid repeat absolute error, factual repeat BCE, residual
RMS, all context cells, train/validation, best/epoch 9. Aggregate within entity,
then within context. Apply the identical auditor to saved historical U snapshots;
there are zero new U fits. Entity SE is descriptive, not seed uncertainty.

The original copy/repeat active >=.05, all three null <=.05, selectivity >=2,
zero-gap invariance and generated-support/validity criteria remain mandatory.
Additionally screen best-validation TV: R−C and R−U must each be negative in the
unweighted mean of the three null cells and nonpositive in the active cell.
This is a conservative exploratory direction check with zero allowed worsening,
not a calibrated significance/noninferiority claim. It prevents declaring a
smaller but less accurate active response a method success. Fixed-epoch and all
individual null outcomes remain visible even if this aggregate screen passes.

Even response+accuracy PASS means eligibility for a **new registered** broader
pilot, not superiority, five-seed confirmation, real-data success or publication.
No later stage, held-out evaluation, external baseline fit or automatic promotion
is authorized by this contract. Code, failures and results will be committed and
pushed to DataGeneration/research/cs-saf; raw checkpoints remain on the workstation.
