# CoF-SeqGen v3 model-level redesign specification

## 1. Status and scope

This is a source-only design specification. It does not authorize or
implement a model, candidate runner, GPU job, sampling job, validation
selection, fresh test, TSTR, privacy analysis, or full experiment.

The official v2.8 conclusion is frozen:

- CTGAN and TVAE have an all-five-guard passing candidate;
- CoF-SeqGen has no passing candidate;
- CoF's remaining failure is
  `receiver_max_abs_signed_frequency = 0.021126555312304892`, above the
  unchanged `0.02` threshold;
- `primary_c2_selection_ready=false`; and
- fresh test, TSTR, privacy analysis, and full/five-seed execution remain
  forbidden.

No additional CoF sampler calibration, decoder calibration, marginal-logit
bias, quantile transport, temperature search, threshold relaxation, or
test-based tuning is permitted. The next admissible intervention is a
model-level discrete-path redesign.

Here, “v3” names a proposed CoF-SeqGen model architecture. It does not alter
the frozen benchmark-v2.5 through benchmark-v2.8 data, metrics, thresholds,
endpoints, or conclusions.

## 2. Frozen evidence and provenance

The following files were audited read-only:

| Evidence | SHA-256 |
|---|---|
| v2.5 full config | `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3` |
| v2.5 frozen-data manifest | `b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05` |
| v2.5 `FINAL_COMPLETE.json` | `e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a` |
| v2.6 selection config | `0884a0144f74f6317ae9e636c0cf5657dedac21ff12cdf545a6b6e5e29be4c4d` |
| frozen CoF 20k checkpoint | `7ce44721cadffef21ade3bf1884421c304f73ea9eff1e9bfb3ca2d9e63cc0c29` |
| v2.6 selection terminal | `3cb9f91a00ea4b3bcb772527e2ff4f49b1a47c645618ce79995847f07337407d` |
| v2.7 selection terminal | `a40630f51e1a07fb509f8f676eabf52f9bd18affbd7a9de47c510e2e1c0792b2` |
| v2.7 CoF empirical-residual evaluation | `bfa153860fa0ceb187454e77f3284fc1302bfadad16decf9d052023a8d62cbad` |
| v2.7 CoF gap-logit evaluation | `a758d8ae9a1faa93c2a8d37eedcf3690d29430da00e1ad17dbcccffa9dd67427` |
| v2.8 CoF gap-distribution evaluation | `45f406b42eff484782cca66423567e94c849f573cc9ef73a926d6d7b051b506d` |
| v2.8 aggregate terminal | `58573dfb93ebed71843f286405e205669140b8cc66811ce61deebe808de513e9` |

The v2.8 report records four CoF guards as PASS and only receiver frequency
as FAIL. Its source checkpoint is still the v2.6 20k checkpoint above. The
v2.8 input inventory remains
`6ae29bd7ab31a1517ceb5ca4745a58b5cd862c197cdb9bad716d71b97d2e4bac`.

## 3. Static audit of the current CoF path

### 3.1 Shared sequence representation

`SeqDenoiser` creates one token per sequence position by summing:

- the projected noisy numerical vector, `num_proj(x_num_t)`;
- a gap-bin embedding, `bin_emb(dt_bin)`; and
- one embedding per categorical channel, including receiver.

The implementation is at `models/seq_denoiser.py:62–69` and
`:118–134`. Learned position, diffusion-time, and entity-label embeddings
are added before a non-causal Transformer encoder
(`models/seq_denoiser.py:124–137`).

The model has four output families:

- `num_head(h)` for amount/numerical clean-value prediction;
- `bin_head(h)` for gap-bin logits;
- independent `cat_heads(h)`, including the receiver logits; and
- `y_head(h)` for the label-consistency logit.

These heads are declared at `models/seq_denoiser.py:87–91` and applied at
`:139–145`.

The shared Transformer therefore permits gap and receiver information to
interact in hidden state `h`, but the output distribution is factorized at
the heads. There is no normalized `p(gap, receiver | h, Y)` in the current
model.

### 3.2 Amount path

The continuous channel alone receives Gaussian VP-cosine forward noise
(`models/cof_seqgen.py:69–82`, `:95–108`). The denoiser predicts clean
`x_num`, and the native objective is masked clean-value MSE
(`models/cof_seqgen.py:163–170`; the weighted v2.6 equivalent is
`models/candidate_components_v2_6.py:153–159`).

Sampling begins from masked Gaussian noise and applies the clean-prediction
DDIM update

`eps_hat = (x_t - alpha_t * amount_hat) / sigma_t`

followed by the next schedule point. The native sampler implements this at
`models/sampler.py:156–199`; the v2.6 candidate sampler has the same path at
`models/candidate_components_v2_6.py:325–385`.

The v2.7 empirical-residual amount candidate fits a class-conditional,
train-only centered quantile map
(`generators/evaluation_only_transforms_v2_7.py:250–270`) and changes only
the amount array after generation, copying gap, receiver, mask, labels, and
lengths unchanged (`:378–424`). This fixed path moved both amount guards to
PASS:

- amount KS `0.003925903617421178`; and
- amount effect `0.004578512140862445`.

The v2.8 CoF parent preserves that amount contract. Its stored amount values
also pass:

- amount KS `0.0033309118429906137`; and
- amount effect `0.005297334999158988`.

These are the validated amount components to retain. v3 must not introduce
a new amount representation, objective, inverse decoder, temperature,
quantile grid, or result-dependent amount adjustment.

### 3.3 Gap path

At training time, gap positions are independently selected for absorbing
MASK corruption (`models/cof_seqgen.py:110–123`). Gap logits use a separate
cross-entropy target (`:166–170`). In v2.6 the same behavior remains, with
its own random gap mask and weighted gap CE
(`models/candidate_components_v2_6.py:91–111`, `:155–159`,
`:202–208`).

At sampling time, gap feedback is chosen from `gap_logits` independently of
receiver. Native sampling uses either gap argmax or a separate multinomial
draw (`models/sampler.py:201–220`), and final output is gap argmax
(`:222–226`). The v2.6 path separately samples gap feedback and separately
decodes final gap logits
(`models/candidate_components_v2_6.py:386–427`).

v2.7 tried a train-only class-conditional marginal gap-logit offset. The
hook changes only gap logits
(`generators/evaluation_only_transforms_v2_7.py:562–584`). It improved gap
KS from `0.02244017136797355` to `0.009416293536285036`, but still failed
the KS threshold and worsened gap effect from `0.25894890573252216` to
`0.3289517776514092`.

v2.8 then replaced generated gaps after sampling using a train-fitted
class-conditional marginal transport. The intervention assigns only
`dt_bin`, while the receiver array is copied unchanged
(`generators/evaluation_only_transforms_v2_8.py:248–295`). This made both
gap guards pass:

- gap KS `0.001900823279718744`; and
- gap effect `0.0067670883118888985`.

That result does not validate the native gap head. It validates a
post-sample marginal gap replacement, which is now frozen as evidence and
must not be extended with another calibrator.

### 3.4 Receiver path

Receiver is embedded through an independent categorical embedding and
decoded through an independent categorical linear head
(`models/seq_denoiser.py:65–66`, `:90`, `:120–123`, `:142`). Training uses
a receiver-specific random mask, independent of the gap mask, and a
separate receiver CE
(`models/candidate_components_v2_6.py:101–111`, `:160–173`).

Sampling uses receiver logits independently from gap logits for both
feedback and final decode
(`models/candidate_components_v2_6.py:395–405`, `:420–427`). The adapter
then stacks the independently decoded receiver categories beside the gap
array (`generators/candidate_model_backends_v2_6.py:642–711`).

The v2.8 gap intervention does not touch receiver. Its final receiver effect
is `0.021126555312304892`, which fails the frozen `0.02` hard guard. The
v2.7 parent value was `0.01810577260056522`, close enough to the boundary
that the unchanged native receiver path was not robust across the
predefined evaluation samples.

### 3.5 Y and length conditioning

Entity label `Y` is represented by a learned embedding broadcast over all
positions (`models/seq_denoiser.py:82–85`, `:129–134`). Classifier-free
dropout replaces some entity labels with the null token during training
(`models/cof_seqgen.py:125–131`).

Length `L` is externally fixed by the shared SamplingPlan. It is represented
by the valid-prefix mask and sequence extent, not by a separately learned
length decoder. The Transformer receives `~valid_mask` as its padding mask,
and all amount/discrete outputs are zeroed outside valid positions
(`models/cof_seqgen.py:133–139`,
`models/candidate_components_v2_6.py:287–317`, `:382–405`,
`generators/candidate_model_backends_v2_6.py:671–710`).

This Y/L contract is common to all models and must remain unchanged.

### 3.6 Coherence path

The code can convert gap and receiver logits to separate softmax
probabilities and pass them to `soft_g` or `SequenceTeacher`
(`models/cof_seqgen.py:141–159`, `:197–228`). That supplies a differentiable
summary-level coupling signal, but it does not replace the independent
discrete likelihood or independent decode.

The frozen v2.5 and v2.6 benchmark configurations set
`coherence_lambda=0.0`
(`configs/benchmark_v2/full_v2_5.yaml:251–273` and
`configs/benchmark_v2/selection_v2_6.yaml:418–437`). The observed candidate
results therefore do not establish that an auxiliary coherence teacher can
repair the native discrete factorization.

## 4. Failure interpretation

The static and runtime evidence supports the following conclusion:

1. The shared sequence backbone, mask, Y conditioning, support contract, and
   amount path are not the remaining v2.8 blocker.
2. Gap and receiver are exposed jointly to the Transformer, but their
   corruption events, likelihood terms, output heads, feedback draws, and
   final decoding decisions are independent.
3. v2.7 and v2.8 repaired individual marginals outside that model-level
   factorization. The sequence of failures moved from amount to gap and
   finally to receiver rather than producing a robust all-five-pass CoF
   candidate.
4. A further receiver bias, temperature, marginal sampler, or joint
   post-sample allocation would be another result-driven calibrator. It
   would not test whether CoF learned the gap–receiver mechanism.

The v3 intervention must therefore replace the discrete probabilistic path,
not add another correction after sampling.

## 5. Frozen components and replaced components

### 5.1 Components to retain

Both v3 candidates must retain, without a candidate-level alternative:

- the non-causal sequence Transformer backbone, learned position embedding,
  and diffusion-time embedding;
- entity-label CFG conditioning;
- the shared train-derived Y prevalence, length distribution, SamplingPlan,
  prefix-mask, padding, and zero-padding contracts;
- the numerical projection and numerical clean-value head;
- the VP-cosine clean-amount objective and DDIM schedule;
- the already fixed train-only centered empirical-residual amount contract
  used by the v2.8 parent, with the same rule and 257-point grid;
- the existing support and invalid-generator routing rules;
- `d_model=128`, two Transformer layers, batch size 256, Adam
  `lr=0.001`, zero weight decay, CFG dropout 0.15, guidance scale 2.0,
  50 diffusion steps, 20,000 requested updates, and a 7,200-second hard
  cap; and
- all five existing guard definitions and thresholds.

Preserving the amount contract means reusing its already specified
train-only algorithm as a fixed base component. It is not an authorization
to tune its grid, fitting split, residual definition, or parameters after
validation.

### 5.2 Components to replace

The following current elements are removed as one coherent model-level
discrete-path replacement:

- separately randomized gap and receiver corruption masks;
- independent `bin_head` and receiver `cat_head` likelihood factorization;
- independent gap and receiver feedback draws;
- independent final gap and receiver argmax/multinomial decodes; and
- all gap/receiver marginal bias, quantile transport, temperature search,
  and post-sample allocation hooks.

Other categorical channels, if introduced later, require a separate
preregistration. The v3 candidate family covers the current single receiver
channel only.

## 6. Finite v3 architecture candidates

The candidate family contains one frozen historical reference and exactly
two trainable architectures. Candidate results may not create a third
candidate.

### 6.1 `cof_v3_ref_v28_frozen`

This is a hash reference to the v2.8 CoF candidate and is not trained or
sampled. It records the frozen failure and cannot be selected.

### 6.2 `cof_v3_c01_direct_joint`

This candidate uses one joint discrete state:

`z = gap * K_receiver + receiver`.

It replaces the separate gap/receiver input contribution with a joint
embedding `E_joint[z]` plus one joint MASK token. A joint linear head emits
`B_gap * K_receiver` logits:

`logits_joint = W_joint h + b_joint`.

The normalized distribution is:

`p(gap, receiver | h, Y) = softmax(logits_joint)`.

Training target and sampling output use the same bijection between `z` and
`(gap, receiver)`. One draw produces one pair; it is impossible for gap and
receiver to be decoded by unrelated random draws.

### 6.3 `cof_v3_c02_factorized_joint`

This candidate represents the same joint distribution as:

`p(gap, receiver | h, Y)`

`= p(gap | h, Y) * p(receiver | gap, h, Y)`.

The gap head emits `B_gap` logits. The conditional receiver head consumes
the Transformer state and the selected/target gap embedding:

`logits_gap = W_gap h + b_gap`

`logits_receiver = MLP_receiver([h, E_gap(gap)])`.

Training uses the true train gap for the conditional receiver likelihood.
Sampling first draws gap, then draws receiver from the receiver logits
conditioned on that exact sampled gap. No receiver marginal offset is
applied.

### 6.4 Shared candidate constraints

The two candidates differ only in direct-joint versus factorized-joint
parameterization. They share:

- training seed `3001`;
- the existing frozen train and validation splits;
- SamplingPlan hash
  `862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27`;
- optimizer, update count, wall cap, checkpoint schedule, batch size,
  backbone, amount path, and sampling seed;
- equal GPU budget;
- no early stopping;
- no result-dependent checkpoint choice; and
- no hyperparameter, temperature, width, depth, weight, or support sweep.

## 7. Train-only objective

### 7.1 Paired discrete corruption

One Bernoulli corruption mask is drawn per valid sequence position and
applied to the gap–receiver pair. The pair is either both visible or both
MASK. This removes the current training mismatch in which one channel can
be masked while the other leaks the target through an independently
sampled mask.

Padding is never corrupted and never contributes to loss.

### 7.2 Direct-joint loss

For `cof_v3_c01_direct_joint`:

`L_discrete = CE(logits_joint[mask], z[mask])`.

### 7.3 Factorized-joint loss

For `cof_v3_c02_factorized_joint`:

`L_discrete`

`= CE(logits_gap[mask], gap[mask])`

`+ CE(logits_receiver[mask, gap], receiver[mask])`.

Both terms are maximum-likelihood terms for the same joint factorization.
They are not marginal matching penalties.

### 7.4 Total loss

For both candidates:

`L_total = L_amount_clean_x0 + L_discrete + L_label`.

The three weights are fixed to `1.0`. `coherence_lambda` remains `0.0` in
this first architecture-isolation experiment. Turning it on would change a
second factor and is outside this candidate family.

All transforms, category support, pair-support masks, and fit state are
derived from the frozen train split only. Validation and test rows must not
be opened by the trainer.

## 8. Sampling contract

1. Y labels, sequence lengths, and valid masks come from the unchanged
   shared SamplingPlan.
2. Numerical generation uses the frozen amount DDIM path.
3. Every valid discrete position starts from one joint MASK state.
4. At each predefined feedback point:
   - direct-joint draws one `z` from the joint logits;
   - factorized-joint draws gap, then receiver conditional on that gap.
5. Both candidates use fixed temperature `1.0`; there is no temperature
   grid or validation-selected temperature.
6. The final pair is drawn through the same joint/factorized operation used
   for feedback.
7. Invalid per-channel support is forbidden. A train-derived global
   joint-support mask may exclude never-observed pairs, but its hash must be
   stored in the checkpoint and artifact. It may not be class- or
   validation-adjusted after training.
8. Padded positions are set to canonical zeros after sampling.
9. Candidate artifacts record source, config, train manifest, SamplingPlan,
   support-mask, checkpoint, sample, and evaluation hashes.
10. No post-sample gap or receiver mutation is permitted.

## 9. Five guards, joint diagnostics, and coherence

The existing five guards remain hard eligibility criteria:

| Guard | Frozen threshold |
|---|---:|
| amount KS | `0.006081138155655141` |
| gap KS | `0.006387882975686154` |
| amount absolute standardized label effect | `0.0363693454591819` |
| gap absolute standardized label effect | `0.051540527275560376` |
| receiver maximum absolute signed frequency | `0.02` |

A candidate is eligible only if all five pass on the frozen validation
split. INVALID, missing, or non-finite results are ineligible and are never
dropped from aggregation.

The joint head also makes the following validation-only diagnostics
mandatory:

- maximum class-conditional joint total variation over `(gap, receiver)`;
- receiver conditional total variation weighted by real
  `p(gap | Y)`;
- `joint_alignment`;
- time-window fanout as the secondary joint positive control; and
- the unchanged continuous association-recovery error as a reported
  development diagnostic.

For the direct joint head, the existing coherence machinery can obtain
marginals by summing the joint probability:

`p(gap) = sum_receiver p(gap, receiver)`

and

`p(receiver) = sum_gap p(gap, receiver)`.

For the factorized head, the same marginals follow from its normalized
factorization. This keeps `soft_g`/SequenceTeacher compatibility without
pretending the heads are independent. No auxiliary coherence loss is
enabled in this candidate family.

The v2.8 selection rule remains unchanged. If both candidates pass all five
guards, select the candidate by lower `max(amount KS, gap KS)`, then lower
`amount KS + gap KS`, then lexicographic candidate ID. Joint TV,
conditional receiver TV, `joint_alignment`, and fanout remain mandatory
diagnostics but cannot alter eligibility or the tie-break. If neither
candidate passes, v3 fresh test and full execution remain forbidden. No new
calibrator may be created from the failure.

The benchmark test endpoint remains the frozen continuous
association-recovery error. It is evaluated only after candidate selection,
source/config/data freeze, and a separate fresh-test authorization.

## 10. RED-to-GREEN contract tests required before execution

| RED condition | Required GREEN behavior |
|---|---|
| Current `bin_head` and receiver `cat_head` can be called independently | v3 discrete output resolves only through the selected joint-head API |
| Gap and receiver use different corruption masks | one recorded pair mask is applied to both targets |
| Joint state encode/decode is not bijective | all valid `(gap, receiver)` pairs round-trip exactly |
| Direct-joint logits do not normalize over the Cartesian support | probabilities are finite, non-negative, and sum to one per valid position |
| Factorized receiver output is unchanged when conditioning gap changes | receiver logits demonstrably depend on the supplied gap |
| Factorized joint probability disagrees with `p(gap)*p(receiver|gap)` | enumerated joint probabilities match the factorization exactly |
| Sampling makes separate unlinked gap/receiver draws | one pair draw, or gap-then-conditional-receiver draw, is recorded per position |
| Validation/test contributes to support or fit state | trainer fails closed before model construction |
| Candidate can invoke a v2.7/v2.8 discrete calibration hook | any marginal bias, transport, allocation, or temperature-selection hook is rejected |
| New discrete path changes amount config or amount fit-state contract | candidate construction fails closed |
| Y or length differs from SamplingPlan | sampling fails before artifact creation |
| Padded rows contain non-zero amount/gap/receiver | mask contract fails and candidate is INVALID |
| Generated gap/receiver leaves train support | support routing marks the candidate INVALID |
| Candidate artifact omits train/config/source/plan/support/checkpoint hashes | terminal `COMPLETE` cannot be created |
| Stored evaluation changes any five-guard threshold | authorization validation fails |
| A marginal-only fixture passes while gap–receiver alignment is destroyed | joint diagnostics detect the destroyed dependence |
| One primary model has no all-five-pass candidate | `primary_c2_selection_ready=false`; no fresh-test authorization is possible |
| A candidate fails | no automatic architecture, budget, temperature, or threshold amendment is generated |

Fixture tests must instrument GPU query, CUDA, model fit, sample, validation,
test, and full-run calls. In the source-only implementation phase all such
counts must remain zero.

## 11. Preregistration and authorization sequence

Before any implementation or execution:

1. freeze a v3 source/config amendment containing only the two candidates
   above;
2. freeze exact module ownership, artifact paths, checkpoint schema, seed,
   budgets, support definition, joint diagnostics, and selection rule;
3. implement the RED-to-GREEN tests and source-only runner;
4. pass focused tests, repository-wide tests, compileall, and diff check;
5. create a separate validation-only authorization; and
6. run only the authorized v3 validation candidates.

Fresh test, TSTR, privacy analysis, and full/five-seed execution remain
forbidden unless an all-five-pass v3 candidate is selected and a later
authorization explicitly permits the next stage.

## 12. Why thresholds and post-hoc calibration remain frozen

The receiver miss is small in absolute terms, but it is still outside the
precommitted hard threshold. Relaxing `0.02` after observing
`0.021126555312304892` would select the threshold from the result.

Likewise, v2.7 and v2.8 already tested the strategy of correcting one output
marginal at a time. Those interventions moved the failure across channels
without establishing that the model learned
`p(gap, receiver | sequence, Y)`. A new receiver offset or joint
post-sample reallocation would continue that pattern and would make model
quality indistinguishable from evaluator-targeted repair.

The v3 proposal instead places gap–receiver dependence inside the trainable,
normalized likelihood and uses the same distribution during feedback and
final sampling. That is the minimum model-level change capable of testing
the stated mechanism while leaving the frozen benchmark contract intact.
