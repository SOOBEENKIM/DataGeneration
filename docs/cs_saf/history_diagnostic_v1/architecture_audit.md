# Architecture, theory, implementation and research-scope audit

This is a bounded audit of the current controlled U/G/C implementation and frozen U/G+A diagnostic. Passing checks does not establish absence of all possible bugs, sufficient optimization, novelty, financial realism or superiority over external models. Historical E/ER findings are not retrospectively reclassified.

## Actual architecture and mathematical meaning

For group c and strict prefix H_t=(gap,mark,value)_{<t}, the encoder computes h_t=GRU(H_t,c), dimension 128; a direct group embedding of dimension 8 is concatenated to h_t. Current gap has 31 positive support representatives; each observed group has a separate rank-16 interaction bank. The generator samples in order

`gap_t -> mark_t -> value_t -> next history`, conditional on the preceding history and group.

First gap is missing; first mark and value remain part of the next history. Length and group are supplied by a fixed plan, not learned. The controlled closed vocabulary has 64 ordinary marks; reserved codes cannot be generated. All three structure variants have 133,549 parameters and paired initial tensors, but equal count does not imply equal function classes or optimization difficulty.

### U: latent copy mixture

Let s_j(H) be the fresh-mark softmax and q(H,g)=sigmoid(b(H)+f_c(H,g)). Then

`P_U(mark=j|H,g) = q 1[j=previous] + (1-q) s_j(H)`.

Observable repetition is `r=q+(1-q)s_previous`, NOT q: a fresh draw can accidentally equal the previous mark. This identity is checked numerically. Latent copy probability and the fresh distribution are not generally separately identifiable from arbitrary observed categorical probabilities; neither is interpreted as a recovered causal mechanism.

### G: observed repeat / nonrepeat factorization

`r(H,g)=sigmoid(b(H)+f_c(H,g))`.

`P_G(previous|H,g)=r`; for j unequal to previous,

`P_G(j|H,g)=(1-r) exp(z_j(H))/sum_{k != previous} exp(z_k(H))`.

The nonrepeat head excludes the previous mark and reserved codes. The resulting probabilities sum to one, and the mark NLL factors into repeat Bernoulli NLL plus the conditional nonrepeat category NLL when needed. First mark uses the common fresh-mark head. G is a valid internal architecture variant created in this project, not an external published baseline or proof of architectural novelty.

**Expressivity restriction:** current gap changes repetition, but not relative probabilities among different nonrepeat marks. In this particular DGP fresh alternatives are uniform, so this restriction alone does not prove the cause of current errors. It can become limiting for different transition laws.

### C: a valid local constraint, not a generated-distribution guarantee

C solves an intercept alpha so that `sum_b a_b(H) sigmoid(score_b(H)+alpha)=rho(H)`, where a is the model gap distribution and rho is a learned base probability. Positive a, 0<rho<1 imply a unique root. Implicit derivatives are checked against numerical gradients. Derivatives follow from the strictly positive denominator `sum_b a_b r_b(1-r_b)`; numerical degeneracy is explicitly rejected.

This fixes a marginal only for ONE given history under its learned gap distribution. Generated histories change after each sampled mark. It does not force the global repeat rate, stationarity, real-data marginal, or desired repeat-gap relationship. Post-hoc corrections can also break the C constraint. C's previous failure remains a failure.

### Existing correction A and rejected B/P

A replaces the observable repeat probability by `sigmoid(logit(r)+delta[c,bin(g)])`, with ten train-only fitted offsets; it redistributes the remaining probability in the existing nonrepeat proportions. It is applied before numeric generation and history feedback. It is not a post-hoc rewrite of a finished table.

B/P changed only these ten offsets. P limited per-group average observed-history prediction cost. Average Brier/NLL control does not mathematically bound each history's gap-response range. Matching coarse generated frequencies also leaves many history-conditional laws unconstrained. These are limitations of the objective, not grounds for reinterpreting the failed experiment as a success.

## Training objective and discretization caveats

The implemented ordinary objective is

`L = mean_valid_gaps NLL_gap + mean_valid_marks NLL_mark + mean_valid_values NLL_value`.

It is a component-normalized weighted likelihood objective. Gap targets exclude the first event; mark/value targets include it. Consequently it is not exactly a single unweighted full-sequence joint NLL. Random batches and their different denominators add a small weighting difference; all internal structures share the same convention. Validation selection recomputes component means over the full split. It is inaccurate to call this exact maximum likelihood of original continuous transaction sequences.

More substantially, current-gap targets are discretized, past gaps in real-history training remain continuous, and generated gaps are representatives. The network is a normalized autoregressive sampler, but the training inputs differ from those visited during its support-valued generation. This is a possible representation/coverage mismatch, not a demonstrated source of this experiment's residual error. A future controlled comparison must change this alone if tested.

The GRU can use prefix history, but has no theorem guaranteeing sufficient recovery of the DGP's hidden regime and remaining duration. A run length is an observable summary, not the hidden semi-Markov duration. A late-position error alone cannot establish that the GRU has forgotten history.

Training-generation differences are not a new research problem. [Scheduled Sampling](https://arxiv.org/abs/1506.03099) changes training inputs and [Professor Forcing](https://arxiv.org/abs/1610.09038) aligns recurrent dynamics. These references motivate caution about novelty; the current diagnostic implements neither method and does not establish their comparative merits.

## Checks actually performed

- 106 selected automated tests: causal encoder, masks, codecs, probability normalization, observable-copy identities, initialization, oracle probability factorization, independent dense hidden-state recursion, no future leakage, generation metrics, replay and new strict-prefix/run/censoring/bootstrap semantics.
- Every saved U/G parent: checkpoint/provenance hashes, frozen state before/after checks, best-epoch selection from recorded validation scores, no train/validation entity overlap, train-only correction metadata; 133,549 parameters.
- Current/future mark and numeric perturbations cannot change the current prediction; later gaps cannot change earlier predictions.
- Replayed decoded stored generation through full-prefix computation and a separate streaming path; compare using the original numerical codec and native sampling normalization.
- Recomputed saved actual-history feature predictions; compare with immutable evaluation-feature manifests.
- C implicit-gradient finite-difference check rerun; previous detailed C normalization/constraint/initialization/training gates still refer to identical source hashes.
- Original 36 A generations and 120 oracle generations verified against their saved hashes. No new fitting or generations in this diagnostic.

Machine-readable bounds and all 12 parents are in [audit.json](audit.json). The initial broad legacy tests had missing-file errors because this worktree did not contain historical artifacts; restoring local links to original artifacts resolved them. No old artifacts or model algorithms were changed to make the checks pass. These links are local prerequisites, not portable Git backups.

## Evaluation and research interpretation boundaries

- The oracle is an exact observable-information filter for the controlled law, never a realized hidden-state target. Real histories use past-continuous/current-bin information; generated gap codes use the binned target law, with a separate representative-as-continuous sensitivity check.
- Replay on another generator's paths assesses prediction at those given paths. It does not identify a unique causal share attributable to architecture versus visited-history distribution.
- Sequence-level bootstrap is conditional on the observed data and frozen weights. Three seeds and reused validation do not become independent data replication; many viewed cells require explicit exploratory wording.
- The DGP includes independent Gaussian numeric values, uniform fresh alternative marks, two supplied groups and short sequences (16-32 events). It does not establish realistic monetary tails, arbitrary receiver relations, learned sequence length, or downstream fraud-detection benefit.
- Repeated validation-guided design is exploratory. No published-baseline superiority or conference-level method contribution follows from this audit. A contribution needs a reproducible substantive error, a justified modification exceeding simple controls, cost preservation, and later independent/external validation.

**Audit conclusion before localization:** no result-invalidating error was found in the inspected current U/G inference and saved-result paths. Several mathematical guarantees that would be needed for stronger claims do not exist; those claims must not be made. The new diagnostic can proceed within these limitations.
