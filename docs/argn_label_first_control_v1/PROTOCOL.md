# Label-first ARGN: matched order by objective diagnostic

Registered 2026-09-27, before these fits. This is a configuration control, not a
new model contribution. The frozen earlier studies and unsuccessful fits remain
unchanged. An exactly matching published Sparkov experiment is not a prerequisite
for this controlled next step.

## Observation and question

The completed D0/D1 verified raw-to-prepared provenance. On 1,133 development
events following fraud, actual continuation is 91.0856%; B_event_weighted predicts
1.1608–4.3716% even with real past and current fields. Its free fraud prevalence
is 0.3896–0.6058%, but mean fraud-run length is 1.010–1.023 (real 10.0526).
Customer-weighted B instead overproduces fraud (19.717–53.363%). S improves
continuation and also mistakenly increases continuation at actual termination.
These are fitted-setting failures, not proof of ARGN's intrinsic incapacity.

Question: does moving the label before the current transaction fields change
these failures without adding state information, parameters or a new loss?

## Closest prior work and exact overlap

- [ARGN v2 §3.3–3.6](https://arxiv.org/html/2501.12012v2): conditional column
  factorization and sequential history already exist. Moving a column is an
  ARGN configuration control. Retain official engine 1.0.4, encoding and heads.
- [TabDiT v2 §3](https://arxiv.org/html/2504.07566v2): row-internal autoregressive
  decoding and latent sequence diffusion are prior methods. We do not add
  diffusion or claim that label-conditioning itself is new. Direct comparison
  is still required before a generator superiority claim.
- [Behavioral fraud benchmark, April 2026 v1 §5.2–5.4, §6.2]
  (https://arxiv.org/html/2604.13125v1): temporal/burst/velocity evaluation already
  exists. The inspected benchmark uses row generation and assigns pseudo-entities;
  its ARGN result is not evidence about the sequential ARGN used here. Dataset
  and sampling conditions also differ. We do not adopt its structural assertions
  as a theorem about our setting.
- [RED-SDS, NeurIPS 2021 §3]
  (https://papers.neurips.cc/paper_files/paper/2021/file/fb4c835feb0a65cc39739320d7a51c02-Paper.pdf)
  already connects a recurrent state to switches and explicit duration counts.
  [REDSLDS, AISTATS 2025](https://proceedings.mlr.press/v258/slupinski25a.html)
  also models recurrent explicit duration. Adding run age alone is not novel.
- [OmegaSDS, May 2026 v1 §2, §4](https://arxiv.org/html/2605.06315v1): recurrent
  switches and regime-dependent dynamics with flow emissions and exact likelihood
  estimation are prior art. Our observable fraud label and heterogeneous fields
  are a different task, not by themselves a novel mechanism. The current order
  control does not implement or reproduce those latent-state estimators.
- [TabCascade v3](https://arxiv.org/html/2601.22816v3): conditional numeric
  refinement is prior art; amount-head development remains a later, separately
  controlled step.

Reading scope: relevant methods/setup sections and abstract/official metadata
where noted, not a claim of exhaustive review or reproduction of these models.

## Registered arms

| New arm | Only change from stored paired baseline | Loss | Paired baseline |
|---|---|---|---|
| B_label_first | move label before current transaction fields | customer mean | B |
| B_event_label_first | same order change | event weighting | B_event_weighted |

Both use order: native length/position → label → category → gap → merchant → amount.
Both disable S exactly as their paired baseline. The inert 8,320-parameter state
projection is retained for parameter and initialization equivalence. No true
current label is supplied during free generation; the model samples it, then
conditions subsequent fields on that sampled label.

Use existing optimization/internal-validation split (619/69), encoded data,
full retained history, DIGIT amount/gap, Medium backbone, seeds 20260930/20261001,
batch 2, accumulation 16, 60 epochs / 180 minutes upper bounds and native minimum
validation-loss checkpoint with patience 4. Same seed gives identical initial
weights, parameter count and data, but order changes can change random-number
consumption and optimization trajectories. No claim of bitwise paired dropout.
Two fits per new arm and two free draws per fit: four fits, eight generated sets.
Reuse the four paired earlier fits rather than retrain identical controls.

Event-weighted arm preserves the previous exact transaction weighting and the
unchanged customer-weighted length loss. Label rebalancing, forced fraud ratios,
forced run lengths, amount head, new state head and detector training are absent.

## Input protection and checks before launch

- Verify frozen preparation/source hashes, earlier checkpoint/generation hashes
  and D1 development-input hashes. Read no held-out final-test event attributes.
- Tests: identical initial weights and capacity; exact order in both training
  and sampling; correct sequential loss dispatch; current/suffix fields cannot
  affect label_t; full teacher equals one-step label prediction; native recurrent
  state filtering; event weighting changes only its registered loss terms.
- Save code/config/protocol hashes before any fit. Fail on drift and preserve
  failures; do not overwrite a started run or silently resume with new code.
- GPU 2/3 UUIDs only, after two low-memory/low-utilization checks and no compute
  process. Never use GPUs 0/1 for this study. Dispatch at most two processes and
  recheck before each new job. Do not terminate foreign processes.

## Evaluation and interpretation

Each completed model automatically generates on the same 147 development customer
contexts with seeds 20261011/20261012. Native generated length; no real future
length, true label, true fraud membership or prevalence/run-length repair.

Reuse frozen metrics: prevalence, initial fraud, onset/continuation/termination,
observed and completed run-length/span distributions, age-specific termination,
amount versus personal history, category/merchant relations, normal amount/gap
quality, length and merchant diversity. Report denominators and censored endpoints;
fraud termination is not customer-sequence termination.

Also probe all 177,997 development events using real past and true structural
length/position tokens, exactly as D1. Save label probabilities, NLL, amount NLL
and cohort/transition summaries. Current transaction fields cannot influence
the first-generated label; teacher and sampled-current-field probabilities
therefore coincide for fixed history/structural tokens (verified in tests and
on each fitted checkpoint). These teacher probabilities do not have the same
conditioning set as the old label-last probabilities, and lower NLL alone is
not a universal ranking. Compare conditional calibration and free generation
together. True-length diagnostic versus free generation is not pure exposure
bias isolation.

This is adaptive development, two fit seeds, not a final significance test.
Every draw remains visible; compare fit means with the matching stored objective,
not only the best draw. No development-based selection is reported as test success.

## Decision after results

1. If order alone jointly improves continuation, termination, prevalence and
   personal-amount fidelity without a clear normal-quality/diversity regression,
   retain it as a stronger baseline; do not credit a new state architecture.
2. If errors persist, identify whether first-event prior, onset, run age or
   normal-return calibration remains wrong. Then register an explicit
   transition/emission candidate and its order-only, simple Markov/duration,
   duration-removal and personal-history-removal controls. Do not relabel the old
   S addition as this new structure.
3. If gains only trade persistence for excess fraud or overlong runs, the current
   change is insufficient. Do not start amount/diffusion changes simultaneously.
4. No positive publication claim follows merely from beating ARGN. CPAR adequate
   training and TabDiT/closest task-matched recent generator comparison remain
   part of the accepted research direction. Sparkov's simulator origin limits
   external validity; Berka supports non-fraud quality only.

The automatic queue covers this complete order control and evaluation. It does
not automatically choose a new architecture based on one metric.
