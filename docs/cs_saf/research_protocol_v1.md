# CS-SAF research protocol v1

Status (2026-09-16): **implemented; all prevalence oracle and CPU gates PASS;
single-seed pilot FAIL at pi=0.25 (null safety), after pi=0.05/0.10 PASS**.
Pi=0.50 training and all confirmatory stages remain unexecuted under the stop rule.
Base commit: `de8fa70`  
Evaluation boundary: **train/validation only; held-out test remains sealed**

The 2026-09-16 pre-training clarification is
[architecture and oracle contract v1](architecture_and_oracle_v1.md).
It fixes the bilinear route, uses observable-repeat likelihood rather than
mislabeling observed repeats as latent copy coins, and defines a train-only
semi-Markov filtering oracle. Conditional input response is not a causal
do(gap) effect. The original pilot thresholds and seed sets remain unchanged.

The [2026-09-16 oracle report](oracle_audit_v1_report_2026_09_16.md) records
two byte-identical executions on existing train data. It establishes oracle
feasibility, not a successful learned model or a completed prevalence study.

The subsequent [pilot execution contract](pilot_execution_contract_v1.md) fixed
the implementation, data materialization and training budget before execution.
The [pilot result](pilot_v1_report_2026_09_16.md) records 12 completed fits and the
failed null-safety gate. The full candidate learned a material active response,
but did not pass the entire pilot. The v1 criteria below remain unchanged;
five-seed confirmation must not proceed from this result.

## Research question

Can a sequential synthetic-data generator preserve both the valid support of
inter-event gaps and rare, context-specific gap-to-mark transition mechanisms,
when aggregate likelihood and marginal fidelity tend to average those
mechanisms away?

## Working insight

We call the target failure mode **conditional mechanism dilution**: a generator
can match global marginals while losing a stochastic transition rule that is
active only in a small observed context. Standard empirical-risk minimization
weights that learning signal in proportion to context prevalence, and aggregate
metrics can hide the resulting failure.

This is a hypothesis to test, not a completed claim.

## Evidence inherited from `main`

- Train-support alignment repeatedly improved gap KS and scaled Wasserstein
  distance in the existing controlled experiments.
- SAF v2--v5 cannot establish static-aware routing because a now-fixed scalar
  codec defect mapped the intended static label to the unknown category.
- With the codec fixed, scalar-gated SAF v6 represented static context but
  failed the frozen intervention-selectivity preflight.
- No held-out test result supports CS-SAF, and no dependency-routing success is
  claimed at this point.

## Required contribution

The conference-oriented paper is viable only if all three parts are supported:

1. **Failure-mode evidence:** decreasing context prevalence degrades conditional
   transition fidelity before it materially degrades aggregate fidelity.
2. **Method:** exact support alignment plus a context-selective gap-to-mark route
   and a context-balanced mechanism objective.
3. **Evaluation:** intervention response, conditional transition fidelity,
   worst-context utility, and null-safety reveal behavior missed by aggregate
   metrics.

Support alignment alone, a new gate alone, or broad superiority on an
undifferentiated fidelity/utility/privacy score is not the intended claim.

## Separation from adjacent work

- Marked temporal point-process work models time--mark dependence primarily for
  likelihood and next-event prediction; CS-SAF targets synthetic-dataset
  generation under exact support and rare-context mechanism fidelity.
- Dependency-repair methods for tabular data commonly reconstruct known
  deterministic rules; CS-SAF must learn a stochastic, history-dependent rule
  from observed sequences.
- Subgroup and logical-consistency diagnostics identify evaluation failures;
  CS-SAF must additionally provide a generative mechanism and null-safe evidence
  that it preserves the conditional response.

## Controlled study

The existing controlled generator, entity-disjoint split, tensorizer, and
support-aligned gap representation remain the base. The study adds only the
factors required to identify dilution:

- context prevalence `pi`: `0.05`, `0.10`, `0.25`, `0.50`;
- dependency state `kappa`: `0` (null) and `1` (active);
- pilot seed: `20260930`;
- confirmatory seeds: `20261001`--`20261005`;
- identical data, parameter budget, optimization budget, copy/new decoder, and
  sampling plan for matched comparisons.

The DGP must first pass an oracle identifiability audit. If the empirical active
cell does not contain a materially larger intervention response than every null
cell, model training is not interpretable and must not start.

## Model family

The primary CS-SAF candidate will contain:

- the existing train-support-aligned gap decoder;
- a direct vector or bilinear interaction between observed static/history
  context and the current gap;
- the existing copy/new mark factorization;
- a context-balanced auxiliary copy loss so rare contexts do not contribute only
  in proportion to their raw frequency.

Required ablations use the same data and compute budget:

- matched autoregressive copy/new baseline;
- support alignment only;
- context route with ordinary average likelihood;
- context-balanced objective without the interaction;
- full CS-SAF.

Ordered hazard is not a primary contribution and is excluded from the first
confirmatory family.

## Frozen preflight criteria

For the pilot intervention audit, all conditions must hold:

1. `kappa=1`, rare-context copy-probability range is at least `0.05`;
2. each noncausal cell has range at most `0.05`;
3. active-cell sensitivity is at least twice the largest noncausal sensitivity;
4. the matched zero-gap control has maximum range at most `1e-8`;
5. generated gaps have zero support violations.

Failure stops the pilot. Thresholds must not be lowered after observing results.

## Confirmatory criteria

The five-seed controlled study proceeds only after the pilot passes. CS-SAF must:

- improve active-cell intervention error and transition-conditional TV over the
  ordinary-likelihood routed model in at least four of five seeds;
- show the same direction in the seed aggregate;
- preserve null safety at `kappa=0`;
- retain zero gap-support violations;
- avoid a material regression in aggregate fidelity and TSTR utility under the
  existing train-only noninferiority margins.

Only after these conditions pass may the study proceed to unchanged real-data
sources and external baselines. Privacy is evaluated under matched sample size
and access conditions; universal superiority on every privacy metric is not a
required or defensible claim.

## Stop rule

- Oracle audit failure: repair only the controlled identification design.
- Pilot model failure: stop this predefined CS-SAF candidate; do not run the
  five-seed or real-data stages and do not relax the gates.
- Five-seed failure: report the boundary of the mechanism and do not access the
  held-out test.
- Confirmatory success: freeze code and checkpoints, then run real-data/baseline
  validation and one final held-out evaluation under a separate authorization.
