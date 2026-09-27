# Frozen-checkpoint follow-up D0/D1

Registered 2026-09-27 before these diagnostic outputs. Adaptive to the completed
B/B+S and event-weight control, not an independent confirmation study. No training,
checkpoint selection, final test events, or new free-rollout repair is permitted.

## Inputs and D0

Use the six completed checkpoints: B, B_S, B_event_weighted, fit seeds 20260930
and 20261001. Preserve their hashes and the existing two free draws per fit.
Compare raw Sparkov fraudTrain.csv with canonical and prepared train/development
events, including identity, order, timestamp, gap, merchant, category, amount,
and label. Read raw IDs first, then parse event attributes only for allowed IDs.
Inspect frozen encoded training/validation sequence lengths and label counts;
IDs were removed at encoding, so do not claim an ID-level encoded round trip.
Tabulate all-normal, mixed, all-fraud customer counts and transitions. These
retrospective cohorts are evaluation strata only, never generation conditions.

## D1

- Teacher: all 147 development customers and all their events, actual prior
  events and actual current non-label fields. Record raw label NLL, sampling-mask
  adjusted fraud probability, amount digit NLL, history length and prior run age.
- One-step: same true prefixes and same actual length/index tokens; sample the
  current transaction using native heads and probability masks. Include every
  first event, actual fraud event, and event following fraud; additionally sample
  four remaining normal events per customer without replacement (seed 20260927).
  Freeze this position list before loading checkpoint weights.
- At each selected prefix use 16 Monte Carlo draws for (a) all current fields
  sampled, (b-e) additionally fixing one of amount, gap, merchant, category to
  its actual current value. Actual label is NEVER fixed in these probes. Holding
  an observed field fixed while sampling earlier fields is a sensitivity probe,
  not sampling from a fully Bayesian observational conditional distribution.
- Capture the label distribution rather than judging one sampled label. Save
  per-prefix means and Monte Carlo standard error. Count unique prefixes and
  customers; repeated draws do not increase the number of independent customers.
- Free rollout: reuse the twelve previously generated sets and evaluate observed
  transitions/cohorts. Their lengths are freely generated, unlike the true-length
  teacher/probe conditions, so the difference is NOT a pure exposure-bias effect.

Cache native teacher history outputs only for one-step queries; synthetic dummy
LSTM recurrence is discarded. Validate cached logits against both teacher logits
and the native true-prefix recurrence, including S memory, before checkpoint runs.
No predicted current label may enter the cached history. Use the pinned runtime,
eval mode, native unknown-token suppression, deterministic seed streams, four CPU
threads, and the two verified idle GPUs. Save code/protocol/input hashes and status.

## Interpretation and next decision

This isolates where the current fitted models fail; it does not prove an inherent
ARGN expressivity limit. If true current fields restore a relationship that current
field generation breaks, inspect that factorization and its learned distributions
before enlarging the past encoder. If even teacher conditioning fails, distinguish
objective, effective sample support, fit/checkpoint selection and representation.
Report normal quality and minority conditions separately. Do not select a new
architecture from overall fraud prevalence alone.

Prior-method overlap and pending comparisons remain those in
[RESEARCH_DECISIONS](../RESEARCH_DECISIONS.md) and
[NEXT_DECISION_PLAN](NEXT_DECISION_PLAN.md): ARGN/TabDiT already generate conditional
sequences; Seq2Synth evaluates sequential fidelity; conditional numeric refinement
is covered by TabCascade. D0/D1 is diagnosis, not a methodological contribution.
