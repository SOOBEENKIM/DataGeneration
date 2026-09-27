# Repository, primary-literature and fitted-model re-audit

2026-09-27, requested by the user after completion of the label-order controls.
This audit preserves all existing unsuccessful results and frozen protocols.
The intended research is an ARGN-based contribution improving customer-sequence
relationships, not an endless search for a favorable seed and not a substitution
of fraud detection/row oversampling for the generator's task.

## Scope and deliverables

1. Fetch every published DataGeneration branch and inventory commits, model
   definitions, runners/configurations, tests and reports. Distinguish inventoried
   files from methods and paths actually inspected; do not claim that a file list
   establishes a complete implementation review.
2. Re-read the supplied finance survey's tabular section and follow its direct
   primary references. Update 2026 transaction, sequential tabular, conditional
   numerical, rare-state and duration-related primary work. Record reading scope,
   data/task/label usage and overlap; do not claim an exhaustive literature census.
3. Audit the chain from controlled CS-SAF/CoF-SAF through the external U/G/D/D_bin
   variants and the ARGN relation/state/loss/order controls. Check task drift,
   objective/metric alignment, inference/training paths and baseline adequacy.
4. Resolve the missing distinction between fitting failure and generalization
   failure before choosing another model modification.
5. Produce a concrete correction/development decision tied to the findings.
   Novelty requires current task-matched comparisons; successful ordinary fixes
   are not retrospectively renamed a methodological contribution.

## Frozen model diagnostic registered before execution

Read all ten saved models: B, B_S, B_event_weighted, B_label_first and
B_event_label_first, fit seeds 20260930/20261001. Evaluate exact stored optimization
(619 customers), internal validation (69), and already-opened development (147)
teacher-forced sequences. No additional large fit, test events, true-label
generation conditioning or artifact mutation is authorized by this diagnostic.

Compute label probabilities/NLL and per-column NLL on valid real events,
aggregated by split and previous label, with first events separated. Also report
continuation/termination/onset diagnostic subsets with support counts; observed
onsets selected using the outcome are not independently calibrated risk sets.
Report both event and customer-mean reductions. True length/position tokens
remain present, as in earlier diagnostics. They are not free-generation inputs.

Train-only simple label references: a Bernoulli first-event prior, a two-state
Markov transition table, and a previous-label/run-age table (ages 1..20, 21+),
all estimated from the same 619 optimization sequences. Use Jeffreys 0.5 count
smoothing; unseen age cells fall back to the previous-label table. These are
label prediction references, not full transaction generators or proposed novelty.
They establish whether the observed transitions can be learned from permitted
data without the neural backbone. No hyperparameter search or development fit.

Fail on changed checkpoint/preparation hashes. Save aggregate tables and hashes;
per-event records remain local. Use only verified idle GPUs for the read-only
forward passes. A clear optimization-split failure directs a learning/input-path
investigation; a training/development gap directs generalization/context analysis.
Neither diagnosis alone proves a model's inherent representational limit.
