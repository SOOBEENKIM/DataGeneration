# CoFSeqGen-SAF pre-execution implementation report

Date: 2026-08-26  
State: `PREEXECUTION_SOURCE_IMPLEMENTED_METRIC_AUDIT_COMPLETE`

## What is complete

### Metric validity audit

The metric suite directly represents the proposed factorization:

- gap-bin-conditioned mark total variation;
- short-gap/repeat-curve L1;
- gap/repeat mutual-information error;
- lag-1 gap and amount autocorrelation errors;
- gap, mark, amount, and length marginal/structural diagnostics;
- exact-trajectory and transition-ngram exposure diagnostics.

The audit uses entity-disjoint halves of train only. It checks a real-vs-real
null and four known alternatives: mark permutation, non-first-gap permutation,
numeric shift/scale, and exact trajectory copy. Mark and gap permutations
retain their targeted marginal distribution. All six registered sensitivity
checks passed for all eight canonical views:

| Dataset/view | Train entities | Train events | Audit checks |
|---|---:|---:|---|
| AMLSim | 6,999 | 929,289 | 6/6 |
| Berka full | 3,150 | 737,846 | 6/6 |
| Berka nested 900 | 630 | 145,947 | 6/6 |
| Citi Bike | 5,238 | 392,813 | 6/6 |
| DGP κ=0 | 31,951 | 766,935 | 6/6 |
| DGP κ=1 | 31,951 | 766,935 | 6/6 |
| H&M 10k | 7,000 | 164,403 | 6/6 |
| Sparkov | 688 | 916,567 | 6/6 |

Thirty-one train-only pseudo-splits were used to freeze q95
marginal/structural non-inferiority margins per dataset. No candidate output,
validation event, or test event was used. Privacy copy sensitivity is audited
as a prespecified risk check rather than converted into a permissive
real-vs-real margin.

The frozen utility tasks are TSTR next-gap regression (Ridge on log-gap) and
TSTR next-mark-repeat prediction (logistic regression), each capped at 50,000
train/test transitions. Both target-permutation checks passed on AMLSim,
Berka full, Berka nested 900, Citi Bike, H&M, and Sparkov. On both DGP cells,
the gap utility check passed but the generic repeat task did not have
corruption sensitivity. That 1/2 result is retained as a power warning; it is
not a reason to replace the task and is not model evidence. DGP mechanism
identification therefore remains anchored to its oracle and primary
gap-conditioned-mark metrics, while H5 utility uses paired TSTR comparisons.

Result artifact:
`artifacts/cof_seqgen_saf/metric_validity_audit_train_only.json`

- bytes: 49,466
- SHA-256:
  `4c0cec50af9dceccad8a709cbf7393aa1523e8f1ff55e8f43db4d65638395ce4`

An audit pass proves that these constructed metrics respond to the registered
corruptions. It is not evidence that SAF-O1 outperforms any model.

### Baseline compatibility design

The benchmark now has non-overlapping roles:

- primary sequential: TabularARGN, TabDiT, CPAR, REaLTabFormer, empirical
  sequence sampler;
- flattened marginal controls: CTGAN, TVAE, GaussianCopula;
- frozen historical context: non-v3 CoF and CCMTPP C1;
- stopped failed-gate context: HCMTTPP H1.

TabPFN is explicitly excluded from generator ranking. The common primary task
uses a shared parent-context and sequence-length plan sampled from train only.
The contract validates raw first-gap, nonnegative-gap, length, event-index,
timestamp consistency, and identity rules before postprocessing. Repaired
outputs and native end-to-end modes are secondary and separately labelled.

The read-only dependency probe found CTGAN/TVAE's `ctgan` module available.
`sdv`, `realtabformer`, and `mostlyai` were unavailable. No installation
was attempted. TabDiT and historical code use repository/frozen-artifact
integration rather than a package probe.

### CoFSeqGen-SAF and ablations

The core implements:

- a shifted GRU whose position t sees only events before t;
- observed-train-atom gap support, retaining an exact zero atom only when observed;
- unordered categorical, ordered hazard, and stable hurdle-lognormal gap
  decoders;
- explicit current-gap routing to mark in U1/O1;
- current-gap and current-mark routing to the numerical head in every
  candidate, preventing H3 from being confounded by a changed value head;
- observed current gap/mark teacher forcing in likelihood training;
- generated current gap/mark routing only during sampling;
- a missing first gap excluded from gap NLL while retaining the first
  mark/value in later history;
- bounded hazard logits, clamped positive scales, and rolling-window
  fixed-length sampling.

The five candidates are SAF-C0, SAF-U0, SAF-O0, SAF-U1, and SAF-O1. Decoded
support-aligned gaps are always members of the fitted train support. The core
does not yet implement auxiliary event heads or learned sequence termination.

## Verification

The full SAF CPU suite passed 43/43 tests covering:

- support preservation and reserved missing code;
- ordered-hazard normalization and gradients;
- finite loss/backpropagation for all five candidates;
- strict no-future leakage;
- retention of the first event in subsequent history;
- isolated routed/unrouted mark-head behavior;
- support-safe sampling and shared length-plan enforcement;
- baseline tiers, train-only plans, views, and raw-output rejection;
- metric family registration and corruption sensitivity;
- actual trajectory/ngram copy detection and next-event TSTR utility power.

The pre-existing canonical data verifier remains green from the preceding
materialization stage.

## Exact remaining boundary

The work is not yet a trained experimental result. The following still require
the next execution authorization and must be done in order:

1. implement the canonical tensorizer, auxiliary heads, checkpoint/artifact
   runner, and external baseline wrappers;
2. resolve and lock baseline dependency versions without changing the
   comparison tiers;
3. run CPU toy end-to-end and one-dataset overfit gates;
4. only then run GPU training and candidate sampling on development roles;
5. apply the frozen metric suite and margins; keep test sealed until selection
   is final.

No superiority, non-inferiority, SOTA, privacy, or publication claim is
authorized by this report.
