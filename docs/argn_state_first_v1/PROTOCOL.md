# First state-conditioned ARGN study (registered before fitting)

2026-09-27. Implements the agreed first B versus B+S study; the amount-head
intervention A is deferred. This is a development experiment, not a confirmed
novelty, convergence, or superiority claim.

## Frozen comparison

- Starting code: audit commit `35637cc1cd64a5549796e76fbf8b92996f00bfc6`.
- Official MOSTLY AI engine 1.0.4 / ARGN Medium, torch 2.5.1 (CUDA build).
- Reuse the frozen Sparkov outer-training DIGIT codec for both gap and amount,
  the same 619 optimization / 69 internal-validation customers, and 147 outer
  development contexts. The codec was fitted on all 688 outer-training customers
  by the official procedure; this inherited internal-validation overlap is
  disclosed. Test events remain unopened.
- Both arms start from fresh weights for seeds 20260930 and 20261001. Same
  initialization, data order, optimizer, loss, stopping policy, and generator.
  Config JSON defines exact budgets (60 epochs, 180 minutes per fit, whichever
  stops first; native internal-validation early stopping remains enabled).
- Fixed within-event order for both training and generation: native sequence
  length/position, category, gap, merchant, amount, fraud label. This removes
  random column-order validation noise. Labels remain jointly generated; there
  is no detector in this first training experiment.
- **Both arms use full frozen encoded customer sequences without additional cropping.** The earlier audit
  used native 100-event windows. This deliberate common change prevents S from
  accessing a longer training history than B, removes cropped-boundary resets,
  and makes internal validation deterministic. B must be freshly trained and
  cannot be substituted with the previous audit's B scores. The inherited official
  encoding already trimmed 58 optimization events at its protected maximum length
  3106 (raw maximum 3123): 816,341 raw optimization events become 816,283 encoded
  events, while all 100,226 internal-validation events remain. This study adds no
  further truncation and gives the same retained histories to B and B+S. Exact
  encoded lengths are recorded in the preparation manifest.
- Native customer-weighted / per-column losses remain unchanged. Natural event
  prevalence is not equivalent to customer weighting; the first experiment does
  not claim to resolve that known possible cause.

## S implementation and controls

Use one bias-free linear projection of ten past-only features, added to the
history representation seen by transaction regressors. The native length and
position regressors are not augmented. All native output heads remain unchanged.

The features are empty-history indicator, previous fraud/unknown indicators,
log consecutive same-label count, log elapsed time within that run, last/mean/
EMA/std log amount, and EMA log inter-event gap. EMA alpha is 1/16. Amount and
duration/gap summaries have feature-only caps of 10,000 and 604,800 seconds;
generated amounts, gaps, labels and run lengths are **not** clipped by this
module. Caps are below every value in the inherited codec's protected upper tail,
making summaries invariant to the native decoder's random extreme replacement.
The log elapsed feature therefore means elapsed time up to seven days, not an
uncensored duration estimate. All caps are fixed before new outcomes are seen.

Features at t consume only rows <t. The same `PastState.advance` computes
training features and updates generated state. Training uses encoded real past;
sampling uses generated past. The first row has zero state, and its sampled gap
is excluded from gap history. A label switch resets run count to one and run
elapsed time to zero; it does not terminate the customer sequence. Categorical
unknown is explicitly represented rather than treated as fraud or normal.

B includes the same projection parameters but receives zero state. A native
equivalence test verifies identical logits; the projection has zero data gradient
for B. Thus nominal parameter count matches, but effective learned capacity
increases in B+S and is reported. This first test estimates the benefit of simple
state-feature augmentation; it does not separate additional learned capacity
from meaningful feature content or prove a new transition architecture. A later
matched-capacity learned-history/shuffled-state control is required for that
stronger claim. Duration and amount groups are named separately in source so
subsequent group-removal studies can isolate their effects.

Sequence length uses native ARGN logic: a synthetic length is generated and then
used for its positional encodings. The inherited training use of true sequence
length is disclosed; S adds no length/future-label information. State memory is
packed alongside LSTM state so native batch filtering removes the correct
customer when their generated sequence ends. No maximum-19 fraud rule, label
repair, external prevalence target, or conditioned real future length is used.

## Evaluation and interpretation fixed before S outcomes

Generate two free-running development datasets per checkpoint (seeds 20261011,
20261012), using all 147 common static contexts and native generated lengths.
Primary diagnostics: observed fraud-run count and elapsed-duration distributions,
continuation/termination by run age, and fraud onset rates. Treat left/right
observation-boundary censoring explicitly and keep event-count versus clock-time
duration separate. Connected outcomes: personal-history-relative amount by fraud
run position, gap/category/merchant relationships. Guardrails: natural event
prevalence, normal-transaction distribution, customer lengths, diversity, raw
amount tails. Rare five-second fraud cells are descriptive only.

Report paired-seed estimates and real-real development resampling intervals;
generation-seed repeats are not independent model fits. No post-hoc composite
score or pass/fail tolerance is declared for this initial diagnostic training.
Numerical acceptance margins must be frozen using real-only variability before
opening generated quality results if a confirmatory decision is later needed.
Development reuse and all failed/unfinished fits must be reported. Improved
teacher loss alone is not success; free generation determines the research
question. CPAR, TabDiT, simple transition/duration baselines and other datasets
follow after this first implementation check. No test-set or paper-level claim
is authorized by the first pair alone.

## Execution safeguards

Before full fitting: causal-prefix tests, hand-worked state transitions,
full-history collation, native-B equivalence, gradients, checkpoint reload,
teacher-versus-stepwise state/logit equivalence, batch-filtering behavior, numeric
decoder invariance, and a GPU optimization smoke test. Code/config/data hashes
are recorded before first fitting and checked on every run. Installed engine and
the completed audit worktree are unchanged. Native training is adapted in memory
only to remove the 14-epoch small-customer cap and disable cuDNN benchmarking;
the resolved source is saved per run. Full-history collation and model class
substitution are explicit adapters, not silent edits to the engine installation.
