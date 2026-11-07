# Frozen history-representation intervention — 2026-09-29

Authorized question: does the real-prefix/generated-prefix discrepancy in the
selected mixture_marginal head explain its timing costs? No fitting, tuning,
checkpoint replacement, push, or sealed-test access is permitted in this study.

## Important input decomposition

Source inspection establishes that the 1322-dimensional head input is NOT all
history: 256 static-context features + 832 strict-past LSTM features + 234 masked
current-column features. The last part includes the available structural,
label/category embeddings. Earlier descriptions of the whole vector as a
history representation were imprecise. Primary interventions change ONLY
columns [256:1088]. All other features, explicit phase, clock, position, validity,
head weights, clipping/quantization and random draws remain fixed.

## Source rationale and scope

Lin et al. 2026, https://arxiv.org/html/2604.27182v1 §III-B, distinguishes the
conditional law from the distribution of its conditioning inputs. We use this
distinction, not its MCMC procedure or an unconditional stability guarantee.
He et al. EMNLP 2021, https://aclanthology.org/2021.emnlp-main.415.pdf §4–5,
uses real/model-prefix interventions to measure exposure effects; a discrepancy
does not by itself prove large deterioration. Neither text-domain conclusions
nor a claim of inevitable accumulation transfers to our transaction experiment.
The present work is a diagnosis of an existing extension, not a novel method.

## Frozen subjects and paths

- One frozen ARGN parent 20260930, selected mixture_marginal step400 and init_only.
- Two native instrumentation generations, seeds20261011/20261012, CPU device.
  Four GPUs were externally busy at registration. These are NEW CPU paths;
  do not present their values as exact replays of the previous CUDA outputs.
  Native all-field tokens and the head's live inputs/noise are captured before
  decoding, with explicit step_keys IDs before batch compaction. This avoids
  irreversible re-encoding of decoded generated records.
- Actual development records and recorded customer-ID/record mapping only.
  Common customer-position support uses min(actual, generated length).
- Actual-history literal: training's actual encoded structural schedule.
  Actual-history aligned: same actual past transaction tokens, but generated
  SLEN/SIDX/SDEC tokens; this controls sequence-length information in the LSTM.
  The aligned intervention is primary; literal is an explicit sensitivity.
- Generated histories must agree with an independent teacher-forced LSTM pass
  over captured native tokens within float32 tolerance. Static context equality,
  strict-past alignment, current-column freezing and native sampled-gap parity
  are mandatory. No re-encoding approximation will be silently accepted.

## Conditional comparisons

On both captured paths, hold either generated or actual current conditions
(phase, explicit clock, current-column features) fixed. Within each anchor,
compare generated history, actual aligned history, and actual literal history.
Use the identical four independent paired sampling seeds20261501–20261504 for
each row and condition. Also reconstruct original native samples with captured
noise. init_only is a negative control: changing only history must have ZERO
effect on its law and samples. First five interval laws must also be invariant.

Primary interpretation concerns normal timing at6–49 and100–199, plus all
noninitial normal events and1024+. Report fraud onset, continuation and return
only with support counts. Evaluate raw log-gap W1, customer-mass-weighted W1,
mean signed log-gap bias, and log(1+gap/strict-past raw median) W1. Retain all
phase disagreements: report broad phase-conditioned population comparisons
separately from a same-real/generated-phase subset. Do not score a mismatched
fraud target as if it were a paired prediction. Conditional NLL against actual
gaps is reported on the actual-condition anchor, not the hybrid generated one.
Also report changes in logits/means/scales, standardized representation distance,
and the share of the residual head's excess error removed by a history swap
when the relevant denominator is positive. Four noises are sampling repeats,
not independent trained models; two paths are not independent refits.

## Decision and limits

An input-shift explanation is supported only if replacing history consistently
reduces relevant costs on both paths under the SAME anchor, and improvements
are appreciable relative to the residual-vs-init_only excess. Direction changes,
small effects and remaining costs must be reported. If effects are small, next
examine likelihood weighting, objective approximation and training/selection
budget, rather than adding another stabilization formula. A substantial effect
would justify a later separately registered generated-history training control.
No threshold selects a new model here. Fixed-prefix conditional resampling is
NOT an intervened full recursive generation or deployable performance. Actual
prefixes are diagnostic oracle inputs; this does not establish a unique cause,
general ARGN limitation, novelty, or superiority to recent generators.
