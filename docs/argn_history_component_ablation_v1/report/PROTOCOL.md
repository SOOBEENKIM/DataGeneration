# Frozen history-head component ablation — 2026-09-30

User authorized the next initial-only/calibration-only comparison after argn_init_validation_v1. This is a fixed-weight 2×2 removal experiment, not a new method fit. Preserve active clock_prefix, all old candidates, source, weights and outputs; no commit/push/promotion/sealed-test reads.

Evidence: the previous same-treatment history model improved normal timing .063958→.052395 and personal fraud-onset timing .120572→.080343 against init_only, while normal-personal timing worsened .006468→.035996. Initial law and calibration were combined; their separate effects on these benefits/costs remain unidentified.

Closest prior methods remain known conditional mixtures (IFL-TPP2020; MoTPP2026 §4.2) and temporal-fidelity evaluation (Seq2Synth2026 §3). We rechecked https://arxiv.org/html/2609.21382v1 and https://arxiv.org/html/2607.15606v3 at the method/evaluation sections. History conditioning and mixture outputs are not novelty. Full native sampling, rather than teacher-forced likelihood alone, determines the current comparison. This neither reproduces official MoTPP nor establishes superiority to recent whole generators. No new literature-based formula is introduced.

## Frozen inputs and four cells

Use the six selected history heads from prior time-fit seeds73001/73002/73003 × frozen native ARGN parents20260930/20261001. Keep existing generation draws20261011/20261012 and existing147 development customers. Existing development is reused; this is mechanism/robustness exploration, not a new confirmatory test. Prior three TIME-fit seeds remain the fit units; new generation is not new refitting.

| Cell | Initial law I | Calibration C | Source |
|---|---|---|---|
| H: history_only | off | zero | reuse12 prior generations |
| I: history_initial_only | on | zero | 12 new generations |
| C: history_calibration_only | off | frozen fitted values | 12 new generations |
| IC: history_init_cal | on | frozen fitted values | reuse12 prior generations |

Also reuse12 matched init_only generations as a reference; they are outside the history 2×2. Total24 NEW full native generations and60 evaluated outputs including36 reused. New neural/GMM/calibration/ARGN/CPAR/latest-generator fits all0. Deriving a state dict by toggling a buffer is not training.

Derive both missing cells from each prior history_init_cal payload. I changes ONLY correction→zeros; C changes ONLY use_initial→False. All other tensors, including the exact neural history head, initial mixture and clock limits, are preserved. Verify prior H differs from IC only in those same two buffers. No calibration refit after removal: this asks what the already-fitted components do, not the best attainable separately refitted variants.

I replaces the entire distribution for normal-stay positions1–5. C shifts component means only for normal-stay, with the existing six-basis coefficient family and ±.35 bound. In IC, the initial replacement overrides C at positions1–5. Consequently, the two interventions are not additive at the conditional-law level; metric interaction is descriptive, not proof of a new synergy theorem. The first event remains native.

All four arms use their OWN recursively generated native history and clock. Label/length sequences must remain exactly paired; category/amount/merchant parameters stay fixed but their output may change through generated history. Those effects are measured. No true-clock oracle, future timestamps, test reads, or unrelated synthetic-prefix/real-next-event target pairs.

## Checks before generation

Freeze source/protocol/evaluator/head hashes before new outputs. Verify all prior395 audited files and all84 prior canonical generation hashes plus reused heads/reports. On every one of six head pairs, check neural tensor identity; conditional-law equality for the cells that should coincide; I has no direct effect outside its first-five normal scope; C preserves weights/scales and non-normal laws. Check native adapter/strict-past clock/reordering and exact labels/lengths in every generation worker, using the previous frozen implementation.

## Evaluation and interpretation

Keep previous metrics and units unchanged: primary full normal log1p-gap W1 and personal fraud-onset gap W1. Explicitly report the previously observed normal-stay personal-gap cost; it remains an auxiliary dimension and does not replace a failed primary target. Also report all6 position bins, customer-equal normal timing, whole fraud time, amount errors, merchant TV and diversity. Publish rare-event support. Recompute raw metrics with the previous independent evaluator and compare preserved existing values and per-worker values.

For each paired fit/parent/draw, report I−H, IC−C (initial effect without/with C); C−H, IC−I (calibration effect without/with I); IC−H; and IC−I−C+H. Smaller error is better. Report every time fit, parent, draw, and grand mean; do not treat the12 draws as12 independent fits or generated rows as independent evidence. Negative interaction on the W1 error scale is not a structural necessity/novelty claim.

The practical descriptive bands remain10% improvement in one primary with the other≤+.01; position costs+.02; amount/merchant+.01; diversity loss5%. Compare IC against I and C as well as H. Additionally list normal-personal changes without inventing a post-hoc pass threshold. Compare each cell with init_only, retaining parent-specific tail costs. No automatic model selection from these screens. If one component alone explains the effects, report that rather than requiring both by construction. A limitation is not itself a no-publication verdict.

## Execution

Use only idle host GPUs, one worker/device, fresh utilization/process checks before dispatch. Persistent tmux with a single-instance lock. Two generations/worker,12 workers. Do not restart a duplicate if waiting. Independent CPU evaluation follows generation automatically. Preserve failed executions and add an explicit amendment on implementation failures; do not silently rewrite this protocol or old source.
