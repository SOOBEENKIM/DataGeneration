# Calibrated U/E matched-history replay v1 — 2026-09-20

Registered before implementation or new replay outcomes, after generation repeats commit `39525ff3fcf8e106dacb685c7d799b413b5a5a0d`. No training, calibration fitting, generation, checkpoint selection, seed expansion, or automatic subsequent experiment is authorized by this contract.

## Research question and claim scope

The working problem is selective preservation of gap–behavior dependence across observed groups, including rare groups: preserve needed relations while suppressing spurious ones, and verify that conditional prediction gains survive recursive generation. Both U and E already use strict-past history. E adds 32 history-head coefficients; it has conditional accuracy/null-response benefits but no demonstrated native generation advantage over equally calibrated U. The current task tests a mechanism relevant to that gap, not a new successful model or novelty proof. Post-fit probability calibration is a generic control, not a unique E contribution.

Prior cross-history work (2026-09-17) already compared U/E/L003 on smaller observed-first-event panels, before calibration. Its oracle controls showed that forcing observed gaps while resampling marks can itself distort the joint law. Here use only the latest fully self-generated calibrated U/E paths, including their generated first events. No new forced-gap generation and no oracle training targets are introduced. This is an extension and check of the existing diagnostic, not a new conceptual invention.

## Frozen inputs and all comparisons

Use Ucal and Ecal for all four prevalences (.05/.10/.25/.50), both kappa conditions (0/1), five existing paired training trials, and all five fresh generation repeats from generation_repeats_v1. Exactly 400 existing 2,048-entity paths and 80 frozen calibrated model states. Each target model evaluates each source path: 800 predictor/path evaluations, 1,638,400 sequence evaluations, zero newly generated sequences. Models, calibration buffers, original context/length/order plans, train metric bins and full-validation reference are immutable and hash verified. The repeated-generation original single tape is excluded.

Every cross evaluation receives the source path's actual gaps, marks, transformed numeric values, observed static codes, valid mask and lengths. For a given source, both targets see identical strict-past events and current gap. Target hidden states are recomputed with its own encoder; do not transplant source hidden states. First-event predictions are excluded from repeat metrics, but the generated first event is retained in all later histories. Copy/fresh coincidences are included: R=P(mark_t=mark_(t-1)), not latent copy q. Use normalized categorical probabilities from the same mark_distribution as native generation. Preserve full-mark distribution disagreement TV on matched inputs; TV between models is not accuracy against truth.

## Primary decomposition and link to native error

For each group, cell, trial and repeat, let C[a,b] be validation-bin-weighted L1 between target a's mean predicted repeat probability on source b paths and the observed validation repeat curve. Use the exact parent train-fitted five bins, reference counts/weights, masks and missing-bin convention. Also retain empirical generated repeat curves and native L1 L[b], independently recomputing and matching all 1,200 source/group native values.

With E=Ecal, U=Ucal:

F = ((C[E,E]-C[U,E]) + (C[E,U]-C[U,U]))/2
H = ((C[E,E]-C[E,U]) + (C[U,E]-C[U,U]))/2
D_expected = C[E,E]-C[U,U] = F+H
S = (L[E]-C[E,E])-(L[U]-C[U,U])
D_native = L[E]-L[U] = F+H+S

F compares predictors on matched inputs. H changes the source distribution of histories AND current gaps under fixed predictors; it is not pure mark-history causation. S is the signed empirical-minus-probability-score remainder, including nonlinear finite-sample effects; it is not an independent zero-mean noise variance and may be negative. C is an error of bin-averaged predicted probabilities, not the expected value of native empirical L1. No component percentage or unique causal parameter attribution. F compares all fitted weights and calibration values, not solely the 32 added coefficients. Cross inputs may be atypical under the target model.

Retain both directional predictor contrasts, both source contrasts, the interaction, all 2x2 matrices, signed bin curves and bin decompositions. Report pooled and both contexts under both kappas, including all three null conditions. Include transition- and entity-weighted matched-input model TV, absolute/signed repeat differences, and fixed early/middle/late steps (1–10,11–20,21–31; zero-based event positions). No hidden-state similarity metric or on-generated-label accuracy ranking.

## Statistics and decision rules

Primary diagnostic group: kappa1/context1; primary components F,H,S and D_native. Average the five repeats within each training trial before describing five paired trial effects. Preserve the parent nested statistics (df4 descriptive t interval, generation-only conditional MC uncertainty, within-repeat variation). Trials differ in model AND fixed sampling plans; prevalence conditions share data. No pseudoreplication, multiplicity-adjusted significance claim, new-data confirmation, or equivalence claim.

A signed component is called consistently material if abs(mean)>=.001 and >=4/5 trial means share that direction, in at least 3/4 prevalences with the same direction. This inherits the earlier diagnostic's descriptive magnitude screen; it is not an acceptable performance margin or publication gate. Report exceptions/intervals even if a screen passes. Matched-input TV has no pass threshold. Do not declare H irrelevant merely because its screen fails. Do not reinterpret old ER failures or calibration outcomes.

Interpretation: material F supports checking the predictor mapping on generated inputs; material H supports studying source composition/feedback (not just mark). Mixed/canceling results require reporting that simple cause attribution is unsupported. This experiment does not choose a model revision or prove that the problem is solved. Any revision requires a separately fixed control and prospective evaluation.

## Technical gates and evidence

Commit this registration before implementation, and implementation before execution. CPU tests verify decomposition including cancellation, empty-bin/reference handling, strict-past causality (changing current mark/value and later events leaves current output unchanged), categorical normalization and TV contraction. CPU and GPU gates on the same committed source compare batched replay with step-by-step encoder updates and prefix replay, both target models on both source samples; compare GPU with CPU within 3e-6 absolute tolerance. Check repeat probability from full distribution against copy-plus-fresh formula. Disable TF32, use one CPU thread and a single low-priority idle-GPU process with a 20% allocator cap; do not interrupt other jobs.

Production stores per-transition repeat probabilities for both predictors and their full-mark TV, path identity/hash, model/contract/source hashes, all scalar results and bin counts. On every cell check a fixed small prefix of each source/model against sequential replay, probability normalization and unchanged model/calibration state. Raw-array independent verification recomputes all native and probability-based curves, decompositions and summaries. Keep failures and document implementation corrections without changing scientific criteria.

Git versions registration/code, all scalar/bin results, plots and report. Raw source paths/checkpoints/prediction arrays remain on the workstation. No test data, real data, model fit or new independent data is accessed.
