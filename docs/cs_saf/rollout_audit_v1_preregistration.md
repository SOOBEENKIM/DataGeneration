# Fixed-checkpoint rollout diagnosis, registered 2026-09-17

This local registration precedes implementation and new diagnostic outcomes. Base: `65be9bdee506dfee8245f05e711840638b1f559e`. The user requested publication only if a useful, validated diagnosis is obtained; registration is committed locally first, without changing the published research branch. All registered results, including contrary findings, will accompany any eventual publication.

## Question and frozen scope

Why does lower observation-history oracle full-mark TV fail to ensure better generated repeat/gap dependence? Reuse best U/E/L003 checkpoints, all four prevalences, both kappas, all five existing trials (120 checkpoints). No fitting, coefficient change, checkpoint reselection, new dataset, held-out test, or latent-state targets. These are exploratory analyses of already-used training/validation data. Scientific improvement of a new method cannot be claimed from this audit.

## A. Endpoint and measure decomposition

On every validation history compute the existing oracle's full-mark TV and repeat absolute error both under the train context gap-bin reference and at the factual current gap bin. Exactly decompose TV into `abs(predicted_repeat - oracle_repeat) + excess_TV`. The nonnegative excess is the information lost when collapsing 64 marks to repeat/nonrepeat; it is not a uniquely identifiable fresh-head causal contribution. Report signed repeat bias, step profiles and teacher-forced repeat curves under the original five train-fitted metric bins. Match existing full-validation grid TV/repeat errors numerically before drawing conclusions.

For each saved native 2048-entity generation, reproduce the published repeat-curve L1 and gap/repeat MI. Re-evaluate probabilities on the actual generated histories. In each metric bin the exact signed identity is:

`generated_repeat - validation_repeat = (generated_repeat - mean_model_repeat) + (mean_model_repeat - mean_oracle_repeat) + (mean_oracle_repeat - validation_repeat)`.

These terms describe realized sampling residual, local conditional discrepancy, and composition/reference discrepancy. Absolute errors and MI are nonlinear: signed terms are not additive percentages of those errors. The last term also includes reference estimation and oracle representation assumptions, and must not be called a pure exposure-bias effect. Report both the original continuous-history oracle and a bin-observation-history oracle (past support-bin likelihood instead of representative-point exponential density); current-gap prediction remains integrated over the support bin. Null oracle filtering ignores gaps in both versions. This sensitivity is needed because generated gaps are support representatives while observed histories contain continuous gaps.

## B. Controlled prefix-anchored sampling

Select 128 validation entities per observed context without replacement with the registered panel seed, separately for each data cell; retain the same panel for all model trials and candidates. Use three registered independent sampling tapes per trial, paired across models and arms. Gap and mark draws use inverse-CDF uniforms and numeric values use an independent normal tape. Fix the first observed mark/value in all arms; use the same lengths and observed static contexts. This is a conditional prefix-anchored diagnostic, not a replacement for the published unconditional generation benchmark.

* OBS: observed continuous gap trajectory and observed numeric values; recursively sampled marks.
* QUANT: only replace that gap trajectory with fitted-bin representatives; keep observed values, recursively sampled marks.
* GAP: sample model gap bins and marks; keep observed numeric values.
* FULL: sample model gaps, marks, and numeric values after the fixed first event.

Compare teacher-forced expected repeat curves with OBS expected curves to assess dependence on generated mark history. Compare OBS to QUANT for representation sensitivity, QUANT to GAP for gap-trajectory sensitivity, GAP to FULL for numeric-feedback sensitivity. Expected curves average the model probabilities actually used along each trajectory; empirical curves and MI are also reported. Each evaluation uses the full validation reference, train-only metric bins, equal panel size, per-step statistics, and both context labels. Report teacher forcing with quantized past gaps separately while keeping current gap coding and all real marks/values unchanged.

Input substitutions are distribution-shift sensitivity diagnostics, not causal DGP interventions, and arm order is not an interaction-free factorial attribution. Supplying an observed entire gap trajectory is not equivalent to sampling the true joint distribution or conditioning optimally on future gaps. Never label an arm as an improved deployable model. Three sampling repetitions share one trained model and are averaged within trial; they are not 15 independent model seeds.

## Predeclared descriptive evidence screen and publication rule

All 120 checkpoints and registered cells must complete; hashes, reserved marks, finite values, strict-past behavior, probability/TV identities, original metric reproduction, and frozen-parameter checks must pass. For active context (kappa=1, label=1), assess E-U and L003-E separately. A useful diagnostic screen passes if either:

1. **Endpoint mismatch:** lower grid full-mark TV coexists with higher grid repeat absolute error, or higher factual full-mark TV, in at least 4/5 paired trials at at least 3/4 prevalences, with both mean directions at least 1e-4 in magnitude; or
2. **Rollout-stage sensitivity:** for any one of the four fixed adjacent stages (TF→OBS, OBS→QUANT, QUANT→GAP, GAP→FULL), the stage changes the paired candidate-minus-comparator expected repeat-curve L1 by the same direction in at least 4/5 trials and at least 3/4 prevalences, with absolute mean effect at least .001 at each qualifying prevalence. Both directions and all stages are reported, with all null contexts shown as controls.

These thresholds are descriptive consistency/materiality screens over multiple exploratory contrasts, not multiplicity-adjusted hypothesis tests, independent data confirmation, or proof of a unique cause. Report five-trial paired descriptive t intervals without interpreting them as confirmatory. A screen pass supports publishing a bounded diagnostic insight, not claiming that model quality improved. If no registered screen passes, preserve local outputs and tell the user; do not push. No additional seeds, thresholds, modes, or parameter searches to obtain a pass. Technical defects may be corrected with a recorded amendment before rerunning affected diagnostics; scientific scope must remain fixed.

## Execution

Commit the implementation and CPU checks before scientific GPU evaluation. Use one available GPU without terminating other jobs. Preserve original checkpoints/data; record source/config/input hashes, environment, panel identities, seeds, checkpoint tensor digests, output checksums, technical failures, and runtime. Keep full per-bin/per-step evidence and publish compact numerical summaries plus a Korean interpretation if the screen passes. The existing failed ER accuracy gate and previous report remain unchanged.
