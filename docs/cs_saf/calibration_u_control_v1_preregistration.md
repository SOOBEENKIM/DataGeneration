# U calibration control v1 — registered 2026-09-20 before new fits

## Question and scope
Does the added history correction in E retain a generation benefit when U receives the same train-only probability calibration? The preceding E/L003 study left this control missing. Its results and failed consistency screens remain unchanged.

This is a bounded exploratory completion of the existing comparison, not an independent confirmation. Reuse all four prevalences (0.05, 0.10, 0.25, 0.50), both kappa conditions (0, 1), five trial indices (0–4), existing U/E checkpoints, and existing Ecal fits. Add exactly 40 Ucal fits; do not retrain base networks, select seeds, tune calibration, or fit to validation/oracle targets. The same train data were already used to train the bases; their checkpoints were selected using the existing validation split. Test remains untouched.

## Equal intervention
For observed context s, apply q'=sigmoid(a_s+b_s z_U), b_s>0, to U's original copy logit. Use the identical four-scalar calibration, observed-repeat likelihood R'=q'+(1-q')f(previous), training transitions, feature precision/batch size, optimizer, bounds and identity initialization as Ecal. No history correction parameters are added to U. Fresh-mark, gap, value and history-encoder weights remain frozen. First-event fresh probabilities remain unchanged. Generated marks still affect subsequent history and therefore subsequent gaps/values.

Use L-BFGS-B with a in [-8,8], b in [0.05,20], initial [0,1], maxiter=500, maxfun=2000, maxls=40, ftol=1e-12, gtol=1e-8. Fit one offset/slope pair per observed context, using all nonfirst training transitions. Fit convergence, finite parameters and nonincreasing train NLL are mandatory. Report bounds hit. No oracle active labels or targets enter fitting.

## Endpoints and comparisons
Compare U, Ucal, E, Ecal on the exact existing train-derived 2,048-entity generation plan, trial sampling seed, and batch size 256. All gaps, marks and values are recursively sampled. One generation tape per trial is reused; sampling variance is not independently estimated.

Primary endpoint: active-context native generated gap-bin repeat-curve L1 (lower better, kappa=1). Primary contrast: Ecal minus Ucal. Additional contrasts: Ucal minus U (generic calibration), Ecal minus E, E minus U, and the factorial interaction (Ecal−E)−(Ucal−U). Report every prevalence and all five paired trial differences, mean/SD and descriptive paired 95% t intervals; no pooled significance or multiplicity-adjusted confirmatory claim.

Secondary endpoints: native gap-repeat MI error, mark transition TV and all existing generation metrics; conditional full-mark TV; unwanted copy/repeat response, averaged equally over the three null cells (kappa=0 contexts 0 and 1, kappa=1 context 0). A negative interaction means calibration helps E more on that metric; it does not by itself prove Ecal better than Ucal.

Retain the prior exploratory consistency screen: negative primary mean and at least 4/5 negative paired effects at at least 3/4 prevalences. The stricter joint lead additionally requires nonpositive mean MI-error and both null-response differences at those prevalences. These are research prioritization screens, not validated application tolerances or final superiority criteria. Report useful improvements and costs even if a screen fails. Do not revise thresholds after results.

## Validation and execution
Before the grid, CPU tests and same-source CPU/GPU gates must demonstrate identity calibration preserves original U likelihood and seeded recursive samples, no history parameters are introduced, fitted original tensors stay unchanged, fitting is deterministic, and checkpoint reload preserves results. Fit first, save calibrated state, then evaluate oracle accuracy. Apply the already documented full-precision duplicate-response verification while preserving historical fit/generation precision. Independently recompute native repeat L1 and MI from saved arrays; verify all reference/output hashes and paired plan identities.

Use at most two currently available GPUs, never terminate unrelated tasks. Preserve incomplete attempts and stop dispatch on technical failure; scientific failures do not stop the grid. Save all outcomes, including negative results, to research/cs-saf. No automatic merge into main.

## Interpretation fixed in advance
If Ecal is consistently better than equally calibrated U, the history correction has a stronger exploratory case; confirm with new data/training seeds and additional sampling tapes before a methods claim. If not, attribute demonstrated gains to calibration where supported and do not claim the E structure improves native generation. Retain Ucal as a simpler candidate if appropriate. Any remaining conditional-vs-generation gap motivates a separately registered diagnostic, not automatic architecture search. Earlier regularization tradeoffs and failed gates remain part of the record.
