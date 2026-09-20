# CS-SAF recoverable research state

Updated 2026-09-21. Current branch: `research/cs-saf-external-audit-v1`.

**Latest experiment COMPLETE: [external Berka/Sparkov relationship and preprocessing audit](external_relations_v1/README.md).**
Registered `e9e61d2`, science source `6726a8e`. Raw/canonical equality checked on
1,990,071 development events. Seven unit checks and independent reductions of
270 profile rows / 40 prediction rows pass (maximum difference 3.34e-14).
CPU ~39.73 seconds on `finx-System-Product-Name`; empirical frequency/mean tables
fit, but zero neural fits, zero new generations, no test outcome analysis.

- Berka: observed-gap-conditioned transition table validation NLL 1.360426 →
  1.239030 (8.92% gain), internal check 9.28% gain. Only 39.04% of validation
  adjacent pairs are unambiguous. Full-pair operation repeat changes from 25.54%
  to 21.31% under reversed unknown within-day order. Require order-invariant
  daily evaluation alongside unambiguous transition results.
- Sparkov: same-merchant repeat only 0.204%; same-category repeat varies
  14.12/12.85/10.12/4.01/8.50% across train-defined gap bins and agrees with train.
  Merchant-conditioned amount log-MAE improves 18.58%. Sparse merchant transition
  tables worsen NLL: 6.4927 → 7.0539 → 8.3999. Post-run coverage: 61.15% of
  validation previous-merchant/gap/current-merchant triples were unseen in train.
  Do not infer absent temporal signal or new-model necessity from that baseline.
- Existing U/G requires two contexts, no auxiliary fields, lengths <=32;
  98.37%/91.84% of Berka/Sparkov validation entities exceed that length. External
  input/sequence handling needs an explicit, tested port before neural comparison.
  Never create fake groups or truncate full trajectories silently. Exclude
  whole-trajectory `entity_any_fraud` from predictors. Seven Sparkov merchant
  names have multiple categories; category is not a deterministic lookup.
- [Next comparison contract](external_relations_v1/next_comparison_contract.md):
  prepare common external schema/evaluator and U_ext/G_ext ports, then register
  budgets and compare official ARGN, CPAR and adequate simple controls. No new
  architecture, external neural result, fraud utility or contribution claim here.
  Failed prior candidates remain failed. Results are on this branch, not main.

**Previous review COMPLETE: [dataset literature and research-process audit](dataset_literature_and_process_audit_2026_09_20.md).**
Fourteen original papers were checked for datasets/evaluation, and 22 completed
CS-SAF result bundles were inventoried (not 22 architecture changes). Controlled
simulation remains useful for diagnosis; current-method external relevance and
utility are unproven. Prioritize problem validation on Berka/Sparkov before
another state/gap correction. Existing materialization is documented, but remote
raw-file availability/hashes were not revalidated in this review. Berka marks
encode operation/type, not counterparties. No training or new evaluation here.

**Previous experiment COMPLETE: [frozen history/run-state diagnostic](history_diagnostic_v1/README.md).**
84 fixed-model/history replays, 36 stored U/G+A generation datasets and 120 stored
oracle datasets; zero fits/new generation, CPU only. Active real-history run-2/3
repeat bias is -4.66/-4.51 percentage points for U/G. Generated continuation is
54.14/54.36% versus oracle 64.26%; all three parent seeds show the discrepancy.
Long-prefix mean-bias screens do not flag progressive collapse; long-run generated
minority cells lack coverage. U/G cross-history errors are similar. Mapping and
history composition both contribute descriptively, without causal attribution.
106 tests and 13,608 independent reductions pass. Existing failures stand; no
external-data superiority, independent confirmation or fraud utility is shown.

**Previous follow-up COMPLETE: frozen U/G with A (observed-history calibration),
B (generated-relation calibration), and P (prediction-protected relation calibration).
24 bounded searches / zero neural refits; 108 model and 120 unique oracle datasets.
No B/P candidate passes the joint preregistered criteria. Stop this recipe.**
[Korean result and decision](rollout_calibration_v1/README.md),
[tables](rollout_calibration_v1/result_tables.md),
[Registration](rollout_calibration_v1/preregistration.md),
[methods](rollout_calibration_v1/methods.md),
[execution notes](rollout_calibration_v1/execution_notes.md),
[verification](rollout_calibration_v1/verification.json).

- Registration `13dc0ac`, fitting implementation `027bc55`, CPU/GPU gates `429172e`.
  Existing seed42 / pi .10 / kappa0/1 / three parent training seeds. Frozen weights,
  same ten correction parameters, sequential feedback; train-only observed targets.
  All 24 selected corrections precede every final model/oracle evaluation.
- Active generated L1 U A/B/P=.018815/.017339/.018857;
  G A/B/P=.019031/.017257/.017133. Improvements 7.85% / -0.22% / 9.32% / 9.97%.
  G/B and G/P improve in 3/3 seeds, U/B in 2/3, U/P in 1/3. All fail the registered
  mean 10% AND .002 improvement; G/P gain=.001898, close but below both thresholds.
- Prediction costs, basic eligibility, other-distribution costs and null generated
  curves pass. Maximum Brier cost .001064 (<.002), mark NLL cost .002873 (<.02).
  Null fixed-history response fails separately: max increases .023367/.016865/
  .024921/.024921 vs limit .01 for U/B, U/P, G/B, G/P (1/9,1/9,2/9,2/9 failures).
- All 12 selected B corrections already satisfy P's training prediction limits.
  B/P outputs are identical for 9/12 parents. No essential protection contribution.
- Same-size continuous-oracle active L1 mean .015876; single-generation 5–95% MC
  range .009115–.022901. BIN/raw .014855; BIN/quantized-reference .014048.
  These are finite-sample references, not lower bounds, equivalence tests, or an
  error decomposition. New A tapes must not be called an improvement over old A.
- Verification PASS: 324 conditional values max difference 4.54e-10; 1,368 repeat
  values 9.98e-17; 468 representative native values 1.25e-16. Hashes/plans/guards
  and pre-evaluation fitting barrier verified. No fitting retries or extra settings.
- One oracle metadata-column collision affected the first evaluation in both
  kappas. Failure artifacts preserved; same seeds rerun after serialization fix;
  first-sample hashes identical. No new model fit/generation. All jobs ended.
- Retain U/G+A; previous C and E/ER failures unchanged. No external/independent
  expansion in this stage. The subsequent history/run diagnostic completed;
  see the latest experiment above. No claim of method novelty yet.
- Full outputs (~152MiB) stay at `artifacts/cs_saf/rollout_calibration_v1/`;
  compact code/results/figures/optimization traces are on this branch, not main.

**Previous Stage 2 COMPLETE: matched U/G/C structure comparison. 18 fresh neural fits,
162 new generated datasets, 331,776 sequences. Primary candidate C FAILS; stop expansion.**

[Korean result and decision](structure_v1/README.md),
[all result tables](structure_v1/result_tables.md),
[methods](structure_v1/methods.md), [registration](structure_v1/preregistration.md),
[artifact verification](structure_v1/verification.json),
[independent arithmetic](structure_v1/independent_arithmetic.json).

- Same existing artificial seed42, pi=.10, kappa0/1, three new paired training seeds.
  All U/G/C have 133,549 parameters, identical initial tensors and shared epoch
  orders, base NLL, budget and selection rule. No rollout/auxiliary/repeat loss.
- U retains the old copy mixture; G directly predicts observable repetition;
  C uses the same tensors with a fixed-history marginal constraint. C is not E/ER.
- Raw/gap/direct on every parent, train-only controls; three identical generation
  tapes and the same 2,048-entity length/group plan. C/raw is primary; posthoc C/gap
  generally breaks the original rho constraint and cannot rescue its primary fail.
- Active generated L1 C/raw .055305 vs G/raw .028320 and U/gap .026435. C is worse
  in 3/3 seeds against both (+95.3%/+109.2% means); all C seed L1 values exceed .03.
- Active Brier C/raw .146700 vs U/gap .141890; each seed exceeds the allowed .002
  cost against both controls. NLL cost passes. Conditional cost passes 18/24 cells,
  basic prediction eligibility 24/24, distribution costs 24/24, null generation 9/9.
- C/gap L1 .039241 remains worse than G/gap .026150 and U/gap in all seeds.
  G/gap vs U/gap difference is only -.000285 with mixed directions; no G superiority.
  Direct U L1 .017730 trades off Brier .185631. Do not assign this to C's contribution.
- Null fixed-history sensitivity is higher for C than G in all three null cell means;
  passing null generated curves is not evidence of reduced spurious sensitivity.
- C constraint max error 2.39e-7; all initial states/orders/input/checkpoint/generation
  hashes and train-only calibration optimum checks pass. Independent recomputation
  of 324 prediction values / 972 generated relation values passes (max 4.91e-7/1e-16).
- Preregistration 892a5d1, scientific execution 6afb05b4ec06b255d0cd9b74a4f9df8276ecbd9e.
  CPU test-index/test-design fixes and TF32 discrepancy were resolved before science;
  all three structures use explicit FP32. No scientific training/evaluation retries.
  All jobs finished; GPU training single admitted job, max reserved 412MiB.
- Do not adopt C as a final model/contribution or automatically add rollout loss.
  C already loses real-history prediction; marginal-head capacity and gradient
  coupling remain unseparated possible causes. U/G+gap remain the working controls.
  Any causal follow-up is unregistered/unexecuted. No independent data/test/real
  finance utility or new external comparison was performed in this internal stage.
- Compact results/code are on this branch, not main. Raw files (~440MiB) remain in
  artifacts/cs_saf/structure_v1. Earlier historical findings are preserved below.

**Previous Stage 1 COMPLETE: four bounded ARGN continuations and shared observable-repeat
controls on 12 parents; 144 generated conditions evaluated, 132 newly generated.**

[Korean report and decision](baseline_adequacy_v1/README.md),
[all result tables](baseline_adequacy_v1/result_tables.md),
[preregistration](baseline_adequacy_v1/preregistration.md),
[implementation/evaluation scope](baseline_adequacy_v1/methods.md),
[verification](baseline_adequacy_v1/verification.json).

- Same existing artificial data seed42, pi=.10, kappa0/1; two ARGN seeds and two
  existing U trials. No independent data, test, real-data utility or new-method training.
- Four continuations add 100 epochs each, restore optimizer/scheduler, and extend
  only stopping patience. Raw selected AND final checkpoints remain inadequate.
  Budget-contribution screen meets in only one seed per kappa, not both.
- Raw / group level / group-by-gap / empirical direct-repeat controls fit train only.
  The shared target is observed repeat probability, not U's latent copy variable.
  Native gap quantization, historical training budgets and length handling still differ.
- U+gap and continued ARGN+direct each pass conditional and generation/cost screens
  in 4/4 parents. Coarse relation matching alone does not require the proposed module.
  Direct is a statistical control, not an implemented/trained neural repeat head.
- U+gap active generation L1 .027880 and Brier .141335; U+direct L1 .011075 but Brier
  .185631. This exposes the curve-matching / per-event prediction tradeoff.
  Continued ARGN+gap active prediction curve L1 .009524 but generated L1 .047604;
  marginal mark costs and repeat-level drift remain in several cells.
- All null centered-shape screens pass (max parent-mean .014981). Large null repeat
  error is not evidence of a large fake gap shape. Small borderline cost failures
  and substantial ones are separately reported, not hidden behind a binary label.
- Independent scalar optimizers agree within 9.92e-8; native replay difference 0;
  original inputs/states unchanged and all hashes checked. Execution implementation
  is a03739e, preregistration 0672e82. All GPU and CPU stage-one workers finished.
- Stop registered Stage 1 here. Any next neural-head / constrained-coupling comparison
  needs matched capacity, likelihood and calibration, plus prospective event-prediction
  cost limits against U+gap. No rollout loss is automatically added. Simple-control
  gains cannot be reassigned to E or the new coupling; old negative results remain.
- This branch contains the preceding external audit too. Neither this extension nor
  its results are merged into main or research/cs-saf. Raw stage-one files remain in
  artifacts/cs_saf/baseline_adequacy_v1; Git contains compact reviewable outputs.

The following is the inherited U/E history from `research/cs-saf`; its findings are
preserved. Its historical "next" proposals are superseded by the current report above.

**Latest equal limited gap calibration COMPLETE: 80 unique corrections,
400 new native datasets (819,200 sequences), 400 reused controls. All GPU jobs
finished. Two interrupted fits were rerun identically after a metadata-only fix.**

[Full report](gap_calibration_v1_report_2026_09_20.md),
[professor brief](professor_brief_gap_calibration_2026_09_20.md),
[statistics, coefficients, verification](gap_calibration_v1_result.json),
[all scalar metrics](gap_calibration_v1_all_metrics.csv),
[registration](gap_calibration_v1_preregistration.md),
[execution amendment](gap_calibration_v1_execution_amendment_01.md).

- Registration `db76b85`, initial implementation `b9251b6`, all completed scientific
  execution `547bce0c0dde6a2c59d7513b0b020f010c9d87b8`. Four prevalences, both kappas,
  five existing training trials, five fixed generation tapes. No new neural fits.
- Same five train-support quantile bins per context for U/E; 8 new effective
  degrees, bounds ±.5, train-weighted centered logits, ridge .001. Every neural
  weight AND old affine scalar frozen. Only observed TRAIN repeat targets fit.
  No oracle/validation fitting, new continuous information or coefficient search.
- Ugap-Ucal and Egap-Ecal native active L1 improve in 5/5 trials at EACH prevalence.
  U reductions 11.0/8.4/10.6/21.7%; E reductions 17.5/16.7/13.4/18.3%. Both primary
  improvement screens meet at 4/4. Active conditional grid/factual TV improves too.
- Egap-Ugap means +.001048/+.001583/+.000857/+.000640; E has lower trial means in
  1/5,2/5,1/5,1/5. Primary improvement 0/4; symmetric cost screen meets at 5/25/50%.
  All between-trial AND fixed-model conditional MC intervals include zero.
  Do not reinterpret directional screens as statistical superiority/inferiority.
- E retains lower mean active conditional TV and all three null-cell copy/repeat
  responses than equally corrected U. But correction increases ALL three null-cell
  means for BOTH parents at EVERY prevalence. U MI error increases at 5/10%.
  No joint lead, including own-parent improvements. Full 13-metric evidence retained.
- Claim revision: do not call E's added history structure a demonstrated contributor
  to native-generation improvement. Keep E's conditional/null benefits as a tradeoff.
  Generic gap calibration improves U too and is not by itself a novel contribution.
  E vs U compares all learned model states; it does not isolate 32 coefficients as
  the sole cause. Earlier ER failures remain unchanged; no overall final model chosen.
- 6 CPU tests and CPU/GPU gates pass; original full 2,048-sequence plans reproduced.
  80 old states unchanged; independent train rows/bin maps/constraints/objectives,
  2,240 conditional means, 2,400 raw native L1/MI values pass, max discrepancy 0.
  All 40 U/E conditional reference measures identical; historical control means
  differ by 0. All 160 group optimizers converge, no bound hit, max |offset| .386088.
- Initial .05/k0/trial0 U/E workers stopped on tuple-vs-JSON-list comparison after
  first generation. Original output archived; JSON-normalized metadata check fixes
  only storage comparison. Identical offsets, checkpoints, conditional endpoints
  and first generated paths verified for both repeats. No later scientific retry.
- Reporting separates max of null-cell means from mean of trial-wise maxima.
  The registered joint screen still checks each cell separately; no rule changed.
  CSV 2,400 rows /31,200 values. Raw artifacts in gap_calibration_v1 and failed archive.
- Same artificial data seed42/explored validation/training trials. No test, real data,
  new external baseline or independent confirmation. Next proposal only: lock U/E
  with/without correction, justify endpoint priorities and any acceptable cost
  prospectively, then preregister independent data/training-seed confirmation.
  Do not keep adding correction capacity until E wins. Later compare external models
  fairly and evaluate behavior–fraud relations/real detection utility.
- Branch remains research/cs-saf, main remains earlier SAF base. Git stores reviewed
  code/contracts/scalars/plots/reports; raw checkpoints and paths stay on workstation.

**Previous calibrated U/E matched-history replay COMPLETE: 400 immutable native
paths, 80 frozen calibrated states, 800 predictor/path evaluations. No training,
calibration fitting or generation. All GPU work finished.**

[Full report](calibrated_replay_v1_report_2026_09_20.md),
[professor brief](professor_brief_calibrated_replay_2026_09_20.md),
[statistics](calibrated_replay_v1_result.json),
[all scalars](calibrated_replay_v1_scalars.csv),
[all bin curves](calibrated_replay_v1_bins.csv),
[registration](calibrated_replay_v1_preregistration.md).

- Registration `5f3c745`, all execution `035473d967d66e9aeff886aaac7dba7aa25a54c6`.
  Reuses both calibrated models and all 5 latest tapes, 4 prevalences, both kappas,
  all 5 paired trials. Both predictors receive exactly the same raw source path;
  each recomputes its own hidden states. Generated first events are retained.
- Active same-input predictor F means: +.003157/+.002620/+.001024/-.000234.
  Positive trial means 4/5,4/5,4/5,2/5; positive materiality screen met at 5/10/25%.
  Both directional source comparisons have the same trial counts. 25% is only
  slightly above .001. Between-trial intervals include zero at 5/25/50%, so no
  confirmatory all-condition inferiority claim.
- Source history/current-gap composition H: +.000476/+.001257/+.000569/+.000323.
  Empirical-versus-probability score remainder S: +.000108/+.001065/+.000339/-.000071.
  H and S each qualify only at 10%; their across-prevalence materiality screens
  fail. Do not interpret that as zero history influence or independent noise.
- F+H+S exactly recovers every previous native Ecal-Ucal contrast (max error0).
  F compares all learned weights/calibration, not just E's additional 32 parameters.
  H includes current gaps and all past generated variables, not pure mark feedback.
  Probability-curve L1 is not the expected empirical L1 or event-level oracle TV.
- Bin evidence: at 5/10/25%, E increases short-gap underprediction and long-gap
  overprediction on BOTH fixed sources; some middle bins improve. At 50%, the
  endpoints improve slightly. All null contexts/pooled groups and stage-specific
  matched-input full-mark TV are retained; model-model TV is not truth accuracy.
- CPU tests 6 PASS; CPU/GPU gates PASS, max CPU/GPU difference2.543e-7.
  Every cell's sequential checks pass, max3.688e-7. Independent verification of
  native L1 1,200, probability-curve L1 2,400 and 600 decompositions PASS,
  max3.469e-18. No technical failure/retry. GPU0 single low-priority process,
  allocator cap20%, 175 seconds, now finished without stopping other work.
- Git exports 1,200 scalar rows and 6,000 bin rows; raw arrays/checkpoints stay
  on workstation. Same artificial seed42/validation, no independent data/training
  seed, test, real-data or new external-baseline experiment.
- Research claim remains selective gap-behavior preservation across groups AND
  native generation. Generic calibration/history use or teacher-forcing mismatch
  is not a novel E contribution. Conditional gains remain; native superiority
  is unsupported. The report links relevant primary literature and limitations.
- Historical next proposal (now completed above): limited gap-dependent probability calibration applied equally
  to U/E, learned solely from observed training labels, with equal capacity/budget,
  original controls and joint conditional/native/null evaluation. It was fitted only
  to observed train labels, never to plotted validation biases or oracle law.
  If E adds no benefit over equally corrected U, revise the E-specific claim rather
  than repeatedly expanding coefficient/model searches. Existing ER failure stays.

**Previous fixed-model generation repeats COMPLETE: 800 generated datasets,
1,638,400 sequences, 160 frozen U/Ucal/E/Ecal states, zero neural/calibration fits.
All four prevalences, both kappas, five trials and five NEW generation tapes
completed. No generation failure/retry; GPU work is finished.**

[Full result](generation_repeats_v1_report_2026_09_20.md),
[professor briefing](professor_brief_generation_repeats_2026_09_20.md),
[all scalar metrics CSV](generation_repeats_v1_all_metrics.csv),
[statistics and verification](generation_repeats_v1_result.json),
[preregistration](generation_repeats_v1_preregistration.md).

- Registration `70ce09d`; all 800 scientific generations from `16bd9ac`.
  Models, calibration buffers, original 2,048-entity plans and batch size 256
  fixed. Only generation seeds change: 2026092100+100*trial+repeat (five repeats).
- Primary Ecal-Ucal mean L1 differences: +.003740/+.004941/+.001932/+.000019.
  Positive tape-averaged trial effects: 4/5, 5/5, 4/5, 2/5. Persistent-cost
  screen MET at 5/10/25%; benefit screen NOT MET. Relative costs approximately
  11.9/16.1/6.7/0.1%. At 50%, differences remain mixed; no equivalence claim.
- Generation-only conditional MC intervals exclude zero at 5/10/25%.
  Between-trial df4 intervals exclude zero only at 10%. Current fixed-model
  sampling uncertainty is distinct from uncertainty across models/data.
  Between-trial variance also includes trial-linked fixed-plan differences.
  Negative raw variance-component estimate at 10% is preserved, not interpreted
  as proof that true between-trial variance is zero.
- Ucal-U primary benefit screen MET at 5/10/25% (5/5,4/5,4/5,3/5 improvements).
  Ecal-E mean L1 improves everywhere, but consistency only at 25%. Both reduce
  MI error in 5/5 trials at each prevalence. No joint improvement screen passes;
  Ucal retains slight null-response and transition-TV costs. Old ER FAIL remains.
- Ecal's prior conditional TV/null-response benefits are unchanged fixed-model
  endpoints; they are not new repeated observations. The generated relation
  benefit of the extra history structure is unsupported by these comparisons.
- CPU unit tests 6 PASS; same-source CPU/GPU gates PASS, including exact full
  historical sample reproduction for four models. Cached 13-metric evaluator
  matches 1,248 historical values (max8.327e-17). Independent raw-array checks:
  2,400 groups / 4,800 L1 and MI values, max difference0; 800 saved model/seed/plan
  identities PASS. All 31,200 native metric values exported, no test accessed.
- Scope: reused seed42 data/validation/checkpoints. Five tapes are averaged
  within five trials, never counted as 25 independent trained models. Original
  tape is reference-only. Finite-sample metric bias/data uncertainty unestimated.
- Historical next proposal: cross-evaluate calibrated U/E on identical saved
  raw generated histories. This is now complete above. This is an
  algebraic diagnostic, not exclusive causal identification or permission to
  claim improved generation. Then independent data/training-seed confirmation
  and fair external comparisons if a justified candidate emerges.
- Raw evidence: `artifacts/cs_saf/generation_repeats_v1/`. Git stores reviewed
  source/contracts/scalars/statistics/plots; raw arrays/checkpoints remain on the
  workstation. Branch research/cs-saf, no main merge.

**Previous equal-calibration U control COMPLETE: 40 four-scalar U fits, zero neural
retraining, 120 reused U/E/Ecal references. All four prevalences, both kappas,
five paired trials completed without a technical retry. GPU jobs have ended.**

[Full result](calibration_u_control_v1_report_2026_09_20.md),
[professor briefing](professor_brief_calibration_u_2026_09_20.md),
[verified numbers and coefficients](calibration_u_control_v1_result.json),
[preregistration](calibration_u_control_v1_preregistration.md).

- Registration `de011db`; all 40 scientific cells from `c39313f`.
- Ucal-U active native L1 mean reductions: 6.8/14.6/9.4/0.3%, improving
  3/5, 4/5, 4/5, 3/5 trials. MI error improves 5/5 at every prevalence.
  Mean null response and mark-transition TV increase slightly.
- Primary Ecal-Ucal L1 means: +.002877/+.002542/+.001873/+.000326;
  Ecal improves only 1/5, 3/5, 1/5, 3/5 trials. All four descriptive paired
  intervals include zero. Registered consistency screen NOT MET (0/4).
  Ucal-U also misses it (2/4). No superiority/equivalence claim; old ER FAIL remains.
- Ecal retains lower conditional full-mark TV and null copy/repeat range than
  Ucal in 5/5 trials at each prevalence. Its conditional advantage does not
  establish a native relation advantage. The conditional TV interval includes
  zero at 50%. Ecal MI and mark-transition TV means worsen at 5/10/25%, improve
  at 50%; all endpoint vectors and factorial interactions are preserved.
- CPU tests 10 PASS (2 CUDA tests skipped), U-specific GPU environment tests 3
  PASS; same-source CPU/GPU gates PASS. Identity calibration exactly preserves
  original U likelihood and seeded samples; no history parameters added.
- 40 frozen-base identities, 80 fit-context objectives, 80 conditional groups,
  40 paired train-target/order/generation-plan/precision checks PASS.
  Independent native arithmetic: 480 groups, 960 values, maximum difference 0.
  Full-precision response checks: 40 PASS, maximum entity discrepancy 3.875e-8.
  Converged fits, no optimizer bound hits. No test access.
- Scope: existing seed42 data and reused validation; one sampling tape per
  training trial. This completes a missing control, not independent confirmation.
- Historical next proposal: fixed-model additional generation tapes. It is now
  completed above. Independent data/training-seed confirmation remains pending.
  Native differences cannot uniquely identify a mark-history mechanism.
- Raw evidence: `artifacts/cs_saf/calibration_u_control_v1/`. Git versions code,
  protocols, summaries, calibration coefficients and figures, not raw checkpoints.
  Research remains on `research/cs-saf`; no merge into `main`.

**Previous frozen copy-probability calibration COMPLETE. 80 four-scalar fits,
zero base neural fits, 120 reused U/E/L003 reference models. All four prevalences,
both kappa conditions and all five paired trials completed. One additional
technical attempt preserved. GPU jobs have finished.**

[Full result](calibration_v1_report_2026_09_20.md),
[professor briefing](professor_brief_calibration_2026_09_20.md),
[verified evidence](calibration_v1_result.json),
[preregistration](calibration_v1_preregistration.md),
[execution amendment](calibration_v1_execution_amendment_01.md).

- Registration `54811a1`; 23 complete cells from `b66c2f1`, 57 from `405b318`.
  The latter changes numeric verification, not fitting or reported science.
- Ecal-E active native L1 means improve 11.1/10.4/10.0/4.0%; improvement
  counts 3/5, 5/5, 4/5, 3/5. L003cal-L003 improves 10.6/7.3/13.8/6.3%,
  counts 3/5, 4/5, 5/5, 3/5. Both improve MI error 5/5 at each prevalence.
- Registered primary consistency screen NOT MET: only 10% and 25% qualify,
  fewer than three of four. Old ER FAIL remains. Mean improvement is not
  independent confirmation or broad superiority.
- Calibration slightly raises mean null response in both arms. L003cal still
  reduces null copy range vs Ecal 5/5 per prevalence, but native L1 and MI
  means are worse at all four prevalences. The tradeoff persists.
- Ecal native L1 vs U improves only in mean at 10/25%, and all four paired
  intervals include zero. Ucal was not run at that stage; the missing control
  is now completed above. Generic calibration is not a history-only contribution.
- 17 CPU tests and same-source CPU/GPU gates PASS; saved-state GPU regression
  PASS. The single TF32 response verification failure is archived. Full-precision
  control keeps the 1e-6 limit, while fit/generation/scientific scores retain
  historical numeric settings. The technical rerun's fitted parameters, model,
  generated arrays and conditional endpoints are identical to its first attempt.
- 80 frozen-base identities, 160 fit-context losses and conditional groups,
  paired generation plans verified. 600 native evaluation groups / 1,200 L1
  and MI values independently recomputed with maximum difference zero.
- Same data seed42 and reused validation; no test/real-data access or independent
  data confirmation. Native sampling uses one existing tape per training trial.
- Historical next proposal: U/Ucal/E/Ecal, followed by fresh data/training
  seeds and multiple paired generation tapes. The same-data missing control is
  now completed above; fresh-data and multiple-tape confirmation remain pending.
  If null suppression is the main aim, retain L003cal as an explicit tradeoff
  and justify an application-based active cost before a new confirmation.
- Raw evidence: `artifacts/cs_saf/calibration_v1/`. Git keeps reviewed source,
  protocols, summaries and figures, not the full checkpoint archive.

**Previous cross-history/joint-oracle diagnosis COMPLETE. 120 fixed models,
2,160 cross-history evaluations, 153,600 oracle sequences, zero new fits.
14 CPU tests, GPU replay gate and independent full-array arithmetic verification PASS.
GPU process finished; all registered outcomes preserved. Default main is still
`de8fa70` and does not contain this latest CS-SAF research.**

[Full report and tradeoff explanation](replay_oracle_v1_report_2026_09_17.md),
[professor briefing](professor_brief_replay_oracle_2026_09_17.md),
[complete contrast evidence](replay_oracle_v1_result.json),
[all binwise calibration evidence](replay_oracle_v1_binwise.json),
[independent verification](replay_oracle_v1_verification.json),
[preregistration](replay_oracle_v1_preregistration.md).

- Base `30f2d37`; registration `5eb7217`; scientific execution `6e022a7`.
  Same four prevalences, both kappas, five paired model trials and three tapes.
- Corrected interpretation: even exact online oracle marks lose joint fidelity
  when forced onto the observed gap path. Active FIX_CONT minus JOINT_CONT L1
  is +.0228 to +.0347, positive in every sampling trial at each prevalence.
  Binned controls agree. Oracle trials are sampling repeats, not learned models.
  The old numerical observation remains; it cannot uniquely diagnose a neural
  mark-history defect or exonerate the gap component.
- Cross-replay: same-history predictor F for L003-E is positive in 5/5 model
  trials at every prevalence for both OBS and FULL. FULL F +.00266 to +.00299.
  Source-history H lacks the registered three-prevalence consistency. This is
  an algebraic decomposition, not a unique causal attribution.
- The active repeat curve is flatter with L003 even on identical E histories:
  more short-gap underprediction and long-gap overprediction in mean. E itself
  also has calibration errors. Full binwise tables include all source/target/null cells.
- Tradeoff remains: three-null copy range improves in 5/5 trials per prevalence,
  while native active repeat L1 costs 2.6/6.7/11.7/11.5% and MI error costs
  7.8/9.1/13.2/18.7% relative to E. Active conditional TV intervals include zero.
  Retrospective required margins are not justified acceptance thresholds or PASS.
  Scientific diagnosis may proceed despite failures; prior ER FAIL is unchanged.
- Next bounded proposal: E/L003 with/without the same train-only repeat-probability
  calibration, checking joint generation and null-response costs, then independent
  data/seed confirmation if promising. Subsequently registered/executed above;
  that historical proposal is no longer the current unexecuted step.
- Raw evidence lives in the separate `research-cs-saf-replay-oracle` worktree under
  `artifacts/cs_saf/replay_oracle_v1/`, linked into the original research worktree.
  Git versions code/contracts/verified summaries, not a full raw-checkpoint backup.

**Previous fixed-checkpoint rollout audit COMPLETE: 120 saved U/E/L003 models,
zero new fits, 368,640 prefix-anchored diagnostic sequences, 1,440 input/output
checksum checks. Registered descriptive diagnostic screen PASS. GPU returned.**

[Full diagnosis and limits](rollout_audit_v1_report_2026_09_17.md),
[professor briefing](professor_brief_rollout_audit_2026_09_17.md),
[compact numerical evidence](rollout_audit_v1_result.json),
[local-before-execution preregistration](rollout_audit_v1_preregistration.md).

- Base `65be9bd`; registration `fd78ae5`; final execution source `fadf272`.
  Four prevalences, both kappas, five paired model trials, three sampling tapes.
- With observed gaps/values held fixed, replacing real marks with recursively
  generated mark histories increases paired expected repeat-curve L1. The
  registered materiality/consistency screen passes E-U at 5/10/25% (not 50%),
  and L003-E at all four prevalences. The 25% stage-effect t intervals include zero;
  this is a descriptive exploratory screen, not confirmatory significance.
- Full-mark TV, grid repeat error and factual TV all improve in active mean for
  both pairs at all prevalences: the endpoint-mismatch screen does not pass.
  Yet L003 worsens aggregate real-history repeat-curve calibration at 25/50%.
- Quantization, own-gap and own-value stages do not pass the cross-prevalence
  screen. Some individual effects remain; these components are not exonerated.
  Null controls have mixed effects; active-exclusive sensitivity is not proven.
- Observed-gap/generated-mark paths are hybrid inputs, not an exact joint-DGP
  causal control or an exact conditional generator given all future gaps.
  Model-specific history occupancy remains confounded with the conditional map.
  No unique root cause, repaired model, new external advantage or independent
  data confirmation is claimed. The original ER accuracy FAIL remains unchanged.
- CPU 9 tests, GPU 12 functional comparisons and all 120 scientific jobs pass.
  First CUDA cumsum failure and TF32 replay correction are recorded in two
  amendments, with original tolerances and scientific comparisons retained.
- Next proposed diagnostic: cross-replay shared generated histories and include
  a proper joint-oracle rollout control before choosing a bounded training fix.
  These next experiments subsequently completed above. Their joint-oracle controls
  qualify the interpretation of this historical audit; its original numbers remain.

**Previous follow-up COMPLETE: 170 new internal GPU fits + 30 reused fits, 40 CPAR
fits (128 epochs each), 200 matched internal generation evaluations, and
40 saved-checkpoint gradient snapshots. All four prevalences and all lambda
points are reported; the original replication FAIL is preserved.**

[Full result and failure analysis](followup_v1_report_2026_09_17.md),
[professor briefing](professor_brief_followup_2026_09_17.md),
[verified evidence](followup_v1_result.json),
[preregistration](followup_v1_preregistration.md).

- Pi=.05/.10/.25/.50, two kappa conditions, five paired trials. No selected-lambda
  expansion: all U/E/.003/.01/.03 cells completed regardless of scientific direction.
- E improves active conditional grid TV vs U in 5/5 trials at each prevalence.
  E adds 32 history coefficients; this is not a regularization-only contrast.
- Lambda .003 passes the original mean-based diagnostic screen at all four
  prevalences. Active E-relative means: −.0001548/−.0004432/−.0006499/−.0008660,
  improving in 2/4/3/4 of five trials. Every descriptive 95% t interval includes zero.
  Do not relabel these mean-screen PASS results as robust superiority.
- Lambda .01 fails the same screen at 5/10/25%, passes at 50%; .03 worsens active
  mean TV at every prevalence. Fixed epoch 9 preserves the strong-penalty cost.
- Free generation reveals a remaining problem: .003 worsens active repeat-curve
  L1 and gap-repeat MI error vs E in mean at every prevalence, and in 5/5 trials
  for both metrics at 10/25/50%. Conditional oracle improvement is not an
  established improvement to the complete sequence generator.
- Every internal arm beats the pinned CPAR execution in 5/5 trials at each
  prevalence on the three active-context primary generation metrics. U does too:
  this does not attribute the CPAR-relative difference to the new penalty.
  CPAR has 11,370 neural parameters vs E's 133,581; optimization and selection
  budgets differ. The pinned continuous-target alignment convention is disclosed.
  One external family does not establish general sequential SOTA superiority.
- Shared-feature penalty gradients affect history indirectly, while direct
  gap-route fit/penalty conflict is also present. The snapshot geometry does not
  identify a unique historical cause or prove that feature separation will help.
- 2,500 artifact checksum entries, 400 checkpoint identities, 1,600 aligned
  array groups and all paired summaries verified. The versioned verifier
  regenerates identical JSON. No missing snapshots. Two PNG/PDF figures reviewed.
- Registration 0bbd049; internal sources fccc7bb (20), 926b505 (20), e408966 (130).
  Gradient jobs use 926b505, CPAR uses e408966. Three early failed gradient
  attempts and both implementation amendments are preserved. GPU availability
  was checked before each launch; unrelated later GPU allocations sometimes
  overlapped, so uninterrupted exclusivity or clean speed comparison is not claimed.
- Same DGP/data seed42 and reused validation. No new-data confirmation, test or
  real-data access. The fixed-checkpoint generation diagnosis is now complete
  above; evidence-led model changes and stronger independent confirmation remain
  unexecuted.

**Historical fixed U/E/ER replication COMPLETE: 30 fresh GPU fits at pi=.05,
five new paired training trials. ER response PASS 5/5; registered accuracy
screen vs E FAIL. Conditional 90-fit prevalence expansion NOT STARTED.**

[Full report](replication_v1_report_2026_09_16.md),
[professor briefing](professor_brief_replication_2026_09_16.md),
[machine-readable evidence](replication_v1_result.json),
[preregistration](replication_v1_preregistration.md),
[figure PDF](replication_v1_pi_0.05_seed_results.pdf).

- Registration `8f5eaef`; CPU/GPU source `e35c6c275eec2b8ae13824d73aa3abee2865579e`.
- Frozen U/E/ER models, no lambda/architecture search or warm start. Five new
  model/order seeds, bank seeds and sampling seeds; same initial distributions.
  Existing data/validation reused; discovery seed excluded from new aggregates.
- 135 tests and six tiny deterministic CPU fits PASS. Remote GPUs 0/1/2/3 checked
  free before each job; all 30 training/evaluation jobs completed without technical failure.
- Response PASS counts: **U 2/5, E 1/5, ER 5/5**. ER worst null copy .022911737.
- Active best-validation TV mean: U **.082549503**, E **.075247675**, ER **.075295470**.
  E-U improves in all five trials; ER-U also improves in all five, mean −.007254033.
  This does not assign U-relative accuracy improvement entirely to regularization.
- ER-E equal-three-null mean TV **−.000395125**, improves 5/5.
  ER-E active TV **+.000047795**, worsens 4/5; one large gain offsets four losses.
  Descriptive 95% seed t interval [−.003575428,+.003671019] includes zero;
  neither consistent accuracy superiority nor equivalence is established.
- Fixed epoch 9 ER-E active TV **+.000297833**, also worse 4/5.
  No post-hoc threshold relaxation; stage-1 gate FAIL and other prevalences withheld.
- 346 checksums, 60 checkpoint tensor identities, 240 aligned array groups,
  6,720 entity statistics and all paired seed intervals/gates independently rechecked.
  Generated 61,440 entities / 1,413,672 nonfirst gaps valid. No missing epoch-9 states.
- Reporting conclusion: repeatable history-head active benefit and residual-penalty
  null benefit, but combined zero-active-cost requirement unfulfilled. At that stage the bounded
  penalty study and external comparison were proposals; the separate follow-up
  above has since completed them. New DGP, real-data and held-out testing remain
  unexecuted; no final method-superiority claim.

**Previous single-seed v4 comparison COMPLETE: retain E raw forward, add the same .01 centered
residual penalty. Two fresh GPU fits; original response gates and registered
best-validation accuracy screens vs E/U PASS. Eligibility for a separately
registered broader pilot, not multi-seed or external-baseline superiority.**

[Full v4 result](v4_pilot_v1_report_2026_09_16.md),
[machine-readable evidence](v4_pilot_v1_result.json),
[preregistration](revision_v4_preregistration.md).

- Registration `642b878`; CPU/GPU source `635664954057bf037ef04c572ac113db8fb28767`.
- New ER predicts b+h+r exactly as E; only the regularizer centers r. Same
  133,581 parameters, pi=.05, seed 20260930, kappa 0/1, no warm start/search.
  Saved E/C/R/U reused with verified hashes, matching data, states/order/measures.
- 123 tests and same-source repeated CPU gate PASS. Remote GPU **2**, selected
  after confirming no compute process; busy GPUs 0/1/3 avoided. Two GPU fits complete.
- ER max null copy/repeat **.018631945/.018336381**, active **.354965944/.349362813**.
  Original response/zero-gap/generated-validity gates PASS.
- Active best-validation mark TV: U .086640340, E .085479746, R .096924737,
  **ER .085439410**. ER-E active delta −.000040335, entity SE .000098938;
  this very small difference is not robust accuracy-superiority evidence.
- Three-null mean TV ER-E **−.000336656**, ER-U **−.000073495**. Individual null
  cells can worsen; no universal per-cell accuracy improvement.
- **Fixed epoch 9 ER-E active TV +.000509623 (worse)**. Preserve this limitation:
  the registered best-checkpoint screen passes but E accuracy improvement is
  checkpoint-sensitive. ER-R active improvement remains at best and epoch 9.
- 101 checksum comparisons, 20 checkpoint tensor identities, 80 aligned entity
  groups, 2,240 statistics and paired factorial contrasts verified. 4,096 generated
  entities / 94,662 nonfirst gaps valid. No held-out, later pi, new seed or baseline run.
- The then-proposed U/E/ER seed/prevalence replication subsequently completed
  its 30-fit stage and stopped at the registered gate; see the latest result above.
  Do not reclassify the single-seed discovery pilot as independent confirmation.

**Previous v3 model result: v3 eight-fit pilot COMPLETE. R passes the original response
criteria at pi=.05 but FAILS the registered accuracy screen against historical U.
Do not promote this candidate to later prevalences or confirmation.**

[Full v3 result](v3_pilot_v1_report_2026_09_16.md),
[machine-readable evidence](v3_pilot_v1_result.json),
[preregistration](revision_v3_preregistration.md).

- Registration `9813573`; final CPU/GPU source `44a2bd3848abd1fdddf0fc37d2cd6894109e3d65`.
- Implemented H (history only), E (extra history/raw route), C (centered route),
  R (same as C + fixed .01 smooth-RMS residual penalty), eight fresh GPU fits.
  Each adds 32 history coefficients; 133,581 stored parameters. H has 1,056
  dormant route parameters. Old U evaluated at saved states, zero new U fits.
- 106 tests PASS; same-source CPU reproducibility, checkpoint, loss-drop,
  support and zero-gap checks PASS. Initial CPU numeric failure preserved;
  constant probabilities now evaluated once then broadcast, no tolerance change.
  All CPU trained states remain bitwise identical before/after this audit fix.
- R maximum null copy/repeat range **.003943601/.003880979**, active
  **.340098053/.334719523**. All original response/validity gates PASS.
- E/C rare-null copy ranges **.056817306/.058481466** still exceed .05.
- Active conditional mark TV (lower better): U **.086640340**, E **.085479746**,
  C **.102974418**, R **.096924737**. R improves C by .006049681 but remains
  .010284397 worse than U. Same direction at fixed epoch 9 and on factual TV.
- Three-null mean TV R−C **−.000395910**, R−U **−.000113982**. Two individual
  null cells are slightly worse than U; no universal null-accuracy improvement.
- Response safety improved in this single pilot; conditional accuracy vs U is
  unresolved. The centering parameterization already worsens active accuracy
  before regularization, so blame cannot be assigned solely to the penalty.
- 97 file checksums, 20 checkpoint tensor identities, 80 aligned array groups,
  2,240 statistics and paired contrasts verified. 16,384 entities / 378,648 gaps
  generated, support/mark/value checks PASS. No held-out, later pi or new seed.

**Previous diagnostic COMPLETE: [fixed-checkpoint route decomposition](route_decomposition_v1_report_2026_09_16.md),
[evidence](route_decomposition_v1_result.json), and
[model/theory/data/literature/claim audit](research_claims_and_related_work_audit_2026_09_16.md).**

- Registered `d12021b` before implementation. Final CPU/GPU source `7d5a1be`.
- 92 tests PASS, same-source CPU PASS; 12 best/epoch-9 snapshot records from
  six existing pi=.05 U/A/B fits, full train/validation, both contexts/all bins.
- **New fits, optimizer steps, generated samples and held-out access: zero.**
- U rare-null best validation repeat BCE: full F **.489598539**, mean-only M
  **.488799468**, residual-only R **.542980485**, whole-route zero Z **.542022948**.
  The train-reference mean is useful. Removing the residual improves validation
  slightly but worsens train (+.000829508); epoch 9 repeats the sign pattern.
- U rare-active validation F **.451531454 -> M .503302035** (+.051770582).
  Useful active gap dependence would also be lost by unconditional removal.
- Registered mean-usefulness and active-residual-usefulness directions hold;
  **null residual harmful in BOTH train and validation does not hold**.
  Do not reclassify this mixed result as a fully supported hypothesis.
- All fixed mechanical checks PASS; 23 file checksums, 12 tensor identities,
  48 aligned entity-array groups and 4,416 statistics independently rechecked.
- First `audit_v1` failed a numerical identity: batch 128 vs parent 256 changed
  CUDA batch-shape arithmetic. Matching 256 resolves it; final historical loss mismatch
  <=1.37e-9, unchanged 1e-6 limit. Failure preserved; final output `audit_v2`.
- No new trained candidate, superiority or dilution claim. Centering alone is
  function-preserving; history capacity and residual regularization require
  separate controls. Related work already covers copy mixtures, interactions,
  sequential generation, group weighting and behavioral fidelity diagnostics.

**Historical training result: the separately registered three-objective diagnostic is
COMPLETE. All U/A/B fail the pi=.05 rare-null response gate. No model success
or expansion to later prevalences is established. V1/v2 failures are preserved.**

Previous training execution: [loss-control report](loss_control_v1_report_2026_09_16.md)
and [machine-readable evidence](loss_control_v1_result.json).

- Registration `d4069c9` preceded implementation/fits. CPU/GPU source:
  `f4bed4b9a17970d2cd146515f3e97d1ae0db6947`.
- **82 tests PASS**, six tiny CPU fits (two each U/A/B), all CPU gates PASS.
- **Six GPU fits completed**, pi=.05 x kappa 0/1 x U/A/B, same v2 architecture,
  initial states, data/order, budget and checkpoint selection. A adds a global
  transition-mean repeat BCE; B retains equal-context repeat BCE.
- U/B's four prior best states reproduced **bitwise**, with identical epochs,
  likelihoods and responses; no trained weight warm-start.
- Kappa=0 rare copy ranges U/A/B: **.055189/.061733/.081949**, all >.05.
  Repeat ranges also fail. Added auxiliary and context redistribution each
  increase this null response; the same ordering holds at fixed epoch 9.
- U train/validation null ranges .054936/.055189; train-defined central-bin
  ranges .053865/.054159; 89.06%/90.33% of entities individually exceed .05.
  This is not restricted to validation outliers or extreme gap bins.
- Turning off the whole U gap route worsens rare-null validation repeat BCE
  **.489599 -> .542023** at fixed weights. The route cannot be treated as
  uniformly useless; history correction and current-gap variation may be mixed.
- B-A validation repeat BCE: rare null **+.006451**, active **-.006080**.
  This is a tradeoff in one exploratory seed, not universal harm or superiority.
- 12,288 generated entities / 283,986 nonfirst gaps; generation checks PASS.
  62 checksum comparisons and checkpoint/sample/diagnostic array identities
  verified. No worker error; each candidate's local scientific gate is FAIL.
- No later prevalence, five-seed, real-data or held-out run. The then-proposed
  train-centered route diagnostic subsequently completed; see latest work above.

Prior v2 implementation and pilot:

V2 implementation preserves the registered 133,549 parameters and fresh-v1
common initialization. Direct bank-gradient isolation, known-label permutation,
strict-past behavior, likelihood/audit agreement, support, checkpoint roundtrip
and fixed auxiliary denominators passed. The revision-aware runner pins the
original v2 YAML hash and inherited v1 budget, requires the same-source CPU gate,
and reuses the immutable data/oracle index. V2 outputs use a separate namespace.
The v2 YAML retains its historical preregistration-state metadata; this file
records current execution state.

V2 execution (2026-09-16, before the separate loss diagnostic):

- Source: `c77d2f9558ca019d14ccf4b833cbc6628728ff44` (committed before CPU/GPU).
- Immutable registration: `d8302e3`; model seed 20260930; 133,549 parameters.
- CPU: two identical histories/best states, train objective drop **48.33%**;
  support, vocabulary, finite-value and zero-gap checks PASS.
- GPU: **4 fits**, CS2-U1/B1 x kappa 0/1, at pi=.05 only.
- Primary CS2-B1 fails kappa=0/label=1: copy **0.081949**, repeat **0.080610**,
  both above .05. The same v1 cell was below .05 (copy 0.045730).
- At kappa=1/label=0, B1 copy response falls from v1 0.046383 to v2 **0.033930**.
  Active kappa=1/label=1 copy/repeat responses remain **0.343783/0.338277**.
- CS2-U1 also fails kappa=0/label=1 (copy **0.055189**, repeat **0.054313**).
  Thus the new failure is not attributable only to the balanced auxiliary.
- All four jobs have exact zero-gap invariance and valid generation. Total
  8,192 entities / 189,324 nonfirst gaps; support violations and reserved marks 0.
- Pi=.10/.25/.50 training stopped by the original rule. No fallback, five-seed,
  real-data or held-out run. Parent exit 2 is scientific FAIL; no worker error.
- Saved checkpoint/sample identities and 21 artifact checksums rechecked.
- The then-proposed three-objective control was subsequently registered and
  completed separately; see the latest result above.

Prior v1 checkpoint forensics (2026-09-16):

- Forensic source: `d96e1b4966f9fec018aee472a26be579ab9a4d1c`.
- All 12 saved best checkpoints analyzed on full train histories. New fits and
  optimizer steps: zero. No new validation/test content or realized latents read.
- Failed-cell train null range: 0.051240, close to existing validation 0.051167.
  Median entity range 0.050248; 51.36% of null entities exceed .05 individually.
- Central observed-gap mass retains 88.49% of the range. The reaction is not
  confined to a few entities, early histories or only tail bins.
- The variation resides in the bilinear copy route. The fresh-mark term only
  attenuates it. Shared direct-route gradients transmit local cross-context
  influence, but null self-descent also raises the response. Strong gradient
  conflict or a unique historical cause is **not** established.
- Three new diagnostic tests; 16 focused tests passed. Checkpoint state hashes
  unchanged, and 24 checkpoint/array checksums verified after analysis.
- V2 registered: replace one shared rank-32 route with two observed-context
  rank-16 routes; preserve 5,440 route / 133,549 total parameters, objectives,
  data and pilot gates. No known active-label mask. Per-context capacity falls;
  shared-feature influence and within-context spurious responses remain possible.

V1 execution retained below:

- Historical base: GitHub `main` at `de8fa70`; repaired SAF v6 failed its pilot.
- Original protocol: `a191392`; exact architecture/oracle: `34cdf66`.
- Model, prevalence data and pilot contract: `924131b`; API fix: `43ceef8`.
- Exact CPU/GPU source: `0d3be638e180695bf62379af8444398bb0c9708e`.
- C0/U0/U1/B0/B1 implemented. The four aligned candidates have matching
  parameter keys/shapes/initialization. U1 and B1 have 133,549 parameters each.
- All eight pi x kappa train/validation views created. Original entity splits
  retained; train-only oracles PASS at every pi. No test bodies materialized.
- Relevant tests: **54 passed**. Two CPU runs have identical histories and
  best-state tensors; train objective decreased **48.16%**.
- GPU: **12 fits**, U1/B1 x kappa 0/1 x pi 0.05/0.10/0.25, seed 20260930.
- CS-B1: pi=0.05 PASS, pi=0.10 PASS, **pi=0.25 FAIL**. At pi=0.25,
  kappa=1/label=0 copy/repeat ranges are **0.051167 / 0.050313**, exceeding 0.05.
  Active ranges are 0.367779 / 0.361659; absent active signal is not the failure.
- Pi=0.50 **data/oracle completed; training not run** under the stop rule.
- All 12 runs: exact zero-gap invariance, no support violations or reserved
  marks, finite generated values. 24,576 generated entities retained locally.
- Validation used for checkpoint selection and predictive-response audit;
  test content not accessed. No five-seed or real-data study started.

Read these files in order when recovering from a missing conversation:

1. [Latest cross-history/oracle diagnosis and tradeoff](replay_oracle_v1_report_2026_09_17.md),
   [professor briefing](professor_brief_replay_oracle_2026_09_17.md), and numerical evidence linked above.
2. [Previous rollout audit](rollout_audit_v1_report_2026_09_17.md), whose OBS
   interpretation is qualified by the latest proper joint-oracle controls.
3. [Lambda/prevalence/external baseline study](followup_v1_report_2026_09_17.md).
4. [Fixed U/E/ER multi-seed replication](replication_v1_report_2026_09_16.md),
   [v4 discovery](v4_pilot_v1_report_2026_09_16.md), and
   [v3 pilot](v3_pilot_v1_report_2026_09_16.md).
5. [Model, preprocessing and related-work audit](research_claims_and_related_work_audit_2026_09_16.md).
   Its then-unexecuted external comparison has since completed in the follow-up.
6. [Route decomposition](route_decomposition_v1_report_2026_09_16.md),
   [loss controls](loss_control_v1_report_2026_09_16.md),
   [v2](v2_pilot_v1_report_2026_09_16.md),
   [checkpoint forensics](checkpoint_forensics_v1_report_2026_09_16.md),
   [v1](pilot_v1_report_2026_09_16.md), and their linked registrations/evidence.
7. [Original architecture/oracle contract](architecture_and_oracle_v1.md),
   [original research protocol](research_protocol_v1.md), and
   [oracle-only signal audit](oracle_audit_v1_report_2026_09_16.md).

The former next hypothesis, retaining E forward while penalizing only the centered
residual, was subsequently implemented as ER and tested in discovery, replication
and the lambda follow-up. It is historical, not an unexecuted next task. The latest
bounded proposal and remaining independent confirmation are stated at the top.

Runtime roots relative to this CS-SAF worktree:

- `data/cs_saf/prevalence_v1/`
- `artifacts/cs_saf/prepared_v1/`
- `artifacts/cs_saf/cpu_gate_v1/`
- `artifacts/cs_saf/pilot_v1/`
- `artifacts/cs_saf/forensics_v1/` (entity diagnostics and gradient arrays)

V2 output namespace: `artifacts/cs_saf/revision_v2/`, containing `cpu_gate_v1/`
and `pilot_v1/` with terminal records.

Loss diagnostic: `artifacts/cs_saf/loss_control_v1/cpu_v1/` and `gpu_v1/`,
with best and epoch-9 checkpoints for all six fits, and train/validation entity arrays.

Previous decomposition: `artifacts/cs_saf/route_decomposition_v1/cpu_v2/` and
`audit_v2/`. The first technical failure is retained in `audit_v1/`.

Latest v3 outputs: `artifacts/cs_saf/revision_v3/cpu_v1/` (preserved failure),
`cpu_v2/` (PASS), and `gpu_v1/` (eight fits plus fixed U evaluation).

Each trained job retains its best checkpoint, history/report, intervention
audit and generated sample. Compact evidence records absolute locations,
source/config/data hashes and checked artifact checksums (100 for v1,
24 checkpoint/array checks in forensics, 21 for v2, 62 for loss control).

Store code/config commits before execution and report commits afterwards.
Preserve failures as well as successes. Runtime data/checkpoints remain on the
workstation; the GitHub repository is not their full backup.
