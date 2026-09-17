# Cross-history replay and joint-oracle controls — 2026-09-17

Registered locally before implementation/outcomes, based on published `30f2d37`. The user requests execution and Git preservation, plus an explanation of branch state and acceptable performance tradeoffs. All registered results will be reported, including null or contradictory results. There is no new training, coefficient search, checkpoint selection, real-data/test access, or automatic subsequent fit.

## A. Evaluate different predictors on exactly the same histories

Reuse all 120 U/E/L003 best checkpoints and the previous audit's immutable panels, OBS/FULL samples and three sampling tapes: four prevalences, both kappas, five paired trials. For each mode and tape evaluate each of three target predictors on each of three source models' exact histories. OBS sources share observed gaps/values and differ in sampled marks; FULL sources can differ in gaps, marks and values. The matrix contains 2,160 predictor/path evaluations, 552,960 replayed entity sequences; these are not new independently trained models.

Let C[a,b] be expected repeat-curve L1 of target predictor a evaluated on source b's stored paths. C is computed separately per observed group with the same five train-fitted metric bins and full-validation reference as before. For a paired comparison A/B, symmetrically decompose its diagonal difference:

* F = ((C[A,A]-C[B,A]) + (C[A,B]-C[B,B]))/2: predictor difference on matched histories.
* H = ((C[A,A]-C[A,B]) + (C[B,A]-C[B,B]))/2: source-history difference within fixed predictors.
* C[A,A]-C[B,B] = F+H exactly, including when the L1 is nonlinear.

This is an algebraic decomposition of this matrix, not a unique causal parameter attribution. H is mark-history sensitivity for OBS; for FULL it includes every generated history component. Report the complete three-by-three matrix, F/H, signed bin curves, empirical metrics and all null contexts. Validate every diagonal against the saved parent probabilities/metrics. Report both directions, without ratios when components cancel. A component is described as consistently material only if its mean has the same sign and magnitude at least .001 in at least three prevalences and at least 4/5 paired trials at each qualifying prevalence. This descriptive threshold is not a significance test or publication gate; interactions and exceptions remain visible.

## B. Determine whether fixing a real gap trajectory itself distorts the comparison

Use the known augmented semi-Markov observation filter, with stationary residual-duration initialization, shared gap/mark state for the active group and independent gap/mark states for null groups. It samples from predictive mixtures and updates beliefs using only observations. No realized latent arrays from the dataset enter any predictor or audit. Gap emissions are exponential, fresh marks uniform over 64 categories, and observed repeat probability includes fresh accidental repeats. All modes preserve each panel's length, observed group, first mark/value, and missing first gap. Numeric values are held at the observed independent values; this is a control for the gap/mark joint law conditional on those independent values, not an amount-generation comparison.

Five fixed oracle modes, each on the same 256 entities and three parent seed tapes for every data cell/trial (153,600 generated oracle sequences):

1. JOINT_CONT: exact sequential continuous-gap/mark joint DGP using filtering mixtures; assimilate continuous current gap before predicting mark, then current mark, then advance duration state.
2. FIX_CONT: supply the observed continuous gap trajectory; use the same exact online continuous-gap oracle for recursively generated marks. This hybrid path is not the true conditional law given the entire future gap trajectory.
3. JOINT_BIN: exact joint DGP projected to fitted support bins; sample gap bins from integrated state-specific emission masses, predict mark from the posterior after that bin, update using bin and mark. Only binned gap information is available in the past.
4. FIX_BIN: supply the panel's quantized observed gaps; otherwise the same binned filter as JOINT_BIN.
5. FIX_MODELINFO: observed continuous gaps in past history, but only the current support bin is used for mark prediction; assimilate full current gap after predicting its mark. This matches the information pattern of the parent copy oracle and provides a closer control for CS-SAF's OBS inputs. It is not a proper joint sampler.

For continuous modes use the original full-validation reference. For binned modes the primary reference is the same validation trajectories with only gaps quantized, using unchanged train metric edges; additionally report the original reference for comparison to the existing generation benchmark. Report oracle predictions on the original panel (teacher forcing) for each information pattern. This separates a hybrid-path diagnostic from representation differences. Compare FIX_CONT−JOINT_CONT and FIX_BIN−JOINT_BIN at matched representation; also compare FIX_MODELINFO−its own teacher-forced reference. Use .001/4-of-5/3-of-4 as the same descriptive materiality screen, not proof that all model error is an artifact.

Streams: retain the parent's separate gap-uniform and mark-uniform tapes. An additional fourth SeedSequence child supplies the continuous mixture-component draw; all components remain independent. Joint-bin generation uses the same gap-uniform tape for inverse CDF. Continuous probabilities use exponential densities; bin probabilities use integrated masses, never representative-point densities. Preserve float32 gap storage, float64 oracle calculations. Repeat expectations are computed before current mark observation. Oracle trials vary sampling randomness, not learned model seeds: do not count five oracle trials as five trained oracle models or independent-data confirmation.

## C. Explain permissible tradeoffs without rewriting old outcomes

Re-express the already published native L003−E generation differences as absolute and E-relative costs. Report for repeat-curve L1 and MI the mean, each paired trial, and the upper end of the descriptive paired 95% t interval (df4). The positive part of that mean/upper bound is a retrospective break-even tolerance, not a newly justified acceptable margin or a retroactive PASS. Also show the existing conditional TV change and null response benefit from the original records. Different endpoints have different units; do not add their raw values into an unregistered composite score.

Explain three decisions separately: technical correctness must hold; a diagnostic can proceed despite some scientific conditions not meeting its screen; a performance claim can allow secondary losses only under an externally justified or prospectively registered noninferiority budget, with uncertainty and group-specific costs reported. Existing ER zero-active-cost FAIL remains unchanged. The previous lambda expansion and diagnostics already proceeded despite that failed stage. No universal requirement that every metric and every seed improve is imposed.

## Technical gate and execution

Commit implementation before CPU/GPU execution. Tests cover independent dense small-state Bayes recursions and joint probability factorization; normalization/stationarity; exact binned emissions; agreement with the existing observable filter for FIX_MODELINFO; strict-past behavior; reproducibility; reserved outputs; and checkpoint/hash immutability. A fixed CPU MC validity check uses 2,048 sequences per group and seed2026091901, comparing exact stationary gap/repeat moments and gap-bin repeat joint probabilities with entity-cluster uncertainty (max of .01 absolute and six cluster SE); it is a sampler validity check, not a model performance experiment.

Use deterministic CPU scans and disable TF32 as in the completed audit. Check an available GPU before launching one read-only model-evaluation process, never stop other users' work. Freeze inputs by manifest/checkpoint hashes, save all oracle paths/replay tables, test source and run metadata. Preserve technical failures if any and amend only demonstrated implementation defects, not scientific endpoints. Publish the complete diagnosis and limitations to the established research branch after validation; do not silently merge the default branch or overwrite historical conclusions.
