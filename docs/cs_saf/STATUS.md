# CS-SAF recoverable research state

Updated 2026-09-20. Branch: `research/cs-saf`.

**Latest frozen copy-probability calibration COMPLETE. 80 four-scalar fits,
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
  intervals include zero. Ucal was not run; a generic calibration benefit
  cannot be assigned to the extra history structure alone.
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
- Next proposal: a separately fixed U/Ucal/E/Ecal comparison to isolate generic
  calibration from added history capacity, with fresh data/training seeds and
  multiple paired generation tapes. No further fits automatically launched.
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
