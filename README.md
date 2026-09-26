# DataGeneration: CoF-SeqGen Research

**Latest ARGN research is on `research/argn-fraud-audit-v1`;
the preceding external audit is on `research/cs-saf-external-audit-v1`;
the earlier U/E study is on `research/cs-saf`, and `main` remains the earlier SAF base.**
Code, preregistrations, reports and numerical evidence are versioned. Large data,
checkpoints and raw paths remain on the workstation; GitHub is not their full backup.

Latest [labelled ARGN relationship audit](docs/argn_fraud_audit_v1/README.md)
uses the official engine with transaction fraud labels, fit-only encoding statistics,
two training seeds and direct class-conditional relationship diagnostics. It preserves
the failed [earlier ARGN relation-path pilot](docs/argn_relation_pilot_v1/README.md).
Errors already occur at the first generated transaction and on real-history predictions;
this does not establish an accumulation-only failure or a novel architecture contribution.

Previous [D empirical-bin amount control](docs/cs_saf/external_binned_amount_v1/README.md)
is **COMPLETE AND CLOSED: two fresh fits, four generated tapes, no calibration.**
Both datasets improve generated relations and Berka's excessive amount tail is
substantially corrected. However, Berka operation marginal TV and amount log-MAE
cost more than the preregistered allowances; Sparkov passes all screens. **Do not
adopt D_bin as the common final model. Stop this amount-output modification direction.**
Preserve D as the reference and D_bin as a completed control; no extra bins,
heads, coefficients, epochs or seeds are running. Registration/source `1579ee4`.
[One-page architecture](docs/cs_saf/external_binned_amount_v1/model_explanation.md).

Previous [frozen D amount/history diagnostic](docs/cs_saf/external_amount_diagnostic_v1/README.md)
is **COMPLETE: zero training and zero new generation**, using two stored checkpoints
and four saved tapes. Berka's amount tail is already excessive on real histories
(above fit q99.9: observed .1091%, predicted .9540%) and increases further under
generated pasts with the same current gap/mark. Sparkov does not show the same
amplification. Hybrid amount/other pasts reveal strong interactions, so this does
not identify past amounts alone as the cause. The next output-control candidate
was subsequently evaluated in the completed control above. Its original
diagnostic findings are preserved; no further head fit or seed expansion is running.

Previous [bounded amount/action output comparison](docs/cs_saf/external_controls_v1/README.md)
is **COMPLETE: 14 new neural fits, 14 common calibration fits and 56 generated
datasets (5,525,464 events).** Registration and scientific source: `e1d44f2`.
Amount-only, action-only and combined U/G changes were compared with a direct
categorical action head D, using the same history information and closely
matched capacity. [One-page model explanation](docs/cs_saf/external_controls_v1/model_explanation.md).

The positive three-component amount output removes negative draws and improves
amount relations, but **U/G's additional repeat-specific contribution is not
established**. On Sparkov, direct D achieves time/action TV .0880 versus combined
U/G .1563/.1881. On Berka, G's .1184 improves that metric over D .1311, but costs
amount TV (.2388 vs .1650), mark NLL and repeat Brier. Neither U/G passes the
registered overall promotion screen, before or after the common calibration.
Berka's positive outputs still have excessive upper tails: generated 99.9th
percentiles are roughly 12–23 times the observed validation value. D also trails
the simple transition control on both primary relations in both datasets and
is not automatically adopted as a final model.

All 34 CPU checks, 1,064 independent generation-metric checks (max difference
1.42e-15) and 28 checkpoint prediction reductions (max difference 6.10e-8) pass.
No scientific run failed or was retried. One fit seed, two generation draws and
reused development data limit the claims. **The bounded comparison is closed;
no extra architecture search or seed expansion is running.**

Previous [external U/G port and Berka/Sparkov pilot](docs/cs_saf/external_port_v1/README.md)
is **COMPLETE: 8 valid neural fits, 4 frozen-network calibration fits and 32
generated datasets (3,066,189 events).** External static/auxiliary fields and
target-complete long trajectories are supported; each U/G prediction still uses
only the most recent 31 events. The external shared-rank adapter is an explicit
change, not the unchanged two-group controlled model.

The current U/G ports are **not established final models**. On Berka, G improves
day-gap/operation TV over U (.2022 vs .2430), but trails official ARGN (.0884)
and a simple transition control (.0931). On Sparkov, U/G improve category-transition
TV over this ARGN run (.1586/.1618 vs .1962), but the transition control reaches
.0774. Amount relations remain substantially worse than empirical resampling.
Raw U/G also produce about 2.3% negative amounts on Berka and 1.3% on Sparkov;
their signed-log Gaussian permits this even on observed histories. Repairing only
negative records cannot close the much larger amount-distribution discrepancy.

All 42 unique engineering tests pass. Independent reductions verify 608 stored
metric values (maximum difference 2.34e-15); checkpoint score reconstruction,
matched initial tensors, input/output hashes and all CPAR tail events also pass.
Two original CPAR attempts are excluded and preserved after a documented tail/API
correction; valid results use **official CPAR + a tail-preserving input adapter**.
One training seed/two generation seeds, different native ARGN lengths/encoding
statistics and unresolved CPAR training adequacy limit interpretation. Aggregate
fidelity does not establish personalized generation, privacy or fraud utility.
All jobs from that pilot finished. Its proposed amount-support/distribution and
ordinary gap-conditioned action follow-up is now completed above; the original
pilot outcomes are preserved. [Full tables and figures](docs/cs_saf/external_port_v1/README.md),
[methods](docs/cs_saf/external_port_v1/methods.md),
[execution record](docs/cs_saf/external_port_v1/execution_notes.md).

Previous [Berka/Sparkov observed-relation audit](docs/cs_saf/external_relations_v1/README.md)
is **COMPLETE: 1,990,071 development transactions checked against raw data,
empirical prediction tables evaluated, zero neural fits or new generations.**
Adding observed gap to a previous-operation table improves Berka validation NLL
by 8.92%, but only 39.04% of its adjacent pairs have unambiguous date ordering.
Changing within-day order changes full-pair repeat rate by 4.22 percentage points.
Sparkov same-merchant repetition is only 0.204%; gap/category and merchant/amount
relations are more appropriate targets than long merchant runs. Its sparse
merchant transition tables worsen NLL, so this is not evidence of a useful strong
baseline or proposed-model superiority. Seven unit checks and 310 independent
aggregate-row checks pass. The [next comparison contract](docs/cs_saf/external_relations_v1/next_comparison_contract.md)
requires explicit external U/G ports: the historical controlled code enforces two
contexts, no auxiliary fields and at most 32 events. That diagnostic did not run
external neural training; the subsequent port/pilot is complete above. Previous
failures and the unproven contribution remain unchanged.

Previous [dataset literature and research-process audit](docs/cs_saf/dataset_literature_and_process_audit_2026_09_20.md)
checks 14 original papers and inventories 22 completed CS-SAF result bundles.
**Prioritize problem validation on external Berka/Sparkov data before another
architecture or state/gap correction.** Simulations are legitimate diagnostic
tools, but repeated exploration on the existing artificial data has not established
current-method external usefulness. Berka operation/type and Sparkov merchant
marks have different meanings. Historical materialization and older CoF external
results are distinguished from current U/G evidence. No new model was trained
for this review; previous failures remain unchanged.

Previous [frozen history/run-state diagnostic](docs/cs_saf/history_diagnostic_v1/README.md)
is **COMPLETE: 84 predictor/history replays on existing U/G+A, 36 stored model
and 120 stored oracle datasets; zero new fits or generations, CPU only.**
Active real-history repeat probability is underestimated after runs of 2-3 by
4.66 percentage points (U) / 4.51 (G). Generated continuation in that state is
54.14% / 54.36% versus 64.26% for the matched oracle; all three parent seeds show
the discrepancy. Long-prefix mean-bias screens do not flag progressive collapse.
Cross-history predictions show similar errors for U/G; both conditional mapping
and visited-history composition matter, without causal attribution. Long-run
minority generation cells have insufficient coverage. The detailed
[architecture/theory/code audit](docs/cs_saf/history_diagnostic_v1/architecture_audit.md)
finds no result-invalidating error in the inspected path, while documenting
component-normalized loss, continuous-training/support-valued-generation history
differences, local rather than global C guarantees, and limited nonrepeat gap
expressivity. 106 tests and 13,608 independent metric checks pass. U/G+A remain
controls; C/B/P failures stand. A state-aware simple control remains a possible
candidate, **not yet registered or trained**; the later dataset review prioritizes
external problem validation first. No independent confirmation,
new-method superiority or fraud utility claim is made.

Previous [frozen U/G rollout-calibration comparison](docs/cs_saf/rollout_calibration_v1/README.md)
is **COMPLETE: 24 correction searches, zero neural refits, 108 model and 120 oracle
generation datasets. Neither B nor prediction-protected P passes the joint criteria.**
Relative to observed-history correction A, active generated L1 improves by 7.85%
for U/B, 9.32% for G/B, and 9.97% for G/P; U/P worsens by 0.22%.
All miss the registered 10% AND .002 mean improvement. Prediction and other
distribution costs pass, but null fixed-history response increases exceed .01
in 1/9, 1/9, 2/9, and 2/9 null cells for U/B, U/P, G/B, and G/P respectively.
Every selected B already meets the training prediction guard; no essential
protection contribution is demonstrated. Same-size continuous-oracle active L1
is .015876 (5–95% Monte Carlo range .009115–.022901), a finite-sample reference,
not an error lower bound or proof of model equivalence. All fits precede evaluation;
no oracle/validation target is used for fitting. One oracle result-column naming
error was preserved and fixed without refitting or changing model outputs.
All verification passed; all jobs ended. **Stop this bounded recipe, retain U/G+A
as controls, and preserve the previous C failure.** No new data, external comparison,
or finance utility claim is made. [Tables](docs/cs_saf/rollout_calibration_v1/result_tables.md),
[methods](docs/cs_saf/rollout_calibration_v1/methods.md),
[registration](docs/cs_saf/rollout_calibration_v1/preregistration.md),
[verification](docs/cs_saf/rollout_calibration_v1/verification.json).

Previous [matched internal structure comparison](docs/cs_saf/structure_v1/README.md)
is **COMPLETE: 18 fresh U/G/C fits and 162 newly generated datasets**.
All models have 133,549 parameters, paired initial states and data orders, the same
base likelihood, training budget and checkpoint rule. C's constrained repeat head
**fails the preregistered primary criteria**: active generated repeat-curve L1 is
.055305, versus .028320 for general head G and .026435 for U+gap; C is worse in
all three paired training seeds. Active Brier .146700 also exceeds U+gap .141890
by more than the allowed .002 in every seed. Equal gap calibration does not
rescue C (.039241 generated L1). Distribution-cost and null-generation screens
pass, but null fixed-history sensitivity is higher in C than G. The fixed-history
constraint is numerically verified; it does not guarantee generated fidelity.
All scientific jobs and independent arithmetic checks are finished. **Do not
advance this candidate to external/independent expansion or add rollout loss
automatically.** Old E/ER failures remain unchanged. New training seeds are not
new synthetic datasets; this is exploratory evidence on existing data seed42.
[Result tables](docs/cs_saf/structure_v1/result_tables.md),
[methods](docs/cs_saf/structure_v1/methods.md),
[preregistration](docs/cs_saf/structure_v1/preregistration.md),
[verification](docs/cs_saf/structure_v1/verification.json).

Previous [bounded baseline adequacy and equal repeat controls](docs/cs_saf/baseline_adequacy_v1/README.md)
is **COMPLETE: 4 ARGN continuations, 12 parent models, 144 generation conditions
(132 new datasets, 12 reused)**. Raw ARGN still fails basic prediction screens
in 4/4 continuations. U+gap and continued ARGN+direct meet both registered
prediction/generation screens in 4/4 parents. Simple controls therefore already
meet this coarse target; their gains are not evidence for a new coupling module.
Direct repetition improves U's generated active L1 from .02788 (gap correction)
to .01107, but worsens event-level Brier from .14133 to .18563. Preserving both
prediction and generation is the remaining question. Independent optimizer,
native replay and artifact checks pass; all stage-one jobs finished. No new
proposed neural architecture was trained in that earlier stage. [Full tables](docs/cs_saf/baseline_adequacy_v1/result_tables.md),
[methods and limitations](docs/cs_saf/baseline_adequacy_v1/methods.md),
[verification](docs/cs_saf/baseline_adequacy_v1/verification.json).
The [preceding official ARGN audit](docs/cs_saf/external_audit_v1/README.md)
and [original architecture proposal, now tested above](docs/cs_saf/external_audit_v1/research_decision.md)
remain available. The historical U/E findings below are unchanged.

Earlier [equal gap-bin calibration](docs/cs_saf/gap_calibration_v1_report_2026_09_20.md)
is **COMPLETE: 80 frozen-network correction fits and 400 new generated datasets**.
Both U and E improve native active repeat-curve L1 in all 5 trials at every
prevalence (U: 8.4–21.7%; E: 13.4–18.3%). However, equally corrected E still has
higher mean L1 than U at every prevalence: +.001048/+.001583/+.000857/+.000640.
E wins only 1/5, 2/5, 1/5, 1/5 trials; its primary benefit screen fails and the
symmetric cost screen meets at 5/25/50%. All paired between-trial and conditional
MC intervals include zero; this is not established universal inferiority.
E retains lower mean conditional TV and each null-cell response, but adding
correction increases null response for BOTH models. U's MI error also rises at
5/10%. No joint screen passes. **E-specific native-generation superiority is
removed from the current claim; generic correction gains are not E's contribution.**
Independent reconstruction of 80 fits, 2,240 conditional means and 2,400 native
values passes with zero arithmetic discrepancy. One metadata-type failure and
two identical technical reruns are preserved. All experiment GPU work finished.
Next proposal: lock these controls, justify acceptable costs before evaluation,
and preregister independent data/training-seed confirmation; not run here.
[Professor brief](docs/cs_saf/professor_brief_gap_calibration_2026_09_20.md),
[all statistics and coefficients](docs/cs_saf/gap_calibration_v1_result.json),
[all 31,200 native values](docs/cs_saf/gap_calibration_v1_all_metrics.csv),
[registration](docs/cs_saf/gap_calibration_v1_preregistration.md),
[technical amendment](docs/cs_saf/gap_calibration_v1_execution_amendment_01.md).

Previous [calibrated U/E matched-history replay](docs/cs_saf/calibrated_replay_v1_report_2026_09_20.md)
is **COMPLETE: 400 immutable native paths, 800 predictor/path evaluations, zero
new training, calibration fits or generations**. On identical source histories
and current gaps, Ecal has higher probability-curve L1 than Ucal at 5/10/25%,
in 4/5 trials at each prevalence. The symmetric predictor term F satisfies the
registered .001 / 4-of-5 / 3-of-4 materiality screen; source-composition H and
empirical-score remainder S qualify only at 10%. Native error is recovered exactly
as F+H+S. F's between-trial intervals include zero at 5/25%; at 50% F is slightly
negative with mixed trials. This is an algebraic diagnosis, not unique causation
or proof that 32 added parameters are responsible. Conditional benefits remain
separate; E's generated-relation superiority is unsupported. All 1,200 empirical
and 2,400 probability-curve L1 values plus 600 decompositions passed independent
verification. GPU work finished. Next proposed bounded change is equal,
train-only gap-dependent calibration for U/E; that comparison is now complete above.
[Professor brief](docs/cs_saf/professor_brief_calibrated_replay_2026_09_20.md),
[full statistics](docs/cs_saf/calibrated_replay_v1_result.json),
[all scalar values](docs/cs_saf/calibrated_replay_v1_scalars.csv),
[all bin curves](docs/cs_saf/calibrated_replay_v1_bins.csv),
[preregistration](docs/cs_saf/calibrated_replay_v1_preregistration.md).

Previous [fixed-model generation repeats](docs/cs_saf/generation_repeats_v1_report_2026_09_20.md)
is **COMPLETE: 800 fresh generation datasets (1,638,400 sequences), zero model or
calibration fits**. Five fresh tapes per frozen model were averaged within each
of five training trials; original tapes were excluded from the new primary mean.
Ecal has higher mean active repeat-curve L1 than Ucal at 5/10/25%, with positive
trial effects in 4/5, 5/5, 4/5 trials, satisfying the preregistered persistent-cost
screen. At 50% the mean difference is small and directions are mixed. Fixed-model
MC intervals exclude zero at 5/10/25%, but between-trial intervals include zero
at 5/25%; neither a unique cause nor general superiority is established.
Ucal-U passes the primary benefit screen at 5/10/25%; no comparison meets the
joint improvement screen. E's previously measured conditional/null-response
advantages remain separate fixed-model evidence. CPU/GPU gates, 1,248 cached-metric
comparisons, 800 saved-plan/model checks and 4,800 independent native values all
passed. GPU work is finished. Its proposed matched-history cross-evaluation
of calibrated U/E is now complete above.
[Professor briefing](docs/cs_saf/professor_brief_generation_repeats_2026_09_20.md),
[all 31,200 scalar metrics](docs/cs_saf/generation_repeats_v1_all_metrics.csv),
[nested statistics and verification](docs/cs_saf/generation_repeats_v1_result.json),
[preregistration](docs/cs_saf/generation_repeats_v1_preregistration.md).

Previous [equal-calibration U control](docs/cs_saf/calibration_u_control_v1_report_2026_09_20.md)
is **COMPLETE: 40 U calibration fits, zero neural retraining, 120 reused U/E/Ecal
references**. Ucal lowers mean active generated repeat-curve L1 at all four
prevalences; Ecal has higher mean L1 than Ucal at every prevalence. All four paired
L1 intervals include zero, and the registered consistency screen is not met.
Ecal nevertheless retains lower conditional TV and null response in 5/5 trials
at each prevalence. Calibration helps U too; E's conditional benefit does not
establish a native generation benefit. 40 frozen-base and paired-input checks,
CPU/GPU gates, and independent recomputation of 960 generation values passed
(maximum discrepancy zero). No technical retries; GPU jobs have ended.
[Professor briefing](docs/cs_saf/professor_brief_calibration_u_2026_09_20.md),
[verified numbers and fitted parameters](docs/cs_saf/calibration_u_control_v1_result.json),
[preregistration](docs/cs_saf/calibration_u_control_v1_preregistration.md),
[recoverable state](docs/cs_saf/STATUS.md).
Its proposed additional-tape evaluation is now complete above; independent
data confirmation remains unexecuted.

Previous [frozen repeat-probability calibration](docs/cs_saf/calibration_v1_report_2026_09_20.md)
is **COMPLETE: 80 four-scalar calibration fits, zero base-model retraining,
120 reused U/E/L003 references**. Calibration decreases active native repeat-curve
L1 in mean at all four prevalences, and decreases MI error in 5/5 paired trials
at every prevalence for both E and L003. However, only two of four prevalences
meet the preregistered primary consistency screen. L003cal still trades lower
null response for worse active generation than Ecal; Ecal has no consistent
primary advantage over U. This is a partial repair, not established model superiority.
17 CPU tests, CPU/GPU gates, a saved-state GPU regression and independent native
arithmetic (1,200 values, maximum discrepancy zero) passed. One TF32 verification
failure and its identical scientific rerun are preserved; GPU execution has ended.
[Professor briefing](docs/cs_saf/professor_brief_calibration_2026_09_20.md),
[verified numbers](docs/cs_saf/calibration_v1_result.json),
[preregistration](docs/cs_saf/calibration_v1_preregistration.md),
[execution amendment](docs/cs_saf/calibration_v1_execution_amendment_01.md).
U+calibration was missing at that stage and is now completed above. Independent-data
confirmation remains unexecuted.

Previous [cross-history replay and joint-oracle diagnosis](docs/cs_saf/replay_oracle_v1_report_2026_09_17.md)
is **COMPLETE: 120 saved models, 2,160 cross-history evaluations, 153,600 oracle
sequences, zero new fits**. The exact online oracle also loses gap–repeat fidelity
when observed gaps are forced and only marks are regenerated. Thus the earlier
OBS result does not uniquely identify a neural history-feedback defect.
On exactly the same generated histories, L003 still has worse repeat-curve
calibration than E at all four prevalences in each of five paired model trials.
Predictor differences are more consistent than source-history differences.
The report preserves all contrary results, quantifies the null-benefit/active-cost
tradeoff, and proposed the bounded calibration comparison subsequently completed
above. Its original diagnostic outcomes remain preserved.
[Professor briefing](docs/cs_saf/professor_brief_replay_oracle_2026_09_17.md),
[numerical evidence](docs/cs_saf/replay_oracle_v1_result.json),
[all binwise contrasts](docs/cs_saf/replay_oracle_v1_binwise.json),
[preregistration](docs/cs_saf/replay_oracle_v1_preregistration.md),
[recoverable status](docs/cs_saf/STATUS.md).

Previous [fixed-checkpoint generation diagnosis](docs/cs_saf/rollout_audit_v1_report_2026_09_17.md)
is **COMPLETE: 120 saved models, zero new fits, 368,640 prefix-anchored diagnostic
sequences**. Replacing real mark histories with recursively generated marks while
retaining observed gaps/values increases the paired expected repeat-curve error:
the preregistered descriptive screen passes for E-U at 3/4 prevalences and
L003-E at 4/4. This identifies a history-feedback sensitivity, not a unique causal
mechanism or an improved trained model. Hybrid gap/mark paths, reused data and
uncertain seed intervals limit the interpretation. All contrary results and
technical amendments are preserved. GPU execution has finished.
[Professor briefing](docs/cs_saf/professor_brief_rollout_audit_2026_09_17.md),
[verified evidence](docs/cs_saf/rollout_audit_v1_result.json),
[preregistration](docs/cs_saf/rollout_audit_v1_preregistration.md).

The preceding [lambda/prevalence/external follow-up](docs/cs_saf/followup_v1_report_2026_09_17.md)
is **COMPLETE: 170 new internal fits, 30 reused fits, 40 CPAR fits and 200 matched
internal generation evaluations**, across 5/10/25/50% prevalence and five paired trials.
Lambda .003 passes the mean-based conditional-accuracy screen at all four
prevalences, but its active paired intervals include zero and its generated
gap–repeat errors worsen vs E. All internal arms, including U, beat this pinned
CPAR implementation on the three active-context primary generation metrics;
this is not evidence that the new regularizer provides that advantage.
The report preserves the old FAIL, numerical results, gradient diagnosis,
CPAR implementation/budget limitations and concrete next experiments.
[Professor briefing](docs/cs_saf/professor_brief_followup_2026_09_17.md),
[verified evidence](docs/cs_saf/followup_v1_result.json),
[research status](docs/cs_saf/STATUS.md).
No independent-data, held-out-test or real-data confirmation has been performed.

Earlier stages below describe their status at the time of those experiments.

Historical [fixed-model replication](docs/cs_saf/replication_v1_report_2026_09_16.md):
**30 fresh GPU fits across five paired training trials, 135 tests and CPU checks PASS**.
ER passes response criteria in 5/5 trials (U 2/5, E 1/5) and improves null TV vs E
in 5/5. Its active TV is worse than E in 4/5, with mean change +.000047795, so the
registered accuracy gate FAILS and the conditional 90-fit prevalence expansion
was not started. ER and E both improve active accuracy vs internal U in all five
trials; this does not establish external-baseline superiority.
[Professor briefing](docs/cs_saf/professor_brief_replication_2026_09_16.md),
[evidence](docs/cs_saf/replication_v1_result.json), [status](docs/cs_saf/STATUS.md).

The [v4 discovery pilot](docs/cs_saf/v4_pilot_v1_report_2026_09_16.md) remains preserved:
its single-seed gate PASS motivated this replication. New trials are reported
separately; no post-hoc seed selection or threshold relaxation is used.

The previous [v3 result](docs/cs_saf/v3_pilot_v1_report_2026_09_16.md) is preserved:
R passed response criteria but failed accuracy against U. At that stage no later prevalence,
new seed, external baseline or held-out test had been run for ER; see the
completed follow-up above for the subsequent experiments.

Earlier v1/v2 state follows:

On `research/cs-saf`, **v2 passed implementation/CPU checks but failed its pilot
at pi=0.05 in the dependency-free rare context**. It reduced another null-cell
response but did not solve full null safety. V1's earlier failure at pi=0.25 is
preserved. Later stopped prevalences, five-seed confirmation and held-out
evaluation remain unexecuted.
See the [research status](docs/cs_saf/STATUS.md) and
[v2 pilot report](docs/cs_saf/v2_pilot_v1_report_2026_09_16.md).

A separate [three-objective diagnostic](docs/cs_saf/loss_control_v1_report_2026_09_16.md)
is complete: 82 tests and all CPU gates passed; six GPU fits reproduced the
earlier U/B results exactly. U/A/B all fail the rare-null response bound. The
added auxiliary and context balancing each increase that response, while
whole-route removal worsens factual repeat loss. No later prevalence study had started at that diagnostic stage.

The subsequent [fixed-checkpoint decomposition](docs/cs_saf/route_decomposition_v1_report_2026_09_16.md)
is complete: 92 tests, CPU/GPU checks, 12 existing snapshots, **zero new fits**.
Retaining the route's train-reference mean preserves useful history correction;
removing only gap variation improves rare-null validation loss slightly but
worsens train loss, and substantially harms active prediction. The registered
both-split null-harm hypothesis is not supported. A new trained solution remains
unproven. See the [model, preprocessing, related-work and claim audit](docs/cs_saf/research_claims_and_related_work_audit_2026_09_16.md)
for the contribution audit at that stage; the latest follow-up report updates
the external execution status.

This repository contains the reproducible research code for CoF-SeqGen and its
support-aligned autoregressive extension (SAF). The `main` branch records the
validated research base through the SAF v6 intervention preflight:

- train-support-aligned gap generation is strongly supported by repeated
  development experiments;
- a static-context codec defect affecting earlier dependency-routing runs has
  been corrected and regression-tested;
- the corrected scalar-gated v6 model failed its preregistered intervention
  preflight, so dependency-routing success is not claimed;
- held-out test data remains sealed.

The conference-oriented extension is developed separately on
`research/cs-saf`. It studies whether support-aligned generation can also
preserve rare, context-specific temporal transition mechanisms.

Raw datasets, model checkpoints, third-party repository clones, and large
runtime artifacts are intentionally excluded. Source contracts, acquisition
scripts, experiment configurations, tests, and compact reports are versioned.
Place locally obtained AMLSim files under `local_data/amlsim/` and Sparkov
files under `local_data/sparkov/`; their expected hashes remain pinned in the
acquisition configuration. The `local_data/` directory is ignored by Git.

The project builds on the TabDiff implementation below and retains its license
and attribution.

---

# TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation

<p align="center">
  <a href="https://github.com/MinkaiXu/TabDiff/blob/main/LICENSE">
    <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
  <a href="https://openreview.net/forum?id=swvURjrt8z">
    <img alt="Openreview" src="https://img.shields.io/badge/review-OpenReview-blue">
  </a>
  <a href="https://arxiv.org/abs/2410.20626">
    <img alt="Paper URL" src="https://img.shields.io/badge/cs.LG-2410.20626-B31B1B.svg">
  </a>
</p>

<div align="center">
  <img src="images/tabdiff_demo.gif" alt="Model Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <p><em>Figure 1: Visualing the generative process of TabDiff. A high-quality version of this video can be found at <a href="images/tabdiff_demo.mp4" download>tabdiff_demo.mp4</a></em></p>
</div>

This repository provides the official implementation of TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation (ICLR 2025).

## Latest Update
- [2025.04]：The categorical-heavy dataset **[Diabetes](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)** evaluated in the paper has now been released!
- [2025.02]：Our code is finally released! We have released part of the tested datasets. The rest will be released soon!

## Introduction

<div align="center">
  <img src="images/tabdiff_flowchart.jpg" alt="Model Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <p><em>Figure 2: The high-level schema of TabDiff</a></em></p>
</div>
TabDiff is a unified diffusion framework designed to model all muti-modal distributions of tabular data in a single model. Its key innovations include:  

1) Framing the joint diffusion process in continuous time,
2) A feature-wised learnable diffusion process that offsets the heterogeneity across different feature distributions,
3) Classifier-free guidance conditional generation for missing column value imputation. 

The schema of TabDiff is presented in the figure above. For more details, please refer to [our paper](https://arxiv.org/abs/2410.20626).


## Environment Setup

Create the main environment with [tabdiff.yaml](tabdiff.yaml). This environment will be used for all tasks except for the evaluation of additional data fidelity metrics (i.e., $\alpha$-precision and $\beta$-recall scores)

```
conda env create -f tabdiff.yaml
```

Create another environment with [synthcity.yaml](synthcity.yaml) to evaluate additional data fidelity metrics

```
conda env create -f synthcity.yaml
```

## Datasets Preparation

### Using the datasets experimented in the paper

Download raw datasets:

```
python download_dataset.py
```

Process datasets:

```
python process_dataset.py
```

### Using your own dataset

First, create a directory for your dataset in [./data](./data):
```
cd data
mkdir <NAME_OF_YOUR_DATASET>
```

Compile your raw tabular data in .csv format. **The first row should be the header** indicating the name of each column, and the remaining rows are records. After finishing these steps, place you data's csv file in the directory you just created and name it as <NAME_OF_YOUR_DATASET>.csv. 

Then, create <NAME_OF_YOUR_DATASET>.json in [./data/Info](./data/Info). Write this file with the metadata of your dataset, covering the following information:
```
{
    "name": "<NAME_OF_YOUR_DATASET>",
    "task_type": "[NAME_OF_TASK]", # binclass or regression
    "header": "infer",
    "column_names": null,
    "num_col_idx": [LIST],  # list of indices of numerical columns
    "cat_col_idx": [LIST],  # list of indices of categorical columns
    "target_col_idx": [list], # list of indices of the target columns (for MLE)
    "file_type": "csv",
    "data_path": "data/<NAME_OF_YOUR_DATASET>/<NAME_OF_YOUR_DATASET>.csv"
    "test_path": null,
}
```

### Important Notes When Creating the Info File
- The MLE evaluation and the imputation task (see later sections for details) assume that one column of your data is the regression or classification target. To enable these tasks, you will need to specify `target_col_idx`. If you don't need to evalute MLE, you can comment out the following line: https://github.com/MinkaiXu/TabDiff/blob/0c4fc3bbfa19046d36c5dce64628df52d5c73d15/tabdiff/main.py#L152
- The fields `target_col_idx`, `num_col_idx` and `cat_col_idx` must be multually exclusive—no column should appear in more than one of these lists. 
- Set the task_type to "regression" if the target column is numerical, or "binclass" if it is categorical.

Finally, run the following command to process your dataset:
```
python process_dataset.py --dataname <NAME_OF_YOUR_DATASET>
```

## Training TabDiff

To train an unconditional TabDiff model across the entire table, run

```
python main.py --dataname <NAME_OF_DATASET> --mode train
```

Current Options of ```<NAME_OF_DATASET>``` are: adult, default, shoppers, magic, beijing, news

Wanb logging is enabled by default. To disable it and log locally, add the ```--no_wandb``` flag.

To disable the learnable noise schedules, add the ```--non_learnable_schedule```. Please note that in order for the code to test/sample from such model properly, you need to add this flag for all commands below.

To specify your own experiment name, which will be used for logging and saving files, add ```--exp_name <your experiment name>```. This flag overwrites the default experiment name (learnable_schedule/non_learnable_schedule), so, similar to ```--non_learnable_schedule```, once added to training, you need to add it to all following commands as well.

## Sampling and Evaluating TabDiff (Density, MLE, C2ST)

To sample synthetic tables from trained TabDiff models and evaluate them, run
```
python main.py --dataname <NAME_OF_DATASET> --mode test --report --no_wandb
```

This will sample 20 synthetic tables randomly. Meanwhile, it will evaluate the density, mle, and c2st scores for each sample and report their average and standard deviation. The results will be printed out in the terminal, and the samples and detailed evaluation results will be placed in ./eval/report_runs/<EXP_NAME>/<NAME_OF_DATASET>/.

## Evaluating on Additional Fidelity Metrics ($\alpha$-precision and $\beta$-recall scores)
To evaluate TabDiff on the additional fidelity metrics ($\alpha$-precision and $\beta$-recall scores), you need to first make sure that you have already generated some samples by the previous commands. Then, you need to switch to the `synthcity` environment (as the synthcity packet used to compute those metrics conflicts with the main environment), by running
```
conda activate synthcity
```
Then, evaluate the metrics by running
```
python eval/eval_quality.py --dataname <NAME_OF_DATASET>
```

Similarly, the results will be printed out in the terminal and added to ./eval/report_runs/<EXP_NAME>/<NAME_OF_DATASET>/

## Evaluating Data Privacy (DCR score)
To evalute the privacy metric DCR score, you first need to retrain all the models, as the metric requires an equal split between the training and testing data (our initial splits employ a 90/10 ratio). To retrain with an equal split, run the training command but append `_dcr` to ```<NAME_OF_DATASET>```
```
python main.py --dataname <NAME_OF_DATASET>_dcr --mode train
```

Then, test the models on DCR with the same `_dcr` suffix
```
python main.py --dataname <NAME_OF_DATASET>_dcr --mode test --report --no_wandb
```



## Missing Value Imputation with Classifier-free Guidance (CFG)
Our current experiments only include imputing the target column. However, our implementation, located at ```sample_impute()``` in [unified_ctime_diffusion.py](./tabdiff/models/unified_ctime_diffusion.py), should support imputing multiple columns with different data types.

### Training Guidance Model
In order to enable classifier-free guidance (CFG), you need to first train an unconditional guidance model on the target column by running the training command with the `--y_only` flag
```
python main.py --dataname <NAME_OF_DATASET> --mode train --y_only
```

### Sampling Imputed Tables
With the trained guidance model, you can then impute the missing target column by running the testing command with the `--impute` flag
```
python main.py --dataname <NAME_OF_DATASET> --mode test --impute --no_wandb
```
This will, by default, randomly produce 50 imputed tables and save them to ./impute/<NAME_OF_DATASET>/<EXP_NAME>.

### Evaluating Imputation
You can then evaluate the imputation quality by running
```
python eval_impute.py --dataname <NAME_OF_DATASET>
```

## License

This work is licensed undeer the MIT License.

## Acknowledgement
This repo is built upon the previous work TabSyn's [[codebase]](https://github.com/amazon-science/tabsyn). Many thanks to Hengrui!

## Citation
Please consider citing our work if you find it helpful in your research!
```
@inproceedings{
shi2025tabdiff,
title={TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation},
author={Juntong Shi and Minkai Xu and Harper Hua and Hengrui Zhang and Stefano Ermon and Jure Leskovec},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=swvURjrt8z}
}
```
## Contact
If you encounter any problem, please file an issue on this GitHub repo.

If you have any question regarding the paper, please contact Minkai at [minkai@stanford.edu](minkai@stanford.edu) or Juntong at [shisteve@usc.edu](shisteve@usc.edu).
