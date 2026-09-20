# Frozen U/G history diagnostic v1

Registered before computing this diagnostic. Parent: e19897b3adf1851ddfc59c11232ad898138f3f0f. Exploratory localization on previously inspected data, NOT independent confirmation. C and rollout B/P failures remain unchanged.

## Scope and budget

- Frozen U/G + existing A (train-only 2 groups x 5 gap-bin logit correction); 2 kappas x 3 training seeds. No weight updates, coefficient fitting, extra generation, test split access, or GPU work.
- Replay actual validation histories and the 36 stored A datasets (2 kappas x 3 parent seeds x 2 generators x 3 tapes) through BOTH predictors of the same parent seed. Total 84 predictor/history evaluations: 12 real + 72 generated.
- Describe the 120 already saved oracle datasets: 2 kappas x 2 information modes x 30 tapes. Primary generated-law reference is JOINT_BIN; JOINT_CONT is sensitivity analysis.
- Use original immutable manifests/hashes; CPU single thread. Artifact hashes, source revision and timestamps retained. No discarded/reselected seeds.

## State definition and coverage

At event index t (zero based), history length is t and run length is the count of consecutive equal marks ending at t-1. Neither current mark nor future observations enter state assignment or prediction. Exclude t=0. A sequence ending is not a nonrepeat event.

- History bins: 1-4, 5-8, 9-16, 17-31.
- Prior-run bins: 1, 2-3, 4-7, 8+.
- Primary views are these two separate partitions, not a selected favorable crossing. All bins/groups/kappas are reported. Joint history x run and five native gap calibration bins are descriptive localization only.
- Before validation diagnostics, export TRAIN-only coverage with these fixed bins. Keep all bins even if sparse. Any cell with <50 distinct sequences OR <200 observed transitions is explicitly low-coverage and cannot support a localized-error screen. No outcome-driven merging.

## Measurements

On identical histories: corrected observable repeat probability p, target-law probability q, signed bias mean(p-q), MAE mean(abs(p-q)), probability squared error mean((p-q)^2), observed repeat Brier and mark NLL. The squared probability error is an expected repeat-Brier regret under the target law at those histories, not observed Brier on generated outcomes. Cross-source comparisons are descriptive, not an identified causal decomposition.

On real histories q uses MODELINFO: current gap bin, past continuous gaps and past marks. On generated support-valued histories q uses BIN: current/past gap-bin observations. This is the target binned process law, not the model's own conditional law. Also report MODELINFO on generated histories (representatives treated as exact past gaps) as a sensitivity check; never silently mix information patterns. Oracle has the analytic DGP law but no realized hidden states or future outcomes. It is evaluation-only.

For each partition: observed continuation probability, state occupancy among valid transitions, and oracle conditional continuation probability. At each state y-q=(p-q)+(y-p) for a self-generated source; retain the signed terms instead of calling every observed difference a model error. Compare distribution of state visits and continuation rates to matched oracle generations. Exact current gap-bin conditioning is reported separately to avoid attributing a changed gap mix solely to run memory.

## Uncertainty and decision rules

- Report per-tape values and means of three tapes WITHIN each parent seed. Three parent seeds are not nine independent training trials and are not new data.
- Sequence-cluster bootstrap, 400 replicates, fixed seed 20264601, gives descriptive 95% intervals for bias and continuation/occupancy. Events within a sequence are not resampled independently. These are not multiplicity-adjusted confirmatory tests.
- Oracle seeds in existing order form ten disjoint consecutive triplets; compare model three-tape means to ten oracle three-tape means. Their min/max and 5-95% range describe finite-sample variation, not a rigorous error floor or equivalence test.
- Flag a CONDITIONAL error candidate only when mean bias magnitude >=.02 (two percentage points), sign agrees in all three parent seeds, each seed magnitude >=.01, all cells meet coverage, and descriptive seed intervals exclude zero in that direction. This is a deliberately practical exploratory screen, not proof of architectural cause. Also publish MAE and squared error even if biases cancel.
- Flag a GENERATED state/continuation discrepancy only when all three parent three-tape means are on the same side of the matched oracle 5-95% interval, mean absolute difference from oracle center >=.02, and every constituent model cell meets coverage. Reference range has only ten triplets; any flag requires later independent verification.
- No flag means insufficient evidence for this preselected class of error, NOT global model correctness. No new architecture/training follows automatically. Do not search additional coefficients/bins/seeds until a favorable result appears.

## Audit before diagnostic

Check strict-past indexing and first-event/mask handling; normalization and reserved marks; U latent-copy versus observed-repeat identity; G nonrepeat exclusion; calibration nonrepeat-ratio preservation; streaming/full-prefix agreement; saved generated-frame decode/encode replay; frozen hashes and training-only calibration; C implicit gradient and fixed-history constraint; and oracle recursion against independent dense-state tests. A failure affecting scientific results blocks dependent interpretation and is documented, not concealed by modifying thresholds.

Deliver code/tests, complete tables, coverage, two state/error figures and a cross-history comparison, an architecture/theory/code audit and a limited next-step decision. No claim of new method superiority, real fraud utility, or exhaustive absence of bugs.
