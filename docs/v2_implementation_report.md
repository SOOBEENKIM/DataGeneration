# Benchmark v2 implementation report

Mode: `IMPLEMENT_AND_SMOKE`

This report is cumulative. Commands, elapsed time, status, and artifact paths
are appended at each phase. No 5-seed full experiment is authorized.

## Phase 0 — Preflight and legacy freeze

Status: PASS

Initial evidence is recorded in `docs/audit/v2_preflight.md`. Legacy assets are
inventoried in `docs/audit/v1_asset_inventory.md` and linked by the SHA-256
manifest `docs/audit/v1_artifact_manifest.json`.

Commands executed:

```text
nvidia-smi
git status --short --branch
git rev-parse HEAD
git log -1 --format=fuller
python --version
python3 --version
<COFSEQ_PYTHON> -c <environment probes>
docker ps --format <...>
```

Observed blocker relevant to later GPU phases: NVIDIA-SMI cannot communicate
with the driver, and the intended PyTorch environment reports no CUDA device.
CPU implementation and validation continue.

Later status correction (2026-07-28): the user confirmed that this laboratory
workstation has four available NVIDIA GPUs. The initial command failure above
is retained as the observation from that preflight, but it is not a permanent
hardware blocker. Learned smoke was not withheld because GPUs were unavailable;
it was withheld because mandatory CPU benchmark gates failed. GPU smoke is
authorized after every revised v2.2 CPU gate passes and the v2.2
preregistration is committed.

Snapshot result:

- source snapshot commit: `0066ad321182b21c5eb46c71c69679c3a97192b1`
- annotated tag: `legacy-kappa-v1` (points to the snapshot commit)
- implementation branch: `benchmark-v2-redesign`
- staged binary over 1 MiB: none
- elapsed: preflight/inventory/freeze completed before 17:26 KST

## Phase 1 — Mask-aware sequence contract

Status: PASS for the required focused suite.

Changes:

- `soft_g` now requires keyword-only `valid_mask`, excludes padded sources,
  zeroes padded targets, and retains valid gradients.
- teacher statistics include only valid positions; the GRU uses packed
  right-padded sequences.
- CoF corruption masks are intersected with the valid mask.
- DDIM passes `src_key_padding_mask=~valid_mask` on every denoiser call and
  restores all padded numerical/discrete values to zero.

Commands and results:

```text
python -m pytest -q tests/test_soft_g_grad.py tests/test_soft_g_mask.py
29 legacy gradient tests + 6 mask tests PASS (1.28 s focused rerun)

python -m pytest -q tests/test_sampler_padding.py
1 PASS (1.18 s)
```

Artifacts: source changes in `models/`; tests in `tests/test_soft_g_mask.py` and
`tests/test_sampler_padding.py`.

## Phase 2 — Benchmark v2 DGP

Status: PASS for implemented analytical/unit tests and smoke generation.

Implemented separate stationary Markov v2a and equilibrium-residual
semi-Markov v2b scenarios, deterministic seed streams, right-padding data
contracts, train-only shared binning, latent audit files, and disjoint scenario
paths.

```text
python -m pytest -q tests/test_benchmark_v2_dgp.py
12 PASS (10.96 s)

python -m scripts.generate_benchmark_v2 \
  --config configs/benchmark_v2/smoke.yaml \
  --output-root data/benchmark_v2_smoke
PASS (2.05 s)

python -m scripts.generate_benchmark_v2 \
  --config configs/benchmark_v2/main.yaml \
  --output-root data/benchmark_v2
PASS under tmux (174.24 s)
```

Artifacts:

- `data/benchmark_v2_smoke/`
- `data/benchmark_v2/benchmark_manifest.json`
- `data/benchmark_v2/{markov_persistence_v2a,joint_semimarkov_v2b}/`
- `artifacts/benchmark_v2/full_generation.log`

## Phase 3 — Full-data CPU benchmark gate

Status: FAIL — forced stop.

```text
python -m scripts.validate_benchmark_v2 \
  --config configs/benchmark_v2/main.yaml \
  --data-root data/benchmark_v2 \
  --output-root artifacts/benchmark_v2/gates
```

The initial validator failed closed: equivalence/oracle/bin robustness evidence
was then `PENDING`, making `overall_status=FAIL`. The CPU continuation below
subsequently computed that evidence and replaced every `PENDING` with an actual
PASS/FAIL result. Baseline and CoF smoke do not run unless all mandatory gate
evidence is PASS.

The full-data command completed in 844.36 seconds. Artifacts:

- `artifacts/benchmark_v2/gates/gate_report.json`
- `artifacts/benchmark_v2/gates/gate_report.md`
- `artifacts/benchmark_v2/gates/row_marginals.csv`
- `artifacts/benchmark_v2/gates/single_row_classifier.csv`
- `artifacts/benchmark_v2/gates/context_classifier_curve.csv`
- `artifacts/benchmark_v2/gates/sequential_statistics.csv`
- `artifacts/benchmark_v2/full_validation.log`

Actual evidence:

- all single-row classifier point AUROCs are between 0.4909 and 0.5105;
- v2a context-2 AUROC rises from 0.4976 at kappa 0 to 0.9497 at kappa 1;
- v2b context-2 AUROC rises from 0.5030 at kappa 0 to 0.6337 at kappa 1;
- v2b primary class contrast rises from 0.00589 to 0.07234;
- gap/amount KS and dt-bin TVD are generally within the 0.02 practical bound;
- receiver TVD exceeds 0.02 for every dataset (0.0317–0.1067), so Gate A fails;
- positionwise/bootstrap CI, iid equivalence, fresh DGP oracle, and 2/4/8-bin
  robustness were still `PENDING` at this initial validation step.

The receiver TVD failure is not tuned away. At kappa 1 it is especially large
for v2a (0.1067), consistent with finite test entities plus long receiver runs;
the DGP has theoretical uniform stationarity but does not satisfy the
preregistered empirical tolerance at the fixed full-data sample. Per the
directive, no learned baseline or CoF training follows this failure.

## Phase 4–5 — Generator/evaluation contracts

Implementation status: PARTIAL; execution is gated by Phase 3.

Implemented common sampling plan, empirical conditional iid and block adapters,
conditional CTGAN/TVAE adapters, independent/joint Markov seams, neural
sequence baseline skeleton, named v2 behavior summaries, coherence references,
direct sequence utilities, fidelity/statistical helpers, and atomic artifact
manifests.

```text
python -m pytest -q \
  tests/test_generator_contract.py tests/test_coherence_v2.py \
  tests/test_artifact_store.py
7 PASS (1.46 s)
```

Combined focused suite:

```text
python -m pytest -q tests/test_soft_g_grad.py tests/test_soft_g_mask.py \
  tests/test_sampler_padding.py tests/test_benchmark_v2_dgp.py \
  tests/test_generator_contract.py tests/test_coherence_v2.py \
  tests/test_artifact_store.py
49 PASS (13.63 s pytest; 14.20 s wall clock)
```

Full legacy-inclusive suite:

```text
python -m pytest -q
54 PASS, 20 FAIL (16.40 s pytest; 17.18 s wall clock)
```

The failures are not hidden. Most are stale legacy call sites that do not yet
pass the newly mandatory `valid_mask`; additional legacy phase-3/phase-4 tests
expect obsolete `L_coh_raw` behavior or cannot unpickle the legacy
`HashEncoder`. Therefore the repository-wide suite is FAIL even though the
focused v2 suite passes.

## Phase 6 — CoF adapter and smoke

Status: NOT RUN (mandatory gate failure).

The adapter source exists at `generators/cof_seqgen_adapter.py`, but executing
CoF smoke would violate the forced-stop rule. Independently, the workstation
has no visible CUDA device, and CoF smoke is GPU-only by instruction.

Baseline smoke was likewise not run because the required ordering is
full-data CPU gate PASS → baseline smoke → CoF smoke.

## Initial forced-stop and Definition of Done

At the initial forced stop, `IMPLEMENT_AND_SMOKE` Definition of Done was not
met. The then-current blocking evidence was:

1. Gate A receiver single-row TVD fails the fixed 0.02 tolerance.
2. Gate B/C/E and positionwise/bootstrap-CI evidence were incomplete and failed
   closed rather than being marked PASS.
3. Repository-wide pytest was 54 PASS / 20 FAIL.
4. At that time NVIDIA-SMI and PyTorch did not expose a GPU in the preflight
   shell. This was later superseded by the user's confirmation of four
   available GPUs and is not the reason learned smoke remains gated.
5. Required gate files such as `iid_bootstrap_{2,4,8}bin.csv`,
   `dgp_oracle.csv`, and positionwise stationarity outputs did not yet exist.

No benchmark parameter was changed after observing any baseline or CoF result;
neither was run. No FULL_EXPERIMENT or 5-seed training was run.

## Full-run commands not executed

The exact commands are recorded in `docs/benchmark_v2/runbook.md`. They remain
unauthorized in IMPLEMENT_AND_SMOKE mode and are additionally blocked by the
failed gate.

## Final repository state

- branch: `benchmark-v2-redesign`
- base/source snapshot SHA: `0066ad321182b21c5eb46c71c69679c3a97192b1`
- tag `legacy-kappa-v1` points to that same SHA
- worktree: dirty with v2 implementation changes
- pre-existing unstaged deletions preserved:
  `images/tabdiff_demo.gif`, `images/tabdiff_demo.mp4`,
  `images/tabdiff_flowchart.jpg`
- legacy `data/`, `results/`, `logs/`, and `figs/` assets were not moved,
  overwritten, deleted, or staged

Primary new source groups are `benchmarks/`, `generators/`, `experiments/`,
`configs/benchmark_v2/`, v2 evaluation modules under `eval/`, v2 scripts,
focused tests, and benchmark-v2 documentation. Runtime outputs are isolated
under `data/benchmark_v2*` and `artifacts/benchmark_v2/`.

## CPU gate-completion checkpoint

Pre-checkpoint state on branch `benchmark-v2-redesign`:

```text
base SHA: 0066ad321182b21c5eb46c71c69679c3a97192b1
modified: docs/v2_implementation_report.md, models/{cof_seqgen,sampler,soft_g,teacher}.py,
          tests/test_soft_g_grad.py
untracked source: benchmarks/, configs/, docs/benchmark_v2/, eval v2 modules,
                  experiments/, generators/, v2 scripts/tests, environment files
pre-existing unstaged deletion: images/tabdiff_demo.{gif,mp4},
                                images/tabdiff_flowchart.jpg
```

Only source/config/test/docs/environment files are included in the checkpoint.
Runtime data/artifacts, checkpoints, and legacy results remain excluded.

Checkpoint result:

```text
checkpoint SHA: 963a40160ee3304dd97367603e1f3ce656002a79
branch: benchmark-v2-redesign
legacy-kappa-v1: 0066ad321182b21c5eb46c71c69679c3a97192b1 (unchanged)
post-checkpoint status: only the three pre-existing image deletions remained
```

## Receiver gate CPU continuation

The receiver-only artifact diagnosis reused the existing full-data v2.0
artifacts and completed 1,000 entity-label permutations for all 20
scenario/kappa/split combinations.

```text
python -m scripts.diagnose_receiver_v2 \
  --data-root data/benchmark_v2 \
  --output-root artifacts/benchmark_v2/receiver_diagnosis \
  --permutations 1000
PASS (22.48 s)
```

The first unoptimized attempt exceeded the 30-second budget and was interrupted
without producing accepted output. The implementation was optimized by deriving
the negative-cluster totals from all-entity totals; the accepted rerun completed
within budget.

Notable null-calibration evidence:

- v2b kappa 1 test pooled TVD 0.05564, permutation median 0.05356,
  excess 0.00208, tail probability 0.318;
- v2a kappa 1 test pooled TVD 0.10670, permutation median 0.04061,
  excess 0.06609, tail probability 0.001;
- permutation tails are explicitly not treated as equivalence tests.

Artifacts are under `artifacts/benchmark_v2/receiver_diagnosis/`.

## Legacy pytest failure classification and repair

The 20 failures from the previous run were classified as:

- production call-site migration omissions: mandatory `valid_mask` absent;
- stale expectations: obsolete `L_coh_raw` and `info["g_std"]`;
- serialization compatibility: `__main__.HashEncoder`;
- fixture mismatch: preserved left-padded v1 data passed to a right-padding GRU;
- actual mask regression: none after focused mask tests.

Repairs preserve the mandatory mask contract. Every production call site now
passes an explicit real mask; no `valid_mask=None` fallback was added. Legacy
pickle files remain unchanged and are loaded through a compatibility alias.
Legacy left-padded fixtures are migrated to right padding in memory only.
The intended loss-info contract is `L_diff`, `L_coh`, and `L_label`; fixed
`g_std` is tested directly as a model buffer.

```text
python -m pytest -q
77 passed, 1 skipped, 13 warnings (112.66 s; 113.59 s wall)
```

The one skipped obsolete seam was subsequently replaced by a current
`CoherenceTeacher` buffer-consistency test.

## CPU Gate B/C/E, positionwise evidence, and context ladder

Status: **FAIL** (all requested evidence computed; no PENDING remains).

```text
python -m scripts.complete_cpu_gates_v2 \
  --config configs/benchmark_v2/main.yaml \
  --data-root data/benchmark_v2 \
  --output-root artifacts/benchmark_v2/gates
FAIL under tmux (990.46 s; 991.15 s wall)

python -m scripts.positionwise_simultaneous_v2 --resamples 2000
FAIL (3.18 s; 3.33 s wall)
```

Results:

- Gate B conditional empirical i.i.d.: FAIL. v2a passes the 4/8-bin cells,
  while v2b fails multiple 4/8-bin cells. At v2b kappa 1, 8-bin C1 mean gap
  is 0.05247, i.i.d. mean is 0.03242, and the improvement upper confidence
  bound 0.02364 exceeds the 0.00787 equivalence margin.
- Gate C fresh parametric oracle: PASS in all required 4/8-bin cells. Each
  oracle draw preserves the fixed test label/length plan and uses a fresh
  production DGP nonce.
- Gate E 2/4/8-bin robustness: FAIL. v2b 8-bin synthetic evaluations have
  insufficient per-bin counts even when the fitted reference has eight
  effective bins.
- Positionwise stationarity: FAIL. The 2,000-resample entity-cluster Gaussian
  multiplier bootstrap gives a joint max-absolute-deviation band over all
  positions. All ten scenario/kappa cells fail; late positions have few fraud
  entities and simultaneous radii 0.155–0.182.
- The CPU-only empirical block ladder (block lengths 1, 2, 4, 8, full) is
  recorded for both 4/8-bin evaluations.

Artifacts:

- `artifacts/benchmark_v2/gates/positionwise_stationarity.csv`
- `artifacts/benchmark_v2/gates/positionwise_stationarity_simultaneous.csv`
- `artifacts/benchmark_v2/gates/positionwise_stationarity_summary.csv`
- `artifacts/benchmark_v2/gates/iid_bootstrap_{2,4,8}bin.csv`
- `artifacts/benchmark_v2/gates/dgp_oracle.csv`
- `artifacts/benchmark_v2/gates/c0_real_split.csv`
- `artifacts/benchmark_v2/gates/bin_robustness.csv`
- `artifacts/benchmark_v2/gates/context_ladder/context_length_curve.csv`
- `artifacts/benchmark_v2/gates/cpu_gate_completion.json`

`gate_report.json` and `gate_report.md` now contain the computed results:
positionwise FAIL, Gate B FAIL, Gate C PASS, and Gate E FAIL. Overall remains
FAIL, so the learned-model forced stop remains in effect.

## Final test verification

```text
python -m pytest -q
78 passed, 14 warnings (100.33 s; 101.25 s wall)

python -m pytest -q tests/test_receiver_gate_calibration.py
2 passed (0.39 s)

python -m compileall -q benchmarks scripts models tests
PASS
```

The second focused test file was added after the 78-test full-suite run.

## v2.1 receiver-gate candidate calibration

Status: PASS for the receiver signed-frequency component; candidate remains
fail-closed overall.

```text
python -m scripts.calibrate_receiver_gate_v2_1 \
  --config configs/benchmark_v2/main_v2_1_candidate.yaml \
  --seeds 30
PASS under tmux (1,203.02 s; 1,203.44 s wall)
```

This generated 4N receiver-only samples for 30 seeds in each of v2a/v2b,
kappa 0/1 (120 replicates). A Bonferroni simultaneous normal interval is
computed over signed category-frequency differences, with each entity as the
independent cluster. The fixed stress test replaces 5% of valid fraud rows by
category 0.

All four cells have null false-fail rate 0/30 and leakage detection power
30/30, satisfying the candidate limits of <=0.05 and >=0.90. Artifacts:

- `artifacts/benchmark_v2/receiver_diagnosis/v2_1_candidate/calibration_raw.csv`
- `artifacts/benchmark_v2/receiver_diagnosis/v2_1_candidate/calibration_summary.csv`
- `artifacts/benchmark_v2/receiver_diagnosis/v2_1_candidate/calibration_manifest.json`

The candidate is not promoted and 4N full data are not generated. The 4N
single-row AUROC entity-bootstrap is not calibrated, and the v2.0 Gate B
failure is substantive rather than a low-N uncertainty failure: for v2b kappa
1 at 8 bins, the improvement upper bound is 0.02364 against a 0.00787 margin.
Increasing N would not make this required scientific equivalence true.
Accordingly, a full v2.1 Gate A–E run is inappropriate until Gate B's
preregistered expectation is reviewed; the stop remains fail-closed.

## Final closure

Final repository-wide verification after adding the candidate tests:

```text
python -m pytest -q
80 passed, 14 warnings (106.19 s; 107.07 s wall)
```

Current mandatory benchmark status remains **FAIL**:

- Gate A raw receiver TVD and positionwise stationarity: FAIL;
- Gate B conditional empirical i.i.d. equivalence: FAIL;
- Gate C fresh DGP oracle: PASS;
- Gate D has passing and failing scenario/kappa subchecks as recorded;
- Gate E bin robustness: FAIL.

No baseline, CoF, CTGAN, TVAE, neural baseline, GPU command, or
`FULL_EXPERIMENT` was run during the CPU continuation. The learned smoke
commands remain gated, and the specified `scripts.run_benchmark_v2` runner is
not present in the repository; this is recorded in the runbook instead of
silently substituting a legacy runner.

GPU availability clarification: four NVIDIA GPUs are available according to
the user. Hardware availability does not override the CPU gate. Once all v2.2
CPU gates pass and preregistration is committed, one independent smoke job per
selected idle GPU is allowed; otherwise no learned job is run.

## v2.2 CPU redesign continuation

Mode remained CPU-only until every v2.2 gate could be decided. No learned
result was available when the endpoint and thresholds were selected.

### GPU status correction

The user confirmed that four NVIDIA GPUs are available on the laboratory
workstation. The original preflight `nvidia-smi` failure remains recorded as
the observation in that shell. It is not interpreted as a hardware blocker.
Learned smoke was not run because mandatory CPU gates failed. GPU selection
and smoke are authorized only after a complete v2.2 CPU PASS and a committed
preregistration.

### Gate B forensic decomposition

```text
python -m scripts.gate_b_forensic_v2 \
  --seeds 10 --bootstrap-resamples 2000
COMPLETE under tmux (1,234.95 s; 1,235.35 s wall)
```

At v2b kappa 1, real/C1 `joint_alignment` standard deviation is 0.05349,
whereas i.i.d. and block-1 collapse to 0.01291/0.01290. I.i.d. loses two
8-bin tails. Its old dropped-bin score is 0.03242 versus C1 0.05219, but the
occupancy-penalized score is 0.28242. H-B1 is supported, H-B2 is rejected, and
H-B3 identifies empty-bin exclusion as an evaluator bug rather than a C1
construction error.

Continuous association recovery restores the intended ordering. At kappa 1,
mean recovery errors are:

```text
C0 floor .00436, oracle .00241, block-full .00429, block-8 .00954,
block-4 .01570, block-2 .03349, block-1 .07222, iid .07221, C1 .07217
```

The detailed report is
`docs/benchmark_v2/gate_b_forensic_diagnosis.md`; artifacts are under
`artifacts/benchmark_v2/gate_b_forensics/`.

### v2.2 preregistration and data

Alternative A was selected without learned results: continuous association
recovery is primary, 8-bin macro coherence confirmatory, and 4/2-bin tables
descriptive. Invalid generator support is an invalid model score and cannot be
excluded to improve a score.

Training N remains 31,951. Only evaluation/audit N is 31,956. New data use
`data/benchmark_v2_2`; v2.0/v2.1 data remain untouched.

```text
python -m scripts.generate_benchmark_v2 \
  --config configs/benchmark_v2/main_v2_2_candidate.yaml \
  --output-root data/benchmark_v2_2
PASS under tmux (281.02 s)
```

The first generated metadata exposed a hard-coded v2.0 schema label. The
generator was corrected to derive dataset/manifest schema from the config, and
the new v2.2 path was deterministically regenerated:

```text
same command after schema fix
PASS under tmux (275.60 s)
```

Final metadata records `benchmark_v2.2-candidate`, n_train 31,951, and n_test
31,956. The preregistration is
`docs/benchmark_v2/preregistered_analysis_v2_2.md`.

### Positionwise and receiver calibration

Position rules were fixed by power analysis before examining the 30-seed
output. For 16 hard prefix comparisons and margin 0.04, Bonferroni z=2.955
implies at least 1,147 positive at-risk entities; the preregistered threshold
is 1,500.

```text
python -m scripts.calibrate_positionwise_v2_2 --seeds 30 --workers 4
PASS under tmux (307.00 s; 307.43 s wall)
```

All 64 hard position cells and all 12 early/middle/late cells pass. The 64
later low-at-risk cells are descriptive.

```text
python -m scripts.calibrate_receiver_gate_v2_2 --seeds 100 --workers 4
PASS under tmux (1,045.76 s; 1,046.18 s wall)
```

For all four scenario/kappa cells, clean false failures are 0/100 with exact
95% upper bound 0.03622. Across leakage categories 0/1/31/63, 2% leakage has
minimum power 0.98 and exact lower bound 0.92962; 5% leakage has power 1.00
and lower bound 0.96378.

### v2.2 full CPU gate

The full-data audit completed in 132.88 seconds (133.65 wall). Revised Gate
B/C/E used five independent sampling/oracle seeds at each kappa, run as five
independent tmux jobs:

```text
kappa 0.0: 496.42 s
kappa 0.3: 493.42 s
kappa 0.5: 498.57 s
kappa 0.7: 493.86 s
kappa 1.0: 493.55 s
```

Final result: **FAIL**.

```text
Gate A single-row AUROC equivalence                 FAIL
Gate A signed receiver frequency                    PASS
Gate A receiver 100-seed calibration                PASS
Gate A positionwise and segments                    PASS
Gate B continuous iid/block ordering (26 checks)    PASS
Gate C parametric oracle (5 checks)                  PASS
Gate D real joint association                       PASS
Gate D channel negative controls                    FAIL
Gate E reference/oracle bin validity                PASS
Gate E bad-generator invalid-score routing          PASS
```

The single-row equivalence intervals are not wholly inside [0.48, 0.52].
Examples are v2b kappa 0 logistic [0.47138, 0.50063] and v2b kappa 1
logistic/HGB upper bounds 0.52262/0.52170. No row-selection seed or interval
was changed after observing this result.

The v2b fanout standardized delta reaches 0.33527, so the required
velocity/gap/fanout/amount negative-control claim is not approximately zero.
This is fail-closed; no post-hoc tolerance is introduced.

The report is under `artifacts/benchmark_v2/v2_2_gate/gate_report.{json,md}`.
The candidate is not promoted.

### Final tests and conditional GPU decision

```text
python -m pytest -q
85 passed, 14 warnings (100.97 s; 101.83 s wall)

python -m compileall -q benchmarks eval generators scripts tests
PASS (0.03 s)
```

Because the v2.2 CPU gate is FAIL:

- `scripts.run_benchmark_v2` was not implemented; its implementation was
  conditional on a complete CPU PASS;
- GPU selection was not performed;
- conditional CTGAN, TVAE, neural baseline, and CoF smoke were not run;
- no 5-seed experiment, sweep, or `FULL_EXPERIMENT` was run.

The reason is mandatory CPU gate failure, not GPU hardware availability.

## v2.3 candidate amendment

The v2.2 config, preregistration, data, artifacts, and continuous primary
endpoint remain unchanged. No learned result was available or used. All new
runtime outputs are isolated under `artifacts/benchmark_v2_3/`, and the
amendment config and documents use explicit v2.3 paths.

### Fanout semantic audit

Code inspection confirmed that fanout counts unique receiver categories in a
gap-derived cumulative-time window. It is therefore a cross-channel summary.
The no-model intervention exchanged intact receiver paths within label/length
strata, destroying their alignment with the fixed gap paths.

```text
python -m scripts.audit_fanout_semantics_v2_3 --permutations 10
PASS (490.91 s; 491.32 s wall)
```

At v2b kappa 1, the joint-alignment delta fell from 0.06959 to 0.00022 and
the fanout delta from -0.34947 to -0.04086. Restoring the original alignment
restored both exactly. Raw receiver counts and intact path multisets were
unchanged; repeat rate, run length, fixed-step unique receiver, raw gap,
gap autocorrelation, gap-only window velocity, and raw amount controls were
preserved to at most `1.33e-15`.

Decision: `joint_alignment` is the primary joint positive control and
time-window fanout is a secondary joint positive control. Fanout is not a
negative control. Detailed evidence is in
`docs/benchmark_v2/fanout_semantic_audit.md` and
`artifacts/benchmark_v2_3/fanout_semantic_audit/`.

### True channel-only negative controls

```text
python -m scripts.audit_channel_controls_v2_3
PASS (62.55 s wall, final run)
```

The fixed absolute standardized-delta threshold is 0.10. Across raw gap,
gap autocorrelation, gap-only window velocity, receiver repeat probability,
receiver run length, fixed-step unique receiver, and raw amount, the maximum
was 0.07148. The separate retained signed-frequency receiver gate is PASS.
Fanout and every other cross-channel window summary are excluded.

Artifacts:

- `artifacts/benchmark_v2_3/channel_controls/channel_only_negative_controls.csv`
- `artifacts/benchmark_v2_3/channel_controls/channel_control_manifest.json`

### 100-clean-seed AUROC calibration

The design in `configs/benchmark_v2/main_v2_3_candidate.yaml` fixed the
samples, seed family, `[0.48, 0.52]` interval, classifiers, validation-only
orientation, leakage features/magnitudes/categories/directions, and exact
binomial criteria before execution.

```text
python -m scripts.calibrate_auroc_gate_v2_3 --seeds 1 --workers 1 \
  --output-root artifacts/benchmark_v2_3/auroc_calibration_smoke
contract smoke COMPLETE (114.32 s; one seed is intentionally ineligible)

python -m scripts.calibrate_auroc_gate_v2_3 --seeds 100 --workers 8
COMPLETE under tmux (26,621.88 s; 26,622.64 s wall)

python -m scripts.summarize_auroc_calibration_v2_3
COMPLETE (1.36 s wall)
```

Result completeness: 2,400 clean rows and 7,200 leakage rows, with all 100
seeds in every scenario/kappa/classifier/method cell.

```text
A: familywise false failures 99/100, exact CI [0.9455, 0.9997]
B: familywise false failures 90/100, exact CI [0.8238, 0.9510]
C: familywise false failures 89/100, exact CI [0.8117, 0.9438]
```

All three fail the clean upper-bound criterion of 0.05. All also fail the
feature/classifier-specific power criteria: the best 2% leakage exact lower
bound is 0.0592 versus 0.80 required, and the best 5% lower bound is 0.7083
versus 0.90 required. Test labels were never used for orientation.

No AUROC method is selected (`selected_method=null`). The interval, DGP,
features, endpoint, and thresholds were not changed after seeing the result.
Detailed evidence is in `docs/benchmark_v2/auroc_gate_calibration_v2_3.md`
and `artifacts/benchmark_v2_3/auroc_calibration/`.

### v2.3 CPU gate and forced stop

```text
python -m scripts.build_v2_3_cpu_gate
COMPLETE (0.03 s); overall FAIL
```

Retained checks all PASS: continuous primary association, iid/block-1/C1
equivalence, block-2/4/8/full ordering, oracle/C0 equivalence, signed receiver
frequency, position/segment stationarity, reference/oracle bin validity, and
invalid-generator support routing. Revised fanout and channel-only checks
PASS. Calibrated single-row AUROC FAILS.

The authoritative result is
`artifacts/benchmark_v2_3/gates/gate_report.json`, with
`learned_smoke_authorized=false`.

Consequently:

- no GPU was selected or used;
- CTGAN, TVAE, neural baseline, and CoF smoke were not run;
- the conditional common learned runner was not invoked;
- no 5-seed experiment or sweep was run.

This is the required benchmark gate forced stop, not a GPU availability
failure.

### v2.3 final verification

```text
python -m pytest -q
89 passed, 14 warnings (101.65 s; 102.58 s wall, final rerun)

python -m compileall -q benchmarks eval generators scripts tests
PASS (0.03 s)

git diff --check
PASS
```

The final artifact index contains 21 files and is stored at
`artifacts/benchmark_v2_3/artifact_index.json`. A search of the v2.3 artifact
tree found no CTGAN, TVAE, neural, CoF, sample, or checkpoint artifact.
Pre-existing unstaged image deletions remain untouched and are excluded from
the v2.3 commit.

## v2.4 integrated-global-maximum AUROC amendment

The first classifier-specific rank-49 Step-A command completed immediately
before the superseding user amendment arrived. Its files are preserved, not
deleted or overwritten, under
`artifacts/benchmark_v2_4/pre_amendment_step_a_rank49/` with a
`SUPERSEDED.md` marker. Its classifier thresholds and 13x multiplier are not
used by v2.4.

The global-maximum amendment was committed before its Step A:

```text
commit a002c16e6e38e127448d0bc8921bc321a609d649
commit 3510e6134b52dd67539133d86ed8cf155b1edc65
```

The latter correction prevents the obsolete two-classifier-mean alternative
from changing the new fixed maximum statistic.

### v2.4 Step A — immutable-artifact reanalysis

Status: PASS (procedure complete; not a CPU benchmark gate).

```text
python3 -m pytest -q tests/test_reanalyze_auroc_gate_v2_4.py
5 passed (0.43 s pytest)

python3 -m scripts.reanalyze_auroc_gate_v2_4 \
  --config configs/benchmark_v2/main_v2_4_candidate.yaml \
  --output-root artifacts/benchmark_v2_4/step_a
COMPLETE (0.58 s wall, 109832 KiB maximum RSS)
```

The integrated statistic is one maximum over four scenario/kappa cells and
both classifiers. The 100-source-seed observed maximum was 0.0238883; the
larger conservative Gaussian-Bonferroni rank-200 proxy was 0.0326937.

The largest modeled requirement was 27x for gap/logistic at 2% (modeled power
0.82). The fixed 1.25 safety factor selects a 34x audit multiplier:
1,086,334 train, 271,626 orientation-validation, and 1,086,504 test entities.
The calculation is explicitly an optimistic lower bound for variance scaling.
No variance-reduction alternative or descriptive 2% downgrade is triggered.

Artifacts:

- `artifacts/benchmark_v2_4/step_a/step_a_manifest.json`
- `artifacts/benchmark_v2_4/step_a/threshold_proxy.json`
- `artifacts/benchmark_v2_4/step_a/threshold_proxy_dimensions.csv`
- `artifacts/benchmark_v2_4/step_a/required_multiplier_summary.csv`
- `artifacts/benchmark_v2_4/step_a/power_multiplier_search.csv`
- `artifacts/benchmark_v2_4/step_a/observed_shift_summary.csv`
- `artifacts/benchmark_v2_4/step_a/clean_global_t_by_seed.csv`

The original v2.3 CSV SHA-256 values match the hashes recorded in the Step-A
manifest. No DGP or learned generator was rerun, and no learned result was
used.

### v2.4 Step B — one-seed engineering smoke

Status: PASS for the required pre-full engineering smoke. This is not a gate
calibration decision.

```text
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 \
python3 -m scripts.calibrate_auroc_gate_v2_4 \
  --mode smoke --workers 8 \
  --output-root artifacts/benchmark_v2_4/step_b_smoke
COMPLETE in detached tmux (21.10 s wall)
```

The smoke completed all 12 cell-seed checkpoints: four selected-N calibration
cells, four selected-N validation cells with every fixed leakage feature and
magnitude, and four nested base-N cells. Selected-N calibration cells averaged
3.36 seconds and validation cells averaged 19.98 seconds. The ideal projected
full 200+200 runtime is 2,334 seconds (0.648 hours) with eight workers, shorter
than the v2.3 run. Maximum worker RSS was 319,168 KiB and the conservative
eight-worker concurrent projection is 2,553,344 KiB.

The required variance-floor risk diagnostic is adverse but non-decisional:
base-N global T was 0.007748, the `1/sqrt(34)` projection was 0.001329, and
selected-N global T was 0.002110 (ratio 1.588). One nested seed cannot estimate
a variance floor. This risk is preserved without changing N, the rank-200
maximum threshold, leakage operators, or hard/descriptive classifications.

Artifacts:

- `artifacts/benchmark_v2_4/step_b_smoke/run_manifest.json`
- `artifacts/benchmark_v2_4/step_b_smoke/smoke_projection.json`
- `artifacts/benchmark_v2_4/step_b_smoke/calibration_manifest.json`
- `artifacts/benchmark_v2_4/step_b_smoke/completed_tasks.jsonl`
- `artifacts/benchmark_v2_4/step_b_smoke/checkpoints/`
- `artifacts/benchmark_v2_4/step_b_smoke/run.log`

### v2.4 Step B — 200+200 calibration

Status: PASS.

```text
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 \
python3 -m scripts.calibrate_auroc_gate_v2_4 \
  --mode full --workers 8 \
  --output-root artifacts/benchmark_v2_4/auroc_calibration
PASS in detached tmux (2,382.11 s / 39:42.11 wall)
```

All 1,600 cell-seed tasks were checkpointed: 800 calibration and 800
validation. The final tables contain 3,200 clean cell rows, 9,600 leakage cell
rows, 400 clean global-T rows, 2,400 leakage global-T rows, and 12 hard power
cells. Maximum worker RSS was 414,404 KiB.

The interpolation-free rank-200/200 threshold is `0.0040435352475451936`.
Independent clean validation had 2 false failures among 200 seeds:
rate 0.01 and Clopper-Pearson exact 95% CI
`[0.00121335, 0.03565467]`. The upper bound is below 0.05.

Every amount, continuous-gap, and receiver-category classifier/magnitude cell
detected 200/200 injected seeds. Their common exact 95% lower bound is
`0.98172466`, exceeding both the 2% requirement 0.80 and 5% requirement 0.90.
Each fixed receiver direction/category also detected 50/50; amount and gap
directions each detected 100/100.

An independent verification confirmed all row counts, exact seed ranges,
strict `T > t_hat` events, no test-label orientation, and that every realised
train/orientation-validation/test injection differs from its requested
probability by at most half an integer row. No N, threshold definition,
operator, or hard/descriptive classification changed after Step B began.

Artifacts:

- `artifacts/benchmark_v2_4/auroc_calibration/calibration_manifest.json`
- `artifacts/benchmark_v2_4/auroc_calibration/run_manifest.json`
- `artifacts/benchmark_v2_4/auroc_calibration/clean_cell_results.csv`
- `artifacts/benchmark_v2_4/auroc_calibration/clean_global_t_by_seed.csv`
- `artifacts/benchmark_v2_4/auroc_calibration/leakage_cell_results.csv`
- `artifacts/benchmark_v2_4/auroc_calibration/leakage_global_t_by_seed.csv`
- `artifacts/benchmark_v2_4/auroc_calibration/leakage_power_summary.csv`
- `artifacts/benchmark_v2_4/auroc_calibration/leakage_operator_breakdown.csv`
- `artifacts/benchmark_v2_4/auroc_calibration/checkpoints/`
- `artifacts/benchmark_v2_4/auroc_calibration/run.log`

### v2.4 Step C — CPU gate assembly

Status: PASS.

```text
python3 -m scripts.verify_v2_4_cpu \
  --output-root artifacts/benchmark_v2_4
112 passed, 14 warnings (54.85 s pytest)
compileall PASS (0.03 s)

python3 -m scripts.build_v2_4_cpu_gate \
  --output-root artifacts/benchmark_v2_4/gates
PASS
```

Every retained v2.2/v2.3 check is PASS, including continuous primary
association, iid/block ordering, oracle, receiver signed frequency,
position/segment stationarity, valid support and invalid-score routing,
fanout as a secondary joint positive control, and true channel-only negative
controls. The only replaced component, integrated global-maximum AUROC, is
PASS. Repository-wide pytest and compileall are also PASS.

The authoritative gate is
`artifacts/benchmark_v2_4/gates/gate_report.json`:
`learned_smoke_authorized=true` and `full_experiment_authorized=false`.
The common `scripts.run_benchmark_v2` runner and its contract tests are in the
112-test PASS result. No learned generator was run before this gate or used in
rule selection.

### Conditional GPU learned smoke

Status: COMPLETE; all four artifact/sequence contracts PASS.

The final preregistration was committed before GPU execution:
`f4ad45b4a59145f491e8910c90ed0f3bcf46b2df`.

Pre-run `nvidia-smi` found four RTX 3090 GPUs. GPU 0 had an unrelated Python
compute process at 18% utilization, so it was not used. GPUs 1, 2, and 3 were
idle. CTGAN, TVAE, and neural ran independently on GPUs 1, 2, and 3; CoF ran
on GPU 3 after neural released it. DDP was not used.

Each invocation used the same bounded common runner:

```text
CUDA_VISIBLE_DEVICES=<1|2|3> python3 -m scripts.run_benchmark_v2 \
  --mode smoke \
  --config configs/benchmark_v2/smoke.yaml \
  --artifact-root artifacts/benchmark_v2_4/learned_smoke \
  --scenario joint_semimarkov_v2b \
  --kappas 1.0 \
  --generators <conditional_ctgan|conditional_tvae|neural_sequence_baseline|cof> \
  --seeds 1 --steps 100 --device cuda:0
```

Results:

| Generator | Wall | Runner total | Peak CUDA | Checkpoints | Contract |
|---|---:|---:|---:|---:|---|
| conditional CTGAN | 14.10 s | 11.29 s | 45,551,104 B | 2 | PASS |
| conditional TVAE | 12.51 s | 9.71 s | 26,598,912 B | 2 | PASS |
| neural sequence | 4.88 s | 2.22 s | 149,859,840 B | 1 | PASS |
| CoF | 3.89 s | 1.72 s | 124,947,456 B | 1 | PASS |

Every run saved `sample.npz`, checkpoint(s), `metrics.json`, `runtime.json`,
and a complete `manifest.json`. Independent reload verification confirmed 256
entities, prefix-contiguous masks, zero padding, supported discrete values,
the fixed seed/scenario/kappa, requested 100 steps, positive recorded peak
CUDA memory, and `full_experiment_authorized=false`.

Smoke TVDs are descriptive only. In particular, TVAE receiver TVD and CoF
gap/receiver TVDs are visibly high after this tiny smoke; they do not alter any
v2.4 metric, threshold, N, or gate rule.

Post-run `nvidia-smi` showed GPUs 1, 2, and 3 again idle at 0% utilization.
GPU 0's unrelated process was left untouched.

Artifacts:

- `artifacts/benchmark_v2_4/learned_smoke/conditional_ctgan/`
- `artifacts/benchmark_v2_4/learned_smoke/conditional_tvae/`
- `artifacts/benchmark_v2_4/learned_smoke/neural_sequence_baseline/`
- `artifacts/benchmark_v2_4/learned_smoke/cof/`
- `artifacts/benchmark_v2_4/learned_smoke/logs/`

No five-seed run, sweep, or FULL_EXPERIMENT was run.

## v2.4 IMPLEMENT_AND_SMOKE Definition of Done

Status: MET.

- pre-amendment artifacts are preserved and marked superseded;
- Step A used immutable v2.3 inputs and fixed the 34x audit N;
- Step B 200+200 global-maximum calibration is PASS;
- retained and revised CPU gates, full pytest, and compileall are PASS;
- final preregistration preceded learned execution;
- common learned runner and scope-rejection contract are implemented;
- CTGAN, TVAE, neural, and CoF one-seed GPU smokes are complete;
- no learned result changed the benchmark;
- no 5-seed FULL_EXPERIMENT or sweep was run;
- pre-existing unstaged image deletions remain untouched.

The v2.4 artifact index contains 1,679 files and is stored at
`artifacts/benchmark_v2_4/artifact_index.json`. It includes the 1,600 resumable
Step-B checkpoints and all smoke samples/checkpoints/manifests.
