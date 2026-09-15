# v2.6 single-factor validation-selection forensic analysis

## Scope and immutable inputs

This analysis explains the preserved selection result; it does not change it.
The analysis base is
`24d9ccec7e429f05cc3faceff0a5cb632d5219ac`. The v2.6 aggregate,
candidate, worker, trajectory, evaluation, config and authorization trees,
plus all v2.5 runtime and frozen data, were read only.

No GPU inventory query, CUDA call, model fit, model sample, data generation,
test/fresh-test access, TSTR, privacy analysis, or five-seed/full execution
was performed.

The principal frozen hashes are:

| Input | SHA-256 |
|---|---|
| single-factor aggregate terminal | `404c820d79cb3d17e21d617f5c255c0d1b97ed6fb7c94b9fdcd23682bdc5bf35` |
| aggregate artifact index | `7620c08d071dabc320c7698a58d313fcb6812df82c29ea6f23fcb12d9e1cd97c` |
| single-factor config | `707ecdbd37bc55f4a3b9ece603b70d26c8274015aa9cb97d5d376aab3d9ecb7c` |
| aggregate authorization | `35ef35ac3e6cfe92a9357438eafc8fc41fb08141fe04c2c0d2a87c0d51d72013` |
| candidate execution authorization | `30b01268d0a94006bde3235b4c09f7a2fe44ec0b2e144b7b0577e6e45bba5adb` |
| v2.5 config | `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3` |
| v2.5 `FINAL_COMPLETE` | `e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a` |
| v2.5 frozen manifest | `b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05` |

The preserved tree inventories are:

| Tree | Files | Bytes | Record digest |
|---|---:|---:|---|
| candidates | 30 | 6,871,674 | `03c0926523decd8114e7e63d77369ece1a065f7732441f76c17149d350720267` |
| workers | 9 | 10,363 | `8180c4d4fb576e1aa8953b4e5298b3ab717b1618adb080bab9b9fddea13fd819` |
| trajectories | 25 | 98,249,656 | `5814437ec364a70dc26c167ba43f548e62065934fe4eca0b49d374d71746897a` |
| evaluations | 15 | 3,439,134 | `32caf2c35a510b996a1211362df6bad7ae3c3fcc92bfee66c3b335d08465c55a` |
| aggregate attempt 001 | 6 | 38,468 | `fd43d5abd73dfb76289f3b58f465892a920d7e1a29579c701823c4502b6080b6` |

## Independent reproduction

The forensic harness does not call the aggregate evaluator. It independently:

1. verifies every frozen file/tree hash;
2. rebuilds the train-only SamplingPlan and verifies its hash;
3. loads the preserved validation split and nine stored synthetic outputs;
4. checks mask, padding, labels, lengths, SamplingPlan identity and discrete
   support;
5. recomputes amount/gap KS, both absolute standardized class effects and
   entity-weighted receiver signed frequency from their defining formulas;
6. compares all values and PASS/FAIL decisions with
   `selection_report.json`.

All 45 guard values match within `1e-12`; all 45 stored PASS/FAIL decisions
match; and all nine sample contracts pass. The final
`NO_PASSING_CANDIDATE` result is therefore reproduced.

Thresholds, in table order below, are:

- amount KS: `0.006081138155655141`;
- gap KS: `0.006387882975686154`;
- amount effect: `0.0363693454591819`;
- gap effect: `0.051540527275560376`;
- receiver effect: `0.02`.

`Δ` is candidate minus frozen control, so a negative delta is an improvement.
`P` and `F` mean PASS and FAIL.

## CTGAN

| Candidate | Amount KS | Gap KS | Amount effect | Gap effect | Receiver effect | Largest failure |
|---|---:|---:|---:|---:|---:|---|
| frozen control | 0.159947 F | 0.042131 F | 0.157397 F | 0.041325 P | 0.029209 F | amount KS, 26.30× |
| shared transformer | 0.131702 F (Δ−0.028245) | 0.064737 F (Δ+0.022606) | 0.165998 F (Δ+0.008600) | 0.079599 F (Δ+0.038274) | 0.026636 F (Δ−0.002573) | amount KS, 21.66× |
| temperature 0.5 | 0.160967 F (Δ+0.001020) | 0.041944 F (Δ−0.000187) | 0.152753 F (Δ−0.004645) | 0.057693 F (Δ+0.016368) | 0.027703 F (Δ−0.001507) | amount KS, 26.47× |

The shared all-train transformer is fitted once and copied into the two
separate-class generators
(`generators/single_factor_backends_v2_6.py:103`). It improves amount KS and
receiver effect but worsens both gap metrics and amount effect. The isolated
temperature intervention is applied to the restored synthesizers
(`experiments/single_factor_runner_v2_6.py:1519`) and produces only small,
mixed movements.

The persistent dominant error is the amount numeric transform/inverse path,
not the worker or selection layer. Neither single change provides an
all-channel correction.

## TVAE

| Candidate | Amount KS | Gap KS | Amount effect | Gap effect | Receiver effect | Largest failure |
|---|---:|---:|---:|---:|---:|---|
| frozen control | 0.079955 F | 0.184933 F | 0.094998 F | 0.238939 F | 0.295483 F | gap KS, 28.95× |
| categorical decode 0.75 | 0.077047 F (Δ−0.002908) | 0.007007 F (Δ−0.177925) | 0.095939 F (Δ+0.000941) | 0.001280 P (Δ−0.237659) | 0.005332 P (Δ−0.290151) | amount KS, 12.67× |
| channel weights 2/2/1 | 0.039711 F (Δ−0.040244) | 0.138833 F (Δ−0.046100) | 0.064132 F (Δ−0.030866) | 0.180046 F (Δ−0.058893) | 0.206270 F (Δ−0.089213) | gap KS, 21.73× |

The categorical-decode-only candidate changes TVAE discrete spans from the
upstream continuous/argmax-like path to temperature-0.75 multinomial choices
(`generators/candidate_model_backends_v2_6.py:244`). Receiver effect and gap
effect move from severe failures to PASS, and gap KS is reduced to only
`0.0006193` above its threshold. Amount KS/effect barely change and remain
far outside the gate.

The 2/2/1 loss-weight candidate improves every metric relative to control but
retains the original categorical decode and still has large gap/receiver
concentration. This cleanly separates a real categorical sampling-path
limitation from the unresolved amount numeric decoder limitation.

## CoF-SeqGen

| Candidate | Amount KS | Gap KS | Amount effect | Gap effect | Receiver effect | Largest failure |
|---|---:|---:|---:|---:|---:|---|
| frozen control | 0.374681 F | 0.022440 F | 0.268484 F | 0.258949 F | 0.018106 P | amount KS, 61.61× |
| epsilon prediction | 0.996989 F (Δ+0.622308) | 0.024659 F (Δ+0.002218) | 3.173450 F (Δ+2.904966) | 0.171499 F (Δ−0.087450) | 0.013467 P (Δ−0.004639) | amount KS, 163.95× |
| variance residual | 0.028221 F (Δ−0.346460) | 0.022440 F (Δ0) | 0.038314 F (Δ−0.230171) | 0.258949 F (Δ0) | 0.018106 P (Δ0) | gap effect, 5.02× |

The epsilon candidate changes only the amount loss target and matching reverse
path (`models/single_factor_components_v2_6.py:19` and `:232`). Its amount KS
and class effect become dramatically worse while receiver remains PASS. This
is strong evidence against adding updates to the same objective as the next
step.

The train-fitted zero-mean variance residual
(`models/single_factor_components_v2_6.py:425`) sharply improves both amount
metrics and leaves gap/receiver values bit-for-bit unchanged, exactly as an
amount-only intervention should. Amount effect narrowly misses by `0.0019443`,
but amount KS remains 4.64× its threshold and the unchanged gap metrics still
fail. The implementation boundary behaves consistently; the candidate is
simply insufficient for ALL-PASS.

## Hypothesis verdicts

### A. Evaluator or selection-runner defect — REFUTED

- 45/45 independently recomputed values match.
- Nine of nine mask/padding/label/length/SamplingPlan/support contracts pass.
- Stored candidate-result, sample, terminal, worker/tree and aggregate-index
  hashes all verify.
- Every recomputed guard decision and each model-level
  `NO_PASSING_CANDIDATE` conclusion matches the preserved report.

No orientation, label mapping, row weighting, padding or receiver
entity-weighting mismatch was found.

### B. Candidate implementation or sampling/decoder path — SUPPORTED

This verdict means path-dependent causal limitation, not a detected contract
or provenance bug.

- TVAE categorical decoding selectively repairs the intended gap/receiver
  channels.
- CoF residual sampling selectively repairs amount and leaves discrete
  channels unchanged.
- CoF epsilon objective selectively makes the amount channel much worse.
- CTGAN transformer/temperature interventions also move their intended
  boundaries, but do not resolve the dominant numeric mismatch.

The observed changes line up with the code boundaries and are inconsistent
with a generic evaluator/runner failure.

### C. Current model, objective and candidate-space limit — SUPPORTED

All nine finite preregistered candidates fail ALL-PASS. Improvements are
partial, trade one failed metric for another, or leave an untouched failed
channel. The result does not support more steps as a sufficient fix and does
not justify changing thresholds.

## Minimal v2.7 proposals

These are proposals only. Each must be separately preregistered and selected
using train/validation only.

### CTGAN

1. One amount-only train-fitted monotone empirical-quantile inverse-map
   candidate, preserving separate-class generation and the categorical path.
2. Separately, one train-only categorical marginal-logit calibration
   candidate; do not bundle it with the amount representation.

### TVAE

1. Freeze the successful 0.75 categorical decoder as control and test one
   amount-only train-fitted inverse-decoder candidate.
2. Separately preregister one bounded categorical-temperature candidate for
   the small remaining gap-KS miss, keeping loss weights fixed.

### CoF-SeqGen

1. Use variance-residual sampling as control and test one train-fitted
   non-Gaussian empirical-residual amount sampler.
2. Separately test one train-only gap-logit marginal-bias candidate, keeping
   the amount sampler and architecture fixed.

No threshold relaxation, test calibration, implementation or rerun is part of
this analysis.

## Evidence artifacts

- `artifacts/benchmark_v2_6/selection_single_factor/forensic_attempt_001/forensic_evidence.json`
- `artifacts/benchmark_v2_6/selection_single_factor/forensic_attempt_001/candidate_guard_evidence.csv`
- `artifacts/benchmark_v2_6/selection_single_factor/forensic_attempt_001/forensic_artifact_index.json`
- `artifacts/benchmark_v2_6/selection_single_factor/forensic_attempt_001/FORENSIC_COMPLETE.json`

Their SHA-256 values are, respectively:

- `5563f450ca8cbbb657acfea101e21b4881496e31c862f499e44e32c2048c0db9`;
- `917fda710cd9bf2a7e0b97f2d7529a055794384481a1506365e766682b77c7f8`;
- `55ace91249b655ce39124176eaa5a529c28ce1c22d752a4eca0af2e141479f3b`;
- `0c2e89a4124d35ccfa565ab504fbedf8819bed2dfe649370f811efe51a408703`.

## Verification

- focused forensic/aggregate/selection tests: 43 passed;
- repository-wide tests: 292 passed, 23 existing third-party warnings;
- `compileall`: PASS;
- `git diff --check`: PASS.
