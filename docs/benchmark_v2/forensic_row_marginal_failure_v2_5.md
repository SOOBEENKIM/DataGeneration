# v2.5 row-marginal failure forensic analysis

## Scope and conclusion

This is a read-only forensic analysis of the completed v2.5 experiment. It
does not revise the endpoint, thresholds, C2 decision rule, DGP, configuration,
model, evaluator, or any runtime artifact. No full runner, frozen-data
generation, CPU/GPU baseline, learned training, retry, or sweep was run.

The evidence supports the following conclusions:

- **A — evaluator/guard implementation defect: REFUTED.** All 59 evaluated
  samples independently reproduce all five stored row-guard statistics with a
  maximum absolute difference of `0.0`. Their label, length, mask, zero-padding,
  row-order, and train-support contracts also pass.
- **B — one common learned sampling/adaptor defect: REFUTED.** The shared
  SamplingPlan contract passes, the runner passes the direct `adapter.sample`
  result to the evaluator without reassignment, and the four learned
  generators use three distinct sampling implementations. Their failure
  fingerprints differ materially. This verdict concerns a defect common to
  all four learned generators; it does not claim that a model-pair-specific
  issue is impossible.
- **C — output-distribution fidelity at the frozen model configurations:
  SUPPORTED.** Every evaluated learned seed fails genuine stored-sample
  distribution checks, with `amount_ks` and `gap_ks` failing in all 18
  evaluated learned seeds. All recorded requested updates completed and no
  learned seed reached the wall cap. The evidence identifies output fidelity
  at the frozen configuration, but does not separately identify optimization,
  objective, or architectural capacity as the mechanism.

The original C2 result remains unchanged and **NOT EVALUABLE** because the
preregistered primary CTGAN comparator is INVALID. No partial mean, threshold
change, or reinterpretation is made here.

Machine-readable evidence:

- `forensic_row_marginal_failure_v2_5.csv`: one row for each of five
  components in every evaluated generator/seed, plus one `NOT_COMPUTED` row
  for each cancelled seed. It contains the threshold, independently
  recomputed statistic, stored statistic, replay difference, real and
  synthetic descriptive values, PASS/FAIL reason, all hard-guard states, and
  sample/evaluation/metrics/runtime/terminal SHA-256.
- `forensic_row_marginal_failure_v2_5.json`: the same rows plus all 65
  seed-level records, manifests, file timestamps, source hashes, distribution
  summaries, preservation checks, and hypothesis verdicts.

At generation time their SHA-256 values were:

| File | SHA-256 | Bytes |
|---|---|---:|
| forensic CSV | `99e4880ee5d41071048d41e764a6232f13afd3a997d59b57f5980838a8917358` | 201,368 |
| forensic JSON | `524aeeac8fc49c2468b2b729707d095188c88ac24b180ddcad65d243452f51f7` | 1,198,155 |

## Immutable provenance verification

All checks below were performed by reading the existing files. The runtime
roots were not written.

| Item | Verified value |
|---|---|
| Source HEAD analyzed | `99a445f6dc893a8c2240d950de4f92877cc07f8a` |
| `full_v2_5.yaml` | `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3` |
| `FINAL_COMPLETE.json` | `e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a` |
| Final artifact index | `c8f73c7d8dfc5513f9f0f10cb9e799c654f35e4367e1e9c3288809889c8e2da9` |
| Final checksum report | `730f8a1c23a965e140bce11ace6787bd8a7120df6fd61185fd41f03bd19cf0f8` |
| Final indexed artifacts | 8/8 size and SHA-256 verified |
| attempt_002 full tree | `83fc03cb49b8fa4578ea5b1ca19590fcb24371d01635afaebf7ddbdfe1afc968`; 983 files; 5,040,277,054 bytes |
| attempt_002 terminal tree | `50fafa1a7641ddf5e11d7aeced0a94ea65e7578ee49cafc4b5e8052afc5ff23c`; 48 files; 38,798 bytes |
| Frozen data manifest | `b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05` |
| Shared SamplingPlan file | `b8876c54983f8af7ed4253f2c25fdf9f351f603bc932a0152ae0a17f896e28e9` |
| Shared SamplingPlan content | `800d40c5f4fcee4577e7d6db95f92f492b7aaa3d8303e3cbeecfd3a374fd5887` |
| Train/validation/test content | `0c03f179930cd4d89ccc8a8b66283e620313e16d15fd53f0674c133bcb80c09d` / `aa1576b4ccae6a2ca30d0472f0782e1231ecc61ab3d224590a772dc3448b9e66` / `26be5e63c047297eb255eccc3276a8a7f17a1fd83a6502d34677bd21cdc75618` |
| attempt_003 continuation | all 17 expected jobs have exactly one terminal artifact |
| v2.4 artifact index / gate report | `c99c79b501ba264b01f1ef54f1bb7fa1c40446c865d95b6078fad6fd47fec08a` / `53fc93c3650e5e6d03f9a085c74a5ad86ef2862f7a918668bce817500319ee14` |
| Capacity index/report/correction | `d0c65a7e1436faae1b6701df1d88f3bb3e4f0321ff005d5ce0db9d237639fcb8` / `c123c33f0f80bcc9fcca2bfee353acc67c2ab2ac06b8eaf5338e24813b812bfc` / `692ae5ebb248e9226c04a5dc0ab1147044f7ed6243a25d6ac01e5f3ea8d434b5` |

The final checksum report itself references the same attempt_002 hashes and
records 48 reused attempt_002 terminals and 17 executed attempt_003
terminals.

## Method

The forensic extractor loads only the frozen train/test/SamplingPlan arrays
and preserved final samples. It implements the row statistics independently
of `eval/model_guards_v2_5.py`, then compares them with each stored
`evaluation.json`:

- amount and gap KS: valid rows only, real test versus synthetic;
- amount and gap label effects: absolute standardized synthetic
  `y=1 - y=0` effect;
- receiver guard: maximum absolute, entity-balanced signed category
  frequency;
- row labels: both `repeat(y_entity, lengths)` and canonical mask traversal
  are calculated and required to be identical;
- gap values: the frozen `tau[dt_bin]` mapping;
- row weighting: amount/gap are row-weighted; receiver frequencies are first
  normalized within entity and then class-averaged.

For every sample, it separately checks exact SamplingPlan labels, lengths and
mask, canonical prefix mask, zero padding, finite amount, train gap/category
support, and row-label ordering. A regression fixture also compares the
independent implementation directly with the production evaluator statistic
function.

This analysis does not call `evaluate_full_seed`, `fit`, `sample`, a DGP,
CUDA, or the full runner.

## All generator/seed terminal states

`C`, `I`, and `X` mean COMPLETE, INVALID, and CANCELLED; the suffix is the
attempt number.

| Generator | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 |
|---|---:|---:|---:|---:|---:|
| empirical_iid | C2 | C2 | C2 | C2 | C2 |
| block_2 | C2 | C2 | C2 | C2 | C2 |
| block_4 | C2 | C2 | C2 | C2 | C2 |
| block_8 | C2 | C2 | C2 | C2 | C2 |
| full_sequence_reference | C2 | C2 | C2 | C2 | C2 |
| independent_markov | C2 | C2 | C2 | C2 | C2 |
| joint_markov | C2 | C2 | C2 | C2 | C2 |
| plug_in_hmm | C2 | C2 | C2 | C2 | C2 |
| plug_in_hsmm | I2 | X2 | X2 | X2 | X2 |
| ctgan_separate_class | I2 | I2 | I2 | I3 | I3 |
| tvae_separate_class | I3 | I3 | I3 | I3 | I3 |
| neural_sequence | I3 | I3 | I3 | X3 | X3 |
| cof_seqgen | I3 | I3 | I3 | I3 | I3 |

Counts are COMPLETE 40, INVALID 19, CANCELLED 6, FAILED 0, and UNAVAILABLE
0. All 65 planned jobs are terminal. The six cancelled jobs have terminal
markers but, correctly, no completed sample/evaluation/metrics/runtime set.
The remaining 59 have those artifacts and are included in the replay.

## Hard guards

| Hard guard | PASS among 59 evaluated | FAIL |
|---|---:|---:|
| canonical mask and zero padding | 59 | 0 |
| train discrete support | 59 | 0 |
| C0/C1 reference contract | 59 | 0 |
| row marginal guards | 40 | 19 |

Thus C0/C1, padding/mask, and train discrete support do not explain any
INVALID result. Only the preregistered direct row-marginal hard guard does.

The frozen thresholds are:

| Component | Threshold |
|---|---:|
| amount KS | 0.006081138155655141 |
| gap KS | 0.006387882975686154 |
| amount absolute standardized label effect | 0.0363693454591819 |
| gap absolute standardized label effect | 0.051540527275560376 |
| receiver maximum absolute signed frequency | 0.02 |

The real test reference has 7,989 entities and 191,832 valid rows. Its amount
mean is 4.9996273, gap mean is 1.5056787, amount absolute label effect is
0.0183051, gap absolute label effect is 0.00568370, and maximum receiver
signed-frequency magnitude is 0.00543175.

## Valid versus invalid distributions

The comparison below uses the exact hard-guard statistic, not a post-hoc
metric.

| Component | COMPLETE mean [min, max], n=40 | INVALID mean [min, max], n=19 |
|---|---:|---:|
| amount KS | 0.002787 [0.001846, 0.005502] | 0.229372 [0.003081, 0.456479] |
| gap KS | 0.002674 [0.001020, 0.004718] | 0.093783 [0.007689, 0.420695] |
| amount abs. label effect | 0.013019 [0.000104, 0.027510] | 0.399090 [0.010622, 2.704458] |
| gap abs. label effect | 0.012517 [0.000504, 0.030799] | 0.211357 [0.003141, 1.198989] |
| receiver max abs. signed frequency | 0.005710 [0.003656, 0.007991] | 0.091486 [0.005635, 0.429739] |

The one non-learned INVALID is `plug_in_hsmm/seed_1`: only gap KS fails,
`0.008506986828145102 > 0.006387882975686154`. HSMM seeds 2–5 were
explicitly cancelled and are not silently omitted.

## Learned-generator failure fingerprints

Each entry is `failed/evaluated [minimum statistic, maximum statistic]`.

| Generator | amount KS | gap KS | amount effect | gap effect | receiver |
|---|---:|---:|---:|---:|---:|
| CTGAN | 5/5 [0.03726, 0.18204] | 5/5 [0.03302, 0.06646] | 5/5 [0.10789, 0.39357] | 2/5 [0.00314, 0.13510] | 0/5 [0.01348, 0.01796] |
| TVAE | 5/5 [0.05110, 0.12953] | 5/5 [0.20937, 0.42069] | 4/5 [0.02705, 0.33189] | 4/5 [0.04574, 1.19899] | 5/5 [0.21942, 0.42974] |
| neural sequence | 3/3 [0.42171, 0.43391] | 3/3 [0.00938, 0.01068] | 3/3 [0.09073, 0.29854] | 0/3 [0.00413, 0.01848] | 0/3 [0.00590, 0.00789] |
| CoF-SeqGen | 5/5 [0.38685, 0.45648] | 5/5 [0.00769, 0.03888] | 4/5 [0.01733, 2.70446] | 4/5 [0.03450, 0.27035] | 2/5 [0.01341, 0.03165] |

The only failures common to every evaluated learned seed are **amount KS**
and **gap KS**. Other components distinguish the models: for example, CTGAN
passes receiver in all five seeds while TVAE fails receiver in all five;
neural sequence passes gap label effect and receiver in all three evaluated
seeds; CoF has seed-varying label-effect and receiver failures.

The CSV preserves every seed/component value and exact reason as either
`statistic <= threshold` or `statistic > threshold`. For KS rows, the
`effect_size` field is the full-distribution KS statistic; the separate
`real_value` and `synthetic_value` fields are descriptive row means and are
not substituted for KS. For label and receiver rows those fields contain the
corresponding real and synthetic effects.

## Hypothesis A: evaluator/guard defect — REFUTED

Evidence:

1. All 295 evaluated component values were recomputed from the frozen test
   and preserved sample arrays without invoking the production evaluator.
   Every value is exactly equal to its stored evaluator value at serialized
   precision; maximum absolute difference is `0.0`.
2. All 59 samples match the shared SamplingPlan labels, lengths, and masks.
   Prefix masks, zero padding, row order, finite amount, and train support all
   pass.
3. `repeat(y_entity, lengths)` is exactly equal to label broadcast through
   the valid mask in all samples, ruling out the proposed row-label/mask
   alignment mismatch.
4. The same formulas and reference roles are used for every generator. There
   is no generator-specific orientation or row-weighting branch in the guard.
5. The stored C0/C1 reference and train-support hard guards pass in all 59
   evaluated cells.

Within the stated failure modes—target/candidate reversal, padding inclusion,
row-weight mismatch, orientation, or label mapping—there is no reproduced
evaluator defect. Consequently no intentionally failing evaluator fixture or
evaluator fix is introduced in this work. The new fixture is a positive
independent-replay and contract regression test.

## Hypothesis B: common learned sampling/adaptor defect — REFUTED

Evidence:

1. Every evaluated learned sample has the exact preregistered entity labels,
   sequence lengths, and mask from the shared SamplingPlan.
2. Static AST inspection locates `sample = adapter.sample(...)` at runner
   line 2585 and `evaluate_full_seed(sample, ...)` at line 2600, with no
   assignment to `sample` between them. There is no common runner decoding or
   post-sampling transform.
3. There are three independent learned sampling implementations:
   CTGAN/TVAE share `ConditionalCTGAN.sample`, neural sequence uses
   `NeuralSequenceBaseline.sample`, and CoF uses `CoFSeqGenAdapter.sample`.
4. Failure fingerprints differ by generator despite the common SamplingPlan.
   In particular, receiver and label-effect behavior is not universally
   distorted.

This refutes a single common adaptor or runner transformation defect capable
of explaining all four learned generators. CTGAN and TVAE intentionally share
a tabular sampling implementation, so a defect specific to that pair cannot
be disproved solely by the three-way implementation split. Their very
different gap and receiver fingerprints provide no positive evidence for
such a pair-specific defect, but that narrower claim remains outside the
universal hypothesis tested here.

## Hypothesis C: frozen-model output fidelity — SUPPORTED

Evidence:

1. All 18 evaluated learned seeds are INVALID on row marginals, and all 18
   fail both amount KS and gap KS.
2. The failures reproduce directly from immutable `sample.npz` files; they
   are not generated by terminal-marker classification.
3. Every evaluated learned run records all requested updates completed and
   `wall_cap_reached=false`: CTGAN 10,000 total class-model updates, TVAE
   20,000, neural sequence 20,000, and CoF 20,000.
4. The output deviations are model-specific and, in many cells, far above
   threshold. For example, learned amount KS reaches 0.45648 and TVAE gap KS
   reaches 0.42069, versus thresholds 0.006081 and 0.006388.

The supported conclusion is deliberately narrow: these frozen training and
sampling configurations did not reproduce the required row distributions.
The artifacts do not distinguish insufficient updates, training objective,
optimization, or architecture. Completion before the two-hour cap is not
proof that extra training would fix the distributions.

## Artifact order and provenance

The runner's logical order is:

1. immutable manifest and RUNNING allocation;
2. training, checkpoint, and progress;
3. `adapter.sample` returns a sample in memory;
4. one `evaluate_full_seed` call computes the contract, association metric,
   support diagnostics, and row guards from that same in-memory sample;
5. sample, final checkpoint, metrics, runtime, and evaluation are atomically
   written and indexed;
6. COMPLETE or INVALID is written last.

As a concrete example, CoF seed 1 attempt_003 has:

| Artifact | UTC mtime | SHA-256 |
|---|---|---|
| manifest | 2026-07-29 15:37:23.142 | `d0d3310809d36679639bca5ab577bee5b697b10d68a86d5db117df25445b09b6` |
| sample | 2026-07-29 15:41:13.299 | `cee28a68862c1cfa25c0fcfa2bbad567fbc38d0fca6617ffaca2f6a806afebac` |
| metrics | 2026-07-29 15:41:13.327 | `5d81f2cb7f7ac2216a0841e9baf7e85608a9824062880f45428b37d3cdc7e66b` |
| runtime | 2026-07-29 15:41:13.327 | `35bfce5ef9e4c0e447ff787d37273aa542637cc7d373a931c636c1d4e1449bf2` |
| evaluation | 2026-07-29 15:41:13.331 | `bab9066eb6e785ba78de0746aee009dd93622f7a86102a631cb9b93275f23062` |
| INVALID | 2026-07-29 15:41:13.355 | `33a481dbe35bce25ec4db8d2ba2d4f69e104f46d05e7009c7551c30bbb2e1d62` |

All seed-level hashes and timestamps are in the JSON; the six core
sample/evaluation/metrics/runtime/terminal hashes are repeated in the CSV.

The 48 attempt_002 manifests use source
`5618a941f8ecbbe60eaed5bcfe82d177eb36a0f7` and relevant code hash
`e64f69ffe1a5606f4eabe6cdc590ae0abae5f0c676b5f5035949cb8c40a8deeb`.
The 17 continuation manifests use source
`99a445f6dc893a8c2240d950de4f92877cc07f8a` and relevant code hash
`20085b42cc2ab8dc9ab4fe1d1d30006ae5733b7243b4e78ea3a8c47564bf82cb`.
Both groups use the unchanged config hash, SamplingPlan content hash,
evaluation version `benchmark-v2.5-evaluation-v1`, and baseline definition
version `benchmark-v2.5`.

## v2.6-only proposal

Any remediation must be a new v2.6 preregistration. It must not reinterpret or
replace this v2.5 result.

1. Use train-only fitting and a separately fixed validation split for all
   model/adaptor selection. The v2.5 test split must not calibrate thresholds,
   transformations, early stopping, or architecture.
2. Before training, preregister a finite candidate list and fixed selection
   rule for update count, objective weights, numeric normalization/inverse
   transform, categorical sampling, and class-specific handling. Preserve the
   same selection budget across compared learned models.
3. Add train/validation-only diagnostics at fixed checkpoints for the five
   existing row guards. These may select among the preregistered candidates;
   they must not change the hard thresholds after seeing held-out test data.
4. Add adapter contract fixtures that round-trip label, length, padding,
   numeric decoding, gap-bin decoding, and receiver categories separately for
   CTGAN/TVAE, neural sequence, and CoF.
5. Freeze the selected v2.6 source/config/data hashes before any v2.6 test
   evaluation. Re-run every affected generator/seed under a new artifact root;
   do not patch or reuse v2.5 terminal outcomes as v2.6 outcomes.
6. Keep INVALID seeds as INVALID in aggregation. Do not exclude them as NaN,
   and do not change C2 based on partial or diagnostic results.

No part of this proposal was implemented or executed in the present work.

## Verification

The forensic implementation adds only a read-only extractor, its tests, and
the three documentation artifacts.

| Check | Result |
|---|---|
| Focused forensic/evaluator/guard pytest | PASS — 11 passed in 0.94 s |
| Repository-wide pytest | PASS — 203 passed, 19 warnings in 105.28 s |
| `python -m compileall -q .` | PASS |
| `git diff --check` | PASS |

The warnings are the existing PyTorch nested-tensor warning and the upstream
CTGAN `cuda`-parameter deprecation warning. No test failure was suppressed.
The benchmark runtime/data/config/authorization roots and the three preexisting
user image deletions are excluded from the forensic commit.
