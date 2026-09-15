# v2.7 no-passing-candidate forensic analysis

## Scope and frozen conclusion

This read-only analysis explains the completed v2.7 validation aggregate. It
does not revise selection:

- CTGAN: `NO_PASSING_CANDIDATE`;
- TVAE: `tvae_v27_c01_amount_inverse_decoder` selected;
- CoF-SeqGen: `NO_PASSING_CANDIDATE`;
- `primary_c2_selection_ready=false`.

Fresh test, TSTR, privacy analysis, five-seed/full execution, threshold
changes, test-based tuning, candidate execution, and result-contingent
candidate extension remain forbidden.

The forensic extractor reads statistical evidence only from the frozen
`selection_report.json` and the six stored non-control `evaluation.json`
files. It does not load a validation sample or test split and does not call
the evaluator or aggregate runner. Hashing immutable runtime files for
provenance is not statistical reuse.

## Immutable inventory

The analysis base is
`fbaaac55cd2506e806e22bb7c26cdbaba1ed2997`.

| Input | SHA-256 |
|---|---|
| v2.7 `AGGREGATE_COMPLETE.json` | `a40630f51e1a07fb509f8f676eabf52f9bd18affbd7a9de47c510e2e1c0792b2` |
| v2.7 aggregate index | `2539c02a5cec95ec85b21e2d1dec2798d905e63aac21723dfcdc103195a07b89` |
| v2.7 aggregate checksum manifest | `da1b2244d718ac9000758c4c6a5411b83e0cd7badd6cd1372ca58f064b1b2496` |
| v2.7 selection report | `3af94b53f5dbaaeca917a04d2b490c0267f70aa915e38e2f751b449451122ec3` |
| v2.7 selection manifest | `53807d3b399d74aea77c2a1b9cced1534bcf6735b7b27cf8fa8b35b74063a07b` |
| aggregate authorization | `5961ce4b042553c7ff058984e37d38b69176118a1e0b3b92898f8f61bccb5d64` |
| candidate execution authorization | `ff8bfdab5177d4b067e74f2cf192590b152236406b2eb4c1664878c20b5beea5` |
| evaluation runner config | `05435443f69f25f56e6d738dcae7c5a62f0c99096483b5ac26c6f525cffae68c` |
| candidate config | `2737723a05ce8628b7c9b8e1afbe7edae323a06971fbaf1eb00465224e0d654b` |
| v2.5 config | `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3` |
| v2.5 `FINAL_COMPLETE.json` | `e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a` |
| v2.5 frozen manifest | `b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05` |
| v2.6 forensic terminal | `0c2e89a4124d35ccfa565ab504fbedf8819bed2dfe649370f811efe51a408703` |
| frozen train file | `c67a6fce4593317e11e1b6f36397bfb38f9607916bf8f5b32329d8fa99b700a8` |
| frozen validation file | `68c15c05c55742bfce0f22560ffe8378d032a7a9fd9a1072e80ec8da5340fba5` |
| train-only SamplingPlan | `862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27` |

The preserved candidate input tree is
`de3fda560550cb8c815774a28be9bcc2350db32075a6776750fc66061b3a520e`
(51 files, 6,993,542 bytes). The aggregate bundle tree is
`18d8e4d0fc5dfd148dd6e67036e063b6ad6db1a1f640c322b50bc825aa824378`
(6 files, 34,223 bytes).

## Stored-evidence consistency

The six non-control `evaluation.json` files contribute 30 stored guard
values. All 30 exactly match the corresponding values in the selection
report. Across all nine candidates, all 45 stored value/threshold decisions
agree with their recorded PASS/FAIL value. Candidate IDs, model IDs,
validation split, train fit split, SamplingPlan hash, and
`test_split_read=false` also agree.

This is a stored-evidence consistency check, not a guard recalculation.
Frozen controls have no new v2.7 evaluation file and are used only through
the hash-bound frozen-control rows in the selection report.

Thresholds are unchanged:

| Guard | Threshold |
|---|---:|
| Amount KS | 0.006081138155655141 |
| Gap KS | 0.006387882975686154 |
| Amount effect | 0.0363693454591819 |
| Gap effect | 0.051540527275560376 |
| Receiver effect | 0.02 |

## CTGAN

`P` and `F` denote stored PASS and FAIL.

| Candidate | Amount KS | Gap KS | Amount effect | Gap effect | Receiver effect |
|---|---:|---:|---:|---:|---:|
| frozen standard | 0.159947378 F | 0.042130957 F | 0.157397385 F | 0.041325401 P | 0.029209443 F |
| amount quantile inverse | 0.003227693 P | 0.041141966 F | 0.008727642 P | 0.032647090 P | 0.026672031 F |
| categorical logit | 0.159254351 F | 0.045025057 F | 0.141863913 F | 0.046268332 P | 0.028411380 F |

### Minimum blockers

`gap_ks` and `receiver_max_abs_signed_frequency` fail in every CTGAN
candidate. Either is by itself a rejection witness for every candidate; an
ALL-PASS repair must resolve both.

The amount-only candidate proves that the train-fitted amount inverse can
move both amount guards inside their limits. Its remaining gap KS is 6.44
times threshold and receiver effect is 1.33 times threshold. The isolated
categorical-logit candidate does not resolve either universal blocker and
leaves the native amount failures intact.

## CoF-SeqGen

| Candidate | Amount KS | Gap KS | Amount effect | Gap effect | Receiver effect |
|---|---:|---:|---:|---:|---:|
| frozen variance residual | 0.028220658 F | 0.022440171 F | 0.038313631 F | 0.258948906 F | 0.018105773 P |
| empirical residual | 0.003925904 P | 0.022440171 F | 0.004578512 P | 0.258948906 F | 0.018105773 P |
| gap logit bias | 0.029113417 F | 0.009416294 F | 0.018892008 P | 0.328951778 F | 0.019471177 P |

### Minimum blockers

`gap_abs_standardized_label_effect` and `gap_ks` fail in every CoF
candidate. Either is by itself a rejection witness for every candidate; both
must be resolved for ALL PASS.

The empirical amount-residual candidate moves both amount guards to PASS and
leaves receiver PASS, but the unchanged gap effect remains 5.02 times its
threshold and gap KS remains 3.51 times threshold. The gap-logit candidate
reduces gap KS from 0.022440171 to 0.009416294, but still fails it and worsens
gap class effect from 0.258948906 to 0.328951778. It also leaves amount KS
failed because amount sampling was not its factor.

## Difference from selected TVAE

The selected TVAE candidate records:

| Candidate | Amount KS | Gap KS | Amount effect | Gap effect | Receiver effect |
|---|---:|---:|---:|---:|---:|
| `tvae_v27_c01_amount_inverse_decoder` | 0.004899962 P | 0.005769090 P | 0.029937812 P | 0.001375065 P | 0.006794393 P |

TVAE combines the same kind of successful amount-only inverse correction
with a frozen categorical path whose gap and receiver guards were already
inside threshold. In contrast:

- CTGAN's amount correction also passes both amount guards, but its frozen
  discrete path still fails gap KS and receiver frequency.
- CoF's amount correction also passes both amount guards and receiver, but
  its frozen gap path still fails both gap guards.
- CTGAN's categorical-logit candidate and CoF's gap-logit candidate do not
  recreate TVAE's across-channel guard profile.

This is an internal stored-result comparison, not a claim about test
performance or the C2 endpoint.

## Hypothesis verdicts

### Aggregate implementation defect — REFUTED

All 30 non-control values match exactly between their stored evaluation and
the aggregate report. All 45 report decisions agree with their stored values
and frozen thresholds. The same aggregation path selects the TVAE all-pass
candidate and rejects only candidates with recorded guard failures.

### Evaluator implementation defect — INCONCLUSIVE

There is no positive evidence of an evaluator defect: every stored sample
contract is PASS, all candidates use the same thresholds and SamplingPlan,
and evaluation-to-report transcription is exact. However, this task
explicitly prohibits reading stored samples and recomputing guards.
Therefore the evaluator formulas are not independently revalidated here and
cannot be fully refuted from these two JSON layers alone.

### Numeric/decode/sampling path limitation — SUPPORTED

The stored intervention pattern is channel-specific:

- CTGAN amount inversion repairs amount but leaves gap/receiver failed.
- CTGAN categorical-logit calibration does not repair its two universal
  discrete blockers.
- CoF empirical residual sampling repairs amount but leaves gap unchanged.
- CoF gap-logit bias improves gap KS but worsens gap class effect.

The evidence supports a limitation at the candidate numeric or discrete
sampling boundaries. It does not identify the exact lower-level causal
subroutine without a separately preregistered intervention.

### Current finite candidate-space limitation — SUPPORTED

Both blocking models have two guard components that fail every member of
their finite candidate family. No existing candidate can satisfy the
all-five-guard intersection. This finding does not authorize adding a
candidate, changing a threshold, or examining test data.

## One next single-factor proposal per model

These are proposals only. They are not implemented, authorized, or launched.
Any later work requires a new train/validation-only preregistration.

### CTGAN: one joint discrete-decoder factor

Evaluate one train-only, class-conditional joint `(gap_bin, receiver)`
categorical decoder calibration as a single discrete-decoder factor. Keep
the amount quantile inverse, frozen checkpoint, separate-class generators,
thresholds, SamplingPlan, endpoint, and DGP unchanged. This one boundary is
chosen because gap KS and receiver frequency are the two universal CTGAN
blockers; it must not be bundled with another numeric or training change.

### CoF-SeqGen: one gap-sampler factor

Evaluate one train-only class-conditional gap-distribution sampler
calibration, replacing only the current gap-logit sampling map. Keep the
empirical amount residual sampler, receiver path, checkpoint,
architecture/objective, thresholds, SamplingPlan, endpoint, and DGP
unchanged. This single gap boundary is chosen because both universal CoF
blockers are gap metrics.

## Versioned evidence

- `forensic_no_passing_candidate_v2_7.json`:
  `e6270814967caeb4e71f9a6f4bb394aa4b31e0096331c4cb180e4cedaf7651d1`
- `forensic_no_passing_candidate_v2_7.csv`:
  `f434ad6c2541b57985475b24365ea7e72a8e5880ff8076f55465106c4b6e1773`

The evidence records zero GPU/CUDA queries, training, restore, sampling,
candidate execution, guard recalculation, validation-sample reads, test
reads, fresh test, TSTR, privacy, and five-seed/full execution.

## Verification

- focused forensic/aggregate/evaluation pytest: 40 passed;
- repository-wide pytest: 332 passed, with 23 existing third-party warnings;
- `compileall`: PASS;
- `git diff --check`: PASS.
