# CoF-SeqGen v3 attempt_002 failure forensic

This is a read-only analysis of stored samples and frozen validation data. It did not query a GPU, restore a model, train, sample, or rerun validation.

## Finding

The production evaluator is independently reproduced, and both samples obey the active train-derived Y/L, mask, padding, codec, and support contracts. The failure is a probability-mass failure within valid joint support, consistent with the current v3 objective/discrete architecture.

## Five-guard reproduction

| candidate | metric | independent | threshold | result | stored match |
|---|---|---:|---:|---|---|
| cof_v3_c01_direct_joint | amount_ks | 0.00237128252888 | 0.00608113815566 | PASS | True |
| cof_v3_c01_direct_joint | gap_ks | 0.0880960937065 | 0.00638788297569 | FAIL | True |
| cof_v3_c01_direct_joint | amount_abs_standardized_label_effect | 0.0259773615566 | 0.0363693454592 | PASS | True |
| cof_v3_c01_direct_joint | gap_abs_standardized_label_effect | 0.0923962404215 | 0.0515405272756 | FAIL | True |
| cof_v3_c01_direct_joint | receiver_max_abs_signed_frequency | 0.0551861707638 | 0.02 | FAIL | True |
| cof_v3_c02_factorized_joint | amount_ks | 0.00475468137189 | 0.00608113815566 | PASS | True |
| cof_v3_c02_factorized_joint | gap_ks | 0.0820637187235 | 0.00638788297569 | FAIL | True |
| cof_v3_c02_factorized_joint | amount_abs_standardized_label_effect | 0.0232211424959 | 0.0363693454592 | PASS | True |
| cof_v3_c02_factorized_joint | gap_abs_standardized_label_effect | 0.473613439488 | 0.0515405272756 | FAIL | True |
| cof_v3_c02_factorized_joint | receiver_max_abs_signed_frequency | 0.302604944129 | 0.02 | FAIL | True |

All ten independent values agree with stored `evaluation.json` within 1e-12 and reproduce the same PASS/FAIL decisions.

## Class-conditional discrete PMF differences

Total variation (TV) compares frozen validation rows with stored synthetic rows within each entity label.

| candidate | Y | gap-bin TV | receiver TV | joint TV |
|---|---:|---:|---:|---:|
| cof_v3_c01_direct_joint | 0 | 0.088397182 | 0.250557153 | 0.358744912 |
| cof_v3_c01_direct_joint | 1 | 0.164030089 | 0.393314947 | 0.568411315 |
| cof_v3_c02_factorized_joint | 0 | 0.101668398 | 0.429934411 | 0.438326691 |
| cof_v3_c02_factorized_joint | 1 | 0.359836819 | 0.528084429 | 0.619729887 |

The factorized candidate is especially class-distorting: its Y=1 gap/receiver/joint TVs are substantially larger than direct-joint. Full PMF vectors and state-level differences are in the JSON/CSV.

## Support, Y/L, mask, and padding

| candidate | active plan exact | padding zero | codec errors | outside support | observed states (real/synth) |
|---|---|---|---:|---:|---|
| cof_v3_c01_direct_joint | True | True | 0 | 0 | 1024/1024 |
| cof_v3_c02_factorized_joint | True | True | 0 | 0 | 1024/1024 |

The active train-policy plan was independently reconstructed from the pre-sampling conditioning binding as `862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27` and exactly matches both samples. `train.npz` and `shared_sampling_plan.npz` were only preservation-hashed, not loaded by the guard/PMF extractor. The guard calculations therefore use only the stored samples, frozen validation, and frozen tau metadata.

The train support mask, real validation, and both synthetic samples all contain 1024/1024 joint states. Therefore support coverage is complete but distributional mass is wrong.

## v2.8 CoF comparison

| candidate | amount KS delta | amount effect delta | gap KS delta | gap effect delta | receiver delta |
|---|---:|---:|---:|---:|---:|
| cof_v3_c01_direct_joint | -0.000959629 | 0.020680027 | 0.086195270 | 0.085629152 | 0.034059615 |
| cof_v3_c02_factorized_joint | 0.001423770 | 0.017923807 | 0.080162895 | 0.466846351 | 0.281478389 |

Both v3 candidates retain amount-guard PASS. Direct improves amount KS relative to v2.8 while its amount effect remains PASS; factorized is slightly worse on both amount measures but remains PASS. In contrast, v2.8 gap guards passed and receiver was 0.0211266, whereas both v3 candidates regress every discrete guard.

## Hypothesis decisions

### A_evaluator_mask_label_mapping: REFUTED

- independent guard implementation matches all ten stored values/checks: True
- both samples exactly match the active train-policy Y/L/mask and zero padding: True
- candidate sample hashes are bound by the stored evaluations

The active plan was reconstructed from the immutable conditioning binding written before candidate execution. Its Y/L/mask and plan hash agree with both samples. The separate preserved SamplingPlan file was hash-inventoried but not loaded for this calculation.

### B_joint_codec_support_decoder_sampling_bug: REFUTED

- codec round trips, bounds, and support compliance all pass: True
- train, validation, and each synthetic sample contain all 1024 joint states
- hash-bound source uses one paired direct state or sampled-gap conditional receiver path

The train support mask is the full 16x64 Cartesian set, so it prevents illegal codes but cannot enforce probability mass. Stored artifacts do not include per-step logits, but no observable codec, mask, support, decode, or branch-contract violation was found.

### C_v3_objective_architecture_limit: SUPPORTED

- direct max class TV: {'gap_bin': 0.16403008893526597, 'receiver': 0.3933149466835676, 'joint': 0.5684113149329025}
- factorized max class TV: {'gap_bin': 0.35983681889105135, 'receiver': 0.5280844291994318, 'joint': 0.6197298871656773}
- all four v3 amount guards pass while all six v3 discrete guards fail
- v2.8 CoF had passing gap guards and only a near-threshold receiver failure; both v3 candidates regress gap and receiver
- v2.8 comparison independently reproduced: True

Mechanisms consistent with the evidence:

- direct joint CE must allocate probability over 1024 dense states under a full support mask
- factorized training teacher-forces true gap while sampling conditions receiver on sampled gap, exposing inference-time error propagation
- sampling starts from a fully masked joint state although training corruption is capped at 0.7
- the fixed 50-step sampler repeatedly resamples discrete states; the objective has no hard marginal-matching term

## Static code audit

| role | path | SHA-256 | audited loci |
|---|---|---|---|
| model_v3 | `models/cof_seqgen_v3.py` | `fbe75b00f94fa90b89f5ca037507f01c4dd86d31ccd7cbe30c17a888d800b25e` | 37-67 joint codec encode/decode ; 70-97 paired corruption mask ; 100-148 direct joint head/sample ; 151-263 factorized gap-conditional receiver path ; 487-497 support masking ; 499-595 paired objective and teacher-forced gap |
| candidate_sampler | `generators/cof_seqgen_v3_candidate.py` | `741b4e23b400b0ad55ae89f5c328468657d573af0c49fe8eaac1982f18ec673f` | 212-240 fixed 50-step/guidance/temperature contract ; 243-250 active Y/L mask and initialization ; 285-344 direct/factorized discrete sampling branches ; 345-376 padding and SamplingPlan output binding ; 504-557 train-only amount state and validation sample path |
| production_evaluator | `eval/evaluation_only_v2_7.py` | `d7cc37e47735cf47ab65ae33fa26e9c2d8096c45540eb92a10d5b806301e7934` | 20-75 frozen threshold/SamplingPlan validation and five guards |
| production_guard | `eval/model_guards_v2_5.py` | `23c530cd53efe142842491ec08ac5e627cc7f1494acdd96765bcf753e5153dab` | 25-43 standardized class effect ; 46-78 entity-weighted receiver frequency ; 81-122 row guard statistics |
| execution_runner | `experiments/cof_seqgen_v3_execution_runner.py` | `b297de8a0a2582de5cf3e80d3e2df1d45724353d535abd3c059970589b842370` | 1171-1194 train-policy plan binding in spawned child ; 1241-1307 train/support/conditioning provenance |

No evaluator fix is proposed because no evaluator defect was found. A future preregistration should change the train-only objective or discrete generative formulation, not thresholds or test-based calibration. The factorized candidate specifically needs removal of teacher-forcing/sampled-gap exposure mismatch; the direct candidate needs a tractable probability-mass objective for the dense joint space. Those are proposals only; no implementation or rerun occurred here.

## Preservation and zero-execution statement

- `v2_5_full_config`: `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3`
- `v2_5_final_complete`: `e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a`
- `v2_6_development_manifest`: `31ddf45dcf7b4f982ed990ec94281c3e6bad0454ec3182e2434e42358006ade5`
- `v2_8_source_config`: `9e8ecaeab15f28ec05cee27150f4c6f76142dce661536094f41a798a83a7ae98`
- `v2_8_evaluation_config`: `2af755a73e2764b060ef880b66e5deea7668919382ae151a11b42f15661feea0`
- `v2_8_aggregate_config`: `b3db5f71080df1bfdc051259eb0267cf17629b90c15512916094b4b47890092d`
- `v2_8_execution_authorization`: `028a6fdc74814ea60e84fa1a027d15abc546c3d5298caa5765db8641d3af6bba`
- `v2_8_aggregate_authorization`: `d83892f88341c22b37c794f422db1f528b9187d1f4a1b3ff8a513a2b63faed8e`
- `v2_8_aggregate_terminal`: `58573dfb93ebed71843f286405e205669140b8cc66811ce61deebe808de513e9`
- `runner_config`: `623ab43352889d8e64c05fd17b883a255f9a92811d713bbedbf7f05ed19082b5`
- `source_config`: `1f9ea3a596509b729e02157a26e60859cccf780621c1439659370c540fec6bd2`
- `attempt_001_authorization`: `d11941f514a65c74e07fe522f738df19447553116e024def5e0224fc9d2176d8`
- `attempt_002_authorization`: `5386a55b0af81dea053433bdbab690ef2114ce04d2890411156a69dd51ce11ca`
- `frozen_manifest`: `b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05`
- `frozen_meta`: `74566c8f06bed7b37060b12fd86e9a38b168942101433e7df8ddf18dc93aa2b4`
- `frozen_train`: `c67a6fce4593317e11e1b6f36397bfb38f9607916bf8f5b32329d8fa99b700a8`
- `frozen_validation`: `68c15c05c55742bfce0f22560ffe8378d032a7a9fd9a1072e80ec8da5340fba5`
- `sampling_plan_file`: `b8876c54983f8af7ed4253f2c25fdf9f351f603bc932a0152ae0a17f896e28e9`
- `direct_attempt_001`: `9d75a436d61caf8ace7c38c212dffe6d4d0f65dc9ffc3ccdd371b4a0f7d7a55c`
- `direct_attempt_002`: `a5ede6b1c13b0a59d328e094444bda9cad54489b8fc3c49282004bd033dc830d`
- `factorized_attempt_001`: `454e68f4bbd77d34ab1bf395e433bb1f195b40d5b425d22f64e49762f2bbe0ca`
- `factorized_attempt_002`: `28f159a587fbf5d9c656bc09417bd84c283b7a5e4537280c0314b23c4d1e0da1`
- `v2_8_candidate_selection`: `7f60deb8a10cd9946e2e8a814b21c463fbf753954e7e03c44b6a7d9efb02e700`

Execution counters are all zero: gpu_queries=0, cuda_calls=0, model_fit_calls=0, model_sample_calls=0, validation_reruns=0, test_split_reads=0, fresh_test_calls=0, tstr_calls=0, privacy_calls=0, five_seed_full_run_calls=0.
