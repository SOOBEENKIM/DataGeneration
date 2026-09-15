# v2.6 validation-selection failure forensic diagnosis

## Scope and frozen conclusion

This is a read-only, validation-only forensic replay of the completed v2.6
selection. It does not change the five thresholds, endpoint, C2 rule,
SamplingPlan, DGP, candidate definitions, training budget, or any v2.5/v2.6
runtime artifact. It did not open the frozen test split, create a fresh test,
query CUDA, train or sample a learned model, or start a five-seed/full run.

The frozen selection conclusion remains:

- CTGAN: `NO_PASSING_CANDIDATE`;
- TVAE: `NO_PASSING_CANDIDATE`;
- CoF-SeqGen: `NO_PASSING_CANDIDATE`;
- neural sequence: secondary `NOT_EVALUABLE`;
- v2.6 primary full run: prohibited.

The complete machine-readable evidence is in
`forensic_selection_failure_v2_6.json`; the 12 learned candidate rows and five
empirical-reference replays are in
`forensic_selection_failure_v2_6.csv`.

## Read-only inventory

The forensic input source state was
`f02004aac873ce9a0d7ee88006ddf6f2ec84d481`. Key immutable inputs were:

| Input | SHA-256 |
|---|---|
| selection config | `0884a0144f74f6317ae9e636c0cf5657dedac21ff12cdf545a6b6e5e29be4c4d` |
| development manifest | `31ddf45dcf7b4f982ed990ec94281c3e6bad0454ec3182e2434e42358006ade5` |
| selection authorization | `b0ceebed14d5d6ebaaf73790fe83d7f08b10985ec547b5bea0f2631daee57935` |
| selection report | `94da87ae7040771b47ad3d07fb6a95c5e7fc6945f37d104cda8be62b087f1236` |
| selection manifest | `f9bd461051e76b697eca91e058788a2110bf46763a652f936e22caeaa9159ad7` |
| selection artifact index | `69a48f51e5f0bea4a6b5a7e39c9a263c131c46365c4b0175d16e5563915d91af` |
| selection checksum manifest | `79fb0ee4a7fef2a0fe9d244036a462474eefe889f10e3eefe6bfe60d4cdfb1e3` |
| `AGGREGATE_COMPLETE.json` | `3cb9f91a00ea4b3bcb772527e2ff4f49b1a47c645618ce79995847f07337407d` |
| frozen train file | `c67a6fce4593317e11e1b6f36397bfb38f9607916bf8f5b32329d8fa99b700a8` |
| frozen validation file | `68c15c05c55742bfce0f22560ffe8378d032a7a9fd9a1072e80ec8da5340fba5` |
| train-only SamplingPlan content | `862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27` |
| v2.5 frozen-data manifest | `b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05` |
| v2.5 config | `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3` |
| v2.5 `FINAL_COMPLETE.json` | `e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a` |

The candidate runtime tree contains 122 files and 381,722,799 bytes. Its
digest is
`db9d2b21f00f584bd9a5c36038d1f18e5026c2a29e42b40d9bcc958789f58d85`,
defined as SHA-256 over sorted
`<file_sha256><two spaces><relative_path>\n` records. Nothing in that tree was
changed, moved, or regenerated.

## Independent replay contract

The new fixture does not call `row_guard_statistics` for its forensic
calculation. It independently reconstructs:

- row traversal from the canonical valid mask;
- row label order from `repeat(y_entity, lengths)`;
- amount and frozen-`tau` gap KS;
- signed pooled-standardized amount/gap class effects;
- entity-balanced receiver signed-frequency effects.

The independent result matched every stored selection statistic for all 12
candidates with maximum absolute error at most `1e-12`. Every stored sample
also independently passed SamplingPlan labels, lengths, mask, canonical
padding, finite amount, train gap/receiver support, and row-label ordering.
Thus the failure values are already present in the stored decoded validation
rows and are not introduced by selection-report serialization.

## Hypothesis A — five-guard threshold feasibility

**REFUTED.**

`EmpiricalConditionalIID` was fitted to frozen train valid rows only. It
sampled rows separately for `y=0` and `y=1` into the exact frozen
SamplingPlan, which has the same 7,989 entities and 192,115 requested valid
rows as every learned candidate. The unchanged validation split was the
reference. Five fixed diagnostic seeds all passed all five unchanged guards:

| Seed | Amount KS | Gap KS | Amount effect | Gap effect | Receiver effect |
|---:|---:|---:|---:|---:|---:|
| 2601 | 0.003188 | 0.003550 | 0.011981 | 0.007928 | 0.003559 |
| 2602 | 0.002885 | 0.002222 | 0.004429 | 0.004556 | 0.003546 |
| 2603 | 0.003638 | 0.003124 | 0.022528 | 0.014214 | 0.004287 |
| 2604 | 0.002563 | 0.003656 | 0.002533 | 0.000231 | 0.004515 |
| 2605 | 0.003527 | 0.001911 | 0.000269 | 0.009257 | 0.005050 |
| Frozen limit | 0.006081 | 0.006388 | 0.036369 | 0.051541 | 0.020000 |

This shows that the frozen acceptance contract is feasible for a
train-derived label-stratified empirical row generator at the actual
validation scale. It does not prove the thresholds are optimal, but it
refutes the stated infeasibility explanation. No threshold redesign is
justified by this result.

## Hypothesis B — numeric encode/decode and class-pair sampling

**SUPPORTED at the decoded model-output boundary; the exact causal
sub-boundary remains INCONCLUSIVE.**

The real validation amount has mean 4.9962, standard deviation 1.0020, and
1/50/99 percentiles 2.6556/5.0002/7.3162. Its signed standardized class
effects are only 0.00339 for amount and 0.00173 for gap; the maximum
entity-balanced receiver class effect is 0.00692.

`A/G/AE/GE/R` below mean amount KS, gap KS, amount class effect, gap class
effect, and receiver class effect. `P` and `F` are the frozen guard outcomes.
Gap/receiver TV are total-variation distances of raw stored marginal
frequencies from validation.

| Model/candidate | A/G/AE/GE/R | Amount mean/std | q01/q50/q99 | Gap TV | Receiver TV |
|---|---|---:|---:|---:|---:|
| CTGAN c00 10k | F/F/F/F/P | 5.098/0.957 | 2.800/4.999/7.588 | 0.127 | 0.088 |
| CTGAN c01 20k | F/F/F/P/F | 5.214/1.161 | 2.462/5.004/7.472 | 0.102 | 0.115 |
| CTGAN c02 z-score | F/F/F/P/F | 5.226/1.032 | 2.827/5.086/7.124 | 0.104 | 0.116 |
| CTGAN c03 tempered | F/F/F/F/P | 4.959/1.016 | 2.449/4.971/7.197 | 0.116 | 0.073 |
| TVAE c00 10k | F/F/F/F/F | 4.984/0.937 | 2.882/4.895/6.873 | 0.496 | 0.737 |
| TVAE c01 20k | F/F/F/F/F | 4.996/1.007 | 2.929/4.802/7.365 | 0.604 | 0.758 |
| TVAE c02 z-score | F/F/F/F/F | 5.063/0.995 | 2.917/5.042/7.410 | 0.579 | 0.797 |
| TVAE c03 weighted/tempered | F/F/F/F/P | 5.348/1.011 | 3.149/5.236/7.409 | 0.170 | 0.048 |
| CoF c00 10k | F/F/F/F/F | 6.851/0.481 | 5.773/6.842/8.015 | 0.074 | 0.116 |
| CoF c01 20k | F/F/F/F/P | 4.933/0.163 | 4.550/4.930/5.338 | 0.036 | 0.070 |
| CoF c02 z-score | F/F/F/P/P | 4.964/0.071 | 4.801/4.965/5.130 | 0.025 | 0.052 |
| CoF c03 weighted/tempered | F/F/F/F/P | 4.963/0.092 | 4.750/4.962/5.190 | 0.041 | 0.067 |

### CTGAN

The adapter flattens valid rows, splits them by label, fits two independent
class models, and writes their decoded rows into plan positions
(`conditional_ctgan.py:137–263`). The candidate backend controls Gumbel
temperature and restores each class transformer
(`candidate_model_backends_v2_6.py:57–137`). Overall amount spread is not
collapsed, but the synthetic class effects are much larger than real:
absolute amount effects are 0.076–0.505 versus real 0.00339, and gap effects
reach 0.220 versus real 0.00173. This localizes the failure to class-specific
learned row distributions before plan assembly, not to mask/length/label
assembly. The present bundled candidates cannot distinguish the GMM inverse,
generator distribution, and class-specific transformer as the sole cause.

### TVAE

TVAE uses the same class-pair flattening and assembly. Its candidate decoder
draws latent rows, applies transformed-span decoding, and invokes the
upstream inverse (`candidate_model_backends_v2_6.py:140–342`). Native c00/c01
concentrate gap and receiver mass: raw marginal TV is 0.496–0.604 for gap and
0.737–0.758 for receiver. At c01, one receiver category holds 28.2% of rows
while the largest real category holds 1.68%. The weighted/tempered c03
reduces receiver TV to 0.048 and makes the receiver hard guard pass, while
gap KS 0.104 and the continuous/class effects still fail. This is direct
evidence that TVAE categorical decode/sampling is a major failure boundary;
it does not establish one temperature or loss weight as the only cause.

### CoF-SeqGen

CoF is not a separate-class row model, so class-pair handling cannot be a
common explanation across all three models. Its candidate loss predicts
clean amount with MSE from a noisy input, while DDIM reconstructs from
Gaussian initialization (`candidate_components_v2_6.py:79–255` and
`269–end`). Its amount output is the distinctive failure: c01–c03 standard
deviations are only 0.163, 0.071, and 0.092 versus real 1.002. The z-score
inverse is exact and plan-bound (`candidate_adapters_v2_6.py:29–108`), yet
variance remains collapsed after inverse. Therefore z-scoring alone is
refuted as a remedy; the continuous loss/sampling boundary is implicated.
Discrete distortions are smaller, and receiver passes in c01–c03.

Across all models, exact plan/support checks plus different channel
fingerprints refute one common mask, padding, label-mapping, or evaluator
implementation error. Because c02/c03 bundle representation, weighting, and
sampling changes and use one selection seed, the forensic evidence cannot
assign causality to a single internal operation.

## Hypothesis C — training cap only

**REFUTED.**

Only c00 and c01 are compared here: they are checkpoints at 10,000 and
20,000 updates from the same native trajectory, so this is the clean
training-duration comparison.

| Model | Guards improved at 20k | Guards worsened at 20k | Count improved |
|---|---|---|---:|
| CTGAN | gap KS, gap effect | amount KS, amount effect, receiver | 2/5 |
| TVAE | gap KS, amount effect, gap effect | amount KS, receiver | 3/5 |
| CoF | amount KS, gap KS, amount effect, receiver | gap effect | 4/5 |

No primary model improved all five guards, and none passed at 20,000.
Increasing updates alone is therefore not supported as the next intervention.

## Minimum next candidate-space changes

These are proposals for a future result-blind amendment, not implementations
or authorizations.

1. **CTGAN:** isolate one factor per candidate. Keep separate class generators
   but compare the current per-class transformer with one train-fitted shared
   transformer, then vary categorical temperature separately. Record
   per-class amount/gap/receiver effects as selection diagnostics. This is the
   minimum test of whether class-specific transform drift, rather than plan
   assembly, creates the large spurious label effects.
2. **TVAE:** retain the c03 categorical sampling path that restored receiver,
   but split its bundled interventions into categorical-decode-only and
   channel-weight-only candidates. Use train-only fixed temperatures, with
   gap and receiver reported separately. Do not add steps until a decode-only
   candidate resolves the observed category concentration.
3. **CoF:** replace the continuous amount objective/sampler only. Compare a
   standard noise-prediction diffusion objective with one explicitly
   preregistered variance-preserving residual-sampling alternative, both
   fitted and selected on train/validation only. Keep the existing discrete
   path unchanged. Z-score-only candidates should be removed because their
   stored outputs show stronger variance collapse.
4. Keep the five thresholds, SamplingPlan, endpoint, DGP, C2 rule, and
   validation-only selection rule unchanged. If a future finite candidate
   family has no all-five-guard PASS candidate for any primary model, stop
   again without fresh-test or full execution.

## Forensic artifact hashes

| Artifact | SHA-256 |
|---|---|
| `forensic_selection_failure_v2_6.json` | `7235f1c6b6d1037a37c901c39948a975b175e59c73935762962d96c25510e8f2` |
| `forensic_selection_failure_v2_6.csv` | `848d1721e8ad0ee7563eff56664d6d4d82afb7e176a49fceecf31c673f5d283c` |
