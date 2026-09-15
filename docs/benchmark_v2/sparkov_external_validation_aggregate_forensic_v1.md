# Sparkov external validation stored-result aggregate and forensic

Status: read-only aggregate of four stored `evaluation.json` and terminal
artifacts. No sample, checkpoint, frozen array, model, GPU, internal test,
Sparkov `fraudTest`, TSTR, privacy, or evaluator was opened or executed.

## Hash-bound inputs

| Model | Attempt | Terminal | Evaluation SHA-256 | Terminal SHA-256 |
|---|---|---|---|---|
| empirical IID | `attempt_002` | COMPLETE | `9da17648abf8deaf5d9715143b48c687d04b88711b06b77fdd39d3c325d8d1b8` | `ad2fc7fe471b2b92dba37e14e7c5b77903c535cd586df9bf3336d82aba1cfa80` |
| CTGAN | `attempt_001` | COMPLETE | `b8293768375c7996ff543cc04771247a88ebbf25fdddb5eafcb7e2175302e33a` | `4190ad0aca4a7cb3d06c1cc62c8a607e347463a184de0f6432559124d7679ac4` |
| TVAE | `attempt_001` | COMPLETE | `d21283b18c477eac22bf5e5dcc76d67ca5304b6130e1abf669bd8de3dfce06c2` | `157b697d3bf0a3da3d9d5f1d42ac855884ce75d0195d535017c45c88fc66e1d7` |
| frozen non-v3 CoF | `attempt_001` | COMPLETE | `1597b95122d527fed21aa4dbd26448719b28fd5c92102d3722352ac12db0f990` | `e891af8bce3bb76d277e2d8139a55f08166745f336550e7e48e84133d6eeb9a6` |

All four evaluations report PASS for mask, padding, and train discrete support.
They share SamplingPlan SHA-256
`665c990d78e01bd31844ca5433f177017955b35915941080bc0648deb957e247`
and threshold SHA-256
`c091c874e8a32c32eb9516e7d56c5946e15342a6837806f8f3809c7972424efb`.
Every terminal and evaluation keeps `test_execution_authorized=false`.

## Stored aggregate ranking

All three scores are lower-is-better. Values below are copied from the stored
`selection` objects; no guard or metric was recomputed.

| Combined rank | Model | Fidelity max ratio | Fidelity rank | Coherence max ratio | Coherence rank | Combined score | Passed metrics / 8 | Tier |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | empirical IID | 2.096461 | 1 | 5.019435 | 3 | 3.557948 | 1 | FIDELITY_WARNING |
| 2 | CTGAN | 10.486262 | 2 | 3.940879 | 2 | 7.213571 | 1 | FIDELITY_WARNING |
| 3 | frozen non-v3 CoF | 13.063355 | 3 | **3.857913** | **1** | 8.460634 | 1 | FIDELITY_WARNING |
| 4 | TVAE | 48.694145 | 4 | 429.970598 | 4 | 239.332371 | 0 | FIDELITY_WARNING |

CoF has the best stored coherence max-ratio, but IID and CTGAN both rank above
CoF on fidelity max-ratio and combined score. All four models have
`fidelity_all_pass=false`; therefore this validation aggregate does not assert
that any model satisfies every external fidelity guard.

## Stored fidelity values

| Model | Amount KS Y0 | Amount KS Y1 | Gap TV Y0 | Gap TV Y1 | Receiver TV Y0 | Receiver TV Y1 |
|---|---:|---:|---:|---:|---:|---:|
| empirical IID | 0.025494 | 0.026078 | 0.012580 | 0.036610 | 0.036233 | 0.234704 |
| CTGAN | 0.142443 | 0.227934 | 0.172277 | 0.200351 | 0.082897 | 0.245236 |
| TVAE | 0.102671 | 0.040873 | 0.824905 | 0.458877 | 0.848287 | 0.573721 |
| frozen non-v3 CoF | 0.230025 | 0.135908 | 0.046261 | 0.308927 | 0.067917 | 0.327984 |

## Stored coherence values

| Model | Short-gap × receiver-repeat error Y0 | Y1 | Coherence max ratio |
|---|---:|---:|---:|
| empirical IID | 0.000653 | 0.001035 | 5.019435 |
| CTGAN | 0.000513 | **0.000000** | 3.940879 |
| TVAE | 0.055947 | 0.001553 | 429.970598 |
| frozen non-v3 CoF | **0.000502** | 0.000518 | **3.857913** |

## Preregistered CoF-versus-TVAE decision

The preregistered continuation rule requires both strict inequalities. Ties do
not pass.

| Required condition | Stored comparison | Result |
|---|---|---|
| CoF combined score < TVAE combined score | `8.460633810333322 < 239.33237112535298` | PASS |
| CoF fidelity max ratio < TVAE fidelity max ratio | `13.06335487722038 < 48.69414455371112` | PASS |

Both attempts are COMPLETE and hard-valid. The TVAE-minus-CoF margins are
230.87173731501966 for combined score and 35.630789676490735 for fidelity
max-ratio. Therefore the exact preregistered decision is:

`CONTINUE_TO_SEPARATE_TEST_PROPOSAL`

`STOP_EXISTING_COF_FAMILY_MODEL_LEVEL_REDESIGN_REQUIRED` is not triggered
because neither required inequality failed. This is only the predefined
CoF-versus-TVAE validation decision. It does not claim CoF is best overall,
does not override its third-place combined/fidelity ranking, and does not
authorize internal test, `fraudTest`, TSTR, privacy, tuning, or any new run.

Machine-readable copies are
`sparkov_external_validation_aggregate_forensic_v1.json` and `.csv`.
