# AMLSim external validation stored-result aggregate and forensic

Status: read-only aggregate of four stored `evaluation.json` and terminal artifacts. No sample, checkpoint, frozen array, Sparkov artifact, internal test, model, GPU, or evaluator was opened or executed.

## Hash-bound inputs

| Model | Attempt | Terminal | Evaluation SHA-256 | Terminal SHA-256 |
|---|---|---|---|---|
| empirical IID | `attempt_002` | COMPLETE | `9510091563d4c5e8f57a654a4b447a9cd0f8087fae0d9d98415f290fba7e1ef3` | `81c8105f62646d833dcbf016b371c1f22e783988d7d5524f84907d22f87c800a` |
| CTGAN | `attempt_001` | COMPLETE | `01e5c7dcb041052d9a6cb2e9c0cb7b814653708f909e610d55b4ce15b87c8542` | `58deb1884cf086a902181d6c30e372d1a04c79e9ff6eb327789a8e3613f38754` |
| TVAE | `attempt_002` | COMPLETE | `b8badf398a03cc6b65a46d22f510afb3e7a06a8756f8889f69ad016817b5a4a0` | `12679b9113d82be0e11aba6019637b7832449112a300ba1144fdff2557954eb0` |
| frozen non-v3 CoF | `attempt_001` | COMPLETE | `db7f7e99f7a43c6cbdb259b663f32628bd44440e01f145e17be305daf1a5d68a` | `e44db66eaaa6c65142df3cde8fa283dd5d05df9f7386a3d35870d51780db7f35` |

All four terminal artifacts say `COMPLETE` and `test_execution_authorized=false`. All four evaluations report PASS for mask, padding, and train discrete support; they use the same SamplingPlan SHA-256 `9e36a197…4a9d` and threshold SHA-256 `e715b01a…0330`.

## Stored ranking

The table transcribes the stored selection fields. Lower combined score is better; no metric was recalculated.

| Rank | Model | Fidelity max ratio | Coherence max ratio | Combined score | Passed metrics / 8 | Tier |
|---:|---|---:|---:|---:|---:|---|
| 1 | empirical IID | 1.8783 | 13.7394 | 7.8088 | 0 | FIDELITY_WARNING |
| 2 | TVAE | 6.8964 | 13.1489 | 10.0226 | 2 | FIDELITY_WARNING |
| 3 | frozen non-v3 CoF | 12.3967 | **12.9044** | 12.6506 | 0 | FIDELITY_WARNING |
| 4 | CTGAN | 12.3592 | 13.7514 | 13.0553 | 0 | FIDELITY_WARNING |

CoF has the lowest stored coherence max-ratio, but the highest fidelity max-ratio. TVAE is the only model with any component PASS: amount KS Y1 and gap TV Y1.

## Stored fidelity values

| Model | Amount KS Y0 | Amount KS Y1 | Gap TV Y0 | Gap TV Y1 | Receiver TV Y0 | Receiver TV Y1 |
|---|---:|---:|---:|---:|---:|---:|
| empirical IID | 0.06074 | 0.06686 | 0.03025 | 0.04992 | 0.53723 | 0.78477 |
| CTGAN | 0.10561 | 0.12136 | 0.23602 | 0.12975 | 0.54650 | 0.78782 |
| TVAE | 0.22301 | **0.05602** | 0.03130 | **0.03236** | 0.84691 | 0.83789 |
| frozen non-v3 CoF | 0.31196 | 0.63456 | 0.23673 | 0.46619 | 0.74000 | 0.93360 |

CoF is not the best model on any listed fidelity value. Its amount and gap values are substantially worse than IID and TVAE, and its receiver values are worse than IID and CTGAN.

## Stored coherence values

| Model | Short-gap × receiver-repeat error Y0 | Y1 | Max ratio |
|---|---:|---:|---:|
| empirical IID | 0.22782 | 0.15617 | 13.7394 |
| CTGAN | 0.22802 | 0.15601 | 13.7514 |
| TVAE | 0.21803 | 0.15506 | 13.1489 |
| frozen non-v3 CoF | **0.21397** | **0.10870** | **12.9044** |

CoF is numerically best on both stored coherence errors and the coherence max-ratio. Those metrics still fail their train-only reference thresholds.

## Can CoF be claimed superior to the baselines?

**No general superiority claim is supported.** CoF ranks third of four on the preregistered stored combined score: it is worse than empirical IID and TVAE and better only than CTGAN. All eight CoF metric components fail, and CoF has the worst fidelity max-ratio. This is also one validation seed with no uncertainty estimate or inferential comparison, so statistical or population-level superiority cannot be claimed.

Two narrow descriptive statements are supported:

1. On these stored AMLSim validation artifacts, CoF has the lowest short-gap × receiver-repeat coherence errors and coherence max-ratio.
2. CoF's stored combined score is lower than CTGAN's, but not lower than IID's or TVAE's.

These statements do not unlock test execution and do not justify a claim that CoF is broadly better than the baseline family.

Machine-readable copies are `amlsim_external_validation_aggregate_forensic_v1.json` and `.csv`.
