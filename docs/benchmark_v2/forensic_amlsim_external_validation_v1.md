# AMLSim external validation v1 forensic audit

Status: read-only terminal-result audit and source-only infrastructure correction. No metric was recomputed, no evaluator formula or scientific threshold was changed, and no external job was launched.

## Preserved state

| Model | Attempt | Terminal | Tree SHA-256 | Evidence |
|---|---|---|---|---|
| empirical IID | `attempt_002` | COMPLETE | `2b602648c70419cb0a055a4d9b2fd3fb9237ccac267cd5f926d4586bdb717248` | evaluation `95100915…1ef3`, COMPLETE `81c8105f…800a` |
| CTGAN | `attempt_001` | COMPLETE | `1da3ef166302a27135d2adace6b4740278ef11e2fe6de8b3312e89f0007d0734` | evaluation `01e5c7db…8542`, COMPLETE `58deb188…754` |
| TVAE | `attempt_001` | FAILED | `d98ed71eb09f2ca07c7f53d42c16df0c1035477f783ed3b5733ccd038c2c9272` | FAILED `b0ca7305…8cdf`, log `b4c95d72…14c60` |
| frozen non-v3 CoF | `attempt_001` | COMPLETE | `e660dcdb324cb0f9394b2c69e03c751cdeee73f4b37c61e8b702d5547b0e541e` | evaluation `db7f7e99…68a`, COMPLETE `e44db66e…7f35` |

The AMLSim frozen bundle tree remained `742735126e3a511dc612271102a9d13008e0a8a85552948c9fb9fd1f9709309d`. The protocol-pinned bundle digest remains `cc8f4c9c6ff415ccd4f78ec193122100145e10084751b971dcc0ce5e705fa024`; the two values use different documented tree algorithms and are not a discrepancy. Runtime files were neither written nor moved during this audit.

## Why empirical IID failed every stored external threshold

The table is a transcription of `evaluation.json` and its immutable train-bootstrap threshold bundle. It is not a fresh metric calculation.

| Stored metric | Y | Value | Train-only threshold | Ratio | PASS |
|---|---:|---:|---:|---:|---|
| amount KS | 0 | 0.0607380 | 0.0323370 | 1.8783 | FAIL |
| amount KS | 1 | 0.0668600 | 0.0613575 | 1.0897 | FAIL |
| gap TV | 0 | 0.0302539 | 0.0190966 | 1.5843 | FAIL |
| gap TV | 1 | 0.0499160 | 0.0395268 | 1.2628 | FAIL |
| receiver TV | 0 | 0.5372273 | 0.3047172 | 1.7630 | FAIL |
| receiver TV | 1 | 0.7847657 | 0.4358637 | 1.8005 | FAIL |
| short-gap × receiver-repeat error | 0 | 0.2278159 | 0.0165813 | 13.7394 | FAIL |
| short-gap × receiver-repeat error | 1 | 0.1561709 | 0.0314149 | 4.9712 | FAIL |

The threshold file records 1,000 entity-cluster bootstrap pairs from train only, rank 951 without interpolation, and `frozen_before_validation=true`. It therefore measures train-internal resampling variation. The validation split contains different sender entities. Empirical IID samples train class marginals while the stored evaluation compares those samples with the corresponding new-entity validation distributions; an excess can therefore reflect real entity-level distribution shift even when the sampler is correct.

Class size matters. Train contains 29,267 Y=0 and 1,082 Y=1 sequences; validation contains 6,311 Y=0 and 231 Y=1 sequences. The smaller positive class has wider stored train-bootstrap thresholds for every fidelity family, but its observed receiver TV still reaches 0.7848 and remains 1.8005 times threshold. Exact valid-row and UNK counts by class are not present in the permitted JSON evidence, so this audit does not invent or infer them from NPZ arrays.

Receiver fidelity is the most visibly high-dimensional marginal problem. The train-fitted vocabulary contains 9,654 observed receivers plus PAD and UNK, for cardinality 9,656. Validation has 3,658 UNK-coded rows among 197,634 valid rows (1.851%), whereas train has no UNK-coded row. That unseen-receiver mass necessarily contributes to train-versus-validation mismatch, but it is far smaller than the stored TV values of 0.5372 and 0.7848. It cannot explain those values alone. Sparse class-conditional mass moving across thousands of known receiver categories, together with the unseen mass, is the supported explanation.

The dominant failure is coherence, not a row-support bug. IID sequence assembly independently resamples rows, so it intentionally removes the temporal coupling between short gaps and receiver repetition. The 13.74× and 4.97× threshold ratios are the expected signature. Meanwhile mask, padding, and train discrete support all PASS. The amount and gap failures show that the result is also not reducible to receiver UNK alone.

The stored selection status is `SELECTED` with tier `FIDELITY_WARNING`, consistent with the preregistered rule that an all-pass failure does not by itself permanently block a later test proposal. This audit does not authorize such a proposal.

## TVAE global-OOM diagnosis

The terminal traceback is unambiguous:

```text
TVAE fit
  -> CheckpointableTVAE._initialize_v2_5
  -> DataTransformer.transform
  -> DataTransformer._parallel_transform
  -> joblib.Parallel(n_jobs=-1)
  -> TerminatedWorkerError, worker SIGKILL(-9)
```

Kernel evidence supplied with the task identifies a global OOM and a killed Python process. CTGAN and TVAE were transforming the large AMLSim input concurrently; CTGAN completed only after TVAE died. This classifies the event as an infrastructure/resource-scheduling failure, not a learned-model or metric result.

The receiver one-hot output dominates scale. A conservative, non-allocating whole-train shape estimate is `914,756 × 9,656 × 8 = 65.81 GiB` for one float64 receiver matrix. Two concurrent AMLSim tabular transforms reach 131.62 GiB before input copies, joblib serialization, other transformed columns, model state, or OS overhead. The analogous Sparkov shape is 4.65 GiB (`898,168 × 695 × 8`), making AMLSim approximately 14.15 times larger. Separate-class fitting transforms class subsets, so these are risk upper bounds rather than measured peak RSS; they nevertheless reproduce the cardinality scaling that makes the confirmed OOM plausible.

## Corrective contract

For external CTGAN and TVAE only:

- `DataTransformer` must use fixed `n_jobs=1` and the in-process synchronous column path; `n_jobs=-1` is fail-closed.
- At most one CTGAN/TVAE heavy transform may be active globally. The authorization job contract uses one shared exclusion group, `external_tabular_transform`.
- The launch schedule separates CTGAN and TVAE into different terminal-barrier waves for both datasets. CoF may share the CTGAN wave because it does not use this transform.
- Algorithm, seed, separate-class handling, Y/L conditioning, train/validation hashes, model hyperparameters, wall caps, metrics, thresholds, and selection rules are unchanged.

The correction removes joblib worker processes and cross-model heavy-transform overlap. It does not claim that the remaining single dense transform has been measured on AMLSim; any future attempt remains subject to the unchanged wall cap and fail-closed terminal handling.

## Can the completed CTGAN result be reused?

Yes, but only under the explicit resource-only mixed-provenance exception implemented by this corrective source. The completed result must exactly match its immutable manifest, evaluation, and COMPLETE hashes plus source `2ae74969…`, prior config `f14e1827…`, AMLSim bundle/train/validation hashes, seed 31001, metric source, SamplingPlan, and threshold bundle. The validator reports `REUSE_ELIGIBLE` only when every field matches. It does not reinterpret or recompute CTGAN.

Therefore the preferred next action is a separately authorized AMLSim TVAE `attempt_002` only. CTGAN does not scientifically require a rerun. If governance rejects mixed source provenance despite the exact resource-only exception, the conservative alternative is CTGAN `attempt_002` followed by TVAE `attempt_002`, strictly sequential; both would require a new authorization. No such authorization or attempt was created here.

Sparkov learned jobs remain blocked until AMLSim TVAE has a terminal artifact, the memory-safe path has been reviewed, and the user issues a separate authorization. When eventually authorized, Sparkov CTGAN and CoF may run together, followed by TVAE alone after a terminal barrier. Sparkov, internal test, fraudTest, TSTR, privacy, and full runs were not accessed or executed in this correction.

Machine-readable evidence is in `forensic_amlsim_external_validation_v1.json` and `forensic_amlsim_external_validation_v1.csv`.
