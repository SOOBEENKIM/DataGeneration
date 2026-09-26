# Berka: first version-qualified replication

Registered 2026-09-27 before computing scores or fitting this run. This is not an assertion that the authors used these exact versions. No new Sparkov experiments are authorized by this protocol.

## Fixed sources and environment

- Author repository: `mostly-ai/paper-tabular-argn`, commit `97781f9cab4ef91e347dfc31650e38ef8dcf2e56`.
- Original author Berka training split: 2,250 accounts / 526,442 transactions. Do not use author holdout for fitting.
- Fresh isolated environment: engine 1.0.4 (2025-02-06), QA 1.5.3 (2025-02-04), torch 2.5.1 CPU. These are releases near paper v2 (2025-02-06); the author repo does not pin the actual experiment versions. Full installed dependencies are captured separately.
- Seed 20260927; CPU with 4 torch threads. Hardware differs from the paper's A10G; do not compare generation speed.

## Data and training

Load CSV with `low_memory=False`, then nullable pandas dtypes. Keep dates as strings and input sequence order. This compatibility choice is supported by the author synthetic account output: 2,247 / 2,250 date values are `_RARE_`, and transaction dates also contain that categorical rare marker. Do not silently parse, repair, sort generated dates, or optimize preprocessing to match a target score.

Execution compatibility correction before any child fitting: store complete key columns as numpy `int64`. In engine 1.0.4, nullable `Int64` keys produce nullable sequence-length counts rejected by `encode_slen_sidx_sdec`; attributes retain their original nullable types. This changes no values, split seed or learned attributes. Attempt 1 failed before parent fitting because the default HF cache was not writable; attempt 2 finished the parent but stopped during child encoding at this dtype assertion. Both are retained; attempt 3 uses a writable local cache and native-compatible key types. Installed engine source remains unchanged.

Declare account_id as parent primary key and transaction foreign key; trans_id as transaction primary key, excluded from learned attributes and scoring. Preserve all other original fields. Generate 2,250 synthetic parents, train the child on real training parents, then generate transactions conditional on the synthetic parents. This follows the public Baseball two-table script, adapted to Berka keys.

Use unmodified engine `split → analyze → encode → train → generate`; native internal 90/10 subject split; `train(max_training_time=300)` and other model defaults. The 300-minute limit is per table; native early stopping applies. Do not alter native validation, model size, encoding heuristics, sequence window, batch size or sampling temperature. Capture resolved settings in workspaces and logs. Run one complete fit first, not five trials presented as completed in advance.

## Evaluation before and after training

1. Score author-provided non-DP run 1 first. It is a reference artifact, never counted as our newly trained model.
2. Use QA 1.5.3's native data preparation, binning, univariate/bivariate aggregation and coherence. Include the context table, exclude primary IDs, preserve date strings. Accuracy-only execution does not evaluate DCR/privacy or produce a fraud-detection score.
3. Preserve the released sampler as the primary compatibility score. Source inspection found `np.random.random(...).astype(int)` occurs before multiplication, so every sampled starting position is zero. Verify on an independent 100-subject fixture. Record that this assesses first-pair relationships.
4. Separately report a predeclared random-adjacent-pair sensitivity matching Appendix E's verbal sampling description, with identical seeds and remaining QA operations. This locally replaces only the sampler during the diagnostic call and restores it afterward; never modify installed QA or training code. The paper additionally describes all-row univariates/bivariates, whereas this QA version samples one row per subject for those too. Thus neither score alone establishes exact paper metric replication.
5. Compare author-artifact and freshly generated scores under the same explicit scoring procedure. Paper table 7 means/ranges (.79 overall [.77,.80], .87 univariate [.85,.88], .68 bivariate [.66,.69], .82 coherence [.81,.83]) are contextual references, not thresholds to tune against.

Follow-up after run 1 matched the approximate table values: evaluate the remaining four public author transaction artifacts with the same fixed seed and both predeclared samplers. Keep the one public author account table and assert foreign-key coverage for every transaction artifact. This checks the reported five-run aggregate; it does not add four new model fits.

## Rerunning

Python 3.10 was used. Install `berka_replication/runtime-freeze.txt` in a separate virtual environment with `--extra-index-url https://download.pytorch.org/whl/cpu`; `pip check` passed in the recorded environment. Preserve the author checkout at the pinned revision. From the project repository, invoke the script with explicit paths:

```bash
python scripts/reproduce_argn_berka.py score --paper-repo /path/to/paper-tabular-argn --out /path/to/author-evaluation --author-run 1
python scripts/reproduce_argn_berka.py train --paper-repo /path/to/paper-tabular-argn --out /path/to/new-run --seed 20260927
python scripts/reproduce_argn_berka.py score --paper-repo /path/to/paper-tabular-argn --out /path/to/new-run-evaluation --synthetic-run /path/to/new-run
```

Set `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=4` for the recorded CPU resource cap. The runner places its default HF cache beside the output directory. It refuses to overwrite a training run, records failures in its manifest, and never reads the external holdout during training or the accuracy-only evaluation.

## Completion wording

Distinguish author-output evaluation, one fresh Berka run, and exact paper reproduction. A fresh run is a version-qualified replication until the missing author version, Berka configuration and sequential scoring protocol are reconciled. The author output's categorical date suppression and non-monotonic dates are observations about that released artifact, not proof that ARGN intrinsically cannot model time. Retain all outputs and failed attempts with their provenance.
