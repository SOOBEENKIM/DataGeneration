# CoFSeqGen-SAF data acquisition and materialization report

## Outcome

Acquisition and canonical materialization completed on 2026-08-26 for every
declared source: the controlled coupling DGP, AMLSim, Sparkov `fraudTrain`,
Berka, H&M, and Citi Bike. No baseline, model, GPU, training, sampling,
evaluation, privacy analysis, or research claim was executed.

| Canonical dataset/cell | Entities | Events | Train / validation / test entities | Failed gates |
|---|---:|---:|---:|---:|
| Controlled joint semi-Markov κ=0 | 45,645 | 1,095,902 | 31,951 / 6,846 / 6,848 | 0 |
| Controlled joint semi-Markov κ=1 | 45,645 | 1,095,902 | 31,951 / 6,846 / 6,848 | 0 |
| AMLSim | 9,999 | 1,323,234 | 6,999 / 1,499 / 1,501 | 0 |
| Sparkov | 983 | 1,296,675 | 688 / 147 / 148 | 0 |
| Berka full | 4,500 | 1,056,320 | 3,150 / 675 / 675 | 0 |
| Berka deterministic nested 900 | 900 | 211,249 | 630 / 135 / 135 | 0 |
| H&M deterministic 10,000 | 10,000 | 236,028 | 7,000 / 1,500 / 1,500 | 0 |
| Citi Bike 2016-02 | 7,483 | 560,874 | 5,238 / 1,122 / 1,123 | 0 |

Every materialized cell passed entity-leakage, timestamp parsing, nonnegative
gap, exactly-one-missing-first-gap-per-entity, train-only vocabulary,
sequence-report, source-manifest, oracle-identity, and mark-context gates.
Raw acquired files are read-only and verified against their acquisition
manifests.

## Reproducibility evidence

The seven previously completed canonical datasets and the newly acquired H&M
dataset were each materialized twice consecutively. After final source-hash
pinning, the two H&M runs produced identical H&M manifest, dataset-report, and
global-summary hashes while leaving the other seven manifests unchanged.

| Artifact | SHA-256 |
|---|---|
| AMLSim canonical manifest | `253ff73ebf06cb80617007b780b6909d6a4688abf4c0ae89b76e4e6cf791df7c` |
| Sparkov canonical manifest | `5ec8d547e72823b90a989a9bb596b2aa96be2b50673518a85b17adf7bd731cc0` |
| Berka full canonical manifest | `76a568f0db09b5fa5b951bb33e0d04bd20870b0df0e2d4b7e26851fe74244237` |
| Berka nested-900 canonical manifest | `ab2a4964a250d6bd38166dc9d93be0a7fba35233f6e09068bbb14ad666783253` |
| H&M canonical manifest | `add788132b5bb487299045b37c4081c148397f14fd40d9bb0f9f3a4d2eda3cb2` |
| H&M dataset report | `1970c73838fb2bf2f3f0efed0d7abb49206893f0ed2c6fd6f42d6afd2140081c` |
| Citi Bike canonical manifest | `5b7dca8c83b52c644001f7a6388116c469e438f7a1d1968cee8a17037c467b93` |
| Controlled κ=0 canonical manifest | `d10d58ff0e070ea818f6487dd16d64b6f18cc16a8df21b5a20778c6ce6484ec6` |
| Controlled κ=1 canonical manifest | `247cc11a636295794b55363072924e5e8aba8007c60e5d61b367a9f8ed0ff43a` |
| Global materialization summary | `91998b4e383b7b5c85fa598402b9e9c0c8a0610f16adc11e0a3bf9fa81d4a11e` |

The independent verifier reports `all_checks_pass: true`. It re-hashes every
acquired source and every file named in each canonical manifest and rejects a
writable raw source, stale config hash, self-referential manifest, or failed
dataset gate.

## Dataset-specific decisions

- Controlled DGP: κ=0 and κ=1 share entity labels, lengths, and split IDs.
  Oracle states are written to `oracle_latents.parquet`, outside canonical
  model inputs. The first sampled interval sets the absolute time origin; the
  canonical first gap remains missing.
- Sparkov: all 1,296,675 supplied `unix_time` values disagree with parsed
  `trans_date_trans_time`. The readable timestamp string is authoritative and
  the disagreement is recorded. `fraudTest.csv` is not pooled.
- Berka: the comparison view matches TabDiT's published five parent and six
  child fields. The deterministic 900-account view is not the unpublished
  TabDiT test split and is marked non-comparable; it is not truncated to
  TabDiT's maximum sequence length 50.
- H&M: the adapter deterministically hashes active customer IDs before taking
  10,000, then makes an entity-disjoint split. It preserves the selected
  article parent rows as `mark_context.parquet`. The exact Seq2Synth sample
  cannot be claimed because its sampled customer IDs and random seed are not
  published. All 31,788,324 transaction rows were scanned; timestamp parse,
  negative-price, negative-gap, and entity-leakage counts are zero. The raw
  CSV hashes are pinned in the acquisition config. Kaggle's CLI returned ZIP
  payloads under CSV names, so acquisition detects the byte signature and
  extracts the named member before hashing. Images and submission files were
  never downloaded.
- Citi Bike: the official 2016-02 source has 560,874 rows and 7,483 bikes,
  whereas Seq2Synth reports 559,644 rows and 5,793 trajectories. It is therefore
  schema-aligned only, not identity-matched.

## Reproduction and verification commands

With competition access and a Kaggle credential stored outside the repository:

```bash
python -m scripts.acquire_cof_seqgen_saf --repo-root . --datasets hm
python -m scripts.materialize_cof_seqgen_saf --repo-root . --datasets hm
python -m scripts.verify_cof_seqgen_saf_data --repo-root .
```
