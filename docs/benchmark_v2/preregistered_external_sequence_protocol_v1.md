# Preregistered external sequence protocol v1 (source-only)

## Status and scope

This document freezes the source-only data protocol for future AMLSim and Sparkov external validation. It does not authorize preprocessing output creation, model training, sampling, validation/test evaluation, threshold calibration, GPU/CUDA access, TSTR, privacy analysis, or a full run. No execution authorization or launch command exists.

The machine-readable protocol is `configs/benchmark_v2/external_sequence_protocol_v1.yaml`, SHA-256 `434e0694df60a8b85ba073e4bcaed9f6c5a1bf43ca2da61be69d87c845e1e3cc`.

## Frozen model identity

External experiments may use only the completed, non-v3 CoF-SeqGen v2.5 implementation:

- source commit: `99a445f6dc893a8c2240d950de4f92877cc07f8a`
- model: `models/cof_seqgen.py`, SHA-256 `caf5fdb3baf367a28c6081a8ba5f4a89e59dacbba2377636acba5c051ae7006e`
- adapter: `generators/cof_seqgen_adapter.py`, SHA-256 `48e9e1280bc23a27abf0b5f6447f6682e3b846d0d1773fb96d99abcf407dd7f3`
- config: `configs/benchmark_v2/full_v2_5.yaml`, SHA-256 `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3`

Architecture changes and v3 candidates are outside this protocol. The v2.5 training configuration identifies the frozen baseline but does not authorize a new external-data fit.

## Shared sequence contract

1. Sort transactions within entity using the dataset-specific keys below.
2. Slice with length 32 and stride 32. Windows never overlap.
3. Keep a final window of length 16–31 and right-pad it to 32 with a prefix-contiguous valid mask. Drop a final window shorter than 16.
4. Reset the first gap of every retained window to zero. No hidden pre-window context enters the sequence.
5. Define sequence label `Y=1` iff at least one valid transaction in that window has the raw fraud/laundering label.
6. Derive each entity's split stratum from whether any retained window for that entity has `Y=1`.
7. Within each stratum, use seed `20260801` for a deterministic entity shuffle and allocate 70%/15%/remaining 15% to train/validation/internal test. A stratum with fewer than three eligible entities fails closed.
8. Every entity, window identifier, and transaction identifier must occur in exactly one split/window. Missing or duplicate identities fail closed.

The split is computed before transform fitting. Oversampling, entity duplication, or window duplication before the split is forbidden.

## AMLSim

- raw source: `transactions.csv`, SHA-256 `cc4885da00fd4c6f854aae4e7bc8dd9c35a9b491f5ef871c51c9194ee1e292f7`
- entity: `SENDER_ACCOUNT_ID`
- stable order: `TIMESTAMP`, then `TX_ID`
- gap unit: integer simulation steps
- receiver: `RECEIVER_ACCOUNT_ID`
- amount: `TX_AMOUNT`
- transaction label: `IS_FRAUD`

`TX_ID` is a mandatory tie-breaker because the raw simulation uses only 200 timestamp steps and has extensive within-sender timestamp ties. `ALERT_ID`, raw labels, and transaction identifiers are provenance only and are not model inputs.

## Sparkov

- development source: `fraudTrain.csv`, SHA-256 `fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d`
- entity: `cc_num`
- stable order: parsed `trans_date_trans_time`, then immutable original row order
- gap unit: seconds
- receiver: `merchant`
- amount: `amt`
- transaction label: `is_fraud`

The empty exported index column and `trans_num` are not model features. `category` is not the receiver; it may be a separately reported merchant-type diagnostic only.

`fraudTest.csv`, SHA-256 `12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0`, is locked. In this source-only stage it may be hash/header checked for preservation, but it cannot be passed to the adapter, split, transform fit, threshold bootstrap, candidate selection, or evaluation. A future temporal robustness evaluation requires a separate authorization and must acknowledge the observed card overlap with `fraudTrain.csv`.

## Train-only transform contract

Only valid rows of retained train windows may fit state:

- amount: `log1p`, followed by population mean/standard-deviation normalization;
- gaps: 16 empirical quantile bins fitted from positive train gaps, with zero as the first edge and positive infinity as the last edge;
- receiver: sorted train-only vocabulary, with padding code 0 and UNK code 1.

Validation/internal-test receivers not present in the train vocabulary map to UNK. Validation/internal-test amounts, gaps, receivers, labels, lengths, and entity membership cannot affect fitted state. The state records hashes of its train transaction and entity identities plus a canonical state hash.

## External fidelity and coherence thresholds

The controlled benchmark's numeric five-guard thresholds are not copied. This implementation only freezes a possible future train-only threshold protocol:

- cluster unit: entity;
- source: train entities only;
- 2,000 cluster bootstrap replicates;
- interpolation-free rank 1,901/2,000 for a 95% cutoff;
- freeze thresholds before looking at validation;
- never reorient or recalibrate on the locked test.

Fidelity families are amount/gap marginals and Y effects, receiver PMF/UNK by Y, length/Y prevalence, support, and mask validity. Coherence families are receiver repeats/runs/transitions, joint gap-receiver by Y, amount-gap/amount-receiver dependence, and the original-versus-row-shuffle order-sensitive delta.

This bootstrap is not run in this commit. Its exact per-metric statistic, simultaneous family rule, and acceptance decision require a later preregistration before any execution.

## Row-shuffle negative control

Within each entity window, preserve Y, length, valid mask, timestamp positions, and row marginals while jointly permuting `(amount, receiver)` across valid positions. Fit and orientation are train-only. Validation may be used only after the statistic and cutoff are frozen; the locked test cannot select or reorient it.

## Source-only completion rule

Plan and dry-run may validate source/config/artifact hashes and raw CSV headers. They must report zero model fit/sample, transform fit, window materialization, external test evaluation, GPU inventory, CUDA, authorization creation, and launch-command creation. Any mismatch fails closed without writing runtime artifacts.
