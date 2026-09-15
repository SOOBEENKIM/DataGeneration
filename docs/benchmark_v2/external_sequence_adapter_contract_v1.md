# External sequence adapter contract v1

## Public interface

`data.external_sequence_adapter` exposes the source-only protocol primitives:

- `build_raw_windows`: schema validation, stable ordering, non-overlapping 32-event windows, 16-event minimum tail, deterministic transaction membership, and window Y;
- `group_stratified_entity_split`: deterministic entity-level 70/15/15 allocation stratified by any-fraud entity label;
- `fit_train_transforms`: train-role-only amount/gap/vocabulary state with identity and state hashes;
- `encode_windows`: right-padded `SequenceBatch` encoding with prefix-contiguous mask and train-only UNK mapping;
- `prepare_external_dataset`: retained-window stratification, train-only fitting, three encoded splits, and entity/window/transaction leakage audit;
- `read_development_csv`: allowed development-source read with pre-read rejection of Sparkov's public test.

The module imports no model, sampler, CUDA, or training runner. It produces in-memory protocol objects only; it has no runtime-artifact writer.

## Observable data contract

`PreparedExternalDataset` contains train, validation, and internal-test `SequenceBatch` values plus raw window membership, split assignment, transform state, and a leakage audit. Valid positions are always the left prefix, padding values are zero, and each sequence has one entity-level Y.

Receiver code 0 is padding, 1 is UNK, and fitted train categories begin at 2. The amount channel contains train-fitted `log1p` z-scores. `dt_bin` contains train-fitted gap-bin indices and resets to bin 0 at each window start.

## Fail-closed conditions

The adapter refuses:

- unsupported/missing dataset columns;
- missing entities or duplicate/missing transaction IDs;
- non-binary fraud labels;
- negative/non-finite amount or gap values;
- a label stratum with fewer than three eligible entities;
- non-train transform fitting;
- empty eligible splits/windows;
- dataset/state mismatch;
- any entity, window, or transaction overlap across splits;
- Sparkov public-test paths or roles before CSV parsing.

## Plan and dry-run

`scripts.plan_external_sequence_protocol` has only `plan` and `dry-run` modes. `plan` validates the frozen non-v3 source/config and preservation anchors without reading raw CSVs. `dry-run` additionally hashes and reads only the headers of AMLSim transactions, Sparkov fraudTrain, and the locked Sparkov fraudTest. Both modes emit JSON to stdout and create no artifact.

The source-only commands are:

```bash
<COFSEQ_PYTHON> -m scripts.plan_external_sequence_protocol \
  --repo-root . \
  --config configs/benchmark_v2/external_sequence_protocol_v1.yaml \
  --mode plan

<COFSEQ_PYTHON> -m scripts.plan_external_sequence_protocol \
  --repo-root . \
  --config configs/benchmark_v2/external_sequence_protocol_v1.yaml \
  --mode dry-run
```

These are validation commands, not external execution commands. No authorization or launch command is defined.
