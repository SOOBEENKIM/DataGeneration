# External sequence protocol v1 frozen-data materialization

## Status and scope

This is a source-only implementation. It does not itself authorize or perform external raw-data materialization. An execution authorization, when separately approved, is an append-only runtime provenance record and is not committed with this source. Plan and dry-run may read only the two development files (AMLSim transactions and Sparkov fraudTrain) for SHA-256 and CSV header verification; they do not parse a raw body or create a runtime directory. Sparkov fraudTest receives zero hash, header, body, and CSV-parse accesses in the materialization plan/dry-run.

The protocol input is `configs/benchmark_v2/external_sequence_protocol_v1.yaml`, SHA-256 `434e0694df60a8b85ba073e4bcaed9f6c5a1bf43ca2da61be69d87c845e1e3cc`. AMLSim uses `transactions.csv`; Sparkov uses only `fraudTrain.csv` as development data. Sparkov `fraudTest.csv` remains locked. The materialization loader rejects its role and filename before `pandas.read_csv` is called.

GPU inventory, CUDA, model fit/sample, fidelity/coherence, TSTR, privacy, and external test evaluation are outside this runner. The external threshold-bootstrap protocol is not executed.

## Frozen protocol

- Entity split is the protocol's seed-20260801 entity-grouped, retained-window-label-stratified 70/15/15 split.
- Events are stably ordered by `(TIMESTAMP, TX_ID)` for AMLSim and `(trans_date_trans_time, original row order)` for Sparkov.
- Windows are non-overlapping 32-event blocks. Tails of length 16–31 are kept and shorter tails are dropped.
- Sequence `Y` is one exactly when a retained window contains at least one fraud/laundering transaction.
- Amount `log1p` population standardization, 16-bin positive-gap quantiles, gap representatives, and receiver vocabulary are fit from valid rows of retained train windows only.
- Receiver code 0 is PAD and code 1 is UNK. Validation/internal-test receivers absent from the train vocabulary must encode as 1.
- Entity, window, and transaction membership must be pairwise disjoint across train, validation, and internal test. The writer independently recomputes these intersections instead of trusting the adapter's audit field.

## Append-only attempt contract

The configured future roots are:

```text
data/external_sequence_protocol_v1/frozen/<dataset>/attempt_001/
artifacts/external_sequence_protocol_v1/materialization/<dataset>/attempt_001/
```

An attempt is rejected before raw-body access if either path already exists. The runner never removes, moves, or overwrites an attempt. `RUNNING.json` is created first. A successful data attempt contains:

- `raw_schema_manifest.json`: raw path, role, SHA-256, byte/row count, ordered columns, and dtypes;
- `entity_split_manifest.json`: typed entity identity, split, stratification method, seed, and label counts;
- `window_manifest.json`: split, typed entity, ordinal, length, label, and transaction-membership hash;
- `train.npz`, `validation.npz`, and `internal_test.npz`: numeric amount, gap bin, receiver code, mask, sequence label, length, and entity identity arrays;
- `train_transform_state.json`: amount state, reversible `+inf` gap-edge encoding, gap representatives, typed receiver vocabulary, PAD/UNK codes, fit-membership hashes, and state hash;
- `summary.json`: window and transaction fraud prevalence, lengths, row/entity counts, receiver cardinality, and PAD/UNK counts by split;
- `leakage_audit.json` and `provenance_manifest.json`.

The artifact attempt contains `checksum_manifest.json` and `artifact_index.json`. `COMPLETE.json` is exclusive-written only after every data file and checksum has been completed. An exception produces `FAILED.json` without deleting partial evidence. A second invocation cannot reuse that path.

## Provenance and execution authorization

Future execute mode is fail-closed and requires a separately created append-only authorization with all of the following exact bindings:

- current source commit and materializer source SHA-256;
- materialization and protocol config SHA-256;
- `attempt_001` and the exact AMLSim/Sparkov development raw paths and SHA-256 values;
- sole allowed operation `frozen_data_materialization`;
- explicit prohibition of GPU/CUDA, model fit/sample, fidelity/coherence, TSTR/privacy, external-test evaluation, and Sparkov public-test parsing.

No such authorization is created here. A mismatch is rejected before a raw body is read. Execute mode also preflights both dataset attempt paths before either dataset body is parsed.

The body-free planning count evidence comes from the frozen feasibility audit. Before applying protocol v1's minimum-tail rule, it contains 45,501 AMLSim and 41,019 Sparkov fraudTrain non-overlapping candidate windows. Because the earlier audit retained every nonempty tail while protocol v1 drops tails shorter than 16, the conservative retained-window planning ranges are 35,502–45,501 for AMLSim and 40,036–41,019 for Sparkov. The planned entity split fractions are 70/15/15. Exact split and retained-window counts cannot be derived from headers without violating the raw-body boundary; they are written to `entity_split_manifest.json` and `window_manifest.json` by an authorized materialization.

## Frozen non-v3 CoF compatibility

The external baseline remains source commit `99a445f6dc893a8c2240d950de4f92877cc07f8a`, config SHA-256 `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3`. Static inspection uses the exact adapter Git blob SHA-256 `48e9e1280bc23a27abf0b5f6447f6682e3b846d0d1773fb96d99abcf407dd7f3`; it does not import or instantiate the model.

The adapter derives numeric width, gap cardinality, receiver cardinality, and maximum sequence length from the training `SequenceBatch`. `SeqDenoiser` constructs its numeric projection and gap/receiver/position embeddings from those values. Therefore amount width 1, gap cardinality 16, receiver width 1, length 32, and the observed upper bounds of 9,928 AMLSim receiver states and 695 Sparkov receiver states require no architecture change. This is only a static input-shape result. No training runner is created and no claim about memory, fit quality, or external fidelity is made.

## Preservation and verification boundary

Plan/dry-run revalidates the frozen v2.5 final marker, v2.5 data manifest, v2.8 aggregate marker, and v3 forensic evidence through the hashes already pinned in the protocol config. It also verifies the frozen non-v3 CoF model, adapter, and config Git blobs. Existing v2.5–v3 runtime/data/checkpoint/aggregate artifacts remain read-only.

Fixture tests exercise the materialization core only under pytest temporary directories. They do not read or write the real external runtime roots and do not invoke a model or accelerator. Repository regression is limited to the source-only external protocol, feasibility audit, and materialization suites because broader model suites can perform model sampling, which is explicitly prohibited in this phase.
