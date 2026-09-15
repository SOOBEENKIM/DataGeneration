# Sparkov external validation source-only wave plan v1

This document is a scheduling specification, not an execution authorization.
It preserves all existing AMLSim, Sparkov IID, frozen-data, authorization, and
controlled-benchmark artifacts read-only.

## Inputs

- Reused reference: Sparkov empirical IID `attempt_002`, terminal COMPLETE.
- Pending jobs: CTGAN `attempt_001`, frozen non-v3 CoF `attempt_001`, and TVAE
  `attempt_001`.
- Frozen dataset: Sparkov `fraudTrain` materialization tree
  `62fefda6207911104a9bff0ebb41d5edf44c20a4c78dccef5875d9768be6addc`.
- Seed: 32001 for all four model records.

## Future wave contract

Wave 1 has two jobs: CTGAN and CoF. They require two distinct idle physical
GPUs, with only one device exposed to each process and runner device `cuda:0`.
The wave has a terminal barrier: both attempts must end in exactly one of
COMPLETE, INVALID, or FAILED before any wave-2 process may begin.

Wave 2 has one job: TVAE on one idle physical GPU, again with one exposed
device and runner device `cuda:0`. It runs alone. This enforces the fixed
low-parallelism memory contract and prevents CTGAN and TVAE transforms from
overlapping.

No job may be retried, tuned, early-stopped based on validation results, or
added to the matrix. Attempt paths are append-only and execution must fail if
an intended attempt path already exists.

## Non-executing validation commands

The following commands inspect manifests and hashes only. They do not create
an authorization or runtime artifact and do not query CUDA/GPU state:

```bash
python -m scripts.prepare_sparkov_external_validation_v1 \
  --repo-root . \
  --config configs/benchmark_v2/external_validation_v1.yaml \
  --mode plan \
  --approval-text "source-only Sparkov validation preparation"

python -m scripts.prepare_sparkov_external_validation_v1 \
  --repo-root . \
  --config configs/benchmark_v2/external_validation_v1.yaml \
  --mode dry-run \
  --approval-text "source-only Sparkov validation preparation"
```

An actual append-only authorization and SSH/tmux launch commands require a
later, explicit execution authorization. They are intentionally absent here.
