# Benchmark v2.5 frozen-data strict-JSON correction

## Scope

This is a source-only serialization correction for v2.5 frozen-data
provenance. It does not change the DGP, generated arrays, kappa, training N,
model config, training path, endpoint, C2 rule, capacity measurements, or any
threshold.

Preparation attempt 001 stopped after writing the three split arrays, shared
SamplingPlan, and row-guard calibration but before `meta.json` and the final
`data_manifest.json`. The preserved runtime failure inventory and traceback
are under:

```text
artifacts/benchmark_v2_5/failed_preparation_attempts/attempt_001/
```

Those runtime artifacts remain excluded from Git.

## Encoding contract

Only the provenance fields `bin_edges` and `tau` in the v2.5 frozen
`meta.json` use the explicit encoding
`benchmark-v2.5-provenance-float-sequence-v1`.

- Every finite float remains a JSON number.
- Negative infinity is `{"__benchmark_v2_5_float__": "-Infinity"}`.
- Positive infinity is `{"__benchmark_v2_5_float__": "+Infinity"}`.
- NaN remains forbidden and aborts preparation.
- Bare non-finite JSON numbers are never permitted; JSON is still written
  with `allow_nan=False`.

The metadata contains a `provenance_float_encoding` object that identifies the
schema, covered fields, and tagged representations. The matching reader
validates this schema and restores the exact float values and infinity signs.
The encoding is reversible and does not feed changed values back into the
DGP.

## Publication order

Preparation writes split arrays, the SamplingPlan, row-guard thresholds, and
strictly encoded `meta.json` first. It hashes every referenced artifact and
creates `data_manifest.json` with `status=COMPLETE` last. A serialization
failure, including NaN, leaves no COMPLETE manifest.

The preparation writer and full-run reader use the same
`experiments.provenance_v2_5.hash_batch` definition, including entity IDs, so
a successfully published manifest is accepted unchanged by the frozen-input
reader.

## Corrected preparation command

The archived failed output does not occupy the exclusive output paths. A
future, separately authorized CPU-only preparation retry must use:

```bash
<COFSEQ_PYTHON> \
  -m scripts.prepare_full_data_v2_5 \
  --config configs/benchmark_v2/full_v2_5.yaml \
  --data-root data/benchmark_v2_5 \
  --artifact-root artifacts/benchmark_v2_5 \
  --prepare-only
```

This corrective change does not itself run that command. Full experiment,
five-seed execution, CPU/GPU baselines, learned models, sweeps, and capacity
preflight remain unexecuted.
