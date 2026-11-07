# Reproduce this local experiment

Run from the `research-argn-state-first` checkout at
`0d620a97bd91e4cd34a246a6ca8557782c07f470`. Prior model/source files are immutable.
This directory is ignored deliberately: unsuccessful candidates are not pushed.
`MANIFEST.json` records protocol, source, data and prior-manifest hashes.
`AMENDMENT_01.json` preserves a pre-fit test-only float32/float64 endpoint issue,
its original source, and the corrected source hash. No model change or data-driven
adjustment followed that test failure. `EVALUATION_MANIFEST.json` freezes the report
before new development generations. Runtime versions are in `RUNTIME.json`.

Use the sibling research-reporting runtime and environment:

```bash
export PYTHONPATH=.:scripts
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
../research-reporting/argn-state-first-2026-09-27/runtime/bin/python artifacts/argn_clock_adaptive_v1/local_source/run_adaptive.py initialize
../research-reporting/argn-state-first-2026-09-27/runtime/bin/python artifacts/argn_clock_adaptive_v1/local_source/run_adaptive.py prepare
../research-reporting/argn-state-first-2026-09-27/runtime/bin/python artifacts/argn_clock_adaptive_v1/local_source/run_adaptive.py dispatch
../research-reporting/argn-state-first-2026-09-27/runtime/bin/python artifacts/argn_clock_adaptive_v1/local_source/report_adaptive.py
```

These are phase entry points, **not commands to overwrite this completed run**.
Initialization/dispatch reject existing manifests/launches. Reproduction should use
a new isolated artifact root with its own manifest, preserving the original.
GPU dispatch needs host NVIDIA visibility; it accepts only configured idle GPU UUIDs,
checks inactivity twice and checks again before launching. Existing compute jobs
are never killed or preempted. The tiny persistent MPS daemon is excluded from job
occupancy; actual user CUDA compute processes are included.

Fits are CPU optimization of 2/6 output coefficients, not retraining the ARGN backbone.
Every new GPU worker reloads the frozen original checkpoint, verifies native routing,
generates four whole customer populations, and checks exact label/length agreement
with the matched previous draw. Audit files rehash raw generated data and recompute
key W1 metrics independently. `STATUS.json`, `fit.log`, `dispatch.log`, per-worker
logs, generation JSONs, and `report/DISPATCH_COMPLETE.json` distinguish running,
failed, and completed work. Final publication/test evaluation is not part of this run.
