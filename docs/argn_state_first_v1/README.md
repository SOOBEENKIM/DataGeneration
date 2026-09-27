# B versus B+S: first state-feature study

See [the frozen protocol](PROTOCOL.md) and [configuration](../../configs/argn_state_first_v1.json).
This folder contains the new study, independent of completed `sparkov_argn_control_v2`.

The required frozen data/codec workspace lives in the sibling audit worktree's
`artifacts/sparkov_argn_control_v2/codec_digit_both`. Train data, checkpoints and
generated transaction rows are ignored by git. No final test events are read.

Runtime: `../research-reporting/argn-state-first-2026-09-27/runtime/bin/python`.
It uses a separate CUDA torch 2.5.1 environment and reads the pinned engine
1.0.4 dependencies from the completed reproduction environment. Both environment
paths are intentionally local, not downloaded project data. Install engine 1.0.4,
torch 2.5.1 with CUDA 12.1, pandas 2.2, numpy 1.26 and pytest 8.3.5 in an equivalent
environment to reproduce elsewhere; use the recorded frozen data and hashes.

From the study worktree, with that interpreter:

```bash
python -m pytest tests/test_argn_state_first.py -q
python scripts/run_argn_state_first.py prepare
CUDA_VISIBLE_DEVICES=1 python scripts/smoke_argn_state_first.py
python scripts/launch_argn_state_first.py --gpus 1 2
```

Preparation, smoke, and launch refuse overwrite. The launcher refuses occupied
GPUs and starts one queue per arm. Each queue trains both fresh seeds first,
then generates both sampling seeds for both checkpoints. Each fit has its own
`START.json`, resolved training source, native progress log, checkpoint and
`FIT.json`; each generated dataset has hashes and generation metadata. Exceptions
write a failed queue status and retain artifacts. No failed run is auto-resumed or
silently replaced.

Progress lives under `artifacts/argn_state_first_v1/`:

- `LAUNCH.json`: PIDs, GPUs, commands and original preflight state.
- `logs/B.log`, `logs/B_S.log`: live training/generation output.
- `queue_B.json`, `queue_B_S.json`: current seed, stage, completion/failure.
- `prepared/MANIFEST.json`: source/data hashes and exact split sizes.
- `smoke/RESULT.json`: disposable integration-check outcomes, not study scores.
- `runs/{arm}_{seed}/`: the real experiments.

The smoke intentionally forces eight generated rows per customer only to exercise
native save/load/generation cheaply with untrained heads. Its checkpoints never
initialize the real fits. The study runner does not modify generated lengths.
