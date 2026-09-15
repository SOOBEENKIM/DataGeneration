# Benchmark v2 runbook

Use `<COFSEQ_PYTHON>` as `python`.

```bash
python -m scripts.generate_benchmark_v2 --config configs/benchmark_v2/main.yaml
python -m scripts.validate_benchmark_v2 --config configs/benchmark_v2/main.yaml --data-root data/benchmark_v2 --output-root artifacts/benchmark_v2/gates
python -m scripts.run_benchmark_v2 --config configs/benchmark_v2/baselines.yaml --data-root data/benchmark_v2 --artifact-root artifacts/benchmark_v2/runs --scenarios markov_persistence_v2a joint_semimarkov_v2b --kappas 0 0.3 0.5 0.7 1 --generators conditional_ctgan conditional_tvae markov_independent markov_joint --seeds 1 2 3 4 5 --device cuda
python -m scripts.run_benchmark_v2 --config configs/benchmark_v2/cof.yaml --data-root data/benchmark_v2 --artifact-root artifacts/benchmark_v2/runs --scenarios markov_persistence_v2a joint_semimarkov_v2b --kappas 0 0.3 0.5 0.7 1 --generators cof --seeds 1 2 3 4 5 --device cuda
```

The last two commands are FULL_EXPERIMENT commands and must not be run in
IMPLEMENT_AND_SMOKE mode. Long commands must run under tmux/screen/scheduler.

The learned smoke commands are also intentionally unexecuted while any
benchmark gate is FAIL. The intended smoke scope is one kappa, seed 1,
`n_train_max=512`, 100 training steps, and at most five diffusion steps.
At the CPU forced-stop checkpoint, `scripts.run_benchmark_v2` is not yet
present, so the examples above are specification commands rather than runnable
commands. This missing runner is reported explicitly and is not bypassed with a
legacy training script.
