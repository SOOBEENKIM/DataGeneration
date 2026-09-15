# Benchmark v2 artifact schema

Runtime outputs are ignored by Git. Gate outputs live under `gates/`, smoke
outputs under `smoke/`, sampling plans under `sampling_plans/`, and generator
runs under `runs/<scenario>/kappa_<value>/<generator>/seed_<seed>/`.

Each complete learned run must contain its resolved config, provenance manifest,
sampling plan, saved synthetic sample, metrics for 2/4/8 bins, sequence metrics,
fidelity report, log, and (where applicable) checkpoint.
