# Frozen internal-validation measurement

Registered after the cap-control fits and before this probe. The relaxed runs
stopped at epochs 22/30 and selected 17/25. Native validation crops random windows
and uses random within-row permutations for flexible ARGN. Measure uncertainty
of that validation signal without changing or refitting any checkpoint.

For each of four checkpoints, evaluate all 69 original internal-check customers
with native batch size 4, window 100 and native sample-loss normalization. Twenty
seeds 20261100..20261119 are shared across checkpoints. Four modes: random crop
and permutation; fixed crop and random permutation; random crop and fixed native
generation order; fixed crop and fixed order. Fixed crop uses NumPy seed
20261100. Preserve weights. The fully fixed mode must be repeatable.

Report distribution of losses, plus differences between paired checkpoints under
the same evaluation seeds. Comparing average losses across different order modes
changes the conditioning task, so do not interpret such changes as improvements.
A noisy stopping signal alone does not prove the selected epoch was wrong, nor
that longer training would improve generation. No new stopping rule is proposed
or trained in this study.
