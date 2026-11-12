# Execution notes

Before taking over either existing worker, CPU preflight caught an import-name
collision: the MeLD vendor's `datasets` directory shadowed Hugging Face datasets
used by imports in the existing independent evaluator. Evaluations now run in a
separate CPU interpreter with a clean import path. Original model/vendor/training
files remain unchanged. This was a new pilot wiring issue caught before any
handoff or new training, not a failure of MeLD generation quality.

Preflight verifies exact epoch capture, refusal of an overshot checkpoint,
best-within-budget selection, process identity and targeted suspend/terminate on
an owned dummy process, unchanged optimizer/EMA statements, internal customer
mapping, twelve identity metric checks, and the actual separate evaluator call.
The identity fixture is real data compared to itself, NOT a newly generated cohort.

Original epochs 1–500 continue in the original worker; the controller only copies
the completed atomic checkpoint. Original source and original manifest are read
and reverified, never rewritten. The 600-named directory contains only a random
latent timing probe (no optimizer update); the user changed the boundary to 500
before a 600-epoch controller was launched.

To resume the longer original VAE attempt later, its original fit directories
retain `vae_last.pt` at epoch500, best checkpoint and curve. The snapshot directory
holds independent verified copies. Restarting requires an explicit new execution
record, not relabelling the pilot's 2000-update generator as the original full run.
The original final evaluator is deliberately deferred; completed CPAR/ARGN reports
remain available. No checkpoint or generated-output quality is claimed in advance.
