# Legacy v1 asset inventory

Inventory captured before the benchmark-v2 source snapshot on 2026-07-28.
Assets remain in their original paths; none were moved, overwritten, or deleted.

| Root | Approximate size | Role |
|---|---:|---|
| `data/` | 72 MiB | legacy processed/raw data and kappa datasets |
| `results/` | 388 KiB | legacy evaluation tables and run outputs |
| `logs/` | 712 KiB | legacy training/evaluation logs |
| `models/` | 148 KiB | source modules and bytecode; no checkpoint found |
| `figs/` | 724 KiB | legacy figures |

Across `data/`, `results/`, `logs/`, and `models/`, 165 files totaling
75,771,367 bytes were observed. The machine-readable manifest records every
excluded runtime artifact with relative path, byte size, modification time, and
SHA-256 digest in `docs/audit/v1_artifact_manifest.json`.

Notable preserved groups:

- `data/kappa_{0.00,0.30,0.50,0.70,1.00}/`
- `data/amlsim/` and `data/sparkov/`
- `results/kappa_*/`
- all pre-existing `results/*.csv` and `logs/*.log`

Git-tracked upstream image files were already absent at preflight:
`images/tabdiff_demo.gif`, `images/tabdiff_demo.mp4`, and
`images/tabdiff_flowchart.jpg`. Their deletion was not caused or staged by this
implementation.
