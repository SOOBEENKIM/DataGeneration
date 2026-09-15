# v2.6 validation-only single-factor candidate amendment

## Status and scope

This amendment is based on forensic commit
`fc503ac9159668cde342f55aa76b40d61d3e0dc0`. It freezes a finite
validation-only candidate family before any new result is produced.

It does not authorize candidate training, candidate sampling, validation
selection, GPU access, fresh-test definition or evaluation, data generation,
or a five-seed/full run. Those actions require a later append-only
authorization matching the eventual implementation source and this config.

The following remain unchanged:

- scenario `joint_semimarkov_v2b`, kappa 1.0 and DGP;
- frozen train and validation splits;
- train-fitted SamplingPlan
  `862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27`;
- continuous association-recovery endpoint;
- all five row-marginal thresholds;
- all-five-guard eligibility and deterministic tie-break rule;
- primary three-model C2 rule and no-pass stop rule.

The original v2.5/v2.6 runtime, config, data, authorization, checkpoints,
candidate samples and selection reports remain read-only. This amendment
writes only to a new future root:
`artifacts/benchmark_v2_6/selection_single_factor/`.

The executable definition is
`configs/benchmark_v2/selection_v2_6_single_factor_amendment.yaml`, SHA-256
`707ecdbd37bc55f4a3b9ece603b70d26c8274015aa9cb97d5d376aab3d9ecb7c`.

## Frozen validation contract

| Item | Frozen value |
|---|---|
| selection seed | 2601 |
| train file | `c67a6fce4593317e11e1b6f36397bfb38f9607916bf8f5b32329d8fa99b700a8` |
| validation file | `68c15c05c55742bfce0f22560ffe8378d032a7a9fd9a1072e80ec8da5340fba5` |
| development manifest | `31ddf45dcf7b4f982ed990ec94281c3e6bad0454ec3182e2434e42358006ade5` |
| amount KS limit | 0.006081138155655141 |
| gap KS limit | 0.006387882975686154 |
| amount-effect limit | 0.0363693454591819 |
| gap-effect limit | 0.051540527275560376 |
| receiver-effect limit | 0.02 |

The selection rule remains:

1. eligible only when all five guards pass;
2. minimize `max(amount_ks, gap_ks)`;
3. then minimize `amount_ks + gap_ks`;
4. then lexicographic candidate ID.

No passing candidate for any primary model prohibits a v2.6 primary full
run. Diagnostics below cannot override a hard guard or select a candidate.

## Single-factor definition

Every model has one immutable control, one evaluation-only intervention
using the immutable control checkpoint, and one newly trained intervention.
Non-control candidates differ from their model control in exactly one
preregistered conceptual dimension. All other effective dimensions must be
byte-equivalent under canonical config comparison.

### CTGAN

Separate `y=0` and `y=1` generators remain mandatory.

| Candidate | Execution | Sole intervention |
|---|---|---|
| `ctgan_sf_c00_frozen_control` | reuse existing c01 result | none |
| `ctgan_sf_c01_shared_transformer` | new 20k trajectory | transformer scope: per-class to one transformer fitted once on all valid train rows |
| `ctgan_sf_c02_temperature_only` | re-evaluate existing c01 checkpoint | categorical temperature: 0.2 to 0.5 |

The shared-transformer candidate still trains two separate class generators.
It does not change amount representation, categorical temperature, update
count or class allocation. The temperature-only candidate keeps the original
per-class transformers and does not retrain.

Required diagnostics are per-class amount mean/std, signed standardized
amount and gap effects, gap frequencies, entity-balanced receiver
frequencies and category-wise signed receiver frequency.

### TVAE

The categorical temperature-multinomial path that restored the receiver
guard in the prior bundled candidate is isolated from its prior z-score,
latent-scale and weighting changes.

| Candidate | Execution | Sole intervention |
|---|---|---|
| `tvae_sf_c00_frozen_control` | reuse existing c01 result | none |
| `tvae_sf_c01_categorical_decode_only` | re-evaluate existing c01 checkpoint | gap/receiver categorical decode: upstream argmax to temperature-multinomial 0.75 |
| `tvae_sf_c02_channel_weight_only` | new 20k trajectory | amount/gap/receiver weights: 1/1/1 to 2/2/1 |

Both interventions retain native amount representation, latent scale 1,
per-class transformers, separate class generators and 20,000 total updates.
There is no update-count candidate.

Required diagnostics are raw gap-bin and receiver-category frequencies,
maximum category frequency, HHI concentration, and class-specific gap and
receiver frequencies.

### CoF-SeqGen

Architecture, native raw amount representation, and the complete discrete
gap/receiver path remain fixed.

| Candidate | Execution | Sole intervention |
|---|---|---|
| `cof_sf_c00_frozen_control` | reuse existing c01 result | none |
| `cof_sf_c01_noise_prediction` | new 20k trajectory | continuous diffusion parameterization: clean-x0 MSE/matching DDIM to epsilon MSE/matching reverse diffusion |
| `cof_sf_c02_variance_preserving_residual` | re-evaluate existing c01 checkpoint | amount residual sampling: none to train-fitted zero-mean variance-deficit residual |

The noise objective and its mathematically matching reverse equation are one
indivisible parameterization dimension. It cannot be combined with residual
sampling.

The residual candidate adds independent zero-mean Gaussian residuals only to
amount. Its variance is fixed before validation as
`max(valid-train amount variance - base train-plan sample variance, 0)`.
The fit uses train only; validation refitting is forbidden. It changes no
discrete value or architecture. Z-score-only is not a candidate.

Required diagnostics are generated amount mean, standard deviation and
1/5/25/50/75/95/99 percentiles.

## Frozen controls

The controls refer to, but never modify or copy over, these completed c01
20k artifacts:

| Model | Checkpoint SHA-256 | Validation sample SHA-256 |
|---|---|---|
| CTGAN | `7e9de7e609ad48e8de687426240fe4fd0a47b5d2d0abcbc70a4edb58f182d10d` | `59ea09140e8b85e0e4b4eed64f7aba31f52c58ea6b0890e3a680d47c310a161b` |
| TVAE | `6f5263a5a40d9513342b33a823a02769bab65f6e1a5bd38c814b87d771a2947c` | `33812d9dec421b4076f41a7ba8774e9dcf44e070216d3c608cb6bead0d984181` |
| CoF | `7ce44721cadffef21ade3bf1884421c304f73ea9eff1e9bfb3ca2d9e63cc0c29` | `3d0025f7a795aea496ca5523bb2b71dca0986b99252c473ced7bd792881c43d1` |

Every later authorization must bind the control result, checkpoint and
sample hashes independently.

## Budget

| Work | Count | Per-item hard cap | Maximum GPU-hours |
|---|---:|---:|---:|
| new training trajectory | 3 | 7,200 seconds | 6.0 |
| evaluation-only candidate | 3 | 1,800 seconds | 1.5 |
| immutable reused control | 3 | zero | zero |
| total | 9 evaluation candidates | — | 7.5 |

CTGAN and TVAE still divide a 20,000-update trajectory and 7,200-second cap
equally across the two class generators. No early stopping, update increase,
result-based extension or replacement candidate is permitted.

With three idle GPUs, the hard-cap wall-time ceiling is approximately 2.5
hours when the three new trajectories run in parallel followed by the three
evaluation-only candidates in parallel. Sequential execution has a 7.5-hour
GPU wall-time ceiling. These are ceilings, not performance observations.

## Implementation boundary

This preparation commit implements:

- strict config/provenance and exactly-one-intervention validation;
- immutable frozen-control hash validation;
- model-specific diagnostic computation from a stored validation sample;
- train/validation-only path validation and test fail-closed behavior;
- an exclusive-create append-only attempt store.

It does not invoke or authorize candidate backend training/sampling. Backend
execution wiring and a matching authorization must be reviewed separately
before any GPU action.
