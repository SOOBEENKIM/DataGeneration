# Benchmark v2.5 baseline definitions

Version: `benchmark-v2.5`.

These definitions are fixed before the full experiment and amended by
`preregistered_full_experiment_v2_5_amendment_1.md`. Every generator uses
train only for fitting, one shared SamplingPlan across every model seed, the
same entity-label prevalence rule, the same entity count, and the same train-derived
label-conditional length distribution. The generated mask must be
prefix-contiguous and all padded numerical, gap, and categorical values must
be zero.

## Shared inputs and outputs

Input is a `SequenceBatch` with one numerical amount channel, one discretized
gap channel, one receiver-category channel, entity label, length, and valid
mask. Output is a `SyntheticBatch` with the exact labels, lengths, and masks in
the fixed `SamplingPlan`.

Discrete generated values must remain within train support and all numerical
values must be finite. Failure is `INVALID`, not a dropped observation. Every
adapter implements `fit(train, config, seed)` and `sample(plan, seed)`.

## Empirical structural references

### Label-conditional empirical i.i.d.

Pool all valid train rows separately for y=0 and y=1. For every valid position,
draw one complete `(amount, dt_bin, receiver)` row with replacement from the
matching class pool. Drawing a complete row retains contemporaneous row
dependence but destroys temporal dependence.

### Contiguous block 2, 4, and 8

For each target sequence, draw same-label train entities and valid contiguous
blocks with replacement until the fixed target length is filled. A block never
crosses a source entity or padding boundary. Final blocks may be truncated.
Lengths 2, 4, and 8 are separate reported cells.

### Full-sequence block reference

Draw a same-label train entity and copy one contiguous block as long as the
available source sequence, repeating with another same-label source if the
target is longer. This is an empirical sequence/oracle reference. It benefits
directly from copying observed full sequences and is **not** interpreted as a
fair learned-model competitor. Its comparison to CoF is descriptive and is
never included in the Holm family.

## Parametric sequence baselines

### Class-conditional independent Markov

For each class independently:

- estimate Laplace-smoothed initial and transition probabilities for
  `dt_bin`;
- estimate a separate Laplace-smoothed initial and transition model for raw
  receiver category;
- retain a class-conditional empirical amount pool.

At generation, gap and receiver chains are sampled independently conditional
on class. Amounts are i.i.d. draws from the matching class pool. This baseline
models within-channel persistence and intentionally omits cross-channel
alignment.

### Class-conditional joint observed Markov

The observed state is exactly:

```text
(dt_bin_t, 1[receiver_t == receiver_(t-1)]).
```

The first repeat bit is defined as zero for fitting. For each class, fit a
Laplace-smoothed Markov chain over the finite `2 * n_gap_bins` state space.
This avoids an arbitrary full gap-by-receiver-category Cartesian state. An
unseen state uses its Laplace-smoothed transition row.

Generation decodes the gap bin and repeat decision jointly. A repeat copies
the previous receiver. A non-repeat draws from the train-only
class-conditional receiver marginal after removing the previous category when
support allows. Amounts use the class empirical pool. This baseline can model
one-step cross-channel alignment but not an explicit non-geometric latent
duration.

### `plug_in_hmm`

This reproducible comparator is a two-state, class-conditional plug-in state
model. The state boundary is inferred deterministically using train only:

```text
short state = dt_bin <= train median dt_bin
```

The non-interpolated empirical median is used. For each class, the inferred
train paths estimate:

- initial and state-transition probabilities;
- gap-bin emission probability by state;
- receiver-repeat emission probability by state;
- state-conditional empirical amount pools;
- class-conditional receiver-category marginal.

No EM, latent-state restart, or test-data fit is used. This comparator is not
claimed to be a standard latent HMM. All categorical probabilities use
Laplace α=1 smoothing. Generation samples
the latent Markov path, then its emissions. Receiver categories are assembled
from repeat decisions with the same non-repeat rule as joint Markov. This is
called a plug-in HMM, not a fully latent maximum-likelihood oracle; that
limitation is reported rather than tuned after results.

### `plug_in_hsmm`

The plug-in duration comparator uses the train-only state inference and
emissions above.
It additionally fits a class/state-specific smoothed empirical discrete
duration distribution over 1–128 and a transition chain between inferred
segments. Generation samples an initial segment state, explicit duration,
emissions for the segment, and the next segment state until the fixed length
is reached.

It uses no EM, latent-state restart, or test-data fit and is not claimed to be
a standard latent HSMM. It directly represents non-geometric duration and may
be especially strong in v2b.

## Separate-class row generators

### CTGAN

This is a **separate-class-model conditional baseline**, not a single
conditional CTGAN:

- fit one CTGAN on valid y=0 rows;
- fit another CTGAN on valid y=1 rows;
- treat `dt_bin` and receiver as discrete columns and amount as continuous;
- use package `ctgan==0.12.1`;
- allocate 5,000 requested generator updates and at most 1 GPU-hour to each
  class model.

The two class models jointly constitute one baseline and have a 2 GPU-hour
maximum. A class model cannot receive the whole baseline budget. For
generation, fix label and length first, sample i.i.d. rows from the matching
model, and place them into the fixed sequence mask. CTGAN therefore models
row-level cross-channel distribution but no temporal dependence.

### TVAE

TVAE has the same separate-class construction and sequence assembly:

- one y=0 TVAE and one y=1 TVAE;
- 10,000 requested minibatch updates per class;
- at most 1 GPU-hour per class;
- 2 GPU-hours total for the baseline;
- package `ctgan==0.12.1`.

It is never called a single conditional model. It is a learned row baseline,
not a sequence model.

## Neural sequence baseline

The neural baseline is a class-conditional autoregressive GRU. Previous
amount, gap embedding, receiver embedding, and label embedding feed the GRU.
Its heads predict the next amount, gap bin, and receiver category. Training
uses the canonical valid mask and teacher forcing on train only.

Frozen settings are hidden width 64, batch size 256 entities, Adam at 0.001,
20,000 requested updates, and a 2 GPU-hour cap. At generation, the fixed label
conditions the autoregressive rollout and all padded positions are restored to
zero.

## CoF-SeqGen

The proposed adapter uses the existing mask-aware CoF-SeqGen architecture and
v2 discrete feedback sampler. Frozen settings include:

- `d_model=128`, two denoiser layers;
- entity batch size 256;
- Adam at 0.001;
- 20,000 requested updates and a 2 GPU-hour cap;
- `coherence_lambda=0.0`, `cfg_dropout=0.15`;
- guidance scale 2.0 and 50 DDIM steps;
- sampling chunk size 256 and discrete feedback after 0.3;
- all-valid loss positions and canonical padding mask.

Train-fitted `tau` and the unchanged window width/temperature are passed to
the model. No model result can change these settings.

“v2.5는 legacy hyperparameter 전체를 복원한 것이 아니다. d_model,
n_layers, diffusion_steps 및 feedback sampler 관련 값만 engineering
reference로 반영했고, batch size 256과 learning rate 0.001은 결과 관찰 전에
고정한 v2.5 고유의 사전등록 선택이다.”

## Failure and reporting

Every adapter is a mandatory registry entry. Import failure, unsupported
runtime, missing checkpoint support, contract failure, OOM, or exception must
produce an immutable `UNAVAILABLE.json` or `FAILED.json`, after which the full
experiment stops. Silent baseline removal, NaN filtering, extra training for
one model, or renaming separate-class CTGAN/TVAE as a single conditional model
is prohibited.
