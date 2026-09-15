# v2.6 learned numeric-path static audit

## Scope

This audit records the actual v2.5 train and sample paths that motivate the
finite v2.6 development candidates. It is a source audit, not a new
performance analysis. The completed v2.5 test artifacts are used only as the
already-frozen forensic observation that learned outputs failed row
marginals. They are not inputs to v2.6 candidate selection.

Audited source state:

- v2.5 forensic commit:
  `5502f84974133c035eb16ea72487e4a1dddd206b`;
- frozen v2.5 config SHA-256:
  `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3`;
- CTGAN package: `ctgan==0.12.1`.

No v2.5 model, generator, evaluator, threshold, test split, runtime artifact,
endpoint, or C2 decision was changed.

## Executive findings

The four learned paths do not share one numeric transform or decoder:

- CTGAN and TVAE pass amount through a train-fitted Bayesian-GMM transformer
  and its inverse transform.
- Neural sequence and CoF train directly on the raw stored amount without an
  explicit standardization or inverse transform.
- CTGAN/TVAE generate i.i.d. rows from separate class models. Neural and CoF
  use one label-conditioned sequence model.
- CTGAN uses Gumbel-softmax inside its transformed generator; TVAE decodes
  transformer spans after a latent VAE sample; neural samples discrete heads
  multinomially; CoF uses multinomial self-feedback but final argmax decoding.
- Neural and CoF use an unweighted sum of amount MSE, gap CE, and receiver CE.
  TVAE uses its learned-sigma reconstruction likelihood and one global loss
  factor. CTGAN has no explicit per-channel reconstruction weights.

This source evidence refutes a single common normalization or decoding
implementation as the explanation for all learned failures. It supports
testing a small, preregistered set of representation, loss-weight, sampling,
and checkpoint alternatives on train/validation only.

## Cross-model path table

| Property | CTGAN | TVAE | Neural sequence | CoF-SeqGen |
|---|---|---|---|---|
| Input rows | Valid rows flattened into `amount_log`, `dt_bin`, `receiver` | Same inherited adapter path | Entity minibatches with full padded sequence and valid mask | Entity minibatches with full padded sequence and valid mask |
| Class handling | Separate model for `y=0` and `y=1` | Separate model for `y=0` and `y=1` | One model; entity label embedding broadcast over positions | One model; entity label used for CFG, with null-label dropout |
| Amount representation | Raw stored amount enters a train-fitted Bayesian-GMM normalizer | Same | Raw amount directly enters GRU and amount head | Raw amount receives cosine Gaussian diffusion and a linear projection |
| Amount normalization | GMM normalized scalar plus component one-hot | Same | None | None |
| Amount inverse | Component argmax and GMM reverse transform | GMM reverse transform with learned-sigma perturbation | Identity | Identity |
| Gap representation | Discrete column, one-hot transformed | Same | Train-supported integer embedding | Train-supported integer embedding plus absorbing MASK |
| Receiver representation | Discrete column, one-hot transformed | Same | Train-supported integer embedding | Train-supported integer embedding plus absorbing MASK |
| Main loss | Joint WGAN-GP adversarial loss plus conditional discrete CE | Per-span reconstruction likelihood plus KL | amount MSE + gap CE + receiver CE | amount MSE + gap CE + receiver CE + label BCE |
| Explicit channel weights | None | No per-channel weights; global reconstruction factor 2 | 1/1/1 | 1/1/1, label 1; coherence weight is 0 in v2.5 |
| Discrete training activation | Gumbel-softmax, temperature 0.2 | Softmax spans reconstructed with CE | Direct CE logits | Direct CE logits under time-dependent masking |
| Discrete sampling | Transformed generator activation then upstream inverse decode | Decoder tanh then upstream inverse decode | Softmax multinomial at temperature 1 | Feedback multinomial at default temperature 1; final decode is argmax |
| Amount sampling | Latent normal through generator and GMM inverse | Latent normal through decoder; learned sigma used in inverse | Deterministic autoregressive conditional mean | Deterministic DDIM trajectory from Gaussian noise |
| Sampling schedule | i.i.d. row sampling by class and SamplingPlan position | Same | Left-to-right autoregressive | 50-step cosine DDIM; CFG scale 2; feedback after 0.3 |
| Length/label output | Fixed SamplingPlan | Fixed SamplingPlan | Fixed SamplingPlan | Fixed SamplingPlan |

## CTGAN path

### Train

`ConditionalCTGAN.fit`:

1. Uses valid rows only.
2. Forms `rows_y = repeat(y_entity, lengths)`.
3. Creates a frame with raw stored `x_num[...,0]` under the misleading
   historical name `amount_log`, plus integer `dt_bin` and `receiver`.
4. Splits the frame by label and fits two independent generators.
5. Gives each class half the baseline wall budget and half the requested
   updates.

The upstream `DataTransformer` treats amount as continuous:

- a `ClusterBasedNormalizer` fits a Bayesian GMM on that class's train rows;
- output is one normalized scalar with `tanh` activation plus a component
  one-hot span;
- gap and receiver are one-hot encoded discrete spans.

The CTGAN generator and discriminator see the concatenated transformed
vector. There is no explicit amount/gap/receiver loss weighting. The
generator loss is adversarial plus conditional CE for the selected discrete
condition. Softmax spans use Gumbel-softmax with fixed temperature `0.2`.

### Sample

For each SamplingPlan label:

1. Draw standard-normal latent noise and the upstream original-frequency
   conditional vector.
2. Generate a transformed row.
3. Apply `tanh` to continuous spans and Gumbel-softmax at temperature `0.2`
   to softmax spans.
4. Inverse-transform amount by component argmax and GMM reverse transform.
5. Reverse one-hot gap and receiver to integer categories.
6. Place i.i.d. rows into all valid positions belonging to the corresponding
   class.

There is no sequence-state feedback, post-sampling correction, or external
amount inverse transform in the adapter.

## TVAE path

TVAE inherits the same valid-row flattening, class split, SamplingPlan
assembly, and transformer definitions as CTGAN.

### Train

The transformed vector is encoded into `mu`, `std`, and `logvar`. Its decoder
produces transformed spans and learned per-output `sigma`.

- Continuous normalized spans use squared reconstruction error divided by
  learned `sigma^2`, plus `log(sigma)`.
- Every softmax span, including the amount mixture component, gap, and
  receiver, uses cross-entropy.
- The summed reconstruction term receives global `loss_factor=2`.
- KL divergence is added.
- No amount/gap/receiver-specific weights exist.

The v2.5 checkpointable loop clamps decoder sigma to `[0.01, 1.0]`.

### Sample

TVAE draws a standard-normal latent vector, applies the decoder and `tanh`,
then invokes the transformer inverse:

- amount component is selected by argmax;
- normalized amount is perturbed using the learned sigma before GMM reverse
  transform;
- gap and receiver use the transformer's discrete reverse path.

Two separate class models generate i.i.d. valid rows.

## Neural sequence path

### Train

The GRU input concatenates:

- the previous raw amount;
- the previous gap embedding;
- the previous receiver embedding;
- the entity label embedding.

Training uses teacher-forced one-position shifts and valid positions only.
No amount normalization or inverse transform exists. The exact v2.5 loss is:

```text
MSE(amount prediction, raw amount)
+ CE(gap logits, gap category)
+ CE(receiver logits, receiver category)
```

All three coefficients are 1. Entity minibatches are sampled uniformly
without replacement on each update. One label-conditioned model is fitted.

### Sample

The model runs left-to-right:

- amount is the deterministic output of the linear amount head;
- gap and receiver are independent multinomial draws from temperature-1
  softmax heads at each step;
- generated outputs are fed into the next step;
- plan labels and lengths are fixed, and padding is reset to zero.

The deterministic amount head supplies no residual-variance sampling path,
which is a static mechanism capable of reducing distributional spread even
when its conditional mean is accurate. This is a hypothesis for
train/validation selection, not a reinterpretation of v2.5.

## CoF-SeqGen path

### Train

CoF uses raw amount directly. A cosine schedule produces
`alpha*x_num + sigma*noise`; there is no train standardization or inverse
transform. Gap and receiver are embedded integers with an extra MASK token.
They are corrupted with probability `min(t_frac, 0.7)`.

The denoiser adds numeric projection, gap embedding, receiver embedding,
position embedding, time embedding, and entity-label/CFG embedding before a
Transformer. The actual executable v2.5 loss is:

```text
MSE(raw amount)
+ CE(gap)
+ CE(receiver)
+ 1.0 * BCE(label)
+ coherence loss
```

The v2.5 config fixes `coherence_lambda=0.0`, so coherence contributes zero.
The source module's introductory comment still says label weight `0.1`, but
the executable expression uses `1.0`; this audit records the executable
value. Amount, gap, and receiver all have coefficient 1.

### Sample

- Start amount from standard Gaussian noise.
- Start gap and receiver from MASK because `start_from_mask=true`.
- Run 50 cosine DDIM steps with entity label, CFG scale 2, and discrete
  feedback beginning after fraction 0.3.
- The adapter does not pass `discrete_temp`, so feedback uses its default
  value 1.
- The final gap and receiver outputs use argmax regardless of feedback
  temperature.
- No numeric inverse transform is applied.

The generated label head is not used to replace the SamplingPlan entity
label.

## Source evidence

| Source | SHA-256 |
|---|---|
| `generators/conditional_ctgan.py` | `0a1606281e8a0fce436aef138eec7e68c87e3dd15e6d008e8906fe4f02608910` |
| `generators/conditional_tvae.py` | `e2a07d10155087f7ddf0b7e5c54ddd78f98cf6c60e6fd3d1a6a3484321019df3` |
| `generators/checkpointable_tabular_v2_5.py` | `9ae957925ff847534622a49197342a6cc6cd30e976e954ffe47da5fc62221a15` |
| `generators/joint_sequence_baseline.py` | `2982951cdedcc2311933d914f265e9985dc0775e6318c715de40ff2646a4a229` |
| `generators/cof_seqgen_adapter.py` | `48e9e1280bc23a27abf0b5f6447f6682e3b846d0d1773fb96d99abcf407dd7f3` |
| `models/cof_seqgen.py` | `caf5fdb3baf367a28c6081a8ba5f4a89e59dacbba2377636acba5c051ae7006e` |
| `models/seq_denoiser.py` | `bd9a057a880bcfe8918b0ccbcc7bef5ffd9e5d0181a4cac0a17b942ac6cf9dc4` |
| `models/sampler.py` | `87e0baba7e5447ed26081beb06d67666c15f0a88840997c2de0da7ad211c38e5` |
| installed `ctgan/data_transformer.py` | `9e0aa6f35efa6cf19741396b0b14868afcaa39bd77127f107ebbe4c819b54386` |
| installed `ctgan/synthesizers/ctgan.py` | `cf8a01b06036d82ba6b62bd21c834d358a55d43792bf15eb4f3834811726329b` |
| installed `ctgan/synthesizers/tvae.py` | `7c464ddba7e2f03223ceeebc2e7ed546aacb7145f48a90937941b80ad74a1837` |

Relevant line locations at the audited state:

- CTGAN/TVAE frame, class split, and assembly:
  `conditional_ctgan.py:137–263`;
- CTGAN transformed activation and sample:
  installed `ctgan.py:193–260, 490–545`;
- DataTransformer continuous/discrete transform and inverse:
  installed `data_transformer.py:36–85, 119–238`;
- TVAE loss and sample:
  installed `tvae.py:79–103, 148–189, 211–236`;
- neural loss and sample:
  `joint_sequence_baseline.py:39–150, 353–411`;
- CoF loss:
  `models/cof_seqgen.py:95–236`;
- CoF adapter and sampling:
  `cof_seqgen_adapter.py:77–258, 287–350`;
- DDIM discrete feedback/final decode:
  `models/sampler.py:88–231`.

## Implication for v2.6

Only four dimensions are eligible for v2.6 development:

1. reversible amount representation fitted on train only;
2. explicitly preregistered channel-loss weights;
3. categorical/numeric sampling rule;
4. requested update trajectory and fixed checkpoint used for validation.

Architecture width/depth, optimizer search, threshold changes, test-derived
calibration, DGP changes, endpoint changes, and C2-rule changes are outside
the candidate space.

## Result-blind implementation amendment

The v2.6-only implementation preserves every audited v2.5 source file and
adds the following isolated paths:

| Concern | v2.6 implementation |
|---|---|
| Train-only representation | `TrainOnlyZScore` fits finite valid train rows, zeros padding after transform, checkpoints float64 mean/std, and inverses valid generated rows before validation |
| CTGAN | new checkpointable subclass applies the candidate Gumbel temperature; the separate-class adapter still receives exact half update/wall budgets |
| TVAE | new checkpointable subclass weights raw-column transformed spans as preregistered and applies latent scale/categorical temperature during sampling |
| Neural | new subclass weights amount/gap/receiver losses and divides gap/receiver logits by the preregistered temperature |
| CoF | new weighted loss component and sampler apply amount/gap/receiver/label weights, feedback temperature, and c03 final categorical temperature |
| SamplingPlan | one plan is fit from train policy and hash-bound before any candidate validation sample |
| Runner | runner-owned bounded child, append-only trajectory/candidate attempts, immutable provenance, and fail-closed path/wall/failure checks |

The c00/c01 native configurations are not duplicated training jobs. Each
model trains one 20,000-update native trajectory and exposes its 10,000 and
20,000 checkpoints as the two existing evaluation candidate IDs. CTGAN and
TVAE checkpoint bundles are balanced across the two class models.

Static import/instantiation and fixture tests do not call an actual model
`fit` or `sample`, query CUDA, invoke a DGP, or open the test split. Candidate
execution and validation selection remain separately unauthorized.
