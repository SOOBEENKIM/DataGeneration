# Benchmark v2.5 preregistration amendment 1

Status: **frozen before capacity measurement; full execution is not
authorized**.

This amendment supplements preparation commit
`4dfe83cd43d50b9304054e11090aa18f761d51c0`. It changes no v2 DGP
parameter, kappa, data split, AUROC operator, gate rule, primary endpoint, or
two-hour per-generator budget. The v2.4 gate-pass tag and all v2.4 runtime
artifacts remain read-only. This amendment authorizes only the timing-only CoF
capacity preflight defined below after the source/config/docs/tests amendment
commit exists. It does not authorize a full model run, five-seed run, sweep,
performance comparison, or sample-quality evaluation.

## C2 primary decision

C2 is evaluated only for `joint_semimarkov_v2b`, kappa=1.0, seeds 1–5, and the
fixed training budgets. Its sole primary endpoint is continuous
association-recovery error:

```text
abs(delta_joint(real test) - delta_joint(synthetic))
```

Lower is better. The 4-bin and 8-bin outputs are
`support_diagnostic_bins=[4,8]`; they cannot enter or change the primary
decision.

The C2 primary multiplicity family contains exactly:

1. `ctgan_separate_class`;
2. `tvae_separate_class`;
3. `empirical_iid`.

For every comparison, the effect is baseline mean error minus CoF mean error.
C2 is supported only if all three effects are strictly positive, all three
Hedges g values are strictly greater than 0.8, and all three two-sided Welch
p-values are below 0.05 after Holm correction within this three-comparison
family. An unpaired bootstrap 95% interval is reported for every effect but is
not an additional decision threshold. If any condition fails, the required
decision text is:
**“C2 was not supported in this preregistered experiment.”**

The secondary Holm family is separate and contains exactly
`neural_sequence`, `independent_markov`, `joint_markov`, `plug_in_hmm`, and
`plug_in_hsmm`. It cannot change the C2 decision.

For a nonsignificant plug-in state comparison, the only permitted
interpretation is “planned sample did not detect a difference.” No
corresponding model-performance identity claim is permitted.

`floor_epsilon=1e-6` is fixed before results. Floor proximity may be described
only when C0 mean error is greater than that epsilon and CoF mean error is no
more than twice C0 mean error. When C0 mean error is at most the epsilon, the
floor-proximity ratio is recorded as uninterpretable.

## One shared SamplingPlan

The train-only label prevalence and train-only label-conditional length
distribution are sampled once with fixed plan seed 10,001 to produce 7,989
label/length/mask rows. Its full SHA-256 is the shared SamplingPlan identity.
Every generator and every model seed 1–5 must use that one plan. Model seeds
may alter initialization, minibatch order, training stochasticity, and
synthetic sampling only; they cannot alter benchmark data, splits, labels,
lengths, or the plan hash.

Every attempt manifest stores the shared full SHA-256. Attempt allocation,
resume, and aggregation fail closed if any stored plan hash is missing or
different. Test labels and test lengths never construct this plan.

## Plug-in state baseline names and budgets

The two state comparators are named `plug_in_hmm` and `plug_in_hsmm`
throughout adapters, config, manifests, tables, and reports. Their state paths
are inferred deterministically from the train split with the frozen short-gap
rule. They use no EM, latent-state restart, or test-data fit. They are not
claimed to be standard latent HMM/HSMM implementations.

`independent_markov`, `joint_markov`, `plug_in_hmm`, and `plug_in_hsmm` each
have a hard `max_wall_seconds=7200` fit cap.

## Read-only legacy engineering audit

The read-only provenance commands resolve the annotated tag as follows:

```text
git rev-parse legacy-kappa-v1^{commit}
0066ad321182b21c5eb46c71c69679c3a97192b1

git show-ref --tags -d legacy-kappa-v1
b72cd99207cc60e1b555139361d25bf2aed4878d refs/tags/legacy-kappa-v1
0066ad321182b21c5eb46c71c69679c3a97192b1 refs/tags/legacy-kappa-v1^{}
```

`b72cd99207cc60e1b555139361d25bf2aed4878d` is the annotated tag object,
not a commit. Its `object` field names the actual immutable target commit
`0066ad321182b21c5eb46c71c69679c3a97192b1`. The earlier amendment wording
that called `b72cd...` a commit was incorrect; there is no conflict between
two commits and the tag was not moved.

All legacy blobs and SHA-256 values in the table below were recalculated
explicitly from target commit
`0066ad321182b21c5eb46c71c69679c3a97192b1` and match the prior values. The
evidence is an executable configuration path: `scripts/compute_fidelity.py`
imports the frozen CoF constants from
`scripts/baseline_coherence_harness.py`, constructs the model, and invokes
the discrete-feedback sampler.

| Read-only path at `legacy-kappa-v1` | Git blob | SHA-256 |
|---|---|---|
| `scripts/compute_fidelity.py` | `2d64022d043206317f04540d56520da04499a8d1` | `06d492563967af8f91fecb7b44afac15fa14c52d15da80e1adb6e98f51e5bbd6` |
| `scripts/baseline_coherence_harness.py` | `392eafbfc908efcb5612436402ced3bdc1277674` | `a2f518760a6ef1a6e6377a8719c724ec3c9471cc918b81a59cf173e89e6c1727` |
| `models/cof_seqgen.py` | `8f4e009c8b4c6ee157bf3fb41b7bcdcb5a4a5781` | `dfaced17a57fd7ebdbf5962b2159177806eb949cc9c52cf7a90fd76e1ca7581f` |
| `models/sampler.py` | `04e5649ab560e121b920f5d22f17cf22904896df` | `b093150f6a2439745e0355e2bb736156ca57327f7449544449a3c523d3d98324` |
| `models/seq_denoiser.py` | `809ade418309021c43ceac4c18493de573078d6c` | `bd9a057a880bcfe8918b0ccbcc7bef5ffd9e5d0181a4cac0a17b942ac6cf9dc4` |

The path verifies:

| Engineering value | Result | Evidence |
|---|---|---|
| `d_model=128` | VERIFIED | harness `COF_D_MODEL`; fidelity model construction |
| `n_layers=2` | VERIFIED | fidelity `SeqDenoiser` construction |
| DDIM/diffusion steps `50` | VERIFIED | harness `COF_T_DIFF`; fidelity sampler call |
| `guidance_scale=2.0` | VERIFIED | fidelity CLI default and sampler call |
| `cfg_dropout=0.15` | VERIFIED | fidelity `CoFSeqGen` construction |
| `discrete_mask_max=0.7` | VERIFIED | `CoFSeqGen.compute_loss` mask cap |
| `feedback_after=0.3` | VERIFIED | fidelity sampler call and sampler parameter |
| `coherence_lambda=0.0` | VERIFIED | fidelity `CoFSeqGen` construction |

This audit is only an engineering reference showing a discrete-feedback
configuration with sufficient sampler iterations. It is not evidence for any
legacy performance claim.

## Amended CoF full configuration

Only the v2.5 full CoF cell changes to `d_model=128`, `n_layers=2`, and
`diffusion_steps=50`. It retains guidance 2.0, CFG dropout 0.15, discrete mask
maximum 0.7, feedback after 0.3, coherence lambda 0, 20,000 requested updates,
7,200 seconds, batch size 256, and Adam learning rate 0.001.
`sampling_chunk_size=256` is fixed for both preflight and any later separately
authorized full run.

“v2.5는 legacy hyperparameter 전체를 복원한 것이 아니다. d_model,
n_layers, diffusion_steps 및 feedback sampler 관련 값만 engineering
reference로 반영했고, batch size 256과 learning rate 0.001은 결과 관찰 전에
고정한 v2.5 고유의 사전등록 선택이다.”

The smoke suite remains `configs/benchmark_v2/smoke.yaml`, and its bounded CoF
adapter settings remain `configs/benchmark_v2/cof.yaml`, including the
100-update, five-step sampler. The full configuration remains
`configs/benchmark_v2/full_v2_5.yaml`. Neither configuration inherits capacity
values from the other.

The earlier CoF estimate of 0.25 GPU-hours per seed is **superseded; capacity
preflight pending**. It cannot be used for scheduling or authorization.

## Timing-only capacity preflight

After this amendment is committed, exactly one idle GPU may run a timing
probe. The probe uses the train split only, the full batch size 256, and
exactly 500 updates: 20 warm-up updates excluded from timing followed by 480
measured updates. It records only mean and p95 update time and peak allocated
GPU memory. It writes no checkpoint.

Sampling uses the actual full path: 50 diffusion steps, guidance 2.0,
discrete feedback after 0.3, and chunk size 256. One warm-up chunk is excluded
and five subsequent shared-plan chunks are timed. Samples stay in memory only,
are checked only for runtime finiteness, and are discarded. No sample,
association recovery, row guard, fidelity, TVD, coherence, or other quality
metric may be persisted or interpreted.

Before starting, the selected GPU must have no compute process, at most
1,024 MiB used memory, and at most 5% utilization. The report records physical
GPU index, UUID, model, driver, CUDA and PyTorch versions, and the selection
reason. CUDA unavailability or the lack of an idle GPU produces
`UNAVAILABLE` and stops.

The projection is fixed as:

```text
(
  p95_train_update_seconds * 20000
  + p95_sampling_seconds_per_entity * 7989
  + 600
) * 1.15
```

The configuration is meaningful within the two-hour cap only if the
projection is at most 7,200 seconds, peak allocated memory leaves at least
20% of physical GPU memory unused, and the probe has no OOM, NaN, or repeated
runtime error. Failure does not authorize a larger cap or smaller model. An
OOM requiring another chunk size requires a new preregistration amendment
before any full run.

The only durable preflight output is
`artifacts/benchmark_v2_5/capacity_preflight/capacity_preflight_report.md`.
It records the amendment source commit, config hash, code hash, measurements,
projection, decision, and updated scheduling table. It is a runtime artifact,
is not committed, does not write below v2.4, and is not performance evidence.

## Authorization boundary

All full-run, five-seed, and sweep counters remain zero. This amendment and
its timing report do not authorize model execution beyond the bounded
preflight. Separate user authorization is required for any full experiment.
