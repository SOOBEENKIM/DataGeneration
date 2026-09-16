# CS-SAF recoverable research state

Updated 2026-09-16. Branch: `research/cs-saf`.

**Latest v4 comparison COMPLETE: retain E raw forward, add the same .01 centered
residual penalty. Two fresh GPU fits; original response gates and registered
best-validation accuracy screens vs E/U PASS. Eligibility for a separately
registered broader pilot, not multi-seed or external-baseline superiority.**

[Full v4 result](v4_pilot_v1_report_2026_09_16.md),
[machine-readable evidence](v4_pilot_v1_result.json),
[preregistration](revision_v4_preregistration.md).

- Registration `642b878`; CPU/GPU source `635664954057bf037ef04c572ac113db8fb28767`.
- New ER predicts b+h+r exactly as E; only the regularizer centers r. Same
  133,581 parameters, pi=.05, seed 20260930, kappa 0/1, no warm start/search.
  Saved E/C/R/U reused with verified hashes, matching data, states/order/measures.
- 123 tests and same-source repeated CPU gate PASS. Remote GPU **2**, selected
  after confirming no compute process; busy GPUs 0/1/3 avoided. Two GPU fits complete.
- ER max null copy/repeat **.018631945/.018336381**, active **.354965944/.349362813**.
  Original response/zero-gap/generated-validity gates PASS.
- Active best-validation mark TV: U .086640340, E .085479746, R .096924737,
  **ER .085439410**. ER-E active delta −.000040335, entity SE .000098938;
  this very small difference is not robust accuracy-superiority evidence.
- Three-null mean TV ER-E **−.000336656**, ER-U **−.000073495**. Individual null
  cells can worsen; no universal per-cell accuracy improvement.
- **Fixed epoch 9 ER-E active TV +.000509623 (worse)**. Preserve this limitation:
  the registered best-checkpoint screen passes but E accuracy improvement is
  checkpoint-sensitive. ER-R active improvement remains at best and epoch 9.
- 101 checksum comparisons, 20 checkpoint tensor identities, 80 aligned entity
  groups, 2,240 statistics and paired factorial contrasts verified. 4,096 generated
  entities / 94,662 nonfirst gaps valid. No held-out, later pi, new seed or baseline run.
- Next proposed work: freeze U/E/ER and separately preregister seed/prevalence
  replication; external sequential baselines and broader validity follow. This
  broader experiment is proposed only. Do not keep modifying until a win or
  reclassify this single-seed adaptive pilot as independent confirmation.

**Previous v3 model result: v3 eight-fit pilot COMPLETE. R passes the original response
criteria at pi=.05 but FAILS the registered accuracy screen against historical U.
Do not promote this candidate to later prevalences or confirmation.**

[Full v3 result](v3_pilot_v1_report_2026_09_16.md),
[machine-readable evidence](v3_pilot_v1_result.json),
[preregistration](revision_v3_preregistration.md).

- Registration `9813573`; final CPU/GPU source `44a2bd3848abd1fdddf0fc37d2cd6894109e3d65`.
- Implemented H (history only), E (extra history/raw route), C (centered route),
  R (same as C + fixed .01 smooth-RMS residual penalty), eight fresh GPU fits.
  Each adds 32 history coefficients; 133,581 stored parameters. H has 1,056
  dormant route parameters. Old U evaluated at saved states, zero new U fits.
- 106 tests PASS; same-source CPU reproducibility, checkpoint, loss-drop,
  support and zero-gap checks PASS. Initial CPU numeric failure preserved;
  constant probabilities now evaluated once then broadcast, no tolerance change.
  All CPU trained states remain bitwise identical before/after this audit fix.
- R maximum null copy/repeat range **.003943601/.003880979**, active
  **.340098053/.334719523**. All original response/validity gates PASS.
- E/C rare-null copy ranges **.056817306/.058481466** still exceed .05.
- Active conditional mark TV (lower better): U **.086640340**, E **.085479746**,
  C **.102974418**, R **.096924737**. R improves C by .006049681 but remains
  .010284397 worse than U. Same direction at fixed epoch 9 and on factual TV.
- Three-null mean TV R−C **−.000395910**, R−U **−.000113982**. Two individual
  null cells are slightly worse than U; no universal null-accuracy improvement.
- Response safety improved in this single pilot; conditional accuracy vs U is
  unresolved. The centering parameterization already worsens active accuracy
  before regularization, so blame cannot be assigned solely to the penalty.
- 97 file checksums, 20 checkpoint tensor identities, 80 aligned array groups,
  2,240 statistics and paired contrasts verified. 16,384 entities / 378,648 gaps
  generated, support/mark/value checks PASS. No held-out, later pi or new seed.

**Previous diagnostic COMPLETE: [fixed-checkpoint route decomposition](route_decomposition_v1_report_2026_09_16.md),
[evidence](route_decomposition_v1_result.json), and
[model/theory/data/literature/claim audit](research_claims_and_related_work_audit_2026_09_16.md).**

- Registered `d12021b` before implementation. Final CPU/GPU source `7d5a1be`.
- 92 tests PASS, same-source CPU PASS; 12 best/epoch-9 snapshot records from
  six existing pi=.05 U/A/B fits, full train/validation, both contexts/all bins.
- **New fits, optimizer steps, generated samples and held-out access: zero.**
- U rare-null best validation repeat BCE: full F **.489598539**, mean-only M
  **.488799468**, residual-only R **.542980485**, whole-route zero Z **.542022948**.
  The train-reference mean is useful. Removing the residual improves validation
  slightly but worsens train (+.000829508); epoch 9 repeats the sign pattern.
- U rare-active validation F **.451531454 -> M .503302035** (+.051770582).
  Useful active gap dependence would also be lost by unconditional removal.
- Registered mean-usefulness and active-residual-usefulness directions hold;
  **null residual harmful in BOTH train and validation does not hold**.
  Do not reclassify this mixed result as a fully supported hypothesis.
- All fixed mechanical checks PASS; 23 file checksums, 12 tensor identities,
  48 aligned entity-array groups and 4,416 statistics independently rechecked.
- First `audit_v1` failed a numerical identity: batch 128 vs parent 256 changed
  CUDA batch-shape arithmetic. Matching 256 resolves it; final historical loss mismatch
  <=1.37e-9, unchanged 1e-6 limit. Failure preserved; final output `audit_v2`.
- No new trained candidate, superiority or dilution claim. Centering alone is
  function-preserving; history capacity and residual regularization require
  separate controls. Related work already covers copy mixtures, interactions,
  sequential generation, group weighting and behavioral fidelity diagnostics.

**Historical training result: the separately registered three-objective diagnostic is
COMPLETE. All U/A/B fail the pi=.05 rare-null response gate. No model success
or expansion to later prevalences is established. V1/v2 failures are preserved.**

Previous training execution: [loss-control report](loss_control_v1_report_2026_09_16.md)
and [machine-readable evidence](loss_control_v1_result.json).

- Registration `d4069c9` preceded implementation/fits. CPU/GPU source:
  `f4bed4b9a17970d2cd146515f3e97d1ae0db6947`.
- **82 tests PASS**, six tiny CPU fits (two each U/A/B), all CPU gates PASS.
- **Six GPU fits completed**, pi=.05 x kappa 0/1 x U/A/B, same v2 architecture,
  initial states, data/order, budget and checkpoint selection. A adds a global
  transition-mean repeat BCE; B retains equal-context repeat BCE.
- U/B's four prior best states reproduced **bitwise**, with identical epochs,
  likelihoods and responses; no trained weight warm-start.
- Kappa=0 rare copy ranges U/A/B: **.055189/.061733/.081949**, all >.05.
  Repeat ranges also fail. Added auxiliary and context redistribution each
  increase this null response; the same ordering holds at fixed epoch 9.
- U train/validation null ranges .054936/.055189; train-defined central-bin
  ranges .053865/.054159; 89.06%/90.33% of entities individually exceed .05.
  This is not restricted to validation outliers or extreme gap bins.
- Turning off the whole U gap route worsens rare-null validation repeat BCE
  **.489599 -> .542023** at fixed weights. The route cannot be treated as
  uniformly useless; history correction and current-gap variation may be mixed.
- B-A validation repeat BCE: rare null **+.006451**, active **-.006080**.
  This is a tradeoff in one exploratory seed, not universal harm or superiority.
- 12,288 generated entities / 283,986 nonfirst gaps; generation checks PASS.
  62 checksum comparisons and checkpoint/sample/diagnostic array identities
  verified. No worker error; each candidate's local scientific gate is FAIL.
- No later prevalence, five-seed, real-data or held-out run. The then-proposed
  train-centered route diagnostic subsequently completed; see latest work above.

Prior v2 implementation and pilot:

V2 implementation preserves the registered 133,549 parameters and fresh-v1
common initialization. Direct bank-gradient isolation, known-label permutation,
strict-past behavior, likelihood/audit agreement, support, checkpoint roundtrip
and fixed auxiliary denominators passed. The revision-aware runner pins the
original v2 YAML hash and inherited v1 budget, requires the same-source CPU gate,
and reuses the immutable data/oracle index. V2 outputs use a separate namespace.
The v2 YAML retains its historical preregistration-state metadata; this file
records current execution state.

V2 execution (2026-09-16, before the separate loss diagnostic):

- Source: `c77d2f9558ca019d14ccf4b833cbc6628728ff44` (committed before CPU/GPU).
- Immutable registration: `d8302e3`; model seed 20260930; 133,549 parameters.
- CPU: two identical histories/best states, train objective drop **48.33%**;
  support, vocabulary, finite-value and zero-gap checks PASS.
- GPU: **4 fits**, CS2-U1/B1 x kappa 0/1, at pi=.05 only.
- Primary CS2-B1 fails kappa=0/label=1: copy **0.081949**, repeat **0.080610**,
  both above .05. The same v1 cell was below .05 (copy 0.045730).
- At kappa=1/label=0, B1 copy response falls from v1 0.046383 to v2 **0.033930**.
  Active kappa=1/label=1 copy/repeat responses remain **0.343783/0.338277**.
- CS2-U1 also fails kappa=0/label=1 (copy **0.055189**, repeat **0.054313**).
  Thus the new failure is not attributable only to the balanced auxiliary.
- All four jobs have exact zero-gap invariance and valid generation. Total
  8,192 entities / 189,324 nonfirst gaps; support violations and reserved marks 0.
- Pi=.10/.25/.50 training stopped by the original rule. No fallback, five-seed,
  real-data or held-out run. Parent exit 2 is scientific FAIL; no worker error.
- Saved checkpoint/sample identities and 21 artifact checksums rechecked.
- The then-proposed three-objective control was subsequently registered and
  completed separately; see the latest result above.

Prior v1 checkpoint forensics (2026-09-16):

- Forensic source: `d96e1b4966f9fec018aee472a26be579ab9a4d1c`.
- All 12 saved best checkpoints analyzed on full train histories. New fits and
  optimizer steps: zero. No new validation/test content or realized latents read.
- Failed-cell train null range: 0.051240, close to existing validation 0.051167.
  Median entity range 0.050248; 51.36% of null entities exceed .05 individually.
- Central observed-gap mass retains 88.49% of the range. The reaction is not
  confined to a few entities, early histories or only tail bins.
- The variation resides in the bilinear copy route. The fresh-mark term only
  attenuates it. Shared direct-route gradients transmit local cross-context
  influence, but null self-descent also raises the response. Strong gradient
  conflict or a unique historical cause is **not** established.
- Three new diagnostic tests; 16 focused tests passed. Checkpoint state hashes
  unchanged, and 24 checkpoint/array checksums verified after analysis.
- V2 registered: replace one shared rank-32 route with two observed-context
  rank-16 routes; preserve 5,440 route / 133,549 total parameters, objectives,
  data and pilot gates. No known active-label mask. Per-context capacity falls;
  shared-feature influence and within-context spurious responses remain possible.

V1 execution retained below:

- Historical base: GitHub `main` at `de8fa70`; repaired SAF v6 failed its pilot.
- Original protocol: `a191392`; exact architecture/oracle: `34cdf66`.
- Model, prevalence data and pilot contract: `924131b`; API fix: `43ceef8`.
- Exact CPU/GPU source: `0d3be638e180695bf62379af8444398bb0c9708e`.
- C0/U0/U1/B0/B1 implemented. The four aligned candidates have matching
  parameter keys/shapes/initialization. U1 and B1 have 133,549 parameters each.
- All eight pi x kappa train/validation views created. Original entity splits
  retained; train-only oracles PASS at every pi. No test bodies materialized.
- Relevant tests: **54 passed**. Two CPU runs have identical histories and
  best-state tensors; train objective decreased **48.16%**.
- GPU: **12 fits**, U1/B1 x kappa 0/1 x pi 0.05/0.10/0.25, seed 20260930.
- CS-B1: pi=0.05 PASS, pi=0.10 PASS, **pi=0.25 FAIL**. At pi=0.25,
  kappa=1/label=0 copy/repeat ranges are **0.051167 / 0.050313**, exceeding 0.05.
  Active ranges are 0.367779 / 0.361659; absent active signal is not the failure.
- Pi=0.50 **data/oracle completed; training not run** under the stop rule.
- All 12 runs: exact zero-gap invariance, no support violations or reserved
  marks, finite generated values. 24,576 generated entities retained locally.
- Validation used for checkpoint selection and predictive-response audit;
  test content not accessed. No five-seed or real-data study started.

Read these files in order when recovering from a missing conversation:

1. [Latest v3 trained result and limits](v3_pilot_v1_report_2026_09_16.md)
   and [evidence](v3_pilot_v1_result.json).
2. [V3 preregistration](revision_v3_preregistration.md).
3. [Latest decomposition and limits](route_decomposition_v1_report_2026_09_16.md),
   [evidence](route_decomposition_v1_result.json), and
   [comprehensive research/claim audit](research_claims_and_related_work_audit_2026_09_16.md).
4. [Decomposition preregistration](route_decomposition_v1_preregistration.md).
5. [Three-objective result and limits](loss_control_v1_report_2026_09_16.md)
   and [evidence](loss_control_v1_result.json).
6. [Loss diagnostic preregistration](loss_control_v1_preregistration.md).
7. [V2 pilot result](v2_pilot_v1_report_2026_09_16.md) and [evidence](v2_pilot_v1_result.json).
8. [V2 preregistration and implementation contract](revision_v2_preregistration.md).
9. [Prior checkpoint analysis](checkpoint_forensics_v1_report_2026_09_16.md)
   and [evidence](checkpoint_forensics_v1_result.json).
10. [V1 pilot report](pilot_v1_report_2026_09_16.md) and [evidence](pilot_v1_result.json).
11. [Frozen v1 pilot contract](pilot_execution_contract_v1.md).
12. [Research protocol](research_protocol_v1.md).
13. [Exact architecture and oracle contract](architecture_and_oracle_v1.md).
14. [Earlier oracle-only result](oracle_audit_v1_report_2026_09_16.md).

**Next hypothesis: retain E's uncentered forward parameterization and apply the
same centered-residual penalty only in the objective.** E has better active
accuracy than C, while R improves C and removes null response. The missing
E+penalty combination could separate forward parameterization from functional
regularization without changing capacity or coefficient. It is proposed, not
registered, implemented or trained. Do not auto-run a fallback after the failed
accuracy screen or tune lambda from these results. A new contract must fix
comparators, budget, accuracy/response gates and stopping rules before execution.
The present R candidate does not advance. Multiple seeds, prevalence and fair
external sequential baselines remain required before a method-success claim.

Runtime roots relative to this CS-SAF worktree:

- `data/cs_saf/prevalence_v1/`
- `artifacts/cs_saf/prepared_v1/`
- `artifacts/cs_saf/cpu_gate_v1/`
- `artifacts/cs_saf/pilot_v1/`
- `artifacts/cs_saf/forensics_v1/` (entity diagnostics and gradient arrays)

V2 output namespace: `artifacts/cs_saf/revision_v2/`, containing `cpu_gate_v1/`
and `pilot_v1/` with terminal records.

Loss diagnostic: `artifacts/cs_saf/loss_control_v1/cpu_v1/` and `gpu_v1/`,
with best and epoch-9 checkpoints for all six fits, and train/validation entity arrays.

Previous decomposition: `artifacts/cs_saf/route_decomposition_v1/cpu_v2/` and
`audit_v2/`. The first technical failure is retained in `audit_v1/`.

Latest v3 outputs: `artifacts/cs_saf/revision_v3/cpu_v1/` (preserved failure),
`cpu_v2/` (PASS), and `gpu_v1/` (eight fits plus fixed U evaluation).

Each trained job retains its best checkpoint, history/report, intervention
audit and generated sample. Compact evidence records absolute locations,
source/config/data hashes and checked artifact checksums (100 for v1,
24 checkpoint/array checks in forensics, 21 for v2, 62 for loss control).

Store code/config commits before execution and report commits afterwards.
Preserve failures as well as successes. Runtime data/checkpoints remain on the
workstation; the GitHub repository is not their full backup.
