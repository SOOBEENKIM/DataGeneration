# v2.7 Evaluation-Only Runner

Status: source-only implementation. Execution, authorization creation, GPU
inventory queries, CUDA calls, sampling, selection, fresh-test evaluation,
TSTR, privacy analysis, training, and five-seed/full runs remain unauthorized.

## Frozen input inventory

The implementation was based on source-preparation commit
`7c3a5a017e5652b567d9fbbe437d6ea04105bcb2`. The following existing inputs
were verified read-only before implementation:

| Input | SHA-256 |
|---|---|
| v2.7 source-preparation config | `2737723a05ce8628b7c9b8e1afbe7edae323a06971fbaf1eb00465224e0d654b` |
| v2.5 full config | `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3` |
| v2.5 `FINAL_COMPLETE.json` | `e47d46b995cdaa8f564ccb4cca56eeda4e9a17dba43c1ed2ca8c1a954e657d8a` |
| v2.5 final artifact index | `c8f73c7d8dfc5513f9f0f10cb9e799c654f35e4367e1e9c3288809889c8e2da9` |
| v2.5 final checksum manifest | `730f8a1c23a965e140bce11ace6787bd8a7120df6fd61185fd41f03bd19cf0f8` |
| v2.5 frozen-data manifest | `b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05` |
| v2.6 single-factor config | `707ecdbd37bc55f4a3b9ece603b70d26c8274015aa9cb97d5d376aab3d9ecb7c` |
| v2.6 aggregate complete | `404c820d79cb3d17e21d617f5c255c0d1b97ed6fb7c94b9fdcd23682bdc5bf35` |
| v2.6 aggregate index | `7620c08d071dabc320c7698a58d313fcb6812df82c29ea6f23fcb12d9e1cd97c` |
| v2.6 forensic complete | `0c2e89a4124d35ccfa565ab504fbedf8819bed2dfe649370f811efe51a408703` |
| v2.6 forensic index | `55ace91249b655ce39124176eaa5a529c28ce1c22d752a4eca0af2e141479f3b` |

The v2.6 candidate, worker, trajectory, evaluation, aggregate, and forensic
tree-record hashes remain, respectively:

`03c0926523decd8114e7e63d77369ece1a065f7732441f76c17149d350720267`,
`8180c4d4fb576e1aa8953b4e5298b3ab717b1618adb080bab9b9fddea13fd819`,
`5814437ec364a70dc26c167ba43f548e62065934fe4eca0b49d374d71746897a`,
`32caf2c35a510b996a1211362df6bad7ae3c3fcc92bfee66c3b335d08465c55a`,
`fd43d5abd73dfb76289f3b58f465892a920d7e1a29579c701823c4502b6080b6`,
and `a54bcdb41adfef8e712abf083dd0f647a024a63ea0889d2ddcef15a84fee9a26`.

## Evaluation plan

There are nine fixed candidates:

- Three frozen controls reference their existing candidate-result,
  validation-sample, and checkpoint bytes by SHA-256. They perform no new
  sampling.
- Six candidates restore the v2.6 20,000-update native checkpoints and make
  one factor change each. They perform no fit, optimizer update, or checkpoint
  training.

The checkpoint inventory is:

| Model | Checkpoint | SHA-256 |
|---|---|---|
| CTGAN | `artifacts/benchmark_v2_6/selection/candidates/_trajectories/ctgan_separate_class/ctgan_native_seed_2601/seed_2601/attempt_001/checkpoints/step_20000.pt` | `7e9de7e609ad48e8de687426240fe4fd0a47b5d2d0abcbc70a4edb58f182d10d` |
| TVAE | `artifacts/benchmark_v2_6/selection/candidates/_trajectories/tvae_separate_class/tvae_native_seed_2601/seed_2601/attempt_001/checkpoints/step_20000.pt` | `6f5263a5a40d9513342b33a823a02769bab65f6e1a5bd38c814b87d771a2947c` |
| CoF-SeqGen | `artifacts/benchmark_v2_6/selection/candidates/_trajectories/cof_seqgen/cof_native_seed_2601/seed_2601/attempt_001/checkpoints/step_20000.pt` | `7ce44721cadffef21ade3bf1884421c304f73ea9eff1e9bfb3ca2d9e63cc0c29` |

CTGAN and TVAE preserve the v2.6 CPU-first deserialization followed by an
explicit selected-device move. CoF preserves selected-device restore.

Train-fitted quantile maps, categorical logit offsets, bounded temperature,
empirical residual maps, and gap-logit offsets use only valid frozen train
rows and the train-only SamplingPlan. Their full parameter state plus source,
config, train file/content, and SamplingPlan hashes is stored in a canonical
hashed fit-state record. Validation data is used only by the fixed five-guard
evaluation after generation. Test-path access fails closed.

The intervention sites are deliberately narrow:

- CTGAN amount inverse mapping changes amount only; its categorical candidate
  adds class-specific train-fitted offsets only to gap and receiver logits.
- TVAE amount inverse decoding changes amount only; its categorical candidate
  selects only from the frozen temperature grid using train-plan outputs.
- CoF empirical residual sampling changes amount only; its gap candidate adds
  class-specific gap-logit bias after classifier-free guidance and preserves
  the frozen variance-residual amount baseline.

## Future authorization and artifact contract

`execute` requires an external append-only authorization manifest that pins
the source commit, relevant-source hash, both config hashes, frozen
train/validation and SamplingPlan hashes, all candidate IDs, all checkpoint
hashes, and the exact evaluation-only scope. This source-preparation task does
not create that manifest.

If separately authorized later, each model owns only:

```text
artifacts/benchmark_v2_7/candidate_selection/
  workers/<model>/ownership.lock
  workers/<model>/WORKER_{COMPLETE|FAILED}.json
  evaluations/<model>/<candidate>/seed_2601/attempt_NNN/
```

All JSON writes and attempt allocation are exclusive-create and append-only.
A child process owns each evaluation, while its parent enforces the fixed
per-candidate wall cap and terminates only that runner-owned child. A failed
candidate leaves `FAILED.json`; a completed candidate leaves its immutable
manifest, fit state, validation sample, five-guard evaluation, result, and
`COMPLETE.json`. A model worker never aggregates or selects candidates.

## Read-only commands

The following commands perform plan and frozen-provenance validation only:

```bash
python -m scripts.run_evaluation_only_v2_7 --mode plan
python -m scripts.run_evaluation_only_v2_7 --mode dry-run
```

No GPU launch command or authorization is defined by this preparation.
