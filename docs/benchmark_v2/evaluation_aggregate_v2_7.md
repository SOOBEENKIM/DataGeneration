# v2.7 validation-only aggregate preparation

## Status and boundary

The source parent is
`70823ea6cf88b78ecbbed897013b2013a9794663`. That source had no v2.7
aggregate-only CLI, so this task implements and tests the aggregate contract
but does not create an aggregate authorization or runtime aggregate. No
aggregate selection result in this document is authoritative.

This preparation performed no GPU query, CUDA initialization, training,
sampling, candidate replay, guard reevaluation, data generation, test access,
fresh test, TSTR, privacy analysis, or five-seed/full run.

## Read-only input inventory

Exactly one `WORKER_COMPLETE.json` and no `WORKER_FAILED.json` exists for each
model:

| Model | Worker terminal SHA-256 |
|---|---|
| CTGAN | `6928afae1d773712b73beb0def77aac994ebfbb811b34cafda4de951a0d36e63` |
| TVAE | `ad2a97955889b005b040a9f6601dbb08c1ce63064afd2c477d0636fe339c49c5` |
| CoF-SeqGen | `7a9dbf8b967f8b7f136493239f44887aa1f5fa7f297049c052173f6bff1c5888` |

There are exactly nine COMPLETE-only candidate attempts. The deterministic
input tree over `workers/` and `evaluations/` has 51 files, 6,993,542 bytes,
and SHA-256
`de3fda560550cb8c815774a28be9bcc2350db32075a6776750fc66061b3a520e`.
The canonical aggregate input inventory SHA-256 is
`c5114fb2e612106ba6b3c7582e19ab63c1c369657d797d69486bdd0cdf54fb53`.

### Candidate artifact hashes

For controls, sample/result are immutable references and guard evidence comes
from the hash-bound v2.6 validation report. No local sample or evaluation was
created for a control.

| Candidate | Manifest | Sample or reference | Evaluation evidence | Result or reference | COMPLETE |
|---|---|---|---|---|---|
| `ctgan_v27_c00_frozen_standard` | `2a5d33c82c8a7b7bd802a3bd978cad3082e5788be323491e9d0acbfea9a1697e` | `59ea09140e8b85e0e4b4eed64f7aba31f52c58ea6b0890e3a680d47c310a161b` | `2698a443b81327080c7cbd34f7cd1ee5feae80d6d74cb067b62d4d83f55736b6` | `bccb0143ca6bd3411284443c3b5cb85d8bdcb9cf81d844ec479305fcf7d1dc51` | `29da9e9346a81238c717d27b4a35a46d9d317b12e906990e4dd8c9582e027e68` |
| `ctgan_v27_c01_amount_quantile_inverse` | `f0be4ee4b59552ef67b08f0677f252576793ab13fd842c1304f22ed86561889f` | `f1e0634576d4b2e65de30138f87c8d8999d18f5969a700921f06dac4f23d7a49` | `77a3a1891af4c8a9748997ec5d55cdef2d7b16714305f08f066dd4ead1415d4a` | `e31d1008317729864f659dd94aa3fa66271f4aae1c0deb832548786700a191c1` | `d1b503d86d2672c01e02581cf66a6c30f907cec26153514d1cd9da03a9f9dce6` |
| `ctgan_v27_c02_categorical_logit` | `66bbd47c54775778e80dc6bb56d784f8791f470df0bda46614339b5a92e6b819` | `c280094869990f5e679eb2acc587ba44746fdd77ba252605b0dcfa01850695a1` | `8447b6bb2bac0adee94bb96644631b3281db3d8419f103ef758bc2b0eb54650d` | `ea18683a0453f00e15a1488985d9bccdac7d585176e26db76c1d0c3e1892e83a` | `72e8673e7d86827aea4f4e723e9ce3619ece888e1aabfd47da4daa8a66587aba` |
| `tvae_v27_c00_frozen_temperature_0_75` | `e1ba419264652e873959e896c1d68a82e7cd148d1d2fa1e7293b2207988486ec` | `9a16aca4bf20c738a7225c77dc2079458001d5cbcd5207f8e59c4730096f32ad` | `2698a443b81327080c7cbd34f7cd1ee5feae80d6d74cb067b62d4d83f55736b6` | `3146d3470a0398acae39e3f0bbc559b52bb2f36d86876ed5bbe11f7d0f907a56` | `efdf9fcc6f22337e9ce40f8710ce917afe80e1747163ad6382c8853dc30094b2` |
| `tvae_v27_c01_amount_inverse_decoder` | `8206b136920db7848899b323ae74de647a5822582601484612fe0c8971cc803b` | `57ed7595d6dc9f83a8c9679606cb18d592ceac935f7d1aef2c0356cd1339c90e` | `56dc7155791440369f4fc23ef791f916cc86417603d7091730efcdd8038581b2` | `5c65e3bd0e0331c95af4fd026bc91008253b06e90414bc54a1cb0d2bba0137c0` | `5a7a462f762ea60e2befe9a547eae00725fb180f4d0ed17983f76dbd3f63de90` |
| `tvae_v27_c02_bounded_temperature` | `330de11a1a408328b108111cbeb66bc5a96a61661d65bab4eaded41e51b8ba7e` | `604eb62d92e5d34c684eecdab75e89920b21774c18daac61ee1c58c0bfc99d28` | `a840cdb4ee2e38213633e6cff2c1ca6a16d6df75864d0cd06cbcb6d6f67ff4a5` | `4d4c98d9c33ee122203034d46e277fec2ea2b71fff2820fc91852168c7775d97` | `7af101e233e701ec11c185b56f5b6ff25020793aec6692bd68d2b2948551ce5b` |
| `cof_v27_c00_frozen_variance_residual` | `d4a4caccebf64d90aabf388e46faf11cdb3d9e72da1a6b930ae433cbbb8d1b31` | `c91b386473bcbbabd0f7f2d7f9b3659b760d282f70061a449254f1a0d03f34a3` | `2698a443b81327080c7cbd34f7cd1ee5feae80d6d74cb067b62d4d83f55736b6` | `d68a78c228520a3edaffae3dfe076c7d59e446575c576905d21f7f53bc639fac` | `f992acd4fe5a830b6062f1b8675e3c3af26dc3022be8a437e08803582075afec` |
| `cof_v27_c01_empirical_residual` | `f04cd5ef65c2d8f9df90e091d88bed94d89b145263dbb8cdac0bc2877da14288` | `364b67ce239b9f692183af00e5f496002d73c9e97d5679d06a7ec168bd8cb2c0` | `bfa153860fa0ceb187454e77f3284fc1302bfadad16decf9d052023a8d62cbad` | `2edba78d3e1d31ef3fab345b93f0c303484fc256ff1db8d71b82445bbf4722bb` | `1b858813a4441e5e8c5e177c0d64eb03f8f99d9748fed6f3ed4d150971af36b9` |
| `cof_v27_c02_gap_logit_bias` | `bece0921f985688fd122386809df6244660c1159827ea3fceeba23f8aea22744` | `c6b02022e7d8c7da55a0db15d9c7ec7886840fa0e28c8ebee413a5839426a375` | `a758d8ae9a1faa93c2a8d37eedcf3690d29430da00e1ad17dbcccffa9dd67427` | `c2987eb5ebf2344b5c448886b69a529657d25fe03cff55275c5e37794a5c6288` | `8494c0177ac21cddcea7c77797687a4e4899f81eb9d43a57955f959565419e53` |

## Frozen provenance

| Input | SHA-256 |
|---|---|
| candidate execution authorization | `ff8bfdab5177d4b067e74f2cf192590b152236406b2eb4c1664878c20b5beea5` |
| evaluation runner config | `05435443f69f25f56e6d738dcae7c5a62f0c99096483b5ac26c6f525cffae68c` |
| candidate config | `2737723a05ce8628b7c9b8e1afbe7edae323a06971fbaf1eb00465224e0d654b` |
| development manifest | `31ddf45dcf7b4f982ed990ec94281c3e6bad0454ec3182e2434e42358006ade5` |
| train file | `c67a6fce4593317e11e1b6f36397bfb38f9607916bf8f5b32329d8fa99b700a8` |
| train content | `0c03f179930cd4d89ccc8a8b66283e620313e16d15fd53f0674c133bcb80c09d` |
| validation file | `68c15c05c55742bfce0f22560ffe8378d032a7a9fd9a1072e80ec8da5340fba5` |
| validation content | `aa1576b4ccae6a2ca30d0472f0782e1231ecc61ab3d224590a772dc3448b9e66` |
| train-only SamplingPlan | `862be1aa149b98e5521d0197a53487c9a9df535fc4853b33b95186cc3fb8cc27` |
| v2.5 frozen-data manifest | `b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05` |
| frozen v2.6 control guard report | `2698a443b81327080c7cbd34f7cd1ee5feae80d6d74cb067b62d4d83f55736b6` |

All nine attempts identify candidate source commit
`de57f79b0b15a9086b6ae26beea25c7f55427a7d` and relevant source hash
`c87491ce2b352c3ef6e862e1e67845546c135a46d94d525b5aea43aea9c54df1`.
The later finalization correction does not alter that candidate provenance.

## Implemented aggregate contract

The new aggregate-only CLI supports `plan`, `dry-run`, and `execute`, all
requiring a separate authorization. It accepts no device or GPU option.

- `plan` validates the frozen train/validation provenance, exact three worker
  terminals, nine candidate terminals, execution authorization, artifact
  hashes, and preserved 51-file input tree. It writes nothing.
- `dry-run` reads only stored `evaluation.json` results plus the hash-bound
  prior guard evidence for controls. It verifies the recorded five-guard
  arithmetic and applies the preregistered rule without opening a validation
  sample or rerunning evaluation. It writes nothing.
- `execute` repeats those checks and exclusive-creates one
  `aggregate_attempt_NNN`. It writes readiness, report, selection manifest,
  checksum manifest, artifact index, and finally `AGGREGATE_COMPLETE.json`.

Eligibility and ordering remain exactly:

1. all five guards PASS;
2. minimum `max(amount_ks, gap_ks)`;
3. minimum `amount_ks + gap_ks`;
4. lexicographic candidate ID.

The authorization schema binds the future aggregate source commit/hash,
both unchanged configs, candidate execution source/authorization, exact
worker and candidate hashes, preserved input-tree digest, frozen
train/validation/SamplingPlan provenance, and an aggregate-only scope. GPU,
CUDA, training, sampling, replay, test access, threshold changes, fresh test,
TSTR, privacy, and five-seed/full execution are false.

No authorization or aggregate artifact was created in this source-only
preparation. Therefore model-level `SELECTED`/`NO_PASSING_CANDIDATE` and
`primary_c2_selection_ready` have not yet been authoritatively produced.

## Verification

- focused aggregate/evaluation/preparation/selection tests: 51 passed;
- repository-wide tests: 328 passed, 23 existing dependency warnings;
- `compileall`: PASS;
- `git diff --check`: PASS;
- real runtime read-only inventory: PASS, 3 workers and 9 candidates;
- aggregate authorization created: false;
- aggregate runtime artifact created: false.
