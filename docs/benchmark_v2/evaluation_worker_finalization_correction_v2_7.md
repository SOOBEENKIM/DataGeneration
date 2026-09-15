# v2.7 evaluation-only worker finalization correction

## Scope and preservation

This source correction starts from commit
`de57f79b0b15a9086b6ae26beea25c7f55427a7d`. It changes only
evaluation-worker terminal orchestration and separately authorized,
finalization-only recovery support. It does not change or rerun a checkpoint,
candidate sample, evaluation, five-guard result, config, SamplingPlan, DGP,
selection, fresh test, TSTR, privacy analysis, or full experiment.

The CTGAN, TVAE, and CoF evaluation-only runtime trees remain append-only and
read-only. Diagnosis and correction used no GPU query, CUDA call, training,
sampling, evaluation, data generation, or selection.

## Read-only runtime inventory

The preserved CTGAN candidate-evaluation subtree contains 15 files and
2,328,524 bytes. Its deterministic subtree-relative record digest is
`f948ef7868382714a26e1b9bf1334268b1905bb830b6be07b6a69c1c534b0999`.

| Candidate | Artifact | SHA-256 |
|---|---|---|
| `ctgan_v27_c00_frozen_standard` | `manifest.json` | `2a5d33c82c8a7b7bd802a3bd978cad3082e5788be323491e9d0acbfea9a1697e` |
| `ctgan_v27_c00_frozen_standard` | `control_reference.json` | `a84148627452641145adf7a7819caf550efab0b30a5edb594d384fbfdd4a7516` |
| `ctgan_v27_c00_frozen_standard` | `COMPLETE.json` | `29da9e9346a81238c717d27b4a35a46d9d317b12e906990e4dd8c9582e027e68` |
| `ctgan_v27_c01_amount_quantile_inverse` | `manifest.json` | `f0be4ee4b59552ef67b08f0677f252576793ab13fd842c1304f22ed86561889f` |
| `ctgan_v27_c01_amount_quantile_inverse` | `validation_sample.npz` | `f1e0634576d4b2e65de30138f87c8d8999d18f5969a700921f06dac4f23d7a49` |
| `ctgan_v27_c01_amount_quantile_inverse` | `evaluation.json` | `77a3a1891af4c8a9748997ec5d55cdef2d7b16714305f08f066dd4ead1415d4a` |
| `ctgan_v27_c01_amount_quantile_inverse` | `COMPLETE.json` | `d1b503d86d2672c01e02581cf66a6c30f907cec26153514d1cd9da03a9f9dce6` |
| `ctgan_v27_c02_categorical_logit` | `manifest.json` | `66bbd47c54775778e80dc6bb56d784f8791f470df0bda46614339b5a92e6b819` |
| `ctgan_v27_c02_categorical_logit` | `validation_sample.npz` | `c280094869990f5e679eb2acc587ba44746fdd77ba252605b0dcfa01850695a1` |
| `ctgan_v27_c02_categorical_logit` | `evaluation.json` | `8447b6bb2bac0adee94bb96644631b3281db3d8419f103ef758bc2b0eb54650d` |
| `ctgan_v27_c02_categorical_logit` | `COMPLETE.json` | `72e8673e7d86827aea4f4e723e9ce3619ece888e1aabfd47da4daa8a66587aba` |

The c01 and c02 fit-state and candidate-result hashes are also bound by their
respective `COMPLETE.json` records. No candidate path was changed during this
work.

The initially markerless CTGAN worker completed naturally during read-only
diagnosis. Its `WORKER_COMPLETE.json` was not created manually and the runner
was not killed. The final worker subtree now contains two files and 1,145
bytes, with record digest
`b763048ea7986e4c37e4152b9c4214d8b1178bf0a7626783ff20ee7a855c7fc4`.
The worker terminal SHA-256 is
`6928afae1d773712b73beb0def77aac994ebfbb811b34cafda4de951a0d36e63`.

The unchanged TVAE and CoF worker subtree record digests are, respectively,
`0bfd09dc08d739f9a6e684ba3cf4639a7f8db4a1bd490d04d8d8a086005ad79e`
and
`375a6a4b437e24471b08ed2c1a0a6cb2f1a666ff9fee1dd259ca2ed8ca334e21`.
The entire candidate-selection tree has 51 files, 6,993,542 bytes, and record
digest
`de3fda560550cb8c815774a28be9bcc2350db32075a6776750fc66061b3a520e`.

## Reproduction and root cause

The final CTGAN candidate wrote `COMPLETE.json` at 19:32:41 KST. The original
parent did not write `WORKER_COMPLETE.json` until 19:42:18 KST. During the
delay, read-only process inspection showed:

- parent PID 1718003 sleeping in `do_sys_poll`;
- a multiprocessing resource tracker;
- a spawned evaluation child;
- a joblib loky resource tracker and worker with multiple threads;
- parent stdout and stderr still open on the CTGAN runner log.

CTGAN's data transformer uses joblib parallel work. The prior parent loop
waited for `child.is_alive()` to become false before consuming the result
queue, while the child did not emit its COMPLETE message until the evaluation
function returned. Candidate files were already complete, but interpreter and
joblib/resource-tracker teardown kept the child process alive. Consequently,
worker finalization was coupled to unbounded process cleanup rather than the
verified candidate terminal event.

The deterministic regression fixture reproduces the ordering defect with a
child that writes a valid candidate terminal and then retains a non-daemon
cleanup resource.

## Corrective contract

The corrected child emits COMPLETE immediately after exclusive creation of
the candidate `COMPLETE.json`. The parent now:

1. polls the result queue while the child is alive;
2. requires the candidate terminal file before accepting COMPLETE;
3. writes the final worker terminal before waiting for child cleanup;
4. allows a bounded cleanup grace;
5. terminates, and if necessary kills, only the runner-owned child after that
   grace.

An abnormal child exit, wall cap, reported exception, or COMPLETE event
without its candidate terminal remains FAILED and produces
`WORKER_FAILED.json`. TVAE and CoF use the same terminal contract. The
candidate artifact format and restore paths are unchanged.

The append-only ownership lock continues to reject a second normal worker, so
the three already COMPLETE CTGAN candidates cannot be replayed by this
correction.

## Finalization-only recovery

A genuinely markerless completed worker can be finalized only through the
new `finalize-only` mode and a separate authorization. The authorization must
bind the corrected source and relevant-source hashes, both unchanged configs,
the original execution authorization hash, ownership-lock hash, exact
candidate manifest and COMPLETE hashes, and the whole preserved candidate
tree digest. Its action scope must explicitly prohibit candidate execution,
replay, sampling, evaluation, training, optimizer updates, selection, fresh
test, TSTR, privacy, and five-seed/full execution.

Recovery verifies exactly one COMPLETE-only attempt for every expected
candidate, writes only the missing worker terminal, and then rechecks every
candidate attempt digest. Missing or mismatched authorization/provenance fails
closed.

The preserved real CTGAN worker is now terminal, so recovery authorization
and an SSH recovery command are neither needed nor permitted for this run.
The finalization-only interface exists only for a future genuinely markerless
runtime.

## Frozen configuration

The runner config remains
`05435443f69f25f56e6d738dcae7c5a62f0c99096483b5ac26c6f525cffae68c`.
The candidate config remains
`2737723a05ce8628b7c9b8e1afbe7edae323a06971fbaf1eb00465224e0d654b`.
No runtime or authorization artifact is part of the corrective source commit.

## Verification

- focused finalization, evaluation-runner, checkpoint-restore, and preparation
  tests: 34 passed, 4 dependency warnings;
- repository-wide tests: 320 passed, 23 dependency warnings;
- `compileall`: PASS;
- `git diff --check`: PASS.

The preparation dry-run regression now snapshots any existing runtime tree
before and after the read-only check instead of assuming that a legitimately
executed runtime root does not exist. This preserves the original no-write
contract at the current lifecycle stage.
