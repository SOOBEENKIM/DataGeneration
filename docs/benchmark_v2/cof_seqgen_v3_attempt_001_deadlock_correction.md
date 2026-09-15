# CoF-SeqGen v3 attempt_001 deadlock diagnosis and corrective design

## Scope and preservation

This correction is infrastructure-only. It does not change the v3 model,
candidate definitions, optimizer, 20,000-update request, 7,200-second hard
cap, frozen train/validation data, SamplingPlan, five guards, or thresholds.
No GPU inventory query, CUDA call, fit, sample, validation evaluation, test
access, or aggregate selection was performed during diagnosis or correction.

The complete read-only inventory is
`docs/benchmark_v2/v3_attempt_001_deadlock_preservation_inventory.json`.
At inventory time `artifacts/benchmark_v3` contained 25 files and 112,147
bytes. Its canonical file/hash inventory digest was
`e5cbc2ede8e02785babb46d73dbb0edf7af2fe34402635978cc423ad44547586`.
The original execution authorization remains byte-for-byte preserved at SHA-256
`d11941f514a65c74e07fe522f738df19447553116e024def5e0224fc9d2176d8`.

Both candidate attempts are terminal `FAILED` with `failure_class=wall_cap`:

| Candidate | Attempt tree SHA-256 | Terminal SHA-256 | Elapsed seconds | Child PID |
| --- | --- | --- | ---: | ---: |
| `cof_v3_c01_direct_joint` | `9d75a436d61caf8ace7c38c212dffe6d4d0f65dc9ffc3ccdd371b4a0f7d7a55c` | `11d44a87d7498268cfe8a58ccd394484f0a31f82b91f3f930f9588013700b264` | 7200.165826693992 | 1791073 |
| `cof_v3_c02_factorized_joint` | `454e68f4bbd77d34ab1bf395e433bb1f195b40d5b425d22f64e49762f2bbe0ca` | `8bf7250f5a5c5958693a0ce7dc3a7940bf596909b5014e6da561e39cffb2a558` | 7200.159225830983 | 1791272 |

The 36-byte `checkpoint.pt` in each attempt is the failure placeholder
`COF_V3_NO_CHECKPOINT_DUE_TO_FAILURE`, not a learned checkpoint. There is no
`progress.jsonl`, numbered checkpoint, sample result, or valid evaluation.
The attempt therefore contains no usable candidate training result.

## Reproduction evidence and static audit

The observed live state was identical for direct-joint and factorized-joint:
the parent waited in `do_sys_poll`, the child waited in
`futex_wait_queue_me`, GPU memory/utilization remained approximately zero,
and no progress/checkpoint/terminal artifact appeared after initial manifest,
conditioning, and joint-support files. The later terminal timestamps show that
both parents waited until the unchanged 7,200-second cap before recording
failure placeholders.

At source HEAD `4289048bd423b138fea0f8e18d4855b5d5a09a53`, the lifecycle was:

1. `run_authorized_v3_candidate` loaded train data and invoked Torch tensor
   operations in the parent, then passed a lambda closing over the plan, train
   context, and artifact store (old lines 919–997).
2. `run_runner_owned_child` explicitly selected multiprocessing `fork`
   (old line 1403).
3. The parent called `process.join(max_wall_seconds)` before reading any child
   queue event (old lines 1410–1411 versus queue read at 1431).
4. The child entered `_authorized_candidate_child`; its first runtime action
   was CUDA device initialization. The first observable training artifact was
   not scheduled until the 1,000-update checkpoint callback.
5. A setup deadlock or early exception therefore remained invisible to the
   parent for the full hard cap. The common failure before model-specific work
   explains why both architectures had the same symptom.

The evidence strongly supports a fork-after-Torch/thread-runtime deadlock at
child startup. A model loss or architecture defect is refuted as the immediate
cause because neither child allocated meaningful GPU memory nor reached the
first optimizer/checkpoint progress point. The exact native library futex site
cannot be recovered after process exit, so the narrowest defensible statement
is that the deadlock occurred in the inherited child startup/runtime state,
before valid training began.

## Corrective lifecycle

The runner now uses only a top-level, pickleable `RunnerChildSpec` with the
`spawn` start method. The clean child reconstructs and revalidates its plan,
source hash, authorization hash, train manifest, SamplingPlan, ownership, and
attempt path instead of inheriting the parent's Torch/thread state.

Parent supervision is event-first and bounded:

- the child emits `child_bootstrap`, then `setup_complete` after CUDA device and
  adapter setup;
- the parent reads queue events continuously instead of joining first;
- no `setup_complete` within 300 seconds records append-only
  `failure_class=startup_timeout`;
- no progress for 900 seconds after setup records append-only
  `failure_class=progress_timeout`;
- the original 7,200-second candidate wall cap remains the outer deadline;
- only the runner-owned child receives SIGTERM, a bounded grace period, and
  SIGKILL if still alive;
- child exception, child `KeyboardInterrupt`/`SystemExit`, and spawn failure
  are distinguished as `child_exception`, `operator_interrupt`, and
  `child_start_error`;
- missing artifacts are filled only by exclusive-create failure placeholders;
- successful lifecycle ordering is progress, checkpoint, candidate terminal,
  then hash-bound worker terminal.

Queue terminal metadata is consumed before bounded child cleanup. This avoids
the prior `join`-before-queue circular wait and prevents queue feeder/resource
cleanup from suppressing terminal finalization.

## Append-only continuation boundary

attempt_001 is permanently ineligible for reuse. The corrected source does not
accept the existing authorization because both source commit and relevant
source hash have changed. It also does not delete the original `ownership.lock`.

A future attempt_002 requires a new authorization with a per-candidate
`continuations` entry containing exactly:

- schema `cof-seqgen-v3-continuation-v1`;
- `previous_attempt=attempt_001` and `next_attempt=attempt_002`;
- the preserved attempt_001 canonical tree SHA-256 shown above;
- the preserved `FAILED.json` SHA-256 shown above;
- previous authorization SHA-256
  `d11941f514a65c74e07fe522f738df19447553116e024def5e0224fc9d2176d8`;
- the corrected source commit/relevant-source hashes and unchanged frozen
  config/data/SamplingPlan scope in the outer authorization.

Only after those hashes are reverified may the runner exclusive-create
`ownership_attempt_002.json` and `attempt_002`. A missing or mismatched field,
an existing attempt_002, or any mutation of attempt_001 fails closed. No such
authorization or attempt was created by this correction, and no launch command
was executed.

## Frozen provenance checked

- v3 runner config SHA-256:
  `623ab43352889d8e64c05fd17b883a255f9a92811d713bbedbf7f05ed19082b5`
- v3 source-preparation config SHA-256:
  `1f9ea3a596509b729e02157a26e60859cccf780621c1439659370c540fec6bd2`
- frozen data manifest SHA-256:
  `b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05`
- frozen SamplingPlan file SHA-256:
  `b8876c54983f8af7ed4253f2c25fdf9f351f603bc932a0152ae0a17f896e28e9`

These files were read only and were not staged for the corrective commit.
