# Benchmark v2.5 scheduler-continuation correction

## Scope

This is an infrastructure-only correction. It does not change or reinterpret
the frozen benchmark config, DGP, model code, endpoint, evaluator, direct
row-marginal hard guards, CTGAN/HSMM outcomes, C2 rule, training steps, or
two-hour cap. No full retry, CPU/GPU baseline, learned training, frozen-data
generation, or real GPU query was performed while implementing it.

## Preserved attempt_002 evidence

The read-only inventory is frozen in
`attempt_002_preservation_manifest_v2_5.json`.

- attempt_002 files: 983
- attempt_002 bytes: 5,040,277,054
- attempt_002 tree SHA-256:
  `83fc03cb49b8fa4578ea5b1ca19590fcb24371d01635afaebf7ddbdfe1afc968`
- terminal markers: 48 files / 38,798 bytes
- terminal-marker tree SHA-256:
  `50fafa1a7641ddf5e11d7aeced0a94ea65e7578ee49cafc4b5e8052afc5ff23c`
- terminal states: 40 COMPLETE, 4 INVALID, 4 CANCELLED, 0 FAILED
- preserved STOPPED marker SHA-256:
  `961b6f6949ef615f75d74cb68382ceca573948ac3d714b823d9acadb05eee892`

CTGAN seeds 1–3 and HSMM seed 1 remain INVALID for their recorded
row-marginal hard-guard failures. HSMM seeds 2–5 remain CANCELLED. C2 remains
`NOT_EVALUABLE` because CTGAN is a primary comparator. These results and
thresholds were not changed.

The pending set is exactly 17 jobs: CTGAN seeds 4–5, TVAE seeds 1–5, neural
sequence seeds 1–5, and CoF seeds 1–5. They are not run during this corrective
work.

## Confirmed scheduler defect and correction

The prior runner treated one temporarily busy authorized GPU as a global
availability failure. The console exclusion recorded physical GPU 2 at 14%
utilization against the frozen 5% selector threshold, even though other GPUs
could be eligible.

The corrected scheduler rereads inventory at each wave. It excludes only
devices that fail the selector and uses
`min(4, live idle GPU count)` concurrent jobs. Thus three, two, or one idle
GPU continues the pending queue without STOPPED state.

At zero idle GPUs it writes append-only WAITING and heartbeat evidence, polls
every 30 seconds, and resumes automatically when a GPU becomes idle. Only
exhaustion of the fixed 1,800-second wait budget creates mandatory STOPPED
state. It never terminates an external process.

## Continuation contract

The external continuation authorization must bind both source commits and
code hashes, the exact scheduler-only commit diff, unchanged config/frozen
data/SamplingPlan/v2.4/capacity provenance, both attempt_002 tree hashes, and
the 48/17 split.

On a future separately started full invocation:

1. all 48 terminal attempt_002 outcomes are verified and replayed read-only;
2. none of those jobs is trained, sampled, evaluated, cancelled, or rewritten;
3. only the 17 missing jobs are scheduled, with allocation floor
   `attempt_003`;
4. CTGAN seeds 4–5 remain pending despite C2 already being NOT_EVALUABLE;
5. finalization requires all 65 jobs to have one explicit terminal state;
6. the final checksum provenance references both preserved attempt_002 hashes
   and the 48/17 continuation counts.

Any config, DGP/benchmark, evaluator, model/generator, data, SamplingPlan,
endpoint, hard-guard, or statistics change rejects this exception.

## Regression coverage

Fake inventory and filesystem fixtures cover:

- one busy GPU with three idle GPUs;
- adaptive concurrency with two or one idle GPU;
- zero-idle append-only wait, heartbeat, poll, resume, and budget exhaustion;
- no STOPPED marker while at least one eligible GPU exists;
- exact read-only reuse of 48 terminal attempt_002 jobs;
- exact 17-job attempt_003 plan, including CTGAN seeds 4–5;
- allocator floor at attempt_003 without placeholder attempts;
- rejection of non-scheduler source changes;
- C2 remaining NOT_EVALUABLE after replayed CTGAN INVALID;
- all 65 terminal states before finalization and attempt_002 provenance in the
  final checksum manifest.

The fixtures do not call real `nvidia-smi`, CUDA, DGP, fit, or sample paths.
