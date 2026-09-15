# Benchmark v2.5 evaluator/runner forensic correction

Status: corrective implementation only. No full retry, five-seed run, frozen
data regeneration, CPU/GPU model execution, threshold change, or model tuning
is authorized by this document.

## Confirmed defect

The frozen config
`configs/benchmark_v2/full_v2_5.yaml` has SHA-256
`81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3`.
It declares `support_diagnostic_bins=[4,8]` and
`diagnostic_bins_cannot_change_primary_decision=true`. Preregistration
amendment 1, SHA-256
`16bf6df52dfa07a85693e0e953c3e965b9e1319f67a95a1c70d40c2c6cf4a20c`,
likewise states that 4-bin and 8-bin outputs cannot enter or change the
primary decision.

At source commit
`a17d386be67c0d6ee78dc4382ceba6972c8de060`,
`eval/full_evaluation_v2_5.py` nevertheless inserted
`confirmatory_support` into `hard_guards`, required
`support["all_bins_valid"]`, set the seed to `INVALID`, and nulled the
top-level `association_recovery_error` when that diagnostic failed. This was
an evaluator implementation defect, not a protocol, DGP, generator, model,
threshold, or training-budget defect.

The corrective evaluator now restricts `hard_guards` to canonical
mask/zero-padding, train discrete support, the frozen C0/C1 reference
contract, and the preregistered direct row-marginal guard. Four-bin and
eight-bin results live under `diagnostics.support_diagnostics`;
`diagnostics.confirmatory_support` preserves the eight-bin view. A diagnostic
support failure no longer nulls the numeric primary error or prevents
`COMPLETE.json`. The finalization payload separately reports, for both bin
counts, all-bin validity, dropped-bin score, occupancy-penalized score, and
invalid-bin count.

## attempt_001 empirical i.i.d. forensic evidence

All five empirical i.i.d. seeds passed every true hard guard. Each was marked
`INVALID` solely because the old evaluator treated the eight-bin support
diagnostic as hard. The old top-level primary error was null, while the
already-computed numeric error remained preserved inside
`evaluation.json["association"]["association_recovery_error"]`.

| Seed | Row guard | 8-bin valid | Invalid bins | Dropped-bin | Occupancy-penalized | Preserved numeric primary error | evaluation.json SHA-256 | INVALID.json SHA-256 |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | PASS | false | 2 | 0.029809794171270398 | 0.2798097941712704 | 0.072282300239678 | `698a364487528c307b50d8acd4035b6373487c80f5af50ba3a103160e739d774` | `d0ea8cfe17972f244efc7d9a4bbefa6baccfe52f5863a823e92f034cc8834eb3` |
| 2 | PASS | false | 2 | 0.03337595502612658 | 0.28337595502612656 | 0.07233201753908129 | `b63ccdb8a57ef86d7ef516755982258d36b365d8226ef246cf6855f2c865d2c6` | `d0ea8cfe17972f244efc7d9a4bbefa6baccfe52f5863a823e92f034cc8834eb3` |
| 3 | PASS | false | 2 | 0.03069406168073956 | 0.28069406168073957 | 0.07198781098326011 | `7c0e0633f98b339d804e9dc434fec0dc2036e38d224136dd198a956d11dc292f` | `d0ea8cfe17972f244efc7d9a4bbefa6baccfe52f5863a823e92f034cc8834eb3` |
| 4 | PASS | false | 2 | 0.029109096120244104 | 0.2791090961202441 | 0.07244387543637992 | `01275764326e97a4a8ba2e37e3e4ee420193e9dbfaa9ecfabd56b865d710caa8` | `d0ea8cfe17972f244efc7d9a4bbefa6baccfe52f5863a823e92f034cc8834eb3` |
| 5 | PASS | false | 2 | 0.031181130301796627 | 0.28118113030179664 | 0.07072906024419177 | `bdee57a8fe8511527d65761f03688e885e2824cca78f347e4bb04e1947d81d07` | `d0ea8cfe17972f244efc7d9a4bbefa6baccfe52f5863a823e92f034cc8834eb3` |

The immutable samples, checkpoints, manifests, metrics, evaluations, and
terminal-marker hashes are listed in
`docs/benchmark_v2/attempt_001_preservation_manifest_v2_5.json`. No
attempt_001 file was modified to apply this correction.

## Row-marginal findings remain hard

No row-marginal threshold or implementation was changed.

- `plug_in_hsmm` seed 1: gap KS
  `0.008506986828145102` exceeded the frozen threshold
  `0.006387882975686154`. Its `evaluation.json` SHA-256 is
  `af518f96f9189c1a18468b15d587e12574c648e4baf0fb70b6a3c88ea1cc1696`
  and `INVALID.json` SHA-256 is
  `d7da15f2adbf9c22cf39cf931c16d273f02d808b59e94b787848dbe2c184adad`.
- `ctgan_separate_class` seed 1: amount standardized effect
  `0.30166068441477434 > 0.0363693454591819`, amount KS
  `0.1820394163376433 > 0.006081138155655141`, and gap KS
  `0.06645773422994083 > 0.006387882975686154`. Its
  `evaluation.json` SHA-256 is
  `0e564d32af3099ed91cd1eca2aab0fe717a4b853dde591a01c20a5d3fd8c0841`
  and `INVALID.json` SHA-256 is
  `9f5d99d383bd3e51612f60344634465b8c98db0addbaf47f85b550902e2b6b94`.

These remain hard INVALID findings. They are not demoted or tuned in response
to attempt_001.

## Runner policy and stopped-run context

The run-level regression fixture materializes all 65 planned terminal cells.
When empirical i.i.d. seed 1 is INVALID because of a row-marginal hard guard,
all unrelated jobs still reach explicit terminal states, C2 becomes
`NOT_EVALUABLE`, and append-only finalization creates
`FINAL_COMPLETE.json`. An INVALID seed is never removed as NaN from a mean.

The preserved real attempt_001 did not stop because a primary comparator was
INVALID. It stopped on the independent global GPU-availability contract:
an authorized GPU became unavailable before a wave; no external process was
terminated. Its `STOPPED.json` SHA-256 is
`d583a1a23174a362e51ca713e7c4cdf84c5e2547e7776ed33bc59483e15765d1`.
At that point 48 attempts had terminal markers: 35 COMPLETE, 9 INVALID, and
4 CANCELLED. This global stop correctly left no final aggregate.

## Authorization required for any later execution

A later execution requires a new, explicit user authorization that:

1. pins the corrective commit and its new relevant-code SHA-256;
2. pins the unchanged config SHA-256
   `81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3`;
3. pins and revalidates the unchanged frozen data manifest, split content
   hashes, shared SamplingPlan, v2.4 gate artifacts, and capacity provenance;
4. explicitly preserves all attempt_001 paths and authorizes only new
   append-only attempts selected by the manifest/code-hash contract;
5. requires the full-run shell to pass the frozen GPU identity and
   availability checks before any GPU job starts; and
6. makes no DGP, endpoint, support-diagnostic, row-guard threshold, model,
   architecture, budget, or C2-rule change.

Until those conditions and a separate explicit authorization are satisfied,
full retry, five-seed execution, frozen-data regeneration, CPU/GPU baseline
execution, and GPU training remain prohibited.

## Corrective provenance and verification

The corrective relevant-code SHA-256 is
`e64f69ffe1a5606f4eabe6cdc590ae0abae5f0c676b5f5035949cb8c40a8deeb`.
The config remains byte-identical at
`81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3`.
The exact corrective commit is the commit containing this report.

Read-only post-change verification found:

- attempt_001: 983 files, 5,039,876,507 bytes, tree SHA-256
  `fc7f37a7e7c359fe7727618f888e2f953f7addc1e8c1d9587b0d8a5399816e8f`;
- new `attempt_002` directories: 0;
- frozen data manifest SHA-256
  `b2529f00bae2e534f90805db6cebdf7f19ee93117f34f71015bc6753a9223e05`,
  with all train/validation/test file and content hashes matching;
- v2.4 artifact index SHA-256
  `c99c79b501ba264b01f1ef54f1bb7fa1c40446c865d95b6078fad6fd47fec08a`
  and gate report SHA-256
  `53fc93c3650e5e6d03f9a085c74a5ad86ef2862f7a918668bce817500319ee14`,
  matching the indexed gate hash; and
- capacity-preflight index SHA-256
  `d0c65a7e1436faae1b6701df1d88f3bb3e4f0321ff005d5ce0db9d237639fcb8`,
  with every indexed artifact hash matching.

Validation commands and results:

```text
pytest focused evaluator/runner/artifact/statistics/row-guard suites
45 passed

pytest -q
183 passed, 19 warnings

python -m compileall -q benchmarks eval experiments generators models scripts tests
PASS

git diff --check
PASS
```
