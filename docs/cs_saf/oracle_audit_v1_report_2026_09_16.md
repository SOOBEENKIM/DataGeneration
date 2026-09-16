# CS-SAF train-only oracle audit, 2026-09-16

Decision: **PASS — conditional signal is observable at the model's gap-bin
resolution. No CS-SAF model has been trained.**

Execution source commit: `34cdf667b5932cefbfa42dcc3e0443210b644749`.
The architecture and oracle criteria were committed before either full audit.
Both executions used the remote workstation's existing `cofseq` environment.
The relevant test suite passed **37 tests**, including 12 new oracle tests.

## Data and information boundary

For each kappa, the audit used all 31,951 existing canonical train entities:
30,406 label-0 entities and 1,545 label-1 entities, totaling 766,935 events and
734,984 nonfirst transitions. The observed minority proportion is about 4.84%.
Both cells share the original split assignment and train static records.

Only train event/static rows were materialized with Parquet predicates.
Validation and held-out event/static content were not accessed. Full split
membership and existing manifest metadata were read for provenance. Realized
oracle latent files were not opened. There was no GPU operation, fitting,
model sampling, or new prevalence-grid materialization.

The oracle integrates the known semi-Markov duration and emission laws using
strictly past observations, then conditions on the same 31 train-fitted gap
bins available to the mark route. It does not expose the hidden regime.
These are predictive input-response curves, not causal do(gap) effects.

## Results

Every metric below first averages histories within entity, then entities
within the specified context. Copy is the DGP mixture component; repeat is
the observable equality event including accidental equality on a fresh draw.

| Kappa | Label | Entities | Mean copy-response range | Mean observable-repeat range | Repeat calibration residual | Entity-cluster SE |
|---|---|---:|---:|---:|---:|---:|
| 0 | 0 | 30,406 | 0.000000 | 0.000000 | +0.000568 | 0.000485 |
| 0 | 1 | 1,545 | 0.000000 | 0.000000 | -0.003750 | 0.002144 |
| 1 | 0 | 30,406 | 0.000000 | 0.000000 | +0.000116 | 0.000484 |
| 1 | 1 | 1,545 | **0.413742** | **0.407278** | +0.001076 | 0.001975 |

All eight decision checks passed: material active response, small null
responses, selectivity, no-gap oracle invariance, sufficient context entities,
observable-repeat calibration, paired splits, and paired train static data.
The static codec and observed-atom support assertions also passed.

The active mean range exceeds the frozen 0.05 threshold. All three null
responses are exactly zero. The ratio to the largest null response is therefore
unbounded, not an arbitrarily floored finite value; JSON records a null ratio
with an explicit explanation. The no-gap oracle maximum response is zero.
Every absolute calibration residual is below the frozen 0.02 tolerance.

## Interpretation and limits

The existing data contain a substantial context-specific conditional signal
even after the current gap is binned. Thus a complete absence of observable
signal cannot explain the prior v6 failure. This does not identify whether
that failure arose from representation, optimization, objective weighting, or
their interaction, and does not show a finite GRU can achieve oracle accuracy.

The known-law oracle's null invariance is implied by the DGP; it is not evidence
that a learned model has null safety. The repeat-calibration check and
production-generator tests provide separate implementation checks.

Reweighting the active response at pi=0.05/0.10/0.25/0.50 yields aggregate
ranges 0.020687/0.041374/0.103436/0.206871. This merely shows how aggregate
weighting scales a fixed conditional signal. It is **not** a prevalence sweep,
learning curve, or proof of conditional mechanism dilution.

The DGP's gap population is continuous. Restriction to observed training atoms
does not prove exact population support recovery or an irreducible-error
theorem. No fidelity/utility/privacy superiority or noninferiority follows
from this audit. The legacy failed noninferiority calibration remains a
separate blocker for confirmatory interpretation.

## Reproducibility and artifacts

The complete compact result is [oracle_audit_v1_result.json](oracle_audit_v1_result.json).
Two independent process executions produced **byte-identical** JSON reports:

`10dafe41b5ef3040195ce3eb852db6da4987932ee31b2489c418cc6bd7d896ef`

Workstation runtime directories, relative to the CS-SAF worktree:

- `artifacts/cs_saf/oracle_v1/attempt_001/`
- `artifacts/cs_saf/oracle_v1/attempt_002/`

Each contains `audit.json` and a terminal `COMPLETE.json` referencing its hash.
The compact result includes execution source/config hashes, Python/NumPy/pandas
versions, filtered train-content hashes, canonical manifests and the original
split hash. Large data are not copied into Git.

At the execution source commit, from a clean checkout with the original local
canonical data available, run the following with a new output directory:

```bash
python -m scripts.audit_cs_saf_oracle \
  --canonical-root /path/to/cof_seqgen_saf/canonical \
  --output-dir artifacts/cs_saf/oracle_v1/new_attempt
```

The command refuses an uncommitted source tree or existing output directory.
Use the recorded source commit for an identical full-report hash; running a
later commit intentionally changes the recorded provenance.

## Next stage

Implement the frozen rank-32 bilinear model and matched objective ablations.
Create the new pi-grid data contract while preserving original entity split
membership. Complete capacity/gradient/leakage and CPU end-to-end/determinism
gates before the registered model pilot. The model pilot, five-seed study,
real-data comparison and final held-out evaluation have not been run.
