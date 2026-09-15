# CoFSeqGen-SAF source protocol and canonical data contract

## Status and exact boundary

This document creates the new `cof_seqgen_saf` research family and its
model-agnostic data contract. The expansion is **Support-Aligned
Autoregressive Factorization**. Its current state is:

> `PREEXECUTION_SOURCE_IMPLEMENTED_METRIC_AUDIT_COMPLETE`

The user first authorized acquisition, raw-data inspection, dataset adapters,
and entity-level split materialization, then explicitly authorized the next
boundary: metric validity audit, baseline compatibility design, and
CoFSeqGen-SAF plus ablation source implementation. The latter boundary was
completed on 2026-08-26. It does not extend to external baseline installation,
GPU queries, canonical model fitting, trained-model sampling,
validation/test evaluation, or empirical claims. Runtime data are written
only under the SAF namespace; every older family and artifact remains
immutable.

## Family separation

This is not an H1 retry and does not alter any existing artifact.

- `cof_hcmttpp_v2 H1` remains permanently frozen at
  `STOP_COF_HCMTTPP_V2_FAMILY_H1_GATE_FAIL`.
- `cof_zdh_v3 D1` remains `SPECIFIED_NOT_IMPLEMENTED` under its own source
  config. SAF records it only as a planned decoder-ablation reference.
- Existing non-v3 CoF, v3, C1, H1, and D1 source, result, forensic, terminal,
  and artifact files are read-only parents or baselines. SAF uses new file and
  future artifact namespaces.

## Research question

> Does support-aligned gap modeling with explicit gap-to-mark dependency
> routing improve temporal-relational fidelity over current sequential tabular
> generators while retaining marginal fidelity, utility, and privacy?

`Autoregressive` means chronological, non-anticipative factorization; it does
not claim causal-effect identification.

## Scientific contribution encoded by the source design

The family is organized around four separable contributions:

1. characterize representation/decoder support mismatch and test whether it
   creates irreducible distributional error;
2. compare support-matched unordered and ordered gap decoders under a fixed
   history model;
3. test explicit within-event dependency routing rather than relying only on a
   shared hidden state;
4. evaluate the resulting complete model against current sequential tabular
   generators under temporal-relational primary endpoints and marginal,
   utility, and privacy constraints.

The planned non-anticipative factorization is

```text
p(events | static context)
  = product_t
      p(gap_t | past, context)
      p(mark_t | past, context, gap_t)
      p(value_t | past, context, gap_t, mark_t).
```

Only past valid events may construct the history. During likelihood training,
the observed current gap is teacher-forced into the mark head and the observed
current gap and mark are teacher-forced into the value head. During generation
only, generated current values are routed forward. This distinction prevents
the old error of injecting a sampled gap into a supervised mark likelihood.
The first event's missing gap is excluded from gap likelihood while its
mark/value still enter later history. Sequence termination remains
unimplemented; all primary generators receive the same train-only
parent-context and length plan.

The finite ablation family is fixed:

| Candidate | Gap decoder | Support aligned | Ordered | gap→mark route |
|---|---|---:|---:|---:|
| SAF-C0 | stable hurdle log-normal | no | no | no |
| SAF-U0 | categorical | yes | no | no |
| SAF-O0 | ordered hazard | yes | yes | no |
| SAF-U1 | categorical | yes | no | yes |
| SAF-O1 | ordered hazard | yes | yes | yes |

SAF-H1 contrasts C0/U0, SAF-H2 contrasts U0/O0 and U1/O1, and SAF-H3
contrasts O0/O1 with U0/U1 as replication. The history encoder, amount head,
optimization protocol, and support state are shared within each contrast.

## Untested hypothesis family

The hypotheses below are directions to be tested, not achieved results.

The `SAF-` prefix prevents confusion with the already frozen HCMTTPP candidate
named H1.

1. **SAF-H1 — support alignment.** Support-aligned gap decoding improves gap
   fidelity relative to a support-mismatched decoder under an otherwise fixed
   comparison.
2. **SAF-H2 — order-aware gap decoding.** An ordered hazard decoder outperforms an
   unordered categorical gap decoder under a fixed support representation.
3. **SAF-H3 — dependency routing.** Explicit generated-gap-to-mark conditioning
   improves recovery of gap/mark joint dependency relative to shared-history
   conditioning alone.
4. **SAF-H4 — complete model.** The eventual full CoFSeqGen-SAF model outperforms
   current sequential tabular generators on a frozen temporal-relational
   fidelity endpoint family.
5. **SAF-H5 — constrained quality.** Its marginal fidelity, downstream utility,
   and trajectory-aware privacy remain within independently frozen
   non-inferiority margins.

The exact metric family is now fixed by
`configs/benchmark_v2/cof_seqgen_saf_metric_audit.yaml`. Its primary
temporal-relational endpoints are gap-bin-conditioned mark TV,
short-gap/repeat-curve L1, and gap/repeat mutual-information error. Marginal,
structural, and privacy diagnostics remain mandatory. Thirty-one
entity-disjoint train-only pseudo-splits calibrate q95 marginal/structural
non-inferiority margins. Known marginal-preserving corruptions audit power.
No candidate output, validation event, or test event entered this procedure.
Result-based endpoint switching remains forbidden.

Utility is frozen as next-gap and next-repeat TSTR with fixed downstream
models and sample caps; it is compared pairwise against the real-data upper
benchmark and generator baselines. Privacy uses actual trajectory/ngram copy
exposure guards and paired baseline comparison. These are not converted into
permissive real-vs-real q95 margins.

## Study data and acquisition boundary

| Dataset | Scientific role | Current acquisition state |
|---|---|---|
| Controlled coupling DGP | mechanism identification | generator dependencies hash-registered; paired κ=0/1 cells materialized |
| AMLSim | complex financial simulation | hash-verified source acquired and materialized |
| Sparkov | external financial simulation | `fraudTrain` acquired and materialized; `fraudTest` excluded |
| Berka | real irregular financial sequences | pinned public revision acquired; full 4,500-account view materialized |
| H&M | real irregular retail sequences | hash-pinned CSV sources acquired; deterministic 10,000-customer view materialized |
| Citi Bike | optional cross-domain robustness | official 2016-02 archive acquired and materialized |

The acquisition registry records source, version, license, acquisition date,
raw file names, byte sizes, and SHA-256 values before an adapter materializes
canonical frames. H&M images are explicitly excluded. Sparkov `fraudTest` is
not pooled into the SAF development source.

The controlled DGP has 45,645 entities per cell so that a 70% training role
retains the prior powered count of 31,951. The κ=0 negative-control and κ=1
full-coupling cells share entity labels, lengths, and split identities. Oracle
gap/receiver states are stored separately from canonical model inputs.

The Berka comparison projection matches TabDiT's published five parent fields
(district, frequency, city, region, account creation time) and six child
fields (time, amount, balance, type, operation, and k-symbol). The full
4,500-account dataset is primary. A deterministic nested 900-account subset is
also emitted, but neither TabDiT nor Seq2Synth publishes the exact test IDs.
It is therefore marked **not comparable** to their 900-account split; matching
the entity count alone is not evidence of matching row identity. TabDiT's
published maximum sequence length of 50 also explains why Seq2Synth's child
row count is much smaller than an untruncated 900-account subset.

Sparkov's supplied `unix_time` disagrees with the readable
`trans_date_trans_time` on every acquired row. The latter is used as the
timestamp authority and the disagreement count is persisted in the audit.

## Canonical entity-sequence representation

Every future adapter must produce three frames under schema
`cofseqgen-saf-canonical-entity-sequence-v1`.

### `static_context`

Exactly one row per `entity_id`. Additional static columns are declared by the
dataset-specific schema and may be empty. A fraud label is not mandatory;
binary `y_entity` is therefore not part of the common contract.

### `events`

The mandatory ordered columns are:

| Column | Contract |
|---|---|
| `entity_id` | grouping identity; provenance only and forbidden as model input |
| `event_id` | non-missing and unique within entity |
| `event_index` | contiguous `0..L-1` in stored order |
| `timestamp` | finite canonical units and nondecreasing within entity |
| `gap` | first event is missing; later values equal timestamp difference |
| `receiver_or_mark` | nullable categorical event mark |
| `amount_or_numeric_value` | nullable numerical event value with at least one finite observation |

Dataset-declared auxiliary numeric columns and then auxiliary categorical
columns follow these columns. Missing marks remain source missing; they are
not converted to an unknown category in the canonical frame.

The first event has no preceding interval and therefore its `gap` is `NaN`.
Encoding it as numeric zero is forbidden. Zero is valid for a non-first gap
when two observed events share a timestamp.

### `entity_splits`

Exactly one row per entity with columns `entity_id, split`. Permitted split
roles are exactly `train`, `validation`, and `test`, all non-empty. The entity
sets in the three frames must match exactly.

The split is deterministic 70/15/15. Where feasible, the pre-transform
stratum is declared static context crossed with a coarse raw sequence-length
bin (`1`, `2-3`, `4-7`, `8-15`, `16-31`, `32-63`, `64-127`, `128+`). Sparse
cells are merged deterministically from the most specific length/static cell
to coarser length and finally static-only or global strata. Raw event counts
are structural metadata, not a learned representation. The split is produced
before fitting any support, vocabulary, quantile, normalization, imputation,
truncation policy, or model. Entities cannot cross roles.

## Missing, unknown, and padding semantics

Every later materializer reserves:

| State | Code |
|---|---:|
| PAD | 0 |
| UNK: observed non-missing value absent from train vocabulary | 1 |
| MISSING: absent in the source, including first-event gap | 2 |
| First learned category or gap-support state | 3 |

These states cannot be merged. Padding is outside a valid trajectory;
source-missing is inside a valid event; unknown is an observed value not fitted
from train. Entity IDs are never encoded as model features.

## Train-only transformation provenance

Future gap support or bins, categorical vocabulary, numerical normalization,
quantiles, imputation state, top-K/tail hierarchy, and any representation
selection must be fitted only from events whose entities are assigned to
`train`.

The canonical implementation creates provenance containing:

- canonical schema hash;
- entity-split assignment hash;
- fit role, which must equal `train`;
- exact train entity count and typed-identity hash;
- exact train event count and typed `(entity_id,event_id)` hash.

Validation or test participation, a changed split, or a changed schema makes
the provenance invalid. Later materialization metadata must persist this
provenance beside encoded arrays.

## Baseline compatibility registry

The primary sequential tier is TabularARGN, TabDiT, CPAR, REaLTabFormer, and
the empirical sequence sampler. CTGAN, TVAE, and GaussianCopula are flattened
marginal controls and cannot be presented as equal sequential competitors.
non-v3 CoF and CCMTPP C1 are frozen lineage context; HCMTTPP H1 remains a
stopped failed-gate context and is never retrained. TabPFN is not a generator
baseline; it may only be used as a downstream utility evaluator.

All primary sequential systems receive the same child-conditional task and
the same parent context/length plan sampled only from train. Raw, unrepaired
output is primary. Any Seq2Synth-style temporal repair or native end-to-end
generation is a separately labelled secondary analysis and cannot rescue or
be pooled with the raw primary result. The compatibility contract performs no
third-party import or installation.

## Endpoint and test-access policy

- Primary family: temporal-relational fidelity.
- Required constraints: marginal fidelity, utility, and privacy
  non-inferiority.
- Development may use controlled and validation roles only under a later
  protocol.
- The held-out test role is forbidden during hypothesis exploration, model
  selection, threshold construction, representation fitting, and checkpoint
  selection.
- No superiority, non-inferiority, SOTA, coherence-recovery, or privacy claim
  is authorized by this source-only package.

## Implemented source artifacts

- Config: `configs/benchmark_v2/cof_seqgen_saf_source_protocol.yaml`
- Canonical contract: `data/cof_seqgen_saf_contract.py`
- Contract tests: `tests/test_cof_seqgen_saf_contract.py`
- Acquisition registry: `configs/benchmark_v2/cof_seqgen_saf_acquisition.yaml`
- Acquisition command: `scripts/acquire_cof_seqgen_saf.py`
- Dataset adapters: `data/cof_seqgen_saf_adapters.py`
- Dataset config: `configs/benchmark_v2/cof_seqgen_saf_datasets.yaml`
- Materialization command: `scripts/materialize_cof_seqgen_saf.py`
- Integrity verifier: `scripts/verify_cof_seqgen_saf_data.py`
- Adapter tests: `tests/test_cof_seqgen_saf_adapters.py`
- Metric implementation: `benchmarks/cof_seqgen_saf_metrics.py`
- Metric audit config: `configs/benchmark_v2/cof_seqgen_saf_metric_audit.yaml`
- Metric audit runner: `scripts/audit_cof_seqgen_saf_metrics.py`
- Train-only audit result: `artifacts/cof_seqgen_saf/metric_validity_audit_train_only.json`
- Baseline compatibility contract: `generators/cof_seqgen_saf_baselines.py`
- Baseline compatibility config: `configs/benchmark_v2/cof_seqgen_saf_baseline_compatibility.yaml`
- Model and five ablations: `models/cof_seqgen_saf.py`
- Model config: `configs/benchmark_v2/cof_seqgen_saf_model.yaml`
- Pre-execution report: `docs/benchmark_v2/cof_seqgen_saf_preexecution_implementation_report_2026_08_26.md`

Acquisition and canonical entity-level materialization are complete for every
declared source. H&M images and submission files were not downloaded, and its
credential remains outside the repository. The source, CPU structural tests,
and train-only metric audit are complete. External baseline installation,
canonical tensorization/runner implementation, GPU training, model sampling,
and validation/test evaluation remain separate boundaries.
