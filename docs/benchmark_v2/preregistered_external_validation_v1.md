# Preregistered external validation v1

Status: source-only; execution is not authorized.

Infrastructure amendment: the scientific protocol below is unchanged. After a
confirmed host OOM in AMLSim TVAE, external CTGAN/TVAE transforms are fixed to
one synchronous in-process `DataTransformer` worker and may not overlap in
time. This changes resource scheduling only: algorithms, seeds, conditioning,
budgets, metrics, thresholds, and selection remain frozen. The failed TVAE
attempt and all completed artifacts remain append-only. Any continuation needs
a new authorization.

This protocol compares four generators on the already frozen AMLSim and
Sparkov `fraudTrain` sequence bundles. It does not alter either bundle, the raw
CSV, a controlled-benchmark artifact, or a v2/v3 result. No result from the
external validation or any test split was available when these rules were
written.

## Frozen inputs and model scope

The two immutable inputs are
`data/external_sequence_protocol_v1/frozen/{amlsim,sparkov}/attempt_001`.
Their corresponding materialization `COMPLETE.json`, artifact index, checksum
manifest, split manifest, transform state, and leakage audit must pass before
an execution authorization can be accepted. Plan and dry-run may read and hash
these JSON manifests and files, but may not open an NPZ array.

The models, in fixed order, are:

1. label-conditional empirical i.i.d. row sampling;
2. separate-class CTGAN;
3. separate-class TVAE; and
4. frozen non-v3 CoF-SeqGen.

The CoF baseline is the exact implementation at source commit
`99a445f6dc893a8c2240d950de4f92877cc07f8a`, with frozen v2.5 config SHA-256
`81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3`.
The model, adapter, and denoiser fingerprints in
`external_validation_v1.yaml` must match both that Git object and the current
file. The model architecture and frozen hyperparameters cannot be overridden.

AMLSim uses seed 31001 and Sparkov uses seed 32001. CTGAN/TVAE retain two
separate class models and their v2.5 shared baseline budget: CTGAN 10,000 total
updates, TVAE 20,000 total updates, each split evenly between classes, and a
7,200-second total training cap. CoF retains 20,000 requested updates and a
7,200-second training cap. Empirical i.i.d. has no learned updates.

The receiver embedding cardinality is obtained from the immutable, train-fitted
external transform state: 9,656 for AMLSim and 695 for Sparkov. This is a data
input dimension supplied to the existing dynamic embedding construction; it is
not an architecture change. PAD is 0 and UNK is 1. A noncontiguous vocabulary,
an incompatible amount/gap shape, or a source fingerprint mismatch blocks the
run rather than triggering an adapter change.

## Split boundary

`train.npz` is the only input to model fitting, transform/reference fitting,
and threshold bootstrap. `validation.npz` is opened only after the train-only
state has been frozen; its Y and lengths define the fixed, matched validation
sampling plan, after which its rows are used only for fidelity and coherence
evaluation. `internal_test.npz` and Sparkov `fraudTest` are rejected before path
or CSV access. Neither is named in an execution manifest's data hashes.

The frozen entity assignment, window IDs, and per-window transaction-membership
hashes are re-audited. Entity, window, or transaction overlap is fail-closed.
The transaction audit is hash-bound because this source-only phase does not
reopen raw CSV records.

## Train-only external reference

Controlled-benchmark five-guard numerical thresholds are not copied. For each
dataset, 1,000 pairs of entity-cluster bootstrap samples are drawn from the
training split only. Each pair is compared with the same metrics listed below.
The threshold is the 951st ordered value, with no interpolation, and the bundle
is written before validation is opened. The short-gap cutoff is the median
positive train-only gap representative (`gap_tau`).

Fidelity is class-conditional and reports:

- amount empirical KS for Y=0 and Y=1;
- gap-bin total variation for Y=0 and Y=1; and
- receiver total variation for Y=0 and Y=1.

Coherence is the absolute real/synthetic error in
`P(short gap AND receiver repeats its preceding valid receiver | Y)`, separately
for Y=0 and Y=1. Mask, padding, and declared train-fitted discrete support are
hard validity contracts. Fidelity/coherence thresholds are analysis references,
not substitutes for those structural contracts.

## Selection and test-unlock rule

For every hard-valid candidate, each metric is divided by its train-only
bootstrap threshold. Let F be the maximum of the six fidelity ratios and C the
maximum of the two coherence ratios. The fixed score is `0.5 F + 0.5 C`.
Candidates are ordered by score, then F, then C, then lexicographic candidate
ID. A candidate with any fidelity ratio above one is retained as
`FIDELITY_WARNING`; failure of a single all-pass gate does not permanently block
a subsequent test proposal.

An external validation selection may become eligible for a separate test
authorization only after all four preregistered models have terminal,
hard-valid, hash-indexed validation artifacts and one selected candidate. The
validation stage never authorizes test access itself. A new preregistration and
explicit authorization would still be required; Sparkov `fraudTest` remains a
separate temporal-robustness target.

## Row-shuffle negative control (design only)

No row-shuffle is executed in this phase. The preregistered negative control
permutes complete transformed rows `(amount, gap, receiver)` together within
each valid window, while preserving Y, length, mask, and the per-window row
multiset. This preserves row-level signal and removes event order, so a drop in
short-gap/receiver-repeat coherence isolates sequence information. Its fit and
any orientation are train-only; implementing or running it requires another
source and execution authorization.

## Prohibitions

This source-only preparation performs no GPU/CUDA query, adapter import, model
fit, checkpoint write, sample, validation metric calculation, internal-test or
Sparkov-public-test read, TSTR, privacy evaluation, or full run. It creates no
execution authorization and no runtime artifact.
