# CS-SAF v2: observed-context-separated bilinear route

Registered 2026-09-16 at `d8302e3`, **before v2 implementation or training**,
after the v1 checkpoint forensics. State at registration:
**PREREGISTERED, NOT IMPLEMENTED, NOT TRAINED**.
This document and the immutable YAML describe that registration; the
[current research status](STATUS.md) records subsequent execution.
Machine-readable contract: [cs_saf_revision_v2.yaml](../../configs/benchmark_v2/cs_saf_revision_v2.yaml).

This is a new exploratory candidate. V1 remains FAIL; its 50% and confirmatory
stages are not resumed. The design has seen the v1 validation pilot result and
train-only forensics, so a future v2 pilot is not independent confirmation.

## Evidence and bounded hypothesis

The [forensic report](checkpoint_forensics_v1_report_2026_09_16.md) finds broad,
history-dependent null response, present in train and validation, arising in
the bilinear copy route rather than the fresh-mark probability. At the failed
25% B1 checkpoint, descending the active-context loss through the shared direct
route locally raises null response. Null-context self-descent also raises it,
and the two context gradients have cosine -0.0299, close to orthogonal. Thus
historical harmful gradient conflict is not established as the sole cause.

The testable hypothesis is narrower: **allocating separate direct bilinear
operators to observed contexts can retain active response while reducing
unwanted cross-context influence under the same total parameter budget**.
At fixed history/static/gap features this removes the measured direct parameter
path exactly. Shared features can still transmit influence, and a null context
can learn its own spurious gap response. Success is not guaranteed.

## One changed factor

V1 uses one rank-32 operator for both observed static labels. V2 replaces that
operator with two rank-16 operators selected by the observed static category.
No label is designated active by the model; both operators have the same form,
initialization distribution, objective treatment and training access.

Let `c_t=[h_t; E_static(s)]` have dimension 136, `e_k` dimension 32 and
`s=static_code-3` in {0,1}. Define:

```text
u_s(c) = tanh(Wc[s] c + bc[s])        # 16 dimensions
v_s(e) = tanh(Wg[s] e)               # 16 dimensions, no bias
r_s(c,e) = sum(w[s] * u_s(c) * v_s(e)) / sqrt(16)
q = sigmoid(base(c) + r_s(c,e))
```

Only the selected operator is evaluated for each entity. Reject unknown/missing
static codes under the controlled-data schema. `w[s]` starts at zero for both
contexts; no oracle target, known active-label mask, kappa input, sparse gate,
new penalty, resampling or postprocessing is introduced.

All common modules remain unchanged: shifted GRU 128, static embedding 8,
gap/mark embeddings 32, support-aligned unordered gap decoder, shared base copy
logit, shared fresh-mark head, and Gaussian value decoder. Repeat probability
remains `q+(1-q)*p_new(previous)`.

Direct-route parameter count is exactly preserved:

```text
v1: 32*(136+1) + 32*32 + 32 = 5,440
v2: 2 * (16*(136+1) + 16*32 + 16) = 5,440
```

The complete controlled model remains **133,549 parameters**. CS2-U1 and
CS2-B1 have identical parameter keys/shapes/initialization. They differ only
in the same ordinary/balanced objectives as v1. Per-context rank falls from
32 to 16: this is an intentional capacity-allocation tradeoff. A comparison
with v1 cannot isolate removal of sharing from per-context rank reduction.

For reproducible initialization, construct fresh v1 common tensors with model
seed 20260930 and the same tensorizer. Keep these common tensors bitwise equal;
discard the old route tensors. Draw the new bank on CPU with a separate
generator seed 20261010, in the order Wc, bc, Wg. Wc and bc use uniform
[-1/sqrt(136),+1/sqrt(136)], Wg uses uniform [-1/sqrt(32),+1/sqrt(32)].
Set both w rows to zero. Never initialize from a trained v1 checkpoint.

## What is held fixed

Keep v1 data, original entity splits, all four prevalences, both kappas,
train-only tensorizers and observed support atoms. Verify the immutable data/
oracle manifests. Existing oracle evidence is applicable because the data and
observable gap representation do not change; no oracle is supplied to training.

Keep base gap+mark+value NLL, the fixed-global-denominator balanced repeat
auxiliary with coefficient 1, AdamW lr=.001, weight decay=1e-5, batch 512,
gradient clip 1, FP32, max 50 epochs, patience 5, no scheduler. Select the best
checkpoint by global validation base NLL excluding auxiliary. This preserves
the existing total route-loss weighting issue; do not silently normalize it.

Use seed 20260930 for both candidates and an identical entity shuffle stream.
Use the same train-only empirical generation plan, 2,048 entities and sampling
seed 20260930. V1 historical results remain descriptive comparisons; do not
retrain v1, switch the primary candidate to U1, or select a winner post hoc.

## Required CPU and structural checks

Before any v2 scientific fit, verify:

- exact parameter budget, pair initialization, and common-tensor initialization
  matching fresh v1;
- own-context gradients reach the selected route bank after its zero-w start;
- at fixed shared features, gradients of one context's route output with
  respect to every other direct bank parameter are exactly zero;
- jointly permuting known label codes, static-embedding rows and bank rows
  preserves outputs, showing no privileged active-label identity;
- strict past, reserved vocabulary, observable-repeat likelihood, support,
  fixed auxiliary denominators and zero-gap invariance;
- the existing two-run CPU gate: 64 train/32 validation entities, 25 epochs,
  loss drop >=10%, identical histories/best-state tensors, reload/generation.

Cross-bank derivative zero is only a direct-route property. It does not apply
to shared GRU, static/gap embeddings, base copy or fresh-mark parameters.

## Registered pilot and stop rule

Primary: **CS2-B1**. Comparator: **CS2-U1**. At each pi=.05/.10/.25/.50, train
both at kappa 0 and 1, four fits per stage, at most 16 fits. Process in that
order and stop after the first primary-candidate failure.

Use all validation nonfirst histories and all 31 train gap bins. Average
histories within entity, then entities within context. For both latent copy
and observable repeat, require active mean range >=.05, each of the three null
means <=.05 and active >=2*maximum null. Require matched zero-gap max <=1e-8,
zero generated support violations, finite values and no reserved marks.

Central-bin diagnostics, group-averaged curves, confidence intervals or rounded
values cannot replace the full-support criteria. No rank/loss/seed sweep,
threshold relaxation or fallback candidate is allowed after observing results.
Technical failures remain distinct from scientific failures and are preserved.

## Execution boundary and interpretation

This registration does not report v2 implementation or learned success.
The next implementation must record its own clean source commit, pass the CPU
gate on that same source and retain outputs under `artifacts/cs_saf/revision_v2`.
Keep all v1 checkpoints and forensic arrays immutable.

Even a v2 pilot PASS only establishes this response/selectivity preflight under
one model seed. Exact intervention-error/conditional-TV aggregation and the
train-only noninferiority-calibration audit remain prerequisites to a separately
specified five-seed study. Real-data and held-out tests remain unexecuted.
