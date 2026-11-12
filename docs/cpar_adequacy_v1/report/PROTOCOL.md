# CPAR adequacy comparison,2026-10-01

Official sdv1.38.0/deepecho0.8.1 using preserved cofseq Python/torch2.1.2 runtime.
ARGN uses its separate engine1.0.4/torch2.5.1 runtime; neither is upgraded.
Official PAR documentation checked2026-10-01: context is sequence-static,
default128epochs,segment_sizeNone,sample_size1,CUDA supported. This is PAR, not a GAN.
https://docs.sdv.dev/sdv/modeling/sequential-synthesizers/parsynthesizer

Same619optimization customers816341events and69internal customers100226events
as the fresh ARGN backbones. No internal target events enter CPAR preprocessing or
optimization. Same5transaction fields and4static fields, gap0 for first event during
fit (first gap removed for interval evaluation). Label categorical, native learned
length. Train128and256epoch checkpoints from ONE256epoch trajectory(seed20261021),
not2independent fits. This controls stochastic initialization for the budget contrast.
Use prior verified mathematically equivalent full-batch gradient microbatch and
sampling execution adapters; model/loss semantics and once-per-epoch optimizer step
unchanged. Hardware/runtime differences are reported.

At both budgets generate2internal draws20261031/32 on supported supplied static
contexts. Never provide true future counts/labels. Native unseen-context exclusions
are logged per customer; unlike historical outer-train143/147 result, this stricter
optimization-only fit can have different support. Other full models must use the
SAME supported IDs in direct tables; full147scores remain separately labeled.

Select mean of normal log-gap W1 and onset-personal log-gap W1 across2internal draws.
Undefined onset distribution=>infinite selection cost; both undefined or exact tie=>128.
Keep missing support visible and do not impute a flattering score. Budget selection is
an adequacy check, not a claim of globally optimal CPAR. No further budget search is
authorized by these results automatically. Selected checkpoint generates2development
draws. Development outcomes do not choose budget. Test remains sealed. No push.
