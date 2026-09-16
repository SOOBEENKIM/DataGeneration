# Execution amendment 02: equivalent CPAR loss vectorization

2026-09-17, before any scientific CPAR fit or CPAR outcome. The user requested
full prevalence and external comparisons even on internal scientific failure.
The installed DeepEcho0.8.1 `_compute_loss` loops over every sequence, channel,
and epoch. A synthetic loss/backward microbenchmark (not a scientific fit)
measured0.96s for1024 sequences/two numeric+one categorical channel on CPU.
With31,951 train sequences and128 epochs×40 fits, Python/kernel overhead is
material. Do not reduce epochs, data, trials or generation counts to hide this.

Use `experiments/cs_saf_cpar_loss.py` to batch exactly the pinned loss expressions.
The upstream source checksum is checked before installation. Patch only the
PARModel class loss method in a context manager during `wrapper.fit`, restoring
it even on exceptions. Keep the SDV preprocessing, model architecture, optimizer,
learning rate, final128-epoch selection, sampling and all evaluation unchanged.
Model artifacts continue to load with the ordinary upstream PARModel class.

Preserve the pinned implementation's unusual N-squared normalization and
continuous-channel last-length target slice exactly. In particular, do NOT
silently replace that slice with the first-length targets. This alignment
convention is a limitation of the pinned CPAR comparator and should be reported;
CPAR results do not establish universal performance of PAR or other implementations.
Floating-point reduction order changes, so training is algorithmically equivalent
within tested tolerance, not promised to be bitwise identical.

Tests cover float64/float32, variable lengths, zero right-padding, continuous,
count, categorical/ordinal fields and meaningful unnormalized loss gradients.
CPU synthetic checks at n3/37/1024 showed exact equality of those gradients,
float64 loss differences <=1.9e-15 and float32 loss differences <=2.4e-7.
GPU regression and end-to-end toy CPAR smoke must pass before scientific fits.
No internal learning equation or scientific criterion is changed.

Pause only our own dispatcher, let all in-flight jobs finish, preserve its
progress/logs under `performance_amendment_2/`, and resume with all completed
fit/audit artifacts. Repeat same-source CPU gates in `cpu_v3` before resuming.
Previously collected internal outcomes and gradient diagnostics remain unchanged.
This is a disclosed implementation acceleration, not an extra baseline or a
posthoc scientific retry. Source changes and package checksums are published.
