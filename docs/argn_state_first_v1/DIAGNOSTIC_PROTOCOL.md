# Adaptive diagnostic after the first evaluation

2026-09-27. The first comparison found severe fraud over-generation in both arms,
including at the first event where S is identically zero. This document registers
follow-up checks after seeing that failure; it is not part of the original
preregistered comparison or a confirmatory result.

1. Audit the native per-customer mean transaction loss. Count label prevalence in
   optimization targets with ordinary event weights versus the actual 1/L_i row
   weights within each customer, including first-event and position strata. Keep
   length-head losses separate. This quantifies an objective/data imbalance; it
   does not by itself prove the cause of generated errors.
2. On each frozen checkpoint, score real development first rows with real length
   tokens (teacher diagnostic). Re-encode each saved generated first row and score
   its fraud probability, then replace exactly one field group at a time by the
   corresponding real customer's value: gap, merchant, amount, category, or length.
   Also use all-real current fields with generated lengths. These are matched-ID
   conditional sensitivity probes, not realizable free generation, improvements,
   or valid causal interventions on the data process. Re-encoded output tails and
   lengths may differ from latent sampled tokens; predictions need not exactly
   reproduce the previous Bernoulli draws.
3. Preserve original fitted weights and generated data. Any subsequent objective
   intervention must be a separately named fresh fit with its own registered
   config, keeping the original failed/mixed comparison visible. Do not interpret
   B+S improvement over a severely miscalibrated B as a sufficient contribution.
