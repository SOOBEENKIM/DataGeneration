# Frozen ARGN with simple transition references

Registered before generating the new datasets, 2026-09-27. This is a known-method
control and causal intervention on one generator component, not a new-method claim.

- Parents: both existing B_event_label_first checkpoints (20260930, 20261001).
- Reference modes: train-only two-state Markov and duration table. First event uses
  the optimization customers' first-label frequency with Jeffreys .5 smoothing.
  Transition rates use optimization events only; duration ages 1–20 and 21+.
  Unseen age cells fall back to the corresponding previous-state rate.
- No neural retraining. Only label logits change before native stochastic sampling.
  Sampled labels condition category, gap, merchant and amount, then enter the history.
  State comes from generated past events. No validation label, actual customer length,
  forced class ratio, forced run ending, or retrospective relabeling is supplied.
- All four controls generate both seeds 20261011/20261012, 147 development contexts.
  Report every draw, comparison with its frozen parent, and the real development set.
- Keep all earlier numeric/episode/conditional/normal/diversity/length metrics.
  A persistence improvement alone is not success on the overall research objective.
- The statistical label head deliberately omits personal and length conditioning.
  Its first-label and length joint distribution may be wrong. That is a measured
  limitation, not a claim that the full target joint distribution is solved.
- Later learned heads must beat both simple controls on joint conditional relations.
  No test event attributes are opened. Existing checkpoints/outputs remain unchanged.

Validation before launch: strict-past ages, held-out teacher probability agreement
with independently tabulated transition references, and unchanged non-label logits
on identical teacher inputs. Generation stays in separate workspaces.
