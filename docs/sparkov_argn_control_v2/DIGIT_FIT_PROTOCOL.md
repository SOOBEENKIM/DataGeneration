# Adaptive official gap-encoding control

Registered after the codec-only results and before DIGIT neural fitting. Native
binned decoding changed fine-gap condition membership; official DIGIT preserved
the tested thresholds. Test whether that resolution survives neural fitting.

Use the exact original internal splits and the same two fit seeds. Reuse frozen
context statistics and all non-gap target statistics; only the gap codec changes
to official TABULAR_NUMERIC_DIGIT with native value protection. The independent
analyze call changed state rare-category handling, so its context statistics are
archived and replaced with the original frozen context statistics before encode.
Require all other target column statistics to be identical.

Use the same cap-relaxed (at most 100 epochs, native early stopping, 120-minute
ceiling), Medium, default window/batch/optimizer settings. Native model-size
heuristics and the number of numeric subcolumns may change with the encoding;
report those differences. This is a codec-configuration control, not an isolated
test of numerical precision at equal parameter count or equal loss weighting.
Loss magnitudes across these encodings are not directly comparable.

Generate both validation and common synthetic contexts with both original draw
seeds. Also test category-first on frozen DIGIT checkpoints, validation contexts
only, with the same draw seeds and planned-length equality check. No new loss,
architecture or proposed model is introduced. Report all metrics; do not assume
the larger representation improves learned generation because roundtrip was exact.
