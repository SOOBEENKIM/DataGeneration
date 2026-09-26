# Fine-gap codec isolation

Adaptive diagnostic registered after inspecting the original roundtrip's <=5s
membership. Aggregate gap TV was small, but that statistic can conceal fine
conditional information loss. Native numeric AUTO selected bins starting at
0, 157, 344, ... seconds. A model receiving that first interval cannot distinguish
the fine values inside it through the encoded gap token alone.

Compare gap-only encode/decode with the frozen native gap codec against the
official TABULAR_NUMERIC_DIGIT codec, with value protection retained. The latter
uses an isolated copy of the exact original training/internal-check partitions;
change only the gap encoding metadata and run native analyze. Neither codec
uses outer validation to fit its statistics. No neural model is refitted.

For decode seeds 20261200..20261209, measure membership, intersection, fraud
counts and support for positive gaps <=5, <=60, <=300 and <=1800 seconds. Keep
all other columns unchanged. Exclude the undefined first gap. Record absolute
gap reconstruction errors as well. Count preservation alone does not establish
conditional-information preservation. Conversely, per-row reconstruction is
not itself the objective of synthetic generation; the argument concerns lost
within-bin distinctions. With only one <=5s validation fraud, these results
cannot establish a precise population-level fraud-rate error in that cell.

A digit-codec improvement here establishes representation resolution, not the
quality of a neural generator trained with the larger digit representation.
