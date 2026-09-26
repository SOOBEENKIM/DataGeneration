# Raw-scale metrics and official two-numeric DIGIT control

Registered after discovering the amount roundtrip issue and before these fits.
The original coarse-bin metric understated numeric error: validation fraud mean
amount changed from 521.5849 to 3712.3325 without neural generation. The final
native amount bin spans 539.94–15305.95; decoding within a wide bin changes amounts
while often retaining the exact same evaluation quantile bin. Coarse TV alone is
therefore insufficient for the requested financial-value fidelity.

Add original-scale mean, median, p90, p99, KS, Wasserstein and log1p Wasserstein
for amount and non-first gap, separately for all/normal/fraud transactions. Add
the continuous amount / prior-20 median ratio (minimum five past observations),
rather than only its coarse <=1 / <=3 / >3 categories. Report valid counts and
invalid values. These supplementary metrics are adaptive; retain the earlier
metrics and show their limitations explicitly.

Create a third codec configuration: official DIGIT for both gap and amount. Use
the exact frozen gap-DIGIT statistics, original categorical statistics, original
context statistics and original internal data split. Replace only amount stats
with a native DIGIT analysis of the same training partitions. The gap-only DIGIT
arm remains a separate control. First verify amount encode/decode on validation;
validation must not fit any codec. Native value protection remains active.

Fit seeds 20260928/20260929, the same cap-relaxed limits and remaining native
settings. Generate both context populations and both draws, plus category-first
on each frozen checkpoint with validation contexts. Model-size heuristics and
numeric subcolumn loss counts change with the encoding and must be disclosed.
These are existing ARGN encoding settings; no proposed architecture or custom
loss is introduced. No further encoding variants are planned in this stage.
