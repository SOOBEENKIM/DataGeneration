# CPAR execution outcome

The final fit completed all 128 full-batch-equivalent Adam updates in 5,675.61
seconds, with the native full sequences and all 688 outer training customers.
The last native loss was 2.9286013. Loss scales are not comparable to ARGN's loss
and this fixed epoch budget does not establish convergence. `provenance/cpar_history.csv`
contains every epoch; `provenance/cpar_fit.json` identifies the saved model hash.

The original process began native 147-context generation after saving the model.
It was stopped only at that stage, after verifying the 128-row history and saved
model SHA256. This is recorded in `provenance/cpar_generation_resume.json`.
No partially generated data from that attempt are used as a quality result.

The saved model's actual native context encoder confirmed the preregistered
support audit: RI (one customer), DE (one), HI (two) raised KeyError. The remaining
143 contexts were generated twice with the verified recurrent-cache and decoding
accelerator, retaining native variable lengths and all output rows. Neither
labels nor numeric values were repaired. The first draw took 23.88 seconds and
generated 17,883 events; the second generated 17,486 events. Detailed generation
metadata remain next to the parquet outputs in the local artifact folder.

The installed SDV/DeepEcho files and saved fitted model were unchanged. Native
continuous-target alignment and normalization conventions were preserved. The
earlier two stopped CPU speed attempts are retained under artifact `attempts/`,
and are not included as independent training repetitions.

The native support limitation and numerical-loss convention are implementation
qualifications on this baseline. The common-support cohort is reported separately
from the original 147-customer ARGN cohort. This is an exploratory one-fit CPAR
comparison, not a claim about an optimally tuned CPAR or all CPAR implementations.
