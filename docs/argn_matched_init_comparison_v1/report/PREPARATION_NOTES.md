# Registration and execution notes, 2026-10-02

The original C/P6/H code, heads and outputs are frozen. The matched init_only
runner calls the exact preserved statistical-fitting routine with redirected NEW
output paths, and uses the same frozen latest backbone/foundation for each seed.
It does not overwrite or retrain those ARGN/foundation models. The statistical
time head does not depend on the ARGN representation; matching the backbone still
matters to its generated labels, categories, amounts, counts and complete rollout.

Before registration, evaluator preflight caught that the independent time evaluator
returns 13 of the 15 registered metrics: merchant TV and merchant diversity were
previously supplied by the native evaluator. The new evaluator now recomputes
both directly from saved rows too. All 15 match a frozen C output within 8.33e-17.
No new fit or generation had started at that point; original metrics are unchanged.
Registration also excludes nonexistent virtual module filenames (for example
PyTorch's virtual _ops.py) from filesystem hashing. This is not a model error.

The manifest freezes 740 files including 620 previous hashes. Foundation arguments
match the original whole-pipeline runner exactly for each of the three seeds.
Native integration checks run again on every new time head before free generation.
New supplemental reporting documents are distinguished from the registered protocol.

Descriptive position-stratum tables include every observed position within each
bin. The canonical six position metrics retain the previous common-length support
restriction; these two summaries are deliberately separate and must not be mixed.
For customer-relative timing, a strict-past rolling median needs five prior gaps;
missing early or fraud-only support is retained as unavailable, not zero error.

MeLD500/2000 is already complete and remains a limited-budget pilot. Its poor
quality does not establish superiority to sufficiently trained MeLD. No new MeLD
job is started by this study. Latest-generator adequacy and unopened-test/second-data
verification remain distinct outstanding evidence.

Host process check at 11:48 KST confirmed tmux, dispatcher PID284628 and active
CPU fits PID284731/287530. Sandbox process lookup cannot see those host PIDs;
a sandbox ProcessLookupError is not evidence the persistent pipeline exited.
Use the host process namespace and STATUS/exit logs before any recovery.
