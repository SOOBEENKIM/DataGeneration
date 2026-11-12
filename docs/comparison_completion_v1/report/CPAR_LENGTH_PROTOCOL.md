# CPAR native length audit and shared synthetic-count sensitivity

Registered before additional generation. Three internally selected CPAR checkpoints
(20261021 prior,20261022/23 new) retain native free-length results as primary.
Inspect learned min/max length, optimizer sequence lengths and actual generated
length distribution. Run 2 additional development draws per checkpoint, providing
ONLY lengths from the matched already-generated C output to official sequence_length.
These counts came from a train-only learned length model, not real future records.
No C labels, gaps, merchants, amounts or past history are supplied to CPAR.
Label this CPAR+shared-synthetic-count, never native CPAR or exact paper reproduction.
Same supported143 customers, static conditions, model parameters and official
conditional sampler; no true-length floor, post-hoc filtering, or relabeling.
This sensitivity isolates consequences of learned termination/count differences;
it does not by itself establish the sole causal reason for CPAR's quality gap.
Report normal/onset/amount/fraud/prevalence/length and absent-tail support. If forced
synthetic counts improve quality, retain the stronger adapted control alongside native.
No changes to the selected checkpoint, no test reads or development-based tuning.
