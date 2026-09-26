# Interpretation safeguards grounded in this dataset and implementation

- Sparkov is simulated transaction data, including simulated fraud. Preserving
  this benchmark's observed relationships does not by itself establish fidelity
  to a bank's real fraud mechanisms. The generator's own
  [source repository](https://github.com/namebrandon/Sparkov_Data_Generation)
  documents its synthetic transactions, customer profiles and fraud generation.
- The generation task starts new sequences conditioned on static customer
  attributes. Each subsequent transaction uses the generated past. Real prefixes
  are used only in the separately identified frozen teacher diagnostic.
- Fraud prevalence is event-weighted. The equally weighted average of customer
  fraud fractions is a different estimand: 8.2084% in training versus 0.5689%
  over events. Training has 52 customers with <=100 transactions, totaling 507
  events, all fraud; validation has 12 such customers, 132 events, all fraud.
  This observation motivated additional retrospective length strata after the
  initial runs. Total future length is not a past-history feature. The resulting
  strata are diagnostic and are not supplied as validation generation conditions.
  Additional exploratory onset/continuation strata separate a fraud transaction
  immediately following a normal transaction from one following another fraud.
  The teacher probe reports actual event support and assigned probabilities for
  these groups; its prefix remains capped at 512 real transactions per customer.
  A low fraud probability or zero recall at a 0.5 threshold for rare onsets alone
  does not establish a generation failure. Evaluate generated onset rates and
  conditional feature distributions against their real-data sampling variation.
  Training short sequences have lengths 7–15, and all other training sequences
  have at least 471 events. Validation includes one 19-event sequence. Report
  generated customer counts in the training length gap (16–470), rather than
  labeling every unseen length invalid or treating a broad <=100 conditional
  fraud-rate difference as a pure label-model failure.
- Engine 1.0.4 learns/generates the sequence length internally and includes its
  positional encoding in the child model. The frozen teacher diagnostic encodes
  the true real sequence length, as native validation does. Consequently its
  label accuracy is not an online fraud-detector result and is not directly
  interchangeable with unconditional/free-running generation quality.
- Per-sequence training loss and event-weighted reporting have different weights.
  The short fraud-only customers make that distinction substantial. This is a
  candidate mechanism involving length/composition, not a demonstrated causal
  effect until a matched loss-weighting experiment is performed.
- CPAR 0.8.1 samples each current field from parameters produced by the same
  recurrent state, without feeding the sampled current merchant into the current
  category sampler. ARGN explicitly supports dependencies between current fields.
  Similar joint-distribution errors can therefore have different mechanisms in
  the two implementations. A baseline's weak result does not establish novelty.
- No numerical transformation, pure-long-rollout explanation, training-cap
  explanation, or architecture explanation should be asserted beyond the paired
  controls that were actually executed. Removing history or shuffling merchant
  information measures frozen-model sensitivity; it does not isolate the cause
  of the original fitting error.

Primary sources: [TabularARGN author repository](https://github.com/mostly-ai/paper-tabular-argn),
[official PARSynthesizer documentation](https://docs.sdv.dev/sdv/modeling/sequential-synthesizers/parsynthesizer),
and the pinned local engine 1.0.4 / DeepEcho 0.8.1 source whose hashes are recorded
with the execution artifacts. CPAR's native numerical-target padding convention
is preserved, as documented in the execution amendment.
