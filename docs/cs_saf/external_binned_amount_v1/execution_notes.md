# Execution notes

- Parent results `5f81c5e`; preregistered source `1579ee4bbe0afe3d822799a8afdb94c0fbf147c5`
  was committed and pushed before either scientific fit. New neural fits2,
  raw generated tapes4, calibration fits0. No grid, restarts, coefficient search
  or automatic follow-up. Both fits completed on the first attempt.
- Five CPU tests passed. CPU/GPU preflight checked all 1,990,071 development
  event codes and fit histogram equality; boundary forward/backward finite.
  GPU preflight performed no optimizer updates. Sparkov has one check amount
  outside the fit range, none in validation; edge-bin CE is not raw likelihood.
- Common non-amount tensors came from each previous D's saved random initial
  checkpoint, not trained weights. The source initializer and existing trained D
  checkpoints/results are unchanged. Both models were trained jointly from the
  matched initialization, so the learned history/action parameters can change.
- Berka bins32 including a zero bin, Sparkov34 positive bins result from the
  single fixed quantile rule and deterministic merging, not selection by quality.
  Parameter count increases are 4,246 (3.33%) and 4,632 (1.60%). This is an
  ordinary output control, not a matched-capacity novelty claim.
- GPU0 and GPU2 were below5% utilization and512MiB when admitted. A process on
  GPU0 observed during an earlier inspection had finished before admission;
  no process was killed, suspended or reset. The exact launch-time query is saved.
- Berka selected26/ended30 with the epoch cap, not patience convergence.
  Sparkov selected8/ended13 by patience. No time cap was hit. Actual budget and
  loss traces are preserved. Different native amount scores are not compared
  across model types; held-out prediction costs use common metrics.
- The standalone verifier independently reduces all76 generation metrics,
  48 tail scalars, validation likelihood components, Brier, log-MAE and empirical
  tail probabilities. Model forward is reused; histogram/ECDF/score reductions
  are separate. Checkpoint epoch, update counts, fit-pool multiset, raw sampled
  support and original source hashes are checked.
- The report uses the predeclared screens unchanged. Post-result operation-rate
  and exact-fit-support tables describe costs/limitations and do not change the
  decision. mark/root failures on Berka refer to the same operation marginal.
- Final decision: no common adoption; Sparkov passes, Berka fails two distinct
  cost types. Record partial improvements and end this amount-output direction.
  No extra model, epoch extension or seed run follows this outcome.
