# Saved-checkpoint forensics v1

Declared before executing these diagnostics, after the v1 pilot failure.
This is exploratory failure analysis, not an independent confirmatory study.
V1's failure at pi=.25 and its unexecuted pi=.50 remain unchanged.

Analyze all 12 existing best checkpoints: U1/B1, kappa 0/1, pi .05/.10/.25.
Read train rows with Parquet predicates and reuse each checkpoint's train-fitted
tensorizer. Do not deserialize combined train/validation caches. Read existing
training-report metadata for provenance only. Do not read new validation/test
content, realized oracle latents, or fit/update any model or data transform.

Before and after every analysis verify checkpoint SHA and model tensor digest.
Retain train entity diagnostics and exact gradient vectors as local artifacts.
The config fixes batch size, position bands and central gap mass before execution.

Diagnostics:

1. Entity-balanced full-support copy/repeat ranges and their entity quantiles,
   fraction above .05, position-stratified transition means, per-bin mean curves,
   and actual observed bin counts. Central 10–90% observed-gap-mass ranges are
   diagnostic only; they do not replace the full-support pilot criterion.
2. Base-logit probability, interaction-logit range, previous-mark fresh
   probability, and the identity R_repeat=(1-p_new(previous))*R_copy.
   The fresh distribution has no current-gap input. Distinguish broad response
   from a few entities/positions or only extreme bins; pooled bin frequency
   cannot establish conditional-history coverage.
3. Compute gradients for the four direct bilinear route tensors only. Hold
   history/static features and gap embeddings fixed. For each label y, compute
   g_y = grad(mean nonfirst repeat BCE_y) and d_y = grad(entity-mean copy range_y).
   Sum batches with fixed global denominators, not means of batch means.
4. Record g_0/g_1 cosine and local cross-context response derivatives
   -dot(d_target,g_source). Positive means an infinitesimal unpreconditioned
   descent direction for the source context raises target response locally.
   No parameter update is performed. This is not the historical AdamW update,
   does not include shared-encoder effects, and cannot prove training causation.

For route-only parameters, nonfirst mark NLL and repeat BCE have the same
gradient: conditional fresh-category loss after a nonrepeat has no dependence
on these parameters. Verify this identity in tests. Under globally aggregated
base mark NLL, a context's conditional route-gradient coefficient is T_y/E,
where E is all train events and T_y its nonfirst transitions. B1 adds 1/2.
Report both this change of context weights and the increase of total route-loss
weight. The stochastic minibatch base ratio is not exactly this global objective.

Interpretation must separate structural facts, measured local evidence and
unresolved hypotheses. Choose at most one new candidate only after inspecting
all planned results. Any revision is explicitly post-v1 exploratory design,
preregistered before its own training. Do not promise that it fixes the failure,
hard-code the DGP's active label, change v1 thresholds, or launch new training
as part of this forensic run.
