# Execution amendment 01: evaluation-mode GRU backward

After the registered 20 additional pi=.05 fits completed, the first three
saved-checkpoint gradient audit jobs failed with
`cudnn RNN backward can only be called in training mode`. Dispatch stopped;
no prevalence expansion or external CPAR fit had started. All 20 fits and their
audits remain unchanged under source fccc7bb. This is a diagnostic implementation
failure, not model training/scientific failure.

Keep all weights/objectives/subsets fixed and leave the model in evaluation
mode except for the GRU reserve-space flag. Verify GRU dropout is exactly0,
then set only `model.encoder.gru.train()` so cuDNN retains backward intermediates.
This GRU has no stochastic dropout, so its mathematical forward is unchanged.
A GPU regression compares the loss/penalty values to cuDNN eval at rtol1e-5 /
atol1e-6, checks finite gradients and unchanged state. A direct test observed
exactly0 maximum difference for all four loss/penalty quantities. No optimizer
step is performed; training code and all scientific criteria remain unchanged.
An initially considered native-backend workaround differed by up to9.72e-6 in
one residual sum and failed that numerical test; it was abandoned before any
successful diagnostic. We did not loosen the test to accept it.

Local gradients are raw parameter-space gradients. They do not reconstruct
AdamW preconditioning, weight decay or historical trajectories. Context-specific
subset averages are not the prevalence-weighted full-training gradient.

Archive the first execution progress/logs/three failed audit folders under
`artifacts/cs_saf/followup_v1/technical_failure_1/`. Preserve `cpu_v1`; rerun the
same CPU gates at `cpu_v2` with corrected source. Resume reusing all20 completed
fits; do not refit to alter results. Successful scientific run identities remain
170 new internal fits plus40 CPAR fits. Failed diagnostic attempts are separately
counted. The new execution progress contains only new allocations; the archived
progress contains the original allocations.

The .003/.03 outcomes had been seen before this technical fix. No hypotheses,
coefficients, sample selection, scientific thresholds, or task order are changed.
The lambda sweep remains exploratory and its original full-grid scope is retained.
