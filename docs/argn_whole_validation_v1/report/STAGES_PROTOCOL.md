# Downstream stage registration, before fresh-head fitting

For each fresh backbone seed: refit balanced category, balanced amount, balanced
gap with the preserved optimizer/selection settings; refit onset amount from its
new balanced amount head; refit the selected unconditional joint length/initial-label
model. Deterministic optimization-only duration/hazard tables and transform limits
are reused exactly. All new real-prefix representations come from that fresh backbone.
No old trained amount/category/gap/onset neural weights are reused.

Refit four phase Gaussian-mixture priors (3 components, n_init3, max_iter100,
reg_covar1e-4, at most50000 phase rows) then the established history-only mixture
head (hidden64,3components,lr3e-4,max2000steps,internal NLL every100,patience6).
Fit seed equals backbone seed. The statistical initializer is the same function as
prior validation. Inactive GMR and first-five wrapper buffers are defaults; neither
is called by these three history heads. No initial treatment is applied to H/P6/C.

Calibrate P6 and C with SAME generated conditional noise, frozen history head,
full optimization sequences, customer-weighted stratified objective,1536max function
evaluations including zero, Powell xtol/ftol.001 and +-0.35 coefficient/output cap.
P6 uses six log-position knots[25,125,350,750,1500,2500]; C uses the preserved
personal_six basis (3 position knots x 2 personal-speed endpoints).
All non-correction tensors are equal within a seed. Calibration is actual-prefix
conditional sampling, not new free-history learning. No development optimization.

18 full native generations=3complete fits x3arms x2draws. Labels/lengths are generated
by each fresh common foundation, never true future outcomes; equal across paired
time arms must be independently checked. They may differ across fresh fit seeds.
Hook replay/strict-past/reordered-batch checks run for every new time payload before
native generation. The independent evaluator is frozen and checked on old outputs
before fresh generations are read. Costs use the established metrics and support.

Descriptive primary screen: C normal error at least10% lower than P6, onset-personal
error no more than.01 higher, normal improvement in each of3fit means. This is a
repeatability screen, not a p-value, novelty claim or automatic adoption. Report all
metric costs even if primary screen passes; do not relabel draws as independent fits.
CPAR/MeLD results and eventual test/transfer are separate phases and not implied done.
