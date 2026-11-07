# Supplemental frozen check, added after first free draw

The first free draw has label prevalence58.66% (official time scaling) and
64.84% (median scaling), far from the real rare-fraud frequency. This motivates
an implementation/adequacy check, not calling LBDTPP a proven weak model.
No outputs are deleted, clipping rules changed, hyperparameters retuned,
checkpoints reselected or full generations restarted in this diagnostic.

With each selected frozen checkpoint and internal-validation data, measure
direct decoding of true embeddings and decoding after one denoising prediction
at diffusion indices0,50,99. Denoising uses the same block mask and true clean
previous blocks as official training. Split positions1–7,8–63,64+. Report label
and mark accuracy, fraud recall/prevalence and log-gap MAE (these are conditional
reconstruction metrics, NOT generation-fidelity scores). This distinguishes a
failed label-code/decoder mapping from failure to denoise without an informative
prefix. It cannot by itself identify one unique structural cause. Source files,
both best checkpoints and fitting/selection protocol stay frozen. No test reads.
