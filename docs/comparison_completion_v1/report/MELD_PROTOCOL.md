# MeLD full-length Sparkov baseline, registered before performance training

Two independent full fits, seeds20261021/22, two development draws20261031/32.
Official vendor b1e5bb0fd95786a7dc11f616892b60ea6b311767 stays immutable.
Primary Methods read at https://pmc.ncbi.nlm.nih.gov/articles/PMC13444125/
§MeLD model architecture /VAE loss /Diffusion loss /Evaluation framework.
Paper: VAE hidden256, embeddings128, two2-layer GRUs, latent32, max1000epochs;
DiT8layers,8heads,hidden256,100000steps,250samplingsteps. Paper uses120visits;
we retain ALL optimization sequences up to3123transactions. Same619/69customer
optimization/internal split asARGN; development147, common143 reported separately.

## Public code and explicitly named adaptations

Use official DeapStack, auto_loss, LightningDiT and Linear velocity transport with
cosine loss/lognormal times from public hiv_lite1 configuration. Channels64 is the
public VAE entrypoint default (paper describes a256-dimensional transformer);
this is a pinned-public-code Sparkov adaptation, NOT exact paper reproduction.
8head/8layer/256hidden DiT with SwiGLU/RMSNorm as public config, patch1 (one event),
32latent channels. No reduced smoke architecture or window truncation.
Use official parser fit ONLY on optimization. Its .001/.999 clipping stays visible;
evaluation real tails remain intact. Gap0 first event; interval metrics exclude it.
Time encoder receives fixed2000anchor+cumulativegap (elapsed-time task, not real
calendar reproduction). Model outputs numeric gap; no true future timestamps.
Parser.invert_fit preserves categorical strings; numeric-only convert_to_table
would not handle this mixed string schema. EOS decoding remains first predicted
EOS inclusive, threshold0.5, up to optimization maximum. Do not force true length.
No missing values generated in this complete-schema task, same as public default.

Static conditions are four supplied fields, joined as text, no ID/label/future count.
CLIP uses previously pinned public checkpoint instead of unavailable author path.
Public LightningDiT.initialize_weights applies Xavier recursively to frozen CLIP
linear layers. Restore public frozen text-encoder weights AFTER constructor and
verify exact checkpoint parity before training; document this adapter correction.
Do not present an unintentionally randomized frozen encoder as an adequate baseline.
This does not modify the vendor or invent a novel conditioning architecture.

## Training/selection

Full float32 VAE: Adam1e-4,weight_decay1e-6, macro batch128 (include last), micro8.
Accumulate reconstruction weighted by valid event count and KL by sequence count,
preserving public pooled loss at fixed predictions; test reduction/gradient parity.
Epoch1-50 beta.01; thereafter public five-epoch cyclical beta up to.01, reduce cap
byhalf after20epochs without internal reconstruction improvement, floor1e-5.
Choose lowest fixed-noise internal reconstruction, inspect every5epochs, save best
and resumable last. Stop after100epochs without improvement after epoch50, cap1000.
VAE validation masks are genuine lengths for reconstruction ONLY; generation never
receives them. Record fraud reconstruction recall/precision/log loss, EOS and
numeric errors; no claim of adequacy from loss alone. If selected VAE predicts zero
true-positive fraud events or nonfinite values, write ADEQUACY_BLOCK and stop before
expensive diffusion. This flags a failed adaptation, not inferiority to our model.

DiT: public AdamW lr2e-4,betas(.9,.95),weight_decay0,EMA.9999. Effective batch32,
micro8,full3123positions; this gives99936tokens/update vs paper122880, not equal
customer batch1024. Public channel min-max normalization fitted to train latents
only. Same transform applied internal and inverted before decoding. CFGdrop.1;
generation uses conditional y without extrapolating guidance (scale1).
Max100000updates, fixed-noise internal transport loss every1000updates, selectEMA;
minimum10000updates, early stop10000updates without internal improvement.
No development-based budget/checkpoint/CFG selection. 250Euler ODE generation.
Resume states include model/EMA/optimizer and RNG. Do not call reaching cap convergence.

After selected model, generate internal once for adequacy and development twice,
report all label/time/amount/category/merchant/length failures without filtering.
No automatic test read even if jobs finish. Comparator adequacy is audited before
final-test protocol; all current ARGNs, outputs and primary target definitions remain.
