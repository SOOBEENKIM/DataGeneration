# MeLD compatibility checks,2026-10-01

Official source pinned b1e5bb0fd95786a7dc11f616892b60ea6b311767; vendor source unchanged.
This is execution compatibility, not a trained strong baseline or paper reproduction.

Completed:
- Official notebook API with an explicit local32customer toy fixture. The notebook's
  input cell is `dt = ...`; no author-provided concrete toy table was claimed.
- All619Sparkov optimization customers /816341rows encoded at full maximum length3123.
  Zero truncation, exact EOS/padding counts, exact fraud/merchant/category roundtrip.
- Full3123position sequences passed a tiny official VAE forward/backward smoke with
  actual rare-fraud support. The3-step model missed all19fraud examples in its small
  training batch; this is visibly inadequate training and NOT MeLD's claimed performance.
- Four optimization steps of the official current DiT/transport on VAE toy latents,
  four-step ODE generation, VAE decoding, conditional/null/drop/CFG checks succeeded.
- Full-length check used3124positions (one explicit patch-padding token) and
  returned the expected shape. No120visit hard limit was encountered.

Important adaptation findings:
1. Official numeric preprocessing clips to .001/.999 training quantiles. For Sparkov,
   amount bounds1.02–1513.9946 changed1293rows, maximum absolute change27434.9054;
   gap bounds5–448964.62seconds changed1621rows, maximum absolute change892506.38.
   The initial exact-numeric-roundtrip check failed for this documented clipping.
   Its log/source are preserved. Revised checks verify parity with the officially
   clipped values and retain original-vs-clipped errors. This is not fixed by quietly
   deleting tails from the real evaluation reference.
2. Public DiT code hardcodes the author's local CLIP path. We explicitly supplied
   public openai/clip-vit-base-patch32 revision3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268.
   The exact author-local CLIP checkpoint is unknown. Conditional plumbing works;
   financial static conditioning accuracy is not yet established. No true future
   labels, amounts, gaps or lengths are included in condition text.
3. Common task currently has gaps, not absolute timestamps. Smoke `date` input is a
   fixed2000anchor plus cumulative gaps; generated gap remains an explicit numerical
   output. This is elapsed-time compatibility, not recovered real calendar timing.
4. Smoke dimensions are deliberately tiny (VAEhidden64/emb32,DiThidden64/depth2).
   They are not the paper's adequate training budget. The current public flow/ODE
   code and CFG update must be distinguished from the published paper's reported
   diffusion settings before registration of a full performance baseline.

Next MeLD phase: fix full architecture/budget and optimization-only parser with
internal validation, preserve all needed fields and learned EOS, generate static
conditions coherently, and record raw clipping/time/length/rare-label diagnostics.
A tail-preserving transform, if compared, must be a separately named adaptation.
No final-test events read, no push, no superiority or novelty conclusion.

Primary reading checked this turn:
- https://www.nature.com/articles/s41467-026-74492-0 Methods architecture/budgets and
  domain-level evaluation. Independent fitting and repeated generation are distinct.
- https://github.com/jeffreyliang-vandy/MeLD current pinned implementation, preprocessing,
  VAE,DiT,CLIP/null conditioning and transport. Not every evaluation script reread.
