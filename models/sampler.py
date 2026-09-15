"""
DDIM reverse-diffusion sampler for CoFSeqGen v1b.

v1b additions:
  - ddim_sample now returns y_logit_final (N, L) raw logits before Bernoulli sample.
    Caller uses this for post-hoc entity-level temperature calibration.
  - calibrate_temperature_entity(): binary search for T s.t.
    mean_entity(P(at least one fraud position)) = target_entity_prevalence.

Calibration logic:
  - After DDIM generates all sequences, stack y_logit_all (N, L).
  - Find T via entity-level binary search: evaluates P(entity fraud) for each T.
  - Resample y_gen ~ Bernoulli(sigmoid(y_logit/T)).
  - Gate: entity fraud_rate_synth should be within 2× of real prevalence.

Why entity-level calibration (not position-level):
  - TSTR uses max-over-sequence labels, not position means.
  - Position-level P=1.6% → entity-level P≈32% for L=24 (the original bug).
  - Entity-level calibration computes the correct objective directly.
"""

import math
import torch
from torch import Tensor
from typing import List, Optional, Tuple

from models.cof_seqgen import CoFSeqGen


def _cosine_alpha_sigma(t: float) -> Tuple[float, float]:
    alpha = math.cos(math.pi * t / 2)
    sigma = math.sin(math.pi * t / 2)
    return alpha, sigma


def calibrate_temperature_entity(
    y_logit: Tensor,           # (N, L) raw logits for all positions
    mask: Tensor,              # (N, L) bool — True = real (non-padding)
    target_prevalence: float,  # entity-level: fraction of sequences with fraud
    n_iter: int = 300,
) -> float:
    """
    Find additive bias b s.t. entity-level fraud rate = target_prevalence.

    Uses logit-bias calibration (NOT temperature scaling) because bias is
    strictly monotone decreasing in b for ALL logit distributions:

        P_entity(b) = mean_seq( 1 - prod_{l: masked}(1 - sigmoid(logit_l - b)) )

    This reaches any target in (0, 1) — temperature scaling can get stuck
    when some entities always have a positive logit at T→0.

    Returns b (float). Caller resamples:  y_gen ~ Bernoulli(sigmoid(logit - b))
    """
    logits = y_logit.float()
    mask_f = mask.float()

    def entity_rate(b: float) -> float:
        p_pos = torch.sigmoid(logits - b)                        # (N, L)
        safe  = p_pos.clamp(max=1.0 - 1e-7)
        log_no_fraud = (torch.log1p(-safe) * mask_f).sum(dim=1) # (N,)
        p_entity = 1.0 - torch.exp(log_no_fraud)                # (N,)
        return p_entity.mean().item()

    # P_entity is strictly decreasing in b:
    #   b very negative → all probs → 1 → entity rate → 1
    #   b very positive → all probs → 0 → entity rate → 0
    lo, hi = -50.0, 50.0
    # Verify feasibility (target must be between entity_rate(hi) and entity_rate(lo))
    p_lo = entity_rate(lo)   # ≈ 1
    p_hi = entity_rate(hi)   # ≈ 0
    target = float(max(min(target_prevalence, p_lo - 1e-7), p_hi + 1e-7))

    for _ in range(n_iter):
        b = (lo + hi) / 2.0
        p = entity_rate(b)
        if abs(p - target) < 1e-8:
            return b
        # P_entity decreasing: p > target → need larger b → lo = b
        if p > target:
            lo = b
        else:
            hi = b

    return (lo + hi) / 2.0


@torch.no_grad()
def ddim_sample(
    model: CoFSeqGen,
    dt_bin_cond: Tensor,           # (B, L) int
    x_cat_cond: Tensor,            # (B, L, d_cat) int
    d_num: int,
    T_steps: int = 50,
    device: str = "cpu",
    return_discrete: bool = True,
    y_cond: Optional[Tensor] = None,      # (B,) long {0,1} — entity label; None=unconditional
    guidance_scale: float = 1.0,          # CFG scale; 1.0=class-conditional only, >1=full CFG
    feedback_discrete: bool = False,      # feed model's bin/cat predictions back each step
    feedback_after: float = 0.3,          # start feedback when t <= (1 - feedback_after)
    discrete_temp: float = 1.0,           # >0=softmax sample, <=0=argmax
    start_from_mask: bool = False,        # init dt_bin/x_cat as MASK tokens (discrete diffusion)
    *,
    valid_mask: Tensor,
) -> Tuple[Tensor, Optional[Tensor], Optional[List[Tensor]], Optional[Tensor], Optional[Tensor]]:
    """
    DDIM sampling for (x_num, dt_bin, x_cat, y_gen, y_logit).

    Class-conditional generation (CFG, Ho & Salimans 2022):
      y_cond: pre-sampled entity fraud labels (B,) long {0,1}.
              None → unconditional (null token, class 2).
      guidance_scale=1.0: class-conditional only (single forward pass).
      guidance_scale>1.0: full CFG — conditioned + unconditional, mixed with scale.

    Returns:
        x_num_gen:    (B, L, d_num) — generated numerical features
        bin_pred:     (B, L)        — argmax of bin_logits at final step
        cat_preds:    list of (B,L) — argmax of cat_logits at final step
        y_gen:        (B, L) long   — Bernoulli sample from sigmoid(y_logit) at T=1
        y_logit:      (B, L) float  — raw logits for post-hoc calibration / diagnostics
    """
    model.eval()
    B, L = dt_bin_cond.shape
    if valid_mask.dtype is not torch.bool or valid_mask.shape != (B, L):
        raise ValueError("valid_mask must be bool with shape [B,L]")

    dt_bin_cond = dt_bin_cond.to(device)
    x_cat_cond  = x_cat_cond.to(device)
    valid_mask = valid_mask.to(device)

    # Discrete diffusion: replace empirical init with full-MASK start
    if start_from_mask:
        MASK_B = model.denoiser.Bbins
        dt_bin_cond = torch.where(
            valid_mask,
            torch.full((B, L), MASK_B, dtype=torch.long, device=device),
            torch.zeros((B, L), dtype=torch.long, device=device),
        )
        d_cat = x_cat_cond.shape[-1]
        cats = [model.denoiser.n_cat_classes[k] for k in range(d_cat)]
        x_cat_cond = torch.stack(
            [
                torch.where(
                    valid_mask,
                    torch.full((B, L), cats[k], dtype=torch.long, device=device),
                    torch.zeros((B, L), dtype=torch.long, device=device),
                )
                for k in range(d_cat)
            ],
            dim=-1
        )

    y_cond_dev = y_cond.to(device) if y_cond is not None else None
    null_cond  = torch.full((B,), 2, dtype=torch.long, device=device)  # unconditional token

    t_schedule = torch.linspace(1.0, 0.0, T_steps + 1, device=device)
    x = torch.randn(B, L, d_num, device=device) * valid_mask[..., None]

    bin_pred_final  = None
    cat_preds_final = None
    y_gen_final     = None
    y_logit_final   = None

    for i in range(T_steps):
        t_curr = t_schedule[i].item()
        t_next = t_schedule[i + 1].item()

        alpha_curr, sigma_curr = _cosine_alpha_sigma(t_curr)
        alpha_next, sigma_next = _cosine_alpha_sigma(t_next)

        t_vec = torch.full((B,), t_curr, device=device)

        if y_cond_dev is not None and guidance_scale > 1.0:
            # Full CFG: two forward passes, then mix
            x_hat_c, bin_c, cat_c, y_logit_c = model.denoiser(
                x, dt_bin_cond, x_cat_cond, t_vec, y_cond=y_cond_dev,
                src_key_padding_mask=~valid_mask,
            )
            x_hat_u, bin_u, cat_u, y_logit_u = model.denoiser(
                x, dt_bin_cond, x_cat_cond, t_vec, y_cond=null_cond,
                src_key_padding_mask=~valid_mask,
            )
            x_num_hat = x_hat_u + guidance_scale * (x_hat_c - x_hat_u)
            bin_logits = bin_u + guidance_scale * (bin_c - bin_u)
            cat_logits = [
                cu + guidance_scale * (cc - cu)
                for cu, cc in zip(cat_u, cat_c)
            ]
            y_logit = y_logit_c   # use conditioned logit for label consistency
        else:
            # Single pass: class-conditional (y_cond) or unconditional (None)
            x_num_hat, bin_logits, cat_logits, y_logit = model.denoiser(
                x, dt_bin_cond, x_cat_cond, t_vec, y_cond=y_cond_dev,
                src_key_padding_mask=~valid_mask,
            )

        eps_hat = (x - alpha_curr * x_num_hat) / (sigma_curr + 1e-8)
        x_num_hat = x_num_hat * valid_mask[..., None]
        x = (alpha_next * x_num_hat + sigma_next * eps_hat) * valid_mask[..., None]

        # ── Discrete channel self-feedback ──────────────────────────────────
        # Feed model's y-conditioned bin/cat predictions back as next-step input.
        # guidance_scale>1 amplifies the y signal in bin_logits/cat_logits before
        # feeding back, helping weak y_emb conditioning overcome marginal echo.
        if feedback_discrete and t_curr <= (1.0 - feedback_after):
            if discrete_temp <= 0:
                dt_bin_cond = bin_logits.argmax(dim=-1)                               # (B, L)
                x_cat_cond  = torch.stack([cl.argmax(-1) for cl in cat_logits], -1)   # (B, L, d_cat)
            else:
                bp = torch.softmax(bin_logits / discrete_temp, dim=-1)
                dt_bin_cond = torch.multinomial(
                    bp.reshape(-1, bp.size(-1)), 1).reshape(B, L)
                cats = []
                for cl in cat_logits:
                    cp = torch.softmax(cl / discrete_temp, dim=-1)
                    cats.append(torch.multinomial(
                        cp.reshape(-1, cp.size(-1)), 1).reshape(B, L))
                x_cat_cond = torch.stack(cats, dim=-1)
            dt_bin_cond = torch.where(valid_mask, dt_bin_cond, 0)
            x_cat_cond = torch.where(valid_mask[..., None], x_cat_cond, 0)

        if i == T_steps - 1 and return_discrete:
            bin_pred_final = torch.where(valid_mask, bin_logits.argmax(dim=-1), 0)
            cat_preds_final = [
                torch.where(valid_mask, cl.argmax(dim=-1), 0) for cl in cat_logits
            ]
            y_logit_final   = y_logit.cpu()                              # (B, L) raw logits
            y_prob          = torch.sigmoid(y_logit)
            y_gen_final     = (torch.rand_like(y_prob) < y_prob).long() # uncalibrated Bernoulli

    return x, bin_pred_final, cat_preds_final, y_gen_final, y_logit_final


def sample_empirical_dt_bin(
    dt_bin_train: Tensor,
    n_samples: int,
    L: int,
    Bbins: int,
    device: str = "cpu",
) -> Tensor:
    N = dt_bin_train.shape[0]
    idx = torch.randint(0, N, (n_samples,))
    return dt_bin_train[idx].to(device)


def sample_empirical_x_cat(
    x_cat_train: Tensor,
    n_samples: int,
    device: str = "cpu",
) -> Tensor:
    N = x_cat_train.shape[0]
    idx = torch.randint(0, N, (n_samples,))
    return x_cat_train[idx].to(device)
