"""
soft_g: Differentiable behavioral summary g(x̂₀)[j] = [vel, gap, rep, amt_sum]

  vel[j]     = soft count of predecessors in causal window [0,j) by time proximity
  gap[j]     = expected inter-transaction Δt at position j  (= dt_hat[j])
  rep[j]     = soft distinct category/receiver count in causal window [0,j)
  amt_sum[j] = soft expected amount from causal window [0,j) (key for AMLSim structuring)

Causal window: STRICTLY m < j (no self, no future).
detach FORBIDDEN — gradient must flow through g to generation parameters.
Amount channel added: gradient also flows through amt → amount generation head.
τ_k, W, temp are dataset-specific — injected at call site, never hardcoded here.

AMLSim note: W is in step units (e.g. 7 or 30); τ_k in step medians from Phase 1 train.
Sparkov note: W is in minutes (e.g. 60*24*7 = 1 week); τ_k in minute medians.
"""

import torch
from torch import Tensor
from typing import Optional


def soft_g(
    bin_probs: Tensor,
    cat_probs: Optional[Tensor],
    amt: Tensor,
    tau: Tensor,
    W: float,
    temp: float,
    *,
    valid_mask: Tensor,
) -> Tensor:
    """
    bin_probs: (B_, L, Bbins) — softmax bin probs (must stay in computation graph)
    cat_probs: (B_, L, K)     — softmax category/receiver probs, or None (no cats)
    amt:       (B_, L)         — amount per position (normalized, stays in comp graph)
    tau:       (Bbins,)        — median Δt per bin, fit on TRAIN only (constant)
    W:         float            — trailing window width in dataset-specific units
    temp:      float            — sigmoid smoothing temperature
    Returns g: (B_, L, 4)      = [vel, gap, rep, amt_sum]
               rep == 0 everywhere when cat_probs is None (e.g. AMLSim)
    """
    B_, L, Bbins = bin_probs.shape
    if valid_mask.shape != (B_, L) or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool with shape [B,L]")

    tau_dev = tau.to(bin_probs.device, bin_probs.dtype)
    mask_f = valid_mask.to(device=bin_probs.device, dtype=bin_probs.dtype)
    dt_hat = (bin_probs * tau_dev[None, None, :]).sum(-1)  # (B_, L)
    dt_hat = dt_hat * mask_f
    T = torch.cumsum(dt_hat, dim=1)                        # (B_, L)

    # diff[b, j, m] = T[b,j] - T[b,m]
    diff = T.unsqueeze(2) - T.unsqueeze(1)  # (B_, L, L)

    # Strict causal mask: 1 where m < j, 0 where m >= j (diagonal excluded)
    causal_mask = torch.tril(
        torch.ones(L, L, device=bin_probs.device, dtype=bin_probs.dtype),
        diagonal=-1,
    )
    assert causal_mask.diagonal(offset=0).sum() == 0, "causal assert: self-attention found"
    assert (L < 2) or (causal_mask[1, 0] == 1 and causal_mask[0, 1] == 0), \
        "causal assert: mask direction wrong"

    # Window weight w[b,j,m] = σ((W - Δt)/temp), zeroed at m >= j
    source_mask = mask_f[:, None, :]
    target_mask = mask_f[:, :, None]
    pair_mask = source_mask * target_mask
    w = torch.sigmoid((W - diff) / temp) * causal_mask[None] * pair_mask

    # [1] Velocity
    vel = w.sum(dim=2)  # (B_, L)

    # [2] Gap = expected Δt
    gap = dt_hat  # (B_, L)

    # [3] Rep / fan-out
    if cat_probs is not None:
        w_exp = w.unsqueeze(-1)                              # (B_, L, L, 1)
        r_exp = cat_probs.unsqueeze(1)                       # (B_, 1, L, K)
        prod = (w_exp * r_exp).clamp(max=1 - 1e-7)          # (B_, L, L, K)
        log_not_seen = torch.log1p(-prod).sum(dim=2)         # (B_, L, K)
        rep = (1.0 - torch.exp(log_not_seen)).sum(dim=-1)   # (B_, L)
    else:
        rep = torch.zeros_like(vel)

    # [4] Amount summary: soft expected amount from causal window
    # amt_sum[b,j] = Σ_{m<j} w[b,j,m] * amt[b,m]
    amt_sum = (w * amt.unsqueeze(1)).sum(dim=2)  # (B_, L)

    g = torch.stack([vel, gap, rep, amt_sum], dim=-1)
    return g * mask_f[..., None]
