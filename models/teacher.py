"""
BehaviorTeacher f_φ: BiGRU encoder over real sequences.

Outputs:
  g_pred:  (B, L, 4)  — predicted behavioral summary [vel, gap, rep, amt_sum]
  y_logit: (B, L)     — fraud prediction logit

Pretraining:
  loss = BCE(y_logit, y_real) + 0.1 * MSE(g_pred, g_real/g_std)
  where g_real = soft_g(one-hot(dt_bin), one-hot(x_cat), amt_real, τ, W, temp)
  and   g_std  = fixed per-component std computed once from the full TRAIN set.

Normalization consistency:
  g_std MUST be the same fixed buffer used in CoFSeqGen.compute_loss().
  Workflow:
    1. g_mean, g_std = compute_g_stats_from_data(x_num_train, ...)
    2. pretrain_teacher(..., g_std=g_std)          → teacher outputs normalized g
    3. CoFSeqGen(..., g_std=g_std)                  → coherence loss uses same space

Gate requirement: AUPRC(y_logit, y_real) > prevalence on val set → freeze.

After freeze: teacher(real_batch) provides g_target (normalized) for L_coh in CoF-SeqGen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor
from typing import List, Optional, Tuple

from models.soft_g import soft_g


class BehaviorTeacher(nn.Module):
    def __init__(
        self,
        d_num: int,
        Bbins: int,
        n_cat_classes: List[int],
        d_hidden: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.Bbins = Bbins
        self.n_cat_classes = n_cat_classes

        self.num_proj = nn.Linear(d_num, d_hidden)
        self.bin_emb = nn.Embedding(Bbins, d_hidden)
        self.cat_embs = nn.ModuleList([nn.Embedding(K, d_hidden) for K in n_cat_classes])

        gru_in = d_hidden * (2 + len(n_cat_classes))
        self.gru = nn.GRU(gru_in, d_hidden, num_layers=n_layers,
                          batch_first=True, bidirectional=True)
        self.g_head = nn.Linear(d_hidden * 2, 4)
        self.y_head = nn.Linear(d_hidden * 2, 1)

    def forward(
        self,
        x_num: Tensor,   # (B, L, d_num)
        dt_bin: Tensor,  # (B, L) int
        x_cat: Tensor,   # (B, L, d_cat) int  (d_cat may be 0)
        *,
        valid_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Returns: g_pred (B,L,4), y_logit (B,L)."""
        if valid_mask.dtype is not torch.bool or valid_mask.shape != dt_bin.shape:
            raise ValueError("valid_mask must be bool with shape [B,L]")
        lengths = valid_mask.sum(1)
        expected = torch.arange(valid_mask.shape[1], device=valid_mask.device)[None] < lengths[:, None]
        if not torch.equal(valid_mask, expected):
            raise ValueError("BehaviorTeacher requires prefix-contiguous right padding")
        h = self.num_proj(x_num) + self.bin_emb(dt_bin)   # (B, L, d_hidden)
        parts = [h, self.bin_emb(dt_bin)]
        for i, emb in enumerate(self.cat_embs):
            if x_cat.shape[-1] > i:
                parts.append(emb(x_cat[:, :, i]))
        h = torch.cat(parts, dim=-1)   # (B, L, gru_in)

        packed = pack_padded_sequence(
            h, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        h_gru, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x_num.shape[1]
        )
        g_pred = self.g_head(h_gru)                      # (B, L, 4)
        y_logit = self.y_head(h_gru).squeeze(-1)         # (B, L)
        g_pred = g_pred * valid_mask[..., None]
        y_logit = y_logit * valid_mask
        return g_pred, y_logit

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)
        self.train()


def compute_g_from_real(
    x_num: Tensor,
    dt_bin: Tensor,
    x_cat: Tensor,
    tau: Tensor,
    W: float,
    temp: float,
    Bbins: int,
    n_cat_classes: List[int],
    *,
    valid_mask: Tensor,
) -> Tensor:
    """
    g_real = soft_g(one-hot(dt_bin), one-hot(x_cat[0]), x_num[:,0], τ, W, temp).
    Hard (1-hot) encoding = exact assignment, no gradient needed.
    Call under torch.no_grad() or detach result.
    """
    bin_probs = F.one_hot(dt_bin.clamp(0, Bbins - 1), num_classes=Bbins).float()

    if n_cat_classes and x_cat.shape[-1] > 0:
        K0 = n_cat_classes[0]
        cat_probs = F.one_hot(x_cat[:, :, 0].clamp(0, K0 - 1), num_classes=K0).float()
    else:
        cat_probs = None

    amt_real = x_num[:, :, 0]  # first col = amount / log-amount
    return soft_g(
        bin_probs, cat_probs, amt_real, tau, W, temp, valid_mask=valid_mask
    )


def compute_g_stats_from_data(
    x_num: Tensor,
    dt_bin: Tensor,
    x_cat: Tensor,
    tau: Tensor,
    W: float,
    temp: float,
    Bbins: int,
    n_cat_classes: List[int],
    *,
    valid_mask: Tensor,
    batch_size: int = 512,
) -> Tuple[Tensor, Tensor]:
    """
    Compute per-component (mean, std) of g over the full dataset.

    Run ONCE on the TRAIN set to obtain a fixed normalization buffer.
    Pass the returned g_std to:
      - pretrain_teacher(..., g_std=g_std)   so teacher outputs normalized g
      - CoFSeqGen(..., g_std=g_std)          so coherence loss uses the same space

    Args:
        x_num:         (N, L, d_num)
        dt_bin:        (N, L) int
        x_cat:         (N, L, d_cat) int  (d_cat=0 for AMLSim)
        tau:           (Bbins,) median Δt per bin — device used as compute device
        W:             window width (dataset units)
        temp:          sigmoid smoothing temperature
        Bbins:         number of time bins
        n_cat_classes: list of category sizes (empty for AMLSim)
        batch_size:    rows processed per batch to avoid OOM

    Returns:
        g_mean: (4,)  per-component mean
        g_std:  (4,)  per-component std, clamped ≥ 1.0
                      clamp prevents division-by-zero on degenerate components
                      (e.g. AMLSim rep=0 always; AMLSim gap with tau_k≈0 blocks)
    """
    N = x_num.shape[0]
    device = tau.device

    g_parts: List[Tensor] = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            g_b = compute_g_from_real(
                x_num[start:end].to(device),
                dt_bin[start:end].to(device),
                x_cat[start:end].to(device),
                tau, W, temp, Bbins, n_cat_classes,
                valid_mask=valid_mask[start:end].to(device),
            )
            mask_b = valid_mask[start:end].to(g_b.device)
            g_parts.append(g_b[mask_b].cpu())

    g_all = torch.cat(g_parts, dim=0)          # (N*L, 4)
    g_mean = g_all.mean(dim=0)                  # (4,)
    g_std  = g_all.std(dim=0).clamp(min=1.0)   # (4,) — clamped for degenerate components
    return g_mean, g_std


def pretrain_teacher(
    teacher: BehaviorTeacher,
    x_num: Tensor,
    dt_bin: Tensor,
    x_cat: Tensor,
    y: Tensor,
    tau: Tensor,
    W: float,
    temp: float,
    g_std: Optional[Tensor] = None,
    *,
    valid_mask: Tensor,
    n_steps: int = 80,
    lr: float = 3e-3,
) -> List[float]:
    """
    Train teacher on real data for n_steps.

    g_std: per-component normalization buffer from compute_g_stats_from_data().
           MUST be the same tensor passed to CoFSeqGen(g_std=...) so both
           teacher and coherence loss operate in the same normalized g space.
           If None (legacy): teacher is trained on raw g — inconsistent if
           CoFSeqGen uses normalized coherence loss.

    Returns per-step loss history.
    """
    teacher.unfreeze()
    optim = torch.optim.Adam(teacher.parameters(), lr=lr)

    with torch.no_grad():
        g_real = compute_g_from_real(
            x_num, dt_bin, x_cat, tau, W, temp,
            teacher.Bbins, teacher.n_cat_classes,
            valid_mask=valid_mask,
        )
        if g_std is not None:
            g_std_dev = g_std.to(g_real.device).clamp(min=1.0)
            g_target = g_real / g_std_dev[None, None, :]   # normalized
        else:
            g_target = g_real   # raw (legacy; inconsistent with normalized CoFSeqGen)

    losses = []
    for _ in range(n_steps):
        optim.zero_grad()
        g_pred, y_logit = teacher(
            x_num, dt_bin, x_cat, valid_mask=valid_mask
        )
        loss_y = F.binary_cross_entropy_with_logits(
            y_logit[valid_mask].float(), y[valid_mask].float()
        )
        loss_g = F.mse_loss(g_pred[valid_mask], g_target[valid_mask])
        loss = loss_y + 0.1 * loss_g
        loss.backward()
        optim.step()
        losses.append(loss.item())

    return losses
