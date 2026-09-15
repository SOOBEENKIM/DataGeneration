"""
CoherenceTeacher f_φ: g(4) → P(fraud=1 | behavior).          [legacy, kept for Sparkov]
SequenceTeacher  f_φ_seq: raw window → per-position P(fraud). [v1c — preferred]

CoherenceTeacher operates on the 4-dim behavioral summary g = [vel, gap, rep, amt_sum].
SequenceTeacher sees the full window (amount trajectory + timing + receiver) via a
BiGRU, enabling much stronger coherence supervision (oracle AUPRC 0.978 vs 0.084 for g).

SequenceTeacher design constraint — soft inputs (CRITICAL):
  During CoF training, timing and receiver heads produce SOFT distributions
  (softmax logits, not discrete argmax). SequenceTeacher receives:
    bin_probs @ bin_emb.weight  →  expected timing embedding   (grad → bin_head)
    cat_probs @ cat_emb.weight  →  expected receiver embedding  (grad → cat_head)
  This keeps gradient flowing to ALL heads through L_coh. Hard argmax kills timing/
  receiver gradients — grad would flow to amount only (C2a, not C2b).

Gradient flow (SequenceTeacher path):
  L_coh → P_model → y_logit      → label head     (align label)
  L_coh → q → SequenceTeacher
           → bin_probs @ bin_emb → bin_head        (align timing)
           → cat_probs @ cat_emb → cat_head        (align receiver/fan-out)
           → num_proj(amt_hat)   → num_head        (align amount)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional


class CoherenceTeacher(nn.Module):
    """f_φ: normalized g (B,L,4) → fraud logit (B,L). Trained on real, then frozen."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, g_norm: Tensor) -> Tensor:
        """g_norm: (..., 4) → logit (...). Last dim squeezed."""
        return self.net(g_norm).squeeze(-1)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)
        self.train()


def pretrain_coherence_teacher(
    f_phi: "CoherenceTeacher",
    x_num: Tensor,          # (N, L, d_num) — real sequences, CPU
    dt_bin: Tensor,         # (N, L) int
    x_cat: Tensor,          # (N, L, d_cat) int
    y: Tensor,              # (N, L) float
    mask: Tensor,           # (N, L) bool — True = real (non-padding)
    tau: Tensor,
    W: float,
    temp: float,
    Bbins: int,
    n_cat_classes: List[int],
    g_std: Tensor,          # (4,) fixed normalization buffer (same as CoFSeqGen)
    steps: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cpu",
) -> List[float]:
    """
    Train f_φ: g/g_std → P(fraud) on real sequences via BCE, then freeze.
    Returns per-step loss history.

    Gate: after training, check val AUPRC > prevalence before using in CoF.
    g_std MUST be the same tensor used in CoFSeqGen (computed from train set once).
    """
    from models.teacher import compute_g_from_real

    f_phi.to(device).unfreeze()
    opt = torch.optim.Adam(f_phi.parameters(), lr=lr)
    g_std_dev = g_std.to(device).clamp(min=1.0)

    # Precompute g_real/g_std on full data (no_grad, batched to avoid OOM)
    # With large n_cat (e.g. 256), soft_g produces (B, L, L, K) tensors — must batch.
    N_full = x_num.shape[0]
    g_chunks = []
    with torch.no_grad():
        for s in range(0, N_full, batch_size):
            e = min(s + batch_size, N_full)
            g_chunk = compute_g_from_real(
                x_num[s:e].to(device), dt_bin[s:e].to(device), x_cat[s:e].to(device),
                tau.to(device), W, temp, Bbins, n_cat_classes,
                valid_mask=mask[s:e].to(device),
            )
            g_chunks.append(g_chunk.cpu())
        g_real   = torch.cat(g_chunks, dim=0).to(device)
        g_real_n = g_real / g_std_dev[None, None, :]   # (N, L, 4)

    # Propagate entity-level label to all positions in that entity.
    # This is necessary when position-level fraud is much rarer than entity-level
    # (e.g. AMLSim: 0.12% position vs 3.64% entity — only 1 fraud position per
    # fraud entity → teacher learns to predict 0 everywhere → AUPRC ≈ random).
    # Entity-label propagation: f_φ learns "does this behavioral position profile
    # belong to a fraud entity?" — which is the correct target for the KL coherence
    # loss (aligning model's fraud prediction with entity behavioral patterns).
    y_entity = (y.float() * mask.float()).max(dim=1).values  # (N,) entity fraud label
    y_ent_expanded = y_entity.unsqueeze(1).expand_as(y)      # (N, L) — propagated
    y_dev    = y_ent_expanded.to(device).float()
    mask_dev = mask.to(device).bool()
    N = x_num.shape[0]

    losses = []
    for _ in range(steps):
        idx  = torch.randint(0, N, (min(batch_size, N),), device=device)
        g_b  = g_real_n[idx]     # (bs, L, 4)
        y_b  = y_dev[idx]        # (bs, L) — entity label at every position
        m_b  = mask_dev[idx]     # (bs, L)

        logit = f_phi(g_b)       # (bs, L)
        loss  = F.binary_cross_entropy_with_logits(
            logit[m_b].float(), y_b[m_b].float()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    f_phi.freeze()
    return losses


# ─── SequenceTeacher (v1c) ────────────────────────────────────────────────────

class SequenceTeacher(nn.Module):
    """
    f_φ_seq: raw window (amount + timing + receiver) → per-position fraud logit (B,L).

    Takes SOFT probability inputs so gradient flows to generator timing/receiver
    heads through L_coh. One-hot is the special case used when training on real data.

    use_time=False, use_recv=False → amount-only variant (C2a attribution baseline).
    use_time=True,  use_recv=True  → full-window variant (C2b signal).
    """
    def __init__(
        self,
        d_num: int,
        Bbins: int,
        K_cat: int,         # vocabulary size for categorical (0 if none)
        h: int = 64,
        n_layers: int = 2,
        use_time: bool = True,
        use_recv: bool = True,
    ):
        super().__init__()
        self.use_time = use_time
        self.use_recv = use_recv
        self.num_proj = nn.Linear(d_num, h)
        if use_time and Bbins > 0:
            self.bin_emb = nn.Embedding(Bbins, h)
        if use_recv and K_cat > 0:
            self.cat_emb = nn.Embedding(K_cat, h)
        self.gru    = nn.GRU(h, h, n_layers, batch_first=True, bidirectional=True)
        self.y_head = nn.Linear(2 * h, 1)

    def forward(
        self,
        amount: Tensor,                    # (B, L, d_num) continuous
        bin_probs: Optional[Tensor],       # (B, L, Bbins) softmax or one-hot
        cat_probs: Optional[Tensor],       # (B, L, K_cat) softmax or one-hot, or None
        mask: Tensor,                      # (B, L) bool — for API compatibility
    ) -> Tensor:
        """Returns per-position logits (B, L). Use [mask] externally to select real positions."""
        tok = self.num_proj(amount)                                       # (B, L, h)
        if self.use_time and bin_probs is not None and hasattr(self, "bin_emb"):
            tok = tok + bin_probs @ self.bin_emb.weight                   # expected bin embed
        if self.use_recv and cat_probs is not None and hasattr(self, "cat_emb"):
            tok = tok + cat_probs @ self.cat_emb.weight                   # expected cat embed
        h, _ = self.gru(tok)                                              # (B, L, 2h)
        return self.y_head(h).squeeze(-1)                                 # (B, L)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)
        self.train()


def pretrain_sequence_teacher(
    f_phi_seq: SequenceTeacher,
    x_num: Tensor,       # (N, L, d_num) real sequences, CPU
    dt_bin: Tensor,      # (N, L) int
    x_cat: Tensor,       # (N, L, d_cat) int
    y: Tensor,           # (N, L) float
    mask: Tensor,        # (N, L) bool
    Bbins: int,
    K_cat: int,          # vocabulary size for cat (0 if none)
    steps: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cpu",
) -> List[float]:
    """
    Train SequenceTeacher on real sequences via entity-propagated BCE, then freeze.
    Returns per-step loss history.

    Real data uses one-hot encoding as the "soft probability" — this is the special
    case of expected-embedding where all probability mass is on one bucket.
    Matches the SequenceTeacher.forward() interface used during CoF training.
    """
    f_phi_seq.to(device).unfreeze()
    opt = torch.optim.Adam(f_phi_seq.parameters(), lr=lr)

    # Entity-level label propagation (same as CoherenceTeacher pretrain)
    y_ent     = (y.float() * mask.float()).max(dim=1).values   # (N,)
    y_ent_exp = y_ent.unsqueeze(1).expand_as(y)                # (N, L)

    x_num_d  = x_num.to(device)
    dt_bin_d = dt_bin.to(device)
    x_cat_d  = x_cat.to(device)
    y_d      = y_ent_exp.to(device).float()
    mask_d   = mask.to(device).bool()
    N = x_num.shape[0]

    losses = []
    for _ in range(steps):
        idx   = torch.randint(0, N, (min(batch_size, N),), device=device)
        xn_b  = x_num_d[idx]
        db_b  = dt_bin_d[idx]
        xc_b  = x_cat_d[idx]
        y_b   = y_d[idx]
        m_b   = mask_d[idx]

        # one-hot = special case of soft probs (expected embedding collapses to lookup)
        bin_p = F.one_hot(db_b, num_classes=Bbins).float()                              # (B,L,Bbins)
        cat_p = F.one_hot(xc_b[:, :, 0], num_classes=K_cat).float() if K_cat > 0 else None

        logit = f_phi_seq(xn_b, bin_p, cat_p, m_b)    # (B, L)
        loss  = F.binary_cross_entropy_with_logits(logit[m_b], y_b[m_b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    f_phi_seq.freeze()
    return losses
