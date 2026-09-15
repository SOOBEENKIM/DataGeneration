"""
SeqDenoiser: sequence-position Transformer denoiser for CoF-SeqGen.

Architecture:
  FeatureTokenizer  → per-position token (B, L, d_model)
  PositionalEncoding → add position embedding
  TimeEmbedding     → broadcast sinusoidal(t) over L
  ClassEmbedding    → broadcast entity fraud label over L (Classifier-Free Guidance)
  TransformerEncoder → self-attention over L (full, not causal — DDPM-style)
  OutputHeads       → x_num_hat, bin_logits, cat_logits, y_logit

Class conditioning (CFG, Ho & Salimans 2022):
  y_emb: Embedding(3, d_model) — 0=non-fraud, 1=fraud, 2=null/unconditional
  y_null weight initialised to zero; learned during training.
  Entity-level label broadcast to all L positions (same fraud flag for the whole sequence).

Input:
  x_num_t: (B, L, d_num)  — noisy numerical features (Gaussian diffusion)
  dt_bin:  (B, L)          — time-bin indices (conditioning, not noised)
  x_cat:   (B, L, d_cat)   — categorical indices (conditioning, not noised)
  t:       (B,)             — diffusion time fraction in [0, 1]
  y_cond:  (B,)  long {0,1,2} or None — entity fraud label; None → unconditional (class 2)

Output:
  x_num_hat:  (B, L, d_num)        — clean numerical prediction
  bin_logits: (B, L, Bbins)        — time-bin logits
  cat_logits: list of (B, L, K_k)  — per-categorical logits (empty list if d_cat=0)
  y_logit:    (B, L)               — fraud prediction logit (label head)
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Tuple


class SeqDenoiser(nn.Module):
    def __init__(
        self,
        d_num: int,
        Bbins: int,
        n_cat_classes: List[int],
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        L_max: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.Bbins = Bbins
        self.n_cat_classes = n_cat_classes

        # ── Timestep embedding ──────────────────────────────────────────────
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        # ── Feature tokenizer ───────────────────────────────────────────────
        self.num_proj = nn.Linear(d_num, d_model)
        # +1 for MASK token: id=Bbins for bin, id=K for each cat
        self.bin_emb = nn.Embedding(Bbins + 1, d_model)
        self.cat_embs = nn.ModuleList([nn.Embedding(K + 1, d_model) for K in n_cat_classes])

        # Learned position encoding (up to L_max positions)
        self.pos_emb = nn.Embedding(L_max, d_model)

        # ── Sequence Transformer ─────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── Class conditioning (CFG) ─────────────────────────────────────────
        # 0=non-fraud, 1=fraud, 2=null/unconditional token
        self.y_emb = nn.Embedding(3, d_model)
        nn.init.zeros_(self.y_emb.weight[2])   # null token starts at zero

        # ── Output heads ─────────────────────────────────────────────────────
        self.num_head = nn.Linear(d_model, d_num)
        self.bin_head = nn.Linear(d_model, Bbins)
        self.cat_heads = nn.ModuleList([nn.Linear(d_model, K) for K in n_cat_classes])
        self.y_head = nn.Linear(d_model, 1)

    @staticmethod
    def _sinusoidal(t: Tensor, dim: int) -> Tensor:
        """t: (B,) ∈ [0,1] → (B, dim)."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        x_num_t: Tensor,                          # (B, L, d_num)
        dt_bin: Tensor,                            # (B, L) int
        x_cat: Tensor,                             # (B, L, d_cat) int  (may have d_cat=0)
        t: Tensor,                                 # (B,) float in [0,1]
        src_key_padding_mask: Tensor = None,       # (B, L) bool — True = pad
        y_cond: Optional[Tensor] = None,           # (B,) long {0,1,2} or None → null token
    ) -> Tuple[Tensor, Tensor, List[Tensor], Tensor]:
        B, L, _ = x_num_t.shape

        # Timestep embedding (B, d_model)
        t_emb = self._sinusoidal(t, self.d_model)
        t_emb = self.t_mlp(t_emb)   # (B, d_model)

        # Feature tokens: sum contributions from all feature types
        h = self.num_proj(x_num_t) + self.bin_emb(dt_bin)   # (B, L, d_model)
        for i, emb in enumerate(self.cat_embs):
            if x_cat.shape[-1] > i:
                h = h + emb(x_cat[:, :, i])

        # Add position encoding and broadcast timestep
        pos = torch.arange(L, device=x_num_t.device)
        h = h + self.pos_emb(pos)[None]          # (B, L, d_model)
        h = h + t_emb[:, None, :]                # (B, L, d_model)

        # Class conditioning: broadcast entity fraud label over all positions
        if y_cond is not None:
            y_e = self.y_emb(y_cond.long().clamp(0, 2))   # (B, d_model)
        else:
            y_e = self.y_emb.weight[2].unsqueeze(0).expand(B, -1)  # null token
        h = h + y_e[:, None, :]                  # (B, L, d_model)

        # Sequence Transformer over L positions
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)  # (B, L, d_model)

        # Output heads
        x_num_hat = self.num_head(h)                            # (B, L, d_num)
        bin_logits = self.bin_head(h)                           # (B, L, Bbins)
        cat_logits = [head(h) for head in self.cat_heads]       # list of (B, L, K_k)
        y_logit = self.y_head(h).squeeze(-1)                    # (B, L)

        return x_num_hat, bin_logits, cat_logits, y_logit
