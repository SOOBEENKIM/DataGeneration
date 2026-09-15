"""
CoFSeqGen v1: entity-sequence diffusion with label-behavioral KL coherence.

Changes from v0:
  - L_coh: g-MSE → KL(P_model(Y|·) ‖ f_φ(g(x̂_0)))
    f_φ = CoherenceTeacher: g → P(fraud), trained on real data, frozen.
  - coherence_teacher replaces BehaviorTeacher.
  - g_gen from x̂_0 retains gradient throughout → coherence loss aligns both
    label head (via P_model) and behavior heads (via q → g_gen → soft_g).

Loss (MASTER §7):
  L = L_diff + λ · L_coh
  L_diff = MSE(x_num_hat, x_num) + CE(bin) + Σ CE(cat) + 0.1 · BCE(label)
  L_coh  = KL(P_model ‖ f_φ(g(x̂_0)/g_std))  — Bernoulli KL, mask-reduced

Gradient flow in L_coh:
  L_coh → P_model → y_logit → label head        (align label output)
  L_coh → q → q_logit → g_gen → soft_g         (align behavior output)
  Teacher params stay frozen; grad flows only through g_gen_n.

Noise schedule: VP-SDE cosine, α(t)=cos(πt/2). x_num only noised.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union

from models.soft_g import soft_g
from models.seq_denoiser import SeqDenoiser
from models.coherence_teacher import CoherenceTeacher, SequenceTeacher


COH_LAMBDA_SWEEP  = [0.0, 0.1, 0.5, 1.0, 2.0]
W_SPARKOV_MINUTES = 60.0
W_AMLSIM_STEPS   = 7.0


class CoFSeqGen(nn.Module):
    def __init__(
        self,
        denoiser: SeqDenoiser,
        tau: Tensor,                   # (Bbins,) median Δt per bin
        W: float,                      # window width in τ units
        temp: float,                   # soft_g sigmoid temperature
        coh_lambda: float,             # λ weight for L_coh (0 = no coherence)
        n_cat_classes: List[int],
        coherence_teacher: Optional[Union[CoherenceTeacher, SequenceTeacher]] = None,
        g_std: Optional[Tensor] = None,  # (4,) for CoherenceTeacher path only
        cfg_dropout: float = 0.15,       # CFG: fraction of training steps that use null label
    ):
        super().__init__()
        self.denoiser = denoiser
        self.register_buffer("tau", tau.float())
        self.W = W
        self.temp = temp
        self.coh_lambda = coh_lambda
        self.n_cat_classes = n_cat_classes
        self.coherence_teacher = coherence_teacher
        self.cfg_dropout = cfg_dropout

        if g_std is not None:
            self.register_buffer("g_std", g_std.float().clamp(min=1.0))
        else:
            self.g_std = None

    # ── Noise schedule ────────────────────────────────────────────────────────

    @staticmethod
    def _alpha_sigma(t_frac: float) -> Tuple[float, float]:
        """VP cosine: x_t = α·x_0 + σ·ε, α²+σ²=1."""
        alpha = math.cos(math.pi * t_frac / 2)
        sigma = math.sin(math.pi * t_frac / 2)
        return alpha, sigma

    @staticmethod
    def add_noise(x_num: Tensor, t_frac: float) -> Tuple[Tensor, Tensor]:
        alpha, sigma = CoFSeqGen._alpha_sigma(t_frac)
        eps = torch.randn_like(x_num)
        return alpha * x_num + sigma * eps, eps

    # ── KL helper ─────────────────────────────────────────────────────────────

    @staticmethod
    def _bern_kl(p: Tensor, q: Tensor, eps: float = 1e-6) -> Tensor:
        """Bernoulli KL(p ‖ q), element-wise. Gradient flows through both p and q."""
        p = p.clamp(eps, 1.0 - eps)
        q = q.clamp(eps, 1.0 - eps)
        return p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))

    # ── Loss computation ──────────────────────────────────────────────────────

    def compute_loss(
        self,
        x_num: Tensor,    # (B, L, d_num)  clean
        dt_bin: Tensor,   # (B, L) int
        x_cat: Tensor,    # (B, L, d_cat) int
        y: Tensor,        # (B, L) float
        mask: Tensor,     # (B, L) bool  True=real
        t_frac: float,
    ) -> Tuple[Tensor, Dict[str, float]]:
        B, L, d_num = x_num.shape

        # Forward diffusion on numericals
        x_num_t, _ = self.add_noise(x_num, t_frac)
        t_vec = torch.full((B,), t_frac, device=x_num.device)

        # Absorbing-state masking for discrete channels (p_mask = min(t_frac, 0.7))
        # Cap at 0.7 to avoid 100% masking (t_frac=1.0) which destabilises CE loss.
        # Model must GENERATE dt_bin/x_cat from MASK given y_cond — no echo shortcut.
        p_mask = min(t_frac, 0.7)
        MASK_B = self.denoiser.Bbins                              # MASK token id for bins
        dt_mask = (torch.rand(B, L, device=x_num.device) < p_mask) & mask
        dt_bin_c = torch.where(dt_mask, torch.full_like(dt_bin, MASK_B), dt_bin)
        x_cat_c = x_cat.clone()
        for k, K in enumerate(self.n_cat_classes):
            MASK_K = K                                            # MASK token id for cat k
            cat_mask = (torch.rand(B, L, device=x_num.device) < p_mask) & mask
            x_cat_c[:, :, k] = torch.where(cat_mask,
                                            torch.full_like(x_cat[:, :, k], MASK_K),
                                            x_cat[:, :, k])

        # Class conditioning: entity-level fraud label (max over real positions)
        y_ent = ((y * mask.float()).max(dim=1).values > 0).long()   # (B,) {0,1}
        # CFG dropout: replace cfg_dropout fraction with null token (class 2)
        y_cond = y_ent.clone()
        if self.cfg_dropout > 0 and self.training:
            drop = torch.rand(B, device=x_num.device) < self.cfg_dropout
            y_cond[drop] = 2   # null/unconditional token

        # Denoiser forward with corrupted discrete inputs
        pad_mask = ~mask
        x_num_hat, bin_logits, cat_logits, y_logit = self.denoiser(
            x_num_t, dt_bin_c, x_cat_c, t_vec,
            src_key_padding_mask=pad_mask,
            y_cond=y_cond,
        )

        # Softmax probs: always needed for L_coh (SequenceTeacher or soft_g path)
        bin_probs = F.softmax(bin_logits, dim=-1)                         # (B, L, Bbins)
        cat_probs = F.softmax(cat_logits[0], dim=-1) if cat_logits else None  # (B, L, K)

        # soft_g: only needed for CoherenceTeacher (g-MLP) path.
        # SequenceTeacher receives raw denoiser outputs directly.
        # NO detach on either path: grad must flow to behavior heads through L_coh.
        _use_seq_teacher = isinstance(self.coherence_teacher, SequenceTeacher)
        if not _use_seq_teacher and self.coh_lambda > 0 and self.coherence_teacher is not None:
            amt_hat = x_num_hat[:, :, 0]
            g_gen = soft_g(
                bin_probs,
                cat_probs,
                amt_hat,
                self.tau,
                self.W,
                self.temp,
                valid_mask=mask,
            )
        else:
            g_gen = None  # not used; avoids expensive soft_g when using SequenceTeacher

        # ── L_diff ───────────────────────────────────────────────────────────
        use_mask = bool(mask.any().item())
        if use_mask:
            L_diff_num = F.mse_loss(x_num_hat[mask], x_num[mask])
            L_diff_bin = F.cross_entropy(
                bin_logits[mask].reshape(-1, self.denoiser.Bbins),
                dt_bin[mask].reshape(-1),
            )
            L_diff_cat = (
                sum(
                    F.cross_entropy(
                        cat_logits[k][mask].reshape(-1, self.n_cat_classes[k]),
                        x_cat[:, :, k][mask].reshape(-1),
                    )
                    for k in range(len(cat_logits))
                )
                if cat_logits
                else x_num.new_zeros(1).squeeze()
            )
            L_label = F.binary_cross_entropy_with_logits(
                y_logit[mask].float(), y[mask].float()
            )
        else:
            L_diff_num = F.mse_loss(x_num_hat, x_num)
            L_diff_bin = F.cross_entropy(
                bin_logits.reshape(-1, self.denoiser.Bbins), dt_bin.reshape(-1)
            )
            L_diff_cat = x_num.new_zeros(1).squeeze()
            L_label = F.binary_cross_entropy_with_logits(
                y_logit.float(), y.float()
            )

        L_diff = L_diff_num + L_diff_bin + L_diff_cat + 1.0 * L_label

        # ── L_coh: KL(P_model ‖ f_φ(window or g)) ───────────────────────────
        # Two paths depending on teacher type (both frozen, no detach):
        #   SequenceTeacher: q_logit = f_φ_seq(x_num_hat, bin_probs, cat_probs, mask) → (B,L)
        #     Grad flows: L_coh → q → expected-embed → bin_probs/cat_probs → bin/cat heads (C2b!)
        #   CoherenceTeacher: q_logit = f_φ(g_gen/g_std) → flat (N_real,)
        #     Grad flows: L_coh → q → g_gen → soft_g → behavior heads (v1b path)
        L_coh = x_num.new_zeros(1).squeeze()
        if self.coh_lambda > 0 and self.coherence_teacher is not None:
            if _use_seq_teacher:
                # SequenceTeacher: soft expected-embedding inputs, per-position output
                q_logit = self.coherence_teacher(x_num_hat, bin_probs, cat_probs, mask)  # (B,L)
                P_model = torch.sigmoid(y_logit)   # (B,L) grad → label head
                q       = torch.sigmoid(q_logit)   # (B,L) grad → bin/cat/num heads
                if use_mask:
                    L_coh = self.coh_lambda * self._bern_kl(P_model[mask], q[mask]).mean()
                else:
                    L_coh = self.coh_lambda * self._bern_kl(
                        P_model.reshape(-1), q.reshape(-1)
                    ).mean()
            elif g_gen is not None and self.g_std is not None:
                # CoherenceTeacher (g-MLP, v1b path)
                g_std_buf = self.g_std  # (4,), requires_grad=False
                if use_mask:
                    g_gen_n   = g_gen[mask]          / g_std_buf
                    y_logit_m = y_logit[mask]
                else:
                    g_gen_n   = g_gen.reshape(-1, 4) / g_std_buf
                    y_logit_m = y_logit.reshape(-1)
                q_logit = self.coherence_teacher(g_gen_n)   # (N_real,)
                P_model = torch.sigmoid(y_logit_m)
                q       = torch.sigmoid(q_logit)
                L_coh = self.coh_lambda * self._bern_kl(P_model, q).mean()

        total = L_diff + L_coh

        return total, {
            "L_diff":  L_diff.item(),
            "L_coh":   L_coh.item() if isinstance(L_coh, Tensor) else float(L_coh),
            "L_label": L_label.item(),
        }
