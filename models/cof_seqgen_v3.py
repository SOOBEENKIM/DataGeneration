"""CoF-SeqGen v3 joint gap/receiver architecture.

This module contains model components only.  It does not perform device
selection, training, sampling runs, evaluation, or artifact writes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class JointStateCodec:
    """Bijection between ``(gap_bin, receiver_code)`` and one joint state."""

    gap_bins: int
    receiver_classes: int

    def __post_init__(self) -> None:
        if self.gap_bins < 1 or self.receiver_classes < 1:
            raise ValueError("joint state dimensions must be positive")

    @property
    def state_count(self) -> int:
        return self.gap_bins * self.receiver_classes

    @property
    def mask_state(self) -> int:
        return self.state_count

    def encode(self, gap: Tensor, receiver: Tensor) -> Tensor:
        if gap.shape != receiver.shape:
            raise ValueError("gap and receiver shapes must match")
        if gap.dtype != torch.long or receiver.dtype != torch.long:
            raise TypeError("gap and receiver must be torch.long")
        if bool(((gap < 0) | (gap >= self.gap_bins)).any()):
            raise ValueError("gap is outside the preregistered support")
        if bool(
            (
                (receiver < 0)
                | (receiver >= self.receiver_classes)
            ).any()
        ):
            raise ValueError("receiver is outside the preregistered support")
        return gap * self.receiver_classes + receiver

    def decode(self, joint_state: Tensor) -> tuple[Tensor, Tensor]:
        if joint_state.dtype != torch.long:
            raise TypeError("joint state must be torch.long")
        if bool(
            ((joint_state < 0) | (joint_state >= self.state_count)).any()
        ):
            raise ValueError("joint state is outside the Cartesian support")
        return (
            torch.div(
                joint_state,
                self.receiver_classes,
                rounding_mode="floor",
            ),
            torch.remainder(joint_state, self.receiver_classes),
        )


def corrupt_joint_state(
    joint_state: Tensor,
    *,
    valid_mask: Tensor,
    mask_probability: float,
    mask_state: int,
    random_values: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply one absorbing-state mask to a complete gap/receiver pair."""

    if valid_mask.dtype != torch.bool or valid_mask.shape != joint_state.shape:
        raise ValueError("valid_mask must be bool with joint-state shape")
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError("mask_probability must be in [0, 1]")
    if random_values is None:
        random_values = torch.rand(
            joint_state.shape,
            device=joint_state.device,
        )
    if random_values.shape != joint_state.shape:
        raise ValueError("random_values must have joint-state shape")
    pair_mask = (random_values < mask_probability) & valid_mask
    corrupted = torch.where(
        pair_mask,
        torch.full_like(joint_state, int(mask_state)),
        joint_state,
    )
    return corrupted, pair_mask


class DirectJointDiscretePath(nn.Module):
    """One embedding and one normalized head for a joint discrete state."""

    def __init__(self, *, codec: JointStateCodec, d_model: int) -> None:
        super().__init__()
        self.codec = codec
        self.joint_embedding = nn.Embedding(
            codec.state_count + 1,
            d_model,
        )
        self.joint_head = nn.Linear(d_model, codec.state_count)

    def embed(self, joint_state: Tensor) -> Tensor:
        return self.joint_embedding(joint_state)

    def logits(self, hidden: Tensor) -> Tensor:
        return self.joint_head(hidden)

    @staticmethod
    def probabilities(logits: Tensor) -> Tensor:
        return F.softmax(logits, dim=-1)

    def sample_and_decode(
        self,
        logits: Tensor,
        *,
        valid_mask: Tensor,
        temperature: float,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if (
            valid_mask.dtype != torch.bool
            or valid_mask.shape != logits.shape[:-1]
        ):
            raise ValueError("valid_mask must match the logit prefix")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if temperature == 0:
            joint = logits.argmax(dim=-1)
        else:
            probabilities = F.softmax(logits / temperature, dim=-1)
            joint = torch.multinomial(
                probabilities.reshape(-1, self.codec.state_count),
                1,
                generator=generator,
            ).reshape(valid_mask.shape)
        joint = torch.where(valid_mask, joint, torch.zeros_like(joint))
        gap, receiver = self.codec.decode(joint)
        return joint, gap, receiver


class FactorizedJointDiscretePath(nn.Module):
    """Normalized ``p(gap) p(receiver | gap)`` discrete path."""

    def __init__(self, *, codec: JointStateCodec, d_model: int) -> None:
        super().__init__()
        self.codec = codec
        self.joint_embedding = nn.Embedding(
            codec.state_count + 1,
            d_model,
        )
        self.gap_head = nn.Linear(d_model, codec.gap_bins)
        self.gap_condition_embedding = nn.Embedding(
            codec.gap_bins,
            d_model,
        )
        self.receiver_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, codec.receiver_classes),
        )

    def embed(self, joint_state: Tensor) -> Tensor:
        return self.joint_embedding(joint_state)

    def gap_logits(self, hidden: Tensor) -> Tensor:
        return self.gap_head(hidden)

    @staticmethod
    def gap_probabilities(logits: Tensor) -> Tensor:
        return F.softmax(logits, dim=-1)

    def receiver_logits(self, hidden: Tensor, gap: Tensor) -> Tensor:
        if gap.shape != hidden.shape[:-1] or gap.dtype != torch.long:
            raise ValueError("conditioning gap must be long with hidden prefix")
        if bool(((gap < 0) | (gap >= self.codec.gap_bins)).any()):
            raise ValueError("conditioning gap is outside train support")
        conditioning = self.gap_condition_embedding(gap)
        return self.receiver_head(torch.cat([hidden, conditioning], dim=-1))

    def receiver_probabilities(
        self,
        hidden: Tensor,
        gap: Tensor,
    ) -> Tensor:
        return F.softmax(self.receiver_logits(hidden, gap), dim=-1)

    def joint_probabilities(self, hidden: Tensor) -> Tensor:
        gap_probability = self.gap_probabilities(self.gap_logits(hidden))
        receiver_by_gap = torch.stack(
            [
                self.receiver_probabilities(
                    hidden,
                    torch.full(
                        hidden.shape[:-1],
                        gap,
                        dtype=torch.long,
                        device=hidden.device,
                    ),
                )
                for gap in range(self.codec.gap_bins)
            ],
            dim=-2,
        )
        return gap_probability.unsqueeze(-1) * receiver_by_gap

    @staticmethod
    def _choice(
        logits: Tensor,
        *,
        temperature: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if temperature == 0:
            return logits.argmax(dim=-1)
        probabilities = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]),
            1,
            generator=generator,
        ).reshape(logits.shape[:-1])

    def sample_and_decode(
        self,
        hidden: Tensor,
        *,
        valid_mask: Tensor,
        temperature: float,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if (
            valid_mask.dtype != torch.bool
            or valid_mask.shape != hidden.shape[:-1]
        ):
            raise ValueError("valid_mask must match the hidden prefix")
        gap = self._choice(
            self.gap_logits(hidden),
            temperature=temperature,
            generator=generator,
        )
        receiver = self._choice(
            self.receiver_logits(hidden, gap),
            temperature=temperature,
            generator=generator,
        )
        gap = torch.where(valid_mask, gap, torch.zeros_like(gap))
        receiver = torch.where(
            valid_mask,
            receiver,
            torch.zeros_like(receiver),
        )
        return gap, receiver


@dataclass(frozen=True)
class V3DenoiserOutput:
    amount_hat: Tensor
    y_logit: Tensor
    hidden: Tensor
    joint_logits: Tensor | None = None
    gap_logits: Tensor | None = None
    receiver_logits: Tensor | None = None


class CoFSeqDenoiserV3(nn.Module):
    """Legacy CoF sequence backbone with a v3 joint discrete path."""

    CANDIDATES = ("direct_joint", "factorized_joint")

    def __init__(
        self,
        *,
        d_num: int,
        codec: JointStateCodec,
        candidate: str,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_length: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if candidate not in self.CANDIDATES:
            raise ValueError("unknown CoF-SeqGen v3 candidate")
        self.codec = codec
        self.candidate = candidate
        self.d_model = d_model
        self.amount_projection = nn.Linear(d_num, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.label_embedding = nn.Embedding(3, d_model)
        nn.init.zeros_(self.label_embedding.weight[2])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        self.amount_head = nn.Linear(d_model, d_num)
        self.y_head = nn.Linear(d_model, 1)
        if candidate == "direct_joint":
            self.discrete_path: nn.Module = DirectJointDiscretePath(
                codec=codec,
                d_model=d_model,
            )
        else:
            self.discrete_path = FactorizedJointDiscretePath(
                codec=codec,
                d_model=d_model,
            )

    @staticmethod
    def _sinusoidal(time: Tensor, dimension: int) -> Tensor:
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half,
                device=time.device,
                dtype=time.dtype,
            )
            / max(half - 1, 1)
        )
        arguments = time[:, None] * frequencies[None]
        result = torch.cat(
            [torch.sin(arguments), torch.cos(arguments)],
            dim=-1,
        )
        if result.shape[-1] < dimension:
            result = F.pad(result, (0, dimension - result.shape[-1]))
        return result

    def encode_hidden(
        self,
        x_num_t: Tensor,
        joint_state: Tensor,
        time: Tensor,
        *,
        valid_mask: Tensor,
        y_cond: Tensor | None,
    ) -> Tensor:
        batch, length, _ = x_num_t.shape
        if (
            joint_state.shape != (batch, length)
            or valid_mask.dtype != torch.bool
            or valid_mask.shape != (batch, length)
            or time.shape != (batch,)
        ):
            raise ValueError("v3 denoiser tensor shapes are invalid")
        discrete = self.discrete_path.embed(joint_state)
        positions = torch.arange(length, device=x_num_t.device)
        hidden = self.amount_projection(x_num_t) + discrete
        hidden = hidden + self.position_embedding(positions)[None]
        hidden = hidden + self.time_mlp(
            self._sinusoidal(time, self.d_model)
        )[:, None, :]
        if y_cond is None:
            label = self.label_embedding.weight[2][None].expand(batch, -1)
        else:
            if y_cond.shape != (batch,):
                raise ValueError("entity labels must have batch shape")
            label = self.label_embedding(y_cond.long().clamp(0, 2))
        hidden = hidden + label[:, None, :]
        return self.transformer(
            hidden,
            src_key_padding_mask=~valid_mask,
        )

    def forward(
        self,
        x_num_t: Tensor,
        joint_state: Tensor,
        time: Tensor,
        *,
        valid_mask: Tensor,
        y_cond: Tensor | None,
        teacher_gap: Tensor | None = None,
    ) -> V3DenoiserOutput:
        hidden = self.encode_hidden(
            x_num_t,
            joint_state,
            time,
            valid_mask=valid_mask,
            y_cond=y_cond,
        )
        amount_hat = self.amount_head(hidden)
        y_logit = self.y_head(hidden).squeeze(-1)
        if self.candidate == "direct_joint":
            path = self.discrete_path
            assert isinstance(path, DirectJointDiscretePath)
            return V3DenoiserOutput(
                amount_hat=amount_hat,
                y_logit=y_logit,
                hidden=hidden,
                joint_logits=path.logits(hidden),
            )
        if teacher_gap is None:
            raise ValueError(
                "factorized training requires teacher-forced true gap"
            )
        path = self.discrete_path
        assert isinstance(path, FactorizedJointDiscretePath)
        return V3DenoiserOutput(
            amount_hat=amount_hat,
            y_logit=y_logit,
            hidden=hidden,
            gap_logits=path.gap_logits(hidden),
            receiver_logits=path.receiver_logits(hidden, teacher_gap),
        )


class CoFSeqGenV3(nn.Module):
    """Trainable v3 objective with one paired gap/receiver corruption."""

    AMOUNT_CONTRACT = (
        "train_fitted_centered_empirical_residual_257_v2_8_frozen"
    )

    def __init__(
        self,
        *,
        denoiser: CoFSeqDenoiserV3,
        codec: JointStateCodec,
        joint_support_mask: Tensor,
        joint_support_sha256: str,
        coherence_lambda: float,
        cfg_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if denoiser.codec != codec:
            raise ValueError("denoiser and model joint codecs differ")
        if (
            joint_support_mask.dtype != torch.bool
            or joint_support_mask.shape != (codec.state_count,)
            or not bool(joint_support_mask.any())
        ):
            raise ValueError("joint support mask is invalid")
        if (
            not isinstance(joint_support_sha256, str)
            or len(joint_support_sha256) != 64
        ):
            raise ValueError("joint support hash is invalid")
        if coherence_lambda != 0.0:
            raise ValueError("v3 architecture isolation fixes coherence_lambda=0")
        if not 0.0 <= cfg_dropout <= 1.0:
            raise ValueError("cfg_dropout must be in [0, 1]")
        self.denoiser = denoiser
        self.codec = codec
        self.joint_support_sha256 = joint_support_sha256
        self.coherence_lambda = 0.0
        self.cfg_dropout = cfg_dropout
        self.register_buffer(
            "joint_support_mask",
            joint_support_mask.clone(),
        )

    @staticmethod
    def _alpha_sigma(time_fraction: float) -> tuple[float, float]:
        if not 0.0 <= time_fraction <= 1.0:
            raise ValueError("time_fraction must be in [0, 1]")
        return (
            math.cos(math.pi * time_fraction / 2),
            math.sin(math.pi * time_fraction / 2),
        )

    def _mask_direct_logits(self, logits: Tensor) -> Tensor:
        return logits.masked_fill(
            ~self.joint_support_mask.to(logits.device),
            float("-inf"),
        )

    def _support_matrix(self, device: torch.device) -> Tensor:
        return self.joint_support_mask.to(device).reshape(
            self.codec.gap_bins,
            self.codec.receiver_classes,
        )

    def compute_loss(
        self,
        *,
        x_num: Tensor,
        gap: Tensor,
        receiver: Tensor,
        y: Tensor,
        valid_mask: Tensor,
        time_fraction: float,
        corruption_random_values: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float | int]]:
        batch, length, _ = x_num.shape
        if (
            gap.shape != (batch, length)
            or receiver.shape != gap.shape
            or y.shape != gap.shape
            or valid_mask.dtype != torch.bool
            or valid_mask.shape != gap.shape
            or not bool(valid_mask.any())
        ):
            raise ValueError("v3 loss tensors are invalid")
        alpha, sigma = self._alpha_sigma(time_fraction)
        clean_amount = x_num * valid_mask[..., None]
        amount_noise = torch.randn_like(clean_amount)
        x_num_t = (
            alpha * clean_amount + sigma * amount_noise
        ) * valid_mask[..., None]
        clean_joint = self.codec.encode(gap, receiver)
        corrupted_joint, pair_mask = corrupt_joint_state(
            clean_joint,
            valid_mask=valid_mask,
            mask_probability=min(time_fraction, 0.7),
            mask_state=self.codec.mask_state,
            random_values=corruption_random_values,
        )
        y_entity = (
            (y * valid_mask.float()).max(dim=1).values > 0
        ).long()
        y_cond = y_entity.clone()
        if self.training and self.cfg_dropout > 0:
            drop = (
                torch.rand(batch, device=x_num.device) < self.cfg_dropout
            )
            y_cond[drop] = 2
        time = torch.full(
            (batch,),
            time_fraction,
            dtype=x_num.dtype,
            device=x_num.device,
        )
        output = self.denoiser(
            x_num_t,
            corrupted_joint,
            time,
            valid_mask=valid_mask,
            y_cond=y_cond,
            teacher_gap=(
                gap
                if self.denoiser.candidate == "factorized_joint"
                else None
            ),
        )
        amount_loss = F.mse_loss(
            output.amount_hat[valid_mask],
            clean_amount[valid_mask],
        )
        discrete_mask = pair_mask if bool(pair_mask.any()) else valid_mask
        if self.denoiser.candidate == "direct_joint":
            assert output.joint_logits is not None
            joint_logits = self._mask_direct_logits(output.joint_logits)
            discrete_loss = F.cross_entropy(
                joint_logits[discrete_mask],
                clean_joint[discrete_mask],
            )
        else:
            assert output.gap_logits is not None
            assert output.receiver_logits is not None
            support = self._support_matrix(output.gap_logits.device)
            allowed_gap = support.any(dim=-1)
            gap_logits = output.gap_logits.masked_fill(
                ~allowed_gap,
                float("-inf"),
            )
            receiver_support = support[gap]
            receiver_logits = output.receiver_logits.masked_fill(
                ~receiver_support,
                float("-inf"),
            )
            discrete_loss = F.cross_entropy(
                gap_logits[discrete_mask],
                gap[discrete_mask],
            ) + F.cross_entropy(
                receiver_logits[discrete_mask],
                receiver[discrete_mask],
            )
        label_loss = F.binary_cross_entropy_with_logits(
            output.y_logit[valid_mask].float(),
            y[valid_mask].float(),
        )
        total = amount_loss + discrete_loss + label_loss
        return total, {
            "amount_loss": float(amount_loss.detach()),
            "discrete_loss": float(discrete_loss.detach()),
            "label_loss": float(label_loss.detach()),
            "coherence_loss": 0.0,
            "paired_mask_positions": int(pair_mask.sum().item()),
        }

    def initial_joint_state(self, valid_mask: Tensor) -> Tensor:
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool")
        return torch.where(
            valid_mask,
            torch.full_like(
                valid_mask,
                self.codec.mask_state,
                dtype=torch.long,
            ),
            torch.zeros_like(valid_mask, dtype=torch.long),
        )

    def sample_discrete_from_hidden(
        self,
        hidden: Tensor,
        *,
        valid_mask: Tensor,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Decode one paired state without any post-sampling calibration."""

        path = self.denoiser.discrete_path
        if isinstance(path, DirectJointDiscretePath):
            logits = self._mask_direct_logits(path.logits(hidden))
            _, gap, receiver = path.sample_and_decode(
                logits,
                valid_mask=valid_mask,
                temperature=temperature,
                generator=generator,
            )
            return gap, receiver
        assert isinstance(path, FactorizedJointDiscretePath)
        support = self._support_matrix(hidden.device)
        gap_logits = path.gap_logits(hidden).masked_fill(
            ~support.any(dim=-1),
            float("-inf"),
        )
        gap = path._choice(
            gap_logits,
            temperature=temperature,
            generator=generator,
        )
        receiver_logits = path.receiver_logits(hidden, gap).masked_fill(
            ~support[gap],
            float("-inf"),
        )
        receiver = path._choice(
            receiver_logits,
            temperature=temperature,
            generator=generator,
        )
        return (
            torch.where(valid_mask, gap, torch.zeros_like(gap)),
            torch.where(
                valid_mask,
                receiver,
                torch.zeros_like(receiver),
            ),
        )
