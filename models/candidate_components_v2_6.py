from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from models.coherence_teacher import SequenceTeacher
from models.cof_seqgen import CoFSeqGen
from models.sampler import _cosine_alpha_sigma
from models.soft_g import soft_g


def weighted_tvae_loss_v2_6(
    reconstructed: Tensor,
    real: Tensor,
    sigmas: Tensor,
    mu: Tensor,
    logvar: Tensor,
    output_info,
    factor: float,
    channel_weights: Mapping[str, float],
) -> tuple[Tensor, Tensor]:
    """TVAE likelihood with preregistered raw-column channel weights."""
    names = ("amount", "gap", "receiver")
    if len(output_info) != len(names):
        raise ValueError("v2.6 TVAE expects amount, gap, receiver columns")
    start = 0
    terms: list[Tensor] = []
    for column_index, column_info in enumerate(output_info):
        weight = float(channel_weights[names[column_index]])
        for span_info in column_info:
            end = start + span_info.dim
            if span_info.activation_fn != "softmax":
                std = sigmas[start]
                error = real[:, start] - torch.tanh(
                    reconstructed[:, start]
                )
                terms.append(
                    weight * (error**2 / 2 / (std**2)).sum()
                )
                terms.append(weight * torch.log(std) * real.size(0))
            else:
                terms.append(
                    weight
                    * F.cross_entropy(
                        reconstructed[:, start:end],
                        torch.argmax(real[:, start:end], dim=-1),
                        reduction="sum",
                    )
                )
            start = end
    if start != reconstructed.size(1):
        raise ValueError("TVAE transformed span accounting mismatch")
    divergence = -0.5 * torch.sum(
        1 + logvar - mu**2 - logvar.exp()
    )
    return (
        sum(terms) * factor / real.size(0),
        divergence / real.size(0),
    )


class WeightedCoFSeqGenV26(CoFSeqGen):
    """CoF model with v2.6-preregistered per-channel loss weights."""

    def __init__(self, *args, channel_weights: Mapping[str, float], **kwargs):
        super().__init__(*args, **kwargs)
        required = {"amount", "gap", "receiver", "label"}
        if set(channel_weights) != required:
            raise ValueError("CoF v2.6 requires four explicit loss weights")
        self.v2_6_channel_weights = {
            key: float(channel_weights[key]) for key in sorted(required)
        }
        if any(value <= 0 for value in self.v2_6_channel_weights.values()):
            raise ValueError("CoF v2.6 loss weights must be positive")

    def compute_loss(
        self,
        x_num: Tensor,
        dt_bin: Tensor,
        x_cat: Tensor,
        y: Tensor,
        mask: Tensor,
        t_frac: float,
    ):
        batch, length, _ = x_num.shape
        x_num_t, _ = self.add_noise(x_num, t_frac)
        t_vec = torch.full((batch,), t_frac, device=x_num.device)
        mask_probability = min(t_frac, 0.7)
        gap_mask = (
            torch.rand(batch, length, device=x_num.device)
            < mask_probability
        ) & mask
        dt_corrupt = torch.where(
            gap_mask,
            torch.full_like(dt_bin, self.denoiser.Bbins),
            dt_bin,
        )
        cat_corrupt = x_cat.clone()
        for channel, categories in enumerate(self.n_cat_classes):
            channel_mask = (
                torch.rand(batch, length, device=x_num.device)
                < mask_probability
            ) & mask
            cat_corrupt[:, :, channel] = torch.where(
                channel_mask,
                torch.full_like(x_cat[:, :, channel], categories),
                x_cat[:, :, channel],
            )
        entity_label = (
            (y * mask.float()).max(dim=1).values > 0
        ).long()
        y_cond = entity_label.clone()
        if self.cfg_dropout > 0 and self.training:
            dropped = (
                torch.rand(batch, device=x_num.device) < self.cfg_dropout
            )
            y_cond[dropped] = 2
        amount_hat, gap_logits, cat_logits, y_logit = self.denoiser(
            x_num_t,
            dt_corrupt,
            cat_corrupt,
            t_vec,
            src_key_padding_mask=~mask,
            y_cond=y_cond,
        )
        gap_probabilities = F.softmax(gap_logits, dim=-1)
        category_probabilities = (
            F.softmax(cat_logits[0], dim=-1) if cat_logits else None
        )
        sequence_teacher = isinstance(
            self.coherence_teacher,
            SequenceTeacher,
        )
        if (
            not sequence_teacher
            and self.coh_lambda > 0
            and self.coherence_teacher is not None
        ):
            generated_summary = soft_g(
                gap_probabilities,
                category_probabilities,
                amount_hat[:, :, 0],
                self.tau,
                self.W,
                self.temp,
                valid_mask=mask,
            )
        else:
            generated_summary = None
        use_mask = bool(mask.any().item())
        if use_mask:
            amount_loss = F.mse_loss(amount_hat[mask], x_num[mask])
            gap_loss = F.cross_entropy(
                gap_logits[mask].reshape(-1, self.denoiser.Bbins),
                dt_bin[mask].reshape(-1),
            )
            receiver_loss = (
                sum(
                    F.cross_entropy(
                        cat_logits[channel][mask].reshape(
                            -1,
                            self.n_cat_classes[channel],
                        ),
                        x_cat[:, :, channel][mask].reshape(-1),
                    )
                    for channel in range(len(cat_logits))
                )
                if cat_logits
                else x_num.new_zeros(())
            )
            label_loss = F.binary_cross_entropy_with_logits(
                y_logit[mask].float(),
                y[mask].float(),
            )
        else:
            amount_loss = F.mse_loss(amount_hat, x_num)
            gap_loss = F.cross_entropy(
                gap_logits.reshape(-1, self.denoiser.Bbins),
                dt_bin.reshape(-1),
            )
            receiver_loss = (
                sum(
                    F.cross_entropy(
                        cat_logits[channel].reshape(
                            -1,
                            self.n_cat_classes[channel],
                        ),
                        x_cat[:, :, channel].reshape(-1),
                    )
                    for channel in range(len(cat_logits))
                )
                if cat_logits
                else x_num.new_zeros(())
            )
            label_loss = F.binary_cross_entropy_with_logits(
                y_logit.float(),
                y.float(),
            )
        weights = self.v2_6_channel_weights
        diffusion_loss = (
            weights["amount"] * amount_loss
            + weights["gap"] * gap_loss
            + weights["receiver"] * receiver_loss
            + weights["label"] * label_loss
        )
        coherence_loss = x_num.new_zeros(())
        if self.coh_lambda > 0 and self.coherence_teacher is not None:
            if sequence_teacher:
                teacher_logit = self.coherence_teacher(
                    amount_hat,
                    gap_probabilities,
                    category_probabilities,
                    mask,
                )
                model_probability = torch.sigmoid(y_logit)
                teacher_probability = torch.sigmoid(teacher_logit)
                if use_mask:
                    coherence_loss = self.coh_lambda * self._bern_kl(
                        model_probability[mask],
                        teacher_probability[mask],
                    ).mean()
                else:
                    coherence_loss = self.coh_lambda * self._bern_kl(
                        model_probability.reshape(-1),
                        teacher_probability.reshape(-1),
                    ).mean()
            elif generated_summary is not None and self.g_std is not None:
                if use_mask:
                    summary = generated_summary[mask] / self.g_std
                    model_logit = y_logit[mask]
                else:
                    summary = (
                        generated_summary.reshape(-1, 4) / self.g_std
                    )
                    model_logit = y_logit.reshape(-1)
                teacher_probability = torch.sigmoid(
                    self.coherence_teacher(summary)
                )
                model_probability = torch.sigmoid(model_logit)
                coherence_loss = self.coh_lambda * self._bern_kl(
                    model_probability,
                    teacher_probability,
                ).mean()
        total = diffusion_loss + coherence_loss
        return total, {
            "L_diff": float(diffusion_loss.detach().cpu()),
            "L_coh": float(coherence_loss.detach().cpu()),
            "L_label": float(label_loss.detach().cpu()),
            "L_amount": float(amount_loss.detach().cpu()),
            "L_gap": float(gap_loss.detach().cpu()),
            "L_receiver": float(receiver_loss.detach().cpu()),
        }


def _tempered_choice(logits: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(
        probabilities.reshape(-1, probabilities.size(-1)),
        1,
    ).reshape(logits.shape[:-1])


@torch.no_grad()
def ddim_sample_v2_6(
    model: CoFSeqGen,
    dt_bin_cond: Tensor,
    x_cat_cond: Tensor,
    d_num: int,
    *,
    steps: int,
    device: str,
    y_cond: Tensor,
    guidance_scale: float,
    feedback_discrete: bool,
    feedback_after: float,
    feedback_temperature: float,
    final_temperature: float | None,
    start_from_mask: bool,
    valid_mask: Tensor,
):
    model.eval()
    batch, length = dt_bin_cond.shape
    if valid_mask.dtype is not torch.bool or valid_mask.shape != (
        batch,
        length,
    ):
        raise ValueError("valid_mask must be bool [B,L]")
    dt_bin_cond = dt_bin_cond.to(device)
    x_cat_cond = x_cat_cond.to(device)
    valid_mask = valid_mask.to(device)
    if start_from_mask:
        dt_bin_cond = torch.where(
            valid_mask,
            torch.full_like(dt_bin_cond, model.denoiser.Bbins),
            0,
        )
        x_cat_cond = torch.stack(
            [
                torch.where(
                    valid_mask,
                    torch.full_like(
                        x_cat_cond[:, :, channel],
                        categories,
                    ),
                    0,
                )
                for channel, categories in enumerate(
                    model.denoiser.n_cat_classes
                )
            ],
            dim=-1,
        )
    y_cond = y_cond.to(device)
    null_cond = torch.full(
        (batch,),
        2,
        dtype=torch.long,
        device=device,
    )
    schedule = torch.linspace(1.0, 0.0, steps + 1, device=device)
    numerical = (
        torch.randn(batch, length, d_num, device=device)
        * valid_mask[..., None]
    )
    final_gap_logits = None
    final_cat_logits = None
    for index in range(steps):
        current = float(schedule[index].item())
        following = float(schedule[index + 1].item())
        alpha_current, sigma_current = _cosine_alpha_sigma(current)
        alpha_following, sigma_following = _cosine_alpha_sigma(following)
        time_vector = torch.full((batch,), current, device=device)
        if guidance_scale > 1:
            conditioned = model.denoiser(
                numerical,
                dt_bin_cond,
                x_cat_cond,
                time_vector,
                y_cond=y_cond,
                src_key_padding_mask=~valid_mask,
            )
            unconditioned = model.denoiser(
                numerical,
                dt_bin_cond,
                x_cat_cond,
                time_vector,
                y_cond=null_cond,
                src_key_padding_mask=~valid_mask,
            )
            amount_hat = unconditioned[0] + guidance_scale * (
                conditioned[0] - unconditioned[0]
            )
            gap_logits = unconditioned[1] + guidance_scale * (
                conditioned[1] - unconditioned[1]
            )
            cat_logits = [
                unconditioned_logit
                + guidance_scale
                * (conditioned_logit - unconditioned_logit)
                for conditioned_logit, unconditioned_logit in zip(
                    conditioned[2],
                    unconditioned[2],
                )
            ]
        else:
            amount_hat, gap_logits, cat_logits, _ = model.denoiser(
                numerical,
                dt_bin_cond,
                x_cat_cond,
                time_vector,
                y_cond=y_cond,
                src_key_padding_mask=~valid_mask,
            )
        noise_hat = (
            numerical - alpha_current * amount_hat
        ) / (sigma_current + 1e-8)
        amount_hat = amount_hat * valid_mask[..., None]
        numerical = (
            alpha_following * amount_hat + sigma_following * noise_hat
        ) * valid_mask[..., None]
        if (
            feedback_discrete
            and current <= 1.0 - feedback_after
        ):
            dt_bin_cond = torch.where(
                valid_mask,
                _tempered_choice(gap_logits, feedback_temperature),
                0,
            )
            x_cat_cond = torch.stack(
                [
                    torch.where(
                        valid_mask,
                        _tempered_choice(logit, feedback_temperature),
                        0,
                    )
                    for logit in cat_logits
                ],
                dim=-1,
            )
        final_gap_logits = gap_logits
        final_cat_logits = cat_logits
    assert final_gap_logits is not None and final_cat_logits is not None
    final_decode_temperature = (
        0.0 if final_temperature is None else final_temperature
    )
    gap = torch.where(
        valid_mask,
        _tempered_choice(
            final_gap_logits,
            final_decode_temperature,
        ),
        0,
    )
    categories = [
        torch.where(
            valid_mask,
            _tempered_choice(logit, final_decode_temperature),
            0,
        )
        for logit in final_cat_logits
    ]
    return numerical, gap, categories
