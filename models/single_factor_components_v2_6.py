from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from models.candidate_components_v2_6 import (
    WeightedCoFSeqGenV26,
    _tempered_choice,
)
from models.coherence_teacher import SequenceTeacher
from models.sampler import _cosine_alpha_sigma
from models.soft_g import soft_g


class EpsilonPredictionCoFSeqGenV26(WeightedCoFSeqGenV26):
    """Single-factor CoF amount objective with an epsilon target.

    The discrete corruption, receiver/gap/label losses, architecture, channel
    weights and coherence path are inherited semantically from the frozen
    v2.6 candidate. Only the denoiser amount head target changes from clean
    x0 to the forward-process epsilon.
    """

    amount_parameterization = "epsilon_prediction"

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
        x_num_t, epsilon = self.add_noise(x_num, t_frac)
        alpha, sigma = self._alpha_sigma(t_frac)
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
                torch.full_like(
                    x_cat[:, :, channel],
                    categories,
                ),
                x_cat[:, :, channel],
            )
        entity_label = (
            (y * mask.float()).max(dim=1).values > 0
        ).long()
        y_cond = entity_label.clone()
        if self.cfg_dropout > 0 and self.training:
            dropped = (
                torch.rand(batch, device=x_num.device)
                < self.cfg_dropout
            )
            y_cond[dropped] = 2
        (
            epsilon_hat,
            gap_logits,
            cat_logits,
            y_logit,
        ) = self.denoiser(
            x_num_t,
            dt_corrupt,
            cat_corrupt,
            t_vec,
            src_key_padding_mask=~mask,
            y_cond=y_cond,
        )
        clean_hat = (
            x_num_t - sigma * epsilon_hat
        ) / max(float(alpha), 1e-6)
        clean_hat = clean_hat * mask[..., None]
        gap_probabilities = F.softmax(gap_logits, dim=-1)
        category_probabilities = (
            F.softmax(cat_logits[0], dim=-1)
            if cat_logits
            else None
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
                clean_hat[:, :, 0],
                self.tau,
                self.W,
                self.temp,
                valid_mask=mask,
            )
        else:
            generated_summary = None
        use_mask = bool(mask.any().item())
        if use_mask:
            amount_loss = F.mse_loss(
                epsilon_hat[mask],
                epsilon[mask],
            )
            gap_loss = F.cross_entropy(
                gap_logits[mask].reshape(
                    -1,
                    self.denoiser.Bbins,
                ),
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
            amount_loss = F.mse_loss(epsilon_hat, epsilon)
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
                    clean_hat,
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
            "amount_parameterization": self.amount_parameterization,
        }


@torch.no_grad()
def epsilon_reverse_sample_v2_6(
    model: EpsilonPredictionCoFSeqGenV26,
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
    """Deterministic VP reverse path matching an epsilon-prediction head."""
    model.eval()
    batch, length = dt_bin_cond.shape
    if (
        valid_mask.dtype is not torch.bool
        or valid_mask.shape != (batch, length)
        or steps < 1
    ):
        raise ValueError("invalid epsilon reverse-sampling contract")
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
    # alpha(1.0) is exactly zero for the frozen cosine schedule. Starting one
    # fixed epsilon below 1 keeps the matching x0 reconstruction finite.
    schedule = torch.linspace(
        1.0 - 1e-4,
        0.0,
        steps + 1,
        device=device,
    )
    numerical = (
        torch.randn(batch, length, d_num, device=device)
        * valid_mask[..., None]
    )
    final_gap_logits = None
    final_cat_logits = None
    for index in range(steps):
        current = float(schedule[index].item())
        following = float(schedule[index + 1].item())
        alpha_current, sigma_current = _cosine_alpha_sigma(
            current
        )
        alpha_following, sigma_following = _cosine_alpha_sigma(
            following
        )
        time_vector = torch.full(
            (batch,),
            current,
            device=device,
        )
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
            epsilon_hat = unconditioned[0] + guidance_scale * (
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
            (
                epsilon_hat,
                gap_logits,
                cat_logits,
                _,
            ) = model.denoiser(
                numerical,
                dt_bin_cond,
                x_cat_cond,
                time_vector,
                y_cond=y_cond,
                src_key_padding_mask=~valid_mask,
            )
        clean_hat = (
            numerical - sigma_current * epsilon_hat
        ) / max(float(alpha_current), 1e-6)
        clean_hat = clean_hat * valid_mask[..., None]
        numerical = (
            alpha_following * clean_hat
            + sigma_following * epsilon_hat
        ) * valid_mask[..., None]
        if (
            feedback_discrete
            and current <= 1.0 - feedback_after
        ):
            dt_bin_cond = torch.where(
                valid_mask,
                _tempered_choice(
                    gap_logits,
                    feedback_temperature,
                ),
                0,
            )
            x_cat_cond = torch.stack(
                [
                    torch.where(
                        valid_mask,
                        _tempered_choice(
                            logit,
                            feedback_temperature,
                        ),
                        0,
                    )
                    for logit in cat_logits
                ],
                dim=-1,
            )
        final_gap_logits = gap_logits
        final_cat_logits = cat_logits
    assert final_gap_logits is not None
    assert final_cat_logits is not None
    decode_temperature = (
        0.0 if final_temperature is None else final_temperature
    )
    gap = torch.where(
        valid_mask,
        _tempered_choice(
            final_gap_logits,
            decode_temperature,
        ),
        0,
    )
    categories = [
        torch.where(
            valid_mask,
            _tempered_choice(logit, decode_temperature),
            0,
        )
        for logit in final_cat_logits
    ]
    return numerical, gap, categories


def variance_preserving_residual(
    amount: Tensor,
    *,
    valid_mask: Tensor,
    train_variance: float,
    seed: int,
) -> tuple[Tensor, Mapping[str, float | int | str]]:
    """Add a train-fitted zero-mean residual without validation refitting."""
    if (
        amount.shape[:-1] != valid_mask.shape
        or valid_mask.dtype is not torch.bool
        or train_variance < 0
    ):
        raise ValueError("invalid residual-sampling input")
    base_variance = float(
        amount[..., 0][valid_mask].double().var(
            unbiased=False
        ).item()
    )
    residual_variance = max(
        float(train_variance) - base_variance,
        0.0,
    )
    generator = torch.Generator(device=amount.device)
    generator.manual_seed(int(seed))
    residual = torch.randn(
        amount.shape,
        generator=generator,
        device=amount.device,
        dtype=amount.dtype,
    ) * residual_variance**0.5
    result = amount + residual * valid_mask[..., None]
    return result, {
        "schema_version": (
            "benchmark-v2.6-train-fitted-residual-v1"
        ),
        "fit_split": "train",
        "train_variance": float(train_variance),
        "base_variance": base_variance,
        "residual_variance": residual_variance,
        "residual_mean": 0.0,
        "seed": int(seed),
    }
