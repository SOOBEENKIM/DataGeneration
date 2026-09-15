from __future__ import annotations

from typing import Mapping

import torch

from models.candidate_components_v2_6 import (
    _cosine_alpha_sigma,
    _tempered_choice,
)


@torch.no_grad()
def ddim_sample_gap_bias_v2_7(
    model,
    dt_bin_cond,
    x_cat_cond,
    d_num,
    *,
    steps,
    device,
    y_cond,
    guidance_scale,
    feedback_discrete,
    feedback_after,
    feedback_temperature,
    final_temperature,
    start_from_mask,
    valid_mask,
    gap_logit_bias_by_label: Mapping[int, tuple[float, ...]],
):
    """Frozen v2.6 DDIM path with one train-fitted gap-logit factor.

    The bias is applied after classifier-free guidance, so it is neither
    multiplied by guidance nor inserted into any amount/receiver path.
    """
    model.eval()
    batch, length = dt_bin_cond.shape
    if valid_mask.dtype is not torch.bool or valid_mask.shape != (
        batch,
        length,
    ):
        raise ValueError("valid_mask must be bool [B,L]")
    bins = int(model.denoiser.Bbins)
    if set(gap_logit_bias_by_label) != {0, 1} or any(
        len(row) != bins for row in gap_logit_bias_by_label.values()
    ):
        raise ValueError("v2.7 gap-logit bias shape mismatch")
    dt_bin_cond = dt_bin_cond.to(device)
    x_cat_cond = x_cat_cond.to(device)
    valid_mask = valid_mask.to(device)
    if start_from_mask:
        dt_bin_cond = torch.where(
            valid_mask,
            torch.full_like(dt_bin_cond, bins),
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
    bias_table = torch.zeros(
        (3, bins),
        dtype=torch.float32,
        device=device,
    )
    for label in (0, 1):
        bias_table[label] = torch.as_tensor(
            gap_logit_bias_by_label[label],
            dtype=torch.float32,
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
        gap_logits = gap_logits + bias_table[y_cond][:, None, :]
        noise_hat = (
            numerical - alpha_current * amount_hat
        ) / (sigma_current + 1e-8)
        amount_hat = amount_hat * valid_mask[..., None]
        numerical = (
            alpha_following * amount_hat
            + sigma_following * noise_hat
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
    if final_gap_logits is None or final_cat_logits is None:
        raise RuntimeError("v2.7 DDIM produced no decode logits")
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
