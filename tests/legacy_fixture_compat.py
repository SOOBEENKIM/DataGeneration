"""In-memory compatibility for preserved legacy left-padded sequence fixtures."""

from __future__ import annotations

import torch


def to_right_padding(x_num, dt_bin, x_cat, y, mask):
    """Move valid tokens to the prefix without modifying fixture files."""
    if torch.equal(
        mask,
        torch.arange(mask.shape[1], device=mask.device)[None]
        < mask.sum(1)[:, None],
    ):
        return x_num, dt_bin, x_cat, y, mask
    out_num = torch.zeros_like(x_num)
    out_bin = torch.zeros_like(dt_bin)
    out_cat = torch.zeros_like(x_cat)
    out_y = torch.zeros_like(y)
    out_mask = torch.zeros_like(mask)
    for index in range(len(mask)):
        length = int(mask[index].sum())
        out_num[index, :length] = x_num[index, mask[index]]
        out_bin[index, :length] = dt_bin[index, mask[index]]
        out_cat[index, :length] = x_cat[index, mask[index]]
        out_y[index, :length] = y[index, mask[index]]
        out_mask[index, :length] = True
    return out_num, out_bin, out_cat, out_y, out_mask
