import torch

from models.sampler import ddim_sample


class Denoiser(torch.nn.Module):
    Bbins = 3
    n_cat_classes = [4]

    def __init__(self):
        super().__init__()
        self.masks = []

    def forward(self, x, dt, cats, t, src_key_padding_mask=None, y_cond=None):
        self.masks.append(src_key_padding_mask.clone())
        b, length, d = x.shape
        return (
            torch.ones_like(x),
            torch.zeros(b, length, self.Bbins),
            [torch.zeros(b, length, self.n_cat_classes[0])],
            torch.zeros(b, length),
        )


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = Denoiser()


def test_sampler_passes_mask_every_step_and_preserves_padding():
    model = Model()
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    result = ddim_sample(
        model,
        torch.zeros(2, 5, dtype=torch.long),
        torch.zeros(2, 5, 1, dtype=torch.long),
        d_num=1,
        T_steps=3,
        start_from_mask=True,
        feedback_discrete=True,
        valid_mask=mask,
    )
    x, dt, cats, _, _ = result
    assert len(model.denoiser.masks) == 3
    assert all(torch.equal(value, ~mask) for value in model.denoiser.masks)
    assert torch.all(x[~mask] == 0)
    assert torch.all(dt[~mask] == 0)
    assert torch.all(cats[0][~mask] == 0)
