import torch
import torch.nn.functional as F

from models.soft_g import soft_g


def run(bp, cp, amount, mask):
    tau = torch.tensor([0.5, 1.0, 2.0], dtype=bp.dtype)
    return soft_g(bp, cp, amount, tau, 4.0, 1.0, valid_mask=mask)


def inputs():
    torch.manual_seed(1)
    bp = F.softmax(torch.randn(2, 6, 3), -1).requires_grad_()
    cp = F.softmax(torch.randn(2, 6, 4), -1).requires_grad_()
    amount = torch.randn(2, 6, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    return bp, cp, amount, mask


def test_padded_values_do_not_change_valid_g():
    bp, cp, amount, mask = inputs()
    expected = run(bp, cp, amount, mask)
    bp2, cp2, amount2 = bp.clone(), cp.clone(), amount.clone()
    bp2[~mask], cp2[~mask], amount2[~mask] = 99, -99, 123
    actual = run(bp2, cp2, amount2, mask)
    assert torch.allclose(expected[mask], actual[mask])


def test_padded_targets_are_zero():
    bp, cp, amount, mask = inputs()
    assert torch.equal(run(bp, cp, amount, mask)[~mask], torch.zeros((5, 4)))


def test_padding_sources_excluded_from_all_history_channels():
    bp, cp, amount, mask = inputs()
    base = run(bp, cp, amount, mask)
    changed = run(bp.masked_fill((~mask)[..., None], 1e4), cp.masked_fill((~mask)[..., None], 1e4), amount.masked_fill(~mask, 1e4), mask)
    assert torch.allclose(base[mask], changed[mask])


def test_valid_gradients_nonzero_and_padded_gradients_zero():
    bp, cp, amount, mask = inputs()
    run(bp, cp, amount, mask).square().sum().backward()
    assert bp.grad[mask].abs().sum() > 0
    assert cp.grad[mask].abs().sum() > 0
    assert amount.grad[mask].abs().sum() > 0
    assert bp.grad[~mask].abs().sum() == 0
    assert cp.grad[~mask].abs().sum() == 0
    assert amount.grad[~mask].abs().sum() == 0


def test_left_right_padding_invariance():
    torch.manual_seed(2)
    bp = F.softmax(torch.randn(1, 4, 3), -1)
    cp = F.softmax(torch.randn(1, 4, 4), -1)
    amount = torch.randn(1, 4)
    valid = torch.ones(1, 4, dtype=torch.bool)
    core = run(bp, cp, amount, valid)
    right = run(F.pad(bp, (0, 0, 0, 2)), F.pad(cp, (0, 0, 0, 2)), F.pad(amount, (0, 2)), torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool))
    left_bp, left_cp, left_amt = F.pad(bp, (0, 0, 2, 0)), F.pad(cp, (0, 0, 2, 0)), F.pad(amount, (2, 0))
    left = run(left_bp, left_cp, left_amt, torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.bool))
    assert torch.allclose(core, right[:, :4])
    assert torch.allclose(core, left[:, 2:])


def test_all_valid_matches_legacy_formula_shape():
    bp, cp, amount, _ = inputs()
    out = run(bp, cp, amount, torch.ones(2, 6, dtype=torch.bool))
    assert out.shape == (2, 6, 4)
    assert torch.isfinite(out).all()
