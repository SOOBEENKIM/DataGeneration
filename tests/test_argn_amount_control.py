from types import SimpleNamespace

import torch
from torch import nn

from benchmarks.argn_amount_control import AmountHeads, conditional_objective


def parent():
    layers = nn.ModuleList([nn.Linear(5, 4), nn.Linear(4, 3)])
    return SimpleNamespace(regressors=SimpleNamespace(get=lambda k: layers, dropout=nn.Dropout(.25)),
                           predictors=SimpleNamespace(predictors={"amount": nn.Linear(3, 2)}))


def test_clones_reproduce_native_and_do_not_modify_parent():
    torch.manual_seed(1)
    p = parent(); x = torch.randn(8, 5); labels = torch.arange(8) % 2
    expected = p.predictors.predictors["amount"](torch.relu(p.regressors.get("")[1](p.regressors.get("")[0](x))))
    for kind in ["shared", "mixture", "routed"]:
        h = AmountHeads(p, ["amount"], kind).eval()
        torch.testing.assert_close(h(x, "amount", labels).softmax(-1), expected.softmax(-1))
        with torch.no_grad():
            h.predictors[0]["amount"].weight.add_(1)
        torch.testing.assert_close(expected, p.predictors.predictors["amount"](torch.relu(p.regressors.get("")[1](p.regressors.get("")[0](x)))))


def test_routing_is_label_local_and_capacity_matches_mixture():
    p = parent(); routed = AmountHeads(p, ["amount"], "routed").eval()
    mixture = AmountHeads(p, ["amount"], "mixture").eval()
    assert sum(x.numel() for x in routed.parameters()) == sum(x.numel() for x in mixture.parameters())
    x = torch.randn(8, 5); y = torch.arange(8) % 2
    old = routed(x, "amount", y).detach()
    with torch.no_grad():
        routed.predictors[1]["amount"].bias[1].add_(2)
    new = routed(x, "amount", y)
    torch.testing.assert_close(new[y == 0], old[y == 0])
    assert not torch.equal(new[y == 1], old[y == 1])
    new[y == 0].sum().backward()
    assert routed.predictors[1]["amount"].weight.grad.abs().sum() == 0


def test_stratified_objective_does_not_change_natural_prior():
    loss = torch.tensor([1., 3., 10., 14.]); y = torch.tensor([0, 0, 1, 1])
    torch.testing.assert_close(conditional_objective(loss, y, .01, False), torch.tensor(2.10))
    torch.testing.assert_close(conditional_objective(loss, y, .01, True), torch.tensor(7.))
