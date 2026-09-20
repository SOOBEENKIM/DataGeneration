"""Equal intervention on U: original architecture, causal law, frozen tensors."""
import pytest
import torch
from experiments.cs_saf_calibration_u import (contract, parent, load_parent,
    calibrated_model, base_digest, batch, subset, state_digest)
from experiments.cs_saf_calibration import contract as previous_contract


@pytest.fixture(scope='module')
def case():
    torch.set_num_threads(1)
    p = parent.load_cache(parent.CACHE/'pi_0.05_kappa_1.pt')
    base, _ = load_parent(p,.05,1,0,'U','cpu')
    return p, base, subset(p['train'],3)


def test_same_registered_intervention_and_no_history_capacity(case):
    p, base, _ = case
    assert contract()['optimizer'] == previous_contract()['optimizer']
    model = calibrated_model(base,p,'cpu')
    assert set(dict(base.named_parameters())) == set(dict(model.named_parameters()))
    assert not hasattr(model,'history_interaction_weight')
    assert set(model.state_dict())-set(base.state_dict()) == {'calibration_offset','calibration_slope'}
    assert base_digest(model) == state_digest(base)
    assert all(not v.requires_grad for v in model.parameters())


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_identity_and_affine_law_with_first_event_and_audit(case,device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('GPU gate covers CUDA separately')
    p, _, train = case
    base, _ = load_parent(p,.05,1,0,'U',device)
    model = calibrated_model(base,p,device)
    x = batch(train,torch.arange(6),device)
    with torch.no_grad():
        c = base.context(base.encoder(**x),x['static_categorical'])
        prev = x['receiver'].roll(1,dims=1); prev[:,0] = 1
        has = x['valid_mask'].clone(); has[:,0] = False
        codes = x['static_categorical'][0][:,None].expand_as(has)
        expected = base.mark_distribution(c,x['gap'],prev,has,static_codes=codes)[0]
        actual = model.mark_distribution(c,x['gap'],prev,has,static_codes=codes)[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for zero in (False,True):
            a = base.response_curves(c[has],prev[has],static_codes=codes[has],zero_gap=zero)
            b = model.response_curves(c[has],prev[has],static_codes=codes[has],zero_gap=zero)
            for av,bv in zip(a,b):torch.testing.assert_close(av,bv,atol=0,rtol=0)
        model.set_calibration(dict(offset=[.3,-.2],slope=[.7,1.4]))
        actual = model.mark_distribution(c,x['gap'],prev,has,static_codes=codes)[0]
        torch.testing.assert_close(actual[:,0],expected[:,0],rtol=0,atol=0)
        slots = codes[has]-3
        raw = base.copy_logits(c[has],x['gap'][has],static_codes=codes[has])
        q = (model.calibration_offset[slots]+model.calibration_slope[slots]*raw).sigmoid()
        fresh = base.new_mark_head(c[has]); fresh[:,:3] = -torch.inf
        f = fresh.softmax(-1)
        target = (1-q[:,None])*f
        target.scatter_add_(1,prev[has][:,None],q[:,None])
        torch.testing.assert_close(actual[has].exp(),target,atol=2e-7,rtol=1e-6)
        grid,_ = model.response_curves(c[has],prev[has],static_codes=codes[has])
        for j,gap in enumerate(base.support.representatives):
            z = base.copy_logits(c[has],torch.full_like(raw,gap),static_codes=codes[has])
            expected_q = (model.calibration_offset[slots]+model.calibration_slope[slots]*z).sigmoid()
            torch.testing.assert_close(grid[:,j],expected_q,atol=2e-7,rtol=1e-6)
        q0,r0 = model.response_curves(c[has],prev[has],static_codes=codes[has],zero_gap=True)
        assert torch.equal(q0.max(1).values,q0.min(1).values)
        assert torch.equal(r0.max(1).values,r0.min(1).values)
    assert base_digest(model) == state_digest(base)
