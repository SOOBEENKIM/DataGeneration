"""Meaningful boundaries for a frozen, observed-target probability adapter."""
import copy
import numpy as np
import pytest
import torch
from scipy.special import expit

from experiments.cs_saf_calibration import (contract, parent, load_parent, calibrated_model,
    extract_train_features, base_digest)
from experiments.cs_saf_pilot import subset, batch, state_digest
from models.cs_saf_calibration import repeat_nll_and_gradient, fit_repeat_calibration


@pytest.fixture(scope='module')
def case():
    torch.set_num_threads(1)
    p=parent.load_cache(parent.CACHE/'pi_0.05_kappa_1.pt')
    base,_=load_parent(p,.05,1,0,'E',torch.device('cpu'))
    return p,base,subset(p['train'],3)


def test_analytic_gradient_against_finite_differences_and_bernoulli():
    z=np.array([-8.,-1.,0.,2.,7.]);f=np.array([.01,.5,.1,.02,.6]);y=np.array([1.,0.,1.,1.,0.])
    theta=np.array([.4,1.2]);value,gradient=repeat_nll_and_gradient(theta,z,f,y)
    r=f+(1-f)*expit(theta[0]+theta[1]*z)
    assert value==pytest.approx(-np.mean(y*np.log(r)+(1-y)*np.log1p(-r)),abs=1e-11)
    for j in range(2):
        delta=np.eye(2)[j]*1e-5
        numeric=(repeat_nll_and_gradient(theta+delta,z,f,y)[0]-repeat_nll_and_gradient(theta-delta,z,f,y)[0])/2e-5
        assert gradient[j]==pytest.approx(numeric,abs=1e-8)


def test_extreme_logits_are_finite():
    loss,g=repeat_nll_and_gradient([8.,20.],np.array([-100.,100.]),np.array([.01,.01]),np.array([1.,0.]))
    assert np.isfinite(loss) and np.isfinite(g).all()


def test_repeat_loss_has_exact_mark_nll_parameter_dependence():
    z=np.array([-.2,.5,1.]);f=np.array([.1,.2,.3]);y=np.array([1.,0.,0.]);actual_f=np.array([.1,.05,.4])
    def mark_nll(theta):
        q=expit(theta[0]+theta[1]*z)
        return -np.log(np.where(y,q+(1-q)*f,(1-q)*actual_f)).mean()
    a=np.array([0.,1.]);b=np.array([.3,1.7])
    assert mark_nll(b)-mark_nll(a)==pytest.approx(
        repeat_nll_and_gradient(b,z,f,y)[0]-repeat_nll_and_gradient(a,z,f,y)[0])


def test_identity_and_nonidentity_distribution_and_first_events(case):
    p,base,train=case;model=calibrated_model(base,p,'cpu')
    x=batch(train,torch.arange(6),'cpu')
    with torch.no_grad():
        c=base.context(base.encoder(**x),x['static_categorical'])
        prev=x['receiver'].roll(1,dims=1);prev[:,0]=1
        has=x['valid_mask'].clone();has[:,0]=False
        codes=x['static_categorical'][0][:,None].expand_as(has)
        expected=base.mark_distribution(c,x['gap'],prev,has,static_codes=codes)[0]
        identity=model.mark_distribution(c,x['gap'],prev,has,static_codes=codes)[0]
        torch.testing.assert_close(identity,expected,atol=0,rtol=0)
        model.set_calibration(dict(offset=[.2,-.4],slope=[1.3,.7]))
        actual,lr,ln=model.mark_distribution(c,x['gap'],prev,has,static_codes=codes)
        torch.testing.assert_close(actual.exp().sum(-1),torch.ones_like(has,dtype=torch.float32),atol=2e-6,rtol=0)
        assert torch.isneginf(actual[...,:3]).all()
        torch.testing.assert_close(actual[:,0],expected[:,0],rtol=0,atol=0)
        repeat=actual.gather(-1,prev[...,None])[...,0].exp()
        torch.testing.assert_close(repeat[has],lr.exp()[has],atol=2e-7,rtol=1e-6)
        torch.testing.assert_close((lr.exp()+ln.exp())[has],torch.ones_like(lr[has]),atol=2e-7,rtol=1e-6)
    assert base_digest(model)==state_digest(base)
    assert all(not p.requires_grad for p in model.parameters())


def test_train_extraction_alignment_and_future_causality(case):
    p,base,train=case
    features=extract_train_features(base,train,'cpu',2)
    changed=copy.deepcopy(train)
    changed['receiver'][:,10:]=3
    changed['numeric_value'][:,10:]+=20
    changed['gap'][:,10:]=.5
    other=extract_train_features(base,changed,'cpu',2)
    before=features['event_index']<10
    for k in features:np.testing.assert_array_equal(features[k][before],other[k][before])
    row=features['entity_index'];col=features['event_index']
    np.testing.assert_array_equal(features['equality'],
        (train['receiver'][row,col]==train['receiver'][row,col-1]).numpy())
    np.testing.assert_array_equal(features['code'],train['codes'][row].numpy())
    assert len(row)==int((train['lengths']-1).sum())


def test_fit_is_deterministic_positive_and_improves_observed_training_loss(case):
    _,base,train=case;features=extract_train_features(base,train,'cpu',2)
    opts=contract()['optimizer'];a=fit_repeat_calibration(features,opts);b=fit_repeat_calibration(features,opts)
    assert a==b and min(a['slope'])>0
    assert all(x['after_nll']<=x['before_nll']+1e-10 for x in a['contexts'].values())
    assert not a['oracle_targets_used'] and not a['active_mask_used']


def test_positive_slopes_and_reserved_context_boundary(case):
    p,base,_=case;model=calibrated_model(base,p,'cpu')
    with pytest.raises(ValueError):model.set_calibration(dict(offset=[0,0],slope=[-1,1]))
    with pytest.raises(ValueError):model.set_calibration(dict(offset=[float('nan'),0],slope=[1,1]))
    with pytest.raises(ValueError):model.slots(torch.tensor([0,3]),(2,))


def test_registration_is_fixed_and_contains_all_cells():
    c=contract()
    assert len(c['prevalences'])*len(c['kappas'])*len(c['trials'])*len(c['parents'])==80
    assert not c['scientific_failure_stops_grid'] and c['fit_split']=='train'
