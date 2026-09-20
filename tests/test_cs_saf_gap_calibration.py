import numpy as np
import pytest
import torch
from scipy.optimize import check_grad
from models.cs_saf_gap_calibration import objective_gradient,fit_gap_offsets
from experiments.cs_saf_gap_calibration import (contract,repeats,adapter,train_bins,extract_features,old_digest,batch,subset,state_digest)


def test_analytic_gradient_including_centered_ridge():
    rng=np.random.default_rng(7);n=200
    args=(rng.normal(size=n),rng.uniform(.001,.1,n),rng.integers(2,size=n),np.arange(n)%5,np.full(5,.2),.001)
    delta=np.array([-.2,.1,.15,-.1,.05])
    error=check_grad(lambda d:objective_gradient(d,*args)[0],lambda d:objective_gradient(d,*args)[1],delta)
    assert error<1e-7


def test_constrained_recovery_bounds_and_determinism():
    # Deterministic fractional repeat rates encoded as binary observed repetitions.
    b=np.repeat(np.arange(5),1000);y=np.concatenate([np.r_[np.ones(n),np.zeros(1000-n)] for n in [380,450,510,560,620]])
    features=dict(logit=np.zeros(10000),fresh_previous=np.full(10000,.02),equality=np.tile(y,2),code=np.repeat([3,4],5000),bin=np.tile(b,2))
    c=contract();r=fit_gap_offsets(features,np.full((2,5),.2),c)
    assert r==fit_gap_offsets(features,np.full((2,5),.2),c)
    d=np.array(r['delta']);assert np.abs(d.mean(1)).max()<1e-9;assert np.abs(d).max()<=.5+1e-9
    assert (np.diff(d,axis=1)>0).all()
    assert all(v['after_nll']<v['before_nll'] for v in r['groups'].values())


@pytest.fixture(scope='module')
def case():
    torch.set_num_threads(1);p=repeats.parent.load_cache(repeats.parent.CACHE/'pi_0.05_kappa_1.pt')
    train=subset(p['train'],4);models={}
    for name in ('Ucal','Ecal'):
        base,_,_=repeats.load_fixed(p,.05,1,0,name,'cpu');m=adapter(base,p,name,'cpu')
        bins=train_bins(base,p['train']);m.set_gap_correction(bins['mapping'],bins['weights'],np.zeros((2,5)))
        models[name]=(base,m,bins)
    return p,train,models


def test_shared_train_map_and_exact_old_tensor_identity(case):
    _,_,models=case
    assert models['Ucal'][2]==models['Ecal'][2]
    for base,m,bins in models.values():
        assert old_digest(m)==state_digest(base)
        assert set(dict(m.named_parameters()))==set(dict(base.named_parameters()))
        assert not any(v.requires_grad for v in m.parameters())


@pytest.mark.parametrize('name',['Ucal','Ecal'])
def test_zero_identity_and_factual_grid_law_zero_control_and_first_event(case,name):
    p,train,models=case;base,m,bins=models[name];x=batch(train,torch.arange(len(train['lengths'])),'cpu')
    with torch.no_grad():
        context=m.context(m.encoder(**x),x['static_categorical']);mask=x['valid_mask'].clone();mask[:,0]=False
        prev=x['receiver'].roll(1,1);prev[:,0]=1;codes=x['static_categorical'][0][:,None].expand_as(mask)
        lp=base.mark_distribution(context,x['gap'],prev,mask,static_codes=codes)[0]
        ident=m.mark_distribution(context,x['gap'],prev,mask,static_codes=codes)[0]
        torch.testing.assert_close(lp,ident,rtol=0,atol=0)
        for zero in (True,False):
            a=base.response_curves(context[mask],prev[mask],static_codes=codes[mask],zero_gap=zero)
            b=m.response_curves(context[mask],prev[mask],static_codes=codes[mask],zero_gap=zero)
            for av,bv in zip(a,b):torch.testing.assert_close(av,bv,rtol=0,atol=0)
        weights=np.array(bins['weights']);d=np.tile([-.2,-.1,0.,.1,.2],(2,1));d-=np.sum(d*weights,axis=1)[:,None]
        m.set_gap_correction(bins['mapping'],weights,d)
        changed=m.mark_distribution(context,x['gap'],prev,mask,static_codes=codes)[0]
        torch.testing.assert_close(changed[:,0],lp[:,0],rtol=0,atol=0)
        for j,g in enumerate(m.support.representatives):
            q,r=m.response_curves(context[mask],prev[mask],static_codes=codes[mask])
            gaps=torch.full_like(x['gap'][mask],g)
            logp,lr,_=m.mark_distribution(context[mask],gaps,prev[mask],torch.ones_like(prev[mask],dtype=torch.bool),static_codes=codes[mask])
            torch.testing.assert_close(lr.exp(),r[:,j],atol=3e-7,rtol=1e-6)
        q,r=m.response_curves(context[mask],prev[mask],static_codes=codes[mask],zero_gap=True)
        assert torch.equal(q.min(1).values,q.max(1).values);assert torch.equal(r.min(1).values,r.max(1).values)
        altered={k:(v.clone() if torch.is_tensor(v) else tuple(a.clone() for a in v)) for k,v in x.items()}
        altered['receiver'][:,1:]=3+(altered['receiver'][:,1:]+9)%64;altered['numeric_value'][:,1:]+=9
        c2=m.context(m.encoder(**altered),altered['static_categorical'])
        torch.testing.assert_close(c2[:,1],context[:,1],atol=0,rtol=0)
        assert old_digest(m)==state_digest(base)
        m.set_gap_correction(bins['mapping'],weights,np.zeros((2,5)))


def test_metric_contract_survives_json_storage_but_rejects_changed_edges():
    from experiments.cs_saf_gap_calibration import metric_states_equal
    import json
    state={'gap_bin_edges':(.2,.4,.6,.8),'mark_groups':('a','b'),'gap_scale':1.2}
    stored=json.loads(json.dumps(state))
    assert metric_states_equal(state,stored)
    stored['gap_bin_edges'][0]=.21
    assert not metric_states_equal(state,stored)
