import numpy as np
import pandas as pd
import torch
from benchmarks.argn_joint_preservation import JointCount,fit_horizon_hazard,horizon_probability
from benchmarks.argn_episode_control import episode_metadata


def test_discrete_joint_probability_normalizes_including_boundary_mass():
    torch.manual_seed(10);m=JointCount(4,3).eval()
    # Nontrivial mixture includes probability outside both protected boundaries.
    with torch.no_grad():
        m.net[-1].bias[8:14]=torch.tensor([1.,3.,6.,1.,3.,6.])
    length=torch.arange(7,61).repeat(2);label=torch.arange(2).repeat_interleave(54)
    nll=m.nll(torch.zeros(108,4),length,label,7,60)
    torch.testing.assert_close((-nll).exp().sum(),torch.tensor(1.,dtype=torch.float64),rtol=1e-6,atol=1e-6)
    nll.mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_context_removal_really_has_no_context_dependence():
    m=JointCount(4,3,False).eval()
    with torch.no_grad():m.net[-1].weight.normal_()
    a=m(torch.zeros(5,4));b=m(torch.randn(5,4)*100)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_initial_state_count_dependency_is_sampled_not_output_relabeling():
    torch.manual_seed(2);m=JointCount(2,3).eval()
    with torch.no_grad():
        m.net[-1].bias[:2]=0
        m.net[-1].bias[8:14]=torch.tensor([7.,7.,7.,2.,2.,2.])
        m.net[-1].bias[14:]=-9
    length,label=m.sample(torch.zeros(2000,2),7,3106)
    assert label.eq(0).sum()>800 and label.eq(1).sum()>800
    assert length[label.eq(0)].float().median()>900
    assert length[label.eq(1)].float().median()<12
    assert length.ge(7).all() and length.le(3106).all()


def fixture():
    rows=[]
    for record in range(30):
        labels=[0]*20
        if record<15:labels[5:8]=[1,1,1]
        if record>=25:labels=[1]*20
        age=0
        for t,y in enumerate(labels):
            age=0 if t==0 else (age+1 if t>1 and labels[t-1]==labels[t-2] else 1)
            rows.append(dict(record=record,event_index=t,label=y,previous_label=labels[t-1] if t else -1,prior_age=age))
    return pd.DataFrame(rows)


def test_onset_exposure_has_no_fictitious_post_window_transition_or_repeat_risk():
    m=fixture();fit=fit_horizon_hazard(m);meta=episode_metadata(m)
    risk=meta.previous_label.eq(0)&meta.ever_fraud.eq(0)
    assert sum(c['events'] for c in fit['counts'])==risk.sum()
    assert sum(c['onsets'] for c in fit['counts'])==15
    assert np.isfinite(fit['rates']).all() and (np.array(fit['rates'])>0).all()


def test_exposure_scaling_keeps_integrated_risk_equal():
    rates=np.full(10,1.7)
    probabilities=[]
    for length in [100,1000]:
        p=horizon_probability(np.arange(length),np.full(length,length),rates)
        probabilities.append(1-np.exp(np.log1p(-p).sum()))
        torch.testing.assert_close(torch.from_numpy(p),horizon_probability(torch.arange(length,dtype=torch.float64),
            torch.full((length,),length,dtype=torch.float64),torch.tensor(rates)))
    np.testing.assert_allclose(probabilities,[1-np.exp(-1.7)]*2)
