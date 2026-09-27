import numpy as np
import pandas as pd
import torch
from benchmarks.argn_time_density import PastClock, TimeDensity, clock_features, log_interval_mass


def test_clock_strict_past_matches_vectorized_and_reordered_state():
    rng=np.random.default_rng(1)
    gaps=rng.integers(0,1_000_000,(3,45)).astype(float)
    gaps[0,:8]=0
    codec=PastClock(123.)
    records=pd.DataFrame(dict(entity_id=np.repeat(np.arange(3),45),event_index=np.tile(np.arange(45),3),gap=gaps.ravel()))
    expected=clock_features(records,123.)
    memory=codec.empty(3);order=np.arange(3)
    for t in range(45):
        if t==17:
            ix=np.array([2,0]);memory=memory[ix];order=order[ix]
        actual=codec.features(memory)
        for i in [0,1]:np.testing.assert_allclose(actual[i],expected[i][order*45+t],rtol=0,atol=0)
        memory=codec.advance(memory,gaps[order,t])


def test_quantized_density_normalizes_including_endpoint_atoms_and_tail_gradients():
    h=TimeDensity(2,'history_clock',components=1)
    h.prior_mean.fill_(2.);h.prior_raw_scale.fill_(.2)
    y=torch.arange(101,dtype=torch.float32);base=torch.zeros(101,2);p=torch.zeros(101,dtype=torch.long)
    nll=h.nll(base,p,torch.zeros(101),torch.ones(101),y,[0,100],1.)
    torch.testing.assert_close((-nll).exp().sum(),torch.tensor(1.,dtype=torch.float64),rtol=1e-9,atol=1e-9)
    a=torch.tensor([-30.,-10.,0.,10.,30.],dtype=torch.float64,requires_grad=True)
    mass=log_interval_mass(a,a+1e-5)
    assert torch.isfinite(mass).all()
    mass.sum().backward();assert torch.isfinite(a.grad).all()
    nll.mean().backward()
    assert all(torch.isfinite(x.grad).all() for x in h.parameters())


def test_relative_location_shift_and_private_sampling():
    a=TimeDensity(2,'history_clock');r=TimeDensity(2,'history_relative');r.load_state_dict(a.state_dict())
    base=torch.randn(20,2);p=torch.arange(20)%4;clock=torch.full((20,),3.);valid=torch.ones(20)
    x=a(base,p,clock,valid);y=r(base,p,clock,valid)
    torch.testing.assert_close(x[0],y[0]);torch.testing.assert_close(x[2],y[2]);torch.testing.assert_close(x[1]+3,y[1])
    np.random.seed(11);expected=np.random.random(3);np.random.seed(11)
    values,_=r.sample(base,p,clock,valid,np.random.default_rng(123),[0,100],1.)
    assert np.array_equal(np.random.random(3),expected)
    assert np.all((values>=0)&(values<=100)&(values==np.floor(values)))
