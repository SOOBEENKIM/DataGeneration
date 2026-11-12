import numpy as np
import torch
from benchmarks.argn_clock_regression import ClockRegression
from benchmarks.argn_clock_location import ClockLocation
from benchmarks.argn_clock_evolution import NumpyDensity


def test_zero_correction_equals_original_and_does_not_change_fraud_density():
    old=ClockRegression(1,'clock_joint').double();h=ClockLocation(1,'clock_rollout').double()
    h.load_state_dict({**old.state_dict(),'correction':torch.zeros(2,dtype=torch.float64)})
    x=torch.tensor([[8.,2.],[9.,3.],[11.,4.],[10.,6.]],dtype=torch.float64);p=torch.arange(4)
    original=old(torch.zeros(4,1),p,x[:,0],torch.ones(4));current=h(torch.zeros(4,1),p,x,torch.ones(4))
    for a,b in zip(original,current):torch.testing.assert_close(a,b,rtol=0,atol=0)
    h.correction[:]=torch.tensor([.08,.2]);logits,mu,sigma=h(torch.zeros(4,1),p,x,torch.ones(4))
    torch.testing.assert_close(mu[1:],original[1][1:],rtol=0,atol=0)
    torch.testing.assert_close(logits,original[0],rtol=0,atol=0);torch.testing.assert_close(sigma,original[2],rtol=0,atol=0)
    q,l,s=NumpyDensity(dict(state_dict=h.state_dict())).parameters(p.numpy(),x.numpy(),h.correction.tolist())
    np.testing.assert_allclose(q,logits.softmax(-1),rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(l,mu,rtol=1e-9,atol=1e-9);np.testing.assert_allclose(s,sigma,rtol=1e-9,atol=1e-9)
