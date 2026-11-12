import numpy as np
import pandas as pd
import torch
from scipy.stats import multivariate_normal, norm
from scipy.special import logsumexp
from benchmarks.argn_clock_evolution import ProgressClock, ProgressDensity, NumpyDensity, progress_features


def test_multivariate_conditional_and_numpy_sampler_parameters():
    h=ProgressDensity(1,'progress_joint',components=2).double()
    mu=np.array([[1.,2.,3.],[2.,-1.,.5]])
    cov=np.array([[[2.,.2,.7],[.2,1.,-.1],[.7,-.1,1.]],[[1.,.1,-.3],[.1,2.,.2],[-.3,.2,1.5]]])
    h.weights[:]=torch.tensor([.3,.7],dtype=torch.float64); h.means[:]=torch.tensor(mu); h.covariances[:]=torch.tensor(cov)
    x=np.array([[-2.,0.],[0.,1.],[1.,2.],[4.,3.],[8.,4.]])
    y=np.array([.2,1.,-1.,2.,0.]); phase=torch.zeros(5,dtype=torch.long)
    logits,loc,scale=h(torch.zeros(5,1),phase,torch.tensor(x),torch.ones(5))
    estimated=logsumexp(logits.numpy()-logsumexp(logits.numpy(),axis=1,keepdims=True)+norm.logpdf(y[:,None],loc.numpy(),scale.numpy()),axis=1)
    joint=logsumexp(np.stack([np.log(w)+multivariate_normal.logpdf(np.c_[x,y],m,c) for w,m,c in zip([.3,.7],mu,cov)],axis=1),axis=1)
    marginal=logsumexp(np.stack([np.log(w)+multivariate_normal.logpdf(x,m[:2],c[:2,:2]) for w,m,c in zip([.3,.7],mu,cov)],axis=1),axis=1)
    np.testing.assert_allclose(estimated,joint-marginal,rtol=1e-9,atol=1e-9)
    p,l,s=NumpyDensity(dict(state_dict=h.state_dict())).parameters(phase.numpy(),x)
    np.testing.assert_allclose(p,logits.softmax(-1),rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(l,loc,rtol=1e-9,atol=1e-9); np.testing.assert_allclose(s,scale,rtol=1e-9,atol=1e-9)
    h.correction[:]=torch.tensor([.1,.2]); mixed=torch.tensor([0,1,2,3,0])
    a,b,c=h(torch.zeros(5,1),mixed,torch.tensor(x),torch.ones(5))
    p,l,s=NumpyDensity(dict(state_dict=h.state_dict())).parameters(mixed.numpy(),x,h.correction.tolist())
    np.testing.assert_allclose(l,b,rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(b.numpy()[1:4],loc.numpy()[1:4],rtol=0,atol=0)


def test_progress_features_are_strict_past_and_survive_reordering():
    rng=np.random.default_rng(91); gaps=rng.integers(0,1000000,(4,40)).astype(float); gaps[:,0]=np.nan
    records=pd.DataFrame([dict(entity_id=i,event_index=t,gap=gaps[i,t]) for i in range(4) for t in range(40)])
    x,v=progress_features(records,12000.); x=x.reshape(4,40,2); v=v.reshape(4,40)
    clock=ProgressClock(12000.); mem=clock.empty(4); ids=np.arange(4)
    for t in range(40):
        if t==17: mem=mem[[3,0]]; ids=ids[[3,0]]
        a,b=clock.features(mem)
        np.testing.assert_allclose(a,x[ids,t],rtol=0,atol=0); np.testing.assert_array_equal(b,v[ids,t])
        mem=clock.advance(mem,gaps[ids,t])
