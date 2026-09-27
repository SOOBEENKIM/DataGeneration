import numpy as np
from benchmarks.argn_phase_gap import encode_values,sample_values


def test_numeric_digits_respect_scalar_value():
    stats=dict(min_decimal=0,min_digits={'E0':0,'E1':0,'E2':0})
    keys=['gap__E2','gap__E1','gap__E0'];x=np.array([0.,1.9,21.1,999.])
    y=encode_values(x,stats,keys,{k:10 for k in keys})
    assert np.array_equal(y[keys[0]]*100+y[keys[1]]*10+y[keys[2]],np.floor(x))


def test_mixture_uses_only_private_rng_and_correct_phase():
    params=dict(limits=[0,1000],phases={str(k):dict(k=1,weights=[1],means=[np.log1p(10**k)],variances=[1e-20]) for k in range(4)})
    np.random.seed(23);expected=np.random.random(3);np.random.seed(23)
    v,n=sample_values(params,np.array([3,0,2,1]),np.random.default_rng(42))
    np.testing.assert_allclose(v,[1000,1,100,10],rtol=1e-8)
    assert np.array_equal(np.random.random(3),expected)
