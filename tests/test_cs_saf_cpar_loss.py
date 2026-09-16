import numpy as np
import pytest
import torch
from deepecho.models.par import PARModel
from experiments.cs_saf_cpar_loss import vectorized_par_loss,equivalent_par_loss


def fixture(n,dtype,device):
    torch.manual_seed(137)
    m=PARModel(cuda=False);m._data_map={
        'numeric':{'type':'continuous','indices':(0,1,2)},
        'count':{'type':'count','indices':(3,4,5),'range':10},
        'category':{'type':'categorical','indices':{'a':6,'b':7,'c':8}},
        'ordinal':{'type':'ordinal','indices':{'low':9,'high':10}},
    }
    lengths=torch.randint(3,35,(n,));x=torch.zeros(33,n,11,dtype=dtype)
    x[:,:,0]=torch.randn(33,n,dtype=dtype);x[:,:,2]=torch.randint(0,2,(33,n)).to(dtype)
    x[:,:,3]=torch.randint(0,10,(33,n)).to(dtype)/10;x[:,:,5]=torch.randint(0,2,(33,n)).to(dtype)
    x[:,:,6:9]=torch.nn.functional.one_hot(torch.randint(0,3,(33,n)),3).to(dtype)
    x[:,:,9:11]=torch.nn.functional.one_hot(torch.randint(0,2,(33,n)),2).to(dtype)
    for i,L in enumerate(lengths):x[min(int(L),33):,i]=0
    x=x.to(device);y=torch.randn_like(x,requires_grad=True)
    return m,x,y,lengths


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('n',[3,37])
def test_variable_lengths_padding_all_types_and_unnormalized_gradients(n,dtype):
    torch.set_num_threads(1);m,x,a,lengths=fixture(n,dtype,'cpu');b=a.detach().clone().requires_grad_()
    old=m._compute_loss(x,a,lengths);new=vectorized_par_loss(m,x,b,lengths);scale=n*n*4
    ga=torch.autograd.grad(old*scale,a)[0];gb=torch.autograd.grad(new*scale,b)[0]
    tol=1e-12 if dtype==torch.float64 else 2e-5
    torch.testing.assert_close(new,old,rtol=tol,atol=tol)
    torch.testing.assert_close(gb,ga,rtol=tol,atol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA numerical regression')
def test_GPU_equivalence_loss_and_unnormalized_gradients():
    m,x,a,lengths=fixture(37,torch.float32,'cuda');b=a.detach().clone().requires_grad_();scale=37*37*4
    old=m._compute_loss(x,a,lengths);new=vectorized_par_loss(m,x,b,lengths)
    ga=torch.autograd.grad(old*scale,a)[0];gb=torch.autograd.grad(new*scale,b)[0]
    torch.testing.assert_close(new,old,rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(gb,ga,rtol=2e-5,atol=2e-5)


def test_class_patch_is_restored_even_on_error():
    original=PARModel._compute_loss
    with pytest.raises(RuntimeError):
        with equivalent_par_loss():
            assert PARModel._compute_loss is vectorized_par_loss
            raise RuntimeError('intentional restoration test')
    assert PARModel._compute_loss is original


def test_128_epochs_not_replaced_by_microbenchmark_budget():
    from experiments.cs_saf_followup import load_contract
    assert load_contract()[0]['external']['epochs']==128
