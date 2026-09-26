import copy
import pytest
import torch
from deepecho.models.par import PARNet,PARModel
from torch.nn.utils.rnn import pack_sequence,pad_packed_sequence
from experiments.sparkov_cpar_dense import dense_forward,equivalent_dense_training,full_batch_gradient,equivalent_microbatch_training
from experiments.cs_saf_cpar_loss import vectorized_par_loss

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_full_sequence_loss_and_all_parameter_gradients(dtype):
    torch.set_num_threads(1);torch.manual_seed(33)
    n=7;lengths=[33,7,24,10,33,16,30]
    original=PARNet(14,5).to(dtype=dtype);dense=copy.deepcopy(original)
    xs=[]
    for length in lengths:
        x=torch.randn(length,14,dtype=dtype);x[:,2]=torch.randint(0,2,(length,)).to(dtype)
        x[:,3:]=torch.nn.functional.one_hot(torch.randint(0,11,(length,)),11).to(dtype);xs.append(x)
    x=pack_sequence(xs,enforce_sorted=False);padded,lens=pad_packed_sequence(x);c=torch.randn(n,5,dtype=dtype)
    a,_=pad_packed_sequence(original(x,c));b=dense_forward(dense,padded,c,lens)
    m=PARModel(cuda=False);m._data_map={0:{'type':'continuous','indices':(0,1,2)},1:{'type':'categorical','indices':{i:i+3 for i in range(11)}}}
    loss_a=m._compute_loss(padded[1:],a[:-1],lens);loss_b=vectorized_par_loss(m,padded[1:],b[:-1],lens)
    loss_a.backward();loss_b.backward()
    tol=2e-12 if dtype==torch.float64 else 3e-5
    torch.testing.assert_close(a,b,rtol=tol,atol=tol)
    torch.testing.assert_close(loss_a,loss_b,rtol=tol,atol=tol)
    for p,q in zip(original.parameters(),dense.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=tol,atol=tol)
    micro=copy.deepcopy(original);micro.zero_grad();m._model=micro;m._ctx_dims=5
    loss_micro=full_batch_gradient(m,padded,c,lens,chunk_size=3)
    torch.testing.assert_close(loss_a,loss_micro,rtol=tol,atol=tol)
    for p,q in zip(original.parameters(),micro.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=tol,atol=tol)

def test_restores_native_fit():
    original=PARModel.fit_sequences
    with equivalent_dense_training():assert PARModel.fit_sequences is not original
    assert PARModel.fit_sequences is original
    with equivalent_microbatch_training():assert PARModel.fit_sequences is not original
    assert PARModel.fit_sequences is original
