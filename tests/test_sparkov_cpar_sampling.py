import torch
from deepecho.models.par import PARModel,PARNet
from experiments.sparkov_cpar_sampling import cached_sequence,vectorized_decode,equivalent_sampling

def fixture():
    torch.set_num_threads(1);torch.manual_seed(27)
    m=PARModel(cuda=False);m._data_dims=9
    m._data_map={0:{'type':'continuous','indices':(0,1,2),'nulls':True,'std':2.5,'mu':3.},1:{'type':'categorical','indices':{'a':3,'b':4,'c':5}},'<TOKEN>':{'type':'categorical','indices':{'<START>':6,'<BODY>':7,'<END>':8}}}
    m._model=PARNet(9,2).eval()
    return m,torch.tensor([[.1,.7]])

def test_cached_sampling_including_early_end_rewrite():
    m,c=fixture()
    for minimum,maximum in [(1,48),(24,24),(20,48)]:
        torch.manual_seed(123)
        with torch.no_grad():a,la=m._sample_sequence(c,minimum,maximum)
        torch.manual_seed(123)
        with torch.no_grad():b,lb=cached_sequence(m,c,minimum,maximum)
        torch.testing.assert_close(a,b,rtol=3e-6,atol=3e-6);torch.testing.assert_close(la,lb,rtol=3e-6,atol=3e-6)
        assert m._tensor_to_data(a)==vectorized_decode(m,a)
    assert a[:,0,8].sum()>0

def test_method_restoration():
    original_sample=PARModel._sample_sequence;original_decode=PARModel._tensor_to_data
    with equivalent_sampling():assert PARModel._sample_sequence is cached_sequence
    assert PARModel._sample_sequence is original_sample and PARModel._tensor_to_data is original_decode
