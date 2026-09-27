import torch
from benchmarks.argn_onset_output import onset_mask


def test_routing_uses_only_available_past_and_current_label():
    f=torch.zeros(5,1,9)
    f[1:,0,0]=1;f[3:,0,1]=1
    labels=torch.tensor([[1],[1],[0],[1],[0]])
    assert onset_mask(f,labels).reshape(-1).tolist()==[False,True,False,False,False]


def test_routing_survives_compacted_customer_batch():
    f=torch.zeros(8,1,9);f[:,0,0]=1;f[::2,0,1]=1
    y=torch.ones(8,1,dtype=torch.long);selected=torch.tensor([7,2,5])
    assert torch.equal(onset_mask(f[selected],y[selected]),onset_mask(f,y)[selected])
