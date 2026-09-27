import torch
from benchmarks.argn_boundary_gap import boundary_mask


def test_routes_only_known_label_changes():
    # no past, normal onset, fraud exit, fraud continuation, normal stay, unknown past
    f=torch.tensor([[0.,0,0],[1,0,0],[1,1,0],[1,1,0],[1,0,0],[1,0,1]])
    y=torch.tensor([1,1,0,1,0,1])
    assert boundary_mask(f,y).tolist()==[False,True,True,False,False,False]
    ix=torch.tensor([2,1,0,4])
    assert torch.equal(boundary_mask(f[ix],y[ix]),boundary_mask(f,y)[ix])
