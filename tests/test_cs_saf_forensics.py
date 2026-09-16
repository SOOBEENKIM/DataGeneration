from copy import deepcopy

import numpy as np
import torch

from experiments.cs_saf_forensics import (
    analyze_model, fixed_feature_curves, flat_gradient, local_direction,
    repeat_nll, route_parameters,
)
from models.cs_saf import CSSAF
from models.cof_seqgen_saf import fit_train_only_gap_support


def fixture_model():
    torch.manual_seed(93)
    model = CSSAF("CS-B1", fit_train_only_gap_support([.1, .5, 1, 3, 8]),
                  receiver_vocab_size=7, static_categorical_vocab_sizes=(5,))
    with torch.no_grad():
        model.interaction_weight.copy_(torch.randn(32)*.7)
    return model


def test_mark_and_repeat_losses_have_identical_direct_route_gradients():
    model = fixture_model().double()
    context = torch.randn(4, 136, dtype=torch.float64)
    previous = torch.tensor([3, 4, 5, 6])
    target = torch.tensor([3, 5, 5, 4])
    logmark, logr, lognr = model.mark_distribution(context, torch.tensor([.1,.5,1.,3.]), previous, torch.ones(4,dtype=torch.bool))
    mark = -logmark.gather(1, target[:,None]).sum()
    repeat = -torch.where(target == previous, logr, lognr).sum()
    a = flat_gradient(mark, route_parameters(model), retain_graph=True)
    b = flat_gradient(repeat, route_parameters(model))
    np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-8)


def test_local_descent_derivative_matches_synthetic_finite_difference():
    model = fixture_model().double()
    context = torch.randn(3,136,dtype=torch.float64)
    previous = torch.tensor([3,4,5])
    logits,q,_,fresh,_,_ = fixed_feature_curves(model,context,previous)
    loss = repeat_nll(logits[:,1],fresh,torch.tensor([True,False,True])).mean()
    g = flat_gradient(loss,route_parameters(model),retain_graph=True)
    response = q.max(1).values.mean()-q.min(1).values.mean()
    d = flat_gradient(response,route_parameters(model))
    values=[]
    for sign in [-1,1]:
        changed=deepcopy(model);offset=0
        with torch.no_grad():
            for p in route_parameters(changed):
                p.add_(torch.from_numpy(g[offset:offset+p.numel()]).reshape(p.shape),alpha=-sign*1e-5)
                offset+=p.numel()
        _,curve,_,_,_,_=fixed_feature_curves(changed,context,previous)
        values.append(float((curve.max(1).values-curve.min(1).values).mean()))
    numerical=(values[1]-values[0])/(2e-5)
    np.testing.assert_allclose(numerical,local_direction(d,g)['unscaled_descent_derivative'],atol=1e-9,rtol=1e-5)


def test_entity_and_gradient_aggregation_survive_unequal_lengths_and_batches():
    model=fixture_model()
    data={'gap':torch.tensor([[float('nan'),.1,.5,1.],[float('nan'),1.,3.,8.],[float('nan'),.1,8.,float('nan')]]),
          'receiver':torch.tensor([[3,3,4,5],[4,5,5,4],[5,6,6,0]]),
          'numeric_value':torch.ones(3,4),'valid_mask':torch.tensor([[True]*4,[True]*4,[True]*3+[False]]),
          'codes':torch.tensor([3,4,3]),'lengths':torch.tensor([4,4,3]),'entity_ids':['a','b','c']}
    cfg={'entity_batch_size':1,'central_observed_gap_mass':[0.,1.],'position_bands':[[1,2],[3,31]]}
    before={k:v.clone() for k,v in model.state_dict().items()}
    first,arr1=analyze_model(model,data,cfg,torch.device('cpu'))
    second,arr2=analyze_model(model,data,dict(cfg,entity_batch_size=3),torch.device('cpu'))
    for key in arr1:
        if 'ids' not in key:
            np.testing.assert_allclose(arr1[key],arr2[key],atol=1e-7,rtol=1e-5)
    assert first['groups']['0']['transitions']==5
    assert first['groups']['1']['entities']==1
    for group in second['groups'].values():
        assert group['repeat_range_identity_max_absolute_error']<1e-7
        assert group['observed_bin_counts'] and sum(group['observed_bin_counts'])==group['transitions']
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in model.parameters())
