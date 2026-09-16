"""Boundary tests for extending a failed preregistration without changing history."""
import numpy as np
import pytest
import torch
from experiments.cs_saf_followup import (ROOT,CACHE,HISTORICAL,CANDIDATES,load_contract,load_cache,make_model,folder_for,gradient_geometry,parameter_block,parent)
from experiments.cs_saf_pilot import subset,batch
from scripts.run_cs_saf_followup import tasks

@pytest.fixture(scope='module')
def payload():
    torch.set_num_threads(1)
    p=load_cache(CACHE/'pi_0.05_kappa_1.pt')
    return dict(p,train=subset(p['train'],3),validation=subset(p['validation'],2))


def test_fixed_grid_never_overwrites_historical_or_selects_winner():
    gpu,cpu=tasks()
    assert len(gpu)==220 and len(cpu)==200 and len(set(gpu))==220
    assert sum(t[0]=='internal' and t[1]==.05 for t in gpu)==20
    assert sum(t[0]=='external' for t in gpu)==40
    assert all(not folder_for(t[1],t[2],t[3],t[4]).is_relative_to(HISTORICAL) for t in gpu if t[0]=='internal')
    assert folder_for(.05,1,0,'E').is_relative_to(HISTORICAL)
    assert load_contract()[0]['internal']['original_FAIL_preserved']


@pytest.mark.parametrize('candidate',CANDIDATES)
def test_exact_frozen_forward_and_scaled_penalty_gradient(payload,candidate):
    cfg,_=load_contract();spec=cfg['candidates'][candidate]
    torch.manual_seed(23);a=make_model(payload,candidate,torch.device('cpu'),0)
    torch.manual_seed(23);b=parent.make_model(payload,spec['parent'],torch.device('cpu'),0)
    if candidate!='U':
        with torch.no_grad():a.route_interaction_weight.uniform_(-.2,.2)
        b.load_state_dict(a.state_dict());b.regularization_coefficient=spec['lambda']
    x=batch(payload['train'],torch.arange(6),torch.device('cpu'))
    ta=a.loss_terms(**x);tb=b.loss_terms(**x)
    assert all(torch.equal(ta[k],tb[k]) for k in ta)
    opts=dict(train_entities=6,transition_counts_by_code={3:50,4:60})
    la=a.objective(ta,**opts);lb=b.objective(tb,**opts)
    assert all(torch.equal(x,y) for x,y in zip(la,lb))
    if candidate!='U':torch.testing.assert_close(la[0],la[1]+spec['lambda']*la[2])
    la[0].backward();lb[0].backward()
    assert all((x.grad is None and y.grad is None) or torch.equal(x.grad,y.grad) for x,y in zip(a.parameters(),b.parameters()))


def test_centered_penalty_reaches_shared_features_but_not_direct_history_coefficients(payload):
    m=make_model(payload,'ER',torch.device('cpu'),0)
    with torch.no_grad():m.route_interaction_weight.fill_(.2);m.history_interaction_weight.fill_(.1)
    x=batch(payload['train'],torch.arange(6),torch.device('cpu'))
    penalty=m.loss_terms(**x)['residual_per_entity'].sum();penalty.backward()
    assert m.route_context_weight.grad.norm()>0
    assert m.history_interaction_weight.grad is None or m.history_interaction_weight.grad.count_nonzero()==0
    assert any(p.grad is not None and p.grad.norm()>0 for p in m.encoder.parameters())


def test_gradient_geometry_handles_zero_vectors_without_false_conflict():
    assert gradient_geometry(torch.zeros(3),torch.ones(3))['cosine'] is None
    assert gradient_geometry(torch.ones(3,dtype=torch.float64),-torch.ones(3,dtype=torch.float64))['cosine']==pytest.approx(-1)


def test_existing_sample_plan_and_decode_roundtrip():
    from experiments.cs_saf_followup_external import canonical_and_plan,sample_to_frame
    d,p,plan,pos=canonical_and_plan(.05,1,0)
    s=torch.load(folder_for(.05,1,0,'E')/'generated_sample.pt',map_location='cpu')['sample']
    frame=sample_to_frame(s,p,plan,pos)
    assert len(frame)==sum(plan.lengths) and frame.entity_id.nunique()==2048
    assert set(frame.receiver_or_mark)<=set(d.events.receiver_or_mark)
    assert frame.groupby('entity_id',sort=False).size().tolist()==list(plan.lengths)
    bad=dict(s,plan_train_indices=s['plan_train_indices'].roll(1))
    with pytest.raises(AssertionError):sample_to_frame(bad,p,plan,pos)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA regression check')
def test_dropout_free_GRU_backward_mode_preserves_eval_forward(payload):
    m=make_model(payload,'ER',torch.device('cuda:0'),0).eval()
    with torch.no_grad():m.route_interaction_weight.fill_(.2)
    before={k:v.clone() for k,v in m.state_dict().items()}
    x=batch(payload['train'],torch.arange(6),torch.device('cuda:0'))
    with torch.no_grad():expected=m.loss_terms(**x)
    assert m.encoder.gru.dropout==0
    m.encoder.gru.train()
    terms=m.loss_terms(**x)
    for k in ('gap_sum','mark_sum','value_sum','residual_per_entity'):
        torch.testing.assert_close(terms[k],expected[k],rtol=1e-5,atol=1e-6)
    loss=terms['mark_sum']/terms['mark_count']+.01*terms['residual_per_entity'].mean()
    grads=torch.autograd.grad(loss,tuple(m.parameters()),allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
