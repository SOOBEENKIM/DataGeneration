import copy
import numpy as np
import pytest
import torch
from models.cs_saf_v3 import CSSAFv3,CANDIDATES
from models.cs_saf_v2 import CSSAFv2,BANK_NAMES
from models.cof_seqgen_saf import fit_train_only_gap_support
from experiments.cs_saf_v3 import mark_tv,accuracy_audit,load_contract


def model(candidate='CS3-R1',vocab=7):
    support=fit_train_only_gap_support([.1,.5,1,3,8])
    pi=torch.tensor([[.1,.2,.3,.1,.3],[.3,.1,.2,.3,.1]],dtype=torch.float64)
    return CSSAFv3(candidate,support,pi,receiver_vocab_size=vocab,static_categorical_vocab_sizes=(5,))


def inputs():
    return {'gap':torch.tensor([[float('nan'),.1,.5,1],[float('nan'),.5,8,float('nan')]]),
        'receiver':torch.tensor([[3,3,4,4],[4,5,5,0]]),'numeric_value':torch.tensor([[1.,2,3,1],[2.,1,3,0]]),
        'valid_mask':torch.tensor([[True]*4,[True]*3+[False]]),'static_categorical':(torch.tensor([3,4]),)}


def open_weights(m):
    with torch.no_grad():
        m.route_interaction_weight.copy_(torch.linspace(-.5,.7,32).reshape(2,16))
        m.history_interaction_weight.copy_(torch.linspace(.6,-.4,32).reshape(2,16))


def test_all_arms_fresh_common_tensors_and_predictions_match_v2_without_rng_change():
    states=[];rng=[]
    for c in CANDIDATES:
        torch.manual_seed(20260930);m=model(c);states.append(m.state_dict());rng.append(torch.get_rng_state())
    torch.manual_seed(20260930);old=CSSAFv2('CS2-U1',m.support,receiver_vocab_size=7,static_categorical_vocab_sizes=(5,))
    assert all(torch.equal(states[0][k],s[k]) for s in states for k in s)
    assert all(torch.equal(rng[0],x) for x in rng) and torch.equal(rng[0],torch.get_rng_state())
    assert all(torch.equal(states[0][k],v) for k,v in old.state_dict().items())
    actual=m.loss_terms(**inputs());expected=old.loss_terms(**inputs())
    assert all(torch.equal(actual[k],v) for k,v in expected.items())
    assert sum(p.numel() for p in m.parameters())-sum(p.numel() for p in old.parameters())==32


def test_centering_is_function_preserving_under_explicit_coefficient_map():
    e=model('CS3-E1');open_weights(e);c=model('CS3-C1');c.load_state_dict(e.state_dict())
    reps=torch.tensor(e.support.representatives);emb=e.gap_route(e._support_code(reps))
    with torch.no_grad():
        for s in (0,1):
            v=torch.tanh(torch.nn.functional.linear(emb,e.route_gap_weight[s]))
            mean=(v*e.reference_probabilities[s,:,None].float()).sum(0)
            c.history_interaction_weight[s].add_(e.route_interaction_weight[s]*mean)
    context=torch.randn(8,136);codes=torch.tensor([3,4]*4)
    torch.testing.assert_close(e.logit_grid(context,codes),c.logit_grid(context,codes),atol=2e-7,rtol=1e-6)
    _,_,delta=c.components(context,codes)
    centered=(delta*c.reference_probabilities[codes-3]).sum(1)
    assert centered.abs().max()<1e-7


@pytest.mark.parametrize('candidate',CANDIDATES)
def test_zero_gap_keeps_history_and_factual_grid_matches_likelihood(candidate):
    m=model(candidate);open_weights(m);c=torch.randn(2,136);codes=torch.tensor([3,4]);prev=torch.tensor([3,4])
    h,_,_=m.components(c,codes);grid=m.logit_grid(c,codes,zero_gap=True)
    torch.testing.assert_close(grid[:,0],m.copy_base(c)[:,0]+h)
    assert torch.count_nonzero(grid-grid[:,:1])==0 and h.abs().max()>0
    for i,gap in enumerate(m.support.representatives):
        q,repeat=m.response_curves(c,prev,static_codes=codes)
        logits=m.copy_logits(c,torch.full((2,),gap),static_codes=codes)
        torch.testing.assert_close(logits.sigmoid(),q[:,i])
        lp,lr,_=m.mark_distribution(c,torch.full((2,),gap),prev,torch.ones(2,dtype=torch.bool),static_codes=codes)
        torch.testing.assert_close(lr.exp(),repeat[:,i])
        torch.testing.assert_close(lp.exp().sum(-1),torch.ones(2))
        torch.testing.assert_close(lp.gather(1,prev[:,None])[:,0].exp(),repeat[:,i])


def test_penalty_has_no_direct_history_or_base_gradient_but_reaches_residual():
    m=model();open_weights(m);c=torch.randn(4,136);codes=torch.tensor([3,4,3,4])
    loss=m.residual_penalty(c,codes).sum()
    params=(m.history_interaction_weight,m.copy_base.weight,m.copy_base.bias,m.route_interaction_weight,m.route_gap_weight,m.route_context_weight)
    gradients=torch.autograd.grad(loss,params,allow_unused=True)
    assert all(x is None for x in gradients[:3])
    assert all(x is not None and torch.count_nonzero(x)>0 for x in gradients[3:])
    with torch.no_grad():m.route_interaction_weight.zero_()
    p=m.residual_penalty(c,codes)
    assert torch.count_nonzero(p)==0 and torch.isfinite(p).all()
    grad=torch.autograd.grad(p.sum(),m.route_interaction_weight)[0]
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad)==0


def test_regularizer_excludes_first_padding_and_uses_fixed_train_denominators():
    m=model();open_weights(m);x=inputs();counts={3:3,4:2}
    terms=m.loss_terms(**x);full=m.objective(terms,train_entities=2,transition_counts_by_code=counts)
    h=m.context(m.encoder(**x),x['static_categorical']);codes=x['static_categorical'][0][:,None].expand_as(x['valid_mask'])
    rho=m.residual_penalty(h,codes);mask=x['valid_mask'].clone();mask[:,0]=False
    torch.testing.assert_close(full[2],rho[mask].sum()/5)
    torch.testing.assert_close(full[0],full[1]+.01*full[2])
    partial=[]
    for i in (0,1):
        one={k:tuple(t[i:i+1] for t in v) if isinstance(v,tuple) else v[i:i+1] for k,v in x.items()}
        partial.append(m.objective(m.loss_terms(**one),train_entities=2,transition_counts_by_code=counts)[2])
    torch.testing.assert_close(full[2],torch.stack(partial).mean())


def test_history_only_is_gap_invariant_and_dormant_banks_have_no_gradient():
    m=model('CS3-H1');open_weights(m);x=inputs()
    loss=m.objective(m.loss_terms(**x),train_entities=2,transition_counts_by_code={3:3,4:2})[0];loss.backward()
    assert m.route_gap_weight.grad is None and m.route_interaction_weight.grad is None
    assert m.history_interaction_weight.grad.abs().max()>0 and m.route_context_weight.grad.abs().max()>0
    q,_=m.response_curves(torch.randn(2,136),torch.tensor([3,4]),static_codes=torch.tensor([3,4]))
    assert torch.count_nonzero(q-q[:,:1])==0


def test_strict_past_and_context_permutation_equivariance():
    m=model();open_weights(m);x=inputs();changed=copy.deepcopy(x)
    changed['receiver'][:,2:]=6;changed['gap'][:,2:]=7;changed['numeric_value'][:,2:]=9
    a=m.encoder(**x);b=m.encoder(**changed)
    torch.testing.assert_close(a[:,:3],b[:,:3],atol=0,rtol=0)
    other=copy.deepcopy(m)
    with torch.no_grad():
        other.encoder.static_categorical[0].weight[[3,4]]=m.encoder.static_categorical[0].weight[[4,3]]
        for name in (*BANK_NAMES,'history_interaction_weight','reference_probabilities'):
            getattr(other,name).copy_(getattr(m,name).flip(0))
    perm=copy.deepcopy(x);perm['static_categorical']=(7-x['static_categorical'][0],)
    for key,value in m.loss_terms(**x).items():
        if key!='static_code':torch.testing.assert_close(value,other.loss_terms(**perm)[key])


def test_mark_TV_counts_all_marks_not_just_latent_copy():
    q=torch.tensor([[.2,.7]],dtype=torch.float64);prev=torch.tensor([1]);f=torch.full((1,4),.25,dtype=torch.float64)
    torch.testing.assert_close(mark_tv(q,f,prev,q),torch.zeros_like(q))
    oq=torch.tensor([[.1,.4]],dtype=torch.float64)
    torch.testing.assert_close(mark_tv(q,f,prev,oq),.75*(q-oq).abs())
    nonuniform=torch.tensor([[.1,.25,.25,.4]],dtype=torch.float64)
    assert (mark_tv(q,nonuniform,prev,q)>0).all()
    p=(1-q[:,:,None])*nonuniform[:,None,:];p[:,:,1]+=q
    op=(1-oq[:,:,None])*torch.ones_like(nonuniform[:,None,:])/4;op[:,:,1]+=oq
    torch.testing.assert_close(mark_tv(q,nonuniform,prev,oq),.5*(p-op).abs().sum(-1))


def test_accuracy_entity_alignment_and_batch_invariance():
    m=model(vocab=67);open_weights(m);x=inputs()
    data={k:v for k,v in x.items() if k!='static_categorical'}
    data.update(codes=x['static_categorical'][0],lengths=x['valid_mask'].sum(1),entity_ids=['a','b'])
    # Two entities per context make batch-size changes meaningful.
    data={k:(v+['c','d'] if isinstance(v,list) else torch.cat([v,v])) for k,v in data.items()}
    s,a=accuracy_audit(m,data,m.reference_probabilities,1,torch.device('cpu'),batch_size=1)
    _,b=accuracy_audit(m,data,m.reference_probabilities,1,torch.device('cpu'),batch_size=2)
    assert a['label_0_entity_ids'].tolist()==['a','c']
    assert a['label_1_entity_ids'].tolist()==['b','d']
    for label in (0,1):
        np.testing.assert_allclose(a[f'label_{label}_metrics'],b[f'label_{label}_metrics'],atol=1e-6)
        assert np.isfinite(a[f'label_{label}_metrics']).all()
    assert s['oracle_uses_realized_latents'] is False


def test_contract_rejects_scope_or_hash_drift_and_fixes_one_penalty():
    c,cfg=load_contract()
    assert c['scientific_fits']==8 and c['candidates']==list(CANDIDATES)
    assert c['objective']['regularization_coefficient']==.01
    assert c['execution']['held_out_test'] is False
    assert cfg['training']['epochs']==50
    with pytest.raises(ValueError):
        CSSAFv3('CS3-R1',model().support,torch.ones(2,5),receiver_vocab_size=7,static_categorical_vocab_sizes=(5,))
