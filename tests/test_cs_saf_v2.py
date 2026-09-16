from copy import deepcopy
import math

import numpy as np
import pytest
import torch

from models.cs_saf import CSSAF
from models.cs_saf_v2 import CSSAFv2, BANK_NAMES, VERSION
from models.cof_seqgen_saf import fit_train_only_gap_support
from experiments.cs_saf_pilot import audit_model, load_pilot_config


def model(candidate="CS2-B1", production=False):
    support = fit_train_only_gap_support(np.linspace(.1, 12, 300), max_positive_states=31) if production else fit_train_only_gap_support([.1,.5,1,3,8])
    return CSSAFv2(candidate, support, receiver_vocab_size=67 if production else 7,
                   static_categorical_vocab_sizes=(5,))


def inputs():
    return {"gap": torch.tensor([[float("nan"),.1,.5,1],[float("nan"),.5,3,float("nan")]]),
            "receiver": torch.tensor([[3,3,4,4],[4,5,5,0]]),
            "numeric_value": torch.tensor([[1.,2,3,1],[2.,1,3,0]]),
            "valid_mask": torch.tensor([[True]*4,[True]*3+[False]]),
            "static_categorical": (torch.tensor([3,4]),)}


def open_banks(m):
    with torch.no_grad():
        m.route_interaction_weight.copy_(torch.linspace(-.4,.8,32).reshape(2,16))


def test_exact_budget_paired_initialization_and_unchanged_common_rng():
    states=[]
    for candidate in ("CS2-U1","CS2-B1"):
        torch.manual_seed(20260930)
        m=model(candidate,production=True);states.append(m.state_dict())
        assert sum(p.numel() for p in m.parameters())==133549
        assert sum(getattr(m,n).numel() for n in BANK_NAMES)==5440
    assert states[0].keys()==states[1].keys()
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    after_v2=torch.get_rng_state().clone()
    torch.manual_seed(20260930)
    reference=CSSAF("CS-B1",m.support,receiver_vocab_size=67,static_categorical_vocab_sizes=(5,))
    assert torch.equal(after_v2,torch.get_rng_state())
    common={k:v for k,v in reference.state_dict().items() if not k.startswith(("context_projection.","gap_projection.")) and k!='interaction_weight'}
    assert set(common)==set(states[0])-set(BANK_NAMES)
    assert all(torch.equal(v,states[0][k]) for k,v in common.items())
    g=torch.Generator().manual_seed(20261010)
    for name,bound in [(BANK_NAMES[0],1/math.sqrt(136)),(BANK_NAMES[1],1/math.sqrt(136)),(BANK_NAMES[2],1/math.sqrt(32))]:
        expected=torch.empty_like(getattr(m,name)).uniform_(-bound,bound,generator=g)
        assert torch.equal(expected,getattr(m,name))
    assert torch.count_nonzero(m.route_interaction_weight)==0


@pytest.mark.parametrize("label",[0,1])
def test_banks_receive_own_gradients_and_exactly_zero_cross_gradients(label):
    m=model();open_banks(m)
    context=torch.randn(4,136)
    q=m.copy_logits(context,torch.tensor([.1,.5,1,3]),static_codes=torch.full((4,),label+3))
    q.sum().backward()
    for name in BANK_NAMES:
        grad=getattr(m,name).grad
        assert torch.isfinite(grad).all()
        assert grad[label].abs().sum()>0
        assert torch.count_nonzero(grad[1-label])==0


def test_zero_initialized_banks_begin_learning_on_both_contexts():
    m=model();x=inputs();optimizer=torch.optim.AdamW(m.parameters(),lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        loss=m.compute_loss(**x,train_entities=2,transition_counts_by_code={3:3,4:2})['loss']
        assert torch.isfinite(loss)
        loss.backward();optimizer.step()
    for name in BANK_NAMES:
        assert all(getattr(m,name).grad[s].abs().sum()>0 for s in (0,1))


def test_simultaneous_label_embedding_bank_permutation_preserves_predictions():
    m=model().eval();open_banks(m);other=deepcopy(m);x=inputs()
    with torch.no_grad():
        for name in BANK_NAMES:
            getattr(other,name).copy_(getattr(m,name).flip(0))
        other.encoder.static_categorical[0].weight[[3,4]]=m.encoder.static_categorical[0].weight[[4,3]]
    changed=dict(x,static_categorical=(7-x['static_categorical'][0],))
    def predict(net,xx):
        context=net.context(net.encoder(**xx),xx['static_categorical'])[:,1]
        return net.mark_distribution(context,xx['gap'][:,1],xx['receiver'][:,0],torch.ones(2,dtype=torch.bool),static_codes=xx['static_categorical'][0])[0]
    torch.testing.assert_close(predict(m,x),predict(other,changed),atol=0,rtol=0)
    a=m.loss_terms(**x);b=other.loss_terms(**changed)
    for key in ['gap_sum','mark_sum','value_sum','repeat_per_entity']:
        torch.testing.assert_close(a[key],b[key],atol=1e-6,rtol=1e-6)


def test_grid_audit_matches_mark_likelihood_and_zero_gap_is_invariant():
    m=model().eval();open_banks(m)
    context=torch.randn(2,136);previous=torch.tensor([3,4]);codes=torch.tensor([3,4])
    copy,repeat=m.response_curves(context,previous,static_codes=codes)
    for k,gap in enumerate(m.support.representatives):
        gaps=torch.full((2,),gap)
        logp,logr,_=m.mark_distribution(context,gaps,previous,torch.ones(2,dtype=torch.bool),static_codes=codes)
        torch.testing.assert_close(logp.gather(1,previous[:,None])[:,0].exp(),repeat[:,k])
        torch.testing.assert_close(logr.exp(),repeat[:,k])
        torch.testing.assert_close(m.copy_logits(context,gaps,static_codes=codes).sigmoid(),copy[:,k])
    zero,zero_repeat=m.response_curves(context,previous,zero_gap=True,static_codes=codes)
    assert torch.count_nonzero(zero.max(1).values-zero.min(1).values)==0
    assert torch.count_nonzero(zero_repeat.max(1).values-zero_repeat.min(1).values)==0
    assert (copy.max(1).values-copy.min(1).values).min()>0


def test_no_current_mark_or_future_information_in_current_distribution():
    m=model().eval();open_banks(m);x=inputs()
    altered=dict(x,receiver=x['receiver'].clone(),numeric_value=x['numeric_value'].clone(),gap=x['gap'].clone())
    altered['receiver'][0,2:]=6;altered['numeric_value'][0,2:]=99;altered['gap'][0,3]=8
    h=m.encoder(**x);h_changed=m.encoder(**altered)
    assert torch.equal(h[:,:3],h_changed[:,:3])
    def predict(hidden,xx):
        c=m.context(hidden,xx['static_categorical'])[:,2]
        return m.mark_distribution(c,xx['gap'][:,2],xx['receiver'][:,1],torch.ones(2,dtype=torch.bool),static_codes=xx['static_categorical'][0])[0]
    assert torch.equal(predict(h,x),predict(h_changed,altered))


def test_balanced_loss_global_denominators_and_candidate_difference():
    torch.manual_seed(4);balanced=model();torch.manual_seed(4);ordinary=model('CS2-U1')
    x=inputs();counts={3:3,4:2}
    b=balanced.compute_loss(**x,train_entities=2,transition_counts_by_code=counts)
    u=ordinary.compute_loss(**x,train_entities=2,transition_counts_by_code=counts)
    torch.testing.assert_close(b['loss'],u['loss']+b['balanced_repeat_nll'])
    pieces=[]
    for i in (0,1):
        one={k:tuple(z[i:i+1] for z in v) if isinstance(v,tuple) else v[i:i+1] for k,v in x.items()}
        pieces.append(balanced.compute_loss(**one,train_entities=2,transition_counts_by_code=counts)['balanced_repeat_nll'])
    torch.testing.assert_close(b['balanced_repeat_nll'],torch.stack(pieces).mean())


@pytest.mark.parametrize("code",[0,1,2,5])
def test_unknown_static_codes_are_rejected(code):
    with pytest.raises(ValueError,match='context'):
        model().copy_logits(torch.zeros(1,136),torch.ones(1),static_codes=torch.tensor([code]))


def test_generation_audit_checkpoint_roundtrip_and_version(tmp_path):
    m=model().eval();open_banks(m);x=inputs()
    torch.save(m.state_dict(),tmp_path/'checkpoint.pt')
    other=model().eval();other.load_state_dict(torch.load(tmp_path/'checkpoint.pt'))
    data={k:v for k,v in x.items() if k!='static_categorical'}
    data.update(codes=x['static_categorical'][0],lengths=torch.tensor([4,3]),entity_ids=['a','b'])
    cfg={'sampling_seed':42,'generation_entities':8,'generation_batch_size':4,'audit_chunk_histories':2}
    audit=audit_model(other,{'train':data,'validation':data},cfg,torch.device('cpu'),sample_output=tmp_path/'samples.pt')
    assert audit['gap_support_violations']==audit['invalid_reserved_marks']==0
    assert audit['finite_generated_values'] and audit['zero_gap_control_max_range']==0
    assert torch.load(tmp_path/'samples.pt')['version']==VERSION
    assert other.architecture_contract()['candidate']=='CS2-B1'


def test_v2_execution_config_preserves_registered_v1_budget_and_gates():
    old,_=load_pilot_config('v1');new,_=load_pilot_config('v2')
    for key in ['prevalences','kappas','training','cpu_gate','generation_entities','sampling_seed','model_seed']:
        assert new[key]==old[key]
    for key in old['pilot_gate']:
        if key!='primary_candidate':assert old['pilot_gate'][key]==new['pilot_gate'][key]
    assert new['pilot_candidates']==['CS2-U1','CS2-B1']
    assert new['pilot_gate']['primary_candidate']=='CS2-B1'
