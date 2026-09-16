import numpy as np
import pytest
import torch

from experiments.cs_saf_route_decomposition import (
    reference_measure, decompose_logits, arm_predictions, audit_payload, analyze,
)
from models.cs_saf_v2 import CSSAFv2
from models.cof_seqgen_saf import fit_train_only_gap_support
from scripts.audit_cs_saf_route_decomposition import contract_and_parent


def model():
    m=CSSAFv2('CS2-U1',fit_train_only_gap_support([.1,.5,1,3,8]),receiver_vocab_size=7,static_categorical_vocab_sizes=(5,))
    with torch.no_grad():m.route_interaction_weight.copy_(torch.linspace(-4,8,32).reshape(2,16))
    return m.eval()


def data():
    return {'gap':torch.tensor([[float('nan'),.1,.5,1],[float('nan'),.5,8,float('nan')]]),
            'receiver':torch.tensor([[3,3,4,4],[4,5,5,0]]),'numeric_value':torch.ones(2,4),
            'valid_mask':torch.tensor([[True]*4,[True]*3+[False]]),'codes':torch.tensor([3,4]),
            'lengths':torch.tensor([4,3]),'entity_ids':['a','b']}


def test_reference_counts_use_transitions_and_float32_model_boundaries():
    m=model();d=data();weights,meta=reference_measure(m,d)
    assert meta['0']['transitions']==3 and meta['1']['transitions']==2
    for y in (0,1):
        bins=m._support_code(d['gap'][y,1:d['lengths'][y]])-3
        expected=torch.bincount(bins,minlength=weights.shape[1]).double()/len(bins)
        torch.testing.assert_close(weights[y],expected,atol=0,rtol=0)


def test_reconstruction_weighted_centering_and_independent_forward_agree():
    m=model();w,_=reference_measure(m,data());context=torch.randn(4,136);codes=torch.tensor([3,4,3,4])
    logits,p=decompose_logits(m,context,codes,w)
    torch.testing.assert_close(logits['F'],p['base']+p['route'],atol=1e-12,rtol=0)
    torch.testing.assert_close((p['residual']*p['weights']).sum(1),torch.zeros(4,dtype=torch.float64),atol=1e-12,rtol=0)
    for k,g in enumerate(m.support.representatives):
        expected=m.copy_logits(context,torch.full((4,),g),static_codes=codes)
        torch.testing.assert_close(logits['F'][:,k],expected.double(),atol=1e-6,rtol=1e-6)
    assert logits['M'].shape==(4,1) and logits['Z'].shape==(4,1)
    assert not torch.equal(logits['M'],logits['Z'])


def test_centering_is_in_logit_space_not_probability_space():
    logits={'F':torch.tensor([[-3.,3.]],dtype=torch.float64),
            'M':torch.tensor([[-1.5]],dtype=torch.float64)}
    fresh=torch.full((1,7),-float('inf'));fresh[:,3:]=np.log(.25)
    p=arm_predictions(logits,fresh,torch.tensor([3]),torch.tensor([3]),torch.tensor([0]))
    w=torch.tensor([.75,.25],dtype=torch.float64)
    assert float(p['M']['copy'])==pytest.approx(torch.sigmoid(torch.tensor(-1.5)).item(),abs=1e-7)
    assert abs(float(p['M']['copy'])-float((p['F']['copy']*w).sum()))>.05


def test_full_mark_and_repeat_loss_differences_match_for_repeat_and_nonrepeat():
    ell=torch.tensor([[-1.,.5],[.7,-.8]],dtype=torch.float64)
    logits={'F':ell,'M':ell.mean(1,keepdim=True),'R':ell-ell.mean(1,keepdim=True),'Z':torch.zeros(2,1,dtype=torch.float64)}
    fresh=torch.randn(2,7);fresh[:,:3]=-torch.inf;fresh=fresh.log_softmax(-1)
    predicted=arm_predictions(logits,fresh,torch.tensor([3,4]),torch.tensor([3,5]),torch.tensor([0,1]))
    for a in ('M','R','Z'):
        torch.testing.assert_close(predicted[a]['mark_NLL']-predicted['F']['mark_NLL'],
                                   predicted[a]['repeat_BCE']-predicted['F']['repeat_BCE'],atol=1e-12,rtol=0)
    assert predicted['M']['copy_range'].count_nonzero()==predicted['Z']['repeat_range'].count_nonzero()==0


def test_zero_route_has_identical_four_arms():
    m=model()
    with torch.no_grad():m.route_interaction_weight.zero_()
    w,_=reference_measure(m,data());logits,_=decompose_logits(m,torch.randn(2,136),torch.tensor([3,4]),w)
    for x in logits.values():assert torch.equal(x,logits['Z'].expand_as(x))


def test_entity_diagnostics_are_batch_invariant_and_leave_weights_unchanged():
    m=model();d=data();w,_=reference_measure(m,d);before={k:v.clone() for k,v in m.state_dict().items()}
    a,ra=analyze(m,d,w,torch.device('cpu'),batch_size=1)
    b,rb=analyze(m,d,w,torch.device('cpu'),batch_size=2)
    for key in ra:
        if key.endswith('entity_ids'):assert np.array_equal(ra[key],rb[key])
        else:np.testing.assert_allclose(ra[key],rb[key],atol=1e-6,rtol=1e-6)
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    assert all(x<1e-6 for x in a['mechanical_errors'].values())
    assert a['groups']['1']['entities']==b['groups']['1']['entities']==1


@pytest.mark.parametrize('defect',['overlap','first_gap','reserved_mark'])
def test_preprocessing_contract_rejects_corrupted_input(defect):
    train,validation=data(),data();validation['entity_ids']=['v1','v2']
    if defect=='overlap':validation['entity_ids'][0]='a'
    if defect=='first_gap':train['gap'][0,0]=0
    if defect=='reserved_mark':validation['receiver'][0,0]=1
    with pytest.raises(ValueError):audit_payload(model(),{'train':train,'validation':validation,'tensorizer_state':{'fit_split':'train'}})


def test_frozen_scope_has_no_training_and_pins_parent_evidence():
    cfg,parent=contract_and_parent()
    assert cfg['checkpoint_count']==12 and cfg['execution']['new_fits']==0
    assert cfg['execution']['held_out_test'] is False
    assert parent['scientific_fits']==6
