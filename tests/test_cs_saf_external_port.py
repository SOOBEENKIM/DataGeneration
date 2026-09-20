import copy
import math

import numpy as np
import pytest
import torch

from data.cof_seqgen_saf_tensorizer import TensorizedSequence
from data.cs_saf_external import TargetWindows
from models.cof_seqgen_saf import GapSupportState
from models.cs_saf_external import ExternalUG
from models.cs_saf_v2 import CSSAFv2
from models.cs_saf_structure import ObservedRepeatStructure

torch.set_num_threads(1)


def support():
    return GapSupportState((0.,1.,3.),(0.,2.,float('inf')),3)


def sequences(lengths=(3,70)):
    out=[]
    for i,n in enumerate(lengths):
        out.append(TensorizedSequence(i,np.r_[np.nan,np.ones(n-1)].astype('float32'),
            np.arange(n)%3+3,np.arange(n,dtype='float32')/100,
            (np.arange(n)%2+3,),np.zeros((n,0),dtype='float32'),(3+i,),np.array([float(i)],dtype='float32')))
    return out


def model(mode='U'):
    torch.manual_seed(4)
    return ExternalUG(mode,support(),receiver_vocab_size=6,static_dim=1,
        static_categorical_vocab_sizes=(7,),auxiliary_categorical_vocab_sizes=(5,),hidden_dim=16,
        mark_embedding_dim=4,gap_embedding_dim=4)


def test_every_target_exactly_once_no_boundary_or_tail_loss():
    seq=sequences();w=TargetWindows(seq)
    b=w.batch(np.arange(len(w)))
    row=torch.arange(len(w));pos=b['target_position']
    assert len(w)==73
    assert torch.equal(b['receiver'][row,pos],torch.as_tensor(np.r_[seq[0].receiver,seq[1].receiver]))
    assert b['target_position'][3]==0 and torch.isnan(b['gap'][3,0])
    assert b['target_position'][35]==31
    assert b['gap'][35,0]==1 # actual gap of window's first event, not fake entity start
    assert torch.equal(b['valid_mask'].sum(1),pos+1)
    assert b['receiver'][72,31]==seq[1].receiver[-1]


@pytest.mark.parametrize('mode',['U','G'])
def test_causality_oov_normalization_and_gradients(mode):
    m=model(mode);b=TargetWindows(sequences()).batch([0,1,35,72])
    first=m.target_outputs(**b)
    changed=copy.deepcopy(b)
    for i,p in enumerate(b['target_position']):
        changed['receiver'][i,p]=1
        changed['numeric_value'][i,p]=100
        changed['auxiliary_categorical'][0][i,p]=1
    after=m.target_outputs(**changed)
    torch.testing.assert_close(first['hidden'],after['hidden'],rtol=0,atol=0)
    torch.testing.assert_close(first['logmark'],after['logmark'],rtol=0,atol=0)
    torch.testing.assert_close(first['logmark'].exp().sum(-1),torch.ones(4))
    assert torch.isfinite(after['logmark'][:,1:]).all()
    assert torch.isneginf(after['logmark'][:,0]).all()
    loss=m.compute_loss(**changed)['loss'];loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.encoder.history_auxiliary_categorical[0].weight.grad.abs().sum()>0


def test_direct_prefix_matches_long_window_prediction():
    m=model();seq=sequences((70,));w=TargetWindows(seq)
    for t in (0,1,30,31,32,69):
        b=w.batch([t]);o=m.target_outputs(**b)
        length=int(b['target_position'][0])+1
        fields={k:(tuple(vv[:,:length] for vv in v) if k=='auxiliary_categorical' else v[:,:length])
                for k,v in b.items() if k in ('gap','receiver','numeric_value','valid_mask','auxiliary_categorical','auxiliary_numeric')}
        h=m.encoder(**fields,static=b['static'],static_categorical=b['static_categorical'])[:,-1]
        torch.testing.assert_close(o['hidden'],h,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('mode',['U','G'])
def test_generation_replay_identical_history_beyond_window(mode):
    m=model(mode).eval();lengths=[37,65]
    captured=[]
    hook=m.encoder.register_forward_hook(lambda module,args,out:captured.append(out[:,-1].clone()))
    sample=m.sample_fixed_lengths(lengths,static=torch.tensor([[0.],[1.]]),static_categorical=(torch.tensor([3,4]),))
    hook.remove()
    seq=[]
    for i,n in enumerate(lengths):
        seq.append(TensorizedSequence(i,sample['gap'][i,:n].numpy(),sample['receiver'][i,:n].numpy(),
            sample['numeric_value'][i,:n].numpy(),tuple(v[i,:n].numpy() for v in sample['auxiliary_categorical']),
            np.zeros((n,0),dtype='float32'),(3+i,),np.array([float(i)],dtype='float32')))
    w=TargetWindows(seq)
    for t in (0,1,31,32,36,37,64):
        targets=[t] if t<37 else []
        targets.append(37+t)
        out=m.target_outputs(**w.batch(targets))
        torch.testing.assert_close(out['hidden'],captured[t],rtol=1e-5,atol=1e-6)
    assert len(sample['gap'][1])==65 and sample['valid_mask'].sum()==102


def test_parameter_and_initial_weight_matching():
    u,g=model('U'),model('G')
    assert u.architecture_contract()['parameters']==g.architecture_contract()['parameters']
    assert u.state_dict().keys()==g.state_dict().keys()
    for k in u.state_dict():torch.testing.assert_close(u.state_dict()[k],g.state_dict()[k],rtol=0,atol=0)


@pytest.mark.parametrize('mode',['U','G'])
def test_calibration_changes_repeat_only_preserves_first_and_normalization(mode):
    m=model(mode);batch=TargetWindows(sequences()).batch([0,1,35,72])
    old=m.target_outputs(**batch)
    m.calibration={'edges':[.5,1.5,2.,3.], 'beta':[.1,.2,.3,.4,.5,.6]}
    new=m.target_outputs(**batch)
    torch.testing.assert_close(old['logmark'][0],new['logmark'][0])
    torch.testing.assert_close(new['logmark'].exp().sum(1),torch.ones(4))
    for i in range(1,4):
        prev=int(old['previous'][i]);keep=[j for j in range(1,6) if j!=prev]
        torch.testing.assert_close(old['logmark'][i,keep].softmax(0),new['logmark'][i,keep].softmax(0))
        assert new['logmark'][i,prev]>old['logmark'][i,prev]


@pytest.mark.parametrize('mode',['U','G'])
@pytest.mark.parametrize('group',[0,1])
def test_old_controlled_mark_head_semantics_preserved_per_bank(mode,group):
    torch.manual_seed(8)
    old=(CSSAFv2('CS2-U1',support(),receiver_vocab_size=7,static_categorical_vocab_sizes=(5,)) if mode=='U'
         else ObservedRepeatStructure('G',support(),receiver_vocab_size=7,static_categorical_vocab_sizes=(5,)))
    new=ExternalUG(mode,support(),receiver_vocab_size=7)
    with torch.no_grad():
        old.route_interaction_weight.normal_()
        new.copy_base.load_state_dict(old.copy_base.state_dict())
        new.new_mark_head.load_state_dict(old.new_mark_head.state_dict())
        new.new_mark_head.weight[1:3]=0;new.new_mark_head.bias[1:3]=-1000
        new.gap_route.load_state_dict(old.gap_route.state_dict())
        new.route_context.weight.zero_();new.route_context.bias.zero_();new.route_gap.weight.zero_()
        new.route_context.weight[:16]=old.route_context_weight[group]
        new.route_context.bias[:16]=old.route_context_bias[group]
        new.route_gap.weight[:16]=old.route_gap_weight[group]
        new.interaction.zero_();new.interaction[:16]=old.route_interaction_weight[group]*math.sqrt(2)
    c=torch.randn(9,136);gap=torch.tensor([0.,1.,3.]*3);prev=torch.tensor([3,4,5]*3)
    has=torch.tensor([False,True,True]*3);codes=torch.full((9,),group+3)
    expected=old.mark_distribution(c,gap,prev,has,static_codes=codes)[0]
    actual=new.mark_log_probabilities(c,gap,prev,has)
    torch.testing.assert_close(expected.exp(),actual.exp(),rtol=2e-5,atol=1e-6)
