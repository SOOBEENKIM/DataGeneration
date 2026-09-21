import copy
import math

import numpy as np
import pytest
from scipy.stats import lognorm
import torch

from data.cof_seqgen_saf_tensorizer import TensorizedSequence
from data.cs_saf_external import TargetWindows
from models.cof_seqgen_saf import GapSupportState
from models.cs_saf_external import ExternalUG
from models.cs_saf_external_controls import ExternalControls

torch.set_num_threads(1)
AST = dict(codec_mean=0., codec_scale=1., log_mean=1., log_scale=2.,
           initial_locations=[-1., 0., 1.], zero_rate=.1)
KWARGS = dict(receiver_vocab_size=7, static_dim=1, static_categorical_vocab_sizes=(7,),
              auxiliary_categorical_vocab_sizes=(5,), hidden_dim=16, mark_embedding_dim=4, gap_embedding_dim=4)


def support():
    return GapSupportState((0., 1., 3.), (0., 2., float('inf')), 3)


def make(mode='G', amount='hurdle_lognormal3', full=True):
    torch.manual_seed(4)
    return ExternalControls(mode, support(), amount_kind=amount, full_gap=full,
                            amount_state=AST, **KWARGS)


def sequences():
    return [TensorizedSequence(i, np.r_[np.nan, np.ones(n-1)].astype('float32'),
            np.arange(n)%3+3, np.log1p(np.arange(n)).astype('float32'),
            (np.arange(n)%2+3,), np.zeros((n,0), dtype='float32'), (3+i,), np.array([float(i)], dtype='float32'))
            for i,n in enumerate((37,65))]


@pytest.mark.parametrize('mode', ['U','G'])
def test_legacy_forward_and_long_sampling_match_parent_exactly(mode):
    torch.manual_seed(9)
    parent = ExternalUG(mode, support(), **KWARGS)
    new = make(mode, 'legacy', False)
    new.initialize_shared(parent.state_dict())
    batch = TargetWindows(sequences()).batch([0,1,31,36,37,101])
    expected, actual = parent.terms(**batch), new.terms(**batch)
    for k in expected[0]:
        torch.testing.assert_close(expected[0][k][0], actual[0][k][0], rtol=0, atol=0)
    for key in ('hidden','logmark','location','scale'):
        torch.testing.assert_close(expected[1][key], actual[1][key], rtol=0, atol=0)
    args = dict(lengths=[37,65], static=torch.tensor([[0.],[1.]]), static_categorical=(torch.tensor([3,4]),))
    torch.manual_seed(10); old = parent.sample_fixed_lengths(**args)
    torch.manual_seed(10); new_sample = new.sample_fixed_lengths(**args)
    for key in ('gap','receiver','numeric_value'):
        torch.testing.assert_close(old[key], new_sample[key], rtol=0, atol=0, equal_nan=True)


def test_hurdle_mixture_density_matches_scipy_including_coordinate_jacobian():
    m = make()
    amount = np.array([0., .1, 1., 10., 100.])
    value = torch.tensor(np.log1p(amount), dtype=torch.float32)
    w = torch.tensor([.2,.3,.5]).log().repeat(5,1)
    loc = torch.tensor([-1.,0.,1.]).repeat(5,1)
    sd = torch.tensor([.3,.5,.8]).repeat(5,1)
    zero = torch.full((5,), math.log(.1/.9))
    actual = m.amount_nll(value, (w,loc,sd,zero)).double().numpy()
    density = sum(weight*lognorm.pdf(amount[1:], s=scale*AST['log_scale'],
                  scale=math.exp(location*AST['log_scale']+AST['log_mean']))
                  for weight,location,scale in zip([.2,.3,.5],[-1.,0.,1.],[.3,.5,.8]))
    expected = np.r_[-math.log(.1), -np.log(.9*density*(1+amount[1:]))]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_sampling_obeys_zero_atom_positive_support_and_history_codec():
    m = make(); n=30000
    h=torch.zeros(n,16); gap=torch.ones(n); mark=torch.full((n,),3)
    torch.manual_seed(74)
    z,raw=m.sample_amount(h,gap,mark)
    assert torch.isfinite(raw).all() and raw.min()==0 and (raw>=0).all()
    assert abs(float(raw.eq(0).float().mean())-.1)<.01
    torch.testing.assert_close(z.double(), torch.log1p(raw), rtol=1e-6, atol=1e-6)
    m.amount_state=dict(AST,zero_rate=0.)
    _,positive=m.sample_amount(h,gap,mark)
    assert (positive>0).all()


@pytest.mark.parametrize('mode', ['U','G','D'])
def test_normalization_strict_past_finite_loss_gradients_and_shared_initial(mode):
    m=make(mode); w=TargetWindows(sequences()); batch=w.batch([0,1,31,36,37,101])
    before=m.target_outputs(**batch)
    changed=copy.deepcopy(batch)
    for i,p in enumerate(batch['target_position']):
        changed['receiver'][i,p]=1
        changed['numeric_value'][i,p]=8
        changed['auxiliary_categorical'][0][i,p]=1
    after=m.target_outputs(**changed)
    torch.testing.assert_close(before['hidden'],after['hidden'],rtol=0,atol=0)
    torch.testing.assert_close(before['logmark'],after['logmark'],rtol=0,atol=0)
    torch.testing.assert_close(before['logmark'].exp().sum(-1),torch.ones(6))
    assert torch.isfinite(before['logmark'][:,1:]).all()
    loss=m.compute_loss(**batch)['loss']; loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.value_head.weight.grad.abs().sum()>0


@pytest.mark.parametrize('mode', ['U','G'])
def test_new_gap_path_changes_nonrepeat_odds_and_zero_initialization_preserves_them(mode):
    m=make(mode); context=torch.randn(2,24); context[1]=context[0]
    gap=torch.tensor([0.,3.]); prev=torch.tensor([3,3]); has=torch.ones(2,dtype=torch.bool)
    before=m.mark_log_probabilities(context,gap,prev,has)
    torch.testing.assert_close(before[0,4]-before[0,5],before[1,4]-before[1,5])
    with torch.no_grad():
        m.fresh_gap.weight.normal_()
    after=m.mark_log_probabilities(context,gap,prev,has)
    assert abs(float((after[0,4]-after[0,5])-(after[1,4]-after[1,5])))>1e-4


@pytest.mark.parametrize('mode', ['U','G','D'])
def test_calibration_preserves_first_and_nonrepeat_relative_probabilities(mode):
    m=make(mode); batch=TargetWindows(sequences()).batch([0,1,35,101])
    old=m.target_outputs(**batch)
    m.calibration={'edges':[.5,1.5,2.,3.], 'beta':[.1,.2,.3,.4,.5,.6]}
    new=m.target_outputs(**batch)
    torch.testing.assert_close(old['logmark'][0],new['logmark'][0])
    torch.testing.assert_close(new['logmark'].exp().sum(1),torch.ones(4))
    for i in range(1,4):
        prev=int(old['previous'][i]); keep=[j for j in range(1,7) if j!=prev]
        torch.testing.assert_close(old['logmark'][i,keep].softmax(0),new['logmark'][i,keep].softmax(0))


@pytest.mark.parametrize('mode', ['U','G','D'])
def test_modified_generation_replays_full_history_after_window_boundary(mode):
    m=make(mode).eval(); lengths=[37,65]; captured=[]
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
        targets=([t] if t<37 else [])+[37+t]
        actual=m.target_outputs(**w.batch(targets))
        torch.testing.assert_close(actual['hidden'],captured[t],rtol=1e-5,atol=1e-6)
    assert sample['valid_mask'].sum()==102
    assert (sample['raw_amount'][sample['valid_mask']]>=0).all()
