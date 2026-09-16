import numpy as np
import pytest
import torch

from experiments.cs_saf_loss_control import (
    LossControlModel, CANDIDATES, COLUMNS, load_contract,
    central_intervals, diagnose, paired_contrasts,
)
from models.cs_saf_v2 import CSSAFv2, BANK_NAMES
from models.cof_seqgen_saf import fit_train_only_gap_support


def model(candidate):
    return LossControlModel(candidate, fit_train_only_gap_support([.1,.5,1,3,8]),
                            receiver_vocab_size=7, static_categorical_vocab_sizes=(5,))


def inputs():
    return {"gap": torch.tensor([[float('nan'),.1,.5,1],[float('nan'),.5,8,float('nan')]]),
            "receiver": torch.tensor([[3,3,4,4],[4,5,5,0]]),
            "numeric_value": torch.tensor([[1.,2,3,1],[2.,1,3,0]]),
            "valid_mask": torch.tensor([[True]*4,[True]*3+[False]]),
            "static_categorical": (torch.tensor([3,4]),)}


def open_banks(m):
    with torch.no_grad():
        m.route_interaction_weight.copy_(torch.linspace(-.4,.8,32).reshape(2,16))


def test_three_objectives_have_identical_initial_states_and_rng():
    states, rng = [], []
    for c in CANDIDATES:
        torch.manual_seed(20260930)
        m = model(c); states.append(m.state_dict()); rng.append(torch.get_rng_state())
    assert all(states[0].keys() == s.keys() for s in states)
    assert all(torch.equal(states[0][k], s[k]) for s in states for k in s)
    assert all(torch.equal(rng[0], r) for r in rng)


@pytest.mark.parametrize('candidate',['CS2-U1','CS2-B1'])
def test_historical_forward_objective_and_gradients_are_unchanged(candidate):
    torch.manual_seed(42); m=model(candidate)
    torch.manual_seed(42); old=CSSAFv2(candidate,m.support,receiver_vocab_size=7,static_categorical_vocab_sizes=(5,))
    open_banks(m);open_banks(old)
    terms=m.loss_terms(**inputs()); previous=old.loss_terms(**inputs())
    assert all(torch.equal(terms[k], previous[k]) for k in terms)
    loss=m.objective(terms,train_entities=2,transition_counts_by_code={3:3,4:2})
    expected=old.objective(previous,train_entities=2,transition_counts_by_code={3:3,4:2})
    assert all(torch.equal(a,b) for a,b in zip(loss,expected))
    loss[0].backward();expected[0].backward()
    assert all((a.grad is None and b.grad is None) or torch.equal(a.grad,b.grad)
               for a,b in zip(m.parameters(),old.parameters()))


def test_global_auxiliary_formula_and_gradient_match_direct_transition_mean():
    m=model('CS2-A1');open_banks(m);terms=m.loss_terms(**inputs())
    loss,base,aux=m.objective(terms,train_entities=2,transition_counts_by_code={3:3,4:2})
    expected=terms['repeat_per_entity'].sum()/5
    torch.testing.assert_close(aux,expected)
    torch.testing.assert_close(loss,base+expected)
    parameters=tuple(m.parameters())
    a=torch.autograd.grad(aux,parameters,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(expected,parameters,allow_unused=True)
    for x,y in zip(a,b):
        if x is None: assert y is None
        else: torch.testing.assert_close(x,y)


@pytest.mark.parametrize('candidate',['CS2-A1','CS2-B1'])
def test_fixed_train_denominators_survive_unequal_length_single_context_batches(candidate):
    m=model(candidate);open_banks(m);x=inputs();counts={3:3,4:2}
    full=m.objective(m.loss_terms(**x),train_entities=2,transition_counts_by_code=counts)[2]
    partial=[]
    for i in (0,1):
        one={k:tuple(t[i:i+1] for t in v) if isinstance(v,tuple) else v[i:i+1] for k,v in x.items()}
        partial.append(m.objective(m.loss_terms(**one),train_entities=2,transition_counts_by_code=counts)[2])
    torch.testing.assert_close(full,torch.stack(partial).mean())
    if candidate=='CS2-A1':
        wrong=(m.loss_terms(**x)['repeat_per_entity']/torch.tensor([3,2])).mean()
        assert not torch.isclose(full,wrong)


def test_direct_route_base_derivative_equals_repeat_derivative_and_global_coefficients():
    m=model('CS2-A1');open_banks(m);terms=m.loss_terms(**inputs())
    parameters=tuple(getattr(m,n) for n in BANK_NAMES)
    a=torch.autograd.grad(terms['mark_sum'],parameters,retain_graph=True)
    b=torch.autograd.grad(terms['repeat_per_entity'].sum(),parameters,retain_graph=True)
    for x,y in zip(a,b): torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-5)
    loss,_,_=m.objective(terms,train_entities=2,transition_counts_by_code={3:3,4:2})
    expected=sum((count/7+count/5)*terms['repeat_per_entity'][i]/count for i,count in enumerate([3,2]))
    a=torch.autograd.grad(loss,parameters,retain_graph=True)
    b=torch.autograd.grad(expected,parameters)
    for x,y in zip(a,b): torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-5)
    assert sum(count/7+count/5 for count in [3,2]) == pytest.approx(sum(count/7+.5 for count in [3,2]))


def as_data(x):
    return {**{k:v for k,v in x.items() if k!='static_categorical'},'codes':x['static_categorical'][0],
            'lengths':x['valid_mask'].sum(1),'entity_ids':['entity_a','entity_b']}


def test_diagnostic_factual_bce_matches_likelihood_and_entity_alignment():
    m=model('CS2-A1');open_banks(m);x=inputs();data=as_data(x)
    intervals=central_intervals(m,data)
    summary,arrays=diagnose(m,data,intervals,torch.device('cpu'),batch_size=1)
    terms=m.loss_terms(**x)
    for label in (0,1):
        values=arrays[f'label_{label}_metrics'][0]
        assert arrays[f'label_{label}_entity_ids'].tolist()==[data['entity_ids'][label]]
        expected=float(terms['repeat_per_entity'][label]/(data['lengths'][label]-1))
        assert values[4]==pytest.approx(expected,abs=1e-6)
        assert values[6]==pytest.approx(values[4]-values[5])
        assert 0<=values[2]<=values[0]+1e-7
        assert 0<=values[3]<=values[1]+1e-7
        assert summary['groups'][str(label)]['entities']==1


def test_initial_zero_route_has_no_bce_difference_or_response():
    m=model('CS2-U1');data=as_data(inputs())
    _,arrays=diagnose(m,data,central_intervals(m,data),torch.device('cpu'))
    for label in (0,1):
        values=arrays[f'label_{label}_metrics']
        assert np.count_nonzero(values[:,:4])==0
        assert np.count_nonzero(values[:,6])==0


def test_paired_contrasts_reject_reordered_entities_and_preserve_signed_differences(tmp_path):
    folders={c:tmp_path/c for c in CANDIDATES}
    for i,(c,p) in enumerate(folders.items()):
        p.mkdir()
        np.savez(p/'entity_diagnostics.npz',best_train_label_0_metrics=np.full((2,len(COLUMNS)),i,dtype=float),
                 best_train_label_0_entity_ids=np.array(['a','b']))
    result=paired_contrasts(folders)
    assert result['CS2-A1_minus_CS2-U1']['metrics']['best_train_label_0_metrics']['copy_range']['mean']==1
    np.savez(folders['CS2-A1']/'entity_diagnostics.npz',best_train_label_0_metrics=np.ones((2,len(COLUMNS))),
             best_train_label_0_entity_ids=np.array(['b','a']))
    with pytest.raises(ValueError,match='align'): paired_contrasts(folders)


def test_registered_scope_is_six_fits_with_unchanged_budget_and_no_expansion():
    contract,cfg=load_contract()
    assert contract['scientific_fits']==6 and contract['prevalence']==.05
    assert contract['candidates']==list(CANDIDATES)
    assert cfg['training']['epochs']==50 and cfg['training']['checkpoint_selection']=='global_validation_base_nll'
    assert contract['boundaries']['held_out_test_access'] is False
    assert contract['boundaries']['later_prevalences'] is False
