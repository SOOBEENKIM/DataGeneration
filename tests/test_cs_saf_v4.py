import ast
from pathlib import Path
import numpy as np
import pytest
import torch
from models.cs_saf_v3 import CSSAFv3
from models.cs_saf_v4 import CSSAFv4, CANDIDATE
from models.cof_seqgen_saf import fit_train_only_gap_support
from experiments.cs_saf_v4 import load_contract, contrasts, accuracy_screens, COLUMNS


def arms():
    support = fit_train_only_gap_support([.1, .5, 1, 3, 8])
    pi = torch.tensor([[.1,.2,.3,.1,.3],[.3,.1,.2,.3,.1]], dtype=torch.float64)
    kwargs = dict(receiver_vocab_size=7, static_categorical_vocab_sizes=(5,))
    er = CSSAFv4(support, pi, **kwargs)
    with torch.no_grad():
        er.history_interaction_weight.uniform_(-.7, .7)
        er.route_interaction_weight.uniform_(-.7, .7)
    e = CSSAFv3('CS3-E1', support, pi, **kwargs)
    r = CSSAFv3('CS3-R1', support, pi, **kwargs)
    e.load_state_dict(er.state_dict()); r.load_state_dict(er.state_dict())
    return er, e, r


def inputs():
    return dict(gap=torch.tensor([[float('nan'),.1,.5,1],[float('nan'),.5,8,float('nan')]]),
        receiver=torch.tensor([[3,3,4,4],[4,5,5,0]]),
        numeric_value=torch.tensor([[1.,2,3,1],[2.,1,3,0]]),
        valid_mask=torch.tensor([[True]*4,[True]*3+[False]]),
        static_categorical=(torch.tensor([3,4]),))


def grads(loss, model):
    return torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True, retain_graph=True)


def same_gradients(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if x is None or y is None:
            assert x is None and y is None
        else:
            assert torch.equal(x, y)


def test_raw_forward_base_and_gradients_exactly_E_penalty_and_gradients_exactly_R():
    torch.manual_seed(83)
    er, e, r = arms()
    context = torch.randn(8, 136); codes = torch.tensor([3,4]*4)
    assert torch.equal(er.logit_grid(context, codes), e.logit_grid(context, codes))
    assert not torch.allclose(er.logit_grid(context, codes), r.logit_grid(context, codes))
    terms = [m.loss_terms(**inputs()) for m in (er, e, r)]
    assert all(torch.equal(terms[0][k], terms[1][k]) for k in terms[0])
    objectives = [m.objective(t, train_entities=2, transition_counts_by_code={3:3,4:2})
                  for m, t in zip((er, e, r), terms)]
    same_gradients(grads(objectives[0][1], er), grads(objectives[1][1], e))
    assert torch.equal(objectives[0][2], objectives[2][2])
    same_gradients(grads(objectives[0][2], er), grads(objectives[2][2], r))
    torch.testing.assert_close(objectives[0][0], objectives[1][0]+.01*objectives[2][2])
    direct = torch.autograd.grad(objectives[0][2],
        (er.history_interaction_weight, er.copy_base.weight, er.copy_base.bias), allow_unused=True)
    assert direct == (None, None, None)
    assert er.architecture_contract()['candidate'] == CANDIDATE
    assert not er.architecture_contract()['centered']


def test_initial_parameters_rng_and_zero_gap_match_E_without_extra_capacity():
    er, e, _ = arms()
    def fresh(kind):
        torch.manual_seed(20260930)
        args = (er.support, er.reference_probabilities)
        kw = dict(receiver_vocab_size=7, static_categorical_vocab_sizes=(5,))
        m = CSSAFv4(*args, **kw) if kind == 'ER' else CSSAFv3('CS3-E1', *args, **kw)
        return m, torch.get_rng_state()
    a, rng_a = fresh('ER'); b, rng_b = fresh('E')
    assert torch.equal(rng_a, rng_b)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    context = torch.randn(137, 136); codes = torch.tensor([3,4]*68+[3])
    h, _, _ = er.components(context, codes)
    assert torch.count_nonzero(h) > 0
    grid = er.logit_grid(context, codes, zero_gap=True)
    torch.testing.assert_close(grid[:,0], er.copy_base(context)[:,0]+h)
    for curve in er.response_curves(context, torch.full((137,),3), zero_gap=True, static_codes=codes):
        assert torch.count_nonzero(curve-curve[:,:1]) == 0
    for k, gap in enumerate(er.support.representatives):
        logits = er.copy_logits(context, torch.full((137,), gap), static_codes=codes)
        assert torch.equal(logits, er.logit_grid(context, codes)[:,k])


def test_training_and_accuracy_persistence_bodies_unchanged_from_frozen_parent():
    trees = [ast.parse(Path(f'experiments/cs_saf_v{v}.py').read_text()) for v in (3, 4)]
    for name in ('train', 'save_accuracy'):
        bodies = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name) for tree in trees]
        assert ast.dump(bodies[0], include_attributes=False) == ast.dump(bodies[1], include_attributes=False)
    c, cfg = load_contract()
    assert c['scientific_fits'] == 2 and c['candidates'] == [CANDIDATE]
    assert c['objective']['regularization_coefficient'] == .01
    assert cfg['pilot_gate']['primary_candidate'] == CANDIDATE
    assert not c['pilot_gate']['automatic_later_stage']


def test_contrasts_align_entities_and_compute_factorial_difference(tmp_path):
    folders = {}
    shifts = {CANDIDATE: .03, 'CS3-E1': .01, 'CS3-C1': .02, 'CS3-R1': .025, 'CS2-U1': 0.}
    key = 'best_validation_label_1_metrics'; ids = key.replace('_metrics', '_entity_ids')
    for c, shift in shifts.items():
        folder = tmp_path/c; folder.mkdir(); folders[c] = folder
        np.savez(folder/'accuracy_arrays.npz', **{key: np.ones((3,len(COLUMNS)))*shift, ids: ['a','b','c']})
    out = contrasts(folders)
    assert out['ER_minus_E'][key]['grid_mark_TV']['mean'] == pytest.approx(.02)
    assert out['penalty_by_forward_interaction'][key]['grid_mark_TV']['mean'] == pytest.approx(.015)
    np.savez(folders['CS3-E1']/'accuracy_arrays.npz', **{key: np.zeros((3,len(COLUMNS))), ids: ['b','a','c']})
    with pytest.raises(ValueError, match='alignment'):
        contrasts(folders)


@pytest.mark.parametrize('null,active,passed', [(-.001,-.002,True),(-.001,0.,True),
    (0.,-.002,False),(-.001,.00001,False),(.00001,-.002,False)])
def test_screen_requires_both_accuracy_directions(null, active, passed):
    paired = {str(k): {f'ER_minus_{c}': {f'best_validation_label_{y}_metrics':
        {'grid_mark_TV': {'mean': active if (k,y)==(1,1) else null}}
        for y in (0,1)} for c in ('E','U')} for k in (0,1)}
    assert all(x['PASS'] == passed for x in accuracy_screens(paired).values())
