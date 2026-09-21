import copy
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import torch
from data.cs_saf_external import TargetWindows
from scripts.diagnose_cs_saf_external_amount import survival, decompose, hidden, paired_parameters, boundary_audit
from tests.test_cs_saf_external_controls import make, sequences


def test_tail_mass_matches_integrated_log_density_and_zero_atom():
    p = dict(weight=np.array([[.1,.3,.6]]), mu=np.array([[-2.,1.,4.]]),
             sigma=np.array([[.3,1.,2.]]), p0=np.array([.2]))
    cuts = np.array([.01, 10., 10000.])
    actual, _ = survival(p, cuts)
    expected = [.8*quad(lambda y: sum(w*norm.pdf(y, loc=m, scale=s) for w,m,s in
                zip(p['weight'][0],p['mu'][0],p['sigma'][0])), np.log(c), np.inf)[0] for c in cuts]
    np.testing.assert_allclose(actual[0], expected, atol=1e-10)


def test_crossed_history_current_and_amount_substitutions_obey_causality():
    model = make('D').eval()
    # Initialization intentionally ignores context in the amount head, so give
    # this engineering fixture a nontrivial context dependence.
    torch.manual_seed(37)
    with torch.no_grad():
        model.value_head.weight.normal_(0, .05)
    rw = TargetWindows(sequences())
    real = rw.batch([0,1,31,32,36,37,38,101])
    gen = copy.deepcopy(real)
    gen['numeric_value'] += 2
    gen['receiver'][gen['valid_mask']] = 6
    gen['gap'][gen['valid_mask'] & torch.isfinite(gen['gap'])] = 3.
    with torch.no_grad():
        p = paired_parameters(model, real, gen)
        same = paired_parameters(model, real, real)
        for k in same:
            for field in same[k]:
                np.testing.assert_array_equal(same[k][field], same['RR'][field])
        # Current amount and current auxiliary are generated AFTER this density.
        changed = copy.deepcopy(real)
        rows = torch.arange(len(real['target_position'])); pos = real['target_position']
        changed['numeric_value'][rows,pos] = 999
        changed['auxiliary_categorical'][0][rows,pos] = 1
        torch.testing.assert_close(hidden(model, changed), hidden(model, real), rtol=0, atol=0)
        first = pos.eq(0).numpy()
        for k in ('GG','AG','OG'):
            np.testing.assert_array_equal(p[k]['mu'][first], p['RG']['mu'][first])
        assert abs(p['GR']['mu']-p['RR']['mu']).max() > 1e-5
        boundary_audit(model, rw)


def test_symmetric_decomposition_accounts_for_interaction_and_total():
    rr,gr,rg,gg = (np.array([v, v/2]) for v in (.01,.02,.04,.1))
    d = decompose(rr,gr,rg,gg)
    np.testing.assert_allclose(d['history']+d['current'], gg-rr, atol=1e-16)
    np.testing.assert_allclose(d['interaction'], [.05,.025], atol=1e-16)
