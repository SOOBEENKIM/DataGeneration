"""Shared train-only controls on observable repetition, for U and official ARGN.

These are calibration/statistical baselines, not a new neural architecture.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch


def assign_bins(code, group, edges):
    out = np.empty(len(code), dtype=np.int64)
    for g in (0, 1):
        ix = np.asarray(group) == g
        out[ix] = np.searchsorted(edges[g], np.asarray(code)[ix], side='right')
    return out


def fit_controls(frame, config):
    assert set(frame.group.unique()) == {0, 1}
    assert set(frame.y.unique()) <= {0, 1}
    edges = [np.quantile(frame.loc[frame.group == g, 'gap_code'], config['coarse_quantiles']).tolist()
             for g in (0, 1)]
    assert all(len(set(e)) == 4 for e in edges), 'coarse gap bins must be identifiable'
    bins = assign_bins(frame.gap_code.to_numpy(), frame.group.to_numpy(), edges)
    results = {}
    for kind in ('level', 'gap', 'direct'):
        parameters, fits = [], []
        for group in (0, 1):
            ix = frame.group.to_numpy() == group
            b = np.zeros(ix.sum(), dtype=int) if kind == 'level' else bins[ix]
            n = 1 if kind == 'level' else 5
            y = frame.y.to_numpy(float)[ix]
            z = logit(frame.p.to_numpy(float)[ix].clip(1e-9, 1-1e-9))
            count = np.bincount(b, minlength=n)
            assert (count > 0).all()
            w = count / count.sum()
            if kind == 'direct':
                q = config['direct_beta_pseudocount']
                values = (np.bincount(b, weights=y, minlength=n)+q)/(count+2*q)
                fit = dict(success=True, counts=count.tolist(), parameters=values.tolist())
            else:
                ridge = config['ridge']
                def objective(delta):
                    t = z + delta[b]
                    loss = np.mean(np.logaddexp(0, t)-y*t) + .5*ridge*np.dot(w, delta**2)
                    grad = np.bincount(b, weights=expit(t)-y, minlength=n)/len(y)+ridge*w*delta
                    return float(loss), grad
                opt = minimize(objective, np.zeros(n), jac=True, method='L-BFGS-B',
                               bounds=[(-config['offset_bound'], config['offset_bound'])]*n,
                               options=dict(maxiter=500, ftol=1e-13, gtol=1e-9))
                assert opt.success and objective(opt.x)[0] <= objective(np.zeros(n))[0]+1e-10
                values = opt.x
                fit = dict(success=bool(opt.success), message=str(opt.message), iterations=int(opt.nit),
                           counts=count.tolist(), parameters=values.tolist(),
                           penalized_nll_before=objective(np.zeros(n))[0], penalized_nll_after=objective(values)[0],
                           bound_hits=np.flatnonzero(np.abs(values)>=config['offset_bound']-1e-6).tolist())
            parameters.append(np.repeat(values, 5).tolist() if n == 1 else values.tolist())
            fits.append(fit)
        results[kind] = dict(kind=kind, edges=edges, parameters=parameters, groups=fits,
                             fit_split='outer_train', validation_used=False, oracle_used=False,
                             target='observable_repeat', native_gap_codes=True)
    return results


def apply_numpy(p, code, group, control):
    if control is None or control['kind'] == 'raw':
        return np.asarray(p).copy()
    b = assign_bins(np.asarray(code), np.asarray(group), control['edges'])
    parameter = np.asarray(control['parameters'])[np.asarray(group, dtype=int), b]
    return parameter if control['kind'] == 'direct' else expit(logit(np.clip(p, 1e-9, 1-1e-9))+parameter)


def apply_torch(p, code, group, control):
    if control is None or control['kind'] == 'raw':
        return p
    bins = torch.empty_like(code, dtype=torch.long)
    for g in (0, 1):
        ix = group == g
        edges = torch.as_tensor(control['edges'][g], dtype=torch.float64, device=p.device)
        bins[ix] = torch.searchsorted(edges, code[ix].double().contiguous(), right=True)
    parameters = torch.as_tensor(control['parameters'], dtype=p.dtype, device=p.device)[group.long(), bins]
    return parameters if control['kind'] == 'direct' else (torch.logit(p.double().clamp(1e-9, 1-1e-9))+parameters.double()).sigmoid().to(p.dtype)


def redistribute(p, previous, repeat_probability):
    """Replace the repeat mass, keeping relative nonrepeat probabilities intact."""
    other = p.clone().scatter(-1, previous[..., None], 0)
    mass = other.sum(-1, keepdim=True)
    if (mass <= 0).any():
        raise FloatingPointError('native model gives no nonrepeat support')
    out = other / mass * (1-repeat_probability[..., None])
    out.scatter_(-1, previous[..., None], repeat_probability[..., None])
    return out
