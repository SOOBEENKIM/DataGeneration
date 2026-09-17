"""Observable-filter joint samplers and explicit fixed-gap hybrid controls.

No model fitting, dataset latent arrays, future conditioning, or GPU operations.
"""
from __future__ import annotations
import numpy as np
import torch

MODES = ('JOINT_CONT', 'FIX_CONT', 'JOINT_BIN', 'FIX_BIN', 'FIX_MODELINFO')


def inverse_cdf(probabilities, uniform):
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(-1) <= 0).any():
        raise ValueError('invalid probabilities')
    cdf = np.cumsum(p/p.sum(-1, keepdims=True), axis=-1); cdf[..., -1] = 1.
    return (np.asarray(uniform)[..., None] >= cdf).sum(-1)


def gap_log_weights(oracle, gap, binned):
    if binned:
        bins = np.searchsorted(oracle.upper, gap, side='left')
        return np.log(oracle.bin_mass[:, bins].T)
    return -np.log(oracle.scales) - np.asarray(gap)[:, None]/oracle.scales


def condition(prior, log_weights):
    weights = np.exp(log_weights-log_weights.max(1, keepdims=True))
    posterior = prior*weights[:, :, None]
    return posterior/posterior.sum((1, 2), keepdims=True)


def mark_probabilities(oracle, prior, gap, previous, active, current_binned):
    posterior = prior.copy()
    posterior[active] = condition(prior[active], gap_log_weights(oracle, gap[active], current_binned))
    q = posterior.sum(-1)@oracle.copy_by_state
    categories = oracle.config.n_receiver_categories
    p = np.broadcast_to(((1-q)/categories)[:, None], (len(q), categories)).copy()
    p[np.arange(len(q)), previous] += q
    return p, q+(1-q)/categories


def advance_beliefs(oracle, mark_prior, gap_prior, gaps, repeats, active, past_binned):
    lg = gap_log_weights(oracle, gaps, past_binned)
    lm = np.log(np.where(repeats[:, None], oracle.repeat_by_state, 1-oracle.repeat_by_state))
    mark_weights = lm.copy(); mark_weights[active] += lg[active]
    new_mark = oracle.advance(condition(mark_prior, mark_weights))
    new_gap = oracle.advance(condition(gap_prior, lg))
    new_gap[active] = new_mark[active]
    return new_mark, new_gap


def _run(oracle, data, kappa, *, information, representatives=None, mode=None, seed=None):
    if information not in ('CONT', 'BIN', 'MODELINFO'): raise ValueError('unknown information pattern')
    if mode is not None and mode not in MODES: raise ValueError('unknown oracle mode')
    result = {k: v.clone() for k, v in data.items()}
    gap = result['gap'].numpy(); marks = result['receiver'].numpy()
    valid = result['valid_mask'].numpy(); active = (result['codes'].numpy() == 4) & bool(kappa)
    if not np.isnan(gap[:, 0]).all(): raise ValueError('first gap must be missing')
    if information == 'BIN' and mode == 'FIX_BIN':
        mask = np.isfinite(gap)
        gap[mask] = np.asarray(representatives, dtype=np.float32)[np.searchsorted(oracle.upper, gap[mask], side='left')]
    if mode:
        streams = np.random.SeedSequence(seed).spawn(4)
        gu = np.random.default_rng(streams[0]).random(gap.shape)
        mu = np.random.default_rng(streams[1]).random(gap.shape)
        component = np.random.default_rng(streams[3]).random(gap.shape)
    mark_prior = oracle.advance(oracle.initial(len(gap)))
    gap_prior = mark_prior.copy()
    predicted = np.zeros(gap.shape, dtype=float)
    for t in range(1, gap.shape[1]):
        ids = np.flatnonzero(valid[:, t]); a = active[ids]
        if mode == 'JOINT_CONT':
            burst = gap_prior[ids].sum(-1)[:, 1]
            z = (component[ids, t] < burst).astype(int)
            gap[ids, t] = (-np.log1p(-gu[ids, t])*oracle.scales[z]).astype(np.float32)
        elif mode == 'JOINT_BIN':
            p = gap_prior[ids].sum(-1)@oracle.bin_mass
            b = inverse_cdf(p, gu[ids, t])
            gap[ids, t] = np.asarray(representatives, dtype=np.float32)[b]
        probabilities, repeat = mark_probabilities(oracle, mark_prior[ids], gap[ids, t], marks[ids, t-1]-3,
                                                    a, information in ('BIN', 'MODELINFO'))
        predicted[ids, t] = repeat
        if mode:
            marks[ids, t] = inverse_cdf(probabilities, mu[ids, t])+3
        equality = marks[ids, t] == marks[ids, t-1]
        mark_prior[ids], gap_prior[ids] = advance_beliefs(oracle, mark_prior[ids], gap_prior[ids],
            gap[ids, t], equality, a, information == 'BIN')
    if (marks[valid] < 3).any() or (marks[valid] >= 3+oracle.config.n_receiver_categories).any():
        raise ValueError('reserved oracle output')
    nonfirst = valid.copy(); nonfirst[:, 0] = False
    if not np.isfinite(gap[nonfirst]).all() or (gap[nonfirst] < 0).any(): raise ValueError('invalid oracle gap')
    if not np.isfinite(predicted).all(): raise ValueError('nonfinite oracle probability')
    return result, predicted


def predict(oracle, data, kappa, information):
    return _run(oracle, data, kappa, information=information)[1]


def sample(oracle, data, kappa, representatives, mode, seed):
    information = 'MODELINFO' if mode == 'FIX_MODELINFO' else ('BIN' if mode.endswith('BIN') else 'CONT')
    return _run(oracle, data, kappa, information=information, representatives=representatives, mode=mode, seed=seed)


def monte_carlo_gate(oracle, representatives, *, n_per_group, seed):
    """Fixed sampler-validity check; no learned models or empirical test split."""
    n = 2*n_per_group; steps = 32
    data = {'gap': torch.full((n, steps), float('nan')), 'receiver': torch.full((n, steps), 3, dtype=torch.long),
            'numeric_value': torch.zeros((n, steps)), 'valid_mask': torch.ones((n, steps), dtype=torch.bool),
            'lengths': torch.full((n,), steps, dtype=torch.long),
            'codes': torch.cat((torch.full((n_per_group,), 3), torch.full((n_per_group,), 4)))}
    marginal = np.array([1-oracle.config.pi_burst, oracle.config.pi_burst])
    mass = marginal@oracle.bin_mass; r = float(marginal@oracle.repeat_by_state)
    reports = {}
    for mode in ('JOINT_CONT', 'JOINT_BIN'):
        generated, _ = sample(oracle, data, 1, representatives, mode, seed)
        g = generated['gap'].numpy()[:, 1:]
        m = generated['receiver'].numpy(); repeat = (m[:, 1:] == m[:, :-1])
        bins = np.searchsorted(oracle.upper, g, side='left')
        for label in (0, 1):
            sl = slice(label*n_per_group, (label+1)*n_per_group)
            features = [repeat[sl].mean(1)]
            expected = [r]
            for b in range(len(oracle.upper)):
                indicator = bins[sl] == b
                features.extend((indicator.mean(1), (indicator*repeat[sl]).mean(1)))
                joint = (marginal*oracle.bin_mass[:, b])@oracle.repeat_by_state if label else mass[b]*r
                expected.extend((mass[b], joint))
            values = np.stack(features, 1); means = values.mean(0)
            se = values.std(0, ddof=1)/np.sqrt(n_per_group)
            tolerances = np.maximum(.01, 6*se); errors = abs(means-np.asarray(expected))
            if (errors > tolerances).any(): raise AssertionError('joint sampler stationary moment check failed')
            reports[f'{mode}/context_{label}'] = {'PASS': True, 'max_error': float(errors.max()),
                'max_error_over_tolerance': float((errors/tolerances).max()), 'features': len(expected),
                'entities': n_per_group}
    return reports
