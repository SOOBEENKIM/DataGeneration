from __future__ import annotations

from math import erf, exp, log, sqrt

import numpy as np


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0))))(x)


def duration_pmf(mu: float, sigma: float, d_max: int) -> np.ndarray:
    edges_lo = np.maximum(np.arange(1, d_max + 1) - 0.5, 0)
    edges_hi = np.arange(1, d_max + 1) + 0.5
    lo = np.where(edges_lo > 0, (np.log(np.maximum(edges_lo, 1e-300)) - mu) / sigma, -np.inf)
    hi = (np.log(edges_hi) - mu) / sigma
    pmf = _normal_cdf(hi) - _normal_cdf(lo)
    pmf[-1] = 1.0 - _normal_cdf(np.array([(log(d_max - 0.5) - mu) / sigma]))[0]
    return pmf / pmf.sum()


def calibrated_duration_pmf(target_mean: float, sigma: float, d_max: int) -> tuple[float, np.ndarray]:
    lo, hi = log(target_mean) - 3, log(target_mean) + 3
    values = np.arange(1, d_max + 1)
    for _ in range(80):
        mid = (lo + hi) / 2
        pmf = duration_pmf(mid, sigma, d_max)
        if float(pmf @ values) < target_mean:
            lo = mid
        else:
            hi = mid
    mu = (lo + hi) / 2
    return mu, duration_pmf(mu, sigma, d_max)


def residual_pmf(pmf: np.ndarray) -> np.ndarray:
    survival = np.cumsum(pmf[::-1])[::-1]
    return survival / (pmf @ np.arange(1, len(pmf) + 1))


def sample_equilibrium_alternating(
    rng: np.random.Generator,
    length: int,
    burst_pmf: np.ndarray,
    normal_pmf: np.ndarray,
    pi_burst: float,
) -> np.ndarray:
    state = int(rng.random() < pi_burst)
    pmfs = (normal_pmf, burst_pmf)
    values = np.arange(1, len(burst_pmf) + 1)
    p = pmfs[state]
    length_biased = values * p / (values @ p)
    total = int(rng.choice(values, p=length_biased))
    residual = total - int(rng.integers(0, total))
    out = np.empty(length, dtype=np.int8)
    pos = 0
    while pos < length:
        take = min(residual, length - pos)
        out[pos : pos + take] = state
        pos += take
        state = 1 - state
        if pos < length:
            residual = int(rng.choice(values, p=pmfs[state]))
    return out
