"""Bounded, causal location outputs; original GMR buffers stay frozen."""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import torch
from benchmarks.argn_clock_regression import ClockRegression
from benchmarks.argn_clock_evolution import ProgressClock
from benchmarks.argn_time_density import time_density_generation

ARMS = ['event_two', 'customer_two', 'position_six', 'personal_six']
POSITION_KNOTS = [25., 125., 350., 750., 1500., 2500.]
JOINT_KNOTS = [25., 350., 2500.]
LIMIT = .35


def hats_numpy(x, knots):
    """Continuous linear interpolation, with constant boundary extrapolation."""
    knots = np.asarray(knots)
    x = np.clip(x, knots[0], knots[-1])
    i = np.searchsorted(knots, x, side='right').clip(1, len(knots)-1)-1
    t = (x-knots[i])/(knots[i+1]-knots[i])
    out = np.zeros((len(x), len(knots)))
    out[np.arange(len(x)), i] = 1-t
    out[np.arange(len(x)), i+1] = t
    return out


def basis_numpy(arm, x, valid, clock_limits):
    if arm in ARMS[:2]:
        return np.c_[np.ones(len(x)), np.exp(-np.expm1(x[:, 1])/50)]
    if arm == 'position_six':
        return hats_numpy(x[:, 1], np.log1p(POSITION_KNOTS))
    pos = hats_numpy(x[:, 1], np.log1p(JOINT_KNOTS))
    q = np.clip((x[:, 0]-clock_limits[0])/(clock_limits[1]-clock_limits[0]), 0, 1)
    q = np.where(valid > .5, q, .5)
    return (pos[:, :, None]*np.c_[1-q, q][:, None, :]).reshape(-1, 6)


def hats_torch(x, knots):
    x = x.clamp(knots[0], knots[-1])
    i = torch.searchsorted(knots, x.contiguous(), right=True).clamp(1, len(knots)-1)-1
    t = (x-knots[i])/(knots[i+1]-knots[i])
    out = x.new_zeros((len(x), len(knots)))
    out.scatter_(1, i[:, None], (1-t)[:, None])
    out.scatter_(1, (i+1)[:, None], t[:, None])
    return out


class AdaptiveClock(ClockRegression):
    def __init__(self, dim, arm, components=3, hidden=64):
        super().__init__(dim, arm, components, hidden)
        assert arm in ARMS
        self.register_buffer('correction', torch.zeros(2 if arm in ARMS[:2] else 6))
        self.register_buffer('clock_limits', torch.tensor([0., 1.]))

    def basis(self, clock, valid):
        if self.arm in ARMS[:2]:
            return torch.stack([torch.ones_like(clock[:, 1]), torch.exp(-torch.expm1(clock[:, 1])/50)], -1)
        knots = POSITION_KNOTS if self.arm == 'position_six' else JOINT_KNOTS
        pos = hats_torch(clock[:, 1], clock.new_tensor(knots).log1p())
        if self.arm == 'position_six':
            return pos
        q = ((clock[:, 0]-self.clock_limits[0])/(self.clock_limits[1]-self.clock_limits[0])).clamp(0, 1)
        q = torch.where(valid > .5, q, .5)
        return (pos[:, :, None]*torch.stack([1-q, q], -1)[:, None, :]).reshape(-1, 6)

    def forward(self, base, phase, clock, valid):
        logits, location, scale = super().forward(base, phase, clock[:, 0], valid)
        delta = (self.basis(clock, valid)@self.correction).clamp(-LIMIT, LIMIT)*phase.eq(0)
        return logits, location+delta[:, None], scale


@contextmanager
def adaptive_generation(*args, **kwargs):
    with patch('benchmarks.argn_time_density.TimeDensity', AdaptiveClock), \
         patch('benchmarks.argn_time_density.PastClock', ProgressClock):
        with time_density_generation(*args, **kwargs) as audit:
            yield audit
