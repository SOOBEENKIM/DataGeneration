"""Initial normal-interval mixture plus a local, rollout-trained handoff correction."""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import torch
from adaptive_clock import AdaptiveClock, LIMIT
from benchmarks.argn_clock_regression import ClockRegression
from benchmarks.argn_clock_evolution import ProgressClock
from benchmarks.argn_time_density import time_density_generation


def handoff_basis(x):
    # Two log-position hats at 6 and 50, with a fixed zero endpoint at 200.
    t = x[:, 1]
    a, b, c = np.log1p([6., 50., 200.])
    first = np.clip((b-t)/(b-a), 0, 1)
    second = np.where(t <= b, (t-a)/(b-a), (c-t)/(c-b))
    return np.c_[first, np.clip(second, 0, 1)] * ((t >= a-1e-12) & (t < c-1e-12))[:, None]


class InitialClock(AdaptiveClock):
    def __init__(self, dim, arm, components=3, hidden=64):
        super().__init__(dim, 'personal_six', components, hidden)
        self.register_buffer('initial_weights', torch.ones(3)/3)
        self.register_buffer('initial_location', torch.zeros(3))
        self.register_buffer('initial_scale', torch.ones(3))
        self.register_buffer('use_initial', torch.tensor(False))
        self.register_buffer('handoff', torch.zeros(2))

    def forward(self, base, phase, clock, valid):
        logits, location, scale = ClockRegression.forward(self, base, phase, clock[:, 0], valid)
        t = clock[:, 1]
        a, b, c = torch.log1p(clock.new_tensor([6., 50., 200.]))
        first = ((b-t)/(b-a)).clamp(0, 1)
        second = torch.where(t <= b, (t-a)/(b-a), (c-t)/(c-b)).clamp(0, 1)
        # Round count only for structural boundaries; floating log1p conversion is harmless.
        n = torch.expm1(t).round()
        basis = torch.stack([first, second], -1) * ((n >= 6) & (n < 200))[:, None]
        delta = (self.basis(clock, valid)@self.correction + basis@self.handoff).clamp(-LIMIT, LIMIT)
        location = location + (delta*phase.eq(0))[:, None]
        initial = self.use_initial & phase.eq(0) & (n >= 1) & (n <= 5)
        logits = torch.where(initial[:, None], self.initial_weights.log()[None], logits)
        location = torch.where(initial[:, None], self.initial_location[None], location)
        scale = torch.where(initial[:, None], self.initial_scale[None], scale)
        return logits, location, scale


@contextmanager
def initial_generation(*args, **kwargs):
    with patch('benchmarks.argn_time_density.TimeDensity', InitialClock), \
         patch('benchmarks.argn_time_density.PastClock', ProgressClock):
        with time_density_generation(*args, **kwargs) as audit:
            yield audit
