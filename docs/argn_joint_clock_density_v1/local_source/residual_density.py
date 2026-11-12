"""Residual distributions around immutable init_only; no backbone mutation."""
from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'artifacts/argn_clock_initial_coupling_v1/local_source'),
                str(ROOT/'artifacts/argn_clock_adaptive_v1/local_source')]
from initial_clock import InitialClock
from benchmarks.argn_clock_evolution import ProgressClock
from benchmarks.argn_time_density import time_density_generation


class ResidualDensity(InitialClock):
    def __init__(self, dim, arm, components=3, hidden=64):
        super().__init__(dim, arm, components, hidden)
        self.arm = arm
        self.register_buffer('base_center', torch.zeros(dim))
        self.register_buffer('base_scale', torch.ones(dim))
        # Identical hidden and final parameter counts in every arm. In the
        # mean-only control the other six outputs are explicitly inactive.
        self.residual = nn.Sequential(nn.Linear(dim+7, hidden), nn.SiLU(),
                                      nn.Linear(hidden, 3*components))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, base, phase, clock, valid):
        logits, location, scale = super().forward(base, phase, clock, valid)
        features = torch.cat([((base-self.base_center)/self.base_scale).clamp(-8,8),
            F.one_hot(phase, 4).to(base.dtype), (clock[:, :1]-10)/2,
            clock[:, 1:2]/8, valid[:,None]], -1)
        dl, dm, ds = self.residual(features).chunk(3, -1)
        # Preserve the entire first five interval laws, including fraud phases.
        active = (torch.expm1(clock[:,1]) > 5.01).to(base.dtype)[:,None]
        location = location + active*.35*torch.tanh(dm)
        if self.arm.startswith('mixture'):
            logits = logits + active*.5*torch.tanh(dl)
            scale = scale*torch.exp(active*.20*torch.tanh(ds))
        return logits, location, scale


def implicit_sample(logits, mu, scale, uniform, normal, quantized=True):
    """Exact mixture forward, univariate implicit-CDF derivative before clipping.

    The integer-second floor uses a straight-through derivative ONLY in training.
    This is a biased surrogate of the quantized rollout, not an unbiased claim.
    """
    probs = logits.softmax(-1)
    with torch.no_grad():
        component = (uniform[:,None] > probs.cumsum(-1)).sum(-1).clamp(max=2)
        z0 = mu.gather(1, component[:,None])[:,0] + scale.gather(1, component[:,None])[:,0]*normal
    a = (z0[:,None]-mu)/scale
    cdf = (probs*torch.special.ndtr(a)).sum(-1)
    pdf = (probs*torch.exp(-a.square()/2)/(scale*np.sqrt(2*np.pi))).sum(-1)
    z = z0 - (cdf-cdf.detach())/pdf.detach().clamp(min=1e-8)
    gap = torch.expm1(z.clamp(0, float(np.log1p(1045691.))))
    if quantized:
        gap = gap + (torch.floor(gap+1e-8)-gap).detach()
    return gap


def clock_tensor(history, positions, fallback):
    """history is chronological last-20 RAW gaps; missing entries are NaN.

    Use the mean of the middle order statistics, as numpy/pandas do; torch's
    median alone would choose the lower middle for even window sizes.
    """
    sorted_gaps = torch.sort(history.clamp(max=604800.).nan_to_num(float('inf')), dim=-1).values
    count = torch.isfinite(history).sum(-1)
    lower = ((count-1)//2).clamp(min=0)
    upper = (count//2).clamp(max=19)
    med = (sorted_gaps.gather(1,lower[:,None])[:,0]+sorted_gaps.gather(1,upper[:,None])[:,0])/2
    valid = (count>=5)&(med>0)&torch.isfinite(med)
    med = torch.where(valid,med,torch.full_like(med,fallback))
    return torch.stack([torch.log1p(med),torch.log1p(positions.to(med.dtype))],-1),valid.to(med.dtype)


@contextmanager
def residual_generation(*args, **kwargs):
    with patch('benchmarks.argn_time_density.TimeDensity', ResidualDensity), \
         patch('benchmarks.argn_time_density.PastClock', ProgressClock):
        with time_density_generation(*args, **kwargs) as audit:
            yield audit
