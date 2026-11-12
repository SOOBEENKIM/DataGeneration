"""Keep the original GMR law; learn only its normal-state location correction."""
from contextlib import contextmanager
from unittest.mock import patch
import torch
from benchmarks.argn_clock_regression import ClockRegression
from benchmarks.argn_clock_evolution import ProgressClock
from benchmarks.argn_time_density import time_density_generation


class ClockLocation(ClockRegression):
    def __init__(self,dim,arm,components=3,hidden=64):
        super().__init__(dim,arm,components,hidden)
        self.register_buffer('correction',torch.zeros(2))

    def forward(self,base,phase,clock,valid):
        logits,location,scale=super().forward(base,phase,clock[:,0],valid)
        index=torch.expm1(clock[:,1])
        delta=(self.correction[0]+self.correction[1]*torch.exp(-index/50))*phase.eq(0)
        return logits,location+delta[:,None],scale


@contextmanager
def location_generation(*args,**kwargs):
    with patch('benchmarks.argn_time_density.TimeDensity',ClockLocation), \
         patch('benchmarks.argn_time_density.PastClock',ProgressClock):
        with time_density_generation(*args,**kwargs) as audit:
            yield audit
