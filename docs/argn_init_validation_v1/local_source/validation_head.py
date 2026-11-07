"""Frozen existing initial/calibration architectures and a matched history control."""
from contextlib import contextmanager
from unittest.mock import patch
import torch
from initial_clock import InitialClock
from adaptive_clock import LIMIT
from benchmarks.argn_time_density import TimeDensity, time_density_generation
from benchmarks.argn_clock_evolution import ProgressClock


class HistoryClock(InitialClock):
    def __init__(self, dim, arm, components=3, hidden=64):
        super().__init__(dim, arm, components, hidden)
        self.history_head=TimeDensity(dim,'history_only',components,hidden)

    def forward(self,base,phase,clock,valid):
        logits,mu,sd=self.history_head(base,phase,clock[:,0],valid)
        delta=(self.basis(clock,valid)@self.correction).clamp(-LIMIT,LIMIT)*phase.eq(0)
        mu=mu+delta[:,None]
        n=torch.expm1(clock[:,1]).round()
        initial=self.use_initial & phase.eq(0) & n.ge(1) & n.le(5)
        logits=torch.where(initial[:,None],self.initial_weights.log()[None],logits)
        mu=torch.where(initial[:,None],self.initial_location[None],mu)
        sd=torch.where(initial[:,None],self.initial_scale[None],sd)
        return logits,mu,sd


def factory(dim,arm,components=3,hidden=64):
    return (HistoryClock if arm.startswith('history') else InitialClock)(dim,arm,components,hidden)


@contextmanager
def validation_generation(*args,**kwargs):
    with patch('benchmarks.argn_time_density.TimeDensity',factory), \
         patch('benchmarks.argn_time_density.PastClock',ProgressClock):
        with time_density_generation(*args,**kwargs) as audit:
            yield audit
