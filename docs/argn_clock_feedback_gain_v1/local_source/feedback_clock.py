"""Bounded local gains on GMR component regression and the existing calibration."""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import torch
from initial_clock import InitialClock
from benchmarks.argn_clock_evolution import ProgressClock
from benchmarks.argn_time_density import time_density_generation


def envelope(x):
    a,b,c=np.log1p([50.,125.,350.]);t=x[:,1]
    return np.clip(np.minimum((t-a)/(b-a),(c-t)/(c-b)),0,1)


class FeedbackClock(InitialClock):
    def __init__(self,dim,arm,components=3,hidden=64):
        super().__init__(dim,'init_only',components,hidden)
        self.register_buffer('feedback_parameters',torch.tensor([0.,1.,0.]))

    def forward(self,base,phase,clock,valid):
        logits,loc,sd=super().forward(base,phase,clock,valid)
        mu,cov=self.means[phase],self.covariances[phase]
        regression=cov[:,:,1,0]/cov[:,:,0,0]*(clock[:,0,None]-mu[:,:,0])
        delta=(self.basis(clock,valid)@self.correction).clamp(-.35,.35)
        low,mid,high=clock.new_tensor([50.,125.,350.]).log1p()
        weight=torch.minimum((clock[:,1]-low)/(mid-low),(high-clock[:,1])/(high-mid)).clamp(0,1)
        n=clock[:,1].expm1().round()
        weight=weight*((n>50)&(n<350)&phase.eq(0))
        a,g,b=self.feedback_parameters
        change=(weight[:,None]*(-a*regression+(g-1)*delta[:,None]+b)).clamp(-.35,.35)
        return logits,loc+change,sd


@contextmanager
def feedback_generation(*args,**kwargs):
    with patch('benchmarks.argn_time_density.TimeDensity',FeedbackClock),patch('benchmarks.argn_time_density.PastClock',ProgressClock):
        with time_density_generation(*args,**kwargs) as audit:yield audit
