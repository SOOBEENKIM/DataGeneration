"""Known Gaussian-mixture regression control for one conditional time output."""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from benchmarks.argn_time_density import TimeDensity,time_density_generation


class ClockRegression(TimeDensity):
    def __init__(self,dim,arm,components=3,hidden=64):
        nn.Module.__init__(self)
        self.dim,self.arm,self.components,self.hidden=dim,arm,components,hidden
        self.register_buffer('weights',torch.ones(4,components)/components)
        self.register_buffer('means',torch.zeros(4,components,2))
        self.register_buffer('covariances',torch.eye(2).expand(4,components,2,2).clone())

    def forward(self,base,phase,clock,valid):
        mu=self.means[phase];cov=self.covariances[phase]
        delta=clock[:,None]-mu[:,:,0];var=cov[:,:,0,0]
        logits=self.weights[phase].log()-.5*(np.log(2*np.pi)+var.log()+delta.square()/var)
        location=mu[:,:,1]+cov[:,:,1,0]/var*delta
        scale=(cov[:,:,1,1]-cov[:,:,1,0].square()/var).clamp(min=1e-8).sqrt()
        return logits,location,scale


@contextmanager
def clock_regression_generation(*args,**kwargs):
    # A new, explicit adapter; the registered temporal implementation is unchanged.
    with patch('benchmarks.argn_time_density.TimeDensity',ClockRegression):
        with time_density_generation(*args,**kwargs) as audit:
            yield audit
