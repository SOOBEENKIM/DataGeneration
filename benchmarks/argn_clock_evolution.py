"""Progress-conditioned time density and a bounded rollout-trained location head."""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from benchmarks.argn_time_density import TimeDensity, PastClock, clock_features, time_density_generation


class ProgressClock(PastClock):
    def features(self, memory):
        clock, valid = super().features(memory)
        return np.c_[clock, np.log1p(memory[:, 0])], valid


def progress_features(metadata, fallback):
    clock, valid = clock_features(metadata, fallback)
    return np.c_[clock, np.log1p(metadata.event_index.to_numpy())], valid


class ProgressDensity(TimeDensity):
    def __init__(self, dim, arm, components=3, hidden=64):
        nn.Module.__init__(self)
        self.dim, self.arm, self.components, self.hidden = dim, arm, components, hidden
        self.register_buffer('weights', torch.ones(4, components)/components)
        self.register_buffer('means', torch.zeros(4, components, 3))
        self.register_buffer('covariances', torch.eye(3).expand(4, components, 3, 3).clone())
        self.register_buffer('correction', torch.zeros(2))

    def forward(self, base, phase, clock, valid):
        mu, cov = self.means[phase], self.covariances[phase]
        delta = clock[:, None, :] - mu[:, :, :2]
        precision = torch.linalg.inv(cov[:, :, :2, :2])
        logits = self.weights[phase].log() - .5*(2*np.log(2*np.pi) +
            torch.linalg.slogdet(cov[:, :, :2, :2])[1] +
            torch.einsum('nki,nkij,nkj->nk', delta, precision, delta))
        beta = torch.einsum('nki,nkij->nkj', cov[:, :, 2, :2], precision)
        location = mu[:, :, 2] + (beta*delta).sum(-1)
        variance = cov[:, :, 2, 2] - (beta*cov[:, :, :2, 2]).sum(-1)
        n = torch.expm1(clock[:, 1])
        adjustment = (self.correction[0] + self.correction[1]*torch.exp(-n/50))*phase.eq(0)
        return logits, location + adjustment[:, None], variance.clamp(min=1e-8).sqrt()


class NumpyDensity:
    """Independent CPU evaluation of the same multivariate conditional law."""
    def __init__(self, payload):
        d = payload['state_dict']
        self.weights = d['weights'].double().numpy()
        self.means = d['means'].double().numpy()
        cov = d['covariances'].double().numpy()
        self.input_dim = self.means.shape[-1] - 1
        xx = cov[..., :self.input_dim, :self.input_dim]
        self.precision = np.linalg.inv(xx)
        self.logdet = np.linalg.slogdet(xx)[1]
        self.beta = np.einsum('pki,pkij->pkj', cov[..., -1, :self.input_dim], self.precision)
        self.scale = np.sqrt(np.maximum(cov[..., -1, -1] -
            (self.beta*cov[..., :self.input_dim, -1]).sum(-1), 1e-8))

    def parameters(self, phase, x, correction=(0., 0.)):
        # x always also carries position for the calibration; original GMR uses x[:,:1].
        delta = x[:, None, :self.input_dim] - self.means[phase, :, :self.input_dim]
        logits = np.log(self.weights[phase]) - .5*(self.input_dim*np.log(2*np.pi) +
            self.logdet[phase] + np.einsum('nki,nkij,nkj->nk', delta, self.precision[phase], delta))
        probs = np.exp(logits-logits.max(-1, keepdims=True)); probs /= probs.sum(-1, keepdims=True)
        loc = self.means[phase, :, -1] + (self.beta[phase]*delta).sum(-1)
        adjustment = (correction[0]+correction[1]*np.exp(-np.expm1(x[:, 1])/50))*(phase == 0)
        return probs, loc+adjustment[:, None], self.scale[phase]


@contextmanager
def evolution_generation(*args, **kwargs):
    with patch('benchmarks.argn_time_density.TimeDensity', ProgressDensity), \
         patch('benchmarks.argn_time_density.PastClock', ProgressClock):
        with time_density_generation(*args, **kwargs) as audit:
            yield audit
