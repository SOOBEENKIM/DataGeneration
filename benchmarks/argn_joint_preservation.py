"""Known conditional count-mixture and exposure-based onset hazard controls.

These change the generative factorization, not completed transaction rows.
Existing category/gap/amount outputs and fraud-exit risks are held fixed.
"""
from contextlib import contextmanager
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.mixture import GaussianMixture
import torch
from torch import nn
from torch.nn import functional as F
from mostlyai.engine._common import SLEN_SUB_COLUMN_PREFIX,encode_slen_sidx_sdec
from benchmarks.argn_episode_control import episode_generation,decode_length,episode_metadata
from benchmarks.argn_past_state import PastState


class JointCount(nn.Module):
    def __init__(self,dim,components=3,conditioned=True):
        super().__init__();self.dim=dim;self.components=components;self.conditioned=conditioned
        self.register_buffer('center',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim))
        self.net=nn.Sequential(nn.Linear(dim,32),nn.Tanh(),nn.Dropout(.1),nn.Linear(32,2+6*components))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        x=(x-self.center)/self.scale
        if not self.conditioned:x=torch.zeros_like(x)
        v=self.net(x);k=self.components
        label=v[:,:2];mix=v[:,2:2+2*k].reshape(-1,2,k)
        mean=v[:,2+2*k:2+4*k].reshape(-1,2,k)
        std=F.softplus(v[:,2+4*k:].reshape(-1,2,k))+.03
        return label,mix,mean,std

    @torch.no_grad()
    def initialize(self,x,length,label,minimum,maximum,seed):
        self.center.copy_(x.mean(0));self.scale.copy_(x.std(0).clamp(min=.1))
        bias=self.net[-1].bias;k=self.components
        rates=torch.bincount(label,minlength=2).float()+.5;rates/=rates.sum();bias[:2]=rates.log()
        for y in [0,1]:
            values=np.log1p(length[label.eq(y)].cpu().numpy().clip(minimum,maximum))[:,None]
            g=GaussianMixture(n_components=k,reg_covar=.0025,n_init=3,random_state=seed).fit(values)
            bias[2+y*k:2+(y+1)*k]=torch.as_tensor(np.log(g.weights_),device=x.device)
            bias[2+2*k+y*k:2+2*k+(y+1)*k]=torch.as_tensor(g.means_[:,0],device=x.device)
            s=np.sqrt(g.covariances_[:,0,0]);raw=np.log(np.expm1(np.maximum(s-.03,.001)))
            bias[2+4*k+y*k:2+4*k+(y+1)*k]=torch.as_tensor(raw,device=x.device)

    def nll(self,x,length,label,minimum,maximum):
        logits,mix,mu,sd=self(x);idx=torch.arange(len(x),device=x.device)
        mix,mu,sd=mix[idx,label],mu[idx,label].double(),sd[idx,label].double()
        n=length.clamp(minimum,maximum).double()
        lo=torch.log1p(n-.5)[:,None];hi=torch.log1p(n+.5)[:,None]
        lower=torch.special.ndtr((lo-mu)/sd);upper=torch.special.ndtr((hi-mu)/sd)
        lower=torch.where(n[:,None].le(minimum),torch.zeros_like(lower),lower)
        upper=torch.where(n[:,None].ge(maximum),torch.ones_like(upper),upper)
        mass=(upper-lower).clamp(min=1e-14)
        return F.cross_entropy(logits,label,reduction='none')-torch.logsumexp(mix.log_softmax(-1)+mass.log(),-1)

    @torch.no_grad()
    def sample(self,x,minimum,maximum):
        logits,mix,mu,sd=self(x);i=torch.arange(len(x),device=x.device)
        label=torch.multinomial(logits.softmax(-1),1).squeeze(1)
        component=torch.multinomial(mix[i,label].softmax(-1),1).squeeze(1)
        log_length=mu[i,label,component]+sd[i,label,component]*torch.randn(len(x),device=x.device)
        length=torch.expm1(log_length.clamp(max=math.log1p(maximum)+2)).round().long().clamp(minimum,maximum)
        return length,label


def fit_horizon_hazard(meta):
    m=episode_metadata(meta)
    risk=m[m.previous_label.eq(0)&m.ever_fraud.eq(0)].copy()
    risk['band']=np.minimum(9,(10*risk.event_index/risk.planned_length).astype(int))
    rates=[];counts=[]
    for band in range(10):
        g=risk[risk.band.eq(band)];exposure=1/g.planned_length.to_numpy();y=g.label.to_numpy()
        def objective(rate):
            z=rate*exposure
            return float((np.where(y==1,-np.log(-np.expm1(-z)),z)).sum()+1e-6*rate-.5*np.log(rate))
        fit=minimize_scalar(objective,bounds=(1e-6,100),method='bounded')
        assert fit.success
        rates.append(float(fit.x));counts.append(dict(band=band,events=len(g),onsets=int(y.sum()),
            normalized_exposure=float(exposure.sum()),rate=float(fit.x)))
    return dict(rates=rates,counts=counts,fit_split='optimization',likelihood='Bernoulli p=1-exp(-lambda_band/L)',
                weak_gamma_prior=dict(shape=1.5,rate=1e-6))


def horizon_probability(position,length,rates):
    if torch.is_tensor(position):
        band=(10*position/length).long().clamp(0,9)
        return -torch.expm1(-rates[band]/length)
    band=np.clip((10*np.asarray(position)/np.asarray(length)).astype(int),0,9)
    return -np.expm1(-np.asarray(rates)[band]/np.asarray(length))


@contextmanager
def preservation_generation(stats,old,outputs,episode_parameters,hazard_parameters,count_payload=None,use_hazard=False):
    import mostlyai.engine._tabular.generation as generation
    with episode_generation(stats,old,outputs,episode_parameters,'joint'):
        parent=generation.SequentialModel;codec=PastState(stats);lk=codec.prefixes['event_is_fraud']+'__cat'
        class PreservationModel(parent):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                self.horizon_rates=torch.tensor(hazard_parameters['rates'],dtype=torch.float64,device=self.device)
                self.joint_count=None
                if count_payload:
                    payload=torch.load(count_payload,map_location=self.device,weights_only=True)
                    devices=[self.device.index or 0] if self.device.type=='cuda' else []
                    with torch.random.fork_rng(devices=devices):
                        head=JointCount(payload['dim'],payload['components'],payload['conditioned']).to(self.device)
                    head.load_state_dict(payload['state_dict']);head.eval().requires_grad_(False)
                    object.__setattr__(self,'joint_count',head)
                self.predictors.register_forward_hook(self.replace_onset)

            def replace_onset(self,module,args,output):
                if args[1]!=lk:return output
                f=self._state_features[:,0];present=f[:,0]>.5
                p=output.softmax(-1)[:,0,codec.codes['1']].double()
                length=decode_length(self.length_tokens,stats['seq_len']['min'])
                if use_hazard:
                    risk=present&f[:,1].le(.5)&self.episode_memory[:,0].eq(0)
                    q=horizon_probability(self.position.double(),length.double(),self.horizon_rates)
                    p=torch.where(risk,q,p)
                if self.first_joint_label is not None:
                    p=torch.where(~present,self.first_joint_label.double(),p)
                logits=torch.full_like(output,-1e9)
                logits[:,0,codec.codes['1']]=p.log().to(output.dtype)
                logits[:,0,codec.codes['0']]=torch.log1p(-p).to(output.dtype)
                return logits

            def forward(self,x,mode,**kwargs):
                state=kwargs.get('history_state');self.first_joint_label=None
                self.position=torch.zeros(kwargs['batch_size'],device=self.device) if state is None else state[2][0,:,0]
                if state is None and self.joint_count is not None:
                    c=torch.cat(kwargs['context'][0],-1)
                    length,label=self.joint_count.sample(c,stats['seq_len']['min'],stats['seq_len']['max'])
                    self.first_joint_label=label
                    encoded=encode_slen_sidx_sdec(pd.Series(length.cpu().numpy()),stats['seq_len']['max'],SLEN_SUB_COLUMN_PREFIX)
                    fixed=dict(kwargs.get('fixed_values',{}))
                    fixed.update({k:torch.as_tensor(encoded[k].to_numpy(),device=self.device).reshape(-1,1) for k in encoded})
                    kwargs['fixed_values']=fixed
                return super().forward(x,mode,**kwargs)
        generation.SequentialModel=PreservationModel
        try:yield
        finally:generation.SequentialModel=parent
