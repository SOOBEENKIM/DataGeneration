"""One bounded, train-centered coarse-gap correction to frozen calibrated models."""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
import torch.nn.functional as F
from models.cs_saf_calibration import CalibratedCSSAF
from models.cs_saf_calibration_u import CalibratedU

NEW_BUFFERS=('gap_bin_map','gap_bin_weights','gap_bin_delta')


class GapCorrection:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.register_buffer('gap_bin_map',torch.zeros((2,len(self.support.representatives)),dtype=torch.long))
        self.register_buffer('gap_bin_weights',torch.full((2,5),.2,dtype=torch.float64))
        self.register_buffer('gap_bin_delta',torch.zeros((2,5)))
        self.requires_grad_(False)

    @torch.no_grad()
    def set_gap_correction(self,mapping,weights,delta):
        m=torch.as_tensor(mapping,device=self.gap_bin_map.device,dtype=torch.long)
        w=torch.as_tensor(weights,device=self.gap_bin_weights.device,dtype=torch.float64)
        d=torch.as_tensor(delta,device=self.gap_bin_delta.device,dtype=self.gap_bin_delta.dtype)
        if m.shape!=self.gap_bin_map.shape or m.min()<0 or m.max()>4:raise ValueError('invalid coarse-bin map')
        if w.shape!=(2,5) or (w<=0).any() or not torch.allclose(w.sum(1),torch.ones(2,device=w.device,dtype=w.dtype),atol=1e-12,rtol=0):raise ValueError('invalid train frequencies')
        if d.shape!=(2,5) or not torch.isfinite(d).all() or d.abs().max()>.5000001:raise ValueError('invalid bounded offsets')
        if (w*d.double()).sum(1).abs().max()>1e-7:raise ValueError('logit offsets must be train centered')
        self.gap_bin_map.copy_(m);self.gap_bin_weights.copy_(w);self.gap_bin_delta.copy_(d)

    def architecture_contract(self):
        return dict(super().architecture_contract(),implementation_version='cs-saf-gap-calibration-v1',
            old_affine_frozen=True,new_effective_degrees_of_freedom=8,new_stored_offsets=10,
            offset_bound=.5,lambda_ridge=.001,gap_bin_map=self.gap_bin_map.tolist(),
            gap_bin_weights=self.gap_bin_weights.tolist(),gap_bin_delta=self.gap_bin_delta.tolist(),
            no_continuous_subbin_information=True)


class GapCalibratedE(GapCorrection,CalibratedCSSAF):
    def logit_grid(self,context,static_codes,*,zero_gap=False):
        z=super().logit_grid(context,static_codes,zero_gap=zero_gap)
        if zero_gap:return z
        slots=self.slots(static_codes,context.shape[:-1])
        return z+self.gap_bin_delta[slots[...,None],self.gap_bin_map[slots]]


class GapCalibratedU(GapCorrection,CalibratedU):
    def copy_logits(self,context,gap,*,zero_gap=False,static_codes=None):
        z=super().copy_logits(context,gap,zero_gap=zero_gap,static_codes=static_codes)
        if zero_gap:return z
        slots=self.slots(static_codes,context.shape[:-1]);k=(self._support_code(gap)-3).clamp_min(0)
        return z+self.gap_bin_delta[slots,self.gap_bin_map[slots,k]]

    @torch.no_grad()
    def response_curves(self,context,previous,*,zero_gap=False,static_codes=None):
        if zero_gap:return super().response_curves(context,previous,zero_gap=True,static_codes=static_codes)
        if context.ndim!=2:raise ValueError('flatten histories for response grid')
        slots=self.slots(static_codes,context.shape[:-1])
        gaps=torch.tensor(self.support.representatives,device=context.device,dtype=context.dtype)
        embedding=self.gap_route(self._support_code(gaps))
        z=self.copy_base(context).expand(-1,len(gaps)).clone()
        for slot in (0,1):
            ix=slots==slot
            if ix.any():
                u=torch.tanh(F.linear(context[ix],self.route_context_weight[slot],self.route_context_bias[slot]))*self.route_interaction_weight[slot]
                v=torch.tanh(F.linear(embedding,self.route_gap_weight[slot]))
                z[ix]+=u@v.T/math.sqrt(16)
        q=(self.calibration_offset[slots,None]+self.calibration_slope[slots,None]*z+
           self.gap_bin_delta[slots[:,None],self.gap_bin_map[slots]]).sigmoid()
        fresh=self.new_mark_head(context).clone();fresh[:,:3]=-torch.inf
        fp=fresh.softmax(-1).gather(1,previous[:,None])
        return q,q+(1-q)*fp


def objective_gradient(delta,z,fp,y,bins,weights,ridge):
    t=z+delta[bins];lq=-np.logaddexp(0.,-t);l1q=-np.logaddexp(0.,t)
    lr=np.logaddexp(lq,l1q+np.log(fp));lnr=l1q+np.log1p(-fp)
    nll=float(-np.where(y,lr,lnr).mean())
    derivative=expit(t)-y*np.exp(lq-lr)
    grad=np.bincount(bins,weights=derivative,minlength=5)/len(z)+ridge*weights*delta
    penalty=.5*ridge*np.dot(weights,delta**2)
    return nll+float(penalty),grad


def fit_gap_offsets(features,weights,config):
    # This interface deliberately accepts no validation/oracle data or metrics.
    required=('logit','fresh_previous','equality','code','bin')
    f={k:np.asarray(features[k]) for k in required}
    if any(v.ndim!=1 or v.shape!=f['logit'].shape for v in f.values()):raise ValueError('unaligned train features')
    if not all(np.isfinite(v).all() for v in f.values()):raise ValueError('nonfinite train features')
    if not np.isin(f['equality'],[0,1]).all() or set(np.unique(f['code']))!={3,4}:raise ValueError('invalid observed labels')
    if not ((f['fresh_previous']>0)&(f['fresh_previous']<1)).all():raise ValueError('invalid fresh probability')
    if not np.isin(f['bin'],range(5)).all():raise ValueError('invalid bin')
    result=[];groups={};weights=np.asarray(weights,dtype=float)
    for slot,code in enumerate((3,4)):
        ix=f['code']==code;b=f['bin'][ix].astype(int);counts=np.bincount(b,minlength=5);w=counts/counts.sum()
        np.testing.assert_allclose(w,weights[slot],atol=1e-12,rtol=0)
        args=(f['logit'][ix].astype(float),f['fresh_previous'][ix].astype(float),f['equality'][ix].astype(float),b,w,config['lambda_ridge'])
        before,_=objective_gradient(np.zeros(5),*args)
        fit=minimize(objective_gradient,np.zeros(5),args=args,jac=True,method=config['optimizer']['method'],
            bounds=[(-config['delta_bound'],config['delta_bound'])]*5,
            constraints=[dict(type='eq',fun=lambda d:float(w@d),jac=lambda d:w)],options=config['optimizer']['options'])
        after,gradient=objective_gradient(fit.x,*args);center=float(w@fit.x)
        if not fit.success or not np.isfinite(fit.x).all() or abs(center)>1e-8 or abs(fit.x).max()>config['delta_bound']+1e-8 or after>before+1e-10:
            raise RuntimeError('constrained calibration invalid: '+str(fit.message))
        penalty=.5*config['lambda_ridge']*np.dot(w,fit.x**2)
        groups[str(slot)]=dict(counts=counts.tolist(),weights=w.tolist(),delta=fit.x.tolist(),
            before_nll=before,after_nll=float(after-penalty),penalty=float(penalty),after_objective=after,
            centered_logit_mean=center,gradient=gradient.tolist(),success=bool(fit.success),
            message=str(fit.message),iterations=int(fit.nit),function_evaluations=int(fit.nfev),
            bound_hits=np.flatnonzero(np.abs(fit.x)>=config['delta_bound']-1e-6).tolist())
        result.append(fit.x.tolist())
    return dict(delta=result,groups=groups,fit_split='train',old_affine_frozen=True,
                oracle_targets_used=False,validation_used_for_fit=False)
