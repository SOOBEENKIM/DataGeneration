"""Vectorized equivalent of pinned DeepEcho 0.8.1 PAR loss (including its conventions).

Do not silently fix upstream normalization or continuous-target alignment. This
is an execution accelerator for the exact pinned baseline, not a new objective.
"""
import torch
import torch.nn.functional as F


def vectorized_par_loss(self,X_padded,Y_padded,seq_len):
    steps,n,_=X_padded.shape
    lengths=torch.as_tensor(seq_len,device=Y_padded.device).clamp(max=steps)
    times=torch.arange(steps,device=Y_padded.device)[:,None]
    mask=times<lengths[None,:]
    # Upstream uses the LAST length_i continuous targets, but FIRST length_i
    # targets for missing indicators, counts and categories. Preserve exactly.
    source=(steps-lengths[None,:]+times).clamp(max=steps-1)
    total=Y_padded.new_zeros(())
    for key,props in self._data_map.items():
        typ=props['type']
        if typ in ('continuous','timestamp'):
            mu_idx,sigma_idx,missing_idx=props['indices']
            mu=Y_padded[:,:,mu_idx];sigma=F.softplus(Y_padded[:,:,sigma_idx])
            target=X_padded[:,:,mu_idx].gather(0,source)
            value=torch.distributions.Normal(mu,sigma).log_prob(target)
            missing=F.logsigmoid(Y_padded[:,:,missing_idx]);true=X_padded[:,:,missing_idx]
            terms=value+true*missing+(1-true)*torch.log(1-torch.exp(missing))
        elif typ=='count':
            r_idx,p_idx,missing_idx=props['indices']
            r=F.softplus(Y_padded[:,:,r_idx])*props['range'];p=torch.sigmoid(Y_padded[:,:,p_idx])
            target=X_padded[:,:,r_idx]*props['range']
            value=torch.distributions.NegativeBinomial(r,p,validate_args=False).log_prob(target)
            missing=F.logsigmoid(Y_padded[:,:,missing_idx]);true=X_padded[:,:,missing_idx]
            terms=value+true*missing+(1-true)*torch.log(1-torch.exp(missing))
        elif typ in ('categorical','ordinal'):
            ix=list(props['indices'].values())
            logprobs=F.log_softmax(Y_padded[:,:,ix],dim=2)
            target=X_padded[:,:,ix].argmax(dim=2,keepdim=True)
            terms=logprobs.gather(2,target).squeeze(2)
        else:raise ValueError('unsupported pinned PAR type')
        total=total+terms.masked_select(mask).sum()
    return -total/(n*len(self._data_map)*n)


from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
PINNED_PAR_SOURCE_SHA256='f177f250aa91e1a24b4f1c64a2bd1acbbc6f287612c9687a350942e2a0f3a6a8'


@contextmanager
def equivalent_par_loss():
    from deepecho.models.par import PARModel
    if hashlib.sha256(Path(inspect.getfile(PARModel)).read_bytes()).hexdigest()!=PINNED_PAR_SOURCE_SHA256:
        raise RuntimeError('PAR source changed; equivalence proof no longer applies')
    original=PARModel._compute_loss
    PARModel._compute_loss=vectorized_par_loss
    try:yield
    finally:PARModel._compute_loss=original
