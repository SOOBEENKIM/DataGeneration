"""Untrained candidate: couple gap and observable repetition at fixed marginals.

This is a known constrained logistic construction, not a novelty or performance
claim. It preserves only the supplied one-history gap/repetition marginals.
It cannot repair an incorrect supplied repetition marginal or guarantee global
trajectory marginals. No historical CS-SAF model imports this candidate.
"""
import torch


class _ConstrainedRepeat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gap_probability, repeat_probability, score):
        if score.shape != gap_probability.shape or repeat_probability.shape != score.shape[:-1]:
            raise ValueError('incompatible coupling shapes')
        if not (torch.isfinite(score).all() and torch.isfinite(gap_probability).all()):
            raise ValueError('nonfinite coupling inputs')
        if not ((gap_probability>0).all() and ((repeat_probability>0)&(repeat_probability<1)).all()):
            raise ValueError('strictly interior probabilities required')
        if not torch.allclose(gap_probability.sum(-1),torch.ones_like(repeat_probability),atol=1e-6,rtol=0):
            raise ValueError('gap probabilities must sum to one')
        center=torch.logit(repeat_probability)
        low=center-score.max(-1).values-2
        high=center-score.min(-1).values+2
        for _ in range(80):
            mid=(low+high)/2
            r=torch.sigmoid(score+mid[...,None])
            below=(gap_probability*r).sum(-1)<repeat_probability
            low=torch.where(below,mid,low); high=torch.where(below,high,mid)
        r=torch.sigmoid(score+((low+high)/2)[...,None])
        ctx.save_for_backward(gap_probability,r)
        return r

    @staticmethod
    def backward(ctx, grad):
        a,r=ctx.saved_tensors; v=r*(1-r)
        z=(a*v).sum(-1,keepdim=True)
        if (z<=torch.finfo(z.dtype).tiny).any():
            raise FloatingPointError('degenerate implicit derivative')
        common=(grad*v).sum(-1,keepdim=True)/z
        return -r*common, common.squeeze(-1), v*(grad-a*common)


def observable_gap_repeat_joint(gap_logits, repeat_logit, relation_score):
    """Return [..., gap, nonrepeat/repeat] joint probabilities.

The first-order implicit gradient includes both marginal branches. Relations
share no trainable parameters here; the candidate encoder/heads are not implemented
in this component. Their proposed contract is in external_audit_v1/research_decision.md.
"""
    a=gap_logits.softmax(-1); rho=repeat_logit.sigmoid()
    r=_ConstrainedRepeat.apply(a,rho,relation_score)
    return torch.stack((a*(1-r),a*r),-1)
