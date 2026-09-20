"""Matched observed-repeat heads; old U and all historical implementations unchanged."""
import math
import torch
import torch.nn.functional as F
from models.cs_saf_v2 import CSSAFv2


class _MarginalLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, score):
        rho = b.sigmoid()
        if not (torch.isfinite(score).all() and torch.isfinite(b).all() and
                (a > 0).all() and ((rho > 0) & (rho < 1)).all()):
            raise FloatingPointError('non-interior or nonfinite coupling inputs')
        low = b - score.max(-1).values - 2
        high = b - score.min(-1).values + 2
        # Double precision root; no unrolled root-finder graph.
        for _ in range(48):
            mid = (low + high) / 2
            below = (a * (score + mid[..., None]).sigmoid()).sum(-1) < rho
            low = torch.where(below, mid, low)
            high = torch.where(below, high, mid)
        logits = score + ((low + high) / 2)[..., None]
        ctx.save_for_backward(a, logits.sigmoid(), rho)
        return logits

    @staticmethod
    def backward(ctx, grad):
        a, r, rho = ctx.saved_tensors
        variance = r * (1-r)
        denominator = (a * variance).sum(-1, keepdim=True)
        if (denominator <= torch.finfo(denominator.dtype).tiny).any():
            raise FloatingPointError('degenerate implicit derivative')
        common = grad.sum(-1, keepdim=True) / denominator
        return (-common*r, common.squeeze(-1)*rho*(1-rho),
                grad-common*a*variance)


def constrained_repeat_logits(gap_logits, base_logit, score):
    return _MarginalLogits.apply(gap_logits.double().softmax(-1),
        base_logit.double(), score.double()).to(score.dtype)


class ObservedRepeatStructure(CSSAFv2):
    def __init__(self, mode, support, **kwargs):
        if mode not in ('G', 'C'):
            raise ValueError('registered modes are G and C')
        super().__init__('CS2-U1', support, **kwargs)
        self.structure_mode = mode

    def relation_scores(self, context, static_codes):
        slots = self.slots(static_codes, context.shape[:-1]).reshape(-1)
        flat = context.reshape(-1, context.shape[-1])
        codes = torch.arange(3, 3+len(self.support.representatives), device=context.device)
        embedding = self.gap_route(codes)
        scores = context.new_zeros((len(flat), len(codes)))
        for group in (0, 1):
            ix = slots == group
            if ix.any():
                u = torch.tanh(F.linear(flat[ix], self.route_context_weight[group], self.route_context_bias[group]))
                v = torch.tanh(F.linear(embedding, self.route_gap_weight[group]))
                scores[ix] = (u*self.route_interaction_weight[group]) @ v.T / math.sqrt(16)
        return scores.reshape(*context.shape[:-1], len(codes))

    def repeat_logits(self, context, static_codes):
        score = self.relation_scores(context, static_codes)
        base = self.copy_base(context).squeeze(-1)
        if self.structure_mode == 'G':
            return base[..., None] + score
        gap_logits = self.gap_decoder.logits(context[..., :self.config.hidden_dim])
        return constrained_repeat_logits(gap_logits, base, score)

    def mark_distribution(self, context, gap, previous, has_previous, *, static_codes=None):
        logits = self.new_mark_head(context).clone()
        logits[..., :3] = -torch.inf
        first = F.log_softmax(logits, dim=-1)
        indicator = F.one_hot(previous, self.config.receiver_vocab_size).bool()
        nonrepeat = F.log_softmax(logits.masked_fill(indicator, -torch.inf), dim=-1)
        all_logits = self.repeat_logits(context, static_codes)
        code = (self._support_code(gap)-3).clamp_min(0)
        selected = all_logits.gather(-1, code[..., None]).squeeze(-1)
        lr, lnr = F.logsigmoid(selected), F.logsigmoid(-selected)
        distribution = torch.where(indicator, lr[..., None], lnr[..., None]+nonrepeat)
        return torch.where(has_previous[..., None], distribution, first), lr, lnr

    def architecture_contract(self):
        return dict(implementation_version='cs-saf-matched-observed-structure-v1',
            mode=self.structure_mode, parameters=sum(p.numel() for p in self.parameters()),
            history='same_strict_past_GRU128', observed_repeat=True,
            previous_mark_excluded_from_nonrepeat=True,
            fixed_history_marginal_constraint=self.structure_mode=='C',
            global_generated_marginals_guaranteed=False, auxiliary_or_rollout_loss=False,
            known_coupling_not_novel_by_itself=True)
