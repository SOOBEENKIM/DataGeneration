"""Registered explicit history coefficients and centered functional gap penalty."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.cs_saf_v2 import CSSAFv2
from models.cof_seqgen_saf import UNK_CODE

CANDIDATES = ('CS3-H1', 'CS3-E1', 'CS3-C1', 'CS3-R1')
VERSION = 'cs-saf-explicit-history-centered-residual-v3'


class CSSAFv3(CSSAFv2):
    def __init__(self, candidate, support, reference_probabilities, **kwargs):
        if candidate not in CANDIDATES:
            raise ValueError('unregistered v3 candidate')
        super().__init__('CS2-U1', support, **kwargs)
        self.cs_candidate_id = candidate
        self.history_interaction_weight = nn.Parameter(torch.zeros(2, 16))
        pi = torch.as_tensor(reference_probabilities, dtype=torch.float64).clone()
        if (pi.shape != (2, len(support.representatives)) or not torch.isfinite(pi).all()
                or (pi < 0).any() or not torch.allclose(pi.sum(1), torch.ones(2, dtype=pi.dtype), atol=1e-12, rtol=0)):
            raise ValueError('two normalized train reference measures required')
        self.register_buffer('reference_probabilities', pi)
        self.regularization_coefficient = .01 if candidate == 'CS3-R1' else 0.
        self.penalty_epsilon = 1e-4

    def components(self, context, static_codes):
        """Flatten internally; no detached means, so centering gradients are exact."""
        shape = context.shape[:-1]
        slots = self.slots(static_codes, shape).reshape(-1)
        c = context.reshape(-1, context.shape[-1])
        reps = torch.tensor(self.support.representatives, device=c.device, dtype=c.dtype)
        embeddings = self.gap_route(self._support_code(reps))
        history = torch.zeros(len(c), device=c.device, dtype=c.dtype)
        raw = torch.zeros((len(c), len(reps)), device=c.device, dtype=c.dtype)
        for s in (0, 1):
            ix = slots == s
            if ix.any():
                u = torch.tanh(F.linear(c[ix], self.route_context_weight[s], self.route_context_bias[s]))
                history[ix] = (u*self.history_interaction_weight[s]).sum(-1)/4
                if self.cs_candidate_id != 'CS3-H1':
                    v = torch.tanh(F.linear(embeddings, self.route_gap_weight[s]))
                    raw[ix] = (u*self.route_interaction_weight[s])@v.T/4
        pi = self.reference_probabilities[slots].to(dtype=c.dtype)
        mean = (raw*pi).sum(-1, keepdim=True)
        delta = raw-mean
        return history.reshape(shape), raw.reshape(*shape, -1), delta.reshape(*shape, -1)

    def logit_grid(self, context, static_codes, *, zero_gap=False):
        h, raw, delta = self.components(context, static_codes)
        baseline = self.copy_base(context).squeeze(-1)+h
        if zero_gap or self.cs_candidate_id == 'CS3-H1':
            return baseline[..., None].expand_as(raw)
        return baseline[..., None]+(raw if self.cs_candidate_id == 'CS3-E1' else delta)

    def copy_logits(self, context, gap, *, zero_gap=False, static_codes=None):
        if gap.shape != context.shape[:-1]:
            raise ValueError('gap/context shape mismatch')
        grid = self.logit_grid(context, static_codes, zero_gap=zero_gap)
        # First/missing gaps have no copy target; a finite placeholder is masked later.
        bins = (self._support_code(gap)-3).clamp_min(0)
        return grid.gather(-1, bins[..., None]).squeeze(-1)

    @torch.no_grad()
    def response_curves(self, context, previous, *, zero_gap=False, static_codes=None):
        if context.ndim != 2:
            raise ValueError('response audit requires flattened histories')
        q = self.logit_grid(context, static_codes, zero_gap=zero_gap).sigmoid()
        fresh = self.new_mark_head(context).clone(); fresh[:, :3] = -torch.inf
        p = fresh.softmax(-1).gather(1, previous[:, None])
        return q, q+(1-q)*p

    def residual_penalty(self, context, static_codes):
        _, _, delta = self.components(context, static_codes)
        pi = self.reference_probabilities[self.slots(static_codes, context.shape[:-1])].to(delta.dtype)
        variance = (delta.square()*pi).sum(-1)
        # Rationalized sqrt(x+eps^2)-eps: exact zero at x=0, no cancellation.
        return variance/((variance+self.penalty_epsilon**2).sqrt()+self.penalty_epsilon)

    def loss_terms(self, *, gap, receiver, numeric_value, valid_mask, static=None,
                   static_categorical=(), auxiliary_categorical=(), auxiliary_numeric=None,
                   target_mask=None):
        if gap.shape[1] > 32 or (target_mask is not None and not torch.equal(target_mask, valid_mask)):
            raise ValueError('one full controlled sequence per example required')
        if (receiver[valid_mask] < 3).any():
            raise ValueError('reserved mark in controlled target')
        hidden = self.encoder(gap, receiver, numeric_value, valid_mask, static,
                              static_categorical, auxiliary_categorical, auxiliary_numeric)
        context = self.context(hidden, static_categorical)
        codes = static_categorical[0][:, None].expand_as(valid_mask)
        has_previous = torch.zeros_like(valid_mask)
        has_previous[:, 1:] = valid_mask[:, 1:] & valid_mask[:, :-1]
        previous = torch.full_like(receiver, UNK_CODE); previous[:, 1:] = receiver[:, :-1]
        log_mark, log_repeat, log_nonrepeat = self.mark_distribution(context, gap, previous, has_previous, static_codes=codes)
        per_mark = -log_mark.gather(-1, receiver.unsqueeze(-1)).squeeze(-1)
        per_gap = self.gap_decoder.nll(hidden, (self._support_code(gap)-3).clamp_min(0))
        location, scale = self._value_parameters(hidden, gap, receiver)
        per_value = scale.log()+.5*math.log(2*math.pi)+.5*((torch.nan_to_num(numeric_value)-location)/scale).square()
        gap_mask, value_mask = valid_mask & torch.isfinite(gap), valid_mask & torch.isfinite(numeric_value)
        repeat = -torch.where(receiver == previous, log_repeat, log_nonrepeat)
        penalty = self.residual_penalty(context, codes).masked_fill(~has_previous, 0)
        return {'gap_sum':per_gap[gap_mask].sum(), 'gap_count':gap_mask.sum(),
                'mark_sum':per_mark[valid_mask].sum(), 'mark_count':valid_mask.sum(),
                'value_sum':per_value[value_mask].sum(), 'value_count':value_mask.sum(),
                'repeat_per_entity':repeat.masked_fill(~has_previous,0).sum(1),
                'static_code':static_categorical[0], 'residual_per_entity':penalty.sum(1)}

    def objective(self, terms, *, train_entities, transition_counts_by_code):
        base = sum(terms[f'{n}_sum']/terms[f'{n}_count'].clamp_min(1) for n in ('gap','mark','value'))
        if set(transition_counts_by_code) != {3,4} or min(transition_counts_by_code.values()) <= 0:
            raise ValueError('positive fixed train counts required')
        penalty = terms['residual_per_entity'].mean()*(train_entities/sum(transition_counts_by_code.values()))
        return base+self.regularization_coefficient*penalty, base, penalty

    def architecture_contract(self):
        return {'implementation_version':VERSION, 'candidate':self.cs_candidate_id,
                'history':'strict_past_GRU_plus_observed_context_rank16_history_head',
                'history_parameters_added':32, 'shared_history_residual_features':True,
                'centered':self.cs_candidate_id in ('CS3-C1','CS3-R1'),
                'current_gap_enabled':self.cs_candidate_id != 'CS3-H1',
                'regularization_coefficient':self.regularization_coefficient,
                'penalty':'train_reference_smooth_RMS', 'epsilon':self.penalty_epsilon,
                'known_active_label_used':False, 'oracle_training_targets':False,
                'dormant_parameter_count':1056 if self.cs_candidate_id=='CS3-H1' else 0}
