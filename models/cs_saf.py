"""Frozen controlled-study CS-SAF candidates; no data loading or execution."""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from models.cof_seqgen_saf import (
    CoFSeqGenSAF, SAFModelConfig, HurdleLogNormalGapDecoder,
    LEARNED_START_CODE, UNK_CODE,
)

VERSION = "cs-saf-bilinear-observable-repeat-v1"
CANDIDATES = {
    "CS-C0": (False, False, False),
    "CS-U0": (True, False, False),
    "CS-U1": (True, True, False),
    "CS-B0": (True, False, True),
    "CS-B1": (True, True, True),
}


class CSSAF(CoFSeqGenSAF):
    def __init__(self, candidate_id, support, *, hidden_dim=128,
                 gap_embedding_dim=32, mark_embedding_dim=32, **tensorizer_kwargs):
        aligned, route, balanced = CANDIDATES[candidate_id]
        config = SAFModelConfig(
            candidate_id="SAF-U1" if aligned else "SAF-C0",
            hidden_dim=hidden_dim, gap_embedding_dim=gap_embedding_dim,
            mark_embedding_dim=mark_embedding_dim, context_window=32,
            max_gap=max(support.representatives), **tensorizer_kwargs)
        if (config.static_dim or len(config.static_categorical_vocab_sizes) != 1
                or config.auxiliary_categorical_vocab_sizes or config.auxiliary_numeric_dim):
            raise ValueError("CS-SAF v1 is restricted to the controlled one-label schema")
        super().__init__(config, support)
        self.cs_candidate_id = candidate_id
        self.route_enabled, self.balanced = route, balanced
        # Replace the inherited scalar gate and mark head; no unused gate tensors.
        del self.copy_base_logit_head, self.copy_gap_gate_head, self.copy_gap_delta_head
        self.encoder.static_categorical[0] = nn.Embedding(config.static_categorical_vocab_sizes[0], 8)
        self.encoder.static_projection = nn.Linear(8, hidden_dim)
        context_dim = hidden_dim+8
        self.copy_base = nn.Linear(context_dim, 1)
        self.context_projection = nn.Linear(context_dim, 32)
        self.gap_projection = nn.Linear(gap_embedding_dim, 32, bias=False)
        self.interaction_weight = nn.Parameter(torch.zeros(32))
        self.new_mark_head = nn.Linear(context_dim, config.receiver_vocab_size)

    def context(self, hidden, static_categorical):
        embedding = self.encoder.static_categorical[0](static_categorical[0])
        if hidden.ndim == 3:
            embedding = embedding[:, None, :].expand(-1, hidden.shape[1], -1)
        return torch.cat([hidden, embedding], dim=-1)

    def copy_logits(self, context, gap, *, zero_gap=False, static_codes=None):
        e = self.gap_route(self._support_code(gap))
        if zero_gap or not self.route_enabled:
            e = torch.zeros_like(e)
        interaction = (torch.tanh(self.context_projection(context))
                       * torch.tanh(self.gap_projection(e)) * self.interaction_weight).sum(-1)
        return self.copy_base(context).squeeze(-1)+interaction/math.sqrt(32)

    def mark_distribution(self, context, gap, previous, has_previous, *, static_codes=None):
        new_logits = self.new_mark_head(context)
        # Controlled data have a closed vocabulary: all reserved codes are excluded.
        new_logits = new_logits.clone()
        new_logits[..., :LEARNED_START_CODE] = -torch.inf
        log_new = F.log_softmax(new_logits, dim=-1)
        logit = self.copy_logits(context, gap, static_codes=static_codes)
        new_component = F.logsigmoid(-logit).unsqueeze(-1)+log_new
        indicator = F.one_hot(previous, self.config.receiver_vocab_size).bool()
        mixture = torch.where(indicator,
            torch.logaddexp(new_component, F.logsigmoid(logit).unsqueeze(-1)), new_component)
        log_mark = torch.where(has_previous.unsqueeze(-1), mixture, log_new)
        # Stable Bernoulli likelihood on observable equality; no latent copy targets.
        previous_log_new = log_new.gather(-1, previous.unsqueeze(-1)).squeeze(-1)
        log_repeat = torch.logaddexp(F.logsigmoid(logit), F.logsigmoid(-logit)+previous_log_new)
        log_nonrepeat = F.logsigmoid(-logit)+torch.log1p(-previous_log_new.exp().clamp(max=1-1e-7))
        return log_mark, log_repeat, log_nonrepeat

    def loss_terms(self, *, gap, receiver, numeric_value, valid_mask, static=None,
                   static_categorical=(), auxiliary_categorical=(), auxiliary_numeric=None,
                   target_mask=None):
        if gap.shape[1] > 32:
            raise ValueError("controlled v1 forbids multiwindow sequences")
        target = valid_mask if target_mask is None else target_mask
        if not torch.equal(target, valid_mask):
            raise ValueError("one full sequence per example is required by fixed denominators")
        if (receiver[valid_mask] < LEARNED_START_CODE).any():
            raise ValueError("reserved mark in controlled target")
        hidden = self.encoder(gap, receiver, numeric_value, valid_mask, static,
                              static_categorical, auxiliary_categorical, auxiliary_numeric)
        context = self.context(hidden, static_categorical)
        has_previous = torch.zeros_like(valid_mask)
        has_previous[:, 1:] = valid_mask[:, 1:] & valid_mask[:, :-1]
        previous = torch.full_like(receiver, UNK_CODE)
        previous[:, 1:] = receiver[:, :-1]
        log_mark, log_repeat, log_nonrepeat = self.mark_distribution(context, gap, previous, has_previous,
            static_codes=static_categorical[0][:, None].expand_as(valid_mask))
        per_mark = -log_mark.gather(-1, receiver.unsqueeze(-1)).squeeze(-1)
        if isinstance(self.gap_decoder, HurdleLogNormalGapDecoder):
            per_gap = self.gap_decoder.nll(hidden, gap)
        else:
            codes = (self._support_code(gap)-LEARNED_START_CODE).clamp_min(0)
            per_gap = self.gap_decoder.nll(hidden, codes)
        location, scale = self._value_parameters(hidden, gap, receiver)
        per_value = scale.log()+0.5*math.log(2*math.pi)+0.5*((torch.nan_to_num(numeric_value)-location)/scale).square()
        gap_mask = valid_mask & torch.isfinite(gap)
        value_mask = valid_mask & torch.isfinite(numeric_value)
        repeat_nll = -torch.where(receiver == previous, log_repeat, log_nonrepeat)
        return {
            "gap_sum": per_gap[gap_mask].sum(), "gap_count": gap_mask.sum(),
            "mark_sum": per_mark[valid_mask].sum(), "mark_count": valid_mask.sum(),
            "value_sum": per_value[value_mask].sum(), "value_count": value_mask.sum(),
            "repeat_per_entity": (repeat_nll.masked_fill(~has_previous, 0)).sum(1),
            "static_code": static_categorical[0],
        }

    def objective(self, terms, *, train_entities, transition_counts_by_code):
        base = sum(terms[f"{name}_sum"]/terms[f"{name}_count"].clamp_min(1)
                   for name in ("gap", "mark", "value"))
        weights = torch.zeros(self.config.static_categorical_vocab_sizes[0], device=base.device)
        if len(transition_counts_by_code) != 2 or min(transition_counts_by_code.values()) <= 0:
            raise ValueError("both context transition counts must be positive")
        for code, count in transition_counts_by_code.items():
            weights[int(code)] = train_entities/(2*count)
        # Entity-uniform batches: N/B * sum_i BCE_i/(2*T_label_i).
        auxiliary = (terms["repeat_per_entity"]*weights[terms["static_code"]]).mean()
        return base+(auxiliary if self.balanced else 0), base, auxiliary

    def compute_loss(self, *, train_entities, transition_counts_by_code, **inputs):
        loss, base, auxiliary = self.objective(self.loss_terms(**inputs),
            train_entities=train_entities, transition_counts_by_code=transition_counts_by_code)
        return {"loss": loss, "base_nll": base, "balanced_repeat_nll": auxiliary}

    def architecture_contract(self):
        return {"implementation_version": VERSION, "family": "CS-SAF",
                "candidate": self.cs_candidate_id, "history": "strictly_past_shifted_gru",
                "gap_to_mark_route": "rank_32_bilinear" if self.route_enabled else "matched_zero_gap",
                "direct_static_embedding_dim": 8, "balanced_observable_repeat_loss": self.balanced,
                "reserved_mark_outputs": "masked", "controlled_schema_only": True}

    @torch.no_grad()
    def response_curves(self, context, previous, *, zero_gap=False, static_codes=None):
        gaps = torch.tensor(self.support.representatives, device=context.device, dtype=context.dtype)
        e = self.gap_route(self._support_code(gaps))
        if zero_gap or not self.route_enabled:
            e = torch.zeros_like(e)
        u = torch.tanh(self.context_projection(context))*self.interaction_weight
        v = torch.tanh(self.gap_projection(e))
        logits = self.copy_base(context)+u@v.T/math.sqrt(32)
        copy = torch.sigmoid(logits)
        fresh = self.new_mark_head(context)
        fresh[:, :LEARNED_START_CODE] = -torch.inf
        pprevious = fresh.softmax(-1).gather(-1, previous[:, None])
        return copy, copy+(1-copy)*pprevious

    @torch.no_grad()
    def sample_fixed_lengths(self, lengths: Sequence[int], *, static_categorical, device=None):
        if not lengths or min(lengths) < 2 or max(lengths) > 32:
            raise ValueError("controlled generation lengths must be in [2,32]")
        device = device or next(self.parameters()).device
        codes = tuple(x.to(device) for x in static_categorical)
        lengths_t = torch.tensor(lengths, device=device)
        valid = torch.arange(max(lengths), device=device)[None, :] < lengths_t[:, None]
        gap = torch.full(valid.shape, float("nan"), device=device)
        receiver = torch.zeros(valid.shape, dtype=torch.long, device=device)
        value = torch.zeros(valid.shape, device=device)
        for t in range(valid.shape[1]):
            local_marks = receiver[:, :t+1].clone()
            local_marks[:, -1] = UNK_CODE
            hidden = self.encoder(gap[:, :t+1], local_marks, value[:, :t+1],
                                  valid[:, :t+1], static_categorical=codes)[:, -1]
            context = self.context(hidden, codes)
            active = valid[:, t]
            if t:
                emitted = self.gap_decoder.sample(hidden)
                if not isinstance(self.gap_decoder, HurdleLogNormalGapDecoder):
                    emitted = self.support.decode_tensor(emitted)
                gap[active, t] = emitted[active]
            previous = receiver[:, t-1] if t else torch.full_like(receiver[:, 0], UNK_CODE)
            logp, _, _ = self.mark_distribution(context, gap[:, t], previous, active & (t > 0),
                                                static_codes=codes[0])
            mark = torch.distributions.Categorical(logits=logp).sample()
            receiver[active, t] = mark[active]
            location, scale = self._value_parameters(hidden, gap[:, t], mark)
            sampled_value = location+scale*torch.randn_like(location)
            value[active, t] = sampled_value[active]
        return {"gap": gap, "receiver": receiver, "numeric_value": value,
                "valid_mask": valid, "lengths": lengths_t}
