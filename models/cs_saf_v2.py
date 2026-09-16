"""Preregistered observed-context route banks; unchanged v1 common modules/loss."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from models.cs_saf import CSSAF

VERSION = "cs-saf-context-separated-bilinear-v2"
BANK_NAMES = ("route_context_weight", "route_context_bias",
              "route_gap_weight", "route_interaction_weight")


class CSSAFv2(CSSAF):
    def __init__(self, candidate_id, support, **kwargs):
        if candidate_id not in ("CS2-U1", "CS2-B1"):
            raise ValueError("only the two preregistered v2 pilot candidates are implemented")
        # Consume exactly the same global initialization stream as fresh v1.
        super().__init__(candidate_id.replace("CS2-", "CS-"), support, **kwargs)
        if (self.config.hidden_dim != 128 or self.config.gap_embedding_dim != 32
                or self.config.mark_embedding_dim != 32
                or self.config.static_categorical_vocab_sizes != (5,)):
            raise ValueError("v2 requires the preregistered dimensions and two known contexts")
        self.cs_candidate_id = candidate_id
        del self.context_projection, self.gap_projection, self.interaction_weight
        self.route_context_weight = nn.Parameter(torch.empty(2, 16, 136))
        self.route_context_bias = nn.Parameter(torch.empty(2, 16))
        self.route_gap_weight = nn.Parameter(torch.empty(2, 16, 32))
        self.route_interaction_weight = nn.Parameter(torch.zeros(2, 16))
        generator = torch.Generator(device="cpu").manual_seed(20261010)
        with torch.no_grad():
            self.route_context_weight.uniform_(-1/math.sqrt(136), 1/math.sqrt(136), generator=generator)
            self.route_context_bias.uniform_(-1/math.sqrt(136), 1/math.sqrt(136), generator=generator)
            self.route_gap_weight.uniform_(-1/math.sqrt(32), 1/math.sqrt(32), generator=generator)

    @staticmethod
    def slots(static_codes, shape):
        if static_codes is None or static_codes.shape != shape:
            raise ValueError("explicit static codes matching the context shape are required")
        if static_codes.dtype not in (torch.int32, torch.int64):
            raise ValueError("static codes must be integer tensors")
        if ((static_codes != 3) & (static_codes != 4)).any():
            raise ValueError("unknown/missing context is outside the controlled v2 contract")
        return static_codes.long()-3

    def context(self, hidden, static_categorical):
        if len(static_categorical) != 1:
            raise ValueError("one observed context is required")
        self.slots(static_categorical[0], hidden.shape[:1])
        return super().context(hidden, static_categorical)

    def copy_logits(self, context, gap, *, zero_gap=False, static_codes=None):
        slots = self.slots(static_codes, context.shape[:-1])
        if gap.shape != slots.shape:
            raise ValueError("gap and context shapes differ")
        embedding = self.gap_route(self._support_code(gap))
        if zero_gap:
            embedding = torch.zeros_like(embedding)
        interaction = torch.zeros(context.shape[:-1], device=context.device, dtype=context.dtype)
        for slot in (0, 1):
            selected = slots == slot
            if selected.any():
                u = torch.tanh(F.linear(context[selected], self.route_context_weight[slot],
                                        self.route_context_bias[slot]))
                v = torch.tanh(F.linear(embedding[selected], self.route_gap_weight[slot]))
                interaction[selected] = (u*v*self.route_interaction_weight[slot]).sum(-1)/math.sqrt(16)
        return self.copy_base(context).squeeze(-1)+interaction

    @torch.no_grad()
    def response_curves(self, context, previous, *, zero_gap=False, static_codes=None):
        if context.ndim != 2:
            raise ValueError("response audit requires flattened histories")
        slots = self.slots(static_codes, context.shape[:-1])
        gaps = torch.tensor(self.support.representatives, device=context.device, dtype=context.dtype)
        embedding = self.gap_route(self._support_code(gaps))
        if zero_gap:
            embedding = torch.zeros_like(embedding)
        logits = self.copy_base(context).expand(-1, len(gaps)).clone()
        for slot in (0, 1):
            selected = slots == slot
            if selected.any():
                u = torch.tanh(F.linear(context[selected], self.route_context_weight[slot],
                                        self.route_context_bias[slot]))*self.route_interaction_weight[slot]
                v = torch.tanh(F.linear(embedding, self.route_gap_weight[slot]))
                logits[selected] += u@v.T/math.sqrt(16)
        copy = logits.sigmoid()
        fresh = self.new_mark_head(context)
        fresh[:, :3] = -torch.inf
        previous_fresh = fresh.softmax(-1).gather(-1, previous[:, None])
        return copy, copy+(1-copy)*previous_fresh

    def architecture_contract(self):
        return {"implementation_version": VERSION, "family": "CS-SAF-v2",
                "candidate": self.cs_candidate_id, "history": "strictly_past_shifted_gru",
                "gap_to_mark_route": "observed_context_separated_rank_16_banks",
                "context_banks": 2, "rank_per_bank": 16, "direct_route_parameters": 5440,
                "direct_static_embedding_dim": 8, "balanced_observable_repeat_loss": self.balanced,
                "reserved_mark_outputs": "masked", "controlled_schema_only": True,
                "bank_initializer_seed": 20261010, "known_active_label_used": False,
                "shared_features_remain_shared": True}
