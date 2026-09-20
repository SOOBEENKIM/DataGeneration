"""The original U forward path with the same four-scalar calibration as E."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from experiments.cs_saf_replication import ReplicationU
from models.cs_saf_calibration import CalibratedCSSAF


class CalibratedU(ReplicationU):
    def __init__(self, support, **kwargs):
        super().__init__('CS2-U1', support, **kwargs)
        self.register_buffer('calibration_offset', torch.zeros(2))
        self.register_buffer('calibration_slope', torch.ones(2))
        self.requires_grad_(False)

    set_calibration = CalibratedCSSAF.set_calibration

    def copy_logits(self, context, gap, *, zero_gap=False, static_codes=None):
        z = super().copy_logits(context, gap, zero_gap=zero_gap, static_codes=static_codes)
        slots = self.slots(static_codes, context.shape[:-1])
        return self.calibration_offset[slots] + self.calibration_slope[slots] * z

    @torch.no_grad()
    def response_curves(self, context, previous, *, zero_gap=False, static_codes=None):
        if context.ndim != 2:
            raise ValueError('response audit requires flattened histories')
        slots = self.slots(static_codes, context.shape[:-1])
        a = self.calibration_offset[slots, None]
        b = self.calibration_slope[slots, None]
        if zero_gap:
            # As in ReplicationU, evaluate a constant once before broadcasting.
            q = (a + b * self.copy_base(context)).sigmoid()
        else:
            # Preserve CSSAFv2's exact raw-logit arithmetic. Inverting sigmoid
            # would introduce saturation/rounding and is not an equal adapter.
            gaps = torch.tensor(self.support.representatives, device=context.device,
                                dtype=context.dtype)
            embedding = self.gap_route(self._support_code(gaps))
            logits = self.copy_base(context).expand(-1, len(gaps)).clone()
            for slot in (0, 1):
                selected = slots == slot
                if selected.any():
                    u = torch.tanh(F.linear(context[selected], self.route_context_weight[slot],
                        self.route_context_bias[slot])) * self.route_interaction_weight[slot]
                    v = torch.tanh(F.linear(embedding, self.route_gap_weight[slot]))
                    logits[selected] += u @ v.T / math.sqrt(16)
            q = (a + b * logits).sigmoid()
        fresh = self.new_mark_head(context).clone()
        fresh[:, :3] = -torch.inf
        fp = fresh.softmax(-1).gather(1, previous[:, None])
        shape = (len(context), len(self.support.representatives))
        return q.expand(shape), (q + (1-q) * fp).expand(shape)

    def architecture_contract(self):
        return dict(super().architecture_contract(),
            implementation_version='cs-saf-u-calibration-control-v1',
            calibration='positive_affine_copy_logit_per_observed_context',
            fitted_scalars=4, base_weights_frozen=True, added_history_parameters=0,
            calibration_offset=self.calibration_offset.tolist(),
            calibration_slope=self.calibration_slope.tolist())
