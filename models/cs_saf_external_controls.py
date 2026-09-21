"""Bounded external output-head ablations; the v1 implementation is unchanged."""
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.cs_saf_external import ExternalUG


class ExternalControls(ExternalUG):
    def __init__(self, mode, support, *, amount_kind, full_gap, amount_state, **kwargs):
        super().__init__('G' if mode == 'D' else mode, support, **kwargs)
        self.control_mode = mode
        self.amount_kind = amount_kind
        self.full_gap = full_gap
        self.amount_state = amount_state
        dim = self.config.hidden_dim + 8
        gd = self.config.gap_embedding_dim
        vocab = self.config.receiver_vocab_size
        if full_gap:
            self.fresh_gap = nn.Linear(gd, vocab, bias=False)
            nn.init.zeros_(self.fresh_gap.weight)
        if mode == 'D':
            assert full_gap
            head_modules = [self.copy_base, self.route_context, self.route_gap,
                            self.new_mark_head, self.fresh_gap]
            target = sum(p.numel() for m in head_modules for p in m.parameters()) + self.interaction.numel()
            input_dim = dim + gd + self.config.mark_embedding_dim + 1
            width = max(1, round((target - vocab) / (input_dim + 1 + vocab)))
            self.direct = nn.Sequential(nn.Linear(input_dim, width), nn.Tanh(), nn.Linear(width, vocab))
            self.direct_width = width
            self.matched_action_target = target
            for name in ('copy_base', 'route_context', 'route_gap', 'new_mark_head', 'fresh_gap', 'interaction'):
                delattr(self, name)
        if amount_kind == 'hurdle_lognormal3':
            self.value_head = nn.Linear(self.value_head.in_features, 10)
            # Three fixed components; global fit-only starting locations, no search.
            with torch.no_grad():
                self.value_head.weight.zero_()
                self.value_head.bias.zero_()
                self.value_head.bias[3:6] = torch.tensor(amount_state['initial_locations'])
                self.value_head.bias[6:9] = math.log(math.expm1(.5))
                p = max(amount_state['zero_rate'], 1e-6)
                self.value_head.bias[9] = math.log(p / (1 - p))
            nodes, weights = np.polynomial.hermite.hermgauss(16)
            self.register_buffer('quadrature_nodes', torch.tensor(nodes, dtype=torch.float32))
            self.register_buffer('quadrature_weights', torch.tensor(weights / math.sqrt(math.pi), dtype=torch.float32))
        elif amount_kind != 'legacy':
            raise ValueError(amount_kind)

    def initialize_shared(self, reference):
        own = self.state_dict()
        shared = {k: v for k, v in reference.items() if k in own and own[k].shape == v.shape}
        self.load_state_dict(shared, strict=False)
        return sorted(shared)

    def mark_log_probabilities(self, context, gap, previous, has_previous):
        embedding = self.gap_route(self._support_code(gap))
        if self.control_mode == 'D':
            prev = self.encoder.history_mark(previous) * has_previous[:, None]
            x = torch.cat((context, embedding, prev, has_previous[:, None].to(context.dtype)), -1)
            logits = self.direct(x)
            logits = logits.masked_fill(torch.arange(logits.shape[-1], device=logits.device)[None, :] == 0, -torch.inf)
            logp = logits.log_softmax(-1)
        else:
            logits = self.new_mark_head(context)
            if self.full_gap:
                logits = logits + self.fresh_gap(embedding)
            logits = logits.masked_fill(torch.arange(logits.shape[-1], device=logits.device)[None, :] == 0, -torch.inf)
            fresh = logits.log_softmax(-1)
            u = torch.tanh(self.route_context(context))
            v = torch.tanh(self.route_gap(embedding))
            r = self.copy_base(context).squeeze(-1) + (u * v * self.interaction).sum(-1) / math.sqrt(self.rank)
            same = F.one_hot(previous, self.config.receiver_vocab_size).bool()
            yes, no = F.logsigmoid(r), F.logsigmoid(-r)
            if self.control_mode == 'U':
                other = no[:, None] + fresh
                logp = torch.where(same, torch.logaddexp(other, yes[:, None]), other)
            else:
                other = logits.masked_fill(same, -torch.inf).log_softmax(-1)
                logp = torch.where(same, yes[:, None], no[:, None] + other)
            logp = torch.where(has_previous[:, None], logp, fresh)
        if self.calibration is not None:
            same = F.one_hot(previous, self.config.receiver_vocab_size).bool()
            edges = torch.as_tensor(self.calibration['edges'], device=gap.device, dtype=gap.dtype)
            code = torch.bucketize(torch.nan_to_num(gap), edges) + 1
            code = torch.where(gap.eq(0), torch.zeros_like(code), code)
            beta = torch.as_tensor(self.calibration['beta'], device=gap.device, dtype=gap.dtype)[code]
            y = logp.gather(1, previous[:, None]).squeeze(1)
            n = logp.masked_fill(same, -torch.inf).logsumexp(-1)
            shifted = y - n + beta
            corrected = torch.where(same, F.logsigmoid(shifted)[:, None],
                                    logp - n[:, None] + F.logsigmoid(-shifted)[:, None])
            logp = torch.where(has_previous[:, None], corrected, logp)
        return logp

    def amount_parameters(self, h, gap, mark):
        features = self._event_core_features(h, gap, mark, torch.zeros_like(gap))[:, :-1]
        out = self.value_head(features)
        weights = out[:, :3].log_softmax(-1)
        locations = out[:, 3:6]
        scales = F.softplus(out[:, 6:9]).clamp(.03, 3.)
        return weights, locations, scales, out[:, 9]

    def amount_nll(self, value, parameters):
        weights, loc, scale, zero_logit = parameters
        s = self.amount_state
        # The model's history representation stays the original signed-log1p codec.
        zero_code = value.new_tensor(-s['codec_mean'] / s['codec_scale'])
        is_zero = value.eq(zero_code)
        u = value.double() * s['codec_scale'] + s['codec_mean']
        safe_u = torch.where(is_zero, torch.ones_like(u), u)
        if torch.any(safe_u <= 0):
            raise ValueError('positive amount target decoded outside its support')
        loga = safe_u + torch.log(-torch.expm1(-safe_u))
        x = ((loga - s['log_mean']) / s['log_scale']).to(value.dtype)
        normal = -.5 * ((x[:, None] - loc) / scale).square() - scale.log() - .5 * math.log(2 * math.pi)
        # Density in the SAME standardized signed-log1p coordinate as legacy.
        jacobian = math.log(s['codec_scale']) + safe_u - loga
        pos = -(weights + normal).logsumexp(-1) + math.log(s['log_scale']) - jacobian.to(value.dtype)
        if s['zero_rate'] > 0:
            pos = pos + F.softplus(zero_logit)
            return torch.where(is_zero, F.softplus(-zero_logit), pos)
        if is_zero.any():
            raise ValueError('unseen zero has no atom in fit-only positive amount model')
        return pos

    def amount_point(self, parameters):
        weights, loc, scale, zero_logit = parameters
        s = self.amount_state
        normal = loc[:, :, None] + math.sqrt(2) * scale[:, :, None] * self.quadrature_nodes
        loga = normal * s['log_scale'] + s['log_mean']
        log1p_mean = (F.softplus(loga) * self.quadrature_weights).sum(-1)
        log1p_mean = (weights.exp() * log1p_mean).sum(-1)
        if s['zero_rate'] > 0:
            log1p_mean = log1p_mean * torch.sigmoid(-zero_logit)
        return (log1p_mean - s['codec_mean']) / s['codec_scale']

    def target_outputs(self, **batch):
        pos = batch['target_position']
        h = self.encoder(**{k: v for k, v in batch.items() if k != 'target_position'})
        row = torch.arange(len(pos), device=pos.device)
        h = h[row, pos]
        gap, mark, value = (batch[k][row, pos] for k in ('gap', 'receiver', 'numeric_value'))
        has = pos > 0
        prev = torch.where(has, batch['receiver'][row, (pos - 1).clamp_min(0)], torch.ones_like(pos))
        context = self.context(h, batch['static'], batch['static_categorical'])
        logmark = self.mark_log_probabilities(context, gap, prev, has)
        if self.amount_kind == 'legacy':
            mu, sd = self._value_parameters(h, gap, mark)
            nll = sd.log() + .5 * math.log(2 * math.pi) + .5 * ((value - mu) / sd).square()
        else:
            parameters = self.amount_parameters(h, gap, mark)
            nll = self.amount_nll(value, parameters)
            mu, sd = self.amount_point(parameters), torch.ones_like(value)
        core = self._event_core_features(h, gap, mark, value)
        auxlogits = []
        for head in self.auxiliary_categorical_heads:
            logits = head(core)
            logits[:, 0] = -torch.inf
            auxlogits.append(logits.log_softmax(-1))
        return dict(hidden=h, gap=gap, mark=mark, numeric=value, previous=prev, has_previous=has,
                    logmark=logmark, location=mu, scale=sd, amount_nll=nll,
                    auxiliary=tuple(v[row, pos] for v in batch['auxiliary_categorical']), auxlogits=tuple(auxlogits))

    def terms(self, **batch):
        o = self.target_outputs(**batch)
        mask = torch.isfinite(o['gap'])
        gap = self.gap_decoder.nll(o['hidden'], (self._support_code(o['gap']) - 3).clamp_min(0))
        mark = -o['logmark'].gather(1, o['mark'][:, None]).squeeze(1)
        terms = {'gap': (gap[mask].sum(), mask.sum()), 'mark': (mark.sum(), mark.new_tensor(len(mark))),
                 'amount': (o['amount_nll'].sum(), mark.new_tensor(len(mark)))}
        for i, (logp, target) in enumerate(zip(o['auxlogits'], o['auxiliary'])):
            nll = -logp.gather(1, target[:, None]).squeeze(1)
            terms[f'aux_{i}'] = (nll.sum(), nll.new_tensor(len(nll)))
        return terms, o

    def sample_amount(self, h, gap, mark):
        if self.amount_kind == 'legacy':
            mu, sd = self._value_parameters(h, gap, mark)
            z = mu + sd * torch.randn_like(mu)
            u = z.double() * self.amount_state['codec_scale'] + self.amount_state['codec_mean']
            return z, u.sign() * torch.expm1(u.abs())
        weights, loc, scale, zero_logit = self.amount_parameters(h, gap, mark)
        component = torch.distributions.Categorical(logits=weights).sample()[:, None]
        normal = loc.gather(1, component).squeeze(1) + scale.gather(1, component).squeeze(1) * torch.randn_like(gap)
        loga = normal.double() * self.amount_state['log_scale'] + self.amount_state['log_mean']
        raw = loga.exp()
        log1p = F.softplus(loga)
        if self.amount_state['zero_rate'] > 0:
            zero = torch.rand_like(gap) < zero_logit.sigmoid()
            raw = torch.where(zero, torch.zeros_like(raw), raw)
            log1p = torch.where(zero, torch.zeros_like(log1p), log1p)
        if not torch.isfinite(raw).all():
            raise FloatingPointError('nonfinite generated amount; no clipping or retry')
        return ((log1p - self.amount_state['codec_mean']) / self.amount_state['codec_scale']).float(), raw

    @torch.no_grad()
    def sample_fixed_lengths(self, lengths, *, static=None, static_categorical=(), device=None):
        if not lengths or min(lengths) < 1:
            raise ValueError('positive lengths required')
        dev = device or next(self.parameters()).device
        lens = torch.as_tensor(lengths, device=dev)
        b, t = len(lengths), max(lengths)
        valid = torch.arange(t, device=dev)[None, :] < lens[:, None]
        gap = torch.full((b, t), float('nan'), device=dev)
        mark = torch.zeros((b, t), dtype=torch.long, device=dev)
        value = torch.zeros((b, t), device=dev)
        raw_amount = torch.zeros((b, t), device=dev, dtype=torch.float64)
        auxiliary = tuple(torch.zeros_like(mark) for _ in self.config.auxiliary_categorical_vocab_sizes)
        auxnum = torch.zeros(b, t, 0, device=dev)
        static = static.to(dev) if static is not None else torch.zeros(b, 0, device=dev)
        static_categorical = tuple(v.to(dev) for v in static_categorical)
        for step in range(t):
            rows = valid[:, step].nonzero().squeeze(1)
            start = max(0, step - self.config.context_window + 1)
            mg = mark[rows, start:step + 1].clone(); mg[:, -1] = 1
            ag = tuple(v[rows, start:step + 1].clone() for v in auxiliary)
            for v in ag:
                v[:, -1] = 1
            h = self.encoder(gap[rows, start:step + 1], mg, value[rows, start:step + 1],
                             valid[rows, start:step + 1], static[rows],
                             tuple(v[rows] for v in static_categorical), ag, auxnum[rows, start:step + 1])[:, -1]
            if step:
                gap[rows, step] = self.support.decode_tensor(self.gap_decoder.sample(h))
            current = gap[rows, step]
            prev = mark[rows, step - 1] if step else torch.ones(len(rows), dtype=torch.long, device=dev)
            context = self.context(h, static[rows], tuple(v[rows] for v in static_categorical))
            logp = self.mark_log_probabilities(context, current, prev, torch.full_like(prev, step > 0, dtype=torch.bool))
            sampled = torch.distributions.Categorical(logits=logp).sample()
            mark[rows, step] = sampled
            numeric, raw = self.sample_amount(h, current, sampled)
            value[rows, step], raw_amount[rows, step] = numeric, raw
            core = self._event_core_features(h, current, sampled, numeric)
            for head, output in zip(self.auxiliary_categorical_heads, auxiliary):
                logits = head(core); logits[:, 0] = -torch.inf
                output[rows, step] = torch.distributions.Categorical(logits=logits).sample()
        return dict(gap=gap, receiver=mark, numeric_value=value, raw_amount=raw_amount,
                    auxiliary_categorical=auxiliary, auxiliary_numeric=auxnum, valid_mask=valid, lengths=lens)

    def architecture_contract(self):
        return dict(version='external-controls-v1', mode=self.control_mode, amount=self.amount_kind,
                    full_gap_to_fresh=self.full_gap, parameters=sum(p.numel() for p in self.parameters()),
                    context_window=self.config.context_window, direct_width=getattr(self, 'direct_width', None),
                    action_parameter_target=getattr(self, 'matched_action_target', None),
                    no_extra_training_objective=True, amount_state=self.amount_state)
