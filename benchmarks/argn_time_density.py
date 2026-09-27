"""A single conditional, quantized log-mixture for transaction intervals.

Personal timing uses a strict-past median clock. All history travels inside the
native recurrent state, so reordering and early sequence termination are safe.
"""
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from benchmarks.argn_past_state import PastState, DURATION_CAP
from benchmarks.argn_phase_gap import encode_values
from benchmarks.argn_onset_output import onset_output_generation

WINDOW = 20
MIN_HISTORY = 5
ARMS = ('history_only', 'history_clock', 'history_relative')


def clock_features(metadata, fallback):
    """Vectorized training equivalent of the recurrent strict-past clock."""
    gap = metadata.gap.clip(upper=DURATION_CAP).where(metadata.event_index.gt(0))
    median = gap.groupby(metadata.entity_id, sort=False).transform(
        lambda x: x.shift().rolling(WINDOW, min_periods=MIN_HISTORY).median())
    valid = median.notna() & median.gt(0)
    return np.log1p(median.where(valid, fallback).to_numpy()), valid.to_numpy(dtype=np.float32)


class PastClock:
    """count of past events, followed by a ring of their non-first gaps."""
    def __init__(self, fallback):
        assert fallback > 0
        self.fallback = float(fallback)

    def empty(self, n):
        m = np.full((n, WINDOW + 1), np.nan, dtype=np.float64)
        m[:, 0] = 0
        return m

    def features(self, memory):
        observed = np.isfinite(memory[:, 1:]).sum(axis=1)
        valid = observed >= MIN_HISTORY
        median = np.full(len(memory), self.fallback)
        if valid.any():
            median[valid] = np.nanmedian(memory[valid, 1:], axis=1)
        valid &= median > 0
        median = np.where(valid, median, self.fallback)
        return np.log1p(median), valid.astype(np.float32)

    def advance(self, memory, gap):
        result = memory.copy()
        count = memory[:, 0].astype(np.int64)
        has_previous = count > 0
        rows = np.flatnonzero(has_previous)
        result[rows, 1 + (count[rows] - 1) % WINDOW] = np.clip(np.asarray(gap)[rows], 0, DURATION_CAP)
        result[:, 0] += 1
        return result


def log_interval_mass(a, b):
    """log(Phi(b)-Phi(a)), stable on either side of the normal tail."""
    log_left, log_right = torch.special.log_ndtr(a), torch.special.log_ndtr(b)
    cdf = log_right + torch.log(-torch.expm1((log_left-log_right).clamp(max=-1e-15)))
    log_survival_left, log_survival_right = torch.special.log_ndtr(-a), torch.special.log_ndtr(-b)
    survival = log_survival_left + torch.log(-torch.expm1((log_survival_right-log_survival_left).clamp(max=-1e-15)))
    return torch.where(a > 0, survival, cdf)


class TimeDensity(nn.Module):
    def __init__(self, dim, arm, components=3, hidden=64):
        super().__init__()
        assert arm in ARMS
        self.dim, self.arm, self.components, self.hidden = dim, arm, components, hidden
        self.register_buffer('center', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))
        self.register_buffer('clock_center', torch.tensor(0.))
        self.register_buffer('clock_scale', torch.tensor(1.))
        self.register_buffer('prior_logits', torch.zeros(4, components))
        self.register_buffer('prior_mean', torch.zeros(4, components))
        self.register_buffer('prior_raw_scale', torch.zeros(4, components))
        self.net = nn.Sequential(nn.Linear(dim+6, hidden), nn.SiLU(), nn.Linear(hidden, 3*components))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base, phase, clock, valid):
        x = (base-self.center)/self.scale
        clock_input = torch.stack([(clock-self.clock_center)/self.clock_scale, valid], -1)
        if self.arm == 'history_only':
            clock_input = torch.zeros_like(clock_input)
        features = torch.cat([x, F.one_hot(phase, 4).to(x.dtype), clock_input], -1)
        logits, location, raw_scale = self.net(features).chunk(3, -1)
        logits = logits + self.prior_logits[phase]
        location = location + self.prior_mean[phase]
        if self.arm == 'history_relative':
            location = location + clock.unsqueeze(-1)
        scale = (F.softplus(raw_scale+self.prior_raw_scale[phase])+.05).clamp(max=5.)
        return logits, location, scale

    def nll(self, base, phase, clock, valid, gap, limits, quantum):
        logits, location, scale = self(base, phase, clock, valid)
        location, scale = location.double(), scale.double()
        y = gap.double().clamp(*limits)
        a = (torch.log1p(y).unsqueeze(-1)-location)/scale
        b = (torch.log1p(y+quantum).unsqueeze(-1)-location)/scale
        log_mass = log_interval_mass(a, b)
        log_mass = torch.where(y.unsqueeze(-1) <= limits[0], torch.special.log_ndtr(b), log_mass)
        log_mass = torch.where(y.unsqueeze(-1) >= limits[1], torch.special.log_ndtr(-a), log_mass)
        return -torch.logsumexp(logits.double().log_softmax(-1)+log_mass, -1)

    @torch.no_grad()
    def sample(self, base, phase, clock, valid, rng, limits, quantum):
        logits, location, scale = self(base, phase, clock, valid)
        probs = logits.softmax(-1).cpu().numpy()
        component = (rng.random((len(base), 1)) > np.cumsum(probs, axis=1)).sum(axis=1).clip(max=self.components-1)
        rows = np.arange(len(base))
        z = rng.normal(location.cpu().numpy()[rows, component], scale.cpu().numpy()[rows, component])
        lo, hi = np.log1p(limits)
        clipped = int(((z < lo) | (z > hi)).sum())
        value = np.expm1(np.clip(z, lo, hi))
        value = np.floor(value/quantum+1e-8)*quantum
        return value, clipped


@contextmanager
def time_density_generation(*args, time_payload, **kwargs):
    import mostlyai.engine._tabular.generation as generation
    stats = args[0]
    codec = PastState(stats)
    audit = []
    with onset_output_generation(*args, **kwargs):
        parent = generation.SequentialModel

        class TimeModel(parent):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                payload = torch.load(time_payload, map_location=self.device, weights_only=True)
                devices = [self.device.index or 0] if self.device.type == 'cuda' else []
                with torch.random.fork_rng(devices=devices):
                    density = TimeDensity(payload['dim'], payload['arm'], payload['components'], payload['hidden']).to(self.device)
                density.load_state_dict(payload['state_dict'])
                density.eval().requires_grad_(False)
                object.__setattr__(self, 'time_density', density)
                self.clock_codec = PastClock(payload['fallback'])
                self.time_limits = payload['limits']
                self.time_quantum = payload['quantum']
                self.time_keys = payload['keys']
                self.time_rng = np.random.default_rng(torch.initial_seed())
                self.time_audit = dict(sampled_values=0, clipped_values=0)
                audit.append(self.time_audit)
                self.regressors.register_forward_pre_hook(self.capture_time_base)
                self.predictors.register_forward_hook(self.replace_time)

            def capture_time_base(self, module, a):
                if a[1] == self.time_keys[0]:
                    self.time_base = torch.cat(a[0][:2], -1)

            def replace_time(self, module, a, output):
                key = a[1]
                if key not in self.time_keys:
                    return output
                if key == self.time_keys[0]:
                    f = self._state_features
                    self.time_mask = f[..., 0].gt(.5) & f[..., 2].le(.5)
                    phase = (2*f[..., 1].gt(.5).long()+self._amount_labels).reshape(-1)
                    clock = torch.as_tensor(self.current_clock[0], device=self.device, dtype=torch.float32)
                    valid = torch.as_tensor(self.current_clock[1], device=self.device)
                    mask = self.time_mask.reshape(-1)
                    values = np.full(len(phase), self.time_limits[0], dtype=np.float64)
                    chosen, clipped = self.time_density.sample(self.time_base.reshape(len(phase), -1)[mask], phase[mask],
                        clock[mask], valid[mask], self.time_rng, self.time_limits, self.time_quantum)
                    values[mask.cpu().numpy()] = chosen
                    self.time_audit['sampled_values'] += len(chosen)
                    self.time_audit['clipped_values'] += clipped
                    self.time_tokens = encode_values(values, stats['columns']['gap'], self.time_keys, self.tgt_cardinalities)
                token = torch.as_tensor(self.time_tokens[key], device=self.device).reshape(output.shape[:-1])
                logits = torch.full_like(output, -torch.inf)
                logits.scatter_(-1, token.unsqueeze(-1), 0.)
                return torch.where(self.time_mask.unsqueeze(-1), logits, output)

            def forward(self, x, mode, **kwargs):
                assert mode == 'gen'
                state = kwargs.get('history_state')
                if state is None:
                    memory = self.clock_codec.empty(kwargs['batch_size'])
                else:
                    assert len(state) == 5
                    memory = state[4][0].detach().cpu().numpy()
                    kwargs['history_state'] = state[:4]
                self.current_clock = self.clock_codec.features(memory)
                output, history, recurrent = super().forward(x, mode, **kwargs)
                event = {k:v.detach().cpu().numpy() for k,v in output.items() if k.startswith(codec.prefixes['gap']+'__')}
                memory = self.clock_codec.advance(memory, codec.numeric(event, 'gap'))
                packed = torch.as_tensor(memory, dtype=torch.float64, device=self.device).unsqueeze(0)
                return output, history, (*recurrent, packed)

        generation.SequentialModel = TimeModel
        try:
            yield audit
        finally:
            generation.SequentialModel = parent
