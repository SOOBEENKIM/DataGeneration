"""Ordinary fit-bin/empirical-value amount control on the unchanged D backbone."""
import numpy as np
import torch
from torch import nn
from models.cs_saf_external_controls import ExternalControls


def fit_amount_pool(amount, codec, positive_bins=32, extra_quantiles=(.99,.999)):
    raw = np.asarray(amount, dtype=np.float64)
    if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError('finite nonnegative fit amounts required')
    z = codec.encode(raw)
    positive = z[raw > 0]
    if not len(positive):
        raise ValueError('positive fit amounts required')
    q = np.unique(np.r_[np.arange(1,positive_bins)/positive_bins, extra_quantiles])
    edges = np.unique(np.quantile(positive,q).astype(np.float32))
    edges = edges[(edges >= positive.min()) & (edges < positive.max())]
    while True:
        codes = np.searchsorted(edges,positive,side='left')
        counts = np.bincount(codes,minlength=len(edges)+1)
        empty = counts[:-1] == 0
        if not empty.any():
            break
        edges = edges[~empty]
    zero = bool((raw == 0).any())
    codes = np.searchsorted(edges,z,side='left') + int(zero)
    if zero:
        codes[raw == 0] = 0
    counts = np.bincount(codes,minlength=len(edges)+1+int(zero))
    assert (counts > 0).all() and counts.sum() == len(raw)
    order = np.argsort(codes,kind='stable')
    values = raw[order]
    starts = np.cumsum(np.r_[0,counts[:-1]])
    means = np.array([np.log1p(values[s:s+n]).mean() for s,n in zip(starts,counts)])
    return dict(edges=edges, counts=counts, starts=starts, values=values,
                mean_log1p=means, zero_bin=np.array(zero), codec_mean=np.array(codec.mean),
                codec_scale=np.array(codec.scale), quantiles=q)


def pool_metadata(pool):
    return dict(fit_events=len(pool['values']), bins=len(pool['counts']),
        zero_bin=bool(pool['zero_bin']), counts=pool['counts'].tolist(),
        edges_codec=pool['edges'].tolist(),
        edges_raw=np.expm1(pool['edges'].astype(float)*float(pool['codec_scale'])+float(pool['codec_mean'])).tolist(),
        minimum=float(pool['values'].min()), maximum=float(pool['values'].max()),
        requested_positive_quantiles=pool['quantiles'].tolist())


def pool_tail_matrix(pool, thresholds):
    cuts = np.asarray(thresholds,float)
    return np.stack([(pool['values'][s:s+n,None]>cuts[None,:]).mean(0)
                     for s,n in zip(pool['starts'],pool['counts'])])


class BinnedAmountD(ExternalControls):
    def __init__(self, support, amount_state, pool, **kwargs):
        super().__init__('D',support,amount_kind='hurdle_lognormal3',full_gap=True,
                         amount_state=amount_state,**kwargs)
        self.amount_kind = 'binned_empirical'
        del self.quadrature_nodes, self.quadrature_weights
        self.value_head = nn.Linear(self.value_head.in_features,len(pool['counts']))
        with torch.no_grad():
            self.value_head.weight.zero_()
            self.value_head.bias.copy_(torch.as_tensor(np.log(pool['counts']/pool['counts'].sum()),dtype=torch.float32))
        self.register_buffer('amount_edges',torch.as_tensor(pool['edges'],dtype=torch.float32))
        # Reconstructed from the fit-only, hashed pool artifact when reloading.
        for name,value in [('pool_values',pool['values']),('pool_starts',pool['starts']),
                           ('pool_counts',pool['counts']),('pool_log_means',pool['mean_log1p'])]:
            self.register_buffer(name,torch.as_tensor(value),persistent=False)
        self.has_zero_bin = bool(pool['zero_bin'])
        self.pool_description = pool_metadata(pool)

    def initialize_from_direct(self, initial):
        own = self.state_dict()
        shared = {k:v for k,v in initial.items() if not k.startswith('value_head.')
                  and k in own and v.shape == own[k].shape}
        expected = set(own)-{'amount_edges','value_head.weight','value_head.bias'}
        assert set(shared) == expected, expected-set(shared)
        self.load_state_dict(shared,strict=False)
        return sorted(shared)

    def amount_parameters(self, h, gap, mark):
        x = self._event_core_features(h,gap,mark,torch.zeros_like(gap))[:,:-1]
        return self.value_head(x).log_softmax(-1)

    def amount_codes(self, value):
        code = torch.bucketize(value.contiguous(),self.amount_edges) + int(self.has_zero_bin)
        if self.has_zero_bin:
            zero = value.new_tensor(-self.amount_state['codec_mean']/self.amount_state['codec_scale'])
            code = torch.where(value.eq(zero),torch.zeros_like(code),code)
        return code

    def amount_nll(self, value, parameters):
        return -parameters.gather(1,self.amount_codes(value)[:,None]).squeeze(1)

    def amount_point(self, parameters):
        mean = parameters.double().exp() @ self.pool_log_means
        s = self.amount_state
        return ((mean-s['codec_mean'])/s['codec_scale']).to(parameters.dtype)

    def sample_amount(self, h, gap, mark):
        logp = self.amount_parameters(h,gap,mark)
        code = torch.distributions.Categorical(logits=logp).sample()
        offset = (torch.rand(len(code),device=code.device,dtype=torch.float64)*self.pool_counts[code]).long()
        raw = self.pool_values[self.pool_starts[code]+offset]
        s = self.amount_state
        z = ((torch.log1p(raw)-s['codec_mean'])/s['codec_scale']).float()
        return z,raw

    def architecture_contract(self):
        result = super().architecture_contract()
        result.update(version='external-binned-amount-v1',pool=self.pool_description,
                      current_auxiliary_used_for_amount=False,whole_model_joint_training=True)
        return result
