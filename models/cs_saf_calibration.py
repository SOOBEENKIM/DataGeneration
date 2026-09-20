"""Frozen CS-SAF copy-logit calibration; observed repeat targets only."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch

from models.cs_saf_v3 import CSSAFv3

VERSION = 'cs-saf-frozen-copy-calibration-v1'
CALIBRATION_BUFFERS = ('calibration_offset', 'calibration_slope')


class CalibratedCSSAF(CSSAFv3):
    def __init__(self, support, reference_probabilities, **kwargs):
        super().__init__('CS3-E1', support, reference_probabilities, **kwargs)
        self.register_buffer('calibration_offset', torch.zeros(2))
        self.register_buffer('calibration_slope', torch.ones(2))
        self.requires_grad_(False)

    def logit_grid(self, context, static_codes, *, zero_gap=False):
        z = super().logit_grid(context, static_codes, zero_gap=zero_gap)
        slots = self.slots(static_codes, context.shape[:-1])
        return self.calibration_offset[slots, None] + self.calibration_slope[slots, None] * z

    @torch.no_grad()
    def set_calibration(self, parameters):
        a = torch.as_tensor(parameters['offset'], device=self.calibration_offset.device,
                            dtype=self.calibration_offset.dtype)
        b = torch.as_tensor(parameters['slope'], device=self.calibration_slope.device,
                            dtype=self.calibration_slope.dtype)
        if a.shape != (2,) or b.shape != (2,) or not torch.isfinite(a).all() or not torch.isfinite(b).all() or (b <= 0).any():
            raise ValueError('two finite offsets and two positive slopes required')
        self.calibration_offset.copy_(a)
        self.calibration_slope.copy_(b)

    def architecture_contract(self):
        return dict(super().architecture_contract(), implementation_version=VERSION,
                    calibration='positive_affine_copy_logit_per_observed_context',
                    fitted_scalars=4, base_weights_frozen=True,
                    calibration_offset=self.calibration_offset.tolist(),
                    calibration_slope=self.calibration_slope.tolist())


def repeat_nll_and_gradient(theta, z, fresh_previous, equality):
    """Stable observable repeat likelihood, including fresh-channel coincidences."""
    t = theta[0] + theta[1] * z
    log_q = -np.logaddexp(0., -t)
    log_one_minus_q = -np.logaddexp(0., t)
    log_repeat = np.logaddexp(log_q, log_one_minus_q + np.log(fresh_previous))
    log_nonrepeat = log_one_minus_q + np.log1p(-fresh_previous)
    loss = -np.where(equality, log_repeat, log_nonrepeat).mean()
    # q/R is bounded; computing it in log space avoids underflow for small q.
    derivative = expit(t) - equality * np.exp(log_q - log_repeat)
    gradient = np.array([derivative.mean(), (derivative * z).mean()])
    return float(loss), gradient


def fit_repeat_calibration(features, optimizer):
    """Pure train-feature fit; this interface accepts no validation/oracle data."""
    required = ('logit', 'fresh_previous', 'equality', 'code')
    arrays = {k: np.asarray(features[k]) for k in required}
    if any(v.ndim != 1 or v.shape != arrays['logit'].shape for v in arrays.values()):
        raise ValueError('aligned one-dimensional training features required')
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('nonfinite training features')
    if set(np.unique(arrays['code'])) != {3, 4}:
        raise ValueError('both observed contexts required')
    if not np.isin(arrays['equality'], [0, 1]).all():
        raise ValueError('observable equality must be binary')
    if not ((arrays['fresh_previous'] > 0) & (arrays['fresh_previous'] < 1)).all():
        raise ValueError('fresh mark probability must be strictly between zero and one')
    groups = {}; offset = []; slope = []
    for code in (3, 4):
        mask = arrays['code'] == code
        args = tuple(np.asarray(arrays[k][mask], dtype=np.float64)
                     for k in ('logit', 'fresh_previous', 'equality'))
        initial = np.asarray(optimizer['initial'], dtype=float)
        before, _ = repeat_nll_and_gradient(initial, *args)
        result = minimize(repeat_nll_and_gradient, initial, args=args, jac=True,
                          method=optimizer['method'], bounds=optimizer['bounds'],
                          options=optimizer['options'])
        after, gradient = repeat_nll_and_gradient(result.x, *args)
        if not result.success or not np.isfinite(result.x).all() or after > before + 1e-10:
            raise RuntimeError(f'calibration did not converge monotonically: {result.message}')
        bounds = np.asarray(optimizer['bounds'])
        groups[str(code-3)] = dict(transitions=int(mask.sum()), before_nll=before,
            after_nll=after, offset=float(result.x[0]), slope=float(result.x[1]),
            success=bool(result.success), message=str(result.message), iterations=int(result.nit),
            function_evaluations=int(result.nfev), gradient=gradient.tolist(),
            bound_hit=bool(np.any(np.minimum(abs(result.x-bounds[:,0]), abs(result.x-bounds[:,1])) < 1e-6)))
        offset.append(float(result.x[0])); slope.append(float(result.x[1]))
    return dict(offset=offset, slope=slope, contexts=groups, fit_split='train',
                target='observed_mark_equality_including_fresh_repeat_mass',
                base_training_data_reused=True, oracle_targets_used=False,
                active_mask_used=False, test_accessed=False)
