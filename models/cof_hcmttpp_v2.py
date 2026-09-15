"""Source-only CoF-HCMTTPP-v2 H1 model components.

Importing this module has no data, device, training, sampling, evaluation, or
artifact side effects.  The only candidate is H1; the stopped CCMTPP-v1
candidate chain is never reopened here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.cof_ccmtpp_v1 import (
    AMOUNT_CONTRACT_NAME,
    CausalEventDecoder,
    ConditionalAmountHead,
    FlatReceiverDecoder,
)


H1_CANDIDATE_ID = "H1"
PERMANENT_V1_CHAIN_STATE = "STOP_CCMTPP_V1_CHAIN_C1_GATE_FAIL"
H1_CENTRAL_BINS = 16
H1_MIN_STRICT_EXCEEDANCES = 128
H1_TAIL_FRACTION = 0.005


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(dtype=torch.float64, device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainOnlyH1TailState:
    """Y-specific strict-exceedance state fitted from positive train gaps."""

    n_positive: tuple[int, int]
    m_strict_exceedances: tuple[int, int]
    u_tail: tuple[float, float]
    p_tail_base: tuple[float, float]
    beta_base: tuple[float, float]
    ordered_positive_u_sha256: tuple[str, str]
    fit_split: str
    provenance: tuple[tuple[str, str], ...]
    state_sha256: str
    validation_rows_used: int = 0
    internal_test_rows_used: int = 0
    fraud_test_rows_used: int = 0

    def __post_init__(self) -> None:
        if self.fit_split != "train":
            raise ValueError("H1 tail state must be train-only")
        if any(
            count != 0
            for count in (
                self.validation_rows_used,
                self.internal_test_rows_used,
                self.fraud_test_rows_used,
            )
        ):
            raise ValueError("non-train rows cannot contribute to H1 tail state")
        if any(count < H1_MIN_STRICT_EXCEEDANCES for count in self.n_positive):
            raise ValueError("each Y class needs enough positive train gaps")
        if any(m < H1_MIN_STRICT_EXCEEDANCES for m in self.m_strict_exceedances):
            raise ValueError("strict-exceedance count is below the frozen minimum")
        if any(not math.isfinite(value) or value <= 0 for value in self.u_tail):
            raise ValueError("tail threshold must be finite and positive")
        if any(not 0 < value < 1 for value in self.p_tail_base):
            raise ValueError("baseline tail probability must be inside (0,1)")
        if any(not math.isfinite(value) or value <= 0 for value in self.beta_base):
            raise ValueError("baseline tail scale must be finite and positive")
        if len(self.state_sha256) != 64:
            raise ValueError("tail-state hash is invalid")


def fit_train_only_h1_tail_state(
    *,
    gap: Tensor,
    y: Tensor,
    valid_mask: Tensor,
    fit_split: str,
    provenance: Mapping[str, str],
) -> TrainOnlyH1TailState:
    """Fit the frozen strict-exceedance threshold and baseline tail state."""

    if fit_split != "train":
        raise ValueError("H1 tail-state fitting is train-only")
    if gap.ndim != 2 or valid_mask.shape != gap.shape or valid_mask.dtype != torch.bool:
        raise ValueError("gap and valid mask must have matching [batch,length] shape")
    if y.shape != (gap.shape[0],) or bool(((y < 0) | (y > 1)).any()):
        raise ValueError("Y must be one binary label per sequence")
    valid_gap = gap[valid_mask]
    if not torch.isfinite(valid_gap).all() or bool((valid_gap < 0).any()):
        raise ValueError("valid train gap must be finite and nonnegative")
    if not provenance or any(not isinstance(key, str) or not isinstance(value, str) for key, value in provenance.items()):
        raise ValueError("tail-state provenance must be a nonempty string mapping")

    expanded_y = y[:, None].expand_as(gap)
    n_positive: list[int] = []
    m_values: list[int] = []
    thresholds: list[float] = []
    probabilities: list[float] = []
    scales: list[float] = []
    input_hashes: list[str] = []
    for label in (0, 1):
        selected = valid_mask & (expanded_y == label) & (gap > 0)
        positive_u = torch.log1p(gap[selected]).to(dtype=torch.float64).sort().values
        n_y = int(positive_u.numel())
        m_y = max(H1_MIN_STRICT_EXCEEDANCES, math.ceil(H1_TAIL_FRACTION * n_y))
        if n_y <= m_y:
            raise ValueError("each Y class needs more positive gaps than strict exceedances")
        unique = torch.unique_consecutive(positive_u)
        threshold: Tensor | None = None
        for candidate in unique.flip(0):
            if int((positive_u > candidate).sum().item()) >= m_y:
                threshold = candidate
                break
        if threshold is None:
            raise ValueError("strict-exceedance threshold is unavailable")
        excess = positive_u[positive_u > threshold] - threshold
        beta = float(excess.mean().item())
        p_tail = float(excess.numel() / n_y)
        threshold_value = float(threshold.item())
        if not math.isfinite(beta) or beta <= 0 or not 0 < p_tail < 1:
            raise ValueError("strict-exceedance baseline is invalid")
        n_positive.append(n_y)
        m_values.append(m_y)
        thresholds.append(threshold_value)
        probabilities.append(p_tail)
        scales.append(beta)
        input_hashes.append(_ordered_tensor_sha256(positive_u))

    state_payload: dict[str, Any] = {
        "schema_version": "cof-hcmttpp-v2-h1-tail-state-v1",
        "n_positive": n_positive,
        "m_strict_exceedances": m_values,
        "u_tail": thresholds,
        "p_tail_base": probabilities,
        "beta_base": scales,
        "ordered_positive_u_sha256": input_hashes,
        "fit_split": "train",
        "provenance": sorted(provenance.items()),
        "validation_rows_used": 0,
        "internal_test_rows_used": 0,
        "fraud_test_rows_used": 0,
    }
    return TrainOnlyH1TailState(
        n_positive=tuple(n_positive),
        m_strict_exceedances=tuple(m_values),
        u_tail=tuple(thresholds),
        p_tail_base=tuple(probabilities),
        beta_base=tuple(scales),
        ordered_positive_u_sha256=tuple(input_hashes),
        fit_split="train",
        provenance=tuple(sorted(provenance.items())),
        state_sha256=_canonical_sha256(state_payload),
    )


@dataclass(frozen=True)
class H1GapParameters:
    zero_logits: Tensor
    height_fractions: Tensor
    derivatives: Tensor
    pi_tail: Tensor
    beta: Tensor
    u_tail: Tensor
    r_tail: Tensor
    r_beta: Tensor
    invalid_mask: Tensor


@dataclass(frozen=True)
class RQSTransformResult:
    value: Tensor
    log_abs_det: Tensor


@dataclass(frozen=True)
class H1SampledGap:
    gap: Tensor
    u: Tensor
    zero_route: Tensor
    tail_route: Tensor
    source: str = "cof_hcmttpp_v2_h1_hurdle_rqs"
    hard_upper_clip_count: int = 0
    redraw_count: int = 0


class InvalidH1GapStateError(RuntimeError):
    """Signals a fail-closed INVALID H1 density or sampling state."""


class H1HurdleRQSGapDecoder(nn.Module):
    """H1 exact-zero hurdle with conditional central RQS and tail."""

    def __init__(
        self,
        *,
        d_model: int,
        tail_state: TrainOnlyH1TailState,
        central_bins: int = H1_CENTRAL_BINS,
        minimum_height_fraction: float = 1.0e-4 / H1_CENTRAL_BINS,
        minimum_derivative: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if central_bins != H1_CENTRAL_BINS:
            raise ValueError("H1 central RQS must have exactly 16 bins")
        if not 0 < minimum_height_fraction * central_bins < 1:
            raise ValueError("minimum RQS height fraction is invalid")
        if minimum_derivative <= 0:
            raise ValueError("minimum RQS derivative must be positive")
        self.d_model = int(d_model)
        self.central_bins = int(central_bins)
        self.minimum_height_fraction = float(minimum_height_fraction)
        self.minimum_derivative = float(minimum_derivative)
        self.tail_state = tail_state
        self.zero_head = nn.Linear(d_model, 1)
        self.spline_head = nn.Linear(d_model, central_bins + central_bins + 1)
        self.tail_gate_residual = nn.Linear(d_model, 1)
        self.tail_scale_residual = nn.Linear(d_model, 1)
        nn.init.zeros_(self.tail_gate_residual.weight)
        nn.init.zeros_(self.tail_gate_residual.bias)
        nn.init.zeros_(self.tail_scale_residual.weight)
        nn.init.zeros_(self.tail_scale_residual.bias)
        self.register_buffer(
            "u_tail_by_y",
            torch.tensor(tail_state.u_tail, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "p_tail_base_by_y",
            torch.tensor(tail_state.p_tail_base, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "beta_base_by_y",
            torch.tensor(tail_state.beta_base, dtype=torch.float64),
            persistent=True,
        )

    def forward(self, hidden: Tensor, *, y: Tensor) -> H1GapParameters:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have [batch,length,d_model] shape")
        if y.shape != (hidden.shape[0],) or bool(((y < 0) | (y > 1)).any()):
            raise ValueError("Y must be one binary label per sequence")
        label = y.long()[:, None].expand(hidden.shape[:2])
        u_tail = self.u_tail_by_y.to(hidden)[label]
        p_base = self.p_tail_base_by_y.to(hidden)[label]
        beta_base = self.beta_base_by_y.to(hidden)[label]

        zero_logits = self.zero_head(hidden).squeeze(-1)
        raw_spline = self.spline_head(hidden)
        raw_height = raw_spline[..., : self.central_bins]
        raw_derivative = raw_spline[..., self.central_bins :]
        height_fractions = self.minimum_height_fraction + (
            1.0 - self.minimum_height_fraction * self.central_bins
        ) * F.softmax(raw_height, dim=-1)
        derivatives = self.minimum_derivative + F.softplus(raw_derivative)

        r_tail = self.tail_gate_residual(hidden).squeeze(-1)
        base_logit = torch.log(p_base) - torch.log1p(-p_base)
        raw_pi_tail = torch.sigmoid(base_logit + r_tail)
        # The algebraic baseline correction preserves the exact train-state
        # value at r=0 without removing the residual gradient.
        pi_tail = p_base + raw_pi_tail - torch.sigmoid(base_logit)
        r_beta = self.tail_scale_residual(hidden).squeeze(-1)
        beta = beta_base * torch.exp(r_beta)
        invalid_mask = (
            ~torch.isfinite(zero_logits)
            | ~torch.isfinite(height_fractions).all(dim=-1)
            | ~torch.isfinite(derivatives).all(dim=-1)
            | ~torch.isfinite(pi_tail)
            | (pi_tail <= 0)
            | (pi_tail >= 1)
            | ~torch.isfinite(beta)
            | (beta <= 0)
        )
        return H1GapParameters(
            zero_logits=zero_logits,
            height_fractions=height_fractions,
            derivatives=derivatives,
            pi_tail=pi_tail,
            beta=beta,
            u_tail=u_tail,
            r_tail=r_tail,
            r_beta=r_beta,
            invalid_mask=invalid_mask,
        )

    @staticmethod
    def _require_valid(parameters: H1GapParameters) -> None:
        if bool(parameters.invalid_mask.any()):
            raise InvalidH1GapStateError("H1 gap parameters are non-finite or invalid")

    def _spline_geometry(
        self, parameters: H1GapParameters
    ) -> tuple[Tensor, Tensor]:
        calculation_dtype = (
            torch.float64
            if parameters.height_fractions.dtype
            in (torch.float16, torch.bfloat16, torch.float32)
            else parameters.height_fractions.dtype
        )
        raw_heights = (
            parameters.height_fractions.to(dtype=calculation_dtype)
            * parameters.u_tail.to(dtype=calculation_dtype).unsqueeze(-1)
        )
        interior = torch.cumsum(raw_heights[..., :-1], dim=-1)
        cumulative = torch.cat(
            (
                torch.zeros_like(raw_heights[..., :1]),
                interior,
                parameters.u_tail.unsqueeze(-1),
            ),
            dim=-1,
        )
        heights = cumulative[..., 1:] - cumulative[..., :-1]
        if not torch.isfinite(cumulative).all() or bool((heights <= 0).any()):
            raise InvalidH1GapStateError("RQS geometry is non-finite or degenerate")
        return heights, cumulative

    @staticmethod
    def _gather(values: Tensor, index: Tensor) -> Tensor:
        return torch.gather(values, -1, index.unsqueeze(-1)).squeeze(-1)

    def _bin_values(
        self,
        *,
        index: Tensor,
        parameters: H1GapParameters,
        heights: Tensor,
        cumulative: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        y0 = self._gather(cumulative[..., :-1], index)
        height = self._gather(heights, index)
        derivative0 = self._gather(parameters.derivatives[..., :-1], index)
        derivative1 = self._gather(parameters.derivatives[..., 1:], index)
        derivative0 = derivative0.to(dtype=heights.dtype)
        derivative1 = derivative1.to(dtype=heights.dtype)
        delta = height * self.central_bins
        return y0, height, derivative0, derivative1, delta

    @staticmethod
    def _rqs_value_and_log_derivative(
        *,
        theta: Tensor,
        y0: Tensor,
        height: Tensor,
        derivative0: Tensor,
        derivative1: Tensor,
        delta: Tensor,
    ) -> tuple[Tensor, Tensor]:
        theta_one_minus = theta * (1.0 - theta)
        denominator = delta + (
            derivative1 + derivative0 - 2.0 * delta
        ) * theta_one_minus
        numerator = height * (
            delta * theta.square() + derivative0 * theta * (1.0 - theta)
        )
        value = y0 + numerator / denominator
        derivative_numerator = delta.square() * (
            derivative1 * theta.square()
            + 2.0 * delta * theta_one_minus
            + derivative0 * (1.0 - theta).square()
        )
        derivative = derivative_numerator / denominator.square()
        if not torch.isfinite(value).all() or not torch.isfinite(derivative).all() or bool((derivative <= 0).any()):
            raise InvalidH1GapStateError("RQS value or Jacobian is invalid")
        return value, torch.log(derivative)

    def central_quantile(
        self, q: Tensor, parameters: H1GapParameters
    ) -> RQSTransformResult:
        """Map conditional-central quantiles in ``(0,1)`` to log gaps."""

        self._require_valid(parameters)
        if q.shape != parameters.zero_logits.shape:
            raise ValueError("central quantile shape must match H1 parameters")
        if not torch.isfinite(q).all() or bool(((q <= 0) | (q >= 1)).any()):
            raise ValueError("central RQS quantile must be in the open interval (0,1)")
        heights, cumulative = self._spline_geometry(parameters)
        index = torch.floor(q * self.central_bins).long().clamp_max(self.central_bins - 1)
        y0, height, derivative0, derivative1, delta = self._bin_values(
            index=index,
            parameters=parameters,
            heights=heights,
            cumulative=cumulative,
        )
        q_calculation = q.to(dtype=heights.dtype)
        theta = q_calculation * self.central_bins - index.to(heights.dtype)
        value, log_derivative = self._rqs_value_and_log_derivative(
            theta=theta,
            y0=y0,
            height=height,
            derivative0=derivative0,
            derivative1=derivative1,
            delta=delta,
        )
        return RQSTransformResult(value=value, log_abs_det=log_derivative)

    def central_inverse(
        self, u: Tensor, parameters: H1GapParameters
    ) -> RQSTransformResult:
        """Invert the conditional-central RQS and return ``log|dq/du|``."""

        self._require_valid(parameters)
        if u.shape != parameters.zero_logits.shape:
            raise ValueError("central inverse shape must match H1 parameters")
        if not torch.isfinite(u).all() or bool(((u <= 0) | (u > parameters.u_tail)).any()):
            raise ValueError("central log gap must be in (0,u_tail]")
        heights, cumulative = self._spline_geometry(parameters)
        u_calculation = u.to(dtype=heights.dtype)
        index = (u_calculation.unsqueeze(-1) >= cumulative[..., 1:]).sum(dim=-1)
        index = index.long().clamp_max(self.central_bins - 1)
        y0, height, derivative0, derivative1, delta = self._bin_values(
            index=index,
            parameters=parameters,
            heights=heights,
            cumulative=cumulative,
        )
        y_delta = u_calculation - y0
        derivative_sum = derivative0 + derivative1 - 2.0 * delta
        a = y_delta * derivative_sum + height * (delta - derivative0)
        b = height * derivative0 - y_delta * derivative_sum
        c = -delta * y_delta
        discriminant = b.square() - 4.0 * a * c
        if not torch.isfinite(discriminant).all() or bool((discriminant < 0).any()):
            raise InvalidH1GapStateError("RQS inverse discriminant is invalid")
        denominator = -b - torch.sqrt(discriminant)
        quadratic_theta = (2.0 * c) / denominator
        theta = torch.where(
            y_delta == 0,
            torch.zeros_like(quadratic_theta),
            torch.where(
                y_delta == height,
                torch.ones_like(quadratic_theta),
                quadratic_theta,
            ),
        )
        if not torch.isfinite(theta).all() or bool(((theta < 0) | (theta > 1)).any()):
            raise InvalidH1GapStateError("RQS inverse root is invalid")
        recovered_u, log_du_dq = self._rqs_value_and_log_derivative(
            theta=theta,
            y0=y0,
            height=height,
            derivative0=derivative0,
            derivative1=derivative1,
            delta=delta,
        )
        tolerance = 2.0e-5 + 2.0e-5 * u_calculation.abs()
        if bool((torch.abs(recovered_u - u_calculation) > tolerance).any()):
            raise InvalidH1GapStateError("RQS inverse does not reconstruct its input")
        q = (index.to(theta.dtype) + theta) / self.central_bins
        return RQSTransformResult(value=q, log_abs_det=-log_du_dq)

    def positive_log_prob_u(
        self, u: Tensor, parameters: H1GapParameters
    ) -> Tensor:
        """Log density of ``u`` conditional on the strictly-positive route."""

        self._require_valid(parameters)
        if u.shape != parameters.zero_logits.shape:
            raise ValueError("positive log-gap shape must match H1 parameters")
        if not torch.isfinite(u).all() or bool((u <= 0).any()):
            raise ValueError("positive log gap must be finite and strictly positive")
        central = u <= parameters.u_tail
        safe_central_u = torch.where(central, u, parameters.u_tail * 0.5)
        central_inverse = self.central_inverse(safe_central_u, parameters)
        central_log_prob = torch.log1p(-parameters.pi_tail) + central_inverse.log_abs_det
        excess = torch.where(
            central,
            torch.zeros_like(u),
            u - parameters.u_tail,
        )
        tail_log_prob = (
            torch.log(parameters.pi_tail)
            - torch.log(parameters.beta)
            - excess / parameters.beta
        )
        result = torch.where(central, central_log_prob, tail_log_prob)
        if not torch.isfinite(result).all():
            raise InvalidH1GapStateError("positive-gap density is non-finite")
        result = result.to(u)
        if not torch.isfinite(result).all():
            raise InvalidH1GapStateError(
                "positive-gap density is non-finite in model precision"
            )
        return result

    def event_nll(
        self,
        gap: Tensor,
        parameters: H1GapParameters,
        *,
        valid_mask: Tensor,
    ) -> Tensor:
        """Per-event exact hurdle NLL in raw-gap coordinates."""

        self._require_valid(parameters)
        if gap.shape != parameters.zero_logits.shape or valid_mask.shape != gap.shape or valid_mask.dtype != torch.bool:
            raise ValueError("gap and valid mask must match H1 parameter shape")
        if not bool(valid_mask.any()):
            raise ValueError("H1 NLL requires at least one valid event")
        if not torch.isfinite(gap[valid_mask]).all() or bool((gap[valid_mask] < 0).any()):
            raise ValueError("valid gap must be finite and nonnegative")
        positive = gap > 0
        safe_gap = torch.where(
            positive & valid_mask,
            gap,
            torch.expm1(parameters.u_tail * 0.5),
        )
        u = torch.log1p(safe_gap)
        positive_log_prob = self.positive_log_prob_u(u, parameters) - u
        zero_log_prob = -F.softplus(-parameters.zero_logits)
        positive_hurdle_log_prob = -F.softplus(parameters.zero_logits)
        event_log_prob = torch.where(
            positive,
            positive_hurdle_log_prob + positive_log_prob,
            zero_log_prob,
        )
        nll = -event_log_prob
        if not torch.isfinite(nll[valid_mask]).all():
            raise InvalidH1GapStateError("H1 event NLL is non-finite")
        return torch.where(valid_mask, nll, torch.zeros_like(nll))

    def nll(
        self,
        gap: Tensor,
        parameters: H1GapParameters,
        *,
        valid_mask: Tensor,
    ) -> Tensor:
        event = self.event_nll(gap, parameters, valid_mask=valid_mask)
        return event[valid_mask].mean()

    def positive_cdf_u(
        self, u: Tensor, parameters: H1GapParameters
    ) -> Tensor:
        """Conditional-positive CDF with continuous mass at ``u_tail``."""

        self._require_valid(parameters)
        if u.shape != parameters.zero_logits.shape:
            raise ValueError("positive CDF input shape must match H1 parameters")
        if not torch.isfinite(u).all() or bool((u < 0).any()):
            raise ValueError("positive CDF input must be finite and nonnegative")
        below = (u > 0) & (u < parameters.u_tail)
        at_threshold = u == parameters.u_tail
        safe_central_u = torch.where(below, u, parameters.u_tail * 0.5)
        central_q = self.central_inverse(safe_central_u, parameters).value
        central_cdf = (1.0 - parameters.pi_tail) * central_q
        excess = torch.clamp(u - parameters.u_tail, min=0.0)
        tail_cdf = 1.0 - parameters.pi_tail + parameters.pi_tail * (
            -torch.expm1(-excess / parameters.beta)
        )
        result = torch.where(
            u <= 0,
            torch.zeros_like(u),
            torch.where(below, central_cdf, tail_cdf),
        )
        result = torch.where(at_threshold, 1.0 - parameters.pi_tail, result)
        if not torch.isfinite(result).all() or bool(((result < 0) | (result > 1)).any()):
            raise InvalidH1GapStateError("positive CDF is invalid")
        return result

    @staticmethod
    def _validate_open_uniform(value: Tensor, *, shape: torch.Size) -> None:
        if value.shape != shape:
            raise ValueError("sampling uniform shape must match H1 parameters")
        if not torch.isfinite(value).all() or bool(((value <= 0) | (value >= 1)).any()):
            raise ValueError("sampling uniforms must be in the open interval (0,1)")

    def sample_from_uniforms(
        self,
        parameters: H1GapParameters,
        *,
        zero_uniform: Tensor,
        tail_uniform: Tensor,
        value_uniform: Tensor,
    ) -> H1SampledGap:
        """Deterministically route open-interval uniforms through H1."""

        self._require_valid(parameters)
        shape = parameters.zero_logits.shape
        for value in (zero_uniform, tail_uniform, value_uniform):
            self._validate_open_uniform(value, shape=shape)
        zero_uniform = zero_uniform.to(parameters.zero_logits)
        tail_uniform = tail_uniform.to(parameters.zero_logits)
        value_uniform = value_uniform.to(parameters.zero_logits)
        zero_route = zero_uniform < torch.sigmoid(parameters.zero_logits)
        tail_route = tail_uniform < parameters.pi_tail
        central_u = self.central_quantile(value_uniform, parameters).value
        tail_u = parameters.u_tail - parameters.beta * torch.log1p(-value_uniform)
        positive_u_calculation = torch.where(tail_route, tail_u, central_u)
        if not torch.isfinite(positive_u_calculation).all():
            raise InvalidH1GapStateError(
                "H1 sampled log gap is non-finite; clipping and redraw are forbidden"
            )
        positive_u = positive_u_calculation.to(parameters.zero_logits)
        u = torch.where(zero_route, torch.zeros_like(positive_u), positive_u)
        gap = torch.expm1(u)
        if not torch.isfinite(u).all() or not torch.isfinite(gap).all() or bool((gap < 0).any()):
            raise InvalidH1GapStateError(
                "H1 sampled gap is invalid; clipping and redraw are forbidden"
            )
        return H1SampledGap(
            gap=gap,
            u=u,
            zero_route=zero_route,
            tail_route=tail_route & ~zero_route,
        )

    def sample(
        self,
        parameters: H1GapParameters,
        *,
        generator: torch.Generator | None = None,
    ) -> H1SampledGap:
        """Sample without hard clipping or redraw from three open uniforms."""

        self._require_valid(parameters)
        reference = parameters.zero_logits
        epsilon = torch.finfo(reference.dtype).eps

        def draw() -> Tensor:
            return torch.rand(
                reference.shape,
                dtype=reference.dtype,
                device=reference.device,
                generator=generator,
            ).clamp(epsilon, 1.0 - epsilon)

        return self.sample_from_uniforms(
            parameters,
            zero_uniform=draw(),
            tail_uniform=draw(),
            value_uniform=draw(),
        )


class CoFHCMTTPPV2H1(nn.Module):
    """The sole H1 candidate with only the C1 gap decoder replaced."""

    CANDIDATES = (H1_CANDIDATE_ID,)

    def __init__(
        self,
        *,
        candidate: str,
        receiver_classes: int,
        tail_state: TrainOnlyH1TailState,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_length: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if candidate != H1_CANDIDATE_ID:
            raise ValueError("CoF-HCMTTPP-v2 implements only H1")
        self.candidate = candidate
        self.receiver_classes = int(receiver_classes)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.max_length = int(max_length)
        self.dropout = float(dropout)
        self.tail_state = tail_state
        self.amount_contract = AMOUNT_CONTRACT_NAME
        self.structure_loss = None
        self.y_balanced_likelihood = False
        self.event_decoder = CausalEventDecoder(
            receiver_classes=receiver_classes,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_length=max_length,
            amount_dimensions=1,
            dropout=dropout,
        )
        self.gap_decoder = H1HurdleRQSGapDecoder(
            d_model=d_model,
            tail_state=tail_state,
        )
        self.receiver_decoder = FlatReceiverDecoder(
            receiver_classes=receiver_classes,
            d_model=d_model,
        )
        self.amount_head = ConditionalAmountHead(
            receiver_classes=receiver_classes,
            d_model=d_model,
        )

    def model_config(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "receiver_classes": self.receiver_classes,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_length": self.max_length,
            "dropout": self.dropout,
            "central_bins": H1_CENTRAL_BINS,
            "changed_factors": ["gap_decoder"],
            "receiver_path": "flat_no_copy",
            "amount_contract": self.amount_contract,
            "y_balanced_likelihood": self.y_balanced_likelihood,
            "structure_loss": self.structure_loss,
            "tail_state_sha256": self.tail_state.state_sha256,
            "permanent_v1_chain_state": PERMANENT_V1_CHAIN_STATE,
        }


_H1_CHECKPOINT_PROVENANCE_KEYS = (
    "source_sha256",
    "config_sha256",
    "train_manifest_sha256",
    "transform_state_sha256",
    "sampling_plan_sha256",
    "amount_contract_sha256",
    "receiver_vocabulary_sha256",
    "tail_state_sha256",
    "validation_rows_used",
    "internal_test_rows_used",
    "fraud_test_rows_used",
)


def _state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tail_state_to_dict(state: TrainOnlyH1TailState) -> dict[str, Any]:
    return {
        "schema_version": "cof-hcmttpp-v2-h1-tail-state-v1",
        "n_positive": list(state.n_positive),
        "m_strict_exceedances": list(state.m_strict_exceedances),
        "u_tail": list(state.u_tail),
        "p_tail_base": list(state.p_tail_base),
        "beta_base": list(state.beta_base),
        "ordered_positive_u_sha256": list(state.ordered_positive_u_sha256),
        "fit_split": state.fit_split,
        "provenance": [list(item) for item in state.provenance],
        "validation_rows_used": state.validation_rows_used,
        "internal_test_rows_used": state.internal_test_rows_used,
        "fraud_test_rows_used": state.fraud_test_rows_used,
        "state_sha256": state.state_sha256,
    }


def _tail_state_from_dict(raw: Mapping[str, Any]) -> TrainOnlyH1TailState:
    if raw.get("schema_version") != "cof-hcmttpp-v2-h1-tail-state-v1":
        raise ValueError("H1 tail-state schema mismatch")
    hash_payload = {key: value for key, value in raw.items() if key != "state_sha256"}
    if raw.get("state_sha256") != _canonical_sha256(hash_payload):
        raise ValueError("H1 tail-state hash mismatch")
    return TrainOnlyH1TailState(
        n_positive=tuple(int(value) for value in raw["n_positive"]),
        m_strict_exceedances=tuple(
            int(value) for value in raw["m_strict_exceedances"]
        ),
        u_tail=tuple(float(value) for value in raw["u_tail"]),
        p_tail_base=tuple(float(value) for value in raw["p_tail_base"]),
        beta_base=tuple(float(value) for value in raw["beta_base"]),
        ordered_positive_u_sha256=tuple(
            str(value) for value in raw["ordered_positive_u_sha256"]
        ),
        fit_split=str(raw["fit_split"]),
        provenance=tuple(
            (str(key), str(value)) for key, value in raw["provenance"]
        ),
        state_sha256=str(raw["state_sha256"]),
        validation_rows_used=int(raw["validation_rows_used"]),
        internal_test_rows_used=int(raw["internal_test_rows_used"]),
        fraud_test_rows_used=int(raw["fraud_test_rows_used"]),
    )


def build_h1_checkpoint_bundle(
    *,
    model: CoFHCMTTPPV2H1,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Create an in-memory, CPU checkpoint bundle without writing artifacts."""

    if tuple(sorted(provenance)) != tuple(sorted(_H1_CHECKPOINT_PROVENANCE_KEYS)):
        raise ValueError("H1 checkpoint provenance keys are incomplete")
    if any(
        int(provenance[key]) != 0
        for key in (
            "validation_rows_used",
            "internal_test_rows_used",
            "fraud_test_rows_used",
        )
    ):
        raise ValueError("H1 checkpoint provenance must be train-only")
    if provenance["tail_state_sha256"] != model.tail_state.state_sha256:
        raise ValueError("H1 checkpoint tail-state provenance mismatch")
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    return {
        "schema_version": "cof-hcmttpp-v2-h1-checkpoint-v1",
        "model_config": model.model_config(),
        "tail_state": _tail_state_to_dict(model.tail_state),
        "provenance": dict(provenance),
        "provenance_sha256": _canonical_sha256(dict(provenance)),
        "state_dict": state,
        "state_dict_sha256": _state_dict_sha256(state),
    }


def load_h1_checkpoint_bundle(
    bundle: Mapping[str, Any],
    *,
    expected_provenance: Mapping[str, Any],
) -> CoFHCMTTPPV2H1:
    """Strictly restore an in-memory H1 checkpoint on CPU."""

    if bundle.get("schema_version") != "cof-hcmttpp-v2-h1-checkpoint-v1":
        raise ValueError("H1 checkpoint schema mismatch")
    if (
        bundle.get("provenance") != dict(expected_provenance)
        or bundle.get("provenance_sha256")
        != _canonical_sha256(dict(expected_provenance))
    ):
        raise ValueError("H1 checkpoint provenance mismatch")
    state_dict = bundle.get("state_dict")
    if not isinstance(state_dict, Mapping) or bundle.get(
        "state_dict_sha256"
    ) != _state_dict_sha256(state_dict):
        raise ValueError("H1 checkpoint state hash mismatch")
    tail_state_raw = bundle.get("tail_state")
    if not isinstance(tail_state_raw, Mapping):
        raise ValueError("H1 checkpoint tail state is missing")
    tail_state = _tail_state_from_dict(tail_state_raw)
    if expected_provenance.get("tail_state_sha256") != tail_state.state_sha256:
        raise ValueError("H1 checkpoint tail-state provenance mismatch")

    raw = dict(bundle["model_config"])
    fixed = {
        "central_bins": H1_CENTRAL_BINS,
        "changed_factors": ["gap_decoder"],
        "receiver_path": "flat_no_copy",
        "amount_contract": AMOUNT_CONTRACT_NAME,
        "y_balanced_likelihood": False,
        "structure_loss": None,
        "tail_state_sha256": tail_state.state_sha256,
        "permanent_v1_chain_state": PERMANENT_V1_CHAIN_STATE,
    }
    for key, expected in fixed.items():
        if raw.pop(key, None) != expected:
            raise ValueError(f"H1 checkpoint factor-isolation mismatch: {key}")
    model = CoFHCMTTPPV2H1(tail_state=tail_state, **raw)
    model.load_state_dict(state_dict, strict=True)
    return model
