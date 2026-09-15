"""CoFSeqGen-SAF core model and preregistered decoder ablations.

This module is deliberately side-effect free: importing it never reads data,
queries CUDA, trains a model, or writes an artifact.  Dataset tensorization and
experiment execution belong to a separately authorized runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


PAD_CODE = 0
UNK_CODE = 1
MISSING_CODE = 2
LEARNED_START_CODE = 3
MODEL_IMPLEMENTATION_VERSION = "saf-gated-static-aware-gap-copy-route-v6"


@dataclass(frozen=True)
class GapSupportState:
    """Train-only, support-aligned discretization for non-negative gaps."""

    representatives: Tuple[float, ...]
    upper_bounds: Tuple[float, ...]
    source_unique_count: int
    fit_split: str = "train"
    zero_is_explicit: bool = True

    def __post_init__(self) -> None:
        if self.fit_split != "train":
            raise ValueError("gap support must be fitted on train only")
        if not self.representatives:
            raise ValueError("at least one gap support state is required")
        if len(self.representatives) != len(self.upper_bounds):
            raise ValueError("representatives and upper_bounds must align")
        reps = np.asarray(self.representatives, dtype=np.float64)
        bounds = np.asarray(self.upper_bounds, dtype=np.float64)
        if not np.isfinite(reps).all() or (reps < 0).any():
            raise ValueError("gap representatives must be finite and non-negative")
        if not np.isfinite(bounds[:-1]).all() or np.any(np.diff(bounds) <= 0):
            raise ValueError("finite upper bounds must be strictly increasing")
        if self.zero_is_explicit and reps[0] != 0.0:
            raise ValueError("the first support state must be the exact zero atom")

    @property
    def n_states(self) -> int:
        return len(self.representatives)

    def to_dict(self) -> Dict[str, object]:
        return {
            "representatives": list(self.representatives),
            "upper_bounds": [
                value if math.isfinite(value) else None
                for value in self.upper_bounds
            ],
            "source_unique_count": self.source_unique_count,
            "fit_split": self.fit_split,
            "zero_is_explicit": self.zero_is_explicit,
        }

    def encode_numpy(self, gaps: np.ndarray) -> np.ndarray:
        values = np.asarray(gaps, dtype=np.float64)
        out = np.full(values.shape, MISSING_CODE, dtype=np.int64)
        finite = np.isfinite(values)
        if np.any(values[finite] < 0):
            raise ValueError("negative gaps violate the canonical contract")
        if finite.any():
            bounds = np.asarray(self.upper_bounds, dtype=np.float64)
            state = np.searchsorted(bounds, values[finite], side="left")
            state = np.minimum(state, self.n_states - 1)
            out[finite] = state + LEARNED_START_CODE
        return out

    def decode_tensor(self, support_index: Tensor) -> Tensor:
        reps = torch.as_tensor(
            self.representatives,
            dtype=torch.float32,
            device=support_index.device,
        )
        return reps[support_index]


def _weighted_median(values: np.ndarray, counts: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    counts = counts[order]
    cutoff = counts.sum() / 2.0
    return float(values[np.searchsorted(np.cumsum(counts), cutoff, side="left")])


def fit_train_only_gap_support(
    gaps: Iterable[float],
    *,
    max_positive_states: int = 31,
) -> GapSupportState:
    """Fit exact-zero plus observed positive support representatives.

    Low-cardinality support is retained exactly.  High-cardinality support is
    partitioned by empirical mass; every decoded representative remains an
    actually observed training atom.
    """

    if max_positive_states < 1:
        raise ValueError("max_positive_states must be positive")
    values = np.asarray(list(gaps), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite training gaps")
    if np.any(values < 0):
        raise ValueError("negative training gaps")

    positive = values[values > 0]
    has_zero = bool(np.any(values == 0))
    representatives = [0.0] if has_zero else []
    if positive.size:
        unique, counts = np.unique(positive, return_counts=True)
        if unique.size <= max_positive_states:
            groups = [(unique[i : i + 1], counts[i : i + 1]) for i in range(unique.size)]
        else:
            cumulative = np.cumsum(counts)
            cuts = np.linspace(0, counts.sum(), max_positive_states + 1)[1:-1]
            split_at = np.unique(np.searchsorted(cumulative, cuts, side="right"))
            value_groups = np.split(unique, split_at)
            count_groups = np.split(counts, split_at)
            groups = [
                (group_values, group_counts)
                for group_values, group_counts in zip(value_groups, count_groups)
                if group_values.size
            ]
        representatives.extend(
            _weighted_median(group_values, group_counts)
            for group_values, group_counts in groups
        )

    reps = np.asarray(representatives, dtype=np.float64)
    if reps.size == 1:
        upper = np.asarray([np.inf], dtype=np.float64)
    else:
        upper = np.concatenate(((reps[:-1] + reps[1:]) / 2.0, [np.inf]))
    return GapSupportState(
        representatives=tuple(float(value) for value in reps),
        upper_bounds=tuple(float(value) for value in upper),
        source_unique_count=int(np.unique(values).size),
        zero_is_explicit=has_zero,
    )


@dataclass(frozen=True)
class SAFCandidate:
    candidate_id: str
    gap_decoder: str
    support_aligned: bool
    ordered: bool
    route_gap_to_mark: bool
    mark_decoder: str


SAF_CANDIDATES: Mapping[str, SAFCandidate] = {
    "SAF-C0": SAFCandidate(
        "SAF-C0", "hurdle_lognormal", False, False, False, "copy_new"
    ),
    "SAF-U0": SAFCandidate(
        "SAF-U0", "unordered", True, False, False, "copy_new"
    ),
    "SAF-O0": SAFCandidate(
        "SAF-O0", "ordered_hazard", True, True, False, "copy_new"
    ),
    "SAF-U1": SAFCandidate(
        "SAF-U1", "unordered", True, False, True, "copy_new"
    ),
    "SAF-O1": SAFCandidate(
        "SAF-O1", "ordered_hazard", True, True, True, "copy_new"
    ),
}


@dataclass(frozen=True)
class SAFModelConfig:
    candidate_id: str = "SAF-O1"
    receiver_vocab_size: int = 32
    static_dim: int = 0
    static_categorical_vocab_sizes: Tuple[int, ...] = ()
    auxiliary_categorical_vocab_sizes: Tuple[int, ...] = ()
    auxiliary_numeric_dim: int = 0
    hidden_dim: int = 128
    gap_embedding_dim: int = 32
    mark_embedding_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    max_gap: float = 3650.0
    context_window: int = 256
    teacher_force_current_gap_for_mark: bool = True
    teacher_force_current_gap_and_mark_for_value: bool = True
    length_policy: str = "shared_train_only_parent_length_plan"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "static_categorical_vocab_sizes",
            tuple(self.static_categorical_vocab_sizes),
        )
        object.__setattr__(
            self,
            "auxiliary_categorical_vocab_sizes",
            tuple(self.auxiliary_categorical_vocab_sizes),
        )
        if self.candidate_id not in SAF_CANDIDATES:
            raise ValueError(f"unknown SAF candidate: {self.candidate_id}")
        if self.receiver_vocab_size <= LEARNED_START_CODE:
            raise ValueError("receiver vocabulary must include learned event codes")
        if self.hidden_dim < 4 or self.gap_embedding_dim < 2 or self.mark_embedding_dim < 2:
            raise ValueError("embedding dimensions are too small")
        if self.num_layers < 1 or self.context_window < 1:
            raise ValueError("num_layers and context_window must be positive")
        all_vocab_sizes = (
            self.static_categorical_vocab_sizes
            + self.auxiliary_categorical_vocab_sizes
        )
        if any(size <= LEARNED_START_CODE for size in all_vocab_sizes):
            raise ValueError("categorical vocabularies must include learned codes")
        if self.static_dim < 0 or self.auxiliary_numeric_dim < 0:
            raise ValueError("numeric dimensions must be non-negative")
        if not self.teacher_force_current_gap_for_mark:
            raise ValueError("training must teacher-force observed current gap into mark")
        if not self.teacher_force_current_gap_and_mark_for_value:
            raise ValueError("training must teacher-force current gap and mark into value")


class UnorderedGapDecoder(nn.Module):
    def __init__(self, hidden_dim: int, n_states: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, n_states)

    def logits(self, hidden: Tensor) -> Tensor:
        return self.projection(hidden)

    def nll(self, hidden: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            self.logits(hidden).reshape(-1, self.projection.out_features),
            target.reshape(-1),
            reduction="none",
        ).reshape(target.shape)

    def sample(self, hidden: Tensor) -> Tensor:
        return torch.distributions.Categorical(logits=self.logits(hidden)).sample()


class OrderedHazardGapDecoder(nn.Module):
    """Ordinal discrete-time hazard with a forced final stopping state."""

    def __init__(self, hidden_dim: int, n_states: int) -> None:
        super().__init__()
        if n_states < 1:
            raise ValueError("n_states must be positive")
        self.n_states = n_states
        self.projection = nn.Linear(hidden_dim, max(1, n_states - 1))

    def probabilities(self, hidden: Tensor) -> Tensor:
        if self.n_states == 1:
            return torch.ones(*hidden.shape[:-1], 1, device=hidden.device)
        bounded_logits = 12.0 * torch.tanh(self.projection(hidden)[..., : self.n_states - 1] / 12.0)
        hazards = torch.sigmoid(bounded_logits)
        # An explicit scan avoids CUDA cumprod/cumsum backward kernels that do
        # not provide deterministic implementations on the locked runtime.
        survival = torch.ones_like(hazards[..., 0])
        stopping = []
        for hazard in hazards.unbind(-1):
            stopping.append(hazard * survival)
            survival = survival * (1.0 - hazard)
        return torch.stack((*stopping, survival), dim=-1)

    def nll(self, hidden: Tensor, target: Tensor) -> Tensor:
        probs = self.probabilities(hidden)
        selected = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return -torch.log(selected.clamp_min(torch.finfo(probs.dtype).tiny))

    def sample(self, hidden: Tensor) -> Tensor:
        return torch.distributions.Categorical(probs=self.probabilities(hidden)).sample()


class HurdleLogNormalGapDecoder(nn.Module):
    """Stable support-mismatched continuous control used only by SAF-C0."""

    def __init__(self, hidden_dim: int, max_gap: float) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 3)
        self.max_gap = float(max_gap)

    def parameters(self, hidden: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        zero_logit, location, raw_scale = self.projection(hidden).unbind(-1)
        scale = F.softplus(raw_scale).clamp(1e-3, 5.0)
        return zero_logit, location.clamp(-12.0, 12.0), scale

    def nll(self, hidden: Tensor, gap: Tensor) -> Tensor:
        zero_logit, location, scale = self.parameters(hidden)
        is_zero = torch.isfinite(gap) & (gap <= 0)
        hurdle = F.binary_cross_entropy_with_logits(
            zero_logit,
            is_zero.to(zero_logit.dtype),
            reduction="none",
        )
        safe_gap = torch.nan_to_num(gap, nan=0.0).clamp_min(0.0)
        log_u = torch.log(torch.log1p(safe_gap).clamp_min(1e-8))
        positive = (
            torch.log(scale)
            + 0.5 * math.log(2.0 * math.pi)
            + 0.5 * ((log_u - location) / scale).square()
            + torch.log1p(safe_gap)
            + torch.log(torch.log1p(safe_gap).clamp_min(1e-8))
        )
        return hurdle + torch.where(is_zero, torch.zeros_like(positive), positive)

    def sample(self, hidden: Tensor) -> Tensor:
        zero_logit, location, scale = self.parameters(hidden)
        is_zero = torch.bernoulli(torch.sigmoid(zero_logit)).bool()
        max_log_u = math.log(math.log1p(self.max_gap))
        log_u = (location + scale * torch.randn_like(location)).clamp(max=max_log_u)
        positive = torch.expm1(torch.exp(log_u)).clamp(0.0, self.max_gap)
        return torch.where(is_zero, torch.zeros_like(positive), positive)


def _categorical_embedding_dim(vocab_size: int) -> int:
    return min(32, max(4, int(math.ceil(math.sqrt(vocab_size)))))


class CausalHistoryEncoder(nn.Module):
    """Shifted GRU: output at t contains only events strictly before t."""

    def __init__(self, config: SAFModelConfig) -> None:
        super().__init__()
        self.config = config
        self.history_gap = nn.Sequential(
            nn.Linear(2, config.gap_embedding_dim),
            nn.Tanh(),
        )
        self.history_mark = nn.Embedding(
            config.receiver_vocab_size,
            config.mark_embedding_dim,
            padding_idx=PAD_CODE,
        )
        self.history_auxiliary_categorical = nn.ModuleList(
            nn.Embedding(
                vocab_size,
                _categorical_embedding_dim(vocab_size),
                padding_idx=PAD_CODE,
            )
            for vocab_size in config.auxiliary_categorical_vocab_sizes
        )
        auxiliary_categorical_dim = sum(
            embedding.embedding_dim
            for embedding in self.history_auxiliary_categorical
        )
        self.input_projection = nn.Linear(
            config.gap_embedding_dim
            + config.mark_embedding_dim
            + 1
            + auxiliary_categorical_dim
            + config.auxiliary_numeric_dim,
            config.hidden_dim,
        )
        self.gru = nn.GRU(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.start = nn.Parameter(torch.zeros(config.hidden_dim))
        self.static_categorical = nn.ModuleList(
            nn.Embedding(
                vocab_size,
                _categorical_embedding_dim(vocab_size),
            )
            for vocab_size in config.static_categorical_vocab_sizes
        )
        static_input_dim = config.static_dim + sum(
            embedding.embedding_dim for embedding in self.static_categorical
        )
        self.static_projection = (
            nn.Linear(static_input_dim, config.hidden_dim)
            if static_input_dim > 0
            else None
        )

    def forward(
        self,
        gap: Tensor,
        receiver: Tensor,
        numeric_value: Tensor,
        valid_mask: Tensor,
        static: Optional[Tensor] = None,
        static_categorical: Sequence[Tensor] = (),
        auxiliary_categorical: Sequence[Tensor] = (),
        auxiliary_numeric: Optional[Tensor] = None,
    ) -> Tensor:
        if gap.ndim != 2 or receiver.shape != gap.shape or numeric_value.shape != gap.shape:
            raise ValueError("gap, receiver, and numeric_value must have shape [B,T]")
        if valid_mask.shape != gap.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean [B,T]")
        invalid_negative = torch.isfinite(gap) & (gap < 0) & valid_mask
        if invalid_negative.any():
            raise ValueError("negative gap")
        if ((receiver < UNK_CODE) & valid_mask).any():
            raise ValueError("valid receiver cannot be PAD")
        if len(auxiliary_categorical) != len(
            self.config.auxiliary_categorical_vocab_sizes
        ):
            raise ValueError("auxiliary categorical field count mismatch")
        if any(values.shape != gap.shape for values in auxiliary_categorical):
            raise ValueError("auxiliary categoricals must have shape [B,T]")
        if self.config.auxiliary_numeric_dim:
            expected = (*gap.shape, self.config.auxiliary_numeric_dim)
            if auxiliary_numeric is None or auxiliary_numeric.shape != expected:
                raise ValueError("auxiliary numeric shape mismatch")
        elif auxiliary_numeric is not None and auxiliary_numeric.numel():
            raise ValueError("auxiliary numeric supplied to a zero-dimension model")

        gap_safe = torch.nan_to_num(gap, nan=0.0).clamp_min(0.0)
        gap_features = torch.stack(
            (torch.log1p(gap_safe), torch.isfinite(gap).to(gap.dtype)),
            dim=-1,
        )
        # Missing first gap is an input feature, not a reason to erase the
        # first event's mark/value from all subsequent history states.
        transition = valid_mask
        event_parts = [
            self.history_gap(gap_features),
            self.history_mark(receiver.clamp(0, self.config.receiver_vocab_size - 1)),
            torch.nan_to_num(numeric_value, nan=0.0).unsqueeze(-1),
        ]
        event_parts.extend(
            embedding(values.clamp(0, embedding.num_embeddings - 1))
            for embedding, values in zip(
                self.history_auxiliary_categorical,
                auxiliary_categorical,
            )
        )
        if auxiliary_numeric is not None:
            event_parts.append(torch.nan_to_num(auxiliary_numeric, nan=0.0))
        event = torch.cat(event_parts, dim=-1)
        event = self.input_projection(event) * transition.unsqueeze(-1)
        batch, steps = gap.shape
        initial = self.start.view(1, -1).expand(batch, -1)
        if self.static_projection is not None:
            if self.config.static_dim:
                if static is None or static.shape != (batch, self.config.static_dim):
                    raise ValueError("static numeric shape does not match static_dim")
                static_parts = [static]
            else:
                static_parts = []
                if static is not None and static.numel():
                    raise ValueError("static numeric supplied to static_dim=0")
            if len(static_categorical) != len(self.static_categorical):
                raise ValueError("static categorical field count mismatch")
            if any(values.shape != (batch,) for values in static_categorical):
                raise ValueError("static categorical fields must have shape [B]")
            static_parts.extend(
                embedding(values.clamp(0, embedding.num_embeddings - 1))
                for embedding, values in zip(
                    self.static_categorical,
                    static_categorical,
                )
            )
            initial = initial + self.static_projection(
                torch.cat(static_parts, dim=-1)
            )
        elif (static is not None and static.numel()) or static_categorical:
            raise ValueError("static context supplied to a context-free model")
        # Static context conditions both p(event_0 | static) and every later
        # history state.  Omitting it from h0 would make the direct static
        # route disappear after the first event.
        initial_state = initial.unsqueeze(0).repeat(
            self.config.num_layers,
            1,
            1,
        )
        encoded, _ = self.gru(event, initial_state.contiguous())
        return torch.cat((initial.unsqueeze(1), encoded[:, :-1]), dim=1)[:, :steps]


class CoFSeqGenSAF(nn.Module):
    """Support-Aligned Autoregressive Factorization and finite ablations."""

    def __init__(self, config: SAFModelConfig, support: GapSupportState) -> None:
        super().__init__()
        self.config = config
        self.candidate = SAF_CANDIDATES[config.candidate_id]
        self.support = support
        self.encoder = CausalHistoryEncoder(config)

        if self.candidate.gap_decoder == "unordered":
            self.gap_decoder: nn.Module = UnorderedGapDecoder(config.hidden_dim, support.n_states)
        elif self.candidate.gap_decoder == "ordered_hazard":
            self.gap_decoder = OrderedHazardGapDecoder(config.hidden_dim, support.n_states)
        else:
            self.gap_decoder = HurdleLogNormalGapDecoder(config.hidden_dim, config.max_gap)

        gap_route_vocab = support.n_states + LEARNED_START_CODE
        self.gap_route = nn.Embedding(gap_route_vocab, config.gap_embedding_dim)
        # Every candidate uses the same parameter-matched copy/new decoder.
        # The base term models history/static repeat propensity.  The routed
        # candidates additionally expose a current-gap delta through a
        # history-dependent gate; controls feed an exact zero gap context.
        # This makes the only U0/U1 and O0/O1 difference the current-gap input.
        if self.candidate.mark_decoder == "copy_new":
            self.copy_base_logit_head: Optional[nn.Linear] = nn.Linear(
                config.hidden_dim,
                1,
            )
            self.copy_gap_gate_head: Optional[nn.Linear] = nn.Linear(
                config.hidden_dim,
                1,
            )
            self.copy_gap_delta_head: Optional[nn.Linear] = nn.Linear(
                config.gap_embedding_dim,
                1,
                bias=False,
            )
            # Start from the no-gap model and let likelihood evidence turn on
            # a gap delta.  This avoids imposing random global gap effects at
            # initialization while retaining a non-zero delta gradient.
            nn.init.zeros_(self.copy_gap_delta_head.weight)
        else:
            self.copy_base_logit_head = None
            self.copy_gap_gate_head = None
            self.copy_gap_delta_head = None
        self.new_mark_head = nn.Linear(
            config.hidden_dim,
            config.receiver_vocab_size,
        )
        self.value_gap = nn.Sequential(
            nn.Linear(2, config.gap_embedding_dim),
            nn.Tanh(),
        )
        self.value_mark = nn.Embedding(
            config.receiver_vocab_size,
            config.mark_embedding_dim,
            padding_idx=PAD_CODE,
        )
        self.value_head = nn.Linear(
            config.hidden_dim + config.gap_embedding_dim + config.mark_embedding_dim,
            2,
        )
        event_core_dim = (
            config.hidden_dim
            + config.gap_embedding_dim
            + config.mark_embedding_dim
            + 1
        )
        self.auxiliary_categorical_heads = nn.ModuleList(
            nn.Linear(event_core_dim, vocab_size)
            for vocab_size in config.auxiliary_categorical_vocab_sizes
        )
        self.current_auxiliary_categorical = nn.ModuleList(
            nn.Embedding(
                vocab_size,
                _categorical_embedding_dim(vocab_size),
                padding_idx=PAD_CODE,
            )
            for vocab_size in config.auxiliary_categorical_vocab_sizes
        )
        auxiliary_route_dim = sum(
            embedding.embedding_dim
            for embedding in self.current_auxiliary_categorical
        )
        self.auxiliary_numeric_head = (
            nn.Linear(
                event_core_dim + auxiliary_route_dim,
                2 * config.auxiliary_numeric_dim,
            )
            if config.auxiliary_numeric_dim
            else None
        )

    def _support_code(self, gap: Tensor) -> Tensor:
        finite = torch.isfinite(gap)
        if (finite & (gap < 0)).any():
            raise ValueError("negative gap")
        finite_bounds = torch.as_tensor(
            self.support.upper_bounds[:-1],
            dtype=gap.dtype,
            device=gap.device,
        )
        state = torch.bucketize(torch.nan_to_num(gap, nan=0.0), finite_bounds)
        learned_code = state + LEARNED_START_CODE
        return torch.where(
            finite,
            learned_code,
            torch.full_like(learned_code, MISSING_CODE),
        )

    def _gap_context(self, gap: Tensor) -> Tensor:
        if self.candidate.support_aligned:
            return self.gap_route(self._support_code(gap))
        safe = torch.nan_to_num(gap, nan=0.0).clamp_min(0.0)
        features = torch.stack(
            (torch.log1p(safe), torch.isfinite(gap).to(gap.dtype)),
            dim=-1,
        )
        return self.value_gap(features)

    def _copy_logits(self, hidden: Tensor, current_gap: Tensor) -> Tensor:
        if (
            self.copy_base_logit_head is None
            or self.copy_gap_gate_head is None
            or self.copy_gap_delta_head is None
        ):
            raise RuntimeError("copy logits require the copy/new mark decoder")
        gap_context = (
            self._gap_context(current_gap)
            if self.candidate.route_gap_to_mark
            else hidden.new_zeros(*hidden.shape[:-1], self.config.gap_embedding_dim)
        )
        base = self.copy_base_logit_head(hidden).squeeze(-1)
        gate = torch.sigmoid(self.copy_gap_gate_head(hidden).squeeze(-1))
        gap_delta = self.copy_gap_delta_head(gap_context).squeeze(-1)
        return base + gate * gap_delta

    def _receiver_log_probabilities(
        self,
        hidden: Tensor,
        current_gap: Tensor,
        previous_receiver: Tensor,
        has_previous: Tensor,
    ) -> Tensor:
        if previous_receiver.shape != hidden.shape[:-1]:
            raise ValueError("previous_receiver must match hidden leading dimensions")
        if has_previous.shape != hidden.shape[:-1] or has_previous.dtype != torch.bool:
            raise ValueError("has_previous must be boolean and match hidden")
        new_log_probabilities = F.log_softmax(self.new_mark_head(hidden), dim=-1)
        if self.copy_base_logit_head is None:
            return new_log_probabilities

        copy_logits = self._copy_logits(hidden, current_gap)
        new_component = (
            F.logsigmoid(-copy_logits).unsqueeze(-1) + new_log_probabilities
        )
        copy_component = F.logsigmoid(copy_logits).unsqueeze(-1)
        previous = previous_receiver.clamp(0, self.config.receiver_vocab_size - 1)
        is_previous = F.one_hot(
            previous,
            num_classes=self.config.receiver_vocab_size,
        ).bool()
        mixture = torch.where(
            is_previous,
            torch.logaddexp(new_component, copy_component),
            new_component,
        )
        return torch.where(
            has_previous.unsqueeze(-1),
            mixture,
            new_log_probabilities,
        )

    def _value_parameters(
        self,
        hidden: Tensor,
        current_gap: Tensor,
        current_receiver: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        safe = torch.nan_to_num(current_gap, nan=0.0).clamp_min(0.0)
        gap_features = torch.stack(
            (torch.log1p(safe), torch.isfinite(current_gap).to(current_gap.dtype)),
            dim=-1,
        )
        features = torch.cat(
            (
                hidden,
                self.value_gap(gap_features),
                self.value_mark(current_receiver.clamp(0, self.config.receiver_vocab_size - 1)),
            ),
            dim=-1,
        )
        location, raw_scale = self.value_head(features).unbind(-1)
        return location, F.softplus(raw_scale).clamp(1e-3, 20.0)

    def _event_core_features(
        self,
        hidden: Tensor,
        current_gap: Tensor,
        current_receiver: Tensor,
        current_numeric: Tensor,
    ) -> Tensor:
        safe = torch.nan_to_num(current_gap, nan=0.0).clamp_min(0.0)
        gap_features = torch.stack(
            (torch.log1p(safe), torch.isfinite(current_gap).to(current_gap.dtype)),
            dim=-1,
        )
        return torch.cat(
            (
                hidden,
                self.value_gap(gap_features),
                self.value_mark(
                    current_receiver.clamp(
                        0,
                        self.config.receiver_vocab_size - 1,
                    )
                ),
                torch.nan_to_num(current_numeric, nan=0.0).unsqueeze(-1),
            ),
            dim=-1,
        )

    def _auxiliary_numeric_parameters(
        self,
        core: Tensor,
        auxiliary_categorical: Sequence[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if self.auxiliary_numeric_head is None:
            empty = core.new_zeros(*core.shape[:-1], 0)
            return empty, empty
        routed = [core]
        routed.extend(
            embedding(values.clamp(0, embedding.num_embeddings - 1))
            for embedding, values in zip(
                self.current_auxiliary_categorical,
                auxiliary_categorical,
            )
        )
        raw = self.auxiliary_numeric_head(torch.cat(routed, dim=-1))
        location, raw_scale = raw.chunk(2, dim=-1)
        return location, F.softplus(raw_scale).clamp(1e-3, 20.0)

    def compute_loss(
        self,
        *,
        gap: Tensor,
        receiver: Tensor,
        numeric_value: Tensor,
        valid_mask: Tensor,
        static: Optional[Tensor] = None,
        static_categorical: Sequence[Tensor] = (),
        auxiliary_categorical: Sequence[Tensor] = (),
        auxiliary_numeric: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Teacher-forced factorized likelihood.

        The first event has missing gap and contributes mark/value losses but no
        gap loss.  No sampled gap is ever injected into a training likelihood.
        """

        hidden = self.encoder(
            gap,
            receiver,
            numeric_value,
            valid_mask,
            static,
            static_categorical,
            auxiliary_categorical,
            auxiliary_numeric,
        )
        if target_mask is None:
            target_mask = valid_mask
        if target_mask.shape != valid_mask.shape or target_mask.dtype != torch.bool:
            raise ValueError("target_mask must be boolean [B,T]")
        if (target_mask & ~valid_mask).any():
            raise ValueError("target_mask cannot include padding")
        gap_mask = target_mask & torch.isfinite(gap)
        if isinstance(self.gap_decoder, HurdleLogNormalGapDecoder):
            per_gap = self.gap_decoder.nll(hidden, gap)
        else:
            target = (self._support_code(gap) - LEARNED_START_CODE).clamp_min(0)
            per_gap = self.gap_decoder.nll(hidden, target)
        gap_loss = per_gap[gap_mask].mean() if gap_mask.any() else hidden.sum() * 0.0

        previous_receiver = torch.full_like(receiver, UNK_CODE)
        previous_receiver[:, 1:] = receiver[:, :-1]
        has_previous = torch.zeros_like(valid_mask)
        has_previous[:, 1:] = valid_mask[:, 1:] & valid_mask[:, :-1]
        receiver_log_probabilities = self._receiver_log_probabilities(
            hidden,
            gap,
            previous_receiver,
            has_previous,
        )
        receiver_log_probability = receiver_log_probabilities.gather(
            -1,
            receiver.unsqueeze(-1),
        ).squeeze(-1)
        receiver_loss = -receiver_log_probability[target_mask].mean()
        location, scale = self._value_parameters(hidden, gap, receiver)
        safe_value = torch.nan_to_num(numeric_value, nan=0.0)
        per_value = (
            torch.log(scale)
            + 0.5 * math.log(2.0 * math.pi)
            + 0.5 * ((safe_value - location) / scale).square()
        )
        value_mask = target_mask & torch.isfinite(numeric_value)
        value_loss = per_value[value_mask].mean() if value_mask.any() else hidden.sum() * 0.0
        core = self._event_core_features(hidden, gap, receiver, numeric_value)
        auxiliary_category_losses = []
        for head, values in zip(
            self.auxiliary_categorical_heads,
            auxiliary_categorical,
        ):
            auxiliary_category_losses.append(
                F.cross_entropy(head(core)[target_mask], values[target_mask])
            )
        auxiliary_categorical_loss = (
            torch.stack(auxiliary_category_losses).mean()
            if auxiliary_category_losses
            else hidden.sum() * 0.0
        )
        auxiliary_location, auxiliary_scale = self._auxiliary_numeric_parameters(
            core,
            auxiliary_categorical,
        )
        if self.config.auxiliary_numeric_dim:
            if auxiliary_numeric is None:
                raise ValueError("auxiliary numeric target is required")
            safe_auxiliary = torch.nan_to_num(auxiliary_numeric, nan=0.0)
            per_auxiliary = (
                torch.log(auxiliary_scale)
                + 0.5 * math.log(2.0 * math.pi)
                + 0.5
                * ((safe_auxiliary - auxiliary_location) / auxiliary_scale).square()
            )
            auxiliary_mask = target_mask.unsqueeze(-1) & torch.isfinite(
                auxiliary_numeric
            )
            auxiliary_numeric_loss = (
                per_auxiliary[auxiliary_mask].mean()
                if auxiliary_mask.any()
                else hidden.sum() * 0.0
            )
        else:
            auxiliary_numeric_loss = hidden.sum() * 0.0
        total = (
            gap_loss
            + receiver_loss
            + value_loss
            + auxiliary_categorical_loss
            + auxiliary_numeric_loss
        )
        return {
            "loss": total,
            "gap_nll": gap_loss,
            "receiver_nll": receiver_loss,
            "value_nll": value_loss,
            "auxiliary_categorical_nll": auxiliary_categorical_loss,
            "auxiliary_numeric_nll": auxiliary_numeric_loss,
        }

    @torch.no_grad()
    def sample_fixed_lengths(
        self,
        lengths: Sequence[int],
        *,
        static: Optional[Tensor] = None,
        static_categorical: Sequence[Tensor] = (),
        device: Optional[torch.device] = None,
    ) -> Dict[str, Tensor]:
        """Generate children for an externally supplied train-only length plan."""

        if not lengths or any(int(length) < 1 for length in lengths):
            raise ValueError("all externally supplied lengths must be positive")
        chosen_device = device or next(self.parameters()).device
        lengths_tensor = torch.as_tensor(lengths, dtype=torch.long, device=chosen_device)
        batch = len(lengths)
        steps = int(lengths_tensor.max().item())
        valid = torch.arange(steps, device=chosen_device).unsqueeze(0) < lengths_tensor.unsqueeze(1)
        gap = torch.full((batch, steps), float("nan"), device=chosen_device)
        receiver = torch.full((batch, steps), PAD_CODE, dtype=torch.long, device=chosen_device)
        numeric = torch.zeros((batch, steps), device=chosen_device)
        auxiliary_categorical = tuple(
            torch.full(
                (batch, steps),
                PAD_CODE,
                dtype=torch.long,
                device=chosen_device,
            )
            for _ in self.config.auxiliary_categorical_vocab_sizes
        )
        auxiliary_numeric = torch.zeros(
            batch,
            steps,
            self.config.auxiliary_numeric_dim,
            device=chosen_device,
        )

        if static is not None:
            static = static.to(chosen_device)
        static_categorical = tuple(
            values.to(chosen_device) for values in static_categorical
        )
        for step in range(steps):
            active = valid[:, step]
            if not active.any():
                continue
            start = max(0, step + 1 - self.config.context_window)
            local_gap = gap[:, start : step + 1].clone()
            local_receiver = receiver[:, start : step + 1].clone()
            local_numeric = numeric[:, start : step + 1].clone()
            local_valid = valid[:, start : step + 1]
            local_auxiliary_categorical = tuple(
                values[:, start : step + 1].clone()
                for values in auxiliary_categorical
            )
            local_auxiliary_numeric = auxiliary_numeric[
                :, start : step + 1
            ].clone()
            local_receiver[:, -1] = UNK_CODE
            for values in local_auxiliary_categorical:
                values[:, -1] = UNK_CODE
            if step > 0:
                local_gap[:, -1] = 0.0
            hidden = self.encoder(
                local_gap,
                local_receiver,
                local_numeric,
                local_valid,
                static,
                static_categorical,
                local_auxiliary_categorical,
                local_auxiliary_numeric,
            )[:, -1]

            if step > 0:
                if isinstance(self.gap_decoder, HurdleLogNormalGapDecoder):
                    sampled_gap = self.gap_decoder.sample(hidden)
                else:
                    state = self.gap_decoder.sample(hidden)
                    sampled_gap = self.support.decode_tensor(state)
                gap[active, step] = sampled_gap[active]
            current_gap = gap[:, step]
            previous_receiver = (
                receiver[:, step - 1]
                if step > 0
                else torch.full(
                    (batch,),
                    UNK_CODE,
                    dtype=torch.long,
                    device=chosen_device,
                )
            )
            has_previous = active & (step > 0)
            receiver_logits = self._receiver_log_probabilities(
                hidden,
                current_gap,
                previous_receiver,
                has_previous,
            )
            # PAD and UNK are materialization mechanics. Source MISSING is a
            # legitimate modeled mark and therefore remains sampleable.
            receiver_logits[:, :MISSING_CODE] = -torch.inf
            sampled_receiver = torch.distributions.Categorical(logits=receiver_logits).sample()
            receiver[active, step] = sampled_receiver[active]
            location, scale = self._value_parameters(hidden, current_gap, sampled_receiver)
            sampled_numeric = location + scale * torch.randn_like(location)
            numeric[active, step] = sampled_numeric[active]
            core = self._event_core_features(
                hidden,
                current_gap,
                sampled_receiver,
                sampled_numeric,
            )
            sampled_auxiliary_categorical = []
            for head, output in zip(
                self.auxiliary_categorical_heads,
                auxiliary_categorical,
            ):
                logits = head(core)
                logits[:, :MISSING_CODE] = -torch.inf
                sampled = torch.distributions.Categorical(logits=logits).sample()
                output[active, step] = sampled[active]
                sampled_auxiliary_categorical.append(sampled)
            auxiliary_location, auxiliary_scale = (
                self._auxiliary_numeric_parameters(
                    core,
                    sampled_auxiliary_categorical,
                )
            )
            if self.config.auxiliary_numeric_dim:
                sampled_auxiliary_numeric = (
                    auxiliary_location
                    + auxiliary_scale * torch.randn_like(auxiliary_location)
                )
                auxiliary_numeric[active, step] = sampled_auxiliary_numeric[active]

        return {
            "gap": gap,
            "receiver": receiver,
            "numeric_value": numeric,
            "auxiliary_categorical": auxiliary_categorical,
            "auxiliary_numeric": auxiliary_numeric,
            "valid_mask": valid,
            "lengths": lengths_tensor,
        }

    def architecture_contract(self) -> Dict[str, object]:
        return {
            "implementation_version": MODEL_IMPLEMENTATION_VERSION,
            "family": "CoFSeqGen-SAF",
            "candidate": asdict(self.candidate),
            "factorization": "gap_then_mark_given_gap_then_value_given_gap_and_mark",
            "history": "strictly_past_shifted_gru",
            "static_context_route": "initial_output_and_all_gru_layer_initial_states",
            "training_route": "observed_current_gap_and_mark_teacher_forcing",
            "generation_route": "generated_current_gap_and_mark",
            "gap_to_mark_route": (
                "history_gated_actual_current_gap_delta_in_copy_logit"
                if self.candidate.route_gap_to_mark
                else "zero_gap_delta_in_parameter_matched_copy_logit"
            ),
            "copy_logit_factorization": (
                "base(history_static)+sigmoid(gate(history_static))*delta(current_gap)"
            ),
            "mark_decoder": self.candidate.mark_decoder,
            "mark_decoder_ablation_matched": all(
                candidate.mark_decoder == self.candidate.mark_decoder
                for candidate in SAF_CANDIDATES.values()
            ),
            "first_gap": "missing_masked_from_gap_likelihood",
            "length_policy": self.config.length_policy,
            "auxiliary_factorization": (
                "core_then_auxiliary_categoricals_then_auxiliary_numerics"
            ),
            "support": self.support.to_dict(),
        }
