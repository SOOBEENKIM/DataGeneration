"""Conditional coherence-preserving marked temporal point process.

This module is deliberately side-effect free: importing it never selects a
device, trains a model, samples a dataset, evaluates a candidate, or writes an
artifact.  It contains only the preregistered ``cof_ccmtpp_v1`` components.
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


PAD_RECEIVER = 0
UNK_RECEIVER = 1
AMOUNT_CONTRACT_NAME = "frozen_non_v3_train_only_encode_inverse_decode_v1"


def _prefix_mask(lengths: Tensor, width: int) -> Tensor:
    positions = torch.arange(width, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


class CausalEventDecoder(nn.Module):
    """Causal history encoder for the next marked event.

    Event inputs are shifted by one position, so output position ``t`` sees
    only events ``< t``. Entity identifiers are intentionally absent from the
    interface.
    """

    def __init__(
        self,
        *,
        receiver_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_length: int = 64,
        amount_dimensions: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if receiver_classes <= UNK_RECEIVER:
            raise ValueError("receiver support must contain PAD, UNK, and data")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.receiver_classes = receiver_classes
        self.d_model = d_model
        self.max_length = max_length
        self.amount_embedding = nn.Linear(amount_dimensions, d_model)
        self.gap_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # One extra index is a BOS symbol. PAD and UNK remain 0 and 1.
        self.bos_receiver = receiver_classes
        self.receiver_embedding = nn.Embedding(
            receiver_classes + 1,
            d_model,
            padding_idx=PAD_RECEIVER,
        )
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.length_embedding = nn.Embedding(max_length + 1, d_model)
        self.label_embedding = nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)

    @staticmethod
    def causal_attention_mask(length: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(
        self,
        *,
        amount: Tensor,
        gap: Tensor,
        receiver: Tensor,
        y: Tensor,
        lengths: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if amount.ndim != 3:
            raise ValueError("amount must have [batch, length, channel] shape")
        batch, width, _ = amount.shape
        expected = (batch, width)
        if (
            gap.shape != expected
            or receiver.shape != expected
            or receiver.dtype != torch.long
            or y.shape != (batch,)
            or lengths.shape != (batch,)
            or valid_mask.shape != expected
            or valid_mask.dtype != torch.bool
        ):
            raise ValueError("causal decoder tensor shapes are invalid")
        if width > self.max_length:
            raise ValueError("sequence exceeds configured maximum length")
        if bool((lengths < 1).any()) or bool((lengths > width).any()):
            raise ValueError("length is outside the configured sequence width")
        if not torch.equal(valid_mask, _prefix_mask(lengths.long(), width)):
            raise ValueError("valid mask must be a contiguous prefix")
        if not torch.isfinite(gap[valid_mask]).all() or bool((gap[valid_mask] < 0).any()):
            raise ValueError("valid continuous gap must be finite and nonnegative")
        if bool(((receiver[valid_mask] < UNK_RECEIVER) | (
            receiver[valid_mask] >= self.receiver_classes
        )).any()):
            raise ValueError("valid receiver code is outside train vocabulary")
        if bool(((y < 0) | (y > 1)).any()):
            raise ValueError("Y must be binary")

        # Shift by one event.  BOS is the only input at position zero.
        shifted_amount = torch.zeros_like(amount)
        shifted_gap = torch.zeros_like(gap)
        shifted_receiver = torch.full_like(receiver, self.bos_receiver)
        if width > 1:
            predecessor_valid = valid_mask[:, :-1]
            shifted_amount[:, 1:] = torch.where(
                predecessor_valid[..., None],
                amount[:, :-1],
                torch.zeros_like(amount[:, :-1]),
            )
            shifted_gap[:, 1:] = torch.where(
                predecessor_valid,
                gap[:, :-1],
                torch.zeros_like(gap[:, :-1]),
            )
            shifted_receiver[:, 1:] = torch.where(
                predecessor_valid,
                receiver[:, :-1],
                torch.full_like(receiver[:, :-1], self.bos_receiver),
            )
        position = torch.arange(width, device=amount.device)
        hidden = (
            self.amount_embedding(shifted_amount)
            + self.gap_embedding(torch.log1p(shifted_gap).unsqueeze(-1))
            + self.receiver_embedding(shifted_receiver)
            + self.position_embedding(position)[None]
            + self.length_embedding(lengths.long())[:, None]
            + self.label_embedding(y.long())[:, None]
        )
        hidden = self.transformer(
            hidden,
            mask=self.causal_attention_mask(width, amount.device),
            src_key_padding_mask=~valid_mask,
        )
        hidden = self.output_norm(hidden)
        return hidden.masked_fill(~valid_mask[..., None], 0.0)


@dataclass(frozen=True)
class GapMixtureParameters:
    mixture_logits: Tensor
    locations: Tensor
    scales: Tensor


@dataclass(frozen=True)
class SampledGap:
    gap: Tensor
    u: Tensor
    source: str = "cof_ccmtpp_v1_continuous_gap_head"


class MixtureLogisticGapHead(nn.Module):
    """Continuous density for ``u = log1p(gap)`` with finite support."""

    def __init__(
        self,
        *,
        d_model: int,
        components: int = 5,
        min_gap: float = 0.0,
        max_gap: float = 86400.0 * 30,
        min_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        if components < 1 or min_gap != 0.0 or max_gap <= min_gap:
            raise ValueError("continuous gap support is invalid")
        if min_scale <= 0:
            raise ValueError("minimum logistic scale must be positive")
        self.components = components
        self.min_gap = float(min_gap)
        self.max_gap = float(max_gap)
        self.min_scale = float(min_scale)
        self.u_max = math.log1p(self.max_gap)
        self.projection = nn.Linear(d_model, components * 3)

    def encode(self, gap: Tensor) -> Tensor:
        if not torch.isfinite(gap).all() or bool(
            ((gap < self.min_gap) | (gap > self.max_gap)).any()
        ):
            raise ValueError("gap is outside the preregistered support")
        return torch.log1p(gap)

    def decode(self, u: Tensor) -> Tensor:
        if not torch.isfinite(u).all() or bool(
            ((u < 0.0) | (u > self.u_max)).any()
        ):
            raise ValueError("log-gap is outside the preregistered support")
        return torch.expm1(u)

    def forward(self, hidden: Tensor) -> GapMixtureParameters:
        raw = self.projection(hidden)
        mixture_logits, locations, raw_scale = raw.chunk(3, dim=-1)
        return GapMixtureParameters(
            mixture_logits=mixture_logits,
            locations=locations.clamp(0.0, self.u_max),
            scales=F.softplus(raw_scale) + self.min_scale,
        )

    def log_prob_u(
        self,
        u: Tensor,
        parameters: GapMixtureParameters,
    ) -> Tensor:
        z = (u.unsqueeze(-1) - parameters.locations) / parameters.scales
        component = (
            -z
            - 2.0 * F.softplus(-z)
            - torch.log(parameters.scales)
        )
        return torch.logsumexp(
            F.log_softmax(parameters.mixture_logits, dim=-1) + component,
            dim=-1,
        )

    def log_prob_gap(
        self,
        gap: Tensor,
        parameters: GapMixtureParameters,
    ) -> Tensor:
        u = self.encode(gap)
        return self.log_prob_u(u, parameters) - u

    def nll(
        self,
        gap: Tensor,
        parameters: GapMixtureParameters,
        *,
        valid_mask: Tensor,
    ) -> Tensor:
        if valid_mask.shape != gap.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid mask must match gap")
        if not bool(valid_mask.any()):
            raise ValueError("gap NLL requires a valid event")
        # Continuous densities may exceed one, so a mathematically valid NLL
        # contribution can be negative. Do not clamp or otherwise change it.
        return -self.log_prob_gap(gap, parameters)[valid_mask].mean()

    def sample(
        self,
        parameters: GapMixtureParameters,
        *,
        generator: torch.Generator | None = None,
    ) -> SampledGap:
        probabilities = F.softmax(parameters.mixture_logits, dim=-1)
        component = torch.multinomial(
            probabilities.reshape(-1, self.components),
            1,
            generator=generator,
        ).reshape(parameters.mixture_logits.shape[:-1])
        location = torch.gather(
            parameters.locations,
            -1,
            component.unsqueeze(-1),
        ).squeeze(-1)
        scale = torch.gather(
            parameters.scales,
            -1,
            component.unsqueeze(-1),
        ).squeeze(-1)
        uniform = torch.rand(
            location.shape,
            dtype=location.dtype,
            device=location.device,
            generator=generator,
        ).clamp(torch.finfo(location.dtype).eps, 1.0 - torch.finfo(location.dtype).eps)
        u = (location + scale * (torch.log(uniform) - torch.log1p(-uniform))).clamp(
            0.0,
            self.u_max,
        )
        return SampledGap(gap=torch.expm1(u), u=u)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ReceiverHierarchyState:
    """Deterministic train-fitted head/tail receiver partition."""

    receiver_classes: int
    pad_code: int
    unk_code: int
    head_codes: tuple[int, ...]
    tail_clusters: tuple[tuple[int, ...], ...]
    counts: tuple[int, ...]
    fit_split: str
    provenance: tuple[tuple[str, str], ...]
    state_sha256: str
    validation_rows_used: int = 0
    test_rows_used: int = 0

    def __post_init__(self) -> None:
        codes = self.head_codes + tuple(
            code for cluster in self.tail_clusters for code in cluster
        )
        if (
            self.pad_code != PAD_RECEIVER
            or self.unk_code != UNK_RECEIVER
            or self.fit_split != "train"
            or len(self.counts) != self.receiver_classes
            or len(set(codes)) != len(codes)
            or set(codes) != set(range(2, self.receiver_classes))
            or any(code < 2 or code >= self.receiver_classes for code in codes)
            or self.validation_rows_used != 0
            or self.test_rows_used != 0
            or len(self.state_sha256) != 64
        ):
            raise ValueError("receiver hierarchy violates PAD/UNK/train-only contract")

    def classification(self, code: int) -> str:
        if code == self.unk_code:
            return "unk"
        if code in self.head_codes:
            return "head"
        if any(code in cluster for cluster in self.tail_clusters):
            return "tail"
        raise ValueError("receiver is outside the train-fitted hierarchy")


def fit_train_receiver_hierarchy(
    *,
    receiver: Tensor,
    valid_mask: Tensor,
    receiver_classes: int,
    fit_split: str,
    provenance: Mapping[str, str],
    head_min_count: int,
    max_head_categories: int,
    tail_cluster_count: int,
) -> ReceiverHierarchyState:
    if fit_split != "train":
        raise ValueError("receiver hierarchy is train-only")
    if receiver.shape != valid_mask.shape or valid_mask.dtype != torch.bool:
        raise ValueError("receiver and valid mask shapes differ")
    if head_min_count < 1 or max_head_categories < 1 or tail_cluster_count < 1:
        raise ValueError("receiver hierarchy rules must be positive")
    values = receiver[valid_mask]
    if bool(((values < 2) | (values >= receiver_classes)).any()):
        raise ValueError("train receiver contains PAD/UNK/out-of-vocabulary code")
    counts_tensor = torch.bincount(values, minlength=receiver_classes).cpu()
    counts = tuple(int(value) for value in counts_tensor.tolist())
    ordered = sorted(
        range(2, receiver_classes),
        key=lambda code: (-counts[code], code),
    )
    heads = tuple(
        code
        for code in ordered
        if counts[code] >= head_min_count
    )[:max_head_categories]
    tail = [code for code in ordered if code not in set(heads)]
    active_clusters = min(tail_cluster_count, max(len(tail), 1))
    clusters: list[list[int]] = [[] for _ in range(active_clusters)]
    for rank, code in enumerate(tail):
        clusters[rank % active_clusters].append(code)
    clusters = [cluster for cluster in clusters if cluster]
    provenance_tuple = tuple(sorted((str(k), str(v)) for k, v in provenance.items()))
    state_payload = {
        "schema_version": "cof-ccmtpp-receiver-hierarchy-v1",
        "receiver_classes": receiver_classes,
        "pad_code": PAD_RECEIVER,
        "unk_code": UNK_RECEIVER,
        "head_codes": list(heads),
        "tail_clusters": clusters,
        "counts": list(counts),
        "fit_split": "train",
        "provenance": dict(provenance_tuple),
        "validation_rows_used": 0,
        "test_rows_used": 0,
    }
    return ReceiverHierarchyState(
        receiver_classes=receiver_classes,
        pad_code=PAD_RECEIVER,
        unk_code=UNK_RECEIVER,
        head_codes=heads,
        tail_clusters=tuple(tuple(cluster) for cluster in clusters),
        counts=counts,
        fit_split="train",
        provenance=provenance_tuple,
        state_sha256=_canonical_sha256(state_payload),
    )


class FlatReceiverDecoder(nn.Module):
    """C1 flat receiver decoder without copy or hierarchy."""

    def __init__(self, *, receiver_classes: int, d_model: int) -> None:
        super().__init__()
        self.receiver_classes = receiver_classes
        self.receiver_head = nn.Linear(d_model, receiver_classes)

    def forward(self, hidden: Tensor) -> Tensor:
        logits = self.receiver_head(hidden)
        logits[..., PAD_RECEIVER] = float("-inf")
        return F.softmax(logits, dim=-1)


def _validate_sampled_gap(sampled_gap: object, prefix: torch.Size) -> SampledGap:
    if (
        not isinstance(sampled_gap, SampledGap)
        or sampled_gap.source != "cof_ccmtpp_v1_continuous_gap_head"
        or sampled_gap.gap.shape != prefix
        or sampled_gap.u.shape != prefix
    ):
        raise ValueError("receiver requires a model-sampled gap representation")
    return sampled_gap


def _causal_pointer(
    *,
    query: Tensor,
    key: Tensor,
    valid_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    batch, width, dimension = query.shape
    positions = torch.arange(width, device=query.device)
    allowed = (
        positions[None, None, :] < positions[None, :, None]
    ) & valid_mask[:, None, :] & valid_mask[:, :, None]
    logits = torch.einsum("btd,bsd->bts", query, key) / math.sqrt(dimension)
    masked = logits.masked_fill(~allowed, -1e9)
    weights = F.softmax(masked, dim=-1) * allowed.float()
    normalizer = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(normalizer > 0, weights / normalizer.clamp_min(1e-12), weights)
    return weights, allowed.any(dim=-1)


@dataclass(frozen=True)
class FlatReceiverDistribution:
    probabilities: Tensor
    copy_gate: Tensor
    pointer_weights: Tensor

    def sample(
        self, *, generator: torch.Generator | None = None
    ) -> Tensor:
        return torch.multinomial(
            self.probabilities.reshape(-1, self.probabilities.shape[-1]),
            1,
            generator=generator,
        ).reshape(self.probabilities.shape[:-1])


class FlatCopyReceiverDecoder(nn.Module):
    """C2 sampled-gap-conditioned copy plus flat-new receiver decoder."""

    def __init__(self, *, receiver_classes: int, d_model: int) -> None:
        super().__init__()
        self.receiver_classes = receiver_classes
        self.gap_condition = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.pointer_query = nn.Linear(d_model, d_model)
        self.receiver_key = nn.Embedding(receiver_classes, d_model, padding_idx=0)
        self.copy_gate = nn.Linear(d_model, 1)
        self.new_receiver_head = nn.Linear(d_model, receiver_classes)

    def forward(
        self,
        *,
        hidden: Tensor,
        history_receiver: Tensor,
        valid_mask: Tensor,
        sampled_gap: SampledGap,
    ) -> FlatReceiverDistribution:
        prefix = hidden.shape[:-1]
        sampled = _validate_sampled_gap(sampled_gap, prefix)
        if history_receiver.shape != prefix or valid_mask.shape != prefix:
            raise ValueError("receiver history shape mismatch")
        conditioned = hidden + self.gap_condition(sampled.u.unsqueeze(-1))
        pointer, has_history = _causal_pointer(
            query=self.pointer_query(conditioned),
            key=self.receiver_key(history_receiver),
            valid_mask=valid_mask,
        )
        gate = torch.sigmoid(self.copy_gate(conditioned).squeeze(-1))
        gate = torch.where(has_history, gate, torch.zeros_like(gate))
        copy_probability = hidden.new_zeros(*prefix, self.receiver_classes)
        copy_probability.scatter_add_(
            -1,
            history_receiver[:, None, :].expand(-1, prefix[1], -1),
            pointer,
        )
        new_logits = self.new_receiver_head(conditioned)
        new_logits[..., PAD_RECEIVER] = float("-inf")
        new_probability = F.softmax(new_logits, dim=-1)
        probability = gate[..., None] * copy_probability + (
            1.0 - gate[..., None]
        ) * new_probability
        return FlatReceiverDistribution(probability, gate, pointer)


@dataclass(frozen=True)
class HierarchicalReceiverDistribution:
    hierarchy: ReceiverHierarchyState
    copy_gate: Tensor
    pointer_weights: Tensor
    history_receiver: Tensor
    route_probabilities: Tensor
    tail_probabilities: tuple[Tensor, ...]

    def total_probability_mass(self) -> Tensor:
        copy_mass = self.pointer_weights.sum(dim=-1)
        new_mass = self.route_probabilities.sum(dim=-1)
        return self.copy_gate * copy_mass + (1.0 - self.copy_gate) * new_mass

    def target_probability(self, target: Tensor) -> Tensor:
        if target.shape != self.copy_gate.shape:
            raise ValueError("receiver target shape mismatch")
        copy_match = self.history_receiver[:, None, :] == target[..., None]
        copy = (self.pointer_weights * copy_match.float()).sum(dim=-1)
        new = torch.zeros_like(self.copy_gate)
        new = torch.where(
            target == self.hierarchy.unk_code,
            self.route_probabilities[..., 0],
            new,
        )
        for index, code in enumerate(self.hierarchy.head_codes):
            new = torch.where(
                target == code,
                self.route_probabilities[..., 1 + index],
                new,
            )
        cluster_offset = 1 + len(self.hierarchy.head_codes)
        for cluster_index, cluster in enumerate(self.hierarchy.tail_clusters):
            route = self.route_probabilities[..., cluster_offset + cluster_index]
            leaf = self.tail_probabilities[cluster_index]
            for leaf_index, code in enumerate(cluster):
                new = torch.where(target == code, route * leaf[..., leaf_index], new)
        probability = self.copy_gate * copy + (1.0 - self.copy_gate) * new
        return torch.where(target == PAD_RECEIVER, torch.zeros_like(probability), probability)

    def sample(
        self, *, generator: torch.Generator | None = None
    ) -> Tensor:
        prefix = self.copy_gate.shape
        uniform = torch.rand(
            prefix,
            dtype=self.copy_gate.dtype,
            device=self.copy_gate.device,
            generator=generator,
        )
        choose_copy = uniform < self.copy_gate
        pointer_choice = torch.multinomial(
            self.pointer_weights.reshape(-1, self.pointer_weights.shape[-1]).clamp_min(0)
            + (self.pointer_weights.sum(dim=-1).reshape(-1, 1) == 0).float(),
            1,
            generator=generator,
        ).reshape(prefix)
        copied = torch.gather(self.history_receiver, 1, pointer_choice)
        route = torch.multinomial(
            self.route_probabilities.reshape(-1, self.route_probabilities.shape[-1]),
            1,
            generator=generator,
        ).reshape(prefix)
        created = torch.full_like(route, self.hierarchy.unk_code)
        for index, code in enumerate(self.hierarchy.head_codes):
            created = torch.where(route == 1 + index, code, created)
        offset = 1 + len(self.hierarchy.head_codes)
        for cluster_index, cluster in enumerate(self.hierarchy.tail_clusters):
            leaf = torch.multinomial(
                self.tail_probabilities[cluster_index].reshape(-1, len(cluster)),
                1,
                generator=generator,
            ).reshape(prefix)
            code_table = torch.tensor(cluster, dtype=torch.long, device=leaf.device)
            selected_code = code_table[leaf]
            created = torch.where(route == offset + cluster_index, selected_code, created)
        return torch.where(choose_copy, copied, created)


class HierarchicalCopyReceiverDecoder(nn.Module):
    """C3/C4 copy gate and train-fitted hierarchical new-receiver route."""

    def __init__(self, *, hierarchy: ReceiverHierarchyState, d_model: int) -> None:
        super().__init__()
        self.hierarchy = hierarchy
        self.gap_condition = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.pointer_query = nn.Linear(d_model, d_model)
        self.receiver_key = nn.Embedding(
            hierarchy.receiver_classes, d_model, padding_idx=PAD_RECEIVER
        )
        self.copy_gate = nn.Linear(d_model, 1)
        route_count = 1 + len(hierarchy.head_codes) + len(hierarchy.tail_clusters)
        self.new_route_head = nn.Linear(d_model, route_count)
        self.tail_leaf_heads = nn.ModuleList(
            nn.Linear(d_model, len(cluster)) for cluster in hierarchy.tail_clusters
        )

    def forward(
        self,
        *,
        hidden: Tensor,
        history_receiver: Tensor,
        valid_mask: Tensor,
        sampled_gap: SampledGap,
    ) -> HierarchicalReceiverDistribution:
        prefix = hidden.shape[:-1]
        sampled = _validate_sampled_gap(sampled_gap, prefix)
        if history_receiver.shape != prefix or valid_mask.shape != prefix:
            raise ValueError("receiver history shape mismatch")
        conditioned = hidden + self.gap_condition(sampled.u.unsqueeze(-1))
        pointer, has_history = _causal_pointer(
            query=self.pointer_query(conditioned),
            key=self.receiver_key(history_receiver),
            valid_mask=valid_mask,
        )
        gate = torch.sigmoid(self.copy_gate(conditioned).squeeze(-1))
        gate = torch.where(has_history, gate, torch.zeros_like(gate))
        routes = F.softmax(self.new_route_head(conditioned), dim=-1)
        tails = tuple(F.softmax(head(conditioned), dim=-1) for head in self.tail_leaf_heads)
        return HierarchicalReceiverDistribution(
            hierarchy=self.hierarchy,
            copy_gate=gate,
            pointer_weights=pointer,
            history_receiver=history_receiver,
            route_probabilities=routes,
            tail_probabilities=tails,
        )


def receiver_nll_diagnostics(
    *,
    distribution: HierarchicalReceiverDistribution,
    target: Tensor,
    valid_mask: Tensor,
    hierarchy: ReceiverHierarchyState,
) -> dict[str, float | int]:
    if hierarchy != distribution.hierarchy:
        raise ValueError("receiver diagnostic hierarchy mismatch")
    probability = distribution.target_probability(target).clamp_min(1e-12)
    nll = -torch.log(probability)
    width = target.shape[1]
    positions = torch.arange(width, device=target.device)
    prior = positions[None, None, :] < positions[None, :, None]
    repeated = (
        (target[:, None, :] == target[..., None])
        & prior
        & valid_mask[:, None, :]
    ).any(dim=-1)
    classes = {
        "head": torch.zeros_like(valid_mask),
        "tail": torch.zeros_like(valid_mask),
        "unk": target == hierarchy.unk_code,
        "repeat": repeated,
        "new": ~repeated,
    }
    for code in hierarchy.head_codes:
        classes["head"] |= target == code
    for cluster in hierarchy.tail_clusters:
        for code in cluster:
            classes["tail"] |= target == code
    result: dict[str, float | int] = {}
    groups = {"overall": valid_mask, **classes}
    for name, group in groups.items():
        selected = group & valid_mask
        count = int(selected.sum().item())
        result[f"{name}_nll"] = float(nll[selected].mean().detach()) if count else 0.0
        result[f"{name}_count"] = count
    return result


def balanced_event_mean(
    event_loss: Tensor,
    *,
    y: Tensor,
    valid_mask: Tensor,
    y_balanced: bool,
) -> Tensor:
    """Aggregate valid event likelihood, optionally balancing the two Y classes."""

    if event_loss.shape != valid_mask.shape or valid_mask.dtype != torch.bool:
        raise ValueError("event loss and valid mask differ")
    if y.shape != (event_loss.shape[0],):
        raise ValueError("Y must be sequence-level")
    if not bool(valid_mask.any()):
        raise ValueError("likelihood requires a valid event")
    if not y_balanced:
        return event_loss[valid_mask].mean()
    class_losses = []
    for label in (0, 1):
        selected = valid_mask & (y == label)[:, None]
        if not bool(selected.any()):
            raise ValueError("Y-balanced likelihood requires both classes")
        class_losses.append(event_loss[selected].mean())
    return torch.stack(class_losses).mean()


class ConditionalAmountHead(nn.Module):
    """Existing normalized-amount path conditioned on sampled mark context."""

    def __init__(self, *, receiver_classes: int, d_model: int) -> None:
        super().__init__()
        self.receiver_embedding = nn.Embedding(
            receiver_classes, d_model, padding_idx=PAD_RECEIVER
        )
        self.gap_embedding = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.amount_head = nn.Linear(d_model, 1)

    def forward(
        self,
        *,
        hidden: Tensor,
        sampled_gap: SampledGap,
        receiver: Tensor,
    ) -> Tensor:
        sampled = _validate_sampled_gap(sampled_gap, hidden.shape[:-1])
        if receiver.shape != hidden.shape[:-1]:
            raise ValueError("amount receiver context shape mismatch")
        context = (
            hidden
            + self.gap_embedding(sampled.u.unsqueeze(-1))
            + self.receiver_embedding(receiver)
        )
        return self.amount_head(context).squeeze(-1)


class CoFCCMTPPV1(nn.Module):
    """Finite C1--C4 conditional marked temporal point-process family."""

    CANDIDATES = ("C1", "C2", "C3", "C4")

    def __init__(
        self,
        *,
        candidate: str,
        receiver_classes: int,
        hierarchy: ReceiverHierarchyState | None = None,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_length: int = 64,
        gap_components: int = 5,
        gap_max: float = 2592000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if candidate not in self.CANDIDATES:
            raise ValueError("candidate is outside the finite candidate family C1-C4")
        if candidate in {"C3", "C4"}:
            if hierarchy is None or hierarchy.receiver_classes != receiver_classes:
                raise ValueError("C3/C4 require a matching train-only hierarchy")
        elif hierarchy is not None:
            raise ValueError("C1/C2 cannot use a head/tail hierarchy")
        self.candidate = candidate
        self.receiver_classes = receiver_classes
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_length = max_length
        self.gap_components = gap_components
        self.gap_max = float(gap_max)
        self.dropout = float(dropout)
        self.amount_contract = AMOUNT_CONTRACT_NAME
        self.structure_loss = None
        self.y_balanced_likelihood = candidate == "C4"
        self.event_decoder = CausalEventDecoder(
            receiver_classes=receiver_classes,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_length=max_length,
            amount_dimensions=1,
            dropout=dropout,
        )
        self.gap_head = MixtureLogisticGapHead(
            d_model=d_model,
            components=gap_components,
            max_gap=gap_max,
        )
        if candidate == "C1":
            self.receiver_decoder: nn.Module = FlatReceiverDecoder(
                receiver_classes=receiver_classes, d_model=d_model
            )
        elif candidate == "C2":
            self.receiver_decoder = FlatCopyReceiverDecoder(
                receiver_classes=receiver_classes, d_model=d_model
            )
        else:
            assert hierarchy is not None
            self.receiver_decoder = HierarchicalCopyReceiverDecoder(
                hierarchy=hierarchy, d_model=d_model
            )
        self.amount_head = ConditionalAmountHead(
            receiver_classes=receiver_classes, d_model=d_model
        )

    @property
    def hierarchy(self) -> ReceiverHierarchyState | None:
        if isinstance(self.receiver_decoder, HierarchicalCopyReceiverDecoder):
            return self.receiver_decoder.hierarchy
        return None

    def model_config(self) -> dict[str, Any]:
        hierarchy = self.hierarchy
        return {
            "candidate": self.candidate,
            "receiver_classes": self.receiver_classes,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_length": self.max_length,
            "gap_components": self.gap_components,
            "gap_max": self.gap_max,
            "dropout": self.dropout,
            "amount_contract": self.amount_contract,
            "structure_loss": None,
            "hierarchy": None if hierarchy is None else _hierarchy_to_dict(hierarchy),
        }

    def compute_loss(
        self,
        *,
        amount: Tensor,
        gap: Tensor,
        receiver: Tensor,
        y: Tensor,
        lengths: Tensor,
        valid_mask: Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, dict[str, float | int]]:
        hidden = self.event_decoder(
            amount=amount,
            gap=gap,
            receiver=receiver,
            y=y,
            lengths=lengths,
            valid_mask=valid_mask,
        )
        gap_parameters = self.gap_head(hidden)
        gap_event_nll = -self.gap_head.log_prob_gap(gap, gap_parameters)
        sampled_gap = self.gap_head.sample(gap_parameters, generator=generator)
        if isinstance(self.receiver_decoder, FlatReceiverDecoder):
            receiver_probability = self.receiver_decoder(hidden)
            target_probability = torch.gather(
                receiver_probability, -1, receiver.unsqueeze(-1)
            ).squeeze(-1)
        elif isinstance(self.receiver_decoder, FlatCopyReceiverDecoder):
            receiver_distribution = self.receiver_decoder(
                hidden=hidden,
                history_receiver=receiver,
                valid_mask=valid_mask,
                sampled_gap=sampled_gap,
            )
            target_probability = torch.gather(
                receiver_distribution.probabilities,
                -1,
                receiver.unsqueeze(-1),
            ).squeeze(-1)
        else:
            assert isinstance(
                self.receiver_decoder, HierarchicalCopyReceiverDecoder
            )
            structured = self.receiver_decoder(
                hidden=hidden,
                history_receiver=receiver,
                valid_mask=valid_mask,
                sampled_gap=sampled_gap,
            )
            target_probability = structured.target_probability(receiver)
        receiver_event_nll = -torch.log(target_probability.clamp_min(1e-12))
        amount_prediction = self.amount_head(
            hidden=hidden,
            sampled_gap=sampled_gap,
            receiver=receiver,
        )
        amount_target = amount.squeeze(-1)
        amount_event_loss = (amount_prediction - amount_target).square()
        event_loss = gap_event_nll + receiver_event_nll + amount_event_loss
        total = balanced_event_mean(
            event_loss,
            y=y,
            valid_mask=valid_mask,
            y_balanced=self.y_balanced_likelihood,
        )
        return total, {
            "total_loss": float(total.detach()),
            "gap_nll": float(gap_event_nll[valid_mask].mean().detach()),
            "receiver_nll": float(receiver_event_nll[valid_mask].mean().detach()),
            "amount_loss": float(amount_event_loss[valid_mask].mean().detach()),
            "structure_loss": 0.0,
            "valid_events": int(valid_mask.sum().item()),
            "y_balanced_likelihood": int(self.y_balanced_likelihood),
        }

    def sample_mark_from_hidden(
        self,
        *,
        hidden: Tensor,
        history_receiver: Tensor,
        valid_mask: Tensor,
        generator: torch.Generator | None = None,
    ) -> "SampledMark":
        """Sample one factorized marked-event tensor from causal states.

        This primitive does not decode a full dataset or apply the external
        amount inverse transform; those operations belong to a future
        authorization-bound runner.
        """

        gap_parameters = self.gap_head(hidden)
        sampled_gap = self.gap_head.sample(gap_parameters, generator=generator)
        if isinstance(self.receiver_decoder, FlatReceiverDecoder):
            probabilities = self.receiver_decoder(hidden)
            sampled_receiver = torch.multinomial(
                probabilities.reshape(-1, self.receiver_classes),
                1,
                generator=generator,
            ).reshape(hidden.shape[:-1])
        elif isinstance(self.receiver_decoder, FlatCopyReceiverDecoder):
            sampled_receiver = self.receiver_decoder(
                hidden=hidden,
                history_receiver=history_receiver,
                valid_mask=valid_mask,
                sampled_gap=sampled_gap,
            ).sample(generator=generator)
        else:
            assert isinstance(self.receiver_decoder, HierarchicalCopyReceiverDecoder)
            sampled_receiver = self.receiver_decoder(
                hidden=hidden,
                history_receiver=history_receiver,
                valid_mask=valid_mask,
                sampled_gap=sampled_gap,
            ).sample(generator=generator)
        normalized_amount = self.amount_head(
            hidden=hidden,
            sampled_gap=sampled_gap,
            receiver=sampled_receiver,
        )
        sampled_receiver = torch.where(
            valid_mask, sampled_receiver, torch.zeros_like(sampled_receiver)
        )
        gap = torch.where(valid_mask, sampled_gap.gap, torch.zeros_like(sampled_gap.gap))
        amount = torch.where(valid_mask, normalized_amount, torch.zeros_like(normalized_amount))
        return SampledMark(
            gap=gap,
            receiver=sampled_receiver,
            normalized_amount=amount,
            gap_source=sampled_gap.source,
            amount_contract=self.amount_contract,
        )


@dataclass(frozen=True)
class SampledMark:
    gap: Tensor
    receiver: Tensor
    normalized_amount: Tensor
    gap_source: str
    amount_contract: str


def _hierarchy_to_dict(state: ReceiverHierarchyState) -> dict[str, Any]:
    return {
        "receiver_classes": state.receiver_classes,
        "pad_code": state.pad_code,
        "unk_code": state.unk_code,
        "head_codes": list(state.head_codes),
        "tail_clusters": [list(cluster) for cluster in state.tail_clusters],
        "counts": list(state.counts),
        "fit_split": state.fit_split,
        "provenance": [list(item) for item in state.provenance],
        "state_sha256": state.state_sha256,
        "validation_rows_used": state.validation_rows_used,
        "test_rows_used": state.test_rows_used,
    }


def _hierarchy_from_dict(raw: Mapping[str, Any]) -> ReceiverHierarchyState:
    return ReceiverHierarchyState(
        receiver_classes=int(raw["receiver_classes"]),
        pad_code=int(raw["pad_code"]),
        unk_code=int(raw["unk_code"]),
        head_codes=tuple(int(value) for value in raw["head_codes"]),
        tail_clusters=tuple(
            tuple(int(value) for value in cluster) for cluster in raw["tail_clusters"]
        ),
        counts=tuple(int(value) for value in raw["counts"]),
        fit_split=str(raw["fit_split"]),
        provenance=tuple((str(key), str(value)) for key, value in raw["provenance"]),
        state_sha256=str(raw["state_sha256"]),
        validation_rows_used=int(raw["validation_rows_used"]),
        test_rows_used=int(raw["test_rows_used"]),
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


_CHECKPOINT_PROVENANCE_KEYS = (
    "source_sha256",
    "config_sha256",
    "train_manifest_sha256",
    "transform_state_sha256",
    "sampling_plan_sha256",
    "amount_contract_sha256",
    "receiver_hierarchy_sha256",
    "validation_rows_used",
    "internal_test_rows_used",
    "fraud_test_rows_used",
)


def build_checkpoint_bundle(
    *, model: CoFCCMTPPV1, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    if tuple(sorted(provenance)) != tuple(sorted(_CHECKPOINT_PROVENANCE_KEYS)):
        raise ValueError("checkpoint provenance keys are incomplete")
    if any(int(provenance[key]) != 0 for key in (
        "validation_rows_used", "internal_test_rows_used", "fraud_test_rows_used"
    )):
        raise ValueError("checkpoint fit provenance must be train-only")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return {
        "schema_version": "cof-ccmtpp-v1-checkpoint-v1",
        "model_config": model.model_config(),
        "provenance": dict(provenance),
        "provenance_sha256": _canonical_sha256(dict(provenance)),
        "state_dict": state,
        "state_dict_sha256": _state_dict_sha256(state),
    }


def load_checkpoint_bundle(
    bundle: Mapping[str, Any], *, expected_provenance: Mapping[str, Any]
) -> CoFCCMTPPV1:
    if bundle.get("schema_version") != "cof-ccmtpp-v1-checkpoint-v1":
        raise ValueError("checkpoint schema mismatch")
    if (
        bundle.get("provenance") != dict(expected_provenance)
        or bundle.get("provenance_sha256") != _canonical_sha256(dict(expected_provenance))
    ):
        raise ValueError("checkpoint provenance mismatch")
    state = bundle.get("state_dict")
    if not isinstance(state, Mapping) or bundle.get("state_dict_sha256") != _state_dict_sha256(state):
        raise ValueError("checkpoint state hash mismatch")
    raw_config = dict(bundle["model_config"])
    if raw_config.pop("amount_contract") != AMOUNT_CONTRACT_NAME:
        raise ValueError("checkpoint amount contract mismatch")
    if raw_config.pop("structure_loss") is not None:
        raise ValueError("checkpoint contains a forbidden structure loss")
    hierarchy_raw = raw_config.pop("hierarchy")
    hierarchy = None if hierarchy_raw is None else _hierarchy_from_dict(hierarchy_raw)
    model = CoFCCMTPPV1(hierarchy=hierarchy, **raw_config)
    model.load_state_dict(state, strict=True)
    return model
