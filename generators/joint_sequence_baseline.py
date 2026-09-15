from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from benchmarks.types import SequenceBatch, SyntheticBatch
from experiments.seed import seed_everything
from .sampling_plan import SamplingPlan


def _batch_hash(batch: SequenceBatch) -> str:
    digest = hashlib.sha256()
    for value in (
        batch.x_num,
        batch.dt_bin,
        batch.x_cat,
        batch.valid_mask,
        batch.y_entity,
        batch.lengths,
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _peak_cuda_memory(device: str) -> int:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated(resolved))


class _JointGRU(nn.Module):
    def __init__(
        self,
        *,
        gap_bins: int,
        receiver_categories: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        gap_dim = min(16, max(4, gap_bins))
        receiver_dim = min(32, max(8, receiver_categories // 2))
        self.gap_embedding = nn.Embedding(gap_bins, gap_dim)
        self.receiver_embedding = nn.Embedding(
            receiver_categories,
            receiver_dim,
        )
        self.label_embedding = nn.Embedding(2, 8)
        self.gru = nn.GRU(
            1 + gap_dim + receiver_dim + 8,
            hidden_size,
            batch_first=True,
        )
        self.amount_head = nn.Linear(hidden_size, 1)
        self.gap_head = nn.Linear(hidden_size, gap_bins)
        self.receiver_head = nn.Linear(hidden_size, receiver_categories)

    def encode_inputs(
        self,
        amount: torch.Tensor,
        gap: torch.Tensor,
        receiver: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        label_embedding = self.label_embedding(label)[:, None, :].expand(
            -1,
            amount.shape[1],
            -1,
        )
        return torch.cat(
            [
                amount,
                self.gap_embedding(gap),
                self.receiver_embedding(receiver),
                label_embedding,
            ],
            dim=-1,
        )

    def forward(
        self,
        amount: torch.Tensor,
        gap: torch.Tensor,
        receiver: torch.Tensor,
        label: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encode_inputs(amount, gap, receiver, label)
        state, hidden_out = self.gru(encoded, hidden)
        return (
            self.amount_head(state),
            self.gap_head(state),
            self.receiver_head(state),
            hidden_out,
        )


class NeuralSequenceBaseline:
    """Class-conditional autoregressive GRU baseline."""

    name = "neural_sequence_baseline"
    baseline_definition_version = "benchmark-v2.5"

    @staticmethod
    def _shift_inputs(
        amount: torch.Tensor,
        gap: torch.Tensor,
        receiver: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        amount_input = torch.zeros_like(amount)
        gap_input = torch.zeros_like(gap)
        receiver_input = torch.zeros_like(receiver)
        amount_input[:, 1:] = amount[:, :-1]
        gap_input[:, 1:] = gap[:, :-1]
        receiver_input[:, 1:] = receiver[:, :-1]
        return amount_input, gap_input, receiver_input

    def _loss(
        self,
        amount: torch.Tensor,
        gap: torch.Tensor,
        receiver: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        amount_input, gap_input, receiver_input = self._shift_inputs(
            amount,
            gap,
            receiver,
        )
        amount_pred, gap_logits, receiver_logits, _ = self.model(
            amount_input,
            gap_input,
            receiver_input,
            labels,
        )
        amount_loss = F.mse_loss(amount_pred[mask], amount[mask])
        gap_loss = F.cross_entropy(gap_logits[mask], gap[mask])
        receiver_loss = F.cross_entropy(
            receiver_logits[mask],
            receiver[mask],
        )
        return amount_loss + gap_loss + receiver_loss

    def _validation_loss(
        self,
        validation: SequenceBatch | None,
    ) -> float | None:
        if validation is None:
            return None
        self.model.eval()
        with torch.no_grad():
            amount = torch.from_numpy(validation.x_num[..., :1]).to(
                self.device
            )
            gap = torch.from_numpy(validation.dt_bin).to(self.device)
            receiver = torch.from_numpy(
                validation.x_cat[..., 0]
            ).to(self.device)
            labels = torch.from_numpy(validation.y_entity).to(self.device)
            mask = torch.from_numpy(validation.valid_mask).to(self.device)
            value = self._loss(
                amount,
                gap,
                receiver,
                labels,
                mask,
            )
        self.model.train()
        return float(value.detach().cpu())

    def fit(
        self,
        train: SequenceBatch,
        *,
        config: Mapping[str, Any],
        seed: int,
    ) -> None:
        seed_everything(seed)
        device = str(config.get("device", "cpu"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for neural baseline smoke")
        self.device = device
        self.gap_bins = int(train.dt_bin.max()) + 1
        self.receiver_categories = int(train.x_cat[..., 0].max()) + 1
        self.model = _JointGRU(
            gap_bins=self.gap_bins,
            receiver_categories=self.receiver_categories,
            hidden_size=int(config.get("hidden_size", 64)),
        ).to(device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config.get("lr", 1e-3)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        train_hash = _batch_hash(train)
        requested_steps = int(
            config.get("requested_steps", config.get("steps", 100))
        )
        batch_size = min(
            int(config.get("batch_size", len(train.lengths))),
            len(train.lengths),
        )
        max_wall_seconds = float(
            config.get("max_wall_seconds", float("inf"))
        )
        checkpoint_interval = int(
            config.get("checkpoint_interval_steps", requested_steps)
        )
        if (
            requested_steps < 1
            or batch_size < 1
            or max_wall_seconds <= 0
            or checkpoint_interval < 1
        ):
            raise ValueError("invalid neural training schedule")
        rng = np.random.default_rng(seed)
        self.loss_history: list[float] = []
        self.current_step = 0
        self.elapsed_seconds = 0.0
        resume = config.get("resume_checkpoint")
        if resume:
            state = torch.load(
                resume,
                map_location=device,
                weights_only=False,
            )
            if state["train_hash"] != train_hash:
                raise ValueError("resume train data hash mismatch")
            if state["requested_steps"] != requested_steps:
                raise ValueError("resume requested steps mismatch")
            self.model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            self.loss_history = list(state["loss_history"])
            self.current_step = int(state["step"])
            self.elapsed_seconds = float(state["elapsed_seconds"])
            rng.bit_generator.state = state["numpy_rng_state"]
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if device.startswith("cuda") and state.get("cuda_rng_state"):
                torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        self._optimizer = optimizer
        self._train_hash = train_hash
        self._requested_steps = requested_steps
        self._numpy_rng = rng
        validation = config.get("validation_batch")
        if validation is not None and not isinstance(
            validation,
            SequenceBatch,
        ):
            raise TypeError("validation_batch must be a SequenceBatch")
        progress_callback = config.get("progress_callback")
        checkpoint_callback = config.get("checkpoint_callback")
        started = time.perf_counter()
        base_elapsed = self.elapsed_seconds
        self.model.train()
        while self.current_step < requested_steps:
            if base_elapsed + time.perf_counter() - started >= max_wall_seconds:
                break
            indices = rng.choice(
                len(train.lengths),
                size=batch_size,
                replace=False,
            )
            amount = torch.from_numpy(train.x_num[indices, ..., :1]).to(
                device
            )
            gap = torch.from_numpy(train.dt_bin[indices]).to(device)
            receiver = torch.from_numpy(
                train.x_cat[indices, ..., 0]
            ).to(device)
            labels = torch.from_numpy(train.y_entity[indices]).to(device)
            mask = torch.from_numpy(train.valid_mask[indices]).to(device)
            optimizer.zero_grad()
            loss = self._loss(
                amount,
                gap,
                receiver,
                labels,
                mask,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite neural baseline loss")
            loss.backward()
            optimizer.step()
            self.loss_history.append(float(loss.detach().cpu()))
            self.current_step += 1
            self.elapsed_seconds = (
                base_elapsed + time.perf_counter() - started
            )
            should_report = (
                self.current_step % checkpoint_interval == 0
                or self.current_step == requested_steps
            )
            if should_report:
                event = {
                    "step": self.current_step,
                    "loss": self.loss_history[-1],
                    "validation_metric": self._validation_loss(validation),
                    "elapsed_seconds": self.elapsed_seconds,
                    "peak_gpu_memory_bytes": _peak_cuda_memory(device),
                }
                if progress_callback is not None:
                    progress_callback(event)
                if checkpoint_callback is not None:
                    checkpoint_callback(event, self)
        self.actual_training_budget = {
            "requested_steps": requested_steps,
            "actual_steps": self.current_step,
            "max_wall_seconds": max_wall_seconds,
            "actual_wall_seconds": self.elapsed_seconds,
            "wall_cap_reached": bool(
                self.current_step < requested_steps
                and self.elapsed_seconds >= max_wall_seconds
            ),
            "peak_gpu_memory_bytes": _peak_cuda_memory(device),
        }
        checkpoint = config.get("checkpoint_path")
        if checkpoint:
            Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
            self.save_training_checkpoint(checkpoint)

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {
            "schema_version": "benchmark-v2.5-neural-sequence",
            "model": self.model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "gap_bins": self.gap_bins,
            "receiver_categories": self.receiver_categories,
            "loss_history": self.loss_history,
            "step": self.current_step,
            "elapsed_seconds": self.elapsed_seconds,
            "requested_steps": self._requested_steps,
            "train_hash": self._train_hash,
            "numpy_rng_state": self._numpy_rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if self.device.startswith("cuda")
                else None
            ),
        }

    def save_training_checkpoint(self, path: str | Path) -> None:
        torch.save(self.checkpoint_state(), path)

    def sample(self, plan: SamplingPlan, *, seed: int) -> SyntheticBatch:
        seed_everything(seed)
        device = self.device
        n, length = plan.valid_mask.shape
        labels = torch.from_numpy(plan.y_entity).to(device)
        amount_out = torch.zeros((n, length, 1), device=device)
        gap_out = torch.zeros((n, length), dtype=torch.long, device=device)
        receiver_out = torch.zeros(
            (n, length),
            dtype=torch.long,
            device=device,
        )
        hidden = None
        self.model.eval()
        with torch.no_grad():
            for position in range(length):
                amount_step = (
                    amount_out[:, position - 1 : position]
                    if position
                    else torch.zeros((n, 1, 1), device=device)
                )
                gap_step = (
                    gap_out[:, position - 1 : position]
                    if position
                    else torch.zeros((n, 1), dtype=torch.long, device=device)
                )
                receiver_step = (
                    receiver_out[:, position - 1 : position]
                    if position
                    else torch.zeros((n, 1), dtype=torch.long, device=device)
                )
                amount_pred, gap_logits, receiver_logits, hidden = self.model(
                    amount_step,
                    gap_step,
                    receiver_step,
                    labels,
                    hidden,
                )
                amount_out[:, position] = amount_pred[:, 0]
                gap_out[:, position] = torch.multinomial(
                    torch.softmax(gap_logits[:, 0], dim=-1),
                    1,
                )[:, 0]
                receiver_out[:, position] = torch.multinomial(
                    torch.softmax(receiver_logits[:, 0], dim=-1),
                    1,
                )[:, 0]
        mask = torch.from_numpy(plan.valid_mask).to(device)
        amount_out[~mask] = 0
        gap_out[~mask] = 0
        receiver_out[~mask] = 0
        return SyntheticBatch(
            amount_out.cpu().numpy().astype(np.float32),
            gap_out.cpu().numpy().astype(np.int64),
            receiver_out[..., None].cpu().numpy().astype(np.int64),
            plan.valid_mask.copy(),
            plan.y_entity.copy(),
            plan.lengths.copy(),
        )
