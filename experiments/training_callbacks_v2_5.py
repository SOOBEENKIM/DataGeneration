from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .full_artifact_store_v2_5 import FullAttemptStore


class AttemptTrainingCallbacks:
    """Wire adapter progress/checkpoint hooks to one immutable attempt."""

    def __init__(
        self,
        store: FullAttemptStore,
        *,
        checkpoint_interval_steps: int,
    ) -> None:
        if checkpoint_interval_steps < 1:
            raise ValueError("checkpoint interval must be positive")
        self.store = store
        self.interval = checkpoint_interval_steps
        self.last_progress_step = 0
        self.last_checkpoint_step = 0

    def progress(self, event: Mapping[str, Any]) -> None:
        step = int(event["step"])
        if step % self.interval != 0:
            return
        if step <= self.last_progress_step:
            raise ValueError("progress step is not strictly increasing")
        self.store.append_progress(event)
        self.store.write_partial_metrics(
            {
                "status": "RUNNING",
                "step": step,
                "loss": event["loss"],
                "validation_metric": event["validation_metric"],
                "elapsed_seconds": event["elapsed_seconds"],
                "peak_gpu_memory_bytes": event[
                    "peak_gpu_memory_bytes"
                ],
            }
        )
        self.last_progress_step = step

    def checkpoint(self, event: Mapping[str, Any], adapter: Any) -> Path:
        step = int(event["step"])
        if step % self.interval != 0:
            raise ValueError("adapter emitted an off-schedule checkpoint")
        if step <= self.last_checkpoint_step:
            raise ValueError("checkpoint step is not strictly increasing")
        if not hasattr(adapter, "save_training_checkpoint"):
            raise TypeError("learned adapter has no checkpoint serializer")
        path = self.store.write_checkpoint(
            step=step,
            writer=adapter.save_training_checkpoint,
        )
        self.last_checkpoint_step = step
        return path

    def save_final_checkpoint(self, adapter: Any) -> Path:
        if hasattr(adapter, "save_training_checkpoint"):
            writer = adapter.save_training_checkpoint
        else:
            writer = lambda path: torch.save(adapter, path)
        return self.store.write_immutable_file(
            "checkpoints/final.pt",
            writer,
        )
