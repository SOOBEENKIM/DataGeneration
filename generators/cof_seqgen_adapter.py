from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from benchmarks.types import SequenceBatch, SyntheticBatch
from experiments.seed import seed_everything
from models.cof_seqgen import CoFSeqGen
from models.sampler import ddim_sample
from models.seq_denoiser import SeqDenoiser
from .sampling_plan import SamplingPlan


class CoFSeqGenAdapter:
    name = "cof"
    baseline_definition_version = "benchmark-v2.5"

    @staticmethod
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

    def _peak_memory(self) -> int:
        device = torch.device(self.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return 0
        return int(torch.cuda.max_memory_allocated(device))

    def _validation_loss(
        self,
        validation: SequenceBatch | None,
        *,
        batch_size: int,
        seed: int,
    ) -> float | None:
        if validation is None:
            return None
        indices = np.arange(min(batch_size, len(validation.lengths)))
        tensors = (
            torch.from_numpy(validation.x_num[indices]).to(self.device),
            torch.from_numpy(validation.dt_bin[indices]).to(self.device),
            torch.from_numpy(validation.x_cat[indices]).to(self.device),
            torch.from_numpy(
                validation.y_position()[indices]
            ).float().to(self.device),
            torch.from_numpy(validation.valid_mask[indices]).to(self.device),
        )
        devices = (
            [torch.device(self.device).index or 0]
            if self.device.startswith("cuda")
            else []
        )
        self.model.eval()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                loss, _ = self.model.compute_loss(*tensors, t_frac=0.5)
        self.model.train()
        return float(loss.detach().cpu())

    def fit(self, train: SequenceBatch, *, config: Mapping[str, Any], seed: int) -> None:
        seed_everything(seed)
        device = str(config.get("device", "cpu"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CoF smoke by the execution directive")
        if float(config.get("discrete_mask_max", 0.7)) != 0.7:
            raise ValueError(
                "the current CoF implementation fixes discrete_mask_max=0.7"
            )
        bins = int(train.dt_bin.max()) + 1
        categories = [int(train.x_cat[..., i].max()) + 1 for i in range(train.x_cat.shape[-1])]
        denoiser = SeqDenoiser(
            train.x_num.shape[-1], bins, categories,
            d_model=int(config.get("d_model", 64)), n_layers=int(config.get("n_layers", 1)),
            L_max=train.x_num.shape[1],
        )
        tau = torch.as_tensor(config["tau"], dtype=torch.float32)
        self.model = CoFSeqGen(
            denoiser, tau, float(config.get("window_width", 7)),
            float(config.get("temperature", 1)), float(config.get("coherence_lambda", 0)),
            categories, cfg_dropout=float(config.get("cfg_dropout", 0.15)),
        ).to(device)
        self.device, self.train = device, train
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config.get("lr", 1e-3)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
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
            raise ValueError("invalid CoF training schedule")
        train_hash = self._batch_hash(train)
        rng = np.random.default_rng(seed)
        self.current_step = 0
        self.elapsed_seconds = 0.0
        self.loss_history: list[float] = []
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
            self.current_step = int(state["step"])
            self.elapsed_seconds = float(state["elapsed_seconds"])
            self.loss_history = list(state["loss_history"])
            rng.bit_generator.state = state["numpy_rng_state"]
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if device.startswith("cuda") and state.get("cuda_rng_state"):
                torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        self._optimizer = optimizer
        self._train_hash = train_hash
        self._requested_steps = requested_steps
        self._numpy_rng = rng
        self.sample_config = {
            "diffusion_steps": int(config.get("diffusion_steps", 5)),
            "guidance_scale": float(config.get("guidance_scale", 2.0)),
            "feedback_discrete": bool(
                config.get("feedback_discrete", True)
            ),
            "feedback_after": float(config.get("feedback_after", 0.3)),
            "start_from_mask": bool(config.get("start_from_mask", True)),
            "sampling_chunk_size": int(
                config.get("sampling_chunk_size", 256)
            ),
        }
        if (
            self.sample_config["diffusion_steps"] < 1
            or self.sample_config["sampling_chunk_size"] < 1
            or not 0 <= self.sample_config["feedback_after"] <= 1
        ):
            raise ValueError("invalid CoF sampling configuration")
        validation = config.get("validation_batch")
        if validation is not None and not isinstance(
            validation,
            SequenceBatch,
        ):
            raise TypeError("validation_batch must be a SequenceBatch")
        progress_callback = config.get("progress_callback")
        checkpoint_callback = config.get("checkpoint_callback")
        update_timing_callback = config.get("update_timing_callback")
        started = time.perf_counter()
        base_elapsed = self.elapsed_seconds
        self.model.train()
        while self.current_step < requested_steps:
            if base_elapsed + time.perf_counter() - started >= max_wall_seconds:
                break
            if update_timing_callback is not None and device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
            update_started = time.perf_counter()
            indices = rng.choice(
                len(train.lengths),
                size=batch_size,
                replace=False,
            )
            tensors = (
                torch.from_numpy(train.x_num[indices]).to(device),
                torch.from_numpy(train.dt_bin[indices]).to(device),
                torch.from_numpy(train.x_cat[indices]).to(device),
                torch.from_numpy(
                    train.y_position()[indices]
                ).float().to(device),
                torch.from_numpy(train.valid_mask[indices]).to(device),
            )
            optimizer.zero_grad()
            loss, _ = self.model.compute_loss(
                *tensors,
                t_frac=(self.current_step + 1) / (requested_steps + 1),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite CoF loss")
            loss.backward()
            optimizer.step()
            if update_timing_callback is not None:
                if device.startswith("cuda"):
                    torch.cuda.synchronize(torch.device(device))
                update_timing_callback(
                    {
                        "step": self.current_step + 1,
                        "update_seconds": time.perf_counter()
                        - update_started,
                    }
                )
            self.current_step += 1
            self.loss_history.append(float(loss.detach().cpu()))
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
                    "validation_metric": self._validation_loss(
                        validation,
                        batch_size=batch_size,
                        seed=seed + 1_000_000 + self.current_step,
                    ),
                    "elapsed_seconds": self.elapsed_seconds,
                    "peak_gpu_memory_bytes": self._peak_memory(),
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
            "peak_gpu_memory_bytes": self._peak_memory(),
        }
        checkpoint = config.get("checkpoint_path")
        if checkpoint:
            Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
            self.save_training_checkpoint(checkpoint)

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {
            "schema_version": "benchmark-v2.5-cof-seqgen",
            "model": self.model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
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
            "sample_config": self.sample_config,
        }

    def save_training_checkpoint(self, path: str | Path) -> None:
        torch.save(self.checkpoint_state(), path)

    def sample(self, plan: SamplingPlan, *, seed: int) -> SyntheticBatch:
        seed_everything(seed)
        n, length = plan.valid_mask.shape
        numerical = np.zeros(
            (n, length, self.train.x_num.shape[-1]),
            dtype=np.float32,
        )
        gaps = np.zeros((n, length), dtype=np.int64)
        categories = np.zeros(
            (n, length, self.train.x_cat.shape[-1]),
            dtype=np.int64,
        )
        chunk_size = self.sample_config["sampling_chunk_size"]
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            count = stop - start
            dt = torch.zeros(
                count,
                length,
                dtype=torch.long,
                device=self.device,
            )
            cats = torch.zeros(
                count,
                length,
                self.train.x_cat.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            mask = torch.from_numpy(plan.valid_mask[start:stop]).to(
                self.device
            )
            x, dt_out, cat_out, _, _ = ddim_sample(
                self.model,
                dt,
                cats,
                self.train.x_num.shape[-1],
                T_steps=self.sample_config["diffusion_steps"],
                device=self.device,
                y_cond=torch.from_numpy(
                    plan.y_entity[start:stop]
                ).to(self.device),
                guidance_scale=self.sample_config["guidance_scale"],
                feedback_discrete=self.sample_config["feedback_discrete"],
                feedback_after=self.sample_config["feedback_after"],
                start_from_mask=self.sample_config["start_from_mask"],
                valid_mask=mask,
            )
            numerical[start:stop] = x.detach().cpu().numpy().astype(
                np.float32
            )
            gaps[start:stop] = dt_out.detach().cpu().numpy().astype(
                np.int64
            )
            categories[start:stop] = torch.stack(
                cat_out,
                -1,
            ).detach().cpu().numpy().astype(np.int64)
        return SyntheticBatch(
            numerical,
            gaps,
            categories,
            plan.valid_mask.copy(), plan.y_entity.copy(), plan.lengths.copy(),
        )
