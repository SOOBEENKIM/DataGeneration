from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import optim

from ctgan import CTGAN, TVAE
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.ctgan import Discriminator, Generator
from ctgan.synthesizers.tvae import Decoder, Encoder, _loss_function


ProgressCallback = Callable[[Mapping[str, Any]], None]
CheckpointCallback = Callable[[Mapping[str, Any]], None]


def _peak_memory(device: torch.device | str) -> int:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated(resolved))


def _move_optimizer(
    optimizer: torch.optim.Optimizer | None,
    device: torch.device | str,
) -> None:
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


class CheckpointableCTGAN(CTGAN):
    """ctgan 0.12.1 training loop with update-level checkpoint hooks.

    The upstream synthesizer does not retain its discriminator or optimizers,
    which prevents exact resume. This subclass retains that state and drops
    only the reconstructible transformed training array from serialized
    checkpoints.
    """

    v2_5_package_contract = "ctgan==0.12.1"

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_v25_train_data", None)
        state.pop("_data_sampler", None)
        return state

    def set_device(self, device):
        super().set_device(device)
        discriminator = getattr(self, "_v25_discriminator", None)
        if discriminator is not None:
            discriminator.to(self._device)
        _move_optimizer(
            getattr(self, "_v25_optimizer_g", None),
            self._device,
        )
        _move_optimizer(
            getattr(self, "_v25_optimizer_d", None),
            self._device,
        )

    def _initialize_v2_5(
        self,
        train_data: pd.DataFrame | np.ndarray,
        discrete_columns,
    ) -> None:
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        transformed = self._transformer.transform(train_data)
        self._v25_train_data = transformed
        self._data_sampler = DataSampler(
            transformed,
            self._transformer.output_info_list,
            self._log_frequency,
        )
        data_dim = self._transformer.output_dimensions
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)
        self._v25_discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac,
        ).to(self._device)
        self._v25_optimizer_g = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )
        self._v25_optimizer_d = optim.Adam(
            self._v25_discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )
        self.loss_values = pd.DataFrame(
            columns=["Step", "Generator Loss", "Discriminator Loss"]
        )
        self._v25_update = 0
        self._v25_elapsed_seconds = 0.0
        self._v25_initialized = True

    def _restore_train_state(
        self,
        train_data: pd.DataFrame | np.ndarray,
    ) -> None:
        transformed = self._transformer.transform(train_data)
        self._v25_train_data = transformed
        self._data_sampler = DataSampler(
            transformed,
            self._transformer.output_info_list,
            self._log_frequency,
        )
        self.set_device(self._device)
        if hasattr(self, "_v25_numpy_rng_state"):
            np.random.set_state(self._v25_numpy_rng_state)
        if hasattr(self, "_v25_torch_rng_state"):
            torch.set_rng_state(self._v25_torch_rng_state.cpu())
        if (
            torch.device(self._device).type == "cuda"
            and getattr(self, "_v25_cuda_rng_state", None) is not None
        ):
            torch.cuda.set_rng_state_all(self._v25_cuda_rng_state)

    def fit_steps(
        self,
        train_data: pd.DataFrame | np.ndarray,
        *,
        discrete_columns=(),
        requested_steps: int,
        max_wall_seconds: float,
        checkpoint_interval: int,
        progress_callback: ProgressCallback | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> Mapping[str, Any]:
        if requested_steps < 1 or max_wall_seconds <= 0:
            raise ValueError("requested steps and wall budget must be positive")
        if checkpoint_interval < 1:
            raise ValueError("checkpoint interval must be positive")
        start = time.perf_counter()
        if not getattr(self, "_v25_initialized", False):
            self._initialize_v2_5(train_data, discrete_columns)
        else:
            self._restore_train_state(train_data)
        mean = torch.zeros(
            self._batch_size,
            self._embedding_dim,
            device=self._device,
        )
        std = mean + 1
        base_elapsed = float(self._v25_elapsed_seconds)
        loss_g = loss_d = torch.tensor(float("nan"))
        while self._v25_update < requested_steps:
            current_elapsed = base_elapsed + time.perf_counter() - start
            if current_elapsed >= max_wall_seconds:
                break
            for _ in range(self._discriminator_steps):
                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1 = m1 = col = opt = None
                    real = self._data_sampler.sample_data(
                        self._v25_train_data,
                        self._batch_size,
                        col,
                        opt,
                    )
                else:
                    c1_np, m1_np, col, opt = condvec
                    c1 = torch.from_numpy(c1_np).to(self._device)
                    m1 = torch.from_numpy(m1_np).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)
                    permutation = np.arange(self._batch_size)
                    np.random.shuffle(permutation)
                    real = self._data_sampler.sample_data(
                        self._v25_train_data,
                        self._batch_size,
                        col[permutation],
                        opt[permutation],
                    )
                    c2 = c1[permutation]
                fake = self._generator(fakez)
                fake_activated = self._apply_activate(fake)
                real_tensor = torch.from_numpy(
                    real.astype("float32")
                ).to(self._device)
                if c1 is not None:
                    fake_input = torch.cat([fake_activated, c1], dim=1)
                    real_input = torch.cat([real_tensor, c2], dim=1)
                else:
                    fake_input, real_input = fake_activated, real_tensor
                y_fake = self._v25_discriminator(fake_input)
                y_real = self._v25_discriminator(real_input)
                penalty = self._v25_discriminator.calc_gradient_penalty(
                    real_input,
                    fake_input,
                    self._device,
                    self.pac,
                )
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                self._v25_optimizer_d.zero_grad(set_to_none=False)
                penalty.backward(retain_graph=True)
                loss_d.backward()
                self._v25_optimizer_d.step()
            fakez = torch.normal(mean=mean, std=std)
            condvec = self._data_sampler.sample_condvec(self._batch_size)
            if condvec is None:
                c1 = m1 = None
            else:
                c1_np, m1_np, _, _ = condvec
                c1 = torch.from_numpy(c1_np).to(self._device)
                m1 = torch.from_numpy(m1_np).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)
            fake = self._generator(fakez)
            fake_activated = self._apply_activate(fake)
            if c1 is None:
                y_fake = self._v25_discriminator(fake_activated)
                conditional_loss = 0
            else:
                y_fake = self._v25_discriminator(
                    torch.cat([fake_activated, c1], dim=1)
                )
                conditional_loss = self._cond_loss(fake, c1, m1)
            loss_g = -torch.mean(y_fake) + conditional_loss
            self._v25_optimizer_g.zero_grad(set_to_none=False)
            loss_g.backward()
            self._v25_optimizer_g.step()
            self._v25_update += 1
            self._v25_elapsed_seconds = (
                base_elapsed + time.perf_counter() - start
            )
            self._v25_numpy_rng_state = np.random.get_state()
            self._v25_torch_rng_state = torch.get_rng_state()
            self._v25_cuda_rng_state = (
                torch.cuda.get_rng_state_all()
                if torch.device(self._device).type == "cuda"
                else None
            )
            event = {
                "step": int(self._v25_update),
                "loss": float(loss_g.detach().cpu()),
                "discriminator_loss": float(loss_d.detach().cpu()),
                "validation_metric": None,
                "elapsed_seconds": float(self._v25_elapsed_seconds),
                "peak_gpu_memory_bytes": _peak_memory(self._device),
            }
            self.loss_values.loc[len(self.loss_values)] = [
                event["step"],
                event["loss"],
                event["discriminator_loss"],
            ]
            if progress_callback is not None:
                progress_callback(event)
            if (
                checkpoint_callback is not None
                and self._v25_update % checkpoint_interval == 0
            ):
                checkpoint_callback(event)
        return {
            "requested_steps": requested_steps,
            "actual_steps": int(self._v25_update),
            "max_wall_seconds": max_wall_seconds,
            "actual_wall_seconds": float(self._v25_elapsed_seconds),
            "wall_cap_reached": bool(
                self._v25_update < requested_steps
                and self._v25_elapsed_seconds >= max_wall_seconds
            ),
            "last_generator_loss": float(loss_g.detach().cpu()),
            "last_discriminator_loss": float(loss_d.detach().cpu()),
            "peak_gpu_memory_bytes": _peak_memory(self._device),
        }


class CheckpointableTVAE(TVAE):
    """TVAE with retained encoder/optimizer and exact update resume."""

    v2_5_package_contract = "ctgan==0.12.1"

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_v25_train_data", None)
        return state

    def set_device(self, device):
        if hasattr(self, "decoder"):
            super().set_device(device)
        else:
            self._device = torch.device(device)
        encoder = getattr(self, "_v25_encoder", None)
        if encoder is not None:
            encoder.to(self._device)
        _move_optimizer(
            getattr(self, "_v25_optimizer", None),
            self._device,
        )

    def _initialize_v2_5(
        self,
        train_data: pd.DataFrame | np.ndarray,
        discrete_columns,
        seed: int,
    ) -> None:
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        transformed = self.transformer.transform(train_data).astype("float32")
        self._v25_train_data = transformed
        data_dim = self.transformer.output_dimensions
        self._v25_encoder = Encoder(
            data_dim,
            self.compress_dims,
            self.embedding_dim,
        ).to(self._device)
        self.decoder = Decoder(
            self.embedding_dim,
            self.decompress_dims,
            data_dim,
        ).to(self._device)
        self._v25_optimizer = optim.Adam(
            list(self._v25_encoder.parameters())
            + list(self.decoder.parameters()),
            lr=float(getattr(self, "_v25_lr", 1e-3)),
            weight_decay=self.l2scale,
        )
        self._v25_rng = np.random.default_rng(seed)
        self._v25_update = 0
        self._v25_elapsed_seconds = 0.0
        self._v25_initialized = True
        self.loss_values = pd.DataFrame(columns=["Step", "Loss"])

    def _restore_train_state(
        self,
        train_data: pd.DataFrame | np.ndarray,
    ) -> None:
        self._v25_train_data = self.transformer.transform(
            train_data
        ).astype("float32")
        self.set_device(self._device)
        if hasattr(self, "_v25_torch_rng_state"):
            torch.set_rng_state(self._v25_torch_rng_state.cpu())
        if (
            torch.device(self._device).type == "cuda"
            and getattr(self, "_v25_cuda_rng_state", None) is not None
        ):
            torch.cuda.set_rng_state_all(self._v25_cuda_rng_state)

    def fit_steps(
        self,
        train_data: pd.DataFrame | np.ndarray,
        *,
        discrete_columns=(),
        requested_steps: int,
        max_wall_seconds: float,
        checkpoint_interval: int,
        seed: int,
        learning_rate: float = 1e-3,
        progress_callback: ProgressCallback | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
    ) -> Mapping[str, Any]:
        if requested_steps < 1 or max_wall_seconds <= 0:
            raise ValueError("requested steps and wall budget must be positive")
        if checkpoint_interval < 1:
            raise ValueError("checkpoint interval must be positive")
        start = time.perf_counter()
        if not getattr(self, "_v25_initialized", False):
            self._v25_lr = float(learning_rate)
            self._initialize_v2_5(train_data, discrete_columns, seed)
        else:
            if float(self._v25_lr) != float(learning_rate):
                raise ValueError("resume learning rate differs from checkpoint")
            self._restore_train_state(train_data)
        base_elapsed = float(self._v25_elapsed_seconds)
        loss = torch.tensor(float("nan"))
        row_count = len(self._v25_train_data)
        while self._v25_update < requested_steps:
            current_elapsed = base_elapsed + time.perf_counter() - start
            if current_elapsed >= max_wall_seconds:
                break
            batch_count = min(self.batch_size, row_count)
            indices = self._v25_rng.choice(
                row_count,
                batch_count,
                replace=False,
            )
            real = torch.from_numpy(
                self._v25_train_data[indices]
            ).to(self._device)
            self._v25_optimizer.zero_grad()
            mu, std, logvar = self._v25_encoder(real)
            embedding = torch.randn_like(std) * std + mu
            reconstructed, sigmas = self.decoder(embedding)
            reconstruction, divergence = _loss_function(
                reconstructed,
                real,
                sigmas,
                mu,
                logvar,
                self.transformer.output_info_list,
                self.loss_factor,
            )
            loss = reconstruction + divergence
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite TVAE loss")
            loss.backward()
            self._v25_optimizer.step()
            self.decoder.sigma.data.clamp_(0.01, 1.0)
            self._v25_update += 1
            self._v25_elapsed_seconds = (
                base_elapsed + time.perf_counter() - start
            )
            self._v25_torch_rng_state = torch.get_rng_state()
            self._v25_cuda_rng_state = (
                torch.cuda.get_rng_state_all()
                if torch.device(self._device).type == "cuda"
                else None
            )
            event = {
                "step": int(self._v25_update),
                "loss": float(loss.detach().cpu()),
                "validation_metric": None,
                "elapsed_seconds": float(self._v25_elapsed_seconds),
                "peak_gpu_memory_bytes": _peak_memory(self._device),
            }
            self.loss_values.loc[len(self.loss_values)] = [
                event["step"],
                event["loss"],
            ]
            if progress_callback is not None:
                progress_callback(event)
            if (
                checkpoint_callback is not None
                and self._v25_update % checkpoint_interval == 0
            ):
                checkpoint_callback(event)
        return {
            "requested_steps": requested_steps,
            "actual_steps": int(self._v25_update),
            "max_wall_seconds": max_wall_seconds,
            "actual_wall_seconds": float(self._v25_elapsed_seconds),
            "wall_cap_reached": bool(
                self._v25_update < requested_steps
                and self._v25_elapsed_seconds >= max_wall_seconds
            ),
            "last_loss": float(loss.detach().cpu()),
            "peak_gpu_memory_bytes": _peak_memory(self._device),
        }
