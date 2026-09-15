from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ctgan.data_sampler import DataSampler

from benchmarks.types import SequenceBatch, SyntheticBatch
from experiments.seed import seed_everything
from generators.checkpointable_tabular_v2_5 import (
    CheckpointableCTGAN,
    CheckpointableTVAE,
    _peak_memory,
)
from generators.cof_seqgen_adapter import CoFSeqGenAdapter
from generators.conditional_ctgan import ConditionalCTGAN, _frame_hash
from generators.conditional_tvae import ConditionalTVAE
from generators.joint_sequence_baseline import NeuralSequenceBaseline
from generators.sampling_plan import SamplingPlan
from models.candidate_components_v2_6 import (
    WeightedCoFSeqGenV26,
    ddim_sample_v2_6,
    weighted_tvae_loss_v2_6,
)
from models.seq_denoiser import SeqDenoiser


def _loss_weights(
    config: Mapping[str, Any],
    *,
    include_label: bool,
) -> dict[str, float]:
    expected = {"amount", "gap", "receiver"}
    if include_label:
        expected.add("label")
    raw = config.get("v2_6_channel_loss_weights")
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ValueError("v2.6 candidate loss weights are incomplete")
    values = {key: float(raw[key]) for key in expected}
    if any(value <= 0 for value in values.values()):
        raise ValueError("v2.6 candidate loss weights must be positive")
    return values


def _sampling_rule(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("v2_6_sampling_rule")
    if not isinstance(value, Mapping):
        raise ValueError("v2.6 candidate sampling rule is missing")
    return value


class CheckpointableCTGANCandidateV26(CheckpointableCTGAN):
    def set_categorical_temperature(self, value: float) -> None:
        if value <= 0:
            raise ValueError("CTGAN categorical temperature must be positive")
        self.v2_6_categorical_temperature = float(value)

    def _apply_activate(self, data):
        transformed = []
        start = 0
        temperature = float(
            getattr(self, "v2_6_categorical_temperature", 0.2)
        )
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                end = start + span_info.dim
                if span_info.activation_fn == "tanh":
                    transformed.append(torch.tanh(data[:, start:end]))
                elif span_info.activation_fn == "softmax":
                    transformed.append(
                        self._gumbel_softmax(
                            data[:, start:end],
                            tau=temperature,
                        )
                    )
                else:
                    raise ValueError(
                        f"unexpected CTGAN activation: "
                        f"{span_info.activation_fn}"
                    )
                start = end
        return torch.cat(transformed, dim=1)


class ConditionalCTGANCandidateV26(ConditionalCTGAN):
    checkpointable_model_class = CheckpointableCTGANCandidateV26
    baseline_definition_version = "benchmark-v2.6-candidate-v1"

    def _new_model(self, config, *, epochs, checkpointable):
        if not checkpointable:
            raise ValueError("v2.6 candidate CTGAN must be checkpointable")
        model = super()._new_model(
            config,
            epochs=epochs,
            checkpointable=True,
        )
        model.set_device(str(config["device"]))
        rule = _sampling_rule(config)
        model.set_categorical_temperature(
            float(rule["categorical_temperature"])
        )
        return model

    def prepare_sampling_from_train(
        self,
        train: SequenceBatch,
    ) -> None:
        rows_y = np.repeat(train.y_entity, train.lengths)
        frame = pd.DataFrame(
            {
                "amount_log": train.x_num[train.valid_mask, 0],
                "dt_bin": train.dt_bin[train.valid_mask],
                "receiver": train.x_cat[train.valid_mask, 0],
            }
        )
        if set(self.models) != {0, 1}:
            raise ValueError(
                "CTGAN candidate restore requires both class models"
            )
        for label, model in self.models.items():
            class_frame = frame.loc[rows_y == label]
            expected_hash = self.training_data_hashes_by_class.get(label)
            if expected_hash != _frame_hash(class_frame):
                raise ValueError(
                    "CTGAN sampling restore train rows differ from checkpoint"
                )
            transformed = model._transformer.transform(class_frame)
            model._data_sampler = DataSampler(
                transformed,
                model._transformer.output_info_list,
                model._log_frequency,
            )


class CheckpointableTVAECandidateV26(CheckpointableTVAE):
    def set_candidate_contract(
        self,
        *,
        channel_weights: Mapping[str, float],
        latent_scale: float,
        categorical_temperature: float | None,
    ) -> None:
        if latent_scale <= 0:
            raise ValueError("TVAE latent scale must be positive")
        if (
            categorical_temperature is not None
            and categorical_temperature <= 0
        ):
            raise ValueError(
                "TVAE categorical temperature must be positive"
            )
        self.v2_6_channel_weights = dict(channel_weights)
        self.v2_6_latent_scale = float(latent_scale)
        self.v2_6_categorical_temperature = categorical_temperature

    def fit_steps(
        self,
        train_data,
        *,
        discrete_columns=(),
        requested_steps: int,
        max_wall_seconds: float,
        checkpoint_interval: int,
        seed: int,
        learning_rate: float = 1e-3,
        progress_callback=None,
        checkpoint_callback=None,
    ):
        if requested_steps < 1 or max_wall_seconds <= 0:
            raise ValueError("requested steps and wall budget must be positive")
        if checkpoint_interval < 1:
            raise ValueError("checkpoint interval must be positive")
        started = time.perf_counter()
        if not getattr(self, "_v25_initialized", False):
            self._v25_lr = float(learning_rate)
            self._initialize_v2_5(train_data, discrete_columns, seed)
        else:
            if float(self._v25_lr) != float(learning_rate):
                raise ValueError("resume learning rate differs")
            self._restore_train_state(train_data)
        base_elapsed = float(self._v25_elapsed_seconds)
        loss = torch.tensor(float("nan"))
        row_count = len(self._v25_train_data)
        while self._v25_update < requested_steps:
            if (
                base_elapsed + time.perf_counter() - started
                >= max_wall_seconds
            ):
                break
            count = min(self.batch_size, row_count)
            indices = self._v25_rng.choice(
                row_count,
                count,
                replace=False,
            )
            real = torch.from_numpy(
                self._v25_train_data[indices]
            ).to(self._device)
            self._v25_optimizer.zero_grad()
            mu, std, logvar = self._v25_encoder(real)
            embedding = torch.randn_like(std) * std + mu
            reconstructed, sigmas = self.decoder(embedding)
            reconstruction, divergence = weighted_tvae_loss_v2_6(
                reconstructed,
                real,
                sigmas,
                mu,
                logvar,
                self.transformer.output_info_list,
                self.loss_factor,
                self.v2_6_channel_weights,
            )
            loss = reconstruction + divergence
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite v2.6 TVAE loss")
            loss.backward()
            self._v25_optimizer.step()
            self.decoder.sigma.data.clamp_(0.01, 1.0)
            self._v25_update += 1
            self._v25_elapsed_seconds = (
                base_elapsed + time.perf_counter() - started
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

    def sample(self, samples: int):
        self.decoder.eval()
        batches = samples // self.batch_size + 1
        values = []
        sigmas = None
        for _ in range(batches):
            noise = (
                torch.randn(self.batch_size, self.embedding_dim)
                * self.v2_6_latent_scale
            ).to(self._device)
            raw, sigmas = self.decoder(noise)
            transformed = []
            start = 0
            for column_index, column_info in enumerate(
                self.transformer.output_info_list
            ):
                for span_info in column_info:
                    end = start + span_info.dim
                    logits = raw[:, start:end]
                    if span_info.activation_fn == "tanh":
                        transformed.append(torch.tanh(logits))
                    elif (
                        column_index > 0
                        and self.v2_6_categorical_temperature is not None
                    ):
                        probabilities = torch.softmax(
                            logits
                            / self.v2_6_categorical_temperature,
                            dim=-1,
                        )
                        choices = torch.multinomial(probabilities, 1)
                        transformed.append(
                            F.one_hot(
                                choices[:, 0],
                                num_classes=span_info.dim,
                            ).to(logits.dtype)
                        )
                    else:
                        transformed.append(torch.tanh(logits))
                    start = end
            values.append(
                torch.cat(transformed, dim=1).detach().cpu().numpy()
            )
        data = np.concatenate(values, axis=0)[:samples]
        assert sigmas is not None
        return self.transformer.inverse_transform(
            data,
            sigmas.detach().cpu().numpy(),
        )


class ConditionalTVAECandidateV26(ConditionalTVAE):
    checkpointable_model_class = CheckpointableTVAECandidateV26
    baseline_definition_version = "benchmark-v2.6-candidate-v1"

    def _new_model(self, config, *, epochs, checkpointable):
        if not checkpointable:
            raise ValueError("v2.6 candidate TVAE must be checkpointable")
        model = super()._new_model(
            config,
            epochs=epochs,
            checkpointable=True,
        )
        model.set_device(str(config["device"]))
        rule = _sampling_rule(config)
        model.set_candidate_contract(
            channel_weights=_loss_weights(
                config,
                include_label=False,
            ),
            latent_scale=float(rule["latent_scale"]),
            categorical_temperature=(
                float(rule["categorical_temperature"])
                if "categorical_temperature" in rule
                else None
            ),
        )
        return model


class NeuralSequenceCandidateV26(NeuralSequenceBaseline):
    baseline_definition_version = "benchmark-v2.6-candidate-v1"

    def fit(self, train, *, config, seed):
        self.v2_6_channel_weights = _loss_weights(
            config,
            include_label=False,
        )
        self.v2_6_categorical_temperature = float(
            _sampling_rule(config)["categorical_temperature"]
        )
        if self.v2_6_categorical_temperature <= 0:
            raise ValueError("neural categorical temperature must be positive")
        return super().fit(train, config=config, seed=seed)

    def _loss(self, amount, gap, receiver, labels, mask):
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
        weights = self.v2_6_channel_weights
        return (
            weights["amount"]
            * F.mse_loss(amount_pred[mask], amount[mask])
            + weights["gap"]
            * F.cross_entropy(gap_logits[mask], gap[mask])
            + weights["receiver"]
            * F.cross_entropy(receiver_logits[mask], receiver[mask])
        )

    def sample(self, plan: SamplingPlan, *, seed: int) -> SyntheticBatch:
        seed_everything(seed)
        device = self.device
        count, length = plan.valid_mask.shape
        labels = torch.from_numpy(plan.y_entity).to(device)
        amount = torch.zeros((count, length, 1), device=device)
        gap = torch.zeros(
            (count, length),
            dtype=torch.long,
            device=device,
        )
        receiver = torch.zeros_like(gap)
        hidden = None
        self.model.eval()
        with torch.no_grad():
            for position in range(length):
                amount_step = (
                    amount[:, position - 1 : position]
                    if position
                    else torch.zeros((count, 1, 1), device=device)
                )
                gap_step = (
                    gap[:, position - 1 : position]
                    if position
                    else torch.zeros(
                        (count, 1),
                        dtype=torch.long,
                        device=device,
                    )
                )
                receiver_step = (
                    receiver[:, position - 1 : position]
                    if position
                    else torch.zeros_like(gap_step)
                )
                (
                    amount_pred,
                    gap_logits,
                    receiver_logits,
                    hidden,
                ) = self.model(
                    amount_step,
                    gap_step,
                    receiver_step,
                    labels,
                    hidden,
                )
                amount[:, position] = amount_pred[:, 0]
                temperature = self.v2_6_categorical_temperature
                gap[:, position] = torch.multinomial(
                    torch.softmax(
                        gap_logits[:, 0] / temperature,
                        dim=-1,
                    ),
                    1,
                )[:, 0]
                receiver[:, position] = torch.multinomial(
                    torch.softmax(
                        receiver_logits[:, 0] / temperature,
                        dim=-1,
                    ),
                    1,
                )[:, 0]
        mask = torch.from_numpy(plan.valid_mask).to(device)
        amount[~mask] = 0
        gap[~mask] = 0
        receiver[~mask] = 0
        return SyntheticBatch(
            amount.cpu().numpy().astype(np.float32),
            gap.cpu().numpy().astype(np.int64),
            receiver[..., None].cpu().numpy().astype(np.int64),
            plan.valid_mask.copy(),
            plan.y_entity.copy(),
            plan.lengths.copy(),
        )


class CoFSeqGenCandidateV26(CoFSeqGenAdapter):
    baseline_definition_version = "benchmark-v2.6-candidate-v1"
    model_class = WeightedCoFSeqGenV26
    sampling_function = staticmethod(ddim_sample_v2_6)

    def fit(self, train: SequenceBatch, *, config, seed: int) -> None:
        seed_everything(seed)
        device = str(config.get("device", "cpu"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for a v2.6 CoF candidate")
        if float(config.get("discrete_mask_max", 0.7)) != 0.7:
            raise ValueError("CoF candidate fixes discrete_mask_max=0.7")
        bins = int(train.dt_bin.max()) + 1
        categories = [
            int(train.x_cat[..., channel].max()) + 1
            for channel in range(train.x_cat.shape[-1])
        ]
        denoiser = SeqDenoiser(
            train.x_num.shape[-1],
            bins,
            categories,
            d_model=int(config.get("d_model", 128)),
            n_layers=int(config.get("n_layers", 2)),
            L_max=train.x_num.shape[1],
        )
        self.model = self.model_class(
            denoiser,
            torch.as_tensor(config["tau"], dtype=torch.float32),
            float(config.get("window_width", 7)),
            float(config.get("temperature", 1)),
            float(config.get("coherence_lambda", 0)),
            categories,
            cfg_dropout=float(config.get("cfg_dropout", 0.15)),
            channel_weights=_loss_weights(config, include_label=True),
        ).to(device)
        self.device = device
        self.train = train
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config.get("lr", 1e-3)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        requested = int(config["requested_steps"])
        batch_size = min(int(config.get("batch_size", 256)), len(train.lengths))
        wall_cap = float(config["max_wall_seconds"])
        checkpoint_interval = int(config["checkpoint_interval_steps"])
        if (
            requested < 1
            or batch_size < 1
            or wall_cap <= 0
            or checkpoint_interval < 1
        ):
            raise ValueError("invalid CoF candidate schedule")
        train_hash = self._batch_hash(train)
        rng = np.random.default_rng(seed)
        self.current_step = 0
        self.elapsed_seconds = 0.0
        self.loss_history = []
        resume = config.get("resume_checkpoint")
        if resume:
            state = torch.load(
                resume,
                map_location=device,
                weights_only=False,
            )
            if (
                state["train_hash"] != train_hash
                or state["requested_steps"] != requested
            ):
                raise ValueError("CoF candidate resume provenance mismatch")
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
        self._requested_steps = requested
        self._numpy_rng = rng
        rule = _sampling_rule(config)
        self.sample_config = {
            "diffusion_steps": int(config.get("diffusion_steps", 50)),
            "guidance_scale": float(rule["guidance_scale"]),
            "feedback_discrete": bool(
                config.get("feedback_discrete", True)
            ),
            "feedback_after": float(config.get("feedback_after", 0.3)),
            "feedback_temperature": float(
                rule["discrete_feedback_temperature"]
            ),
            "final_temperature": (
                float(rule["final_categorical_temperature"])
                if rule["final_categorical_decode"]
                == "temperature_multinomial"
                else None
            ),
            "start_from_mask": bool(config.get("start_from_mask", True)),
            "sampling_chunk_size": int(
                config.get("sampling_chunk_size", 256)
            ),
        }
        progress_callback = config.get("progress_callback")
        checkpoint_callback = config.get("checkpoint_callback")
        started = time.perf_counter()
        base_elapsed = self.elapsed_seconds
        self.model.train()
        while self.current_step < requested:
            if (
                base_elapsed + time.perf_counter() - started
                >= wall_cap
            ):
                break
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
                t_frac=(self.current_step + 1) / (requested + 1),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite v2.6 CoF loss")
            loss.backward()
            optimizer.step()
            self.current_step += 1
            self.loss_history.append(float(loss.detach().cpu()))
            self.elapsed_seconds = (
                base_elapsed + time.perf_counter() - started
            )
            if (
                self.current_step % checkpoint_interval == 0
                or self.current_step == requested
            ):
                event = {
                    "step": self.current_step,
                    "loss": self.loss_history[-1],
                    "validation_metric": None,
                    "elapsed_seconds": self.elapsed_seconds,
                    "peak_gpu_memory_bytes": self._peak_memory(),
                }
                if progress_callback is not None:
                    progress_callback(event)
                if checkpoint_callback is not None:
                    checkpoint_callback(event, self)
        self.actual_training_budget = {
            "requested_steps": requested,
            "actual_steps": self.current_step,
            "max_wall_seconds": wall_cap,
            "actual_wall_seconds": self.elapsed_seconds,
            "wall_cap_reached": bool(
                self.current_step < requested
                and self.elapsed_seconds >= wall_cap
            ),
            "peak_gpu_memory_bytes": self._peak_memory(),
        }
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            self.save_training_checkpoint(checkpoint_path)

    def checkpoint_state(self):
        state = dict(super().checkpoint_state())
        state["schema_version"] = "benchmark-v2.6-cof-candidate-v1"
        state["v2_6_channel_loss_weights"] = dict(
            self.model.v2_6_channel_weights
        )
        return state

    def sample(self, plan: SamplingPlan, *, seed: int) -> SyntheticBatch:
        seed_everything(seed)
        count, length = plan.valid_mask.shape
        numerical = np.zeros(
            (count, length, self.train.x_num.shape[-1]),
            dtype=np.float32,
        )
        gaps = np.zeros((count, length), dtype=np.int64)
        categories = np.zeros(
            (count, length, self.train.x_cat.shape[-1]),
            dtype=np.int64,
        )
        chunk_size = self.sample_config["sampling_chunk_size"]
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            size = stop - start
            gap_input = torch.zeros(
                size,
                length,
                dtype=torch.long,
                device=self.device,
            )
            category_input = torch.zeros(
                size,
                length,
                self.train.x_cat.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            mask = torch.from_numpy(plan.valid_mask[start:stop]).to(
                self.device
            )
            amount, gap, category = self.sampling_function(
                self.model,
                gap_input,
                category_input,
                self.train.x_num.shape[-1],
                steps=self.sample_config["diffusion_steps"],
                device=self.device,
                y_cond=torch.from_numpy(
                    plan.y_entity[start:stop]
                ).to(self.device),
                guidance_scale=self.sample_config["guidance_scale"],
                feedback_discrete=self.sample_config[
                    "feedback_discrete"
                ],
                feedback_after=self.sample_config["feedback_after"],
                feedback_temperature=self.sample_config[
                    "feedback_temperature"
                ],
                final_temperature=self.sample_config[
                    "final_temperature"
                ],
                start_from_mask=self.sample_config["start_from_mask"],
                valid_mask=mask,
            )
            numerical[start:stop] = amount.cpu().numpy().astype(np.float32)
            gaps[start:stop] = gap.cpu().numpy().astype(np.int64)
            categories[start:stop] = torch.stack(
                category,
                dim=-1,
            ).cpu().numpy().astype(np.int64)
        return SyntheticBatch(
            numerical,
            gaps,
            categories,
            plan.valid_mask.copy(),
            plan.y_entity.copy(),
            plan.lengths.copy(),
        )


BACKEND_REGISTRY_V2_6 = {
    "ctgan_separate_class": ConditionalCTGANCandidateV26,
    "tvae_separate_class": ConditionalTVAECandidateV26,
    "neural_sequence": NeuralSequenceCandidateV26,
    "cof_seqgen": CoFSeqGenCandidateV26,
}
