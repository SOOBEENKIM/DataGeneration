from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN

from benchmarks.types import SequenceBatch, SyntheticBatch
from .checkpointable_tabular_v2_5 import CheckpointableCTGAN
from .sampling_plan import SamplingPlan


def _by_label(value: Mapping[Any, Any], label: int) -> Any:
    if label in value:
        return value[label]
    return value[str(label)]


def _frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(frame.to_numpy()).tobytes())
    digest.update("|".join(frame.columns).encode())
    return digest.hexdigest()


class ConditionalCTGAN:
    name = "conditional_ctgan"
    model_class = CTGAN
    checkpointable_model_class = CheckpointableCTGAN
    baseline_definition_version = "benchmark-v2.5"

    def _model_kwargs(
        self,
        config: Mapping[str, Any],
        *,
        epochs: int,
    ) -> dict[str, Any]:
        return {
            "epochs": epochs,
            "batch_size": int(config.get("batch_size", 500)),
            "generator_lr": float(config.get("generator_lr", 2e-4)),
            "generator_decay": float(
                config.get("generator_decay", 1e-6)
            ),
            "discriminator_lr": float(
                config.get("discriminator_lr", 2e-4)
            ),
            "discriminator_decay": float(
                config.get("discriminator_decay", 1e-6)
            ),
            "verbose": False,
            "cuda": bool(config.get("cuda", False)),
        }

    def _new_model(
        self,
        config: Mapping[str, Any],
        *,
        epochs: int,
        checkpointable: bool,
    ):
        model_type = (
            self.checkpointable_model_class
            if checkpointable
            else self.model_class
        )
        return model_type(**self._model_kwargs(config, epochs=epochs))

    def _fit_checkpointable_model(
        self,
        model,
        class_frame: pd.DataFrame,
        *,
        config: Mapping[str, Any],
        seed: int,
        label: int,
        requested_steps: int,
        max_wall_seconds: float,
        checkpoint_interval: int,
        progress_callback,
        checkpoint_callback,
    ) -> Mapping[str, Any]:
        del config, seed, label
        return model.fit_steps(
            class_frame,
            discrete_columns=["dt_bin", "receiver"],
            requested_steps=requested_steps,
            max_wall_seconds=max_wall_seconds,
            checkpoint_interval=checkpoint_interval,
            progress_callback=progress_callback,
            checkpoint_callback=checkpoint_callback,
        )

    @staticmethod
    def _validate_budget(
        config: Mapping[str, Any],
    ) -> tuple[dict[int, int], dict[int, float], float] | None:
        if "max_wall_seconds_total" not in config:
            return None
        total = float(config["max_wall_seconds_total"])
        wall = {
            label: float(
                _by_label(config["max_wall_seconds_per_class"], label)
            )
            for label in (0, 1)
        }
        steps = {
            label: int(
                _by_label(config["requested_steps_per_class"], label)
            )
            for label in (0, 1)
        }
        requested_total = int(config["requested_steps_total"])
        if total <= 0 or any(value <= 0 for value in wall.values()):
            raise ValueError("CTGAN/TVAE wall budgets must be positive")
        if not np.isclose(sum(wall.values()), total):
            raise ValueError(
                "two class-model wall budgets must sum to one baseline budget"
            )
        if not np.isclose(wall[0], total / 2) or not np.isclose(
            wall[1],
            total / 2,
        ):
            raise ValueError("each class model must receive half the budget")
        if sum(steps.values()) != requested_total or any(
            value < 1 for value in steps.values()
        ):
            raise ValueError(
                "class requested steps must be positive and sum to total"
            )
        return steps, wall, total

    def fit(self, train: SequenceBatch, *, config: Mapping[str, Any], seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        budget = self._validate_budget(config)
        resume_path = config.get("resume_checkpoint")
        if resume_path:
            restored = torch.load(
                resume_path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(restored, type(self)):
                raise TypeError("resume checkpoint has the wrong adapter type")
            self.__dict__.update(restored.__dict__)
        else:
            self.models = {}
            self.actual_training_budget = {}
            self.training_data_hashes_by_class = {}
        self.d_num, self.d_cat = train.x_num.shape[-1], train.x_cat.shape[-1]
        rows_y = np.repeat(train.y_entity, train.lengths)
        frame = pd.DataFrame({
            "amount_log": train.x_num[train.valid_mask, 0],
            "dt_bin": train.dt_bin[train.valid_mask],
            "receiver": train.x_cat[train.valid_mask, 0],
        })
        checkpointable = budget is not None
        checkpoint_interval = int(config.get("checkpoint_interval_steps", 100))
        if checkpointable and checkpoint_interval < 1:
            raise ValueError("checkpoint interval must be positive")
        for label in (0, 1):
            class_frame = frame.loc[rows_y == label]
            current_hash = _frame_hash(class_frame)
            existing_hash = self.training_data_hashes_by_class.get(label)
            if existing_hash is not None and existing_hash != current_hash:
                raise ValueError("resume train rows differ from checkpoint")
            self.training_data_hashes_by_class[label] = current_hash
            model = self.models.get(label)
            if model is None:
                np.random.seed(seed + label)
                torch.manual_seed(seed + label)
                model = self._new_model(
                    config,
                    epochs=int(config.get("epochs", 1)),
                    checkpointable=checkpointable,
                )
                if hasattr(model, "set_random_state"):
                    model.set_random_state(seed + label)
                self.models[label] = model
            elif checkpointable and not isinstance(
                model,
                self.checkpointable_model_class,
            ):
                raise TypeError("resume model is not checkpointable")
            if not checkpointable:
                model.fit(
                    class_frame,
                    discrete_columns=["dt_bin", "receiver"],
                )
                continue
            assert budget is not None
            steps, class_wall, total_wall = budget
            adapter_progress = config.get("progress_callback")
            adapter_checkpoint = config.get("checkpoint_callback")

            def event_with_label(event):
                class_step = event["step"]
                baseline_step = (
                    class_step
                    if label == 0
                    else steps[0] + class_step
                )
                return {
                    **event,
                    "step": baseline_step,
                    "class_step": class_step,
                    "class_label": label,
                }

            def progress(event):
                if adapter_progress is not None:
                    adapter_progress(event_with_label(event))

            def checkpoint(event):
                if adapter_checkpoint is not None:
                    adapter_checkpoint(
                        event_with_label(event),
                        self,
                    )

            result = self._fit_checkpointable_model(
                model,
                class_frame,
                config=config,
                seed=seed,
                label=label,
                requested_steps=steps[label],
                max_wall_seconds=class_wall[label],
                checkpoint_interval=checkpoint_interval,
                progress_callback=progress,
                checkpoint_callback=checkpoint,
            )
            self.actual_training_budget[label] = result
        if budget is not None:
            steps, class_wall, total_wall = budget
            self.budget_allocation = {
                "requested_steps_total": sum(steps.values()),
                "requested_steps_per_class": steps,
                "max_wall_seconds_total": total_wall,
                "max_wall_seconds_per_class": class_wall,
                "actual": self.actual_training_budget,
            }

    def save_training_checkpoint(self, path: str | Path) -> None:
        torch.save(self, path)

    def sample(self, plan: SamplingPlan, *, seed: int) -> SyntheticBatch:
        np.random.seed(seed)
        n, max_l = plan.valid_mask.shape
        x_num, dt_bin = np.zeros((n, max_l, self.d_num), np.float32), np.zeros((n, max_l), np.int64)
        x_cat = np.zeros((n, max_l, self.d_cat), np.int64)
        for label in (0, 1):
            positions = np.argwhere(plan.valid_mask & (plan.y_entity[:, None] == label))
            rows = self.models[label].sample(len(positions))
            x_num[positions[:, 0], positions[:, 1], 0] = rows["amount_log"].to_numpy(np.float32)
            dt_bin[positions[:, 0], positions[:, 1]] = rows["dt_bin"].to_numpy(np.int64)
            x_cat[positions[:, 0], positions[:, 1], 0] = rows["receiver"].to_numpy(np.int64)
        return SyntheticBatch(x_num, dt_bin, x_cat, plan.valid_mask.copy(), plan.y_entity.copy(), plan.lengths.copy())
