from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch
from .sampling_plan import SamplingPlan


def _budget_start(config: Mapping[str, Any]) -> tuple[float, float]:
    maximum = float(config.get("max_wall_seconds", 7200))
    if maximum <= 0:
        raise ValueError("max_wall_seconds must be positive")
    return time.perf_counter(), maximum


def _budget_check(started: float, maximum: float, name: str) -> None:
    if time.perf_counter() - started >= maximum:
        raise TimeoutError(f"{name} reached max_wall_seconds={maximum}")


def _fit_chain(sequences: list[np.ndarray], classes: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if not sequences:
        raise ValueError("both entity labels must occur in the training split")
    if classes < 1 or alpha <= 0:
        raise ValueError("classes and laplace_alpha must be positive")
    initial = np.full(classes, alpha)
    transitions = np.full((classes, classes), alpha)
    for sequence in sequences:
        if len(sequence) == 0:
            raise ValueError("zero-length sequences are outside the v2 contract")
        initial[sequence[0]] += 1
        if len(sequence) > 1:
            np.add.at(transitions, (sequence[:-1], sequence[1:]), 1)
    return initial / initial.sum(), transitions / transitions.sum(1, keepdims=True)


def _draw_nonrepeat(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    previous: int,
) -> int:
    """Draw a category different from ``previous`` whenever support allows."""
    adjusted = probabilities.copy()
    if len(adjusted) > 1:
        adjusted[previous] = 0.0
        total = adjusted.sum()
        if total > 0:
            adjusted /= total
            return int(rng.choice(len(adjusted), p=adjusted))
    return int(rng.choice(len(probabilities), p=probabilities))


class ClassConditionalMarkov:
    name = "markov_independent"
    baseline_definition_version = "benchmark-v2.5"

    def fit(self, train: SequenceBatch, *, config: Mapping[str, Any], seed: int) -> None:
        del seed
        started, maximum = _budget_start(config)
        alpha = float(config.get("laplace_alpha", 1.0))
        if train.x_num.shape[-1] != 1 or train.x_cat.shape[-1] != 1:
            raise ValueError("benchmark v2.5 adapters require one numerical and one categorical channel")
        self.train, self.models = train, {}
        self.gap_classes = int(train.dt_bin[train.valid_mask].max()) + 1
        self.cat_classes = int(train.x_cat[..., 0][train.valid_mask].max()) + 1
        for label in (0, 1):
            _budget_check(started, maximum, self.name)
            indices = np.flatnonzero(train.y_entity == label)
            gap_sequences = [train.dt_bin[i, : train.lengths[i]] for i in indices]
            cat_sequences = [train.x_cat[i, : train.lengths[i], 0] for i in indices]
            self.models[label] = {
                "gap": _fit_chain(gap_sequences, self.gap_classes, alpha),
                "cat": _fit_chain(cat_sequences, self.cat_classes, alpha),
                "amounts": train.x_num[(train.y_entity[:, None] == label) & train.valid_mask, 0],
            }
        self.actual_training_budget = {
            "max_wall_seconds": maximum,
            "actual_wall_seconds": time.perf_counter() - started,
            "wall_cap_reached": False,
        }

    def sample(self, plan: SamplingPlan, *, seed: int) -> SyntheticBatch:
        rng = np.random.default_rng(seed)
        n, max_l = plan.valid_mask.shape
        x_num, dt_bin = np.zeros((n, max_l, 1), np.float32), np.zeros((n, max_l), np.int64)
        x_cat = np.zeros((n, max_l, 1), np.int64)
        for i, (label, length) in enumerate(zip(plan.y_entity, plan.lengths)):
            model = self.models[int(label)]
            for target, key in ((dt_bin, "gap"), (x_cat[..., 0], "cat")):
                initial, transitions = model[key]
                target[i, 0] = rng.choice(len(initial), p=initial)
                for t in range(1, int(length)):
                    target[i, t] = rng.choice(len(initial), p=transitions[target[i, t - 1]])
            x_num[i, :length, 0] = rng.choice(model["amounts"], int(length), replace=True)
        return SyntheticBatch(x_num, dt_bin, x_cat, plan.valid_mask.copy(), plan.y_entity.copy(), plan.lengths.copy())


class JointObservedMarkov:
    """Class-conditional Markov chain over ``(dt_bin, receiver_repeat)``.

    The repeat indicator is used instead of the full receiver-category Cartesian
    product. This is the preregistered v2.5 joint observed state and directly
    represents cross-channel alignment while keeping a finite, auditable state
    space. A non-repeat transition draws from the train-only class-conditional
    receiver marginal with the previous category removed.
    """

    name = "markov_joint"
    baseline_definition_version = "benchmark-v2.5"

    def fit(
        self,
        train: SequenceBatch,
        *,
        config: Mapping[str, Any],
        seed: int,
    ) -> None:
        del seed
        started, maximum = _budget_start(config)
        if train.x_num.shape[-1] != 1 or train.x_cat.shape[-1] != 1:
            raise ValueError("benchmark v2.5 adapters require one numerical and one categorical channel")
        alpha = float(config.get("laplace_alpha", 1.0))
        if alpha <= 0:
            raise ValueError("laplace_alpha must be positive")
        self.gap_classes = int(train.dt_bin[train.valid_mask].max()) + 1
        self.cat_classes = int(train.x_cat[..., 0][train.valid_mask].max()) + 1
        self.joint_classes = self.gap_classes * 2
        self.models: dict[int, dict[str, Any]] = {}
        for label in (0, 1):
            _budget_check(started, maximum, self.name)
            indices = np.flatnonzero(train.y_entity == label)
            joint_sequences = []
            category_counts = np.full(self.cat_classes, alpha, dtype=float)
            amount_values = []
            for index in indices:
                _budget_check(started, maximum, self.name)
                length = int(train.lengths[index])
                gap = train.dt_bin[index, :length]
                category = train.x_cat[index, :length, 0]
                repeat = np.zeros(length, dtype=np.int64)
                if length > 1:
                    repeat[1:] = category[1:] == category[:-1]
                joint_sequences.append(gap * 2 + repeat)
                np.add.at(category_counts, category, 1)
                amount_values.append(train.x_num[index, :length, 0])
            if not amount_values:
                raise ValueError("both entity labels must occur in the training split")
            self.models[label] = {
                "joint": _fit_chain(joint_sequences, self.joint_classes, alpha),
                "category": category_counts / category_counts.sum(),
                "amounts": np.concatenate(amount_values),
            }
        self.actual_training_budget = {
            "max_wall_seconds": maximum,
            "actual_wall_seconds": time.perf_counter() - started,
            "wall_cap_reached": False,
        }

    def sample(self, plan: SamplingPlan, *, seed: int) -> SyntheticBatch:
        rng = np.random.default_rng(seed)
        n, max_length = plan.valid_mask.shape
        x_num = np.zeros((n, max_length, 1), dtype=np.float32)
        dt_bin = np.zeros((n, max_length), dtype=np.int64)
        x_cat = np.zeros((n, max_length, 1), dtype=np.int64)
        for index, (label_value, length_value) in enumerate(
            zip(plan.y_entity, plan.lengths)
        ):
            label, length = int(label_value), int(length_value)
            model = self.models[label]
            initial, transitions = model["joint"]
            joint = np.empty(length, dtype=np.int64)
            joint[0] = rng.choice(self.joint_classes, p=initial)
            # The first position has no predecessor, so its repeat bit is not
            # interpreted. Later states have an exact repeat/non-repeat meaning.
            x_cat[index, 0, 0] = rng.choice(
                self.cat_classes,
                p=model["category"],
            )
            for position in range(1, length):
                joint[position] = rng.choice(
                    self.joint_classes,
                    p=transitions[joint[position - 1]],
                )
                if joint[position] % 2:
                    x_cat[index, position, 0] = x_cat[index, position - 1, 0]
                else:
                    x_cat[index, position, 0] = _draw_nonrepeat(
                        rng,
                        model["category"],
                        int(x_cat[index, position - 1, 0]),
                    )
            dt_bin[index, :length] = joint // 2
            x_num[index, :length, 0] = rng.choice(
                model["amounts"],
                length,
                replace=True,
            )
        return SyntheticBatch(
            x_num,
            dt_bin,
            x_cat,
            plan.valid_mask.copy(),
            plan.y_entity.copy(),
            plan.lengths.copy(),
        )
