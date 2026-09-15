from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from benchmarks.types import SequenceBatch, SyntheticBatch
from .class_conditional_markov import _draw_nonrepeat, _fit_chain
from .sampling_plan import SamplingPlan


def _normalized_counts(values: np.ndarray, classes: int, alpha: float) -> np.ndarray:
    counts = np.full(classes, alpha, dtype=float)
    np.add.at(counts, values, 1)
    return counts / counts.sum()


def _runs(states: np.ndarray) -> list[tuple[int, int]]:
    if len(states) == 0:
        return []
    output: list[tuple[int, int]] = []
    start = 0
    for position in range(1, len(states)):
        if states[position] != states[start]:
            output.append((int(states[start]), position - start))
            start = position
    output.append((int(states[start]), len(states) - start))
    return output


def _deadline(config: Mapping[str, Any]) -> tuple[float, float]:
    max_wall_seconds = float(config.get("max_wall_seconds", 7200))
    if max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be positive")
    return time.perf_counter(), max_wall_seconds


def _check_deadline(started: float, maximum: float, name: str) -> None:
    if time.perf_counter() - started >= maximum:
        raise TimeoutError(f"{name} reached max_wall_seconds={maximum}")


class ClassConditionalPlugInHMM:
    """Two-state class-conditional plug-in state model for benchmark v2.5.

    State assignments are inferred deterministically from the training split
    using a preregistered short-gap quantile. No EM, latent-state restart, or
    test-data fitting occurs. This plug-in comparator is not represented as a
    standard latent maximum-likelihood HMM.
    """

    name = "plug_in_hmm"
    baseline_definition_version = "benchmark-v2.5"

    def fit(
        self,
        train: SequenceBatch,
        *,
        config: Mapping[str, Any],
        seed: int,
    ) -> None:
        del seed
        started, max_wall_seconds = _deadline(config)
        if train.x_num.shape[-1] != 1 or train.x_cat.shape[-1] != 1:
            raise ValueError("benchmark v2.5 adapters require one numerical and one categorical channel")
        alpha = float(config.get("laplace_alpha", 1.0))
        quantile = float(config.get("short_gap_quantile", 0.5))
        if alpha <= 0 or not 0 < quantile < 1:
            raise ValueError("laplace_alpha must be positive and short_gap_quantile must be in (0, 1)")
        valid_gaps = train.dt_bin[train.valid_mask]
        self.short_gap_bin_threshold = int(
            np.quantile(valid_gaps, quantile, method="inverted_cdf")
        )
        self.gap_classes = int(valid_gaps.max()) + 1
        self.cat_classes = int(
            train.x_cat[..., 0][train.valid_mask].max()
        ) + 1
        self.models: dict[int, dict[str, Any]] = {}
        for label in (0, 1):
            _check_deadline(
                started,
                max_wall_seconds,
                self.name,
            )
            indices = np.flatnonzero(train.y_entity == label)
            state_sequences: list[np.ndarray] = []
            gaps_by_state = [[], []]
            repeats_by_state = [[], []]
            amounts_by_state = [[], []]
            category_values = []
            for index in indices:
                _check_deadline(
                    started,
                    max_wall_seconds,
                    self.name,
                )
                length = int(train.lengths[index])
                gap = train.dt_bin[index, :length]
                category = train.x_cat[index, :length, 0]
                amount = train.x_num[index, :length, 0]
                state = (gap <= self.short_gap_bin_threshold).astype(np.int64)
                repeat = np.zeros(length, dtype=np.int64)
                if length > 1:
                    repeat[1:] = category[1:] == category[:-1]
                state_sequences.append(state)
                category_values.append(category)
                for hidden in (0, 1):
                    chosen = state == hidden
                    gaps_by_state[hidden].append(gap[chosen])
                    repeats_by_state[hidden].append(repeat[chosen])
                    amounts_by_state[hidden].append(amount[chosen])
            if not state_sequences:
                raise ValueError("both entity labels must occur in the training split")
            initial, transition = _fit_chain(state_sequences, 2, alpha)
            emission = []
            for hidden in (0, 1):
                gaps = np.concatenate(gaps_by_state[hidden])
                repeats = np.concatenate(repeats_by_state[hidden])
                amounts = np.concatenate(amounts_by_state[hidden])
                # Quantile-derived states can leave a state empty only in
                # degenerate data; fail closed instead of fabricating an HMM.
                if not len(gaps) or not len(amounts):
                    raise ValueError("train-only hidden-state inference produced an empty state")
                emission.append(
                    {
                        "gap": _normalized_counts(
                            gaps,
                            self.gap_classes,
                            alpha,
                        ),
                        "repeat": _normalized_counts(repeats, 2, alpha),
                        "amounts": amounts.astype(np.float32, copy=True),
                    }
                )
            categories = np.concatenate(category_values)
            self.models[label] = {
                "initial": initial,
                "transition": transition,
                "emission": emission,
                "category": _normalized_counts(
                    categories,
                    self.cat_classes,
                    alpha,
                ),
            }
        self.fit_metadata = {
            "state_fit_split": "train",
            "short_gap_quantile": quantile,
            "short_gap_bin_threshold": self.short_gap_bin_threshold,
            "hidden_states": 2,
            "state_inference": "deterministic_train_only",
            "uses_em": False,
            "uses_latent_state_restarts": False,
            "uses_test_data": False,
            "max_wall_seconds": max_wall_seconds,
        }
        self.actual_training_budget = {
            "max_wall_seconds": max_wall_seconds,
            "actual_wall_seconds": time.perf_counter() - started,
            "wall_cap_reached": False,
        }

    def _sample_states(
        self,
        rng: np.random.Generator,
        model: Mapping[str, Any],
        length: int,
    ) -> np.ndarray:
        states = np.empty(length, dtype=np.int64)
        states[0] = rng.choice(2, p=model["initial"])
        for position in range(1, length):
            states[position] = rng.choice(
                2,
                p=model["transition"][states[position - 1]],
            )
        return states

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
            states = self._sample_states(rng, model, length)
            x_cat[index, 0, 0] = rng.choice(
                self.cat_classes,
                p=model["category"],
            )
            for position, hidden_value in enumerate(states):
                hidden = int(hidden_value)
                emission = model["emission"][hidden]
                dt_bin[index, position] = rng.choice(
                    self.gap_classes,
                    p=emission["gap"],
                )
                x_num[index, position, 0] = rng.choice(
                    emission["amounts"],
                )
                if position:
                    repeat = bool(rng.choice(2, p=emission["repeat"]))
                    if repeat:
                        x_cat[index, position, 0] = x_cat[
                            index,
                            position - 1,
                            0,
                        ]
                    else:
                        x_cat[index, position, 0] = _draw_nonrepeat(
                            rng,
                            model["category"],
                            int(x_cat[index, position - 1, 0]),
                        )
        return SyntheticBatch(
            x_num,
            dt_bin,
            x_cat,
            plan.valid_mask.copy(),
            plan.y_entity.copy(),
            plan.lengths.copy(),
        )


class ClassConditionalPlugInHSMM(ClassConditionalPlugInHMM):
    """Explicit-duration counterpart of the plug-in state comparator."""

    name = "plug_in_hsmm"

    def fit(
        self,
        train: SequenceBatch,
        *,
        config: Mapping[str, Any],
        seed: int,
    ) -> None:
        super().fit(train, config=config, seed=seed)
        started = time.perf_counter()
        max_wall_seconds = float(config.get("max_wall_seconds", 7200))
        parent_seconds = float(
            self.actual_training_budget["actual_wall_seconds"]
        )
        remaining_seconds = max_wall_seconds - parent_seconds
        if remaining_seconds <= 0:
            raise TimeoutError(
                f"{self.name} reached max_wall_seconds={max_wall_seconds}"
            )
        alpha = float(config.get("duration_laplace_alpha", 1.0))
        duration_max = int(config.get("duration_max", 128))
        if alpha <= 0 or duration_max < 1:
            raise ValueError("duration smoothing and maximum must be positive")
        self.duration_max = duration_max
        for label in (0, 1):
            _check_deadline(
                started,
                remaining_seconds,
                self.name,
            )
            indices = np.flatnonzero(train.y_entity == label)
            segment_sequences: list[np.ndarray] = []
            duration_values = [[], []]
            for index in indices:
                _check_deadline(
                    started,
                    remaining_seconds,
                    self.name,
                )
                length = int(train.lengths[index])
                gap = train.dt_bin[index, :length]
                states = (gap <= self.short_gap_bin_threshold).astype(np.int64)
                runs = _runs(states)
                segment_sequences.append(
                    np.asarray([state for state, _ in runs], dtype=np.int64)
                )
                for state, duration in runs:
                    duration_values[state].append(min(duration, duration_max))
            segment_initial, segment_transition = _fit_chain(
                segment_sequences,
                2,
                alpha,
            )
            durations = []
            for hidden in (0, 1):
                values = np.asarray(duration_values[hidden], dtype=np.int64)
                if not len(values):
                    raise ValueError("train-only duration fit produced an empty state")
                # Index zero represents duration one.
                durations.append(
                    _normalized_counts(values - 1, duration_max, alpha)
                )
            self.models[label]["segment_initial"] = segment_initial
            self.models[label]["segment_transition"] = segment_transition
            self.models[label]["duration"] = durations
        self.fit_metadata = {
            **self.fit_metadata,
            "duration_fit_split": "train",
            "duration_max": duration_max,
            "duration_family": "smoothed_empirical_discrete",
            "max_wall_seconds": max_wall_seconds,
        }
        self.actual_training_budget = {
            "max_wall_seconds": max_wall_seconds,
            "actual_wall_seconds": (
                parent_seconds + time.perf_counter() - started
            ),
            "wall_cap_reached": False,
        }

    def _sample_states(
        self,
        rng: np.random.Generator,
        model: Mapping[str, Any],
        length: int,
    ) -> np.ndarray:
        output: list[int] = []
        hidden = int(rng.choice(2, p=model["segment_initial"]))
        while len(output) < length:
            duration = int(
                rng.choice(
                    self.duration_max,
                    p=model["duration"][hidden],
                )
            ) + 1
            output.extend([hidden] * min(duration, length - len(output)))
            hidden = int(
                rng.choice(
                    2,
                    p=model["segment_transition"][hidden],
                )
            )
        return np.asarray(output, dtype=np.int64)
