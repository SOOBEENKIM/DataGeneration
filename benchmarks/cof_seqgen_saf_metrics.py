"""Train-only metric definitions and validity audit for CoFSeqGen-SAF.

The audit is deliberately independent of model outputs. It calibrates grouping,
scales, and null margins from entity-disjoint subsets of the training split and
checks metric sensitivity with controlled, marginal-preserving corruptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.linear_model import LogisticRegression, Ridge


REQUIRED_COLUMNS = (
    "entity_id",
    "event_index",
    "gap",
    "receiver_or_mark",
    "amount_or_numeric_value",
)


class SAFMetricError(ValueError):
    pass


@dataclass(frozen=True)
class MetricState:
    """All learned metric state, fitted only from a training reference."""

    gap_bin_edges: Tuple[float, ...]
    mark_groups: Tuple[str, ...]
    gap_scale: float
    amount_scale: float
    length_scale: float
    fit_split: str = "train"
    fit_entity_count: int = 0

    def __post_init__(self) -> None:
        if self.fit_split != "train":
            raise SAFMetricError("metric state must be fitted on train only")
        if self.fit_entity_count < 2:
            raise SAFMetricError("at least two train entities are required")
        if self.gap_scale <= 0 or self.amount_scale <= 0 or self.length_scale <= 0:
            raise SAFMetricError("metric scales must be positive")


METRIC_REGISTRY: Mapping[str, Mapping[str, object]] = {
    "gap_conditioned_mark_tv": {
        "family": "temporal_relational",
        "primary": False,
        "direction": "lower",
        "targets": "absolute_mark_distribution_by_gap",
    },
    "transition_conditioned_mark_tv": {
        "family": "temporal_relational",
        "primary": True,
        "direction": "lower",
        "targets": "gap_and_previous_mark_to_current_mark_routing",
    },
    "short_gap_repeat_curve_l1": {
        "family": "temporal_relational",
        "primary": True,
        "direction": "lower",
        "targets": "gap_to_mark_routing",
    },
    "gap_repeat_mi_error": {
        "family": "temporal_relational",
        "primary": True,
        "direction": "lower",
        "targets": "gap_to_mark_routing",
    },
    "gap_lag1_acf_error": {
        "family": "temporal_relational",
        "primary": False,
        "direction": "lower",
        "targets": "temporal_coherence",
    },
    "amount_lag1_acf_error": {
        "family": "temporal_relational",
        "primary": False,
        "direction": "lower",
        "targets": "temporal_coherence",
    },
    "gap_ks": {"family": "marginal", "primary": False, "direction": "lower"},
    "gap_scaled_w1": {"family": "marginal", "primary": False, "direction": "lower"},
    "gap_zero_rate_error": {"family": "marginal", "primary": False, "direction": "lower"},
    "mark_sparse_tv": {"family": "marginal", "primary": False, "direction": "lower"},
    "amount_ks": {"family": "marginal", "primary": False, "direction": "lower"},
    "amount_scaled_w1": {"family": "marginal", "primary": False, "direction": "lower"},
    "length_scaled_w1": {"family": "structural", "primary": False, "direction": "lower"},
    "next_gap_tstr_mae_ratio": {
        "family": "utility",
        "primary": False,
        "direction": "lower",
        "targets": "next_gap_prediction",
    },
    "next_repeat_tstr_brier_ratio": {
        "family": "utility",
        "primary": False,
        "direction": "lower",
        "targets": "next_mark_repeat_prediction",
    },
    "trajectory_exact_match_rate": {
        "family": "privacy",
        "primary": False,
        "direction": "lower",
    },
    "transition_ngram_exposure_rate": {
        "family": "privacy",
        "primary": False,
        "direction": "lower",
    },
}


def _validate(events: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(events.columns)
    if missing:
        raise SAFMetricError(f"event frame missing columns: {sorted(missing)}")
    if events.empty:
        raise SAFMetricError("event frame is empty")
    frame = events.copy()
    frame["event_index"] = pd.to_numeric(frame["event_index"], errors="raise").astype(int)
    frame["gap"] = pd.to_numeric(frame["gap"], errors="coerce")
    frame["amount_or_numeric_value"] = pd.to_numeric(
        frame["amount_or_numeric_value"], errors="coerce"
    )
    frame = frame.sort_values(["entity_id", "event_index"], kind="mergesort")
    expected_index = frame.groupby("entity_id", sort=False).cumcount().to_numpy()
    if not np.array_equal(frame["event_index"].to_numpy(), expected_index):
        raise SAFMetricError("event_index must be contiguous within entity")
    gaps = frame["gap"].to_numpy(float)
    first = expected_index == 0
    later = ~first
    if (
        not np.isnan(gaps[first]).all()
        or not np.isfinite(gaps[later]).all()
        or (gaps[later] < 0).any()
    ):
        raise SAFMetricError("first gap must be missing and later gaps finite nonnegative")
    return frame.reset_index(drop=True)


def _stable_mark(value: object) -> str:
    return "<MISSING>" if pd.isna(value) else str(value)


def _robust_scale(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return float(max(q75 - q25, np.std(values), 1e-8))


def fit_metric_state(
    train_events: pd.DataFrame,
    *,
    gap_bins: int = 5,
    max_mark_groups: int = 20,
) -> MetricState:
    frame = _validate(train_events)
    if gap_bins < 2 or max_mark_groups < 2:
        raise SAFMetricError("gap_bins and max_mark_groups must be at least two")
    gaps = frame["gap"].dropna().to_numpy(float)
    if gaps.size == 0:
        raise SAFMetricError("training data has no modeled gaps")
    internal = np.unique(np.quantile(gaps, np.linspace(0, 1, gap_bins + 1)[1:-1]))
    mark_counts = frame["receiver_or_mark"].map(_stable_mark).value_counts()
    marks = tuple(mark_counts.sort_values(ascending=False, kind="stable").index[:max_mark_groups])
    lengths = frame.groupby("entity_id").size().to_numpy(float)
    return MetricState(
        gap_bin_edges=tuple(float(value) for value in internal),
        mark_groups=marks,
        gap_scale=_robust_scale(gaps),
        amount_scale=_robust_scale(frame["amount_or_numeric_value"].to_numpy(float)),
        length_scale=_robust_scale(lengths),
        fit_entity_count=int(frame["entity_id"].nunique()),
    )


def _gap_bin(values: np.ndarray, state: MetricState) -> np.ndarray:
    return np.searchsorted(np.asarray(state.gap_bin_edges), values, side="right")


def _group_marks(series: pd.Series, state: MetricState) -> np.ndarray:
    allowed = set(state.mark_groups)
    return np.asarray([
        value if value in allowed else "<OTHER>"
        for value in series.map(_stable_mark)
    ], dtype=object)


def _tv(left: Sequence[object], right: Sequence[object]) -> float:
    if len(left) == 0 and len(right) == 0:
        return 0.0
    if len(left) == 0 or len(right) == 0:
        return 1.0
    left_series = pd.Series(list(left), dtype="object").value_counts(normalize=True)
    right_series = pd.Series(list(right), dtype="object").value_counts(normalize=True)
    support = left_series.index.union(right_series.index)
    return float(0.5 * np.abs(left_series.reindex(support, fill_value=0) - right_series.reindex(support, fill_value=0)).sum())


def _conditional_mark_tv(reference: pd.DataFrame, candidate: pd.DataFrame, state: MetricState) -> float:
    ref = reference[reference["gap"].notna()].copy()
    can = candidate[candidate["gap"].notna()].copy()
    ref["bin"] = _gap_bin(ref["gap"].to_numpy(float), state)
    can["bin"] = _gap_bin(can["gap"].to_numpy(float), state)
    ref["mark_group"] = _group_marks(ref["receiver_or_mark"], state)
    can["mark_group"] = _group_marks(can["receiver_or_mark"], state)
    total = max(len(ref), 1)
    error = 0.0
    all_bins = range(len(state.gap_bin_edges) + 1)
    for bin_id in all_bins:
        ref_bin = ref.loc[ref["bin"] == bin_id, "mark_group"]
        can_bin = can.loc[can["bin"] == bin_id, "mark_group"]
        if len(ref_bin) == 0:
            continue
        error += len(ref_bin) / total * _tv(ref_bin, can_bin)
    return float(error)


def _mark_transition_frame(frame: pd.DataFrame, state: MetricState) -> pd.DataFrame:
    rows = frame[frame["gap"].notna()].copy()
    previous = frame.groupby("entity_id", sort=False)["receiver_or_mark"].shift()
    rows["gap_bin"] = _gap_bin(rows["gap"].to_numpy(float), state)
    rows["previous_mark_group"] = _group_marks(
        previous.loc[rows.index], state
    )
    rows["current_mark_group"] = _group_marks(
        rows["receiver_or_mark"], state
    )
    return rows


def _transition_conditioned_mark_tv(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    state: MetricState,
) -> float:
    """Weighted TV of p(mark_t | mark_{t-1}, gap_bin_t).

    Conditioning on the previous mark makes this endpoint sensitive to
    copy/repeat dependence while preserving the original train-only gap bins
    and sparse mark grouping.
    """

    ref = _mark_transition_frame(reference, state)
    can = _mark_transition_frame(candidate, state)
    total = max(len(ref), 1)
    error = 0.0
    for (gap_bin, previous_group), ref_cell in ref.groupby(
        ["gap_bin", "previous_mark_group"],
        sort=False,
    ):
        can_cell = can.loc[
            (can["gap_bin"] == gap_bin)
            & (can["previous_mark_group"] == previous_group),
            "current_mark_group",
        ]
        error += len(ref_cell) / total * _tv(
            ref_cell["current_mark_group"],
            can_cell,
        )
    return float(error)


def _transition_frame(frame: pd.DataFrame, state: MetricState) -> pd.DataFrame:
    rows = frame[frame["gap"].notna()].copy()
    previous = frame.groupby("entity_id", sort=False)["receiver_or_mark"].shift()
    rows["repeat"] = (
        rows["receiver_or_mark"].map(_stable_mark).to_numpy()
        == previous.loc[rows.index].map(_stable_mark).to_numpy()
    ).astype(int)
    rows["gap_bin"] = _gap_bin(rows["gap"].to_numpy(float), state)
    return rows


def _repeat_curve_error(reference: pd.DataFrame, candidate: pd.DataFrame, state: MetricState) -> float:
    ref = _transition_frame(reference, state)
    can = _transition_frame(candidate, state)
    error = 0.0
    total = max(len(ref), 1)
    for bin_id in range(len(state.gap_bin_edges) + 1):
        ref_values = ref.loc[ref["gap_bin"] == bin_id, "repeat"]
        if ref_values.empty:
            continue
        can_values = can.loc[can["gap_bin"] == bin_id, "repeat"]
        can_mean = float(can_values.mean()) if not can_values.empty else 0.0
        error += len(ref_values) / total * abs(float(ref_values.mean()) - can_mean)
    return float(error)


def _mutual_information(frame: pd.DataFrame, state: MetricState) -> float:
    transitions = _transition_frame(frame, state)
    if transitions.empty:
        return 0.0
    table = pd.crosstab(transitions["gap_bin"], transitions["repeat"]).to_numpy(float)
    pxy = table / table.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / expected[mask])))


def _lag1(frame: pd.DataFrame, column: str) -> float:
    right = frame[column].to_numpy(float)
    left = frame.groupby("entity_id", sort=False)[column].shift().to_numpy(float)
    pair_mask = np.isfinite(left) & np.isfinite(right)
    left = left[pair_mask]
    right = right[pair_mask]
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _trajectory_signatures(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    signatures = set()
    for _, group in frame.groupby("entity_id", sort=False):
        signature = tuple(
            (
                None if pd.isna(row.gap) else round(float(row.gap), 8),
                _stable_mark(row.receiver_or_mark),
                None if pd.isna(row.amount_or_numeric_value) else round(float(row.amount_or_numeric_value), 8),
            )
            for row in group.itertuples(index=False)
        )
        signatures.add(signature)
    return signatures


def _ngram_signatures(frame: pd.DataFrame, n: int = 3) -> set[tuple[str, ...]]:
    output: set[tuple[str, ...]] = set()
    for _, group in frame.groupby("entity_id", sort=False):
        marks = [_stable_mark(value) for value in group["receiver_or_mark"]]
        output.update(tuple(marks[index : index + n]) for index in range(max(0, len(marks) - n + 1)))
    return output


def evaluate_privacy_exposure(
    reference_events: pd.DataFrame,
    candidate_events: pd.DataFrame,
) -> Dict[str, float]:
    """Compute trajectory-level copy diagnostics from actual event content."""

    reference = _validate(reference_events)
    candidate = _validate(candidate_events)
    ref_trajectories = _trajectory_signatures(reference)
    can_trajectories = _trajectory_signatures(candidate)
    ref_ngrams = _ngram_signatures(reference)
    can_ngrams = _ngram_signatures(candidate)
    return {
        "trajectory_exact_match_rate": len(ref_trajectories & can_trajectories)
        / max(len(can_trajectories), 1),
        "transition_ngram_exposure_rate": len(ref_ngrams & can_ngrams)
        / max(len(can_ngrams), 1),
    }


def _utility_examples(
    events: pd.DataFrame,
    state: MetricState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = _validate(events)
    previous_gap = frame.groupby("entity_id", sort=False)["gap"].shift().to_numpy(float)
    previous_amount = frame.groupby("entity_id", sort=False)[
        "amount_or_numeric_value"
    ].shift().to_numpy(float)
    previous_mark = frame.groupby("entity_id", sort=False)[
        "receiver_or_mark"
    ].shift()
    current_gap = frame["gap"].to_numpy(float)
    current_mark = frame["receiver_or_mark"].map(_stable_mark).to_numpy()
    valid = np.isfinite(current_gap)
    group_names = list(state.mark_groups) + ["<OTHER>"]
    group_index = {value: index for index, value in enumerate(group_names)}
    previous_group = np.asarray(
        [
            group_index.get(_stable_mark(value), len(group_names) - 1)
            for value in previous_mark
        ]
    )
    one_hot = np.eye(len(group_names), dtype=np.float32)[previous_group]
    features = np.column_stack(
        (
            np.log1p(np.nan_to_num(previous_gap, nan=0.0).clip(min=0.0)),
            np.isfinite(previous_gap).astype(float),
            np.nan_to_num(previous_amount, nan=0.0) / state.amount_scale,
            one_hot,
        )
    )
    repeat = (
        current_mark
        == previous_mark.map(_stable_mark).to_numpy()
    ).astype(int)
    return features[valid], np.log1p(current_gap[valid]), repeat[valid]


def _cap_examples(
    arrays: Tuple[np.ndarray, ...],
    cap: int,
    seed: int,
) -> Tuple[np.ndarray, ...]:
    size = len(arrays[0])
    if size <= cap:
        return arrays
    indices = np.random.default_rng(seed).choice(size, size=cap, replace=False)
    return tuple(array[indices] for array in arrays)


def evaluate_next_event_utility(
    source_events: pd.DataFrame,
    real_target_events: pd.DataFrame,
    state: MetricState,
    *,
    max_train_examples: int = 50_000,
    max_test_examples: int = 50_000,
    seed: int = 20260826,
) -> Dict[str, float]:
    """TSTR-style next-event utility with fixed, lightweight downstream models."""

    if max_train_examples < 100 or max_test_examples < 100:
        raise SAFMetricError("utility caps must be at least 100")
    source = _cap_examples(
        _utility_examples(source_events, state),
        max_train_examples,
        seed,
    )
    target = _cap_examples(
        _utility_examples(real_target_events, state),
        max_test_examples,
        seed + 1,
    )
    x_train, gap_train, repeat_train = source
    x_test, gap_test, repeat_test = target
    gap_model = Ridge(alpha=1.0)
    gap_model.fit(x_train, gap_train)
    gap_prediction = gap_model.predict(x_test)
    gap_mae = float(np.mean(np.abs(gap_test - gap_prediction)))
    gap_null = float(np.mean(np.abs(gap_test - np.median(gap_train))))

    if np.unique(repeat_train).size < 2:
        repeat_probability = np.full(len(repeat_test), float(np.mean(repeat_train)))
    else:
        repeat_model = LogisticRegression(
            C=1.0,
            max_iter=200,
            solver="lbfgs",
            random_state=seed,
        )
        repeat_model.fit(x_train, repeat_train)
        repeat_probability = repeat_model.predict_proba(x_test)[:, 1]
    brier = float(np.mean((repeat_test - repeat_probability) ** 2))
    prevalence = float(np.mean(repeat_train))
    brier_null = float(np.mean((repeat_test - prevalence) ** 2))
    return {
        "next_gap_tstr_mae_ratio": gap_mae / max(gap_null, 1e-12),
        "next_repeat_tstr_brier_ratio": brier / max(brier_null, 1e-12),
        "train_examples": int(len(x_train)),
        "test_examples": int(len(x_test)),
    }


def audit_utility_validity(
    reference_events: pd.DataFrame,
    probe_events: pd.DataFrame,
    state: MetricState,
    *,
    seed: int,
) -> Dict[str, object]:
    null = evaluate_next_event_utility(reference_events, probe_events, state, seed=seed)
    gap_corrupt = evaluate_next_event_utility(
        corrupt_train_events(reference_events, "permute_nonfirst_gaps", seed=seed + 1),
        probe_events,
        state,
        seed=seed,
    )
    mark_corrupt = evaluate_next_event_utility(
        corrupt_train_events(reference_events, "permute_marks", seed=seed + 2),
        probe_events,
        state,
        seed=seed,
    )
    checks = {
        "gap_target_corruption_worsens_next_gap": (
            gap_corrupt["next_gap_tstr_mae_ratio"]
            > null["next_gap_tstr_mae_ratio"]
        ),
        "mark_target_corruption_worsens_next_repeat": (
            mark_corrupt["next_repeat_tstr_brier_ratio"]
            > null["next_repeat_tstr_brier_ratio"]
        ),
    }
    return {
        "real_vs_real": null,
        "gap_corruption": gap_corrupt,
        "mark_corruption": mark_corrupt,
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
    }


def evaluate_metric_suite(
    reference_events: pd.DataFrame,
    candidate_events: pd.DataFrame,
    state: MetricState,
    *,
    include_privacy: bool = True,
) -> Dict[str, float]:
    reference = _validate(reference_events)
    candidate = _validate(candidate_events)
    ref_gap = reference["gap"].dropna().to_numpy(float)
    can_gap = candidate["gap"].dropna().to_numpy(float)
    ref_amount = reference["amount_or_numeric_value"].dropna().to_numpy(float)
    can_amount = candidate["amount_or_numeric_value"].dropna().to_numpy(float)
    if not len(can_gap) or not len(can_amount):
        raise SAFMetricError("candidate needs finite gap and amount observations")
    ref_lengths = reference.groupby("entity_id").size().to_numpy(float)
    can_lengths = candidate.groupby("entity_id").size().to_numpy(float)
    scores = {
        "gap_conditioned_mark_tv": _conditional_mark_tv(reference, candidate, state),
        "transition_conditioned_mark_tv": _transition_conditioned_mark_tv(
            reference, candidate, state
        ),
        "short_gap_repeat_curve_l1": _repeat_curve_error(reference, candidate, state),
        "gap_repeat_mi_error": abs(_mutual_information(reference, state) - _mutual_information(candidate, state)),
        "gap_lag1_acf_error": abs(_lag1(reference, "gap") - _lag1(candidate, "gap")),
        "amount_lag1_acf_error": abs(_lag1(reference, "amount_or_numeric_value") - _lag1(candidate, "amount_or_numeric_value")),
        "gap_ks": float(ks_2samp(ref_gap, can_gap).statistic),
        "gap_scaled_w1": float(wasserstein_distance(ref_gap, can_gap) / state.gap_scale),
        "gap_zero_rate_error": abs(float(np.mean(ref_gap == 0)) - float(np.mean(can_gap == 0))),
        "mark_sparse_tv": _tv(_group_marks(reference["receiver_or_mark"], state), _group_marks(candidate["receiver_or_mark"], state)),
        "amount_ks": float(ks_2samp(ref_amount, can_amount).statistic),
        "amount_scaled_w1": float(wasserstein_distance(ref_amount, can_amount) / state.amount_scale),
        "length_scaled_w1": float(wasserstein_distance(ref_lengths, can_lengths) / state.length_scale),
    }
    if include_privacy:
        scores.update(evaluate_privacy_exposure(reference, candidate))
    return scores


def _evaluate_noninferiority_suite(
    reference_events: pd.DataFrame,
    candidate_events: pd.DataFrame,
    state: MetricState,
) -> Dict[str, float]:
    """Compute only margin endpoints, avoiding unrelated temporal/privacy work."""

    reference = _validate(reference_events)
    candidate = _validate(candidate_events)
    ref_gap = reference["gap"].dropna().to_numpy(float)
    can_gap = candidate["gap"].dropna().to_numpy(float)
    ref_amount = reference["amount_or_numeric_value"].dropna().to_numpy(float)
    can_amount = candidate["amount_or_numeric_value"].dropna().to_numpy(float)
    ref_lengths = reference.groupby("entity_id").size().to_numpy(float)
    can_lengths = candidate.groupby("entity_id").size().to_numpy(float)
    return {
        "gap_ks": float(ks_2samp(ref_gap, can_gap).statistic),
        "gap_scaled_w1": float(wasserstein_distance(ref_gap, can_gap) / state.gap_scale),
        "gap_zero_rate_error": abs(float(np.mean(ref_gap == 0)) - float(np.mean(can_gap == 0))),
        "mark_sparse_tv": _tv(_group_marks(reference["receiver_or_mark"], state), _group_marks(candidate["receiver_or_mark"], state)),
        "amount_ks": float(ks_2samp(ref_amount, can_amount).statistic),
        "amount_scaled_w1": float(wasserstein_distance(ref_amount, can_amount) / state.amount_scale),
        "length_scaled_w1": float(wasserstein_distance(ref_lengths, can_lengths) / state.length_scale),
    }


def _stable_entity_order(values: Iterable[object], seed: int) -> list[object]:
    return sorted(
        set(values),
        key=lambda value: hashlib.sha256(f"{seed}|{type(value).__name__}|{value}".encode()).hexdigest(),
    )


def _subset(frame: pd.DataFrame, entities: Sequence[object]) -> pd.DataFrame:
    return frame[frame["entity_id"].isin(set(entities))].copy()


def corrupt_train_events(
    events: pd.DataFrame,
    corruption: str,
    *,
    seed: int,
) -> pd.DataFrame:
    """Known train-only alternatives used to test metric sensitivity."""

    frame = _validate(events)
    rng = np.random.default_rng(seed)
    output = frame.copy()
    if corruption == "permute_marks":
        output["receiver_or_mark"] = rng.permutation(output["receiver_or_mark"].to_numpy())
    elif corruption == "permute_nonfirst_gaps":
        mask = output["gap"].notna()
        output.loc[mask, "gap"] = rng.permutation(output.loc[mask, "gap"].to_numpy())
    elif corruption == "shift_scale_amount":
        values = output["amount_or_numeric_value"].to_numpy(float)
        finite = np.isfinite(values)
        scale = _robust_scale(values)
        values[finite] = values[finite] * 1.35 + 2.0 * scale
        output["amount_or_numeric_value"] = values
    elif corruption == "copy_trajectories":
        output["entity_id"] = [f"copied::{value}" for value in output["entity_id"]]
    else:
        raise SAFMetricError(f"unknown corruption: {corruption}")
    return output


def _permute_marks_within_gap_bins(
    events: pd.DataFrame,
    state: MetricState,
    *,
    seed: int,
) -> pd.DataFrame:
    """Destroy transitions while preserving current-mark counts per gap bin."""

    output = _validate(events).copy()
    nonfirst = output["gap"].notna()
    bin_ids = _gap_bin(output.loc[nonfirst, "gap"].to_numpy(float), state)
    rng = np.random.default_rng(seed)
    for bin_id in np.unique(bin_ids):
        indices = output.loc[nonfirst].index[bin_ids == bin_id]
        values = output.loc[indices, "receiver_or_mark"].to_numpy(copy=True)
        output.loc[indices, "receiver_or_mark"] = values[
            rng.permutation(len(values))
        ]
    return output


def audit_metric_validity(train_events: pd.DataFrame, *, seed: int = 20260826) -> Dict[str, object]:
    """Run a train-only real-vs-real null and targeted corruption audit."""

    frame = _validate(train_events)
    entities = _stable_entity_order(frame["entity_id"], seed)
    if len(entities) < 8:
        raise SAFMetricError("metric validity audit requires at least eight train entities")
    midpoint = len(entities) // 2
    reference = _subset(frame, entities[:midpoint])
    probe = _subset(frame, entities[midpoint:])
    state = fit_metric_state(reference)
    null = evaluate_metric_suite(reference, probe, state, include_privacy=False)
    corrupted = {
        name: evaluate_metric_suite(
            reference,
            corrupt_train_events(probe, name, seed=seed + index + 1),
            state,
            include_privacy=False,
        )
        for index, name in enumerate(
            ("permute_marks", "permute_nonfirst_gaps", "shift_scale_amount")
        )
    }
    corrupted["permute_marks_within_gap_bins"] = evaluate_metric_suite(
        reference,
        _permute_marks_within_gap_bins(
            probe,
            state,
            seed=seed + 8,
        ),
        state,
        include_privacy=False,
    )
    copied = evaluate_privacy_exposure(
        reference,
        corrupt_train_events(reference, "copy_trajectories", seed=seed + 9),
    )
    utility = audit_utility_validity(reference, probe, state, seed=seed)
    joint_family = (
        "transition_conditioned_mark_tv",
        "short_gap_repeat_curve_l1",
        "gap_repeat_mi_error",
    )
    checks = {
        "mark_permutation_preserves_mark_marginal": corrupted["permute_marks"]["mark_sparse_tv"] <= null["mark_sparse_tv"] + 0.05,
        "mark_permutation_detected_jointly": any(
            corrupted["permute_marks"][metric] > null[metric] + 1e-12
            for metric in joint_family
        ),
        "within_gap_mark_permutation_preserves_absolute_conditional_tv": abs(
            corrupted["permute_marks_within_gap_bins"][
                "gap_conditioned_mark_tv"
            ]
            - null["gap_conditioned_mark_tv"]
        )
        <= 1e-12,
        "within_gap_mark_permutation_detected_by_transition_tv": (
            corrupted["permute_marks_within_gap_bins"][
                "transition_conditioned_mark_tv"
            ]
            > null["transition_conditioned_mark_tv"] + 1e-12
        ),
        "gap_permutation_preserves_gap_marginal": corrupted["permute_nonfirst_gaps"]["gap_ks"] <= null["gap_ks"] + 0.05,
        "gap_permutation_detected_jointly": any(
            corrupted["permute_nonfirst_gaps"][metric] > null[metric] + 1e-12
            for metric in joint_family
        ),
        "numeric_corruption_detected": corrupted["shift_scale_amount"]["amount_scaled_w1"] > null["amount_scaled_w1"],
        "copied_trajectories_detected": copied["trajectory_exact_match_rate"] == 1.0,
    }
    return {
        "audit_scope": "train_only_entity_disjoint",
        "seed": seed,
        "reference_entities": int(reference["entity_id"].nunique()),
        "probe_entities": int(probe["entity_id"].nunique()),
        "metric_state": asdict(state),
        "real_vs_real_null": null,
        "corruptions": {**corrupted, "copy_trajectories": copied},
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "utility_validity": utility,
        "interpretation": "Failure is a metric-power diagnostic, never evidence for or against a model.",
    }


def calibrate_train_only_margins(
    train_events: pd.DataFrame,
    *,
    seeds: Sequence[int] = tuple(range(31)),
    quantile: float = 0.95,
) -> Dict[str, float]:
    """Calibrate noninferiority margins from repeated train-only pseudo-splits."""

    if not 0.5 < quantile < 1:
        raise SAFMetricError("margin quantile must lie between 0.5 and 1")
    frame = _validate(train_events)
    records: list[Dict[str, float]] = []
    for seed in seeds:
        entities = _stable_entity_order(frame["entity_id"], int(seed))
        midpoint = len(entities) // 2
        if midpoint < 2 or len(entities) - midpoint < 2:
            raise SAFMetricError("margin calibration needs at least four train entities")
        reference = _subset(frame, entities[:midpoint])
        probe = _subset(frame, entities[midpoint:])
        records.append(
            _evaluate_noninferiority_suite(
                reference,
                probe,
                fit_metric_state(reference),
            )
        )
    noninferiority = [
        name
        for name, spec in METRIC_REGISTRY.items()
        if spec["family"] in {"marginal", "structural"}
    ]
    return {
        name: float(np.quantile([record[name] for record in records], quantile))
        for name in noninferiority
    }
