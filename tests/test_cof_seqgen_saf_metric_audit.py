from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.cof_seqgen_saf_metrics import (
    METRIC_REGISTRY,
    audit_metric_validity,
    corrupt_train_events,
    audit_utility_validity,
    evaluate_privacy_exposure,
    evaluate_metric_suite,
    fit_metric_state,
)
from benchmarks import cof_seqgen_saf_metrics as metric_module


def _coupled_events(n_entities: int = 40) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(8)
    for entity in range(n_entities):
        previous = "A"
        timestamp = 0.0
        for index in range(12):
            if index == 0:
                gap = np.nan
            else:
                gap = float((entity + index) % 4)
                timestamp += gap
            mark = previous if index and gap <= 1 else ("B" if previous == "A" else "A")
            value = float(2 * (0 if np.isnan(gap) else gap) + (mark == "B") + rng.normal(0, 0.01))
            rows.append(
                {
                    "entity_id": f"e{entity}",
                    "event_index": index,
                    "gap": gap,
                    "receiver_or_mark": mark,
                    "amount_or_numeric_value": value,
                }
            )
            previous = mark
    return pd.DataFrame(rows)


def test_metric_registry_directly_targets_factorization() -> None:
    primary = {name for name, spec in METRIC_REGISTRY.items() if spec.get("primary")}
    assert primary == {
        "transition_conditioned_mark_tv",
        "short_gap_repeat_curve_l1",
        "gap_repeat_mi_error",
    }


def test_empty_conditional_support_is_maximal_tv_not_half_credit() -> None:
    assert metric_module._tv(["A", "A"], []) == 1.0


def test_known_corruptions_preserve_marginals_and_move_target_metrics() -> None:
    events = _coupled_events()
    state = fit_metric_state(events)
    mark_corrupt = corrupt_train_events(events, "permute_marks", seed=1)
    gap_corrupt = corrupt_train_events(events, "permute_nonfirst_gaps", seed=2)
    mark_scores = evaluate_metric_suite(events, mark_corrupt, state)
    gap_scores = evaluate_metric_suite(events, gap_corrupt, state)
    assert mark_scores["mark_sparse_tv"] == 0.0
    assert mark_scores["gap_conditioned_mark_tv"] > 0
    assert gap_scores["gap_ks"] == 0.0
    assert gap_scores["short_gap_repeat_curve_l1"] > 0


def test_transition_conditioned_tv_detects_dependency_within_gap_bin() -> None:
    events = _coupled_events()
    state = fit_metric_state(events)
    corrupted = events.copy()
    nonfirst = corrupted["gap"].notna()
    bin_ids = np.searchsorted(
        np.asarray(state.gap_bin_edges),
        corrupted.loc[nonfirst, "gap"].to_numpy(float),
        side="right",
    )
    rng = np.random.default_rng(19)
    for bin_id in np.unique(bin_ids):
        indices = corrupted.loc[nonfirst].index[bin_ids == bin_id]
        values = corrupted.loc[indices, "receiver_or_mark"].to_numpy(copy=True)
        corrupted.loc[indices, "receiver_or_mark"] = values[
            rng.permutation(len(values))
        ]
    scores = evaluate_metric_suite(events, corrupted, state)
    assert scores["gap_conditioned_mark_tv"] == 0.0
    assert scores["transition_conditioned_mark_tv"] > 0.1


def test_train_only_validity_audit_detects_all_registered_corruptions() -> None:
    result = audit_metric_validity(_coupled_events())
    assert result["audit_scope"] == "train_only_entity_disjoint"
    assert result["all_checks_pass"] is True
    assert result["checks"][
        "within_gap_mark_permutation_preserves_absolute_conditional_tv"
    ] is True
    assert result["checks"][
        "within_gap_mark_permutation_detected_by_transition_tv"
    ] is True
    assert result["corruptions"]["copy_trajectories"]["trajectory_exact_match_rate"] == 1.0


def test_privacy_copy_detection_uses_trajectory_content_not_entity_id() -> None:
    events = _coupled_events(8)
    copied = corrupt_train_events(events, "copy_trajectories", seed=9)
    scores = evaluate_privacy_exposure(events, copied)
    assert scores["trajectory_exact_match_rate"] == 1.0
    assert scores["transition_ngram_exposure_rate"] == 1.0


def test_next_event_utility_detects_target_corruption() -> None:
    events = _coupled_events(40)
    entities = sorted(events["entity_id"].unique())
    reference = events[events["entity_id"].isin(entities[:20])]
    probe = events[events["entity_id"].isin(entities[20:])]
    result = audit_utility_validity(
        reference,
        probe,
        fit_metric_state(reference),
        seed=3,
    )
    assert result["all_checks_pass"] is True
