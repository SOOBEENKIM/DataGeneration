"""Preregistered validation-only selection for external protocol v1."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class ExternalSelectionError(RuntimeError):
    """Raised when threshold or candidate evidence is not admissible."""


FIDELITY_METRICS = (
    "amount_ks_y0",
    "amount_ks_y1",
    "gap_total_variation_y0",
    "gap_total_variation_y1",
    "receiver_total_variation_y0",
    "receiver_total_variation_y1",
)
COHERENCE_METRICS = (
    "short_gap_receiver_repeat_error_y0",
    "short_gap_receiver_repeat_error_y1",
)
HARD_VALIDITY = ("mask", "padding", "train_discrete_support")
EXTERNAL_MODELS = (
    "empirical_iid",
    "ctgan_separate_class",
    "tvae_separate_class",
    "cof_seqgen_frozen_non_v3",
)


def validate_external_analysis_policy(config: Mapping[str, Any]) -> Mapping[str, Any]:
    analysis = config.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ExternalSelectionError("external analysis policy is missing")
    bootstrap = analysis.get("bootstrap")
    expected_bootstrap = {
        "source_split": "train",
        "cluster_unit": "entity",
        "replicates": 1000,
        "order_statistic_rank": 951,
        "interpolation": "none",
        "freeze_before_validation": True,
    }
    if bootstrap != expected_bootstrap:
        raise ExternalSelectionError("external bootstrap policy is not frozen train-only")
    if analysis.get("controlled_five_guard_thresholds_reused") is not False:
        raise ExternalSelectionError("controlled five-guard thresholds cannot be reused")
    fidelity = analysis.get("fidelity")
    expected_fidelity = {
        "amount": ["amount_ks_y0", "amount_ks_y1"],
        "gap": ["gap_total_variation_y0", "gap_total_variation_y1"],
        "receiver": ["receiver_total_variation_y0", "receiver_total_variation_y1"],
    }
    if fidelity != expected_fidelity:
        raise ExternalSelectionError("external fidelity metric family changed")
    coherence = analysis.get("coherence")
    if (
        coherence.get("definition") != "short_gap_times_receiver_repeat_by_class"
        or coherence.get("metrics") != list(COHERENCE_METRICS)
        or coherence.get("short_gap_threshold_source")
        != "train_positive_gap_median"
    ):
        raise ExternalSelectionError("external coherence metric contract changed")
    selection = analysis.get("selection")
    if (
        float(selection.get("fidelity_weight", -1)) != 0.5
        or float(selection.get("coherence_weight", -1)) != 0.5
        or selection.get("all_fidelity_pass_required") is not False
        or selection.get("tie_break")
        != [
            "combined_score",
            "fidelity_max_ratio",
            "coherence_max_ratio",
            "candidate_id_lexicographic",
        ]
    ):
        raise ExternalSelectionError("external selection rule changed")
    row_shuffle = analysis.get("row_shuffle_negative_control")
    if (
        row_shuffle.get("execution_enabled") is not False
        or row_shuffle.get("design_only") is not True
        or row_shuffle.get("fit_and_orientation") != "train_only"
        or row_shuffle.get("preserve")
        != ["Y", "length", "valid_mask", "per_window_transformed_row_multiset"]
        or row_shuffle.get("permute_together")
        != ["amount", "gap", "receiver"]
    ):
        raise ExternalSelectionError("row-shuffle negative-control scope changed")
    test_unlock = analysis.get("test_unlock")
    if (
        test_unlock.get("requires_separate_authorization") is not True
        or test_unlock.get("all_models_terminal_and_selected") is not True
        or test_unlock.get("hard_validity_required") is not True
        or test_unlock.get("all_fidelity_pass_required") is not False
        or test_unlock.get("internal_test_and_sparkov_fraudTest_remain_locked")
        is not True
    ):
        raise ExternalSelectionError("external test-unlock rule changed")
    return {
        "controlled_five_guard_thresholds_reused": False,
        "bootstrap": dict(bootstrap),
        "fidelity": dict(fidelity),
        "coherence": dict(coherence),
        "selection": dict(selection),
        "row_shuffle_negative_control": dict(row_shuffle),
        "test_unlock": dict(test_unlock),
    }


def validate_train_bootstrap_thresholds(thresholds: Mapping[str, Any]) -> None:
    if (
        thresholds.get("schema_version")
        != "external-train-bootstrap-thresholds-v1"
        or thresholds.get("source_split") != "train"
        or thresholds.get("cluster_unit") != "entity"
        or int(thresholds.get("bootstrap_replicates", 0)) != 1000
        or int(thresholds.get("order_statistic_rank", 0)) != 951
        or thresholds.get("interpolation") != "none"
        or thresholds.get("controlled_benchmark_thresholds_copied") is not False
        or thresholds.get("frozen_before_validation") is not True
    ):
        raise ExternalSelectionError(
            "external thresholds must be train-only preregistered bootstrap evidence"
        )
    for key in ("train_sha256", "threshold_config_sha256"):
        value = thresholds.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ExternalSelectionError(f"external threshold provenance missing: {key}")
    expected = (("fidelity", FIDELITY_METRICS), ("coherence", COHERENCE_METRICS))
    for family, metrics in expected:
        observed = thresholds.get(family)
        if not isinstance(observed, Mapping) or set(observed) != set(metrics):
            raise ExternalSelectionError(f"external {family} threshold family mismatch")
        if any(
            not math.isfinite(float(observed[metric]))
            or float(observed[metric]) <= 0
            for metric in metrics
        ):
            raise ExternalSelectionError(f"external {family} thresholds must be positive")


def _candidate_score(
    candidate: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> Mapping[str, Any]:
    hard = candidate.get("hard_validity")
    if not isinstance(hard, Mapping) or set(hard) != set(HARD_VALIDITY):
        raise ExternalSelectionError("candidate hard-validity contract is incomplete")
    if not all(hard.values()):
        raise ExternalSelectionError("hard-invalid candidate cannot enter selection")
    ratios: dict[str, float] = {}
    passes: dict[str, bool] = {}
    for family, metrics in (
        ("fidelity", FIDELITY_METRICS),
        ("coherence", COHERENCE_METRICS),
    ):
        values = candidate.get(family)
        if not isinstance(values, Mapping) or set(values) != set(metrics):
            raise ExternalSelectionError(f"candidate {family} evidence is incomplete")
        for metric in metrics:
            value = float(values[metric])
            if not math.isfinite(value) or value < 0:
                raise ExternalSelectionError(f"candidate metric is invalid: {metric}")
            ratio = value / float(thresholds[family][metric])
            ratios[metric] = ratio
            passes[metric] = ratio <= 1.0
    fidelity_ratio = max(ratios[metric] for metric in FIDELITY_METRICS)
    coherence_ratio = max(ratios[metric] for metric in COHERENCE_METRICS)
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "fidelity_all_pass": all(passes[metric] for metric in FIDELITY_METRICS),
        "fidelity_max_ratio": fidelity_ratio,
        "coherence_max_ratio": coherence_ratio,
        "combined_score": 0.5 * fidelity_ratio + 0.5 * coherence_ratio,
        "metric_ratios": ratios,
        "metric_pass": passes,
        "source_sha256": candidate["source_sha256"],
        "sample_sha256": candidate["sample_sha256"],
    }


def select_external_candidate(
    *,
    model_id: str,
    candidates: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Select by fixed joint fidelity/coherence score, not an all-pass gate."""

    validate_train_bootstrap_thresholds(thresholds)
    if not candidates:
        raise ExternalSelectionError("selection has no candidate evidence")
    scored = [_candidate_score(candidate, thresholds) for candidate in candidates]
    identifiers = [record["candidate_id"] for record in scored]
    if len(set(identifiers)) != len(identifiers):
        raise ExternalSelectionError("selection candidate IDs are not unique")
    selected = min(
        scored,
        key=lambda record: (
            record["combined_score"],
            record["fidelity_max_ratio"],
            record["coherence_max_ratio"],
            record["candidate_id"],
        ),
    )
    return {
        "schema_version": "external-validation-selection-v1",
        "status": "SELECTED",
        "model_id": model_id,
        **selected,
        "selection_tier": (
            "FIDELITY_PASS"
            if selected["fidelity_all_pass"]
            else "FIDELITY_WARNING"
        ),
        "permanent_test_blocked_by_all_pass": False,
        "separate_test_authorization_required": True,
        "tie_break": [
            "combined_score",
            "fidelity_max_ratio",
            "coherence_max_ratio",
            "candidate_id_lexicographic",
        ],
    }


def finalize_external_validation_selection(
    *,
    dataset: str,
    selections: Mapping[str, Mapping[str, Any]],
    threshold_bundle_sha256: str,
    frozen_bundle_tree_sha256: str,
) -> Mapping[str, Any]:
    """Finalize validation evidence without authorizing any test access."""

    if set(selections) != set(EXTERNAL_MODELS):
        raise ExternalSelectionError("selection requires exactly four preregistered models")
    for value in (threshold_bundle_sha256, frozen_bundle_tree_sha256):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ExternalSelectionError("selection provenance hash is invalid")
    warnings = []
    selected_ids: dict[str, str] = {}
    for model in EXTERNAL_MODELS:
        record = selections[model]
        if (
            record.get("status") != "SELECTED"
            or record.get("dataset") != dataset
            or record.get("model_id") != model
            or record.get("hard_valid") is not True
        ):
            raise ExternalSelectionError(
                f"selection is missing terminal hard-valid evidence: {model}"
            )
        selection_hash = record.get("selection_sha256")
        if not isinstance(selection_hash, str) or len(selection_hash) != 64:
            raise ExternalSelectionError(f"selection hash is invalid: {model}")
        selected_ids[model] = str(record["candidate_id"])
        if record.get("fidelity_all_pass") is not True:
            warnings.append(model)
    return {
        "schema_version": "external-validation-finalization-v1",
        "status": "VALIDATION_SELECTION_COMPLETE",
        "dataset": dataset,
        "selected_candidates": selected_ids,
        "fidelity_warning_models": warnings,
        "threshold_bundle_sha256": threshold_bundle_sha256,
        "frozen_bundle_tree_sha256": frozen_bundle_tree_sha256,
        "all_fidelity_pass_required": False,
        "test_unlock_eligible": True,
        "test_execution_authorized": False,
        "separate_test_authorization_required": True,
    }
