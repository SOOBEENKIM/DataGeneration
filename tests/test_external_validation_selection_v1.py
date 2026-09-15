import pytest
import yaml
from pathlib import Path

from eval.external_validation_protocol_v1 import (
    ExternalSelectionError,
    finalize_external_validation_selection,
    select_external_candidate,
    validate_external_analysis_policy,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/benchmark_v2/external_validation_v1.yaml"


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


def _thresholds():
    return {
        "schema_version": "external-train-bootstrap-thresholds-v1",
        "source_split": "train",
        "cluster_unit": "entity",
        "bootstrap_replicates": 1000,
        "order_statistic_rank": 951,
        "interpolation": "none",
        "controlled_benchmark_thresholds_copied": False,
        "frozen_before_validation": True,
        "train_sha256": "a" * 64,
        "threshold_config_sha256": "b" * 64,
        "fidelity": {metric: 0.10 for metric in FIDELITY_METRICS},
        "coherence": {metric: 0.20 for metric in COHERENCE_METRICS},
    }


def _candidate(candidate_id, fidelity_value, coherence_value):
    return {
        "candidate_id": candidate_id,
        "hard_validity": {
            "mask": True,
            "padding": True,
            "train_discrete_support": True,
        },
        "fidelity": {metric: fidelity_value for metric in FIDELITY_METRICS},
        "coherence": {metric: coherence_value for metric in COHERENCE_METRICS},
        "source_sha256": "c" * 64,
        "sample_sha256": "d" * 64,
    }


def test_selection_uses_fidelity_and_coherence_without_all_pass_permanent_block():
    candidates = [
        _candidate("candidate_a", fidelity_value=0.11, coherence_value=0.05),
        _candidate("candidate_b", fidelity_value=0.16, coherence_value=0.01),
    ]

    selected = select_external_candidate(
        model_id="cof_seqgen_frozen_non_v3",
        candidates=candidates,
        thresholds=_thresholds(),
    )

    assert selected["status"] == "SELECTED"
    assert selected["candidate_id"] == "candidate_a"
    assert selected["fidelity_all_pass"] is False
    assert selected["selection_tier"] == "FIDELITY_WARNING"
    assert selected["permanent_test_blocked_by_all_pass"] is False
    assert selected["separate_test_authorization_required"] is True

    invalid_thresholds = {**_thresholds(), "source_split": "validation"}
    with pytest.raises(ExternalSelectionError, match="train-only"):
        select_external_candidate(
            model_id="cof_seqgen_frozen_non_v3",
            candidates=candidates,
            thresholds=invalid_thresholds,
        )


def test_external_analysis_policy_keeps_row_shuffle_design_only_and_test_locked():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    policy = validate_external_analysis_policy(config)

    assert policy["controlled_five_guard_thresholds_reused"] is False
    assert policy["bootstrap"] == {
        "source_split": "train",
        "cluster_unit": "entity",
        "replicates": 1000,
        "order_statistic_rank": 951,
        "interpolation": "none",
        "freeze_before_validation": True,
    }
    assert policy["row_shuffle_negative_control"]["execution_enabled"] is False
    assert policy["row_shuffle_negative_control"]["preserve"] == [
        "Y",
        "length",
        "valid_mask",
        "per_window_transformed_row_multiset",
    ]
    assert policy["row_shuffle_negative_control"]["permute_together"] == [
        "amount",
        "gap",
        "receiver",
    ]
    assert policy["test_unlock"]["requires_separate_authorization"] is True
    assert policy["test_unlock"]["all_fidelity_pass_required"] is False


def test_selection_finalization_requires_all_models_but_never_unlocks_test_directly():
    model_ids = (
        "empirical_iid",
        "ctgan_separate_class",
        "tvae_separate_class",
        "cof_seqgen_frozen_non_v3",
    )
    selections = {
        model: {
            "status": "SELECTED",
            "dataset": "amlsim",
            "model_id": model,
            "candidate_id": f"{model}_final",
            "hard_valid": True,
            "fidelity_all_pass": model == "empirical_iid",
            "selection_sha256": (str(index) * 64)[:64],
        }
        for index, model in enumerate(model_ids, start=1)
    }
    result = finalize_external_validation_selection(
        dataset="amlsim",
        selections=selections,
        threshold_bundle_sha256="a" * 64,
        frozen_bundle_tree_sha256="b" * 64,
    )
    assert result["status"] == "VALIDATION_SELECTION_COMPLETE"
    assert result["test_unlock_eligible"] is True
    assert result["test_execution_authorized"] is False
    assert result["separate_test_authorization_required"] is True
    assert result["all_fidelity_pass_required"] is False

    incomplete = dict(selections)
    incomplete.pop("ctgan_separate_class")
    with pytest.raises(ExternalSelectionError, match="exactly four"):
        finalize_external_validation_selection(
            dataset="amlsim",
            selections=incomplete,
            threshold_bundle_sha256="a" * 64,
            frozen_bundle_tree_sha256="b" * 64,
        )
