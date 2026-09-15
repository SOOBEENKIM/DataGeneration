from pathlib import Path

import numpy as np
import pytest

from benchmarks.types import SequenceBatch
from generators.sampling_plan import SamplingPlan
import scripts.forensic_selection_failure_v2_6 as forensic
from scripts.forensic_selection_failure_v2_6 import (
    build_forensic_report,
    decompose_batch,
    empirical_iid_feasibility,
    training_cap_assessment,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _batch() -> SequenceBatch:
    lengths = np.asarray([2, 2, 2, 2], dtype=np.int64)
    valid = np.ones((4, 2), dtype=np.bool_)
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    x_num = np.zeros((4, 2, 1), dtype=np.float32)
    x_num[..., 0] = np.asarray(
        [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7]],
        dtype=np.float32,
    )
    dt_bin = np.asarray(
        [[0, 1], [0, 1], [0, 1], [0, 1]],
        dtype=np.int64,
    )
    x_cat = dt_bin[..., None].copy()
    return SequenceBatch(
        x_num=x_num,
        dt_bin=dt_bin,
        x_cat=x_cat,
        valid_mask=valid,
        y_entity=y,
        lengths=lengths,
        entity_ids=np.asarray(["a", "b", "c", "d"]),
    )


def test_empirical_iid_feasibility_is_train_fitted_and_plan_stratified():
    train = _batch()
    validation = _batch()
    plan = SamplingPlan.from_batch(validation)
    thresholds = {
        "amount_ks": 1.0,
        "gap_ks": 1.0,
        "amount_abs_standardized_label_effect": 100.0,
        "gap_abs_standardized_label_effect": 100.0,
        "receiver_max_abs_signed_frequency": 1.0,
    }

    result = empirical_iid_feasibility(
        train=train,
        validation=validation,
        plan=plan,
        tau=np.asarray([0.5, 2.0]),
        receiver_categories=2,
        thresholds=thresholds,
        seeds=(2601,),
    )

    assert result["reference"] == "train_fitted_label_conditional_empirical_iid"
    assert result["sampling_plan_hash"] == plan.plan_hash
    assert result["all_trials_pass"] is True
    assert result["trials"][0]["all_five_guards_pass"] is True
    assert result["trials"][0]["sample_contract"]["labels_equal_plan"] is True
    assert result["trials"][0]["sample_contract"]["lengths_equal_plan"] is True
    assert result["trials"][0]["sample_contract"]["mask_equal_plan"] is True


def test_distribution_decomposition_preserves_channel_and_class_boundaries():
    summary = decompose_batch(
        _batch(),
        tau=np.asarray([0.5, 2.0]),
        receiver_categories=2,
    )

    assert summary["amount"]["overall"]["mean"] == pytest.approx(np.mean(
        np.arange(8) / 10
    ))
    assert summary["amount"]["by_label"]["0"]["mean"] == pytest.approx(0.15)
    assert summary["amount"]["by_label"]["1"]["mean"] == pytest.approx(0.55)
    assert summary["gap_bin_frequency"]["overall"] == [0.5, 0.5]
    assert summary["gap_bin_frequency"]["by_label"]["0"] == [0.5, 0.5]
    assert summary["receiver_frequency"]["overall"] == [0.5, 0.5]
    assert len(summary["amount"]["overall"]["quantiles"]) == 7
    assert summary["class_effect"]["amount_signed_standardized"] > 0
    assert summary["class_effect"]["gap_signed_standardized"] == 0


def test_training_cap_hypothesis_requires_every_guard_to_improve():
    rows = []
    for model in ("ctgan_separate_class", "tvae_separate_class", "cof_seqgen"):
        rows.extend(
            [
                {
                    "model_id": model,
                    "candidate_id": "c00_native_checkpoint_10000",
                    "sampled_checkpoint_step": 10_000,
                    "statistics": {
                        key: 0.2
                        for key in (
                            "amount_ks",
                            "gap_ks",
                            "amount_abs_standardized_label_effect",
                            "gap_abs_standardized_label_effect",
                            "receiver_max_abs_signed_frequency",
                        )
                    },
                },
                {
                    "model_id": model,
                    "candidate_id": "c01_native_checkpoint_20000",
                    "sampled_checkpoint_step": 20_000,
                    "statistics": {
                        "amount_ks": 0.1,
                        "gap_ks": 0.1,
                        "amount_abs_standardized_label_effect": 0.1,
                        "gap_abs_standardized_label_effect": 0.3,
                        "receiver_max_abs_signed_frequency": 0.1,
                    },
                },
            ]
        )

    result = training_cap_assessment(rows)

    assert result["hypothesis"] == "REFUTED"
    assert result["all_models_all_guards_improved"] is False
    assert all(
        model["improved_guard_count"] == 4
        for model in result["models"].values()
    )


def test_real_forensic_replay_reads_no_test_and_matches_all_stored_results(
    monkeypatch,
):
    original_load = np.load
    opened: list[Path] = []

    def traced_load(path, *args, **kwargs):
        opened.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(forensic.np, "load", traced_load)
    report, rows = build_forensic_report(REPOSITORY)

    assert len(report["candidate_records"]) == 12
    assert len(rows) == 17
    assert all(
        record["independent_replay_max_abs_error"] <= 1e-12
        for record in report["candidate_records"]
    )
    assert report["empirical_iid_feasibility"]["all_trials_pass"] is True
    assert report["scope"]["gpu_queries"] == 0
    assert report["scope"]["learned_fit_calls"] == 0
    assert report["scope"]["learned_sample_calls"] == 0
    assert {path.name for path in opened} == {
        "train.npz",
        "validation.npz",
        "validation_sample.npz",
    }
    assert all(
        not ({part.lower() for part in path.parts} & {"test", "test.npz"})
        for path in opened
    )
