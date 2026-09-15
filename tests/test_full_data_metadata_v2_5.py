import copy
import functools
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from benchmarks.types import DatasetBundle, SequenceBatch
from eval.model_guards_v2_5 import RowGuardThresholds
from experiments.frozen_metadata_v2_5 import (
    FLOAT_TAG,
    METADATA_FLOAT_ENCODING_SCHEMA,
    decode_frozen_metadata,
    encode_frozen_metadata,
)
import scripts.prepare_full_data_v2_5 as preparation


CONFIG = Path("configs/benchmark_v2/full_v2_5.yaml")


def _fixture_batch() -> SequenceBatch:
    lengths = np.asarray([3, 2, 3, 2], dtype=np.int64)
    valid_mask = np.arange(3)[None, :] < lengths[:, None]
    x_num = np.zeros((4, 3, 1), dtype=np.float32)
    x_num[..., 0][valid_mask] = np.linspace(1.0, 2.0, valid_mask.sum())
    dt_bin = np.zeros((4, 3), dtype=np.int64)
    dt_bin[valid_mask] = np.arange(valid_mask.sum()) % 2
    x_cat = np.zeros((4, 3, 1), dtype=np.int64)
    x_cat[..., 0][valid_mask] = np.arange(valid_mask.sum()) % 3
    return SequenceBatch(
        x_num=x_num,
        dt_bin=dt_bin,
        x_cat=x_cat,
        valid_mask=valid_mask,
        y_entity=np.asarray([0, 1, 0, 1], dtype=np.int64),
        lengths=lengths,
        entity_ids=np.asarray(["a", "b", "c", "d"]),
    )


def _patch_cpu_preparation_boundaries(monkeypatch, bundle, batch):
    monkeypatch.setattr(
        preparation,
        "generate_benchmark",
        lambda unused_config, unused_seed: bundle,
    )
    monkeypatch.setattr(
        preparation,
        "generate_fixed_binning_split",
        lambda unused_config, **unused_kwargs: (
            batch,
            {"metadata": {"fixture": True}},
        ),
    )
    monkeypatch.setattr(
        preparation,
        "calibrate_row_guard_thresholds",
        lambda *unused_args, **unused_kwargs: RowGuardThresholds(
            amount_ks=0.1,
            gap_ks=0.1,
            amount_abs_standardized_label_effect=0.1,
            gap_abs_standardized_label_effect=0.1,
            receiver_max_abs_signed_frequency=0.1,
            calibration_trials=1,
            calibration_seed=24_500,
        ),
    )


def test_strict_json_metadata_round_trip_preserves_infinity_signs():
    metadata = {
        "schema_version": "benchmark-v2.5-frozen-data",
        "scenario": "joint_semimarkov_v2b",
        "bin_edges": [-np.inf, 0.25, 1.5, np.inf],
        "tau": [0.1, 0.75, 2.5],
    }

    encoded = encode_frozen_metadata(metadata)
    payload = json.dumps(encoded, allow_nan=False)
    loaded = json.loads(payload)
    decoded = decode_frozen_metadata(loaded)

    assert loaded["provenance_float_encoding"]["schema_version"] == (
        METADATA_FLOAT_ENCODING_SCHEMA
    )
    assert loaded["bin_edges"][0] == {FLOAT_TAG: "-Infinity"}
    assert loaded["bin_edges"][-1] == {FLOAT_TAG: "+Infinity"}
    assert loaded["bin_edges"][1:-1] == [0.25, 1.5]
    np.testing.assert_array_equal(
        np.asarray(decoded["bin_edges"]),
        np.asarray(metadata["bin_edges"]),
    )
    np.testing.assert_array_equal(
        np.asarray(decoded["tau"]),
        np.asarray(metadata["tau"]),
    )


def test_metadata_nan_remains_a_hard_failure():
    metadata = {
        "schema_version": "benchmark-v2.5-frozen-data",
        "bin_edges": [-np.inf, 1.0, np.inf],
        "tau": [0.5, np.nan],
    }

    with pytest.raises(ValueError, match="NaN is forbidden"):
        encode_frozen_metadata(metadata)


def test_tagged_infinity_requires_declared_encoding_schema():
    metadata = {
        "bin_edges": [{FLOAT_TAG: "-Infinity"}, 1.0],
        "tau": [0.5],
    }

    with pytest.raises(ValueError, match="encoding schema is required"):
        decode_frozen_metadata(metadata)


def test_prepare_success_publishes_complete_manifest_last_without_models(
    tmp_path,
    monkeypatch,
):
    batch = _fixture_batch()
    bundle = DatasetBundle(
        train=batch,
        test=batch,
        metadata={
            "bin_edges": [-np.inf, 0.75, np.inf],
            "tau": [0.25, 1.25],
        },
    )
    _patch_cpu_preparation_boundaries(monkeypatch, bundle, batch)

    calls = {"full": 0, "gpu": 0, "learned_fit": 0, "learned_sample": 0}

    def forbidden(name, original):
        @functools.wraps(original)
        def call(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} must not run during preparation")

        return call

    import torch
    from generators.cof_seqgen_adapter import CoFSeqGenAdapter
    from generators.conditional_ctgan import ConditionalCTGAN
    from generators.conditional_tvae import ConditionalTVAE
    from generators.joint_sequence_baseline import NeuralSequenceBaseline
    import scripts.run_full_experiment_v2_5 as full_runner

    monkeypatch.setattr(
        full_runner,
        "run_full_experiment",
        forbidden("full", full_runner.run_full_experiment),
    )
    for name in ("is_available", "device_count"):
        original = getattr(torch.cuda, name)
        monkeypatch.setattr(
            torch.cuda,
            name,
            forbidden("gpu", original),
        )
    for adapter_type in (
        ConditionalCTGAN,
        ConditionalTVAE,
        NeuralSequenceBaseline,
        CoFSeqGenAdapter,
    ):
        monkeypatch.setattr(
            adapter_type,
            "fit",
            forbidden("learned_fit", adapter_type.fit),
        )
        monkeypatch.setattr(
            adapter_type,
            "sample",
            forbidden("learned_sample", adapter_type.sample),
        )

    raw = copy.deepcopy(yaml.safe_load(CONFIG.read_text()))
    raw["sampling_plan"]["entity_count"] = 4
    data_root = tmp_path / "data" / "benchmark_v2_5"
    artifact_root = tmp_path / "artifacts" / "benchmark_v2_5"

    manifest = preparation.prepare_data(
        raw,
        config_path=CONFIG,
        data_root=data_root,
        artifact_root=artifact_root,
    )

    destination = (
        data_root
        / "frozen/joint_semimarkov_v2b/kappa_1.00"
    )
    manifest_path = destination / "data_manifest.json"
    stored_manifest = json.loads(manifest_path.read_text())
    stored_metadata = json.loads((destination / "meta.json").read_text())
    referenced = [
        destination / f"{split}.npz"
        for split in ("train", "validation", "test")
    ] + [
        destination / "shared_sampling_plan.npz",
        destination / "meta.json",
        artifact_root / "prerun_calibration/row_guard_thresholds.json",
    ]

    assert manifest["status"] == stored_manifest["status"] == "COMPLETE"
    assert all(path.is_file() for path in referenced)
    assert manifest_path.stat().st_mtime_ns >= max(
        path.stat().st_mtime_ns for path in referenced
    )
    assert stored_metadata["bin_edges"][0] == {FLOAT_TAG: "-Infinity"}
    assert stored_metadata["bin_edges"][-1] == {FLOAT_TAG: "+Infinity"}
    loaded_inputs = full_runner.load_and_validate_frozen_inputs(
        repository_root=Path("."),
        config_path=CONFIG,
        data_manifest_path=manifest_path,
    )
    np.testing.assert_array_equal(
        loaded_inputs.tau,
        np.asarray(bundle.metadata["tau"]),
    )
    assert calls == {
        "full": 0,
        "gpu": 0,
        "learned_fit": 0,
        "learned_sample": 0,
    }


def test_prepare_nan_failure_never_publishes_complete_manifest(
    tmp_path,
    monkeypatch,
):
    batch = _fixture_batch()
    bundle = DatasetBundle(
        train=batch,
        test=batch,
        metadata={
            "bin_edges": [-np.inf, 0.75, np.inf],
            "tau": [0.25, np.nan],
        },
    )
    _patch_cpu_preparation_boundaries(monkeypatch, bundle, batch)
    raw = copy.deepcopy(yaml.safe_load(CONFIG.read_text()))
    raw["sampling_plan"]["entity_count"] = 4
    data_root = tmp_path / "data" / "benchmark_v2_5"
    artifact_root = tmp_path / "artifacts" / "benchmark_v2_5"

    with pytest.raises(ValueError, match="NaN is forbidden"):
        preparation.prepare_data(
            raw,
            config_path=CONFIG,
            data_root=data_root,
            artifact_root=artifact_root,
        )

    destination = (
        data_root
        / "frozen/joint_semimarkov_v2b/kappa_1.00"
    )
    assert not (destination / "meta.json").exists()
    assert not (destination / "data_manifest.json").exists()
    assert (destination / "train.npz").is_file()
    assert (
        artifact_root
        / "prerun_calibration/row_guard_thresholds.json"
    ).is_file()
