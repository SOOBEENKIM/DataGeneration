from pathlib import Path

from benchmarks.temporal_coupling_v2 import BenchmarkConfig, generate_benchmark
from experiments.full_artifact_store_v2_5 import validate_manifest
from experiments.provenance_v2_5 import hash_batch, hash_code


def test_batch_and_code_hashes_are_deterministic_and_sha256_sized():
    bundle = generate_benchmark(
        BenchmarkConfig(
            n_train=20,
            n_test=10,
            fraud_rate=0.5,
            min_length=4,
            max_length=5,
        ),
        42,
    )
    assert hash_batch(bundle.train) == hash_batch(bundle.train)
    assert len(hash_batch(bundle.train)) == 64
    code = hash_code(
        Path("."),
        paths=[Path("generators/contracts_v2_5.py")],
    )
    assert len(code) == 64


def test_manifest_validator_requires_all_three_data_hashes():
    manifest = {
        "schema_version": "benchmark-v2.5-full-attempt",
        "git_commit": "a" * 40,
        "config_hash": "b" * 64,
        "code_hash": "c" * 64,
        "scenario": "joint_semimarkov_v2b",
        "kappa": 1.0,
        "generator": "cof_seqgen",
        "seed": 5,
        "sampling_plan_hash": "0" * 64,
        "data_hashes": {
            "train": "d" * 64,
            "validation": "e" * 64,
            "test": "f" * 64,
        },
        "gpu": {"id": 3},
        "cuda_version": "12.1",
        "pytorch_version": "2.1.2",
        "requested_training_budget": {"max_wall_seconds": 7200},
        "actual_training_budget": {"steps": 0, "wall_seconds": 0.0},
        "baseline_definition_version": "benchmark-v2.5",
        "evaluation_version": "benchmark-v2.5-evaluation-v1",
    }
    validate_manifest(manifest)
