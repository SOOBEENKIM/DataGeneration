from pathlib import Path

from experiments.full_artifact_store_v2_5 import FullAttemptStore
from experiments.training_callbacks_v2_5 import AttemptTrainingCallbacks


def manifest():
    return {
        "schema_version": "benchmark-v2.5-full-attempt",
        "git_commit": "a" * 40,
        "config_hash": "b" * 64,
        "code_hash": "c" * 64,
        "scenario": "joint_semimarkov_v2b",
        "kappa": 1.0,
        "generator": "neural_sequence",
        "seed": 1,
        "sampling_plan_hash": "0" * 64,
        "data_hashes": {
            "train": "d" * 64,
            "validation": "e" * 64,
            "test": "f" * 64,
        },
        "gpu": {"id": 1},
        "cuda_version": "12.1",
        "pytorch_version": "2.1.2",
        "requested_training_budget": {
            "steps": 20000,
            "max_wall_seconds": 7200,
        },
        "actual_training_budget": {
            "steps": 0,
            "wall_seconds": 0.0,
        },
        "baseline_definition_version": "benchmark-v2.5",
        "evaluation_version": "benchmark-v2.5-evaluation-v1",
    }


class Adapter:
    def save_training_checkpoint(self, path: Path):
        path.write_bytes(b"state")


def event(step):
    return {
        "step": step,
        "loss": 1.0,
        "validation_metric": 1.1,
        "elapsed_seconds": 2.0,
        "peak_gpu_memory_bytes": 3,
    }


def test_callbacks_filter_progress_and_atomically_save_checkpoint_and_final(
    tmp_path,
):
    store, _ = FullAttemptStore.select(
        tmp_path / "benchmark_v2_5",
        manifest(),
    )
    callbacks = AttemptTrainingCallbacks(
        store,
        checkpoint_interval_steps=100,
    )
    callbacks.progress(event(1))
    assert not (store.path / "progress.jsonl").exists()
    callbacks.progress(event(100))
    checkpoint = callbacks.checkpoint(event(100), Adapter())
    final = callbacks.save_final_checkpoint(Adapter())
    assert checkpoint.read_bytes() == b"state"
    assert final.read_bytes() == b"state"
    assert (store.path / "partial_metrics.json").is_file()
