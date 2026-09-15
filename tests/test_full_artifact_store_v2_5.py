import json

import pytest

from experiments.full_artifact_store_v2_5 import (
    ArtifactContractError,
    FullAttemptStore,
    RunAlreadyComplete,
)


def manifest(**changes):
    value = {
        "schema_version": "benchmark-v2.5-full-attempt",
        "git_commit": "a" * 40,
        "config_hash": "b" * 64,
        "code_hash": "c" * 64,
        "scenario": "joint_semimarkov_v2b",
        "kappa": 1.0,
        "generator": "plug_in_hmm",
        "seed": 1,
        "sampling_plan_hash": "0" * 64,
        "data_hashes": {
            "train": "d" * 64,
            "validation": "e" * 64,
            "test": "f" * 64,
        },
        "gpu": {"id": None, "name": None},
        "cuda_version": None,
        "pytorch_version": "2.1.2",
        "requested_training_budget": {
            "steps": None,
            "max_wall_seconds": None,
        },
        "actual_training_budget": {
            "steps": 0,
            "wall_seconds": 0.0,
        },
        "baseline_definition_version": "benchmark-v2.5",
        "evaluation_version": "benchmark-v2.5-evaluation-v1",
    }
    value.update(changes)
    return value


def test_exact_manifest_resumes_latest_hashed_checkpoint(tmp_path):
    root = tmp_path / "benchmark_v2_5"
    store, selected = FullAttemptStore.select(root, manifest())
    assert selected.attempt == 1
    assert selected.resume is False
    store.append_progress(
        {
            "step": 100,
            "loss": 1.0,
            "validation_metric": None,
            "elapsed_seconds": 2.0,
            "peak_gpu_memory_bytes": 0,
        }
    )
    store.write_partial_metrics({"step": 100})
    checkpoint = store.write_checkpoint(
        step=100,
        writer=lambda path: path.write_bytes(b"checkpoint"),
    )
    resumed, resume_selection = FullAttemptStore.select(root, manifest())
    assert resumed.path == store.path
    assert resume_selection.resume is True
    assert resume_selection.latest_checkpoint == checkpoint
    assert json.loads(
        (store.path / "checkpoints/latest").read_text()
    )["step"] == 100


def test_manifest_mismatch_or_terminal_failure_creates_new_attempt(tmp_path):
    root = tmp_path / "benchmark_v2_5"
    first, _ = FullAttemptStore.select(root, manifest())
    changed = manifest(config_hash="9" * 64)
    second, selected = FullAttemptStore.select(root, changed)
    assert selected.attempt == 2
    assert first.path != second.path
    second.fail(
        exception="CUDA out of memory",
        last_checkpoint=None,
        peak_gpu_memory_bytes=123,
    )
    third, selected = FullAttemptStore.select(root, changed)
    assert selected.attempt == 3
    assert third.path.name == "attempt_003"
    assert (second.path / "FAILED.json").is_file()


def test_sampling_plan_mismatch_allocates_new_attempt_without_resume(tmp_path):
    root = tmp_path / "benchmark_v2_5"
    first, _ = FullAttemptStore.select(root, manifest())
    second, selected = FullAttemptStore.select(
        root,
        manifest(sampling_plan_hash="1" * 64),
    )
    assert selected.attempt == 2
    assert selected.resume is False
    assert second.path != first.path


def test_continuation_allocator_starts_at_attempt_003_without_placeholders(
    tmp_path,
):
    root = tmp_path / "benchmark_v2_5"

    store, selected = FullAttemptStore.select(
        root,
        manifest(),
        minimum_attempt=3,
    )

    assert selected.attempt == 3
    assert store.path.name == "attempt_003"
    assert not (store.path.parent / "attempt_001").exists()
    assert not (store.path.parent / "attempt_002").exists()


def test_interrupted_attempt_resumes_checkpoint_in_new_append_only_attempt(
    tmp_path,
):
    root = tmp_path / "benchmark_v2_5"
    first, _ = FullAttemptStore.select(root, manifest())
    checkpoint = first.write_checkpoint(
        step=100,
        writer=lambda path: path.write_bytes(b"interrupt checkpoint"),
    )
    first.interrupted(
        interruption="sigterm",
        last_checkpoint=checkpoint.name,
        actual_elapsed_seconds=3.0,
    )

    second, selected = FullAttemptStore.select(root, manifest())

    assert second.path.name == "attempt_002"
    assert selected.attempt == 2
    assert selected.resume is True
    assert selected.latest_checkpoint == checkpoint
    assert (first.path / "INTERRUPTED.json").is_file()
    assert (first.path / "manifest.json").read_bytes() == (
        second.path / "manifest.json"
    ).read_bytes()


def test_complete_requires_all_artifacts_and_guards_and_never_overwrites(
    tmp_path,
):
    root = tmp_path / "benchmark_v2_5"
    store, _ = FullAttemptStore.select(root, manifest())
    with pytest.raises(ArtifactContractError):
        store.complete({"hard_guards": {"mask": "PASS"}})
    for relative in (
        "sample.npz",
        "checkpoints/final.pt",
        "metrics.json",
        "runtime.json",
        "evaluation.json",
    ):
        store.write_immutable_bytes(relative, relative.encode())
    complete = store.complete(
        {
            "association_recovery_error": 0.01,
            "hard_guards": {"mask": "PASS", "support": "PASS"},
        }
    )
    assert json.loads(complete.read_text())["status"] == "COMPLETE"
    with pytest.raises(RunAlreadyComplete):
        FullAttemptStore.select(root, manifest())
    with pytest.raises(ArtifactContractError):
        store.write_immutable_bytes("metrics.json", b"replacement")


def test_wrong_artifact_root_and_nonpreregistered_scope_fail_closed(tmp_path):
    with pytest.raises(ArtifactContractError):
        FullAttemptStore.select(tmp_path / "other", manifest())
    with pytest.raises(ArtifactContractError):
        FullAttemptStore.select(
            tmp_path / "benchmark_v2_5",
            manifest(kappa=0.0),
        )
    store, _ = FullAttemptStore.select(
        tmp_path / "benchmark_v2_5",
        manifest(seed=2),
    )
    with pytest.raises(ArtifactContractError, match="escapes"):
        store.write_immutable_bytes("../outside.json", b"bad")


def test_third_identical_infrastructure_failure_forces_stop(tmp_path):
    root = tmp_path / "benchmark_v2_5"
    for attempt in range(1, 4):
        store, selected = FullAttemptStore.select(root, manifest())
        assert selected.attempt == attempt
        marker = store.fail(
            exception="scheduler lost worker",
            last_checkpoint=None,
            peak_gpu_memory_bytes=0,
            failure_class="infrastructure",
            failure_fingerprint="scheduler-lost-worker",
        )
    third = json.loads(marker.read_text())
    assert third["consecutive_identical_infrastructure_failures"] == 3
    assert third["mandatory_stop"] is True
    with pytest.raises(
        ArtifactContractError,
        match="three consecutive",
    ):
        FullAttemptStore.select(root, manifest())


def test_heartbeat_resource_and_invalid_status_are_crash_safe(tmp_path):
    root = tmp_path / "benchmark_v2_5"
    store, _ = FullAttemptStore.select(root, manifest())
    store.write_heartbeat({"step": 1, "status": "RUNNING"})
    store.write_heartbeat({"step": 2, "status": "RUNNING"})
    heartbeat = json.loads((store.path / "heartbeat.json").read_text())
    assert heartbeat["step"] == 2
    store.append_resource(
        {
            "elapsed_seconds": 3.0,
            "cpu_rss_bytes": 4,
            "gpu_memory_bytes": 5,
            "gpu_utilization_percent": 6,
        }
    )
    assert len((store.path / "resource.jsonl").read_text().splitlines()) == 1
    marker = store.invalid(
        reason="row guard failed",
        hard_guards={"row_marginal_guards": "FAIL"},
    )
    assert json.loads(marker.read_text())["status"] == "INVALID"
    with pytest.raises(ArtifactContractError):
        store.write_heartbeat({"step": 3, "status": "RUNNING"})


def test_attempt_artifact_index_is_append_only_and_hashed(tmp_path):
    store, _ = FullAttemptStore.select(
        tmp_path / "benchmark_v2_5",
        manifest(),
    )
    artifact = store.write_immutable_bytes("raw.json", b'{"value":1}\n')
    entry = store.append_artifact_index_entry(
        "raw.json",
        role="raw_result",
    )
    assert entry["sha256"]
    assert entry["bytes"] == artifact.stat().st_size
    lines = (store.path / "artifact_index.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == entry
