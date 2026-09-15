import hashlib
import functools
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest
import numpy as np
import yaml

from benchmarks.types import SequenceBatch
from experiments.full_artifact_store_v2_5 import FullAttemptStore
from experiments.provenance_v2_5 import hash_batch
from generators.sampling_plan import SamplingPlan
from scripts.run_full_experiment_v2_5 import (
    _finalize_full_results,
    AuthorizationError,
    FullRunStateStore,
    RunPolicyController,
    RunnerInterrupted,
    apply_final_decision_contract,
    build_gpu_waves,
    build_job_plan,
    execute_job_schedule,
    publish_finalization_artifacts,
    plan_scheduler_continuation,
    replay_continuation_results,
    run_dry_run,
    run_full_experiment,
    run_live_gpu_schedule,
    run_owned_attempt_process,
    run_plan,
    select_idle_gpus,
    validate_real_adapter_integrations,
    validate_full_authorization,
    validate_scheduler_continuation_changed_paths,
)


CONFIG = Path("configs/benchmark_v2/full_v2_5.yaml")
PLAN_HASH = "1" * 64


def hanging_fixture_worker(*, attempt_path, manifest, sleep_seconds):
    store = FullAttemptStore(Path(attempt_path), manifest)
    store.write_checkpoint(
        step=1,
        writer=lambda path: path.write_bytes(b"last atomic checkpoint"),
    )
    time.sleep(sleep_seconds)
    return {"status": "COMPLETE"}


def interrupting_fixture_worker(*, attempt_path, manifest, interrupt):
    store = FullAttemptStore(Path(attempt_path), manifest)
    store.write_checkpoint(
        step=1,
        writer=lambda path: path.write_bytes(b"interrupt checkpoint"),
    )
    if interrupt == "keyboard":
        raise KeyboardInterrupt("fixture keyboard interrupt")
    if interrupt == "system_exit":
        raise SystemExit(17)
    if interrupt == "sigterm":
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
    raise AssertionError(interrupt)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_frozen_plan_fixture(tmp_path):
    repository = tmp_path / "repository"
    config_path = repository / CONFIG
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(CONFIG.read_bytes())
    data_directory = (
        repository
        / "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00"
    )
    data_directory.mkdir(parents=True)
    n, length = 4, 3
    lengths = np.asarray([3, 2, 3, 2], dtype=np.int64)
    valid = np.arange(length)[None, :] < lengths[:, None]
    x_num = np.zeros((n, length, 1), dtype=np.float32)
    dt_bin = np.zeros((n, length), dtype=np.int64)
    x_cat = np.zeros((n, length, 1), dtype=np.int64)
    x_num[valid, 0] = np.arange(valid.sum(), dtype=np.float32)
    dt_bin[valid] = np.arange(valid.sum()) % 2
    x_cat[..., 0][valid] = np.arange(valid.sum()) % 3
    batch = SequenceBatch(
        x_num=x_num,
        dt_bin=dt_bin,
        x_cat=x_cat,
        valid_mask=valid,
        y_entity=np.asarray([0, 1, 0, 1], dtype=np.int64),
        lengths=lengths,
        entity_ids=np.asarray(["a", "b", "c", "d"]),
    )
    split_paths = {}
    for split in ("train", "validation", "test"):
        path = data_directory / f"{split}.npz"
        np.savez_compressed(
            path,
            **{
                field: getattr(batch, field)
                for field in SequenceBatch.__dataclass_fields__
            },
        )
        split_paths[split] = path
    plan = SamplingPlan.from_batch(batch)
    plan_path = data_directory / "shared_sampling_plan.npz"
    plan.save(str(plan_path))
    meta_path = data_directory / "meta.json"
    meta_path.write_text(json.dumps({"tau": [0.5, 1.5]}) + "\n")
    threshold_path = (
        repository
        / "artifacts/benchmark_v2_5/prerun_calibration/"
        "row_guard_thresholds.json"
    )
    threshold_path.parent.mkdir(parents=True)
    threshold_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "learned_results_used": False,
                "thresholds": {
                    "amount_ks": 1.0,
                    "gap_ks": 1.0,
                    "amount_abs_standardized_label_effect": 10.0,
                    "gap_abs_standardized_label_effect": 10.0,
                    "receiver_max_abs_signed_frequency": 1.0,
                    "calibration_trials": 200,
                    "calibration_seed": 24500,
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    manifest_path = data_directory / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "config_sha256": sha256(config_path),
                "scenario": "joint_semimarkov_v2b",
                "kappa": 1.0,
                "data_hashes": {
                    split: hash_batch(batch)
                    for split in ("train", "validation", "test")
                },
                "file_sha256": {
                    split: sha256(path)
                    for split, path in split_paths.items()
                },
                "sampling_plan": {
                    "path": str(plan_path),
                    "sha256": plan.plan_hash,
                    "file_sha256": sha256(plan_path),
                },
                "metadata_sha256": sha256(meta_path),
                "row_guard_thresholds_path": str(threshold_path),
                "row_guard_thresholds_sha256": sha256(threshold_path),
            },
            sort_keys=True,
        )
        + "\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    return repository, config_path, manifest_path, split_paths["train"]


def write_authorized_snapshot(tmp_path, **changes):
    approval_text = "fixture explicit full approval"
    data_manifest = {
        "schema_version": "benchmark-v2.5-frozen-data-manifest",
        "status": "COMPLETE",
        "config_sha256": sha256(CONFIG),
        "scenario": "joint_semimarkov_v2b",
        "kappa": 1.0,
        "data_hashes": {
            "train": "2" * 64,
            "validation": "3" * 64,
            "test": "4" * 64,
        },
        "sampling_plan": {"sha256": PLAN_HASH},
    }
    data_path = tmp_path / "data_manifest.json"
    data_path.write_text(json.dumps(data_manifest, sort_keys=True) + "\n")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    authorization = {
        "schema_version": "benchmark-v2.5-full-authorization-v1",
        "source_commit": commit,
        "config_sha256": sha256(CONFIG),
        "v2_4": {
            "gate_tag": "benchmark-v2.4-gate-pass",
            "gate_commit": (
                "35ec654061f99e8e5ccd54381d40329546ed2eb7"
            ),
            "artifact_index_sha256": (
                "c99c79b501ba264b01f1ef54f1bb7fa1c40446c865d95b6078fad6fd47fec08a"
            ),
            "gate_report_sha256": (
                "53fc93c3650e5e6d03f9a085c74a5ad86ef2862f7a918668bce817500319ee14"
            ),
        },
        "capacity_preflight": {
            "artifact_index_sha256": sha256(
                "artifacts/benchmark_v2_5/capacity_preflight/artifact_index.json"
            ),
            "correction_note_sha256": sha256(
                "artifacts/benchmark_v2_5/capacity_preflight/"
                "capacity_preflight_correction_001.md"
            ),
        },
        "frozen_data": {
            "manifest_sha256": sha256(data_path),
            "data_hashes": data_manifest["data_hashes"],
            "sampling_plan_sha256": PLAN_HASH,
        },
        "scope": {
            "scenario": "joint_semimarkov_v2b",
            "kappa": 1.0,
            "seeds": [1, 2, 3, 4, 5],
            "generators": list(
                yaml.safe_load(CONFIG.read_text())["baselines"]
            ),
            "analysis": ["primary", "secondary"],
        },
        "approval_text": approval_text,
        "approval_text_sha256": hashlib.sha256(
            approval_text.encode()
        ).hexdigest(),
    }
    authorization.update(changes)
    auth_path = tmp_path / "authorization.json"
    auth_path.write_text(json.dumps(authorization, sort_keys=True) + "\n")
    return auth_path, data_path


def test_plan_contains_exactly_thirteen_generators_by_five_seeds():
    raw = yaml.safe_load(CONFIG.read_text())
    jobs = build_job_plan(raw, sampling_plan_hash=PLAN_HASH)
    assert len(jobs) == 65
    assert {job.generator for job in jobs} == set(raw["baselines"])
    assert {job.seed for job in jobs} == {1, 2, 3, 4, 5}
    assert {job.sampling_plan_hash for job in jobs} == {PLAN_HASH}


def test_plan_validates_full_frozen_data_hashes_before_planning(tmp_path):
    repository, config_path, manifest_path, train_path = (
        write_frozen_plan_fixture(tmp_path)
    )
    result = run_plan(
        repository_root=repository,
        config_path=config_path,
        data_manifest_path=manifest_path,
    )
    assert result["job_count"] == 65
    assert result["frozen_data_deep_validation"] is True
    assert result["adapter_preflight"]["status"] == "PASS"
    assert result["adapter_preflight"]["calls"] == {
        "fit": 0,
        "sample": 0,
        "dgp": 0,
        "cuda": 0,
    }
    with train_path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(RuntimeError, match="train file hash mismatch"):
        run_plan(
            repository_root=repository,
            config_path=config_path,
            data_manifest_path=manifest_path,
        )


def test_all_real_adapters_import_and_map_config_without_training(
    monkeypatch,
):
    import torch
    import benchmarks.temporal_coupling_v2 as dgp
    from generators.full_registry_v2_5 import baseline_registry

    calls = {"fit": 0, "sample": 0, "dgp": 0, "cuda": 0}

    def forbidden(name, original):
        @functools.wraps(original)
        def call(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} must not run in adapter preflight")

        return call

    classes = {
        type(spec.factory())
        for spec in baseline_registry().values()
    }
    for adapter_type in classes:
        monkeypatch.setattr(
            adapter_type,
            "fit",
            forbidden("fit", adapter_type.fit),
        )
        monkeypatch.setattr(
            adapter_type,
            "sample",
            forbidden("sample", adapter_type.sample),
        )
    monkeypatch.setattr(
        dgp,
        "generate_benchmark",
        forbidden("dgp", dgp.generate_benchmark),
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        forbidden("cuda", torch.cuda.is_available),
    )
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        forbidden("cuda", torch.cuda.device_count),
    )

    result = validate_real_adapter_integrations(
        yaml.safe_load(CONFIG.read_text())
    )
    assert result["status"] == "PASS"
    assert len(result["adapters"]) == 13
    assert [item["generator"] for item in result["adapters"]] == list(
        yaml.safe_load(CONFIG.read_text())["baselines"]
    )
    assert all(item["fit_signature_valid"] for item in result["adapters"])
    assert all(item["sample_signature_valid"] for item in result["adapters"])
    assert all(
        item["callback_resume_contract_valid"]
        for item in result["adapters"]
        if item["device_class"] == "gpu"
    )
    assert result["calls"] == {
        "fit": 0,
        "sample": 0,
        "dgp": 0,
        "cuda": 0,
    }
    assert calls == result["calls"]


@pytest.mark.parametrize("device_class", ("cpu", "gpu"))
def test_owned_child_hard_cap_is_enforced_for_cpu_and_gpu(
    tmp_path,
    device_class,
):
    attempt_manifest = {
        "schema_version": "benchmark-v2.5-full-attempt",
        "git_commit": "a" * 40,
        "config_hash": "b" * 64,
        "code_hash": "c" * 64,
        "scenario": "joint_semimarkov_v2b",
        "kappa": 1.0,
        "generator": f"fixture_{device_class}",
        "seed": 1,
        "sampling_plan_hash": PLAN_HASH,
        "data_hashes": {
            "train": "d" * 64,
            "validation": "e" * 64,
            "test": "f" * 64,
        },
        "gpu": {"id": 0 if device_class == "gpu" else None},
        "cuda_version": "fixture" if device_class == "gpu" else None,
        "pytorch_version": "fixture",
        "requested_training_budget": {
            "steps": 100,
            "max_wall_seconds": 0.25,
        },
        "actual_training_budget": {"steps": 0, "wall_seconds": 0.0},
        "baseline_definition_version": "benchmark-v2.5",
        "evaluation_version": "benchmark-v2.5-evaluation-v1",
    }
    store, _ = FullAttemptStore.select(
        tmp_path / "benchmark_v2_5",
        attempt_manifest,
    )
    started = time.monotonic()
    result = run_owned_attempt_process(
        worker=hanging_fixture_worker,
        worker_kwargs={
            "attempt_path": str(store.path),
            "manifest": attempt_manifest,
            "sleep_seconds": 30.0,
        },
        store=store,
        max_wall_seconds=0.25,
        termination_grace_seconds=0.15,
        process_start_method="fork",
    )
    elapsed = time.monotonic() - started
    marker = json.loads((store.path / "FAILED.json").read_text())
    assert elapsed < 2.0
    assert result["status"] == "FAILED"
    assert result["failure_class"] == "wall_cap"
    assert marker["failure_class"] == "wall_cap"
    assert marker["actual_elapsed_seconds"] >= 0.25
    assert marker["last_checkpoint"] == "step_00000001.pt"
    assert marker["runner_owned_child_terminated"] is True


@pytest.mark.parametrize(
    "interrupt",
    ("keyboard", "system_exit", "sigterm"),
)
def test_operator_interrupt_is_not_misclassified_as_failed(
    tmp_path,
    interrupt,
):
    attempt_manifest = {
        "schema_version": "benchmark-v2.5-full-attempt",
        "git_commit": "a" * 40,
        "config_hash": "b" * 64,
        "code_hash": "c" * 64,
        "scenario": "joint_semimarkov_v2b",
        "kappa": 1.0,
        "generator": f"fixture_{interrupt}",
        "seed": 1,
        "sampling_plan_hash": PLAN_HASH,
        "data_hashes": {
            "train": "d" * 64,
            "validation": "e" * 64,
            "test": "f" * 64,
        },
        "gpu": {"id": None},
        "cuda_version": None,
        "pytorch_version": "fixture",
        "requested_training_budget": {
            "steps": 100,
            "max_wall_seconds": 5,
        },
        "actual_training_budget": {"steps": 0, "wall_seconds": 0.0},
        "baseline_definition_version": "benchmark-v2.5",
        "evaluation_version": "benchmark-v2.5-evaluation-v1",
    }
    store, _ = FullAttemptStore.select(
        tmp_path / "benchmark_v2_5",
        attempt_manifest,
    )
    with pytest.raises(RunnerInterrupted):
        run_owned_attempt_process(
            worker=interrupting_fixture_worker,
            worker_kwargs={
                "attempt_path": str(store.path),
                "manifest": attempt_manifest,
                "interrupt": interrupt,
            },
            store=store,
            max_wall_seconds=5,
            termination_grace_seconds=0.2,
            process_start_method="fork",
        )
    marker = json.loads((store.path / "INTERRUPTED.json").read_text())
    assert marker["status"] == "INTERRUPTED"
    assert marker["interruption"] == interrupt
    assert marker["last_checkpoint"] == "step_00000001.pt"
    assert not (store.path / "FAILED.json").exists()


def test_partial_finalization_recovers_append_only_in_new_attempt(tmp_path):
    full_root = tmp_path / "benchmark_v2_5" / "full"
    partial = full_root / "finalization_attempt_001"
    partial.mkdir(parents=True)
    old_partial = partial / "raw_results.csv"
    old_partial.write_bytes(b"partial-attempt-one\n")
    old_hash = hashlib.sha256(old_partial.read_bytes()).hexdigest()
    legacy_root_partial = full_root / "aggregate_statistics.csv"
    legacy_root_partial.write_bytes(b"legacy-root-partial\n")
    payloads = {
        "full_experiment_report.md": b"# complete fixture\n",
        "raw_results.csv": b"generator,seed,status\nfixture,1,COMPLETE\n",
        "aggregate_statistics.csv": b"generator,status\nfixture,VALID\n",
        "effect_sizes_and_holm.csv": b"generator,status\nfixture,VALID\n",
        "per_seed_runtime.csv": b"generator,seed,total_seconds\nfixture,1,1\n",
        "analysis.json": b'{"status":"COMPLETE"}\n',
    }
    result = publish_finalization_artifacts(
        full_root=full_root,
        artifacts=payloads,
        provenance={"manifest_hash": "a" * 64},
    )
    second = full_root / "finalization_attempt_002"
    marker = json.loads((full_root / "FINAL_COMPLETE.json").read_text())
    index = json.loads((second / "artifact_index.json").read_text())
    checksum = json.loads(
        (second / "checksum_manifest_report.json").read_text()
    )
    assert result["status"] == "FINAL_COMPLETE"
    assert result["attempt"] == 2
    assert marker["finalization_attempt"] == "finalization_attempt_002"
    assert second.is_dir()
    assert hashlib.sha256(old_partial.read_bytes()).hexdigest() == old_hash
    assert legacy_root_partial.read_bytes() == b"legacy-root-partial\n"
    assert index["artifact_count"] == len(payloads) + 1
    assert checksum["artifact_index_sha256"] == sha256(
        second / "artifact_index.json"
    )
    assert json.loads(
        (second / "prior_partial_finalizations.json").read_text()
    )["attempts"][0]["status"] == "PARTIAL"


def test_secondary_failure_does_not_block_c2_primary_jobs(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    jobs = build_job_plan(raw, sampling_plan_hash=PLAN_HASH)
    executed = []
    cancelled = []

    def execute(job):
        executed.append((job.generator, job.seed))
        if job.generator == "block_2" and job.seed == 1:
            return {
                "status": "FAILED",
                "failure_class": "code",
                "failure_fingerprint": "fixture-secondary",
                "attempt": 1,
                "manifest_hash": "a" * 64,
            }
        return {
            "status": "COMPLETE",
            "failure_class": None,
            "attempt": 1,
            "manifest_hash": "a" * 64,
        }

    def cancel(job, reason):
        cancelled.append((job.generator, job.seed, reason))
        return {
            "status": "CANCELLED",
            "failure_class": "dependency_cancelled",
            "attempt": 1,
            "manifest_hash": "a" * 64,
        }

    state = FullRunStateStore(
        tmp_path / "benchmark_v2_5" / "full",
        run_manifest_hash="b" * 64,
    )
    result = execute_job_schedule(
        jobs,
        execute=execute,
        cancel=cancel,
        run_state=state,
    )
    assert result["mandatory_stopped"] is False
    assert {
        generator
        for generator, unused_seed in executed
        if generator
        in {
            "empirical_iid",
            "ctgan_separate_class",
            "tvae_separate_class",
            "cof_seqgen",
        }
    } == {
        "empirical_iid",
        "ctgan_separate_class",
        "tvae_separate_class",
        "cof_seqgen",
    }
    assert ("block_2", 2) not in executed
    assert [(generator, seed) for generator, seed, unused in cancelled] == [
        ("block_2", 2),
        ("block_2", 3),
        ("block_2", 4),
        ("block_2", 5),
    ]
    events = [
        json.loads(line)
        for line in (state.root / "run_state.jsonl").read_text().splitlines()
    ]
    assert all(
        {"generator", "seed", "attempt", "status", "failure_class",
         "timestamp", "manifest_hash"}
        <= event.keys()
        for event in events
    )


def test_primary_invalid_makes_c2_not_evaluable_without_stopping_others(
    tmp_path,
):
    raw = yaml.safe_load(CONFIG.read_text())
    jobs = build_job_plan(raw, sampling_plan_hash=PLAN_HASH)
    executed = []

    def execute(job):
        executed.append((job.generator, job.seed))
        return {
            "status": (
                "INVALID"
                if job.generator == "empirical_iid" and job.seed == 1
                else "COMPLETE"
            ),
            "failure_class": (
                "hard_guard"
                if job.generator == "empirical_iid" and job.seed == 1
                else None
            ),
            "attempt": 1,
            "manifest_hash": "a" * 64,
        }

    state = FullRunStateStore(
        tmp_path / "benchmark_v2_5" / "full",
        run_manifest_hash="b" * 64,
    )
    result = execute_job_schedule(
        jobs,
        execute=execute,
        cancel=lambda job, reason: pytest.fail("primary was cancelled"),
        run_state=state,
    )
    decision = apply_final_decision_contract(
        {
            "primary_family": [
                "ctgan_separate_class",
                "tvae_separate_class",
                "empirical_iid",
            ],
            "secondary_family": [
                "neural_sequence",
                "independent_markov",
                "joint_markov",
                "plug_in_hmm",
                "plug_in_hsmm",
            ],
            "c2_supported": True,
        },
        c2_evaluable=result["c2_evaluable"],
    )
    assert result["all_jobs_terminal"] is True
    assert len(executed) == 65
    assert result["c2_evaluable"] is False
    assert decision["c2_status"] == "NOT_EVALUABLE"
    assert decision["c2_supported"] is False
    assert decision["c2_decision"] == (
        "C2 was not evaluable in this preregistered experiment."
    )


def test_primary_row_guard_invalid_still_finalizes_all_65_terminal_jobs(
    tmp_path,
):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["scope"]["bootstrap_resamples"] = 20
    jobs = build_job_plan(raw, sampling_plan_hash=PLAN_HASH)
    artifact_root = tmp_path / "benchmark_v2_5"
    hard_pass = {
        "canonical_mask_and_zero_padding": "PASS",
        "train_discrete_support": "PASS",
        "c0_c1_reference_contract": "PASS",
        "row_marginal_guards": "PASS",
    }
    support = {
        "4_bin": {
            "all_bins_valid": True,
            "dropped_bin_macro_gap": 0.01,
            "occupancy_penalized_gap": 0.01,
            "invalid_bin_count": 0,
        },
        "8_bin": {
            "all_bins_valid": False,
            "dropped_bin_macro_gap": 0.02,
            "occupancy_penalized_gap": 0.145,
            "invalid_bin_count": 1,
        },
    }
    for job in jobs:
        attempt = (
            artifact_root
            / "full"
            / job.scenario
            / f"kappa_{job.kappa:.2f}"
            / job.generator
            / f"seed_{job.seed}"
            / "attempt_001"
        )
        attempt.mkdir(parents=True)
        (attempt / "manifest.json").write_text(
            json.dumps({"sampling_plan_hash": PLAN_HASH})
        )
        primary_invalid = (
            job.generator == "empirical_iid" and job.seed == 1
        )
        if primary_invalid:
            (attempt / "INVALID.json").write_text(
                json.dumps(
                    {
                        "status": "INVALID",
                        "hard_guards": {
                            **hard_pass,
                            "row_marginal_guards": "FAIL",
                        },
                        "association_recovery_error": None,
                    }
                )
            )
            (attempt / "evaluation.json").write_text(
                json.dumps(
                    {
                        "diagnostics": {
                            "support_diagnostics": support,
                        }
                    }
                )
            )
        else:
            error = 0.01 + job.seed / 10_000
            (attempt / "COMPLETE.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "hard_guards": hard_pass,
                        "association_recovery_error": error,
                    }
                )
            )
            (attempt / "metrics.json").write_text(
                json.dumps(
                    {
                        "hard_guards": hard_pass,
                        "association_recovery_error": error,
                        "support_diagnostics": support,
                    }
                )
            )
            (attempt / "runtime.json").write_text(
                json.dumps({"total_seconds": 1.0})
            )
            (attempt / "evaluation.json").write_text(
                json.dumps(
                    {
                        "diagnostics": {
                            "support_diagnostics": support,
                        }
                    }
                )
            )

    finalized = _finalize_full_results(
        artifact_root=artifact_root,
        jobs=jobs,
        raw=raw,
        c2_evaluable=False,
        run_manifest_hash="a" * 64,
        continuation_provenance={
            "preserved_attempt": 2,
            "new_attempt": 3,
            "reused_terminal_jobs": 48,
            "executed_pending_jobs": 17,
            "attempt_002_tree_sha256": "b" * 64,
            "attempt_002_terminal_marker_tree_sha256": "c" * 64,
        },
    )

    final_marker = artifact_root / "full" / "FINAL_COMPLETE.json"
    final_attempt = (
        artifact_root
        / "full"
        / finalized["final_marker"]["finalization_attempt"]
    )
    assert len(jobs) == 65
    assert sum(
        1
        for job in jobs
        if any(
            (
                artifact_root
                / "full"
                / job.scenario
                / f"kappa_{job.kappa:.2f}"
                / job.generator
                / f"seed_{job.seed}"
                / "attempt_001"
                / terminal
            ).is_file()
            for terminal in (
                "COMPLETE.json",
                "FAILED.json",
                "INVALID.json",
                "UNAVAILABLE.json",
                "CANCELLED.json",
            )
        )
    ) == 65
    assert final_marker.is_file()
    assert finalized["analysis"]["c2_status"] == "NOT_EVALUABLE"
    assert finalized["analysis"]["c2_aggregate_computed"] is False
    support_report = (
        final_attempt / "support_diagnostics.csv"
    ).read_text()
    markdown_report = (
        final_attempt / "full_experiment_report.md"
    ).read_text()
    assert "4-bin" in markdown_report
    assert "8-bin" in markdown_report
    assert "dropped_bin_macro_gap" in support_report
    assert "occupancy_penalized_gap" in support_report
    assert "invalid_bin_count" in support_report
    checksum = json.loads(
        (final_attempt / "checksum_manifest_report.json").read_text()
    )
    assert checksum["provenance"]["continuation"] == {
        "preserved_attempt": 2,
        "new_attempt": 3,
        "reused_terminal_jobs": 48,
        "executed_pending_jobs": 17,
        "attempt_002_tree_sha256": "b" * 64,
        "attempt_002_terminal_marker_tree_sha256": "c" * 64,
    }


def test_three_consecutive_infrastructure_failures_force_mandatory_stop(
    tmp_path,
):
    raw = yaml.safe_load(CONFIG.read_text())
    jobs = build_job_plan(raw, sampling_plan_hash=PLAN_HASH)
    executed = []

    def execute(job):
        executed.append((job.generator, job.seed))
        return {
            "status": "FAILED",
            "failure_class": "infrastructure",
            "failure_fingerprint": "scheduler-link-lost",
            "attempt": 1,
            "manifest_hash": "a" * 64,
        }

    state = FullRunStateStore(
        tmp_path / "benchmark_v2_5" / "full",
        run_manifest_hash="b" * 64,
    )
    result = execute_job_schedule(
        jobs,
        execute=execute,
        cancel=lambda job, reason: pytest.fail("unexpected cancel"),
        run_state=state,
    )
    stopped = json.loads((state.root / "STOPPED.json").read_text())
    assert result["mandatory_stopped"] is True
    assert result["all_jobs_terminal"] is False
    assert len(executed) == 3
    assert stopped["mandatory"] is True
    assert "three consecutive identical infrastructure failures" in (
        stopped["reason"]
    )
    assert not (state.root / "COMPLETE.json").exists()


@pytest.mark.parametrize(
    "mutation",
    (
        lambda raw: raw["baselines"].pop("cof_seqgen"),
        lambda raw: raw["scope"].update(model_seeds=[1, 2, 3, 4]),
        lambda raw: raw["scope"].update(kappas=[0.0]),
    ),
)
def test_plan_rejects_nonfrozen_generator_seed_or_kappa(mutation):
    raw = yaml.safe_load(CONFIG.read_text())
    mutation(raw)
    with pytest.raises(ValueError, match="frozen|exact 13"):
        build_job_plan(raw, sampling_plan_hash=PLAN_HASH)


def test_full_authorization_manifest_is_mandatory(tmp_path):
    with pytest.raises(AuthorizationError, match="missing"):
        validate_full_authorization(
            tmp_path / "authorization.json",
            repository_root=Path("."),
            config_path=CONFIG,
            data_manifest_path=tmp_path / "data_manifest.json",
        )


def test_full_authorization_mismatch_fails_closed(tmp_path):
    auth_path, data_path = write_authorized_snapshot(
        tmp_path,
        config_sha256="0" * 64,
    )
    with pytest.raises(AuthorizationError, match="config"):
        validate_full_authorization(
            auth_path,
            repository_root=Path("."),
            config_path=CONFIG,
            data_manifest_path=data_path,
        )


def test_full_authorization_verifies_embedded_approval_text_hash(tmp_path):
    auth_path, data_path = write_authorized_snapshot(tmp_path)
    validated = validate_full_authorization(
        auth_path,
        repository_root=Path("."),
        config_path=CONFIG,
        data_manifest_path=data_path,
    )
    assert validated["scope"]["seeds"] == [1, 2, 3, 4, 5]
    value = json.loads(auth_path.read_text())
    value["approval_text"] = "tampered approval"
    auth_path.write_text(json.dumps(value, sort_keys=True) + "\n")
    with pytest.raises(AuthorizationError, match="approval text hash"):
        validate_full_authorization(
            auth_path,
            repository_root=Path("."),
            config_path=CONFIG,
            data_manifest_path=data_path,
        )


def test_full_mode_checks_authorization_before_any_data_or_job_action(
    tmp_path,
):
    with pytest.raises(AuthorizationError, match="missing"):
        run_full_experiment(
            repository_root=Path("."),
            config_path=CONFIG,
            data_manifest_path=tmp_path / "missing-data.json",
            authorization_path=tmp_path / "missing-authorization.json",
            artifact_root=tmp_path / "benchmark_v2_5",
            data_root=tmp_path / "data" / "benchmark_v2_5",
        )
    assert list(tmp_path.rglob("attempt_*")) == []


def test_gpu_selector_excludes_busy_or_nonidle_devices():
    rows = [
        {
            "physical_index": 0,
            "uuid": "GPU-0",
            "name": "RTX",
            "memory_used_mib": 10,
            "utilization_percent": 0,
        },
        {
            "physical_index": 1,
            "uuid": "GPU-1",
            "name": "RTX",
            "memory_used_mib": 18,
            "utilization_percent": 0,
        },
        {
            "physical_index": 2,
            "uuid": "GPU-2",
            "name": "RTX",
            "memory_used_mib": 2048,
            "utilization_percent": 0,
        },
        {
            "physical_index": 3,
            "uuid": "GPU-3",
            "name": "RTX",
            "memory_used_mib": 20,
            "utilization_percent": 8,
        },
    ]
    selected, excluded = select_idle_gpus(
        rows,
        process_uuids={"GPU-0"},
    )
    assert [gpu["physical_index"] for gpu in selected] == [1]
    assert {gpu["physical_index"] for gpu in excluded} == {0, 2, 3}


def test_live_scheduler_excludes_one_busy_gpu_and_uses_three_idle_gpus(
    tmp_path,
):
    raw = yaml.safe_load(CONFIG.read_text())
    jobs = [
        job
        for job in build_job_plan(raw, sampling_plan_hash=PLAN_HASH)
        if job.device_class == "gpu"
    ][:6]
    rows = [
        {
            "physical_index": index,
            "uuid": f"GPU-{index}",
            "name": "RTX",
            "memory_used_mib": 20,
            "utilization_percent": 14 if index == 2 else 0,
        }
        for index in range(4)
    ]
    assigned_waves = []
    calls = {"inventory": 0, "dgp": 0, "gpu": 0, "fit": 0, "sample": 0}

    def inventory_provider():
        calls["inventory"] += 1
        return rows, set()

    def execute_wave(assignments):
        assigned_waves.append(
            [assignment.gpu["physical_index"] for assignment in assignments]
        )
        return [
            {
                "status": "COMPLETE",
                "attempt": 3,
                "manifest_hash": "a" * 64,
                "failure_class": None,
            }
            for assignment in assignments
        ]

    state = FullRunStateStore(
        tmp_path / "benchmark_v2_5" / "full",
        run_manifest_hash="b" * 64,
    )
    result = run_live_gpu_schedule(
        jobs,
        inventory_provider=inventory_provider,
        execute_wave=execute_wave,
        cancel=lambda job, reason: pytest.fail("unexpected cancellation"),
        run_state=state,
        max_concurrency=4,
        wait_budget_seconds=60,
        poll_interval_seconds=5,
    )

    assert result["status"] == "COMPLETE"
    assert assigned_waves == [[0, 1, 3], [0, 1, 3]]
    assert not (state.root / "STOPPED.json").exists()
    assert calls == {
        "inventory": 2,
        "dgp": 0,
        "gpu": 0,
        "fit": 0,
        "sample": 0,
    }


def test_live_scheduler_reduces_concurrency_to_two_idle_gpus(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    jobs = [
        job
        for job in build_job_plan(raw, sampling_plan_hash=PLAN_HASH)
        if job.device_class == "gpu"
    ][:5]
    rows = [
        {
            "physical_index": index,
            "uuid": f"GPU-{index}",
            "name": "RTX",
            "memory_used_mib": 20,
            "utilization_percent": 14 if index in {2, 3} else 0,
        }
        for index in range(4)
    ]
    assigned_waves = []

    def execute_wave(assignments):
        assigned_waves.append(
            [assignment.gpu["physical_index"] for assignment in assignments]
        )
        return [
            {
                "status": "COMPLETE",
                "attempt": 3,
                "manifest_hash": "a" * 64,
                "failure_class": None,
            }
            for assignment in assignments
        ]

    state = FullRunStateStore(
        tmp_path / "benchmark_v2_5" / "full",
        run_manifest_hash="b" * 64,
    )
    result = run_live_gpu_schedule(
        jobs,
        inventory_provider=lambda: (rows, set()),
        execute_wave=execute_wave,
        cancel=lambda job, reason: pytest.fail("unexpected cancellation"),
        run_state=state,
        max_concurrency=4,
        wait_budget_seconds=60,
        poll_interval_seconds=5,
    )

    assert result["status"] == "COMPLETE"
    assert assigned_waves == [[0, 1], [0, 1], [0]]
    assert not (state.root / "STOPPED.json").exists()


def test_live_scheduler_reduces_concurrency_to_one_idle_gpu(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    jobs = [
        job
        for job in build_job_plan(raw, sampling_plan_hash=PLAN_HASH)
        if job.device_class == "gpu"
    ][:3]
    rows = [
        {
            "physical_index": index,
            "uuid": f"GPU-{index}",
            "name": "RTX",
            "memory_used_mib": 20,
            "utilization_percent": 0 if index == 1 else 14,
        }
        for index in range(4)
    ]
    assigned_waves = []

    def execute_wave(assignments):
        assigned_waves.append(
            [assignment.gpu["physical_index"] for assignment in assignments]
        )
        return [
            {
                "status": "COMPLETE",
                "attempt": 3,
                "manifest_hash": "a" * 64,
                "failure_class": None,
            }
            for assignment in assignments
        ]

    state = FullRunStateStore(
        tmp_path / "benchmark_v2_5" / "full",
        run_manifest_hash="b" * 64,
    )
    result = run_live_gpu_schedule(
        jobs,
        inventory_provider=lambda: (rows, set()),
        execute_wave=execute_wave,
        cancel=lambda job, reason: pytest.fail("unexpected cancellation"),
        run_state=state,
        max_concurrency=4,
        wait_budget_seconds=60,
        poll_interval_seconds=5,
    )

    assert result["status"] == "COMPLETE"
    assert assigned_waves == [[1], [1], [1]]
    assert not (state.root / "STOPPED.json").exists()


def test_live_scheduler_waits_append_only_and_resumes_when_gpu_is_idle(
    tmp_path,
):
    raw = yaml.safe_load(CONFIG.read_text())
    job = next(
        job
        for job in build_job_plan(raw, sampling_plan_hash=PLAN_HASH)
        if job.device_class == "gpu"
    )
    busy = [
        {
            "physical_index": index,
            "uuid": f"GPU-{index}",
            "name": "RTX",
            "memory_used_mib": 20,
            "utilization_percent": 14,
        }
        for index in range(4)
    ]
    idle = [
        {
            **row,
            "utilization_percent": 0,
        }
        for row in busy
    ]
    inventories = iter(((busy, set()), (busy, set()), (idle, set())))
    clock = {"now": 0.0}
    assigned = []

    def sleep(seconds):
        clock["now"] += seconds

    def execute_wave(assignments):
        assigned.extend(assignments)
        return [
            {
                "status": "COMPLETE",
                "attempt": 3,
                "manifest_hash": "a" * 64,
                "failure_class": None,
            }
            for unused in assignments
        ]

    state = FullRunStateStore(
        tmp_path / "benchmark_v2_5" / "full",
        run_manifest_hash="b" * 64,
    )
    result = run_live_gpu_schedule(
        [job],
        inventory_provider=lambda: next(inventories),
        execute_wave=execute_wave,
        cancel=lambda unused_job, reason: pytest.fail(
            "unexpected cancellation"
        ),
        run_state=state,
        max_concurrency=4,
        wait_budget_seconds=20,
        poll_interval_seconds=5,
        monotonic=lambda: clock["now"],
        sleep=sleep,
    )

    wait = state.root / "gpu_wait_attempt_001"
    heartbeats = [
        json.loads(line)
        for line in (wait / "heartbeat.jsonl").read_text().splitlines()
    ]
    assert result["status"] == "COMPLETE"
    assert (wait / "WAITING_FOR_GPU.json").is_file()
    assert (wait / "RESUMED.json").is_file()
    assert [row["elapsed_wait_seconds"] for row in heartbeats] == [0.0, 5.0]
    assert len(assigned) == 1
    assert not (state.root / "STOPPED.json").exists()


def test_live_scheduler_stops_only_after_zero_idle_wait_budget(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    job = next(
        job
        for job in build_job_plan(raw, sampling_plan_hash=PLAN_HASH)
        if job.device_class == "gpu"
    )
    busy = [
        {
            "physical_index": index,
            "uuid": f"GPU-{index}",
            "name": "RTX",
            "memory_used_mib": 20,
            "utilization_percent": 14,
        }
        for index in range(4)
    ]
    clock = {"now": 0.0}
    calls = {"inventory": 0, "wave": 0}

    def inventory_provider():
        calls["inventory"] += 1
        return busy, set()

    def sleep(seconds):
        clock["now"] += seconds

    def execute_wave(unused_assignments):
        calls["wave"] += 1
        pytest.fail("no wave may run without an idle GPU")

    state = FullRunStateStore(
        tmp_path / "benchmark_v2_5" / "full",
        run_manifest_hash="b" * 64,
    )
    result = run_live_gpu_schedule(
        [job],
        inventory_provider=inventory_provider,
        execute_wave=execute_wave,
        cancel=lambda unused_job, reason: pytest.fail(
            "unexpected cancellation"
        ),
        run_state=state,
        max_concurrency=4,
        wait_budget_seconds=10,
        poll_interval_seconds=5,
        monotonic=lambda: clock["now"],
        sleep=sleep,
    )

    wait = state.root / "gpu_wait_attempt_001"
    heartbeats = (wait / "heartbeat.jsonl").read_text().splitlines()
    stopped = json.loads((state.root / "STOPPED.json").read_text())
    assert result["status"] == "STOPPED"
    assert len(heartbeats) == 3
    assert (wait / "WAIT_BUDGET_EXHAUSTED.json").is_file()
    assert not (wait / "RESUMED.json").exists()
    assert stopped["mandatory"] is True
    assert "wait budget exhausted" in stopped["reason"]
    assert "no external process was terminated" in stopped["reason"]
    assert calls == {"inventory": 3, "wave": 0}


def test_scheduler_continuation_reuses_48_terminal_jobs_and_plans_17_attempt_003(
    tmp_path,
):
    raw = yaml.safe_load(CONFIG.read_text())
    jobs = build_job_plan(raw, sampling_plan_hash=PLAN_HASH)
    artifact_root = tmp_path / "benchmark_v2_5"
    prior_source = "1" * 40
    prior_code = "2" * 64
    config_hash = "3" * 64
    data_hashes = {
        "train": "4" * 64,
        "validation": "5" * 64,
        "test": "6" * 64,
    }
    terminal_status = {}
    for job in jobs:
        if job.generator in {
            "empirical_iid",
            "block_2",
            "block_4",
            "block_8",
            "full_sequence_reference",
            "independent_markov",
            "joint_markov",
            "plug_in_hmm",
        }:
            terminal_status[(job.generator, job.seed)] = "COMPLETE"
        elif job.generator == "plug_in_hsmm":
            terminal_status[(job.generator, job.seed)] = (
                "INVALID" if job.seed == 1 else "CANCELLED"
            )
        elif job.generator == "ctgan_separate_class" and job.seed <= 3:
            terminal_status[(job.generator, job.seed)] = "INVALID"

    before = {}
    for job in jobs:
        status = terminal_status.get((job.generator, job.seed))
        if status is None:
            continue
        attempt = (
            artifact_root
            / "full"
            / job.scenario
            / f"kappa_{job.kappa:.2f}"
            / job.generator
            / f"seed_{job.seed}"
            / "attempt_002"
        )
        attempt.mkdir(parents=True)
        manifest = {
            "git_commit": prior_source,
            "code_hash": prior_code,
            "config_hash": config_hash,
            "scenario": job.scenario,
            "kappa": job.kappa,
            "generator": job.generator,
            "seed": job.seed,
            "sampling_plan_hash": PLAN_HASH,
            "data_hashes": data_hashes,
            "baseline_definition_version": "benchmark-v2.5",
            "evaluation_version": "benchmark-v2.5-evaluation-v1",
        }
        marker = {
            "status": status,
            "failure_class": (
                "hard_guard"
                if status == "INVALID"
                else (
                    "dependency_cancelled"
                    if status == "CANCELLED"
                    else None
                )
            ),
        }
        (attempt / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n"
        )
        marker_path = attempt / f"{status}.json"
        marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n")
        before[attempt] = {
            path.name: sha256(path)
            for path in sorted(attempt.iterdir())
        }

    attempt_files = sorted(
        path
        for path in (artifact_root / "full").rglob("*")
        if path.is_file() and "attempt_002" in path.parts
    )
    terminal_files = [
        path
        for path in attempt_files
        if path.name
        in {
            "COMPLETE.json",
            "FAILED.json",
            "INVALID.json",
            "UNAVAILABLE.json",
            "CANCELLED.json",
        }
    ]

    def tree_digest(paths):
        payload = "".join(
            f"{sha256(path)}  "
            f"{path.relative_to(artifact_root / 'full').as_posix()}\n"
            for path in paths
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    result = plan_scheduler_continuation(
        artifact_root=artifact_root,
        jobs=jobs,
        continuation={
            "previous_source_commit": prior_source,
            "previous_relevant_code_sha256": prior_code,
            "config_sha256": config_hash,
            "data_hashes": data_hashes,
            "sampling_plan_sha256": PLAN_HASH,
            "terminal_attempt": 2,
            "new_attempt": 3,
            "expected_terminal_jobs": 48,
            "expected_pending_jobs": 17,
            "attempt_tree_sha256": tree_digest(attempt_files),
            "terminal_marker_tree_sha256": tree_digest(terminal_files),
        },
    )

    pending = {
        (item["job"]["generator"], item["job"]["seed"])
        for item in result["pending"]
    }
    assert result["status"] == "CONTINUATION_READY"
    assert len(result["reused"]) == 48
    assert len(result["pending"]) == 17
    assert {
        (item["job"]["generator"], item["job"]["seed"])
        for item in result["reused"]
    } == set(terminal_status)
    assert pending == {
        ("ctgan_separate_class", 4),
        ("ctgan_separate_class", 5),
        *(("tvae_separate_class", seed) for seed in range(1, 6)),
        *(("neural_sequence", seed) for seed in range(1, 6)),
        *(("cof_seqgen", seed) for seed in range(1, 6)),
    }
    assert all(item["target_attempt"] == 3 for item in result["pending"])
    assert all(item["attempt"] == 2 for item in result["reused"])
    assert all(item["reused"] is True for item in result["reused"])
    assert all(
        {
            path.name: sha256(path)
            for path in sorted(attempt.iterdir())
        }
        == hashes
        for attempt, hashes in before.items()
    )
    assert list(artifact_root.rglob("attempt_003")) == []

    state = FullRunStateStore(
        artifact_root / "full" / "continuation_attempt_003",
        run_manifest_hash="7" * 64,
    )
    policy = RunPolicyController(state)
    replay_continuation_results(
        jobs=jobs,
        continuation_plan=result,
        policy=policy,
    )
    assert len(policy.results) == 48
    assert policy.c2_evaluable is False
    assert any(
        "ctgan_separate_class seed 1 status=INVALID" in reason
        for reason in policy.c2_reasons
    )
    for job in jobs:
        if (job.generator, job.seed) not in pending:
            continue
        policy.observe(
            job,
            {
                "status": "COMPLETE",
                "attempt": 3,
                "manifest_hash": "8" * 64,
                "failure_class": None,
            },
        )
    summary = policy.summary(expected_jobs=65)
    assert summary["all_jobs_terminal"] is True
    assert summary["c2_evaluable"] is False
    assert len(summary["results"]) == 65


@pytest.mark.parametrize(
    "forbidden",
    (
        "configs/benchmark_v2/full_v2_5.yaml",
        "benchmarks/temporal_coupling_v2.py",
        "eval/full_evaluation_v2_5.py",
        "generators/full_registry_v2_5.py",
        "models/cof_model.py",
        "data/benchmark_v2_5/frozen/data_manifest.json",
    ),
)
def test_scheduler_continuation_rejects_non_scheduler_source_changes(
    forbidden,
):
    with pytest.raises(AuthorizationError, match="scheduler-only"):
        validate_scheduler_continuation_changed_paths(
            [
                "scripts/run_full_experiment_v2_5.py",
                forbidden,
            ]
        )


def test_scheduler_assigns_at_most_one_heavy_job_per_gpu():
    raw = yaml.safe_load(CONFIG.read_text())
    jobs = [
        job
        for job in build_job_plan(raw, sampling_plan_hash=PLAN_HASH)
        if job.device_class == "gpu"
    ]
    gpus = [
        {"physical_index": index, "uuid": f"GPU-{index}"}
        for index in (1, 2, 3)
    ]
    waves = build_gpu_waves(jobs, gpus)
    assert sum(len(wave) for wave in waves) == 20
    for wave in waves:
        assigned = [item.gpu["physical_index"] for item in wave]
        assert len(assigned) == len(set(assigned))
        assert len(assigned) <= 3


def test_dry_run_exercises_contracts_without_dgp_gpu_fit_or_sample(tmp_path):
    result = run_dry_run(
        repository_root=Path("."),
        config_path=CONFIG,
        work_root=tmp_path / "runner-dry-run",
    )
    assert result["status"] == "PASS"
    assert result["job_count"] == 65
    assert result["authorization_validated"] is True
    assert result["exact_resume_verified"] is True
    assert result["mismatch_new_attempt_verified"] is True
    assert result["invalid_routing_verified"] is True
    assert result["scheduler_job_count"] == 20
    assert result["calls"] == {
        "dgp": 0,
        "gpu": 0,
        "fit": 0,
        "sample": 0,
    }


def test_final_decision_uses_exact_primary_family_and_fixed_english_failure():
    analysis = {
        "primary_family": [
            "ctgan_separate_class",
            "tvae_separate_class",
            "empirical_iid",
        ],
        "secondary_family": [
            "neural_sequence",
            "independent_markov",
            "joint_markov",
            "plug_in_hmm",
            "plug_in_hsmm",
        ],
        "c2_supported": False,
        "c2_decision": "superseded presentation text",
        "pairwise": {
            "neural_sequence": {
                "mean_effect": 100.0,
                "holm_adjusted_p": 0.0,
                "hedges_g": 99.0,
            }
        },
    }
    fixed = apply_final_decision_contract(analysis)
    assert fixed["primary_family"] == [
        "ctgan_separate_class",
        "tvae_separate_class",
        "empirical_iid",
    ]
    assert fixed["c2_supported"] is False
    assert fixed["c2_decision"] == (
        "C2 was not supported in this preregistered experiment."
    )


def test_v2_5_runtime_artifacts_are_not_git_commit_targets():
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "artifacts/benchmark_v2_5/full/future-runtime.json",
            "data/benchmark_v2_5/future-data.npz",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert len(result.stdout.splitlines()) == 2
