"""Append-only external validation v1 runner.

Plan and dry-run validate provenance without importing a generator, loading an
NPZ array, touching raw CSV data, or querying an accelerator. Execution is
fail-closed until a separate authorization is provided.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import yaml


class ExternalValidationError(RuntimeError):
    """Raised when external validation provenance or scope is invalid."""


ZERO_EXECUTION_COUNTS = {
    "gpu_inventory_queries": 0,
    "cuda_calls": 0,
    "model_import_calls": 0,
    "model_fit_calls": 0,
    "checkpoint_write_calls": 0,
    "model_sample_calls": 0,
    "validation_metric_calls": 0,
    "internal_test_load_calls": 0,
    "sparkov_fraud_test_accesses": 0,
    "tstr_calls": 0,
    "privacy_calls": 0,
    "full_run_calls": 0,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def claim_external_validation_attempt(
    attempt_path: Path, manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Claim one dataset/model attempt with immutable exclusive ownership."""

    required = {
        "dataset",
        "model",
        "source_sha256",
        "config_sha256",
        "train_sha256",
        "validation_sha256",
    }
    if not required.issubset(manifest):
        raise ExternalValidationError("attempt manifest lacks frozen provenance")
    if attempt_path.exists():
        raise ExternalValidationError("external validation attempt already exists")
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        attempt_path.mkdir()
    except FileExistsError as error:
        raise ExternalValidationError(
            "external validation attempt already exists"
        ) from error
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    manifest_hash = hashlib.sha256(canonical).hexdigest()
    _exclusive_json(
        attempt_path / "OWNERSHIP.json",
        {
            "schema_version": "external-validation-ownership-v1",
            "dataset": manifest["dataset"],
            "model": manifest["model"],
            "manifest": dict(manifest),
            "manifest_sha256": manifest_hash,
            "claimed_at": _utc_now(),
            "append_only": True,
        },
    )
    running = {
        "schema_version": "external-validation-running-v1",
        "status": "RUNNING",
        "dataset": manifest["dataset"],
        "model": manifest["model"],
        "manifest_sha256": manifest_hash,
        "started_at": _utc_now(),
    }
    _exclusive_json(attempt_path / "RUNNING.json", running)
    return running


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ExternalValidationError(f"JSON manifest is not an object: {path}")
    return value


def _load_config(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ExternalValidationError("external validation config must be a mapping")
    return value


def _git_blob(repo_root: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _tree_digest(bundle_root: Path, artifact_root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for root in (bundle_root, artifact_root):
        root_label = str(root)
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = f"{root_label}:{path.relative_to(root)}".encode("utf-8")
            payload = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            count += 1
            size += len(payload)
    return digest.hexdigest(), count, size


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_only_tree_inventory(root: Path) -> Mapping[str, Any]:
    if not root.is_dir():
        raise ExternalValidationError(f"preserved artifact tree is missing: {root}")
    files: dict[str, Mapping[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files[path.relative_to(root).as_posix()] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "tree_sha256": _canonical_sha256({"files": files}),
        "file_count": len(files),
        "byte_count": sum(record["bytes"] for record in files.values()),
    }


def _expected_attempt(config: Mapping[str, Any], model: str) -> str:
    attempts = config["runtime"]["attempts_by_model"]
    try:
        return str(attempts[model])
    except KeyError as error:
        raise ExternalValidationError(
            f"continuation attempt is not registered for model: {model}"
        ) from error


_EXPECTED_LAUNCH_WAVES = (
    (
        "cpu_wave",
        "cpu",
        (
            ("amlsim", "empirical_iid", "cpu_a", "cpu", 0),
            ("sparkov", "empirical_iid", "cpu_b", "cpu", 0),
        ),
    ),
    (
        "amlsim_gpu_wave_1",
        "gpu",
        (
            (
                "amlsim",
                "ctgan_separate_class",
                "physical_gpu_a",
                "cuda:0",
                1,
            ),
            (
                "amlsim",
                "cof_seqgen_frozen_non_v3",
                "physical_gpu_c",
                "cuda:0",
                1,
            ),
        ),
    ),
    (
        "amlsim_gpu_wave_2",
        "gpu",
        (
            (
                "amlsim",
                "tvae_separate_class",
                "physical_gpu_b",
                "cuda:0",
                1,
            ),
        ),
    ),
    (
        "sparkov_gpu_wave_1",
        "gpu",
        (
            (
                "sparkov",
                "ctgan_separate_class",
                "physical_gpu_a",
                "cuda:0",
                1,
            ),
            (
                "sparkov",
                "cof_seqgen_frozen_non_v3",
                "physical_gpu_c",
                "cuda:0",
                1,
            ),
        ),
    ),
    (
        "sparkov_gpu_wave_2",
        "gpu",
        (
            (
                "sparkov",
                "tvae_separate_class",
                "physical_gpu_b",
                "cuda:0",
                1,
            ),
        ),
    ),
)


def validate_external_launch_schedule(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the scheduling-only two/three/three wave contract."""

    schedule = config.get("launch_schedule")
    if not isinstance(schedule, Mapping):
        raise ExternalValidationError("external launch schedule is missing")
    if (
        schedule.get("schema_version")
        != "external-validation-memory-safe-five-wave-schedule-v1"
        or schedule.get("scheduling_only") is not True
        or schedule.get("scientific_contract_unchanged") is not True
    ):
        raise ExternalValidationError("external launch schedule scope changed")
    waves = schedule.get("waves")
    if not isinstance(waves, list) or len(waves) != 5:
        raise ExternalValidationError("external launch schedule requires five waves")
    observed_pairs: set[tuple[str, str]] = set()
    for wave, (wave_id, device_class, expected_jobs) in zip(
        waves, _EXPECTED_LAUNCH_WAVES
    ):
        if not isinstance(wave, Mapping) or (
            wave.get("wave_id") != wave_id
            or wave.get("device_class") != device_class
            or wave.get("parallel") is not True
            or wave.get("terminal_barrier_after") is not True
        ):
            raise ExternalValidationError(
                f"external launch wave contract changed: {wave_id}"
            )
        jobs = wave.get("jobs")
        if not isinstance(jobs, list) or len(jobs) != len(expected_jobs):
            raise ExternalValidationError(
                f"external launch wave job count changed: {wave_id}"
            )
        for job, expected in zip(jobs, expected_jobs):
            if not isinstance(job, Mapping):
                raise ExternalValidationError("external launch job is not a mapping")
            observed = (
                str(job.get("dataset")),
                str(job.get("model")),
                str(job.get("device_slot")),
                str(job.get("runner_device")),
                int(job.get("cuda_visible_devices_count", -1)),
            )
            if observed != expected:
                raise ExternalValidationError(
                    f"external launch job mapping changed: {wave_id}"
                )
            pair = observed[:2]
            if pair in observed_pairs:
                raise ExternalValidationError("external launch job is duplicated")
            observed_pairs.add(pair)
        if device_class == "gpu" and len(
            {str(job["device_slot"]) for job in jobs}
        ) != len(jobs):
            raise ExternalValidationError(
                f"GPU workers must use distinct physical slots: {wave_id}"
            )
    expected_pairs = {
        (dataset, model)
        for _, _, jobs in _EXPECTED_LAUNCH_WAVES
        for dataset, model, _, _, _ in jobs
    }
    if observed_pairs != expected_pairs:
        raise ExternalValidationError("external launch schedule matrix changed")
    memory_policy = config.get("memory_safety", {}).get("data_transformer", {})
    heavy_models = set(memory_policy.get("models", ()))
    if (
        heavy_models != {"ctgan_separate_class", "tvae_separate_class"}
        or int(memory_policy.get("fixed_n_jobs", -1)) != 1
        or memory_policy.get("unbounded_n_jobs_forbidden") is not True
        or int(memory_policy.get("max_concurrent_heavy_transforms", -1)) != 1
    ):
        raise ExternalValidationError("external transform memory policy changed")
    for wave in waves:
        heavy_count = sum(
            job["model"] in heavy_models for job in wave["jobs"]
        )
        if heavy_count > 1:
            raise ExternalValidationError(
                "CTGAN and TVAE heavy transforms cannot share a wave"
            )
    return json.loads(json.dumps(schedule, allow_nan=False))


def _launch_job_contract(
    config: Mapping[str, Any], dataset: str, model: str
) -> Mapping[str, Any]:
    schedule = validate_external_launch_schedule(config)
    for wave in schedule["waves"]:
        for job in wave["jobs"]:
            if job["dataset"] == dataset and job["model"] == model:
                memory_policy = config["memory_safety"]["data_transformer"]
                heavy_transform = model in memory_policy["models"]
                return {
                    "wave_id": wave["wave_id"],
                    "device_slot": job["device_slot"],
                    "runner_device": job["runner_device"],
                    "cuda_visible_devices_count": job[
                        "cuda_visible_devices_count"
                    ],
                    "terminal_barrier_after_wave": wave[
                        "terminal_barrier_after"
                    ],
                    "prior_wave_terminal_required": (
                        wave["wave_id"] != "cpu_wave"
                    ),
                    "data_transformer_n_jobs": (
                        int(memory_policy["fixed_n_jobs"])
                        if heavy_transform
                        else 0
                    ),
                    "heavy_transform_exclusion_group": (
                        "external_tabular_transform"
                        if heavy_transform
                        else None
                    ),
                }
    raise ExternalValidationError("external launch job is not scheduled")


def validate_external_wave_barrier(
    *, repo_root: Path, config_path: Path, dataset: str, model: str
) -> Mapping[str, Any]:
    """Require every job in preceding waves to have one terminal marker."""

    config = _load_config(config_path)
    _validate_config(config)
    schedule = validate_external_launch_schedule(config)
    target_wave_index: int | None = None
    for index, wave in enumerate(schedule["waves"]):
        if any(
            job["dataset"] == dataset and job["model"] == model
            for job in wave["jobs"]
        ):
            target_wave_index = index
            break
    if target_wave_index is None:
        raise ExternalValidationError("external launch job is not scheduled")
    terminals: list[Mapping[str, Any]] = []
    runtime_root = repo_root / config["runtime"]["root"]
    marker_names = tuple(config["runtime"]["terminal_markers"])
    for wave in schedule["waves"][:target_wave_index]:
        for job in wave["jobs"]:
            attempt_path = (
                runtime_root
                / job["dataset"]
                / job["model"]
                / _expected_attempt(config, job["model"])
            )
            observed = [
                marker
                for marker in marker_names
                if (attempt_path / marker).is_file()
            ]
            if len(observed) != 1:
                raise ExternalValidationError(
                    "external launch terminal barrier is not satisfied: "
                    f"{wave['wave_id']}/{job['dataset']}/{job['model']}"
                )
            terminals.append(
                {
                    "wave_id": wave["wave_id"],
                    "dataset": job["dataset"],
                    "model": job["model"],
                    "attempt": _expected_attempt(config, job["model"]),
                    "terminal_marker": observed[0],
                }
            )
    return {
        "status": "PASS",
        "target_wave": schedule["waves"][target_wave_index]["wave_id"],
        "prior_terminal_count": len(terminals),
        "prior_terminals": terminals,
        "artifact_writes": 0,
        "gpu_inventory_queries": 0,
    }


def validate_external_continuation_preservation(
    *, repo_root: Path, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Rehash immutable prior attempts and authorization history read-only."""

    correction = config["performance_correction"]
    metric_path = repo_root / correction["metric_source_path"]
    metric_hash = _sha256_file(metric_path)
    if metric_hash != correction["metric_source_sha256"]:
        raise ExternalValidationError("performance-corrected metric source hash mismatch")
    artifacts: dict[str, Mapping[str, Any]] = {}
    for name, expected in config["continuation"]["preserved_artifacts"].items():
        observed = _read_only_tree_inventory(repo_root / expected["path"])
        pinned = {
            "tree_sha256": expected["tree_sha256"],
            "file_count": int(expected["file_count"]),
            "byte_count": int(expected["byte_count"]),
        }
        if observed != pinned:
            raise ExternalValidationError(
                f"preserved continuation artifact hash mismatch: {name}"
            )
        terminal = expected.get("terminal")
        if terminal is not None and not (
            repo_root / expected["path"] / f"{terminal}.json"
        ).is_file():
            raise ExternalValidationError(
                f"preserved continuation terminal mismatch: {name}"
            )
        artifacts[name] = {
            "path": expected["path"],
            **observed,
            **({"terminal": terminal} if terminal is not None else {}),
        }
    return {
        "status": "PASS",
        "metric_source_sha256": metric_hash,
        "artifacts": artifacts,
        "preservation_sha256": _canonical_sha256({"artifacts": artifacts}),
        "mutation_calls": 0,
    }


def validate_external_completed_result_reuse(
    *,
    repo_root: Path,
    config_path: Path,
    dataset: str,
    model: str,
    config_override: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate a completed result under the resource-only reuse exception."""

    config = dict(config_override) if config_override is not None else _load_config(
        config_path
    )
    corrective = config.get("corrective_continuation", {})
    if (
        corrective.get("status") != "source_only_not_authorized"
        or corrective.get("resource_policy_only_change") is not True
        or corrective.get("scientific_contract_unchanged") is not True
    ):
        raise ExternalValidationError("corrective reuse scope changed")
    try:
        record = corrective["completed_result_reuse"][dataset][model]
    except KeyError as error:
        raise ExternalValidationError(
            f"completed result is not registered for reuse: {dataset}/{model}"
        ) from error
    attempt_path = (repo_root / str(record["attempt_path"])).resolve()
    # A continuation may deliberately write to a new isolated runtime root while
    # reusing a hash-bound result from the preserved v1 runtime.  Therefore the
    # reuse source is anchored to the canonical, read-only external-validation
    # tree instead of the *new* target runtime root.
    preserved_runtime_root = (
        repo_root / "artifacts/external_validation_v1"
    ).resolve()
    if preserved_runtime_root not in attempt_path.parents:
        raise ExternalValidationError(
            "reuse artifact is outside preserved external-validation runtime"
        )
    paths = {
        "manifest_sha256": attempt_path / "manifest.json",
        "evaluation_sha256": attempt_path / "evaluation.json",
        "terminal_sha256": attempt_path / "COMPLETE.json",
    }
    mismatched_files = [
        key
        for key, path in paths.items()
        if not path.is_file() or _sha256_file(path) != record.get(key)
    ]
    if mismatched_files:
        raise ExternalValidationError(
            "reuse artifact hash mismatch: " + ", ".join(mismatched_files)
        )
    manifest = _load_json(paths["manifest_sha256"])
    evaluation = _load_json(paths["evaluation_sha256"])
    terminal = _load_json(paths["terminal_sha256"])
    expected = corrective["required_equivalence"]
    equivalence_failures = []
    manifest_expected = {
        "source_commit": expected["source_commit"],
        "config_sha256": expected["config_sha256"],
        "bundle_tree_sha256": expected["bundle_tree_sha256"],
        "train_sha256": expected["train_sha256"],
        "validation_sha256": expected["validation_sha256"],
        "metric_source_sha256": expected["metric_source_sha256"],
        "seed": int(expected["seed"]),
        "dataset": dataset,
        "model": model,
    }
    equivalence_failures.extend(
        key
        for key, value in manifest_expected.items()
        if manifest.get(key) != value
    )
    if evaluation.get("sampling_plan_sha256") != expected["sampling_plan_sha256"]:
        equivalence_failures.append("sampling_plan_sha256")
    if evaluation.get("thresholds_sha256") != expected["thresholds_sha256"]:
        equivalence_failures.append("thresholds_sha256")
    if terminal.get("status") != "COMPLETE":
        equivalence_failures.append("terminal_status")
    if equivalence_failures:
        raise ExternalValidationError(
            "completed result reuse provenance mismatch: "
            + ", ".join(equivalence_failures)
        )
    return {
        "status": "REUSE_ELIGIBLE",
        "dataset": dataset,
        "model": model,
        "attempt_path": str(attempt_path),
        "manifest_sha256": record["manifest_sha256"],
        "evaluation_sha256": record["evaluation_sha256"],
        "terminal_sha256": record["terminal_sha256"],
        "scientific_contract_unchanged": True,
        "resource_policy_only_change": True,
        "model_execution_calls": 0,
        "runtime_artifact_writes": 0,
    }


def _validate_failed_tvae_attempt_preservation(
    *, repo_root: Path, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Rehash the failed AMLSim TVAE attempt and its runner log read-only."""

    expected = config["corrective_continuation"][
        "failed_attempt_preservation"
    ]
    attempt_path = (repo_root / expected["path"]).resolve()
    observed = _read_only_tree_inventory(attempt_path)
    if any(
        observed[key] != expected[key]
        for key in ("tree_sha256", "file_count", "byte_count")
    ):
        raise ExternalValidationError(
            "failed TVAE attempt_001 preservation mismatch"
        )
    file_checks = {
        "manifest_sha256": attempt_path / "manifest.json",
        "failed_sha256": attempt_path / "FAILED.json",
        "runner_log_sha256": repo_root / expected["runner_log_path"],
    }
    mismatches = [
        name
        for name, path in file_checks.items()
        if not path.is_file() or _sha256_file(path) != expected[name]
    ]
    if mismatches:
        raise ExternalValidationError(
            "failed TVAE attempt_001 file hash mismatch: "
            + ", ".join(mismatches)
        )
    failed = _load_json(attempt_path / "FAILED.json")
    if (
        failed.get("status") != "FAILED"
        or failed.get("model") != "tvae_separate_class"
        or failed.get("dataset") != "amlsim"
    ):
        raise ExternalValidationError(
            "failed TVAE attempt_001 terminal contract mismatch"
        )
    return {
        "status": "PRESERVED",
        "path": expected["path"],
        **observed,
        "manifest_sha256": expected["manifest_sha256"],
        "failed_sha256": expected["failed_sha256"],
        "runner_log_path": expected["runner_log_path"],
        "runner_log_sha256": expected["runner_log_sha256"],
        "artifact_writes": 0,
    }


def build_amlsim_tvae_continuation_authorization(
    *, repo_root: Path, config_path: Path, approval_text: str
) -> Mapping[str, Any]:
    """Build the exact AMLSim TVAE attempt_002 grant without writing it."""

    if not approval_text.strip():
        raise ExternalValidationError(
            "AMLSim TVAE continuation requires explicit approval text"
        )
    config = _load_config(config_path)
    _validate_config(config)
    corrective = config["corrective_continuation"]
    target = corrective["next_target"]
    exact_target = {
        "dataset": "amlsim",
        "model": "tvae_separate_class",
        "attempt": "attempt_002",
        "seed": 31001,
        "requested_steps": 20000,
        "training_hard_cap_seconds": 7200,
        "job_hard_cap_seconds": 10800,
        "runner_device": "cuda:0",
        "cuda_visible_devices_count": 1,
        "data_transformer_n_jobs": 1,
        "heavy_transform_exclusion_group": "external_tabular_transform",
        "authorization_required": True,
    }
    if target != exact_target:
        raise ExternalValidationError(
            "AMLSim TVAE continuation target contract changed"
        )
    policy = config["memory_safety"]["data_transformer"]
    if (
        int(policy["fixed_n_jobs"]) != 1
        or policy["unbounded_n_jobs_forbidden"] is not True
        or int(policy["max_concurrent_heavy_transforms"]) != 1
    ):
        raise ExternalValidationError(
            "AMLSim TVAE continuation memory policy changed"
        )
    bundle = _verify_bundle(repo_root, "amlsim", config["datasets"]["amlsim"])
    preservation = validate_external_continuation_preservation(
        repo_root=repo_root, config=config
    )
    reused = {
        model: validate_external_completed_result_reuse(
            repo_root=repo_root,
            config_path=config_path,
            dataset="amlsim",
            model=model,
        )
        for model in (
            "empirical_iid",
            "ctgan_separate_class",
            "cof_seqgen_frozen_non_v3",
        )
    }
    failed = _validate_failed_tvae_attempt_preservation(
        repo_root=repo_root, config=config
    )
    launch = _launch_job_contract(
        config, "amlsim", "tvae_separate_class"
    )
    launch.update(
        {
            "data_transformer_n_jobs": 1,
            "heavy_transform_exclusion_group": (
                "external_tabular_transform"
            ),
        }
    )
    return {
        "schema_version": (
            "external-validation-performance-continuation-job-authorization-v1"
        ),
        "scope": "one_performance_corrective_external_validation_job",
        "continuation_kind": "amlsim_tvae_memory_safe_attempt_002",
        "approval_text": approval_text.strip(),
        "job_count": 1,
        "single_job_only": True,
        "source_commit": _git_head(repo_root),
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "execution_source_sha256": _sha256_file(
            repo_root / "experiments/external_validation_runner_v1.py"
        ),
        "config_sha256": _sha256_file(config_path),
        "metric_source_sha256": preservation["metric_source_sha256"],
        "continuation_preservation_sha256": preservation[
            "preservation_sha256"
        ],
        "dataset": "amlsim",
        "model": "tvae_separate_class",
        "seed": 31001,
        "attempt": "attempt_002",
        "bundle_tree_sha256": bundle["tree_sha256"],
        "train_sha256": bundle["train_sha256"],
        "validation_sha256": bundle["validation_sha256"],
        "allowed_operations": [
            "train_only_model_fit",
            "validation_generation_and_evaluation",
            "train_only_bootstrap",
        ],
        "forbidden_operations": [
            "internal_test_access",
            "sparkov_fraudTest_access",
            "raw_csv_access",
            "controlled_artifact_mutation",
            "test_evaluation",
            "tstr",
            "privacy",
            "materialization_execution",
            "architecture_or_hyperparameter_change",
            "threshold_or_selection_rule_change",
            "full_or_five_seed_execution",
            "retry",
            "early_stopping",
            "hyperparameter_sweep",
            "gpu_inventory_query",
            "existing_artifact_overwrite_delete_move",
            "aggregate_selection",
        ],
        "training_hard_cap_seconds": 7200,
        "job_hard_cap_seconds": 10800,
        "expected_gpu_hours_upper": 2.0,
        "device_class": "gpu",
        **launch,
        "requested_steps": 20000,
        "validation_conditioning_plan": (
            "exact_y_lengths_from_hash_bound_validation_npz"
        ),
        "retry_allowed": False,
        "early_stopping_allowed": False,
        "sweep_allowed": False,
        "memory_safety": dict(config["memory_safety"]),
        "preserved_completed_results": reused,
        "failed_attempt_001_preservation": failed,
        "scientific_definition_unchanged": True,
        "seed_conditioning_selection_unchanged": True,
        "job_cap_changed": False,
        "sparkov_authorized": False,
        "internal_test_authorized": False,
        "fraud_test_authorized": False,
        "tstr_authorized": False,
        "privacy_authorized": False,
        "full_run_authorized": False,
        "gpu_inventory_query_authorized": False,
        "aggregate_authorized": False,
        "existing_artifact_mutation_authorized": False,
    }


def build_amlsim_tvae_continuation_plan(
    *,
    repo_root: Path,
    config_path: Path,
    approval_text: str,
    mode: str,
) -> Mapping[str, Any]:
    """Validate the one-job continuation without creating an artifact."""

    if mode not in {"plan", "dry-run"}:
        raise ExternalValidationError(
            "AMLSim TVAE continuation mode must be plan or dry-run"
        )
    authorization = build_amlsim_tvae_continuation_authorization(
        repo_root=repo_root,
        config_path=config_path,
        approval_text=approval_text,
    )
    validated = validate_external_validation_authorization(
        repo_root=repo_root,
        config_path=config_path,
        authorization=authorization,
        # The source-only historical plan remains inspectable after the
        # append-only attempt has reached terminal state.  Execute validation
        # still uses the default True and therefore refuses overwrite/retry.
        require_attempt_absent=False,
    )
    return {
        "schema_version": "amlsim-tvae-memory-safe-continuation-plan-v1",
        "mode": mode,
        "status": "PASS",
        "target": {
            "dataset": validated["dataset"],
            "model": validated["model"],
            "attempt": authorization["attempt"],
        },
        "job_count": 1,
        "authorization_sha256_if_written": _document_sha256(
            authorization
        ),
        "authorization_root": str(
            repo_root
            / _load_config(config_path)["corrective_continuation"][
                "authorization_root"
            ]
        ),
        "preserved_completed_results": authorization[
            "preserved_completed_results"
        ],
        "failed_attempt_001_preservation": authorization[
            "failed_attempt_001_preservation"
        ],
        "memory_safety": authorization["memory_safety"],
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),
        "authorization_created": False,
        "runtime_artifacts_created": False,
    }


def _verify_bundle(
    repo_root: Path,
    dataset: str,
    record: Mapping[str, Any],
    *,
    hash_frozen_body: bool = False,
) -> Mapping[str, Any]:
    bundle_relative = Path(record["bundle_root"])
    artifact_relative = Path(record["artifact_root"])
    bundle_root = repo_root / bundle_relative
    artifact_root = repo_root / artifact_relative
    complete_path = artifact_root / "COMPLETE.json"
    index_path = artifact_root / "artifact_index.json"
    checksum_path = artifact_root / "checksum_manifest.json"
    provenance_path = bundle_root / "provenance_manifest.json"
    summary_path = bundle_root / "summary.json"
    complete = _load_json(complete_path)
    index = _load_json(index_path)
    checksums = _load_json(checksum_path)
    provenance = _load_json(provenance_path)
    summary = _load_json(summary_path)
    if complete.get("status") != "COMPLETE" or complete.get("dataset") != dataset:
        raise ExternalValidationError(f"frozen bundle is not COMPLETE: {dataset}")
    if _sha256_file(complete_path) != record["complete_sha256"]:
        raise ExternalValidationError(f"COMPLETE hash mismatch: {dataset}")
    if _sha256_file(index_path) != complete["artifact_index_sha256"]:
        raise ExternalValidationError(f"artifact index hash mismatch: {dataset}")
    if _sha256_file(checksum_path) != complete["checksum_manifest_sha256"]:
        raise ExternalValidationError(f"checksum manifest hash mismatch: {dataset}")
    if index.get("data_tree_files") != checksums.get("data_files"):
        raise ExternalValidationError(f"index/checksum disagreement: {dataset}")
    for relative, expected in checksums["data_files"].items():
        path = bundle_root / relative
        if not path.is_file():
            raise ExternalValidationError(f"frozen data file is missing: {dataset}/{relative}")
        if hash_frozen_body and _sha256_file(path) != expected:
            raise ExternalValidationError(f"frozen data hash mismatch: {dataset}/{relative}")
    observed_tree = str(record["tree_sha256"])
    file_count = len(checksums["data_files"]) + 4
    byte_count: int | None = None
    if hash_frozen_body:
        observed_tree, file_count, byte_count = _tree_digest(
            bundle_relative, artifact_relative
        )
        # The digest labels use repository-relative roots; file bytes come from repo_root.
        if not bundle_relative.is_absolute():
            digest = hashlib.sha256()
            file_count = 0
            byte_count = 0
            for relative_root, absolute_root in (
                (bundle_relative, bundle_root),
                (artifact_relative, artifact_root),
            ):
                for path in sorted(item for item in absolute_root.rglob("*") if item.is_file()):
                    label = f"{relative_root}:{path.relative_to(absolute_root)}".encode("utf-8")
                    payload = path.read_bytes()
                    digest.update(len(label).to_bytes(8, "big"))
                    digest.update(label)
                    digest.update(len(payload).to_bytes(8, "big"))
                    digest.update(payload)
                    file_count += 1
                    byte_count += len(payload)
            observed_tree = digest.hexdigest()
        if observed_tree != record["tree_sha256"]:
            raise ExternalValidationError(f"frozen bundle tree hash mismatch: {dataset}")
    leakage_audit = _audit_leakage_paths(
        bundle_root=bundle_root,
        dataset=dataset,
        summary=summary,
    )
    return {
        "status": "PASS",
        "tree_sha256": observed_tree,
        "file_count": file_count,
        "byte_count": byte_count,
        "complete_sha256": record["complete_sha256"],
        "train_sha256": checksums["data_files"]["train.npz"],
        "validation_sha256": checksums["data_files"]["validation.npz"],
        "internal_test_sha256_locked": checksums["data_files"]["internal_test.npz"],
        "transform_sha256": checksums["data_files"]["train_transform_state.json"],
        "provenance_sha256": checksums["data_files"]["provenance_manifest.json"],
        "transform_state_sha256": provenance["transform_state_sha256"],
        "receiver_vocabulary_cardinality": summary["receiver_vocabulary_cardinality"],
        "leakage_audit": leakage_audit,
        "manifest_json_reads": 8,
        "bundle_hash_reads": 3,
        "frozen_body_hash_reads": len(checksums["data_files"]) if hash_frozen_body else 0,
        "tree_digest_status": (
            "RECOMPUTED_PASS" if hash_frozen_body else "PINNED_MANIFEST_ONLY"
        ),
        "npz_load_calls": 0,
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("status") != "source_only_not_authorized":
        raise ExternalValidationError("external validation execution is not authorized")
    if list(config.get("models", [])) != [
        "empirical_iid",
        "ctgan_separate_class",
        "tvae_separate_class",
        "cof_seqgen_frozen_non_v3",
    ]:
        raise ExternalValidationError("external model scope changed")
    runtime = config["runtime"]
    expected_attempts = {
        "empirical_iid": "attempt_002",
        "ctgan_separate_class": "attempt_001",
        "tvae_separate_class": "attempt_001",
        "cof_seqgen_frozen_non_v3": "attempt_001",
    }
    if (
        runtime.get("attempts_by_model") != expected_attempts
        or runtime["append_only"] is not True
        or runtime["overwrite"] != "forbidden"
        or runtime["authorization_required_for_execute"] is not True
    ):
        raise ExternalValidationError("append-only runtime contract changed")
    correction = config.get("performance_correction", {})
    if (
        correction.get("kind") != "receiver_total_variation_complexity_only"
        or correction.get("scientific_definition_unchanged") is not True
        or correction.get("seed_conditioning_selection_unchanged") is not True
        or correction.get("job_cap_changed") is not False
    ):
        raise ExternalValidationError("performance correction scope changed")
    continuation = config.get("continuation", {})
    if (
        continuation.get("schema_version")
        != "external-validation-performance-continuation-v1"
        or continuation.get("job_count") != 8
        or set(continuation.get("preserved_artifacts", {}))
        != {
            "amlsim_empirical_iid_attempt_001",
            "sparkov_empirical_iid_attempt_001",
            "prior_authorization_history",
            "prior_continuation_authorization_history",
            "prior_wave_scheduled_authorization_history",
        }
    ):
        raise ExternalValidationError("performance continuation contract changed")
    validate_external_launch_schedule(config)


def validate_external_validation_authorization(
    *,
    repo_root: Path,
    config_path: Path,
    authorization: Mapping[str, Any],
    require_attempt_absent: bool = True,
) -> Mapping[str, Any]:
    """Validate a future single dataset/model authorization without execution."""

    config = _load_config(config_path)
    _validate_config(config)
    if (
        authorization.get("schema_version")
        != "external-validation-performance-continuation-job-authorization-v1"
        or authorization.get("scope")
        != "one_performance_corrective_external_validation_job"
    ):
        raise ExternalValidationError("authorization scope is not one dataset/model")
    dataset = str(authorization.get("dataset", ""))
    model = str(authorization.get("model", ""))
    if dataset not in config["datasets"] or model not in config["models"]:
        raise ExternalValidationError("authorization dataset/model is not preregistered")
    is_tvae_continuation = authorization.get("continuation_kind") == (
        "amlsim_tvae_memory_safe_attempt_002"
    )
    expected_attempt = (
        str(config["corrective_continuation"]["next_target"]["attempt"])
        if is_tvae_continuation
        else _expected_attempt(config, model)
    )
    if authorization.get("attempt") != expected_attempt:
        raise ExternalValidationError("authorization continuation attempt mapping changed")
    launch_contract = _launch_job_contract(config, dataset, model)
    launch_mismatches = [
        key
        for key, value in launch_contract.items()
        if authorization.get(key) != value
    ]
    if launch_mismatches:
        raise ExternalValidationError(
            "authorization launch schedule mismatch: "
            + ", ".join(launch_mismatches)
        )

    if is_tvae_continuation:
        corrective = config["corrective_continuation"]
        exact_contract = {
            "dataset": "amlsim",
            "model": "tvae_separate_class",
            "attempt": "attempt_002",
            "seed": 31001,
            "requested_steps": 20000,
            "training_hard_cap_seconds": 7200,
            "job_hard_cap_seconds": 10800,
            "runner_device": "cuda:0",
            "cuda_visible_devices_count": 1,
            "data_transformer_n_jobs": 1,
            "heavy_transform_exclusion_group": (
                "external_tabular_transform"
            ),
            "authorization_required": True,
        }
        if corrective.get("next_target") != exact_contract:
            raise ExternalValidationError(
                "AMLSim TVAE continuation target changed"
            )
        authorization_contract = {
            "dataset": dataset,
            "model": model,
            "attempt": authorization.get("attempt"),
            "seed": authorization.get("seed"),
            "requested_steps": authorization.get("requested_steps"),
            "training_hard_cap_seconds": authorization.get(
                "training_hard_cap_seconds"
            ),
            "job_hard_cap_seconds": authorization.get(
                "job_hard_cap_seconds"
            ),
            "runner_device": authorization.get("runner_device"),
            "cuda_visible_devices_count": authorization.get(
                "cuda_visible_devices_count"
            ),
            "data_transformer_n_jobs": authorization.get(
                "data_transformer_n_jobs"
            ),
            "heavy_transform_exclusion_group": authorization.get(
                "heavy_transform_exclusion_group"
            ),
            "authorization_required": True,
        }
        if (
            authorization_contract != exact_contract
            or authorization.get("job_count") != 1
            or authorization.get("single_job_only") is not True
            or authorization.get("memory_safety") != config["memory_safety"]
            or authorization.get("scientific_definition_unchanged") is not True
            or authorization.get("seed_conditioning_selection_unchanged") is not True
            or authorization.get("job_cap_changed") is not False
        ):
            raise ExternalValidationError(
                "AMLSim TVAE continuation contract mismatch"
            )
        reused = {
            reused_model: validate_external_completed_result_reuse(
                repo_root=repo_root,
                config_path=config_path,
                dataset="amlsim",
                model=reused_model,
            )
            for reused_model in (
                "empirical_iid",
                "ctgan_separate_class",
                "cof_seqgen_frozen_non_v3",
            )
        }
        if authorization.get("preserved_completed_results") != reused:
            raise ExternalValidationError(
                "AMLSim TVAE completed-result reuse mismatch"
            )
        failed = _validate_failed_tvae_attempt_preservation(
            repo_root=repo_root, config=config
        )
        if authorization.get("failed_attempt_001_preservation") != failed:
            raise ExternalValidationError(
                "AMLSim TVAE failed-attempt preservation mismatch"
            )
        false_scopes = (
            "sparkov_authorized",
            "internal_test_authorized",
            "fraud_test_authorized",
            "tstr_authorized",
            "privacy_authorized",
            "full_run_authorized",
            "gpu_inventory_query_authorized",
            "aggregate_authorized",
            "existing_artifact_mutation_authorized",
        )
        if any(authorization.get(field) is not False for field in false_scopes):
            raise ExternalValidationError(
                "AMLSim TVAE continuation forbidden scope changed"
            )

    bundle = _verify_bundle(repo_root, dataset, config["datasets"][dataset])
    preservation = validate_external_continuation_preservation(
        repo_root=repo_root, config=config
    )
    expected = {
        "source_commit": _git_head(repo_root),
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "execution_source_sha256": _sha256_file(
            repo_root / "experiments/external_validation_runner_v1.py"
        ),
        "config_sha256": _sha256_file(config_path),
        "metric_source_sha256": preservation["metric_source_sha256"],
        "continuation_preservation_sha256": preservation["preservation_sha256"],
        "bundle_tree_sha256": bundle["tree_sha256"],
        "train_sha256": bundle["train_sha256"],
        "validation_sha256": bundle["validation_sha256"],
    }
    mismatches = [
        key for key, value in expected.items() if authorization.get(key) != value
    ]
    if mismatches:
        raise ExternalValidationError(
            "authorization provenance mismatch: " + ", ".join(mismatches)
        )

    allowed = {
        "train_only_bootstrap",
        "train_only_model_fit",
        "validation_generation_and_evaluation",
    }
    forbidden = {
        "internal_test_access",
        "sparkov_fraudTest_access",
        "raw_csv_access",
        "controlled_artifact_mutation",
        "test_evaluation",
        "tstr",
        "privacy",
        "materialization_execution",
        "architecture_or_hyperparameter_change",
        "threshold_or_selection_rule_change",
        "full_or_five_seed_execution",
        "retry",
        "early_stopping",
        "hyperparameter_sweep",
        "gpu_inventory_query",
        "existing_artifact_overwrite_delete_move",
        "aggregate_selection",
    }
    if set(authorization.get("allowed_operations", ())) != allowed:
        raise ExternalValidationError("authorization allowed-operation scope changed")
    if set(authorization.get("forbidden_operations", ())) != forbidden:
        raise ExternalValidationError("authorization forbidden-operation scope changed")

    attempt_path = (
        repo_root
        / config["runtime"]["root"]
        / dataset
        / model
        / expected_attempt
    )
    if require_attempt_absent and attempt_path.exists():
        raise ExternalValidationError("external validation attempt already exists")
    canonical = json.dumps(
        authorization, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "status": "PASS",
        "dataset": dataset,
        "model": model,
        "attempt_path": str(attempt_path),
        "authorization_sha256": hashlib.sha256(canonical).hexdigest(),
        "parent_authorization_sha256": authorization.get(
            "parent_authorization_sha256"
        ),
        "bundle_tree_sha256": bundle["tree_sha256"],
        "metric_source_sha256": preservation["metric_source_sha256"],
        "continuation_preservation_sha256": preservation["preservation_sha256"],
        "npz_load_calls": 0,
        "model_import_calls": 0,
        "runtime_artifacts_created": False,
        "continuation_kind": authorization.get("continuation_kind"),
    }


def build_external_execution_manifest(
    *,
    repo_root: Path,
    config_path: Path,
    authorization: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build the immutable future attempt manifest without loading data."""

    validated = validate_external_validation_authorization(
        repo_root=repo_root,
        config_path=config_path,
        authorization=authorization,
    )
    config = _load_config(config_path)
    dataset = validated["dataset"]
    model = validated["model"]
    cof_contract = validate_external_input_and_frozen_cof_contract(
        repo_root=repo_root,
        config_path=config_path,
        dataset=dataset,
    )
    canonical = json.dumps(
        authorization, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    manifest = {
        "schema_version": "external-validation-attempt-v1",
        "source_commit": _git_head(repo_root),
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "source_sha256": _sha256_file(Path(__file__).resolve()),
        "execution_source_sha256": authorization["execution_source_sha256"],
        "config_sha256": _sha256_file(config_path),
        "metric_source_sha256": validated["metric_source_sha256"],
        "continuation_preservation_sha256": validated[
            "continuation_preservation_sha256"
        ],
        "authorization_sha256": hashlib.sha256(canonical).hexdigest(),
        "dataset": dataset,
        "model": model,
        "seed": int(config["datasets"][dataset]["seed"]),
        "attempt": authorization["attempt"],
        "wave_id": authorization["wave_id"],
        "device_slot": authorization["device_slot"],
        "runner_device": authorization["runner_device"],
        "cuda_visible_devices_count": authorization[
            "cuda_visible_devices_count"
        ],
        "data_transformer_n_jobs": authorization[
            "data_transformer_n_jobs"
        ],
        "heavy_transform_exclusion_group": authorization[
            "heavy_transform_exclusion_group"
        ],
        "memory_safety_policy_sha256": _canonical_sha256(
            config["memory_safety"]
        ),
        "attempt_path": validated["attempt_path"],
        "bundle_tree_sha256": validated["bundle_tree_sha256"],
        "data_hashes": {
            "train": authorization["train_sha256"],
            "validation": authorization["validation_sha256"],
        },
        "train_sha256": authorization["train_sha256"],
        "validation_sha256": authorization["validation_sha256"],
        "fit_split": "train",
        "reference_fit_split": "train",
        "evaluation_split": "validation",
        "frozen_cof_source_commit": config["frozen_cof"]["source_commit"],
        "frozen_cof_config_sha256": config["frozen_cof"]["config_sha256"],
        "frozen_cof_model_sha256": cof_contract["model_source_sha256"],
        "frozen_cof_adapter_sha256": cof_contract["adapter_source_sha256"],
        "frozen_cof_denoiser_sha256": cof_contract["denoiser_source_sha256"],
        "receiver_vocabulary_cardinality": cof_contract[
            "receiver_vocabulary_cardinality"
        ],
        "receiver_cardinality_role": "train_only_input_vocabulary_not_architecture_change",
        "append_only": True,
        "training_hard_cap_seconds": authorization.get(
            "training_hard_cap_seconds"
        ),
        "job_hard_cap_seconds": authorization.get("job_hard_cap_seconds"),
        "retry_allowed": authorization.get("retry_allowed", False),
        "runtime_artifacts_created": False,
    }
    if authorization.get("continuation_kind"):
        manifest.update(
            {
                "continuation_kind": authorization["continuation_kind"],
                "preserved_completed_results": authorization[
                    "preserved_completed_results"
                ],
                "failed_attempt_001_preservation": authorization[
                    "failed_attempt_001_preservation"
                ],
                "single_job_only": authorization["single_job_only"],
            }
        )
    return manifest


def _document_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _nested_numeric(value: Mapping[str, Any], dotted_path: str) -> float:
    current: Any = value
    for key in dotted_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise ExternalValidationError(
                f"Sparkov stop-rule field is missing: {dotted_path}"
            )
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ExternalValidationError(
            f"Sparkov stop-rule field is not numeric: {dotted_path}"
        )
    result = float(current)
    if not math.isfinite(result):
        raise ExternalValidationError(
            f"Sparkov stop-rule field is non-finite: {dotted_path}"
        )
    return result


def evaluate_sparkov_cof_stop_rule(
    *,
    cof_evaluation: Mapping[str, Any],
    tvae_evaluation: Mapping[str, Any],
    cof_terminal: str,
    tvae_terminal: str,
) -> Mapping[str, Any]:
    """Apply the preregistered strict CoF-vs-TVAE stop rule.

    Lower is better.  Continuing to a *separate proposal* requires CoF to be
    strictly lower than TVAE on both the combined score and the fidelity
    maximum ratio.  This function never authorizes test execution.
    """

    combined_path = "selection.combined_score"
    fidelity_path = "selection.fidelity_max_ratio"
    cof_combined = _nested_numeric(cof_evaluation, combined_path)
    tvae_combined = _nested_numeric(tvae_evaluation, combined_path)
    cof_fidelity = _nested_numeric(cof_evaluation, fidelity_path)
    tvae_fidelity = _nested_numeric(tvae_evaluation, fidelity_path)

    def _hard_valid(evaluation: Mapping[str, Any]) -> bool:
        guards = evaluation.get("hard_validity")
        return bool(
            isinstance(guards, Mapping)
            and guards
            and all(value is True for value in guards.values())
        )

    terminal_and_valid = (
        cof_terminal == "COMPLETE"
        and tvae_terminal == "COMPLETE"
        and _hard_valid(cof_evaluation)
        and _hard_valid(tvae_evaluation)
    )
    combined_strictly_better = cof_combined < tvae_combined
    fidelity_strictly_better = cof_fidelity < tvae_fidelity
    continue_proposal = (
        terminal_and_valid
        and combined_strictly_better
        and fidelity_strictly_better
    )
    return {
        "schema_version": "sparkov-cof-stop-rule-decision-v1",
        "direction": "lower_is_better",
        "cof_terminal": cof_terminal,
        "tvae_terminal": tvae_terminal,
        "hard_validity_pass": terminal_and_valid,
        "values": {
            "cof_combined_score": cof_combined,
            "tvae_combined_score": tvae_combined,
            "cof_fidelity_max_ratio": cof_fidelity,
            "tvae_fidelity_max_ratio": tvae_fidelity,
        },
        "comparisons": {
            "combined_score_strictly_better": combined_strictly_better,
            "fidelity_max_ratio_strictly_better": fidelity_strictly_better,
        },
        "decision": (
            "CONTINUE_TO_SEPARATE_TEST_PROPOSAL"
            if continue_proposal
            else "STOP_EXISTING_COF_FAMILY_MODEL_LEVEL_REDESIGN_REQUIRED"
        ),
        "test_execution_authorized": False,
        "model_tuning_authorized": False,
    }


def _validate_sparkov_iid_reference(
    *, repo_root: Path, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    preparation = config.get("sparkov_validation_preparation", {})
    record = preparation.get("iid_reference", {})
    attempt_path = (repo_root / str(record.get("attempt_path", ""))).resolve()
    expected_root = (
        repo_root / "artifacts/external_validation_v1/sparkov/empirical_iid"
    ).resolve()
    if expected_root not in attempt_path.parents:
        raise ExternalValidationError("Sparkov IID reference path changed")
    paths = {
        "manifest_sha256": attempt_path / "manifest.json",
        "evaluation_sha256": attempt_path / "evaluation.json",
        "terminal_sha256": attempt_path / "COMPLETE.json",
    }
    mismatches = [
        name
        for name, path in paths.items()
        if not path.is_file() or _sha256_file(path) != record.get(name)
    ]
    if mismatches:
        raise ExternalValidationError(
            "Sparkov IID reference hash mismatch: " + ", ".join(mismatches)
        )
    manifest = _load_json(attempt_path / "manifest.json")
    evaluation = _load_json(attempt_path / "evaluation.json")
    terminal = _load_json(attempt_path / "COMPLETE.json")
    expected_manifest = {
        "dataset": "sparkov",
        "model": "empirical_iid",
        "attempt": "attempt_002",
        "source_commit": record.get("source_commit"),
        "config_sha256": record.get("config_sha256"),
        "bundle_tree_sha256": record.get("bundle_tree_sha256"),
        "train_sha256": record.get("train_sha256"),
        "validation_sha256": record.get("validation_sha256"),
        "metric_source_sha256": record.get("metric_source_sha256"),
        "seed": int(record.get("seed", -1)),
    }
    provenance_mismatches = [
        name
        for name, expected in expected_manifest.items()
        if manifest.get(name) != expected
    ]
    if evaluation.get("sampling_plan_sha256") != record.get(
        "sampling_plan_sha256"
    ):
        provenance_mismatches.append("sampling_plan_sha256")
    if evaluation.get("thresholds_sha256") != record.get("thresholds_sha256"):
        provenance_mismatches.append("thresholds_sha256")
    if terminal.get("status") != "COMPLETE":
        provenance_mismatches.append("terminal_status")
    if provenance_mismatches:
        raise ExternalValidationError(
            "Sparkov IID reference provenance mismatch: "
            + ", ".join(provenance_mismatches)
        )
    return {
        "status": "REUSE_ELIGIBLE",
        "dataset": "sparkov",
        "model": "empirical_iid",
        "attempt": "attempt_002",
        "attempt_path": str(attempt_path),
        **{name: str(record[name]) for name in paths},
        "sampling_plan_sha256": record["sampling_plan_sha256"],
        "thresholds_sha256": record["thresholds_sha256"],
        "model_execution_calls": 0,
        "runtime_artifact_writes": 0,
    }


def build_sparkov_validation_authorization(
    *, repo_root: Path, config_path: Path, approval_text: str
) -> Mapping[str, Any]:
    """Build, but do not write, the source-only Sparkov grant template."""

    if not approval_text.strip():
        raise ExternalValidationError(
            "Sparkov source-only authorization requires approval text"
        )
    config = _load_config(config_path)
    _validate_config(config)
    preparation = config.get("sparkov_validation_preparation", {})
    if (
        preparation.get("schema_version")
        != "external-validation-sparkov-source-preparation-v1"
        or preparation.get("status") != "source_only_not_authorized"
        or preparation.get("scientific_contract_unchanged") is not True
        or preparation.get("model_architecture_or_config_change_allowed")
        is not False
    ):
        raise ExternalValidationError("Sparkov preparation scope changed")
    expected_waves = [
        {
            "wave_id": "sparkov_gpu_wave_1",
            "parallel": True,
            "distinct_physical_gpus": True,
            "terminal_barrier_after": True,
            "models": [
                "ctgan_separate_class",
                "cof_seqgen_frozen_non_v3",
            ],
        },
        {
            "wave_id": "sparkov_gpu_wave_2",
            "parallel": False,
            "distinct_physical_gpus": True,
            "terminal_barrier_after": True,
            "models": ["tvae_separate_class"],
        },
    ]
    expected_stop_rule = {
        "comparison": "cof_seqgen_frozen_non_v3_vs_tvae_separate_class",
        "direction": "lower_is_better",
        "combined_field": "selection.combined_score",
        "combined_operator": "strict_less_than",
        "fidelity_field": "selection.fidelity_max_ratio",
        "fidelity_operator": "strict_less_than",
        "required_relation": "both_metrics_strictly_better",
        "ties": "stop_existing_cof_family",
        "invalid_or_noncomplete": "stop_existing_cof_family",
        "pass_decision": "CONTINUE_TO_SEPARATE_TEST_PROPOSAL",
        "stop_decision": (
            "STOP_EXISTING_COF_FAMILY_MODEL_LEVEL_REDESIGN_REQUIRED"
        ),
        "result_based_rule_change_allowed": False,
        "test_execution_authorized": False,
    }
    if preparation.get("wave_contract") != expected_waves:
        raise ExternalValidationError("Sparkov wave contract changed")
    if preparation.get("stop_rule") != expected_stop_rule:
        raise ExternalValidationError("Sparkov stop rule changed")
    bundle = _verify_bundle(repo_root, "sparkov", config["datasets"]["sparkov"])
    adapter_contract = build_external_adapter_contract(
        repo_root=repo_root,
        config_path=config_path,
        dataset="sparkov",
    )
    preservation = validate_external_continuation_preservation(
        repo_root=repo_root, config=config
    )
    iid_reference = _validate_sparkov_iid_reference(
        repo_root=repo_root, config=config
    )
    head = _git_head(repo_root)
    runner_hash = _sha256_file(Path(__file__).resolve())
    execution_hash = _sha256_file(
        repo_root / "experiments/external_validation_runner_v1.py"
    )
    config_hash = _sha256_file(config_path)
    allowed = [
        "train_only_bootstrap",
        "train_only_model_fit",
        "validation_generation_and_evaluation",
    ]
    forbidden = [
        "internal_test_access",
        "sparkov_fraudTest_access",
        "raw_csv_access",
        "controlled_artifact_mutation",
        "test_evaluation",
        "tstr",
        "privacy",
        "materialization_execution",
        "architecture_or_hyperparameter_change",
        "threshold_or_selection_rule_change",
        "full_or_five_seed_execution",
        "retry",
        "early_stopping",
        "hyperparameter_sweep",
        "gpu_inventory_query",
        "existing_artifact_overwrite_delete_move",
        "aggregate_selection",
    ]
    configured_jobs = preparation.get("learned_jobs")
    if not isinstance(configured_jobs, list) or len(configured_jobs) != 3:
        raise ExternalValidationError("Sparkov preparation requires three jobs")
    jobs: list[Mapping[str, Any]] = []
    for registered in configured_jobs:
        model = str(registered.get("model", ""))
        if model not in {
            "ctgan_separate_class",
            "tvae_separate_class",
            "cof_seqgen_frozen_non_v3",
        }:
            raise ExternalValidationError("Sparkov learned model scope changed")
        launch = _launch_job_contract(config, "sparkov", model)
        if (
            registered.get("dataset") != "sparkov"
            or registered.get("attempt") != "attempt_001"
            or registered.get("wave_id") != launch["wave_id"]
            or int(registered.get("data_transformer_n_jobs", -1))
            != int(launch["data_transformer_n_jobs"])
        ):
            raise ExternalValidationError("Sparkov learned job mapping changed")
        model_contract = adapter_contract[model]
        jobs.append(
            {
                "schema_version": (
                    "external-validation-performance-continuation-job-authorization-v1"
                ),
                "scope": "one_performance_corrective_external_validation_job",
                "source_commit": head,
                "runner_source_sha256": runner_hash,
                "execution_source_sha256": execution_hash,
                "config_sha256": config_hash,
                "metric_source_sha256": preservation["metric_source_sha256"],
                "continuation_preservation_sha256": preservation[
                    "preservation_sha256"
                ],
                "dataset": "sparkov",
                "model": model,
                "seed": 32001,
                "attempt": "attempt_001",
                "bundle_tree_sha256": bundle["tree_sha256"],
                "train_sha256": bundle["train_sha256"],
                "validation_sha256": bundle["validation_sha256"],
                "allowed_operations": allowed,
                "forbidden_operations": forbidden,
                "training_hard_cap_seconds": int(
                    model_contract["max_wall_seconds"]
                ),
                "job_hard_cap_seconds": 10800,
                "expected_gpu_hours_upper": 2.0,
                "device_class": "gpu",
                **launch,
                "requested_steps": int(model_contract["requested_steps"]),
                "validation_conditioning_plan": (
                    "exact_y_lengths_from_hash_bound_validation_npz"
                ),
                "retry_allowed": False,
                "early_stopping_allowed": False,
                "sweep_allowed": False,
                "sparkov_fraudTest_authorized": False,
                "internal_test_authorized": False,
                "execution_authorized": False,
            }
        )
    waves = []
    for wave in preparation["wave_contract"]:
        models = list(wave["models"])
        wave_jobs = [
            {
                "dataset": job["dataset"],
                "model": job["model"],
                "attempt": job["attempt"],
            }
            for job in jobs
            if job["model"] in models
        ]
        if [job["model"] for job in wave_jobs] != models:
            raise ExternalValidationError("Sparkov wave model order changed")
        waves.append(
            {
                "wave_id": wave["wave_id"],
                "parallel": wave["parallel"],
                "distinct_physical_gpus": wave["distinct_physical_gpus"],
                "terminal_barrier_after": wave["terminal_barrier_after"],
                "jobs": wave_jobs,
            }
        )
    return {
        "schema_version": "external-validation-sparkov-source-authorization-v1",
        "scope": "sparkov_iid_reuse_plus_three_learned_validation_jobs",
        "approval_text": approval_text.strip(),
        "source_commit": head,
        "runner_source_sha256": runner_hash,
        "execution_source_sha256": execution_hash,
        "config_sha256": config_hash,
        "metric_source_sha256": preservation["metric_source_sha256"],
        "bundle_tree_sha256": bundle["tree_sha256"],
        "iid_reference": iid_reference,
        "job_count": 3,
        "jobs": jobs,
        "waves": waves,
        "allowed_operations": allowed,
        "forbidden_operations": forbidden,
        "stop_rule": json.loads(
            json.dumps(preparation["stop_rule"], allow_nan=False)
        ),
        "frozen_cof": dict(config["frozen_cof"]),
        "scientific_contract_unchanged": True,
        "model_architecture_or_config_change_allowed": False,
        "execution_authorized": False,
        "aggregate_authorized": False,
        "test_execution_authorized": False,
        "gpu_inventory_query_authorized": False,
        "internal_test_authorized": False,
        "sparkov_fraudTest_authorized": False,
        "existing_artifact_mutation_authorized": False,
    }


def validate_sparkov_validation_authorization(
    *,
    repo_root: Path,
    config_path: Path,
    authorization: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Fail closed unless a source-only Sparkov template is exact."""

    config = _load_config(config_path)
    _validate_config(config)
    expected_stop_rule = config["sparkov_validation_preparation"]["stop_rule"]
    if authorization.get("stop_rule") != expected_stop_rule:
        raise ExternalValidationError("Sparkov stop rule changed")
    expected = build_sparkov_validation_authorization(
        repo_root=repo_root,
        config_path=config_path,
        approval_text=str(authorization.get("approval_text", "")),
    )
    if authorization != expected:
        raise ExternalValidationError("Sparkov source-only authorization changed")
    for job in authorization["jobs"]:
        validate_external_validation_authorization(
            repo_root=repo_root,
            config_path=config_path,
            authorization=job,
            # This validator is for the source-only historical template.  It
            # must remain auditable after the append-only Sparkov attempts
            # become terminal.  Execute validation still uses the default
            # True and therefore refuses overwrite/retry.
            require_attempt_absent=False,
        )
    return {
        "status": "PASS",
        "authorization_sha256_if_written": _document_sha256(authorization),
        "job_count": 3,
        "iid_reference": authorization["iid_reference"],
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),
        "authorization_created": False,
        "runtime_artifacts_created": False,
    }


def build_sparkov_validation_plan(
    *,
    repo_root: Path,
    config_path: Path,
    approval_text: str,
    mode: str,
) -> Mapping[str, Any]:
    """Plan the Sparkov-only comparison without creating authorization/runtime."""

    if mode not in {"plan", "dry-run"}:
        raise ExternalValidationError("Sparkov preparation mode must be plan or dry-run")
    authorization = build_sparkov_validation_authorization(
        repo_root=repo_root,
        config_path=config_path,
        approval_text=approval_text,
    )
    validated = validate_sparkov_validation_authorization(
        repo_root=repo_root,
        config_path=config_path,
        authorization=authorization,
    )
    return {
        "schema_version": "external-validation-sparkov-source-plan-v1",
        "mode": mode,
        "status": "PASS",
        "job_count": 3,
        "jobs": [
            {
                "dataset": job["dataset"],
                "model": job["model"],
                "attempt": job["attempt"],
                "wave_id": job["wave_id"],
                "data_transformer_n_jobs": job[
                    "data_transformer_n_jobs"
                ],
            }
            for job in authorization["jobs"]
        ],
        "waves": authorization["waves"],
        "iid_reference": authorization["iid_reference"],
        "stop_rule": authorization["stop_rule"],
        "authorization_sha256_if_written": validated[
            "authorization_sha256_if_written"
        ],
        "authorization_created": False,
        "runtime_artifacts_created": False,
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),
    }


def build_external_validation_batch_authorization(
    *,
    repo_root: Path,
    config_path: Path,
    approval_text: str,
) -> Mapping[str, Any]:
    """Build the exact eight-job authorization without writing or execution."""

    if not approval_text.strip():
        raise ExternalValidationError("batch authorization requires explicit approval text")
    config = _load_config(config_path)
    _validate_config(config)
    dry_run = build_external_validation_plan(
        repo_root=repo_root,
        config_path=config_path,
        mode="dry-run",
    )
    head = _git_head(repo_root)
    runner_hash = _sha256_file(Path(__file__).resolve())
    config_hash = _sha256_file(config_path)
    preservation = validate_external_continuation_preservation(
        repo_root=repo_root, config=config
    )
    completed_result_reuse = {
        model: validate_external_completed_result_reuse(
            repo_root=repo_root,
            config_path=config_path,
            dataset="amlsim",
            model=model,
        )
        for model in (
            "empirical_iid",
            "ctgan_separate_class",
            "cof_seqgen_frozen_non_v3",
        )
    }
    allowed = [
        "train_only_bootstrap",
        "train_only_model_fit",
        "validation_generation_and_evaluation",
    ]
    forbidden = [
        "internal_test_access",
        "sparkov_fraudTest_access",
        "raw_csv_access",
        "controlled_artifact_mutation",
        "test_evaluation",
        "tstr",
        "privacy",
        "materialization_execution",
        "architecture_or_hyperparameter_change",
        "threshold_or_selection_rule_change",
        "full_or_five_seed_execution",
        "retry",
        "early_stopping",
        "hyperparameter_sweep",
        "gpu_inventory_query",
        "existing_artifact_overwrite_delete_move",
        "aggregate_selection",
    ]
    jobs = []
    for dataset in ("amlsim", "sparkov"):
        bundle = dry_run["bundles"][dataset]
        adapter_contract = dry_run["adapter_contracts"][dataset]
        for model in config["models"]:
            model_contract = adapter_contract[model]
            launch_contract = _launch_job_contract(config, dataset, model)
            learned = model != "empirical_iid"
            jobs.append(
                {
                    "schema_version": (
                        "external-validation-performance-continuation-job-authorization-v1"
                    ),
                    "scope": "one_performance_corrective_external_validation_job",
                    "source_commit": head,
                    "runner_source_sha256": runner_hash,
                    "execution_source_sha256": _sha256_file(
                        repo_root / "experiments/external_validation_runner_v1.py"
                    ),
                    "config_sha256": config_hash,
                    "metric_source_sha256": preservation[
                        "metric_source_sha256"
                    ],
                    "continuation_preservation_sha256": preservation[
                        "preservation_sha256"
                    ],
                    "dataset": dataset,
                    "model": model,
                    "seed": int(config["datasets"][dataset]["seed"]),
                    "attempt": _expected_attempt(config, model),
                    "bundle_tree_sha256": bundle["tree_sha256"],
                    "train_sha256": bundle["train_sha256"],
                    "validation_sha256": bundle["validation_sha256"],
                    "allowed_operations": allowed,
                    "forbidden_operations": forbidden,
                    "training_hard_cap_seconds": int(
                        model_contract.get("max_wall_seconds", 0)
                    ),
                    "job_hard_cap_seconds": 10800 if learned else 7200,
                    "expected_gpu_hours_upper": 2.0 if learned else 0.0,
                    "device_class": "gpu" if learned else "cpu",
                    **launch_contract,
                    "requested_steps": int(model_contract["requested_steps"]),
                    "validation_conditioning_plan": (
                        "exact_y_lengths_from_hash_bound_validation_npz"
                    ),
                    "retry_allowed": False,
                    "early_stopping_allowed": False,
                    "sweep_allowed": False,
                }
            )
    return {
        "schema_version": (
            "external-validation-performance-continuation-authorization-v1"
        ),
        "scope": "exact_eight_performance_corrective_external_validation_jobs",
        "approval_text": approval_text.strip(),
        "source_commit": head,
        "runner_source_sha256": runner_hash,
        "execution_source_sha256": _sha256_file(
            repo_root / "experiments/external_validation_runner_v1.py"
        ),
        "config_sha256": config_hash,
        "metric_source_sha256": preservation["metric_source_sha256"],
        "continuation_preservation": preservation,
        "attempts_by_model": dict(config["runtime"]["attempts_by_model"]),
        "launch_schedule": validate_external_launch_schedule(config),
        "wave_barrier_enforced": True,
        "memory_safety": dict(config["memory_safety"]),
        "completed_result_reuse": completed_result_reuse,
        "scientific_definition_unchanged": True,
        "seed_conditioning_selection_unchanged": True,
        "job_cap_changed": False,
        "v2_5_provenance": {
            "config_sha256": _sha256_file(
                repo_root / "configs/benchmark_v2/full_v2_5.yaml"
            ),
            "final_complete_sha256": _sha256_file(
                repo_root / "artifacts/benchmark_v2_5/full/FINAL_COMPLETE.json"
            ),
            "frozen_manifest_sha256": _sha256_file(
                repo_root
                / "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00/data_manifest.json"
            ),
        },
        "frozen_cof": dict(config["frozen_cof"]),
        "bundle_trees": {
            dataset: dry_run["bundles"][dataset]["tree_sha256"]
            for dataset in ("amlsim", "sparkov")
        },
        "job_count": 8,
        "jobs": jobs,
        "aggregate_authorized": False,
        "test_execution_authorized": False,
        "gpu_inventory_query_authorized": False,
        "materialization_authorized": False,
        "existing_artifact_mutation_authorized": False,
    }


def validate_external_validation_batch_authorization(
    *,
    repo_root: Path,
    config_path: Path,
    authorization: Mapping[str, Any],
    require_all_attempts_absent: bool = True,
) -> Mapping[str, Any]:
    """Validate the exact eight-job authorization using manifests only."""

    if (
        authorization.get("schema_version")
        != "external-validation-performance-continuation-authorization-v1"
        or authorization.get("scope")
        != "exact_eight_performance_corrective_external_validation_jobs"
    ):
        raise ExternalValidationError("batch authorization scope changed")
    jobs = authorization.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 8 or authorization.get("job_count") != 8:
        raise ExternalValidationError("batch authorization requires exactly eight jobs")
    config = _load_config(config_path)
    expected_pairs = {
        (dataset, model)
        for dataset in ("amlsim", "sparkov")
        for model in config["models"]
    }
    observed_pairs = {
        (str(job.get("dataset")), str(job.get("model"))) for job in jobs
    }
    if observed_pairs != expected_pairs or len(observed_pairs) != len(jobs):
        raise ExternalValidationError("batch authorization requires exactly eight unique jobs")
    expected_top = {
        "source_commit": _git_head(repo_root),
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "execution_source_sha256": _sha256_file(
            repo_root / "experiments/external_validation_runner_v1.py"
        ),
        "config_sha256": _sha256_file(config_path),
        "metric_source_sha256": _sha256_file(
            repo_root / config["performance_correction"]["metric_source_path"]
        ),
    }
    mismatches = [
        key for key, value in expected_top.items() if authorization.get(key) != value
    ]
    if mismatches:
        raise ExternalValidationError(
            "batch authorization provenance mismatch: " + ", ".join(mismatches)
        )
    expected_v2_5 = {
        "config_sha256": _sha256_file(
            repo_root / "configs/benchmark_v2/full_v2_5.yaml"
        ),
        "final_complete_sha256": _sha256_file(
            repo_root / "artifacts/benchmark_v2_5/full/FINAL_COMPLETE.json"
        ),
        "frozen_manifest_sha256": _sha256_file(
            repo_root
            / "data/benchmark_v2_5/frozen/joint_semimarkov_v2b/kappa_1.00/data_manifest.json"
        ),
    }
    if authorization.get("v2_5_provenance") != expected_v2_5:
        raise ExternalValidationError("batch authorization v2.5 provenance mismatch")
    if authorization.get("frozen_cof") != config["frozen_cof"]:
        raise ExternalValidationError("batch authorization frozen CoF fingerprint mismatch")
    expected_trees = {
        dataset: config["datasets"][dataset]["tree_sha256"]
        for dataset in ("amlsim", "sparkov")
    }
    if authorization.get("bundle_trees") != expected_trees:
        raise ExternalValidationError("batch authorization bundle tree mismatch")
    preservation = validate_external_continuation_preservation(
        repo_root=repo_root, config=config
    )
    if authorization.get("continuation_preservation") != preservation:
        raise ExternalValidationError("batch authorization preservation mismatch")
    if authorization.get("attempts_by_model") != config["runtime"]["attempts_by_model"]:
        raise ExternalValidationError("batch authorization attempt mapping changed")
    expected_schedule = validate_external_launch_schedule(config)
    if authorization.get("launch_schedule") != expected_schedule:
        raise ExternalValidationError("batch authorization launch schedule changed")
    if authorization.get("wave_barrier_enforced") is not True:
        raise ExternalValidationError("batch authorization wave barrier changed")
    if authorization.get("memory_safety") != config["memory_safety"]:
        raise ExternalValidationError("batch authorization memory policy changed")
    expected_reuse = {
        model: validate_external_completed_result_reuse(
            repo_root=repo_root,
            config_path=config_path,
            dataset="amlsim",
            model=model,
        )
        for model in (
            "empirical_iid",
            "ctgan_separate_class",
            "cof_seqgen_frozen_non_v3",
        )
    }
    if authorization.get("completed_result_reuse") != expected_reuse:
        raise ExternalValidationError(
            "batch authorization completed-result reuse changed"
        )
    if (
        authorization.get("scientific_definition_unchanged") is not True
        or authorization.get("seed_conditioning_selection_unchanged") is not True
        or authorization.get("job_cap_changed") is not False
    ):
        raise ExternalValidationError("batch authorization correction scope changed")
    if not str(authorization.get("approval_text", "")).strip():
        raise ExternalValidationError("batch authorization approval text is missing")
    for field in (
        "aggregate_authorized",
        "test_execution_authorized",
        "gpu_inventory_query_authorized",
        "materialization_authorized",
        "existing_artifact_mutation_authorized",
    ):
        if authorization.get(field) is not False:
            raise ExternalValidationError(f"batch authorization forbidden scope changed: {field}")
    for job in jobs:
        validate_external_validation_authorization(
            repo_root=repo_root,
            config_path=config_path,
            authorization=job,
            require_attempt_absent=require_all_attempts_absent,
        )
        learned = job["model"] != "empirical_iid"
        launch_contract = _launch_job_contract(
            config, str(job["dataset"]), str(job["model"])
        )
        expected_cap = 7200 if learned else 0
        expected_steps = {
            "empirical_iid": 0,
            "ctgan_separate_class": 10000,
            "tvae_separate_class": 20000,
            "cof_seqgen_frozen_non_v3": 20000,
        }[job["model"]]
        if (
            int(job.get("seed", -1))
            != int(config["datasets"][job["dataset"]]["seed"])
            or job.get("attempt") != _expected_attempt(config, job["model"])
            or int(job.get("training_hard_cap_seconds", -1)) != expected_cap
            or int(job.get("job_hard_cap_seconds", -1))
            != (10800 if learned else 7200)
            or int(job.get("requested_steps", -1)) != expected_steps
            or float(job.get("expected_gpu_hours_upper", -1))
            != (2.0 if learned else 0.0)
            or job.get("device_class") != ("gpu" if learned else "cpu")
            or any(job.get(key) != value for key, value in launch_contract.items())
            or job.get("validation_conditioning_plan")
            != "exact_y_lengths_from_hash_bound_validation_npz"
            or job.get("retry_allowed") is not False
            or job.get("early_stopping_allowed") is not False
            or job.get("sweep_allowed") is not False
        ):
            raise ExternalValidationError("batch authorization job contract changed")
    return {
        "status": "PASS",
        "job_count": 8,
        "authorization_sha256": _document_sha256(authorization),
        "execution_counts": {
            "npz_load_calls": 0,
            "raw_csv_accesses": 0,
            "model_import_calls": 0,
            "gpu_inventory_queries": 0,
        },
    }


def select_batch_authorized_job(
    *,
    repo_root: Path,
    config_path: Path,
    authorization: Mapping[str, Any],
    dataset: str,
    model: str,
) -> Mapping[str, Any]:
    validated = validate_external_validation_batch_authorization(
        repo_root=repo_root,
        config_path=config_path,
        authorization=authorization,
        require_all_attempts_absent=False,
    )
    matching = [
        job
        for job in authorization["jobs"]
        if job["dataset"] == dataset and job["model"] == model
    ]
    if len(matching) != 1:
        raise ExternalValidationError("batch authorization job selection is ambiguous")
    return {
        **matching[0],
        "parent_authorization_sha256": validated["authorization_sha256"],
    }


def write_external_validation_authorization(
    path: Path, authorization: Mapping[str, Any]
) -> Path:
    """Exclusive-create an append-only authorization document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_json(path, authorization)
    return path


def resolve_external_bundle_access(
    *,
    repo_root: Path,
    config_path: Path,
    dataset: str,
    split: str,
    purpose: str,
) -> Path:
    """Resolve an allowed NPZ path without opening it."""

    config = _load_config(config_path)
    _validate_config(config)
    if dataset not in config["datasets"]:
        raise ExternalValidationError(f"unsupported external dataset: {dataset}")
    if split in {"fraudTest", "public_test", "sparkov_fraudTest"}:
        raise ExternalValidationError("Sparkov fraudTest is locked before path access")
    if split == "internal_test":
        raise ExternalValidationError("internal_test is locked before NPZ access")
    if purpose in {"model_fit", "transform_fit", "bootstrap_reference_fit"}:
        if split != "train":
            raise ExternalValidationError("validation cannot fit models or reference state")
    elif purpose == "candidate_generation_and_evaluation":
        if split != "validation":
            raise ExternalValidationError("candidate evaluation may read validation only")
    else:
        raise ExternalValidationError(f"unsupported external data purpose: {purpose}")
    return repo_root / config["datasets"][dataset]["bundle_root"] / f"{split}.npz"


def _identity_key(value: Mapping[str, Any]) -> tuple[str, str]:
    return str(value["type"]), str(value["value"])


def _audit_leakage_paths(
    *, bundle_root: Path, dataset: str, summary: Mapping[str, Any]
) -> Mapping[str, Any]:
    entity_manifest = _load_json(bundle_root / "entity_split_manifest.json")
    window_manifest = _load_json(bundle_root / "window_manifest.json")
    leakage = _load_json(bundle_root / "leakage_audit.json")
    assignments: dict[tuple[str, str], str] = {}
    duplicate_entities = 0
    for item in entity_manifest["assignments"]:
        key = _identity_key(item["entity"])
        if key in assignments:
            duplicate_entities += 1
        assignments[key] = str(item["split"])
    window_ids: set[str] = set()
    membership_hashes: set[str] = set()
    duplicate_window_ids = 0
    duplicate_memberships = 0
    missing_entity = 0
    wrong_entity_split = 0
    window_count = 0
    for split, windows in window_manifest["splits"].items():
        if len(windows) != int(summary["splits"][split]["sequences"]):
            raise ExternalValidationError(
                f"window/summary sequence count mismatch: {dataset}/{split}"
            )
        for window in windows:
            window_count += 1
            window_id = str(window["window_id"])
            membership = str(window["transaction_ids_sha256"])
            if window_id in window_ids:
                duplicate_window_ids += 1
            window_ids.add(window_id)
            if membership in membership_hashes:
                duplicate_memberships += 1
            membership_hashes.add(membership)
            entity = _identity_key(window["entity"])
            if entity not in assignments:
                missing_entity += 1
            elif assignments[entity] != split:
                wrong_entity_split += 1
            length = int(window["length"])
            if not 16 <= length <= 32 or int(window["transaction_count"]) != length:
                raise ExternalValidationError(
                    f"invalid frozen window length/membership count: {dataset}"
                )
            if int(window["y"]) not in {0, 1}:
                raise ExternalValidationError(f"invalid frozen window label: {dataset}")
    observed = {
        key: int(leakage[key])
        for key in (
            "entity_overlap_count",
            "window_overlap_count",
            "transaction_overlap_count",
        )
    }
    failures = list(observed.values()) + [
        duplicate_entities,
        duplicate_window_ids,
        duplicate_memberships,
        missing_entity,
        wrong_entity_split,
    ]
    if leakage.get("status") != "PASS" or any(failures):
        raise ExternalValidationError(
            f"frozen leakage manifest contract failed: {dataset}"
        )
    return {
        "status": "PASS",
        **observed,
        "window_count": window_count,
        "duplicate_entity_assignments": duplicate_entities,
        "duplicate_window_ids": duplicate_window_ids,
        "duplicate_window_membership_hashes": duplicate_memberships,
        "window_entities_missing_from_split": missing_entity,
        "wrong_entity_split_count": wrong_entity_split,
        "transaction_evidence": "hash_bound_materializer_independent_audit",
        "limitation": (
            "transaction IDs are represented by per-window SHA-256; this stage "
            "verifies the immutable zero-overlap audit rather than reopening raw CSV"
        ),
    }


def audit_frozen_leakage_manifests(
    *, repo_root: Path, config_path: Path, dataset: str
) -> Mapping[str, Any]:
    """Revalidate leakage evidence using only immutable JSON manifests."""

    config = _load_config(config_path)
    _validate_config(config)
    if dataset not in config["datasets"]:
        raise ExternalValidationError(f"unsupported external dataset: {dataset}")
    return _verify_bundle(repo_root, dataset, config["datasets"][dataset])[
        "leakage_audit"
    ]


def validate_external_input_and_frozen_cof_contract(
    *, repo_root: Path, config_path: Path, dataset: str
) -> Mapping[str, Any]:
    """Validate train-fit input state and exact frozen non-v3 CoF sources."""

    config = _load_config(config_path)
    _validate_config(config)
    if dataset not in config["datasets"]:
        raise ExternalValidationError(f"unsupported external dataset: {dataset}")
    frozen = config["frozen_cof"]
    commit = str(frozen["source_commit"])
    for key in ("model", "adapter", "denoiser", "config"):
        path_key = f"{key}_path"
        hash_key = f"{key}_sha256"
        blob = _git_blob(repo_root, commit, str(frozen[path_key]))
        if hashlib.sha256(blob).hexdigest() != frozen[hash_key]:
            raise ExternalValidationError(f"frozen CoF {key} Git fingerprint mismatch")
        if _sha256_file(repo_root / frozen[path_key]) != frozen[hash_key]:
            raise ExternalValidationError(f"current CoF {key} source fingerprint mismatch")
    adapter_text = (repo_root / frozen["adapter_path"]).read_text(encoding="utf-8")
    denoiser_text = (repo_root / frozen["denoiser_path"]).read_text(encoding="utf-8")
    required_tokens = (
        "bins = int(train.dt_bin.max()) + 1",
        "int(train.x_cat[..., i].max()) + 1",
        "train.x_num.shape[-1], bins, categories",
        "L_max=train.x_num.shape[1]",
        "nn.Embedding(K + 1, d_model)",
    )
    combined = adapter_text + "\n" + denoiser_text
    if any(token not in combined for token in required_tokens):
        raise ExternalValidationError(
            "frozen CoF does not prove dynamic external input cardinality support"
        )
    bundle_root = repo_root / config["datasets"][dataset]["bundle_root"]
    transform = _load_json(bundle_root / "train_transform_state.json")
    summary = _load_json(bundle_root / "summary.json")
    provenance = _load_json(bundle_root / "provenance_manifest.json")
    if transform.get("fit_role") != "train":
        raise ExternalValidationError("external transform state is not train-only")
    if transform.get("state_sha256") != provenance.get("transform_state_sha256"):
        raise ExternalValidationError("external transform provenance hash mismatch")
    if int(transform["pad_code"]) != 0 or int(transform["unk_code"]) != 1:
        raise ExternalValidationError("external PAD/UNK contract mismatch")
    vocabulary = transform["receiver_vocabulary"]
    codes = sorted(int(item["code"]) for item in vocabulary)
    if codes != list(range(2, len(codes) + 2)):
        raise ExternalValidationError("external receiver vocabulary is not contiguous")
    cardinality = len(codes) + 2
    if cardinality != int(summary["receiver_vocabulary_cardinality"]):
        raise ExternalValidationError("external receiver cardinality manifest mismatch")
    gap_bins = len(transform["gap_edges"]) - 1
    if gap_bins != 16 or len(transform["gap_tau"]) != gap_bins:
        raise ExternalValidationError("external gap input shape is not frozen at 16 bins")
    if not math.isfinite(float(transform["amount_log_mean"])) or not (
        math.isfinite(float(transform["amount_log_std"]))
        and float(transform["amount_log_std"]) > 0
    ):
        raise ExternalValidationError("external amount transform is invalid")
    if any(
        int(split["length_max"]) != 32
        for split in summary["splits"].values()
    ):
        raise ExternalValidationError("external sequence length contract changed")
    return {
        "status": "PASS",
        "dataset": dataset,
        "pad_code": 0,
        "unk_code": 1,
        "receiver_vocabulary_cardinality": cardinality,
        "receiver_cardinality_role": (
            "train_only_input_vocabulary_not_architecture_change"
        ),
        "amount_input_width": 1,
        "gap_bin_cardinality": gap_bins,
        "sequence_length": 32,
        "architecture_change_required": False,
        "frozen_source_commit": commit,
        "frozen_config_sha256": frozen["config_sha256"],
        "model_source_sha256": frozen["model_sha256"],
        "adapter_source_sha256": frozen["adapter_sha256"],
        "denoiser_source_sha256": frozen["denoiser_sha256"],
        "transform_state_sha256": transform["state_sha256"],
    }


def build_external_adapter_contract(
    *, repo_root: Path, config_path: Path, dataset: str
) -> Mapping[str, Any]:
    """Map frozen v2.5 budgets without importing or mutating an adapter."""

    config = _load_config(config_path)
    _validate_config(config)
    input_contract = validate_external_input_and_frozen_cof_contract(
        repo_root=repo_root,
        config_path=config_path,
        dataset=dataset,
    )
    for model, record in config["adapter_sources"].items():
        if _sha256_file(repo_root / record["path"]) != record["sha256"]:
            raise ExternalValidationError(f"external adapter fingerprint mismatch: {model}")
    frozen_path = repo_root / config["frozen_cof"]["config_path"]
    frozen_bytes = frozen_path.read_bytes()
    frozen_config = yaml.safe_load(frozen_bytes)
    baselines = frozen_config["baselines"]
    ctgan = baselines["ctgan_separate_class"]
    tvae = baselines["tvae_separate_class"]
    cof = baselines["cof_seqgen"]
    return {
        "dataset": dataset,
        "model_import_calls": 0,
        "frozen_config_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "frozen_config_mutated": False,
        "empirical_iid": {
            "adapter": config["adapter_sources"]["empirical_iid"],
            "requested_steps": 0,
            "device_class": "cpu",
            "fit_split": "train",
        },
        "ctgan_separate_class": {
            "adapter": config["adapter_sources"]["ctgan_separate_class"],
            "requested_steps": int(ctgan["requested_steps_total"]),
            "class_requested_steps": dict(ctgan["requested_steps_per_class"]),
            "max_wall_seconds": int(ctgan["max_wall_seconds_total"]),
            "class_wall_seconds": {
                str(key): int(value)
                for key, value in ctgan["max_wall_seconds_per_class"].items()
            },
            "device_class": "gpu",
            "fit_split": "train",
        },
        "tvae_separate_class": {
            "adapter": config["adapter_sources"]["tvae_separate_class"],
            "requested_steps": int(tvae["requested_steps_total"]),
            "class_requested_steps": dict(tvae["requested_steps_per_class"]),
            "max_wall_seconds": int(tvae["max_wall_seconds_total"]),
            "class_wall_seconds": {
                str(key): int(value)
                for key, value in tvae["max_wall_seconds_per_class"].items()
            },
            "device_class": "gpu",
            "fit_split": "train",
        },
        "cof_seqgen_frozen_non_v3": {
            "adapter": config["adapter_sources"]["cof_seqgen_frozen_non_v3"],
            "requested_steps": int(cof["requested_steps"]),
            "max_wall_seconds": int(cof["max_wall_seconds"]),
            "device_class": "gpu",
            "fit_split": "train",
            "architecture_override": {},
            "runtime_data_bindings": {
                "tau": "train_transform_state.gap_tau",
                "receiver_cardinality": input_contract[
                    "receiver_vocabulary_cardinality"
                ],
                "receiver_cardinality_role": "input_vocabulary_only",
                "amount_input_width": input_contract["amount_input_width"],
                "gap_bin_cardinality": input_contract["gap_bin_cardinality"],
                "sequence_length": input_contract["sequence_length"],
            },
        },
    }


def build_external_validation_plan(
    *, repo_root: Path, config_path: Path, mode: str
) -> Mapping[str, Any]:
    if mode not in {"plan", "dry-run"}:
        raise ExternalValidationError("mode must be plan or dry-run")
    config = _load_config(config_path)
    _validate_config(config)
    from eval.external_validation_protocol_v1 import validate_external_analysis_policy

    analysis_policy = validate_external_analysis_policy(config)
    models = list(config["models"])
    runtime_root = Path(config["runtime"]["root"])
    preservation = validate_external_continuation_preservation(
        repo_root=repo_root, config=config
    )
    datasets: dict[str, Any] = {}
    bundles: dict[str, Any] = {}
    cof_contracts: dict[str, Any] = {}
    adapter_contracts: dict[str, Any] = {}
    manifest_reads = 0
    hash_reads = 0
    frozen_body_hash_reads = 0
    for dataset, record in config["datasets"].items():
        jobs = {
            model: {
                "attempt": _expected_attempt(config, model),
                "attempt_path": str(
                    runtime_root / dataset / model / _expected_attempt(config, model)
                ),
                "attempt_exists": (
                    repo_root
                    / runtime_root
                    / dataset
                    / model
                    / _expected_attempt(config, model)
                ).exists(),
                "append_only": True,
                "execution_authorized": False,
            }
            for model in models
        }
        datasets[dataset] = {
            "bundle_root": record["bundle_root"],
            "jobs": jobs,
            "internal_test_access": "fail_closed",
            "sparkov_fraud_test_access": "fail_closed",
        }
        if mode == "dry-run":
            bundles[dataset] = _verify_bundle(repo_root, dataset, record)
            cof_contracts[dataset] = validate_external_input_and_frozen_cof_contract(
                repo_root=repo_root,
                config_path=config_path,
                dataset=dataset,
            )
            adapter_contracts[dataset] = build_external_adapter_contract(
                repo_root=repo_root,
                config_path=config_path,
                dataset=dataset,
            )
            manifest_reads += int(bundles[dataset]["manifest_json_reads"])
            hash_reads += int(bundles[dataset]["bundle_hash_reads"])
            frozen_body_hash_reads += int(
                bundles[dataset]["frozen_body_hash_reads"]
            )
        else:
            bundles[dataset] = {"status": "NOT_READ_IN_PLAN"}
            cof_contracts[dataset] = {"status": "NOT_READ_IN_PLAN"}
            adapter_contracts[dataset] = {"status": "NOT_READ_IN_PLAN"}
    return {
        "schema_version": "external-validation-plan-v1",
        "mode": mode,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "models": models,
        "job_count": len(models) * len(datasets),
        "datasets": datasets,
        "bundles": bundles,
        "frozen_cof_contracts": cof_contracts,
        "adapter_contracts": adapter_contracts,
        "analysis_policy": analysis_policy,
        "performance_correction": dict(config["performance_correction"]),
        "attempts_by_model": dict(config["runtime"]["attempts_by_model"]),
        "launch_schedule": validate_external_launch_schedule(config),
        "continuation_preservation": preservation,
        "all_target_attempts_absent": all(
            not job["attempt_exists"]
            for dataset in datasets.values()
            for job in dataset["jobs"].values()
        ),
        "access_counts": {
            "manifest_json_reads": manifest_reads,
            "bundle_hash_reads": hash_reads,
            "frozen_body_hash_reads": frozen_body_hash_reads,
            "npz_load_calls": 0,
            "raw_csv_hash_reads": 0,
            "raw_csv_header_reads": 0,
            "raw_csv_body_reads": 0,
            "internal_test_npz_load_calls": 0,
            "sparkov_fraud_test_accesses": 0,
        },
        "execution_counts": dict(ZERO_EXECUTION_COUNTS),
        "runtime_artifacts_created": False,
        "execution_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "dry-run", "execute"), required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--dataset", choices=("amlsim", "sparkov"))
    parser.add_argument("--model", choices=(
        "empirical_iid",
        "ctgan_separate_class",
        "tvae_separate_class",
        "cof_seqgen_frozen_non_v3",
    ))
    parser.add_argument("--device")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "execute":
        if args.authorization is None:
            raise ExternalValidationError(
                "external validation execute requires a separate authorization"
            )
        if args.dataset is None or args.model is None or args.device is None:
            raise ExternalValidationError(
                "execute requires explicit dataset, model, and device"
            )
        authorization = _load_json(args.authorization.resolve())
        if (
            authorization.get("schema_version")
            in {
                "external-validation-batch-authorization-v1",
                "external-validation-performance-continuation-authorization-v1",
            }
        ):
            authorization = select_batch_authorized_job(
                repo_root=args.repo_root.resolve(),
                config_path=args.config.resolve(),
                authorization=authorization,
                dataset=args.dataset,
                model=args.model,
            )
        validated = validate_external_validation_authorization(
            repo_root=args.repo_root.resolve(),
            config_path=args.config.resolve(),
            authorization=authorization,
        )
        if validated["dataset"] != args.dataset or validated["model"] != args.model:
            raise ExternalValidationError("CLI dataset/model differs from authorization")
        if args.device != authorization.get("runner_device"):
            raise ExternalValidationError(
                "CLI device differs from the authorization launch schedule"
            )
        if authorization.get("continuation_kind") != (
            "amlsim_tvae_memory_safe_attempt_002"
        ):
            validate_external_wave_barrier(
                repo_root=args.repo_root.resolve(),
                config_path=args.config.resolve(),
                dataset=args.dataset,
                model=args.model,
            )
        from experiments.external_validation_runner_v1 import (
            execute_external_validation_job_bounded,
        )

        result = execute_external_validation_job_bounded(
            repo_root=args.repo_root.resolve(),
            config_path=args.config.resolve(),
            authorization=authorization,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.authorization is not None:
        raise ExternalValidationError("plan/dry-run reject execution authorization")
    if any(value is not None for value in (args.dataset, args.model, args.device)):
        raise ExternalValidationError("plan/dry-run reject execution-only options")
    result = build_external_validation_plan(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        mode=args.mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
