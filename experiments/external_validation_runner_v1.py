"""Authorized external validation v1 execution worker.

This module is imported only by ``--mode execute``.  Plan and dry-run therefore
cannot import model dependencies or load arrays.  One invocation owns exactly
one dataset/model attempt and never opens an external test split.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
from typing import Any, Mapping

import numpy as np
import yaml

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.external_validation_metrics_v1 import (
    ExternalMetricError,
    compute_external_validation_metrics,
    validate_external_hard_contract,
)
from eval.external_validation_protocol_v1 import (
    COHERENCE_METRICS,
    FIDELITY_METRICS,
    select_external_candidate,
    validate_external_analysis_policy,
)
from generators.sampling_plan import SamplingPlan


class ExternalExecutionError(RuntimeError):
    """Raised when an authorized external worker violates its contract."""


def validate_external_transform_memory_policy(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the external-only, resource-policy transform contract."""

    policy = config.get("memory_safety", {}).get("data_transformer")
    expected = {
        "models": ["ctgan_separate_class", "tvae_separate_class"],
        "fixed_n_jobs": 1,
        "unbounded_n_jobs_forbidden": True,
        "execution_mode": "synchronous_column_transform",
        "max_concurrent_heavy_transforms": 1,
        "algorithm_seed_conditioning_metrics_unchanged": True,
    }
    if policy != expected:
        raise ExternalExecutionError(
            "external DataTransformer memory-safety policy changed"
        )
    return dict(policy)


def estimate_external_transform_memory(
    *,
    valid_rows: int,
    receiver_cardinality: int,
    output_dtype_bytes: int,
    concurrent_tabular_jobs: int,
) -> Mapping[str, Any]:
    """Return a non-allocating dense receiver transform risk estimate."""

    values = (
        valid_rows,
        receiver_cardinality,
        output_dtype_bytes,
        concurrent_tabular_jobs,
    )
    if any(int(value) < 1 for value in values):
        raise ExternalExecutionError("transform risk dimensions must be positive")
    dense_bytes = int(valid_rows) * int(receiver_cardinality) * int(
        output_dtype_bytes
    )
    concurrent_bytes = dense_bytes * int(concurrent_tabular_jobs)
    gib = 1024**3
    return {
        "valid_rows": int(valid_rows),
        "receiver_cardinality": int(receiver_cardinality),
        "output_dtype_bytes": int(output_dtype_bytes),
        "concurrent_tabular_jobs": int(concurrent_tabular_jobs),
        "receiver_dense_matrix_bytes": dense_bytes,
        "receiver_dense_matrix_gib": dense_bytes / gib,
        "concurrent_receiver_dense_matrix_bytes": concurrent_bytes,
        "concurrent_receiver_dense_matrix_gib": concurrent_bytes / gib,
        "upstream_parallel_n_jobs": -1,
        "risk": "GLOBAL_OOM_PLAUSIBLE",
        "arrays_allocated": 0,
    }


@contextmanager
def bounded_data_transformer_policy(*, n_jobs: int):
    """Force CTGAN's column transform through one in-process worker.

    The upstream transform dispatches inputs with at least 500 rows through
    ``Parallel(n_jobs=-1)``. External tabular jobs instead use the upstream
    synchronous column methods in their original order. This changes only
    resource scheduling, not fitted transforms, output columns, seeds, or data.
    """

    if int(n_jobs) != 1:
        raise ExternalExecutionError(
            "external DataTransformer n_jobs must be exactly 1"
        )
    from ctgan.data_transformer import DataTransformer

    original = DataTransformer._parallel_transform

    def synchronous_transform(instance, raw_data, column_transform_info_list):
        return instance._synchronous_transform(
            raw_data, column_transform_info_list
        )

    audit = {
        "configured_n_jobs": 1,
        "joblib_worker_processes": 0,
        "execution_mode": "synchronous_column_transform",
        "upstream_n_jobs_minus_one_reachable": False,
    }
    DataTransformer._parallel_transform = synchronous_transform
    try:
        yield audit
    finally:
        DataTransformer._parallel_transform = original


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    _exclusive_bytes(path, _json_bytes(value))


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    with os.fdopen(descriptor, "ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _exclusive_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_batch(path: Path) -> SequenceBatch:
    with np.load(path, allow_pickle=False) as archive:
        fields = {
            name: archive[name]
            for name in SequenceBatch.__dataclass_fields__
        }
    return SequenceBatch(**fields)


def _synthetic_from_sequence(batch: SequenceBatch) -> SyntheticBatch:
    return SyntheticBatch(
        x_num=batch.x_num.copy(),
        dt_bin=batch.dt_bin.copy(),
        x_cat=batch.x_cat.copy(),
        valid_mask=batch.valid_mask.copy(),
        y_entity=batch.y_entity.copy(),
        lengths=batch.lengths.copy(),
    )


def _entity_bootstrap(batch: SequenceBatch, rng: np.random.Generator) -> SequenceBatch:
    entities, inverse = np.unique(batch.entity_ids, return_inverse=True)
    selected_entities = rng.integers(0, len(entities), size=len(entities))
    indices = np.concatenate(
        [np.flatnonzero(inverse == selected) for selected in selected_entities]
    )
    return SequenceBatch(
        x_num=batch.x_num[indices],
        dt_bin=batch.dt_bin[indices],
        x_cat=batch.x_cat[indices],
        valid_mask=batch.valid_mask[indices],
        y_entity=batch.y_entity[indices],
        lengths=batch.lengths[indices],
        entity_ids=np.arange(len(indices), dtype=np.int64),
    )


def fit_train_only_bootstrap_reference(
    *,
    train: SequenceBatch,
    gap_tau: np.ndarray,
    seed: int,
    policy: Mapping[str, Any],
    train_sha256: str,
    threshold_config_sha256: str,
) -> Mapping[str, Any]:
    """Fit the preregistered threshold bundle using training entities only."""

    replicates = int(policy["replicates"])
    rank = int(policy["order_statistic_rank"])
    if (
        policy.get("source_split") != "train"
        or policy.get("cluster_unit") != "entity"
        or policy.get("interpolation") != "none"
        or rank < 1
        or rank > replicates
    ):
        raise ExternalExecutionError("invalid train-only bootstrap policy")
    positions = np.arange(train.valid_mask.shape[1])[None, :]
    positive_gap = train.valid_mask & (positions > 0) & (gap_tau[train.dt_bin] > 0)
    if not positive_gap.any():
        raise ExternalExecutionError("train has no positive gaps for coherence threshold")
    short_gap_threshold = float(np.median(gap_tau[train.dt_bin[positive_gap]]))
    observed = {metric: [] for metric in (*FIDELITY_METRICS, *COHERENCE_METRICS)}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        left = _entity_bootstrap(train, rng)
        right = _entity_bootstrap(train, rng)
        result = compute_external_validation_metrics(
            real=left,
            synthetic=_synthetic_from_sequence(right),
            gap_tau=gap_tau,
            short_gap_threshold=short_gap_threshold,
        )
        for family in ("fidelity", "coherence"):
            for metric, value in result[family].items():
                observed[metric].append(float(value))
    return {
        "schema_version": "external-train-bootstrap-thresholds-v1",
        "source_split": "train",
        "cluster_unit": "entity",
        "bootstrap_replicates": replicates,
        "order_statistic_rank": rank,
        "interpolation": "none",
        "controlled_benchmark_thresholds_copied": False,
        "frozen_before_validation": True,
        "train_sha256": train_sha256,
        "threshold_config_sha256": threshold_config_sha256,
        "short_gap_threshold": short_gap_threshold,
        "fidelity": {
            metric: float(np.sort(observed[metric])[rank - 1])
            for metric in FIDELITY_METRICS
        },
        "coherence": {
            metric: float(np.sort(observed[metric])[rank - 1])
            for metric in COHERENCE_METRICS
        },
    }


def _adapter_and_config(
    *,
    model: str,
    frozen_config: Mapping[str, Any],
    transform: Mapping[str, Any],
    device: str,
    progress_callback,
    checkpoint_callback,
):
    """Lazily import exactly the authorized frozen adapter."""

    baseline = frozen_config["baselines"]
    if model == "empirical_iid":
        from generators.empirical_conditional_iid import EmpiricalConditionalIID

        return EmpiricalConditionalIID(), {}
    if model == "ctgan_separate_class":
        from generators.conditional_ctgan import ConditionalCTGAN

        source = baseline[model]
        optimizer = source["optimizer"]
        return ConditionalCTGAN(), {
            "batch_size": source["batch_size"],
            "generator_lr": optimizer["generator"]["lr"],
            "generator_decay": optimizer["generator"]["weight_decay"],
            "discriminator_lr": optimizer["discriminator"]["lr"],
            "discriminator_decay": optimizer["discriminator"]["weight_decay"],
            "requested_steps_total": source["requested_steps_total"],
            "requested_steps_per_class": source["requested_steps_per_class"],
            "max_wall_seconds_total": source["max_wall_seconds_total"],
            "max_wall_seconds_per_class": source["max_wall_seconds_per_class"],
            "checkpoint_interval_steps": source["checkpoint_interval_steps"],
            "cuda": device.startswith("cuda"),
            "progress_callback": progress_callback,
            "checkpoint_callback": checkpoint_callback,
        }
    if model == "tvae_separate_class":
        from generators.conditional_tvae import ConditionalTVAE

        source = baseline[model]
        optimizer = source["optimizer"]
        return ConditionalTVAE(), {
            "batch_size": source["batch_size"],
            "lr": optimizer["lr"],
            "weight_decay": optimizer["weight_decay"],
            "requested_steps_total": source["requested_steps_total"],
            "requested_steps_per_class": source["requested_steps_per_class"],
            "max_wall_seconds_total": source["max_wall_seconds_total"],
            "max_wall_seconds_per_class": source["max_wall_seconds_per_class"],
            "checkpoint_interval_steps": source["checkpoint_interval_steps"],
            "cuda": device.startswith("cuda"),
            "progress_callback": progress_callback,
            "checkpoint_callback": checkpoint_callback,
        }
    if model == "cof_seqgen_frozen_non_v3":
        from generators.cof_seqgen_adapter import CoFSeqGenAdapter

        source = baseline["cof_seqgen"]
        optimizer = source["optimizer"]
        return CoFSeqGenAdapter(), {
            "d_model": source["d_model"],
            "n_layers": source["n_layers"],
            "batch_size": source["batch_size"],
            "lr": optimizer["lr"],
            "weight_decay": optimizer["weight_decay"],
            "requested_steps": source["requested_steps"],
            "max_wall_seconds": source["max_wall_seconds"],
            "checkpoint_interval_steps": source["checkpoint_interval_steps"],
            "coherence_lambda": source["coherence_lambda"],
            "cfg_dropout": source["cfg_dropout"],
            "guidance_scale": source["guidance_scale"],
            "diffusion_steps": source["diffusion_steps"],
            "sampling_chunk_size": source["sampling_chunk_size"],
            "discrete_mask_max": source["discrete_mask_max"],
            "start_from_mask": source["start_from_mask"],
            "feedback_discrete": source["feedback_discrete"],
            "feedback_after": source["feedback_after"],
            "tau": transform["gap_tau"],
            "device": device,
            "progress_callback": progress_callback,
            "checkpoint_callback": checkpoint_callback,
        }
    raise ExternalExecutionError(f"unsupported external model: {model}")


def _save_checkpoint_exclusive(adapter: Any, path: Path) -> None:
    if path.exists():
        raise ExternalExecutionError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        adapter.save_training_checkpoint(temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def execute_external_validation_job(
    *,
    repo_root: Path,
    config_path: Path,
    authorization: Mapping[str, Any],
    device: str,
) -> Mapping[str, Any]:
    """Execute one pre-authorized dataset/model job append-only."""

    # Imported here to keep plan/dry-run free of execution-only dependencies.
    from scripts.run_external_validation_v1 import (
        build_external_execution_manifest,
        claim_external_validation_attempt,
        resolve_external_bundle_access,
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_external_analysis_policy(config)
    transform_memory_policy = validate_external_transform_memory_policy(config)
    manifest = dict(
        build_external_execution_manifest(
            repo_root=repo_root,
            config_path=config_path,
            authorization=authorization,
        )
    )
    dataset = str(manifest["dataset"])
    model = str(manifest["model"])
    if model == "empirical_iid" and device != "cpu":
        raise ExternalExecutionError("empirical IID external control requires cpu")
    if model != "empirical_iid" and not device.startswith("cuda"):
        raise ExternalExecutionError("learned external models require an explicit CUDA device")
    attempt_path = Path(manifest["attempt_path"])
    claim_external_validation_attempt(attempt_path, manifest)
    _exclusive_json(attempt_path / "manifest.json", manifest)
    started = time.monotonic()
    try:
        train_path = resolve_external_bundle_access(
            repo_root=repo_root,
            config_path=config_path,
            dataset=dataset,
            split="train",
            purpose="model_fit",
        )
        validation_path = resolve_external_bundle_access(
            repo_root=repo_root,
            config_path=config_path,
            dataset=dataset,
            split="validation",
            purpose="candidate_generation_and_evaluation",
        )
        if _sha256(train_path) != manifest["data_hashes"]["train"] or _sha256(
            validation_path
        ) != manifest["data_hashes"]["validation"]:
            raise ExternalExecutionError("authorized external split hash changed")
        train = _load_batch(train_path)
        transform_path = train_path.parent / "train_transform_state.json"
        transform = json.loads(transform_path.read_text(encoding="utf-8"))
        gap_tau = np.asarray(transform["gap_tau"], dtype=float)
        threshold_config_hash = hashlib.sha256(
            _json_bytes(config["analysis"]["bootstrap"])
        ).hexdigest()
        thresholds = fit_train_only_bootstrap_reference(
            train=train,
            gap_tau=gap_tau,
            seed=int(manifest["seed"]) + 10,
            policy=config["analysis"]["bootstrap"],
            train_sha256=manifest["data_hashes"]["train"],
            threshold_config_sha256=threshold_config_hash,
        )
        _exclusive_json(attempt_path / "train_bootstrap_thresholds.json", thresholds)

        def progress(event: Mapping[str, Any]) -> None:
            _append_jsonl(attempt_path / "progress.jsonl", dict(event))

        def checkpoint(event: Mapping[str, Any], adapter: Any) -> None:
            step = int(event["step"])
            _save_checkpoint_exclusive(adapter, attempt_path / "checkpoints" / f"step_{step:06d}.pt")

        frozen_config = yaml.safe_load(
            (repo_root / config["frozen_cof"]["config_path"]).read_text(encoding="utf-8")
        )
        adapter, adapter_config = _adapter_and_config(
            model=model,
            frozen_config=frozen_config,
            transform=transform,
            device=device,
            progress_callback=progress,
            checkpoint_callback=checkpoint,
        )
        transform_policy_audit: Mapping[str, Any] = {
            "configured_n_jobs": 0,
            "joblib_worker_processes": 0,
            "execution_mode": "not_applicable",
            "upstream_n_jobs_minus_one_reachable": False,
        }
        if model in transform_memory_policy["models"]:
            with bounded_data_transformer_policy(
                n_jobs=int(transform_memory_policy["fixed_n_jobs"])
            ) as active_policy:
                transform_policy_audit = dict(active_policy)
                adapter.fit(
                    train, config=adapter_config, seed=int(manifest["seed"])
                )
        else:
            adapter.fit(train, config=adapter_config, seed=int(manifest["seed"]))
        if hasattr(adapter, "save_training_checkpoint"):
            _save_checkpoint_exclusive(adapter, attempt_path / "checkpoints" / "final.pt")
        # Validation arrays are deliberately loaded only after every train-only
        # fit and reference artifact has been frozen.
        validation = _load_batch(validation_path)
        plan = SamplingPlan.from_batch(validation)
        _exclusive_npz(
            attempt_path / "validation_sampling_plan.npz",
            {
                "y_entity": plan.y_entity,
                "lengths": plan.lengths,
                "valid_mask": plan.valid_mask,
                "plan_hash": np.asarray(plan.plan_hash),
            },
        )
        synthetic = adapter.sample(plan, seed=int(manifest["seed"]) + 1)
        hard = validate_external_hard_contract(
            real_validation=validation,
            synthetic=synthetic,
            gap_cardinality=len(transform["gap_tau"]),
            receiver_cardinality=int(
                json.loads((train_path.parent / "summary.json").read_text(encoding="utf-8"))[
                    "receiver_vocabulary_cardinality"
                ]
            ),
        )
        metrics = compute_external_validation_metrics(
            real=validation,
            synthetic=synthetic,
            gap_tau=gap_tau,
            short_gap_threshold=float(thresholds["short_gap_threshold"]),
        )
        candidate = {
            "candidate_id": f"{dataset}_{model}_frozen_v1",
            "hard_validity": hard,
            "fidelity": metrics["fidelity"],
            "coherence": metrics["coherence"],
            "source_sha256": manifest["runner_source_sha256"],
            "sample_sha256": "pending",
        }
        sample_path = attempt_path / "sample.npz"
        _exclusive_npz(
            sample_path,
            {
                "x_num": synthetic.x_num,
                "dt_bin": synthetic.dt_bin,
                "x_cat": synthetic.x_cat,
                "valid_mask": synthetic.valid_mask,
                "y_entity": synthetic.y_entity,
                "lengths": synthetic.lengths,
            },
        )
        candidate["sample_sha256"] = _sha256(sample_path)
        selection = select_external_candidate(
            model_id=model,
            candidates=[candidate],
            thresholds=thresholds,
        )
        evaluation = {
            "schema_version": "external-validation-evaluation-v1",
            "dataset": dataset,
            "model": model,
            "sampling_plan_sha256": _sha256(attempt_path / "validation_sampling_plan.npz"),
            "hard_validity": hard,
            "metrics": metrics,
            "thresholds_sha256": _sha256(attempt_path / "train_bootstrap_thresholds.json"),
            "selection": selection,
            "test_execution_authorized": False,
        }
        _exclusive_json(attempt_path / "metrics.json", metrics)
        _exclusive_json(attempt_path / "evaluation.json", evaluation)
        _exclusive_json(
            attempt_path / "runtime.json",
            {
                "elapsed_seconds": time.monotonic() - started,
                "device": device,
                "seed": manifest["seed"],
                "model": model,
                "data_transformer_memory_policy": dict(
                    transform_policy_audit
                ),
            },
        )
        index = {
            str(path.relative_to(attempt_path)): _sha256(path)
            for path in sorted(attempt_path.rglob("*"))
            if path.is_file() and path.name not in {"artifact_index.json", "COMPLETE.json"}
        }
        _exclusive_json(attempt_path / "artifact_index.json", index)
        complete = {
            "status": "COMPLETE",
            "dataset": dataset,
            "model": model,
            "artifact_index_sha256": _sha256(attempt_path / "artifact_index.json"),
            "test_execution_authorized": False,
        }
        _exclusive_json(attempt_path / "COMPLETE.json", complete)
        return complete
    except ExternalMetricError as error:
        invalid = {
            "status": "INVALID",
            "dataset": dataset,
            "model": model,
            "failure_class": "hard_validity_contract",
            "message": str(error),
            "elapsed_seconds": time.monotonic() - started,
            "test_execution_authorized": False,
        }
        _exclusive_json(attempt_path / "INVALID.json", invalid)
        return invalid
    except Exception as error:
        _exclusive_json(
            attempt_path / "FAILED.json",
            {
                "status": "FAILED",
                "dataset": dataset,
                "model": model,
                "failure_class": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.monotonic() - started,
                "test_execution_authorized": False,
            },
        )
        raise


def _external_validation_child_entry(result_queue: Any, kwargs: Mapping[str, Any]) -> None:
    try:
        result_queue.put(
            {"kind": "result", "value": dict(execute_external_validation_job(**kwargs))}
        )
    except BaseException as error:
        result_queue.put(
            {
                "kind": "exception",
                "exception": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        raise


def execute_external_validation_job_bounded(
    *,
    repo_root: Path,
    config_path: Path,
    authorization: Mapping[str, Any],
    device: str,
) -> Mapping[str, Any]:
    """Run one authorized job in a runner-owned child under a hard wall cap."""

    import multiprocessing
    import queue

    from scripts.run_external_validation_v1 import (
        build_external_execution_manifest,
        claim_external_validation_attempt,
    )

    max_wall_seconds = float(authorization.get("job_hard_cap_seconds", 0))
    if max_wall_seconds <= 0:
        raise ExternalExecutionError("authorization lacks a positive whole-job hard cap")
    manifest = dict(
        build_external_execution_manifest(
            repo_root=repo_root,
            config_path=config_path,
            authorization=authorization,
        )
    )
    attempt_path = Path(manifest["attempt_path"])
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_external_validation_child_entry,
        args=(
            result_queue,
            {
                "repo_root": repo_root,
                "config_path": config_path,
                "authorization": dict(authorization),
                "device": device,
            },
        ),
    )
    started = time.monotonic()
    process.start()
    process.join(max_wall_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10.0)
        if process.is_alive():
            process.kill()
            process.join(10.0)
        if process.is_alive():
            raise ExternalExecutionError("runner-owned external child survived SIGKILL")
        if not attempt_path.exists():
            claim_external_validation_attempt(attempt_path, manifest)
            _exclusive_json(attempt_path / "manifest.json", manifest)
        terminals = [
            name
            for name in ("COMPLETE.json", "INVALID.json", "FAILED.json")
            if (attempt_path / name).is_file()
        ]
        if not terminals:
            _exclusive_json(
                attempt_path / "FAILED.json",
                {
                    "status": "FAILED",
                    "dataset": manifest["dataset"],
                    "model": manifest["model"],
                    "failure_class": "wall_cap",
                    "message": (
                        "runner-owned whole-job deadline reached at "
                        f"{max_wall_seconds} seconds"
                    ),
                    "elapsed_seconds": time.monotonic() - started,
                    "runner_owned_child_terminated": True,
                    "test_execution_authorized": False,
                },
            )
        return json.loads((attempt_path / (terminals[0] if terminals else "FAILED.json")).read_text())
    try:
        envelope = result_queue.get(timeout=10.0)
    except queue.Empty:
        envelope = {
            "kind": "exception",
            "exception": f"child exited {process.exitcode} without result metadata",
            "traceback": "",
        }
    if envelope["kind"] == "result":
        return dict(envelope["value"])
    failed_path = attempt_path / "FAILED.json"
    if failed_path.is_file():
        return json.loads(failed_path.read_text(encoding="utf-8"))
    raise ExternalExecutionError(
        "external child failed without append-only terminal marker: "
        + str(envelope.get("exception"))
    )
