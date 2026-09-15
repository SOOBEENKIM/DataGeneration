"""CPU-synthetic execution-readiness gate for H1 attempt_003.

This module does not authorize or execute an experiment.  It writes only to a
caller-supplied scratch directory and exercises the production ownership
attach boundary plus one in-memory CPU optimizer step.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import queue
from typing import Any, Mapping

import yaml

from experiments.cof_hcmttpp_v2_execution_runner import (
    ExecutionJob,
    H1AttemptStore,
    H1RunnerContractError,
)


_ATTEMPT = "attempt_003"
_CANDIDATE = "H1"
_SEED = 4001
_DATASETS = ("amlsim", "sparkov")
_SCOPE = "SOURCE_ONLY_CPU_SYNTHETIC_EXECUTION_READINESS"
_PROCESS_TIMEOUT_SECONDS = 120.0


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _frozen_model_and_optimizer(repository_root: Path) -> dict[str, Any]:
    path = repository_root / "configs/benchmark_v2/cof_hcmttpp_v2_source_only.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    budget = raw["fixed_model_and_budget"]
    return {
        "seed": int(budget["seed"]),
        "d_model": int(budget["d_model"]),
        "n_heads": int(budget["n_heads"]),
        "n_layers": int(budget["n_layers"]),
        "max_length": int(budget["max_length"]),
        "dropout": float(budget["dropout"]),
        "optimizer": str(budget["optimizer"]),
        "learning_rate": float(budget["learning_rate"]),
        "weight_decay": float(budget["weight_decay"]),
    }


def _child_backend_entry(payload: Mapping[str, Any], output_queue: Any) -> None:
    try:
        from generators.cof_hcmttpp_v2_execution_backend import (
            run_cpu_readiness_attempt_003,
        )

        output_queue.put({"status": "PASS", "result": run_cpu_readiness_attempt_003(payload)})
    except BaseException as error:  # child must always report a bounded outcome
        output_queue.put(
            {
                "status": "ERROR",
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )


def _mutation_probe_entry(payload: Mapping[str, Any], output_queue: Any) -> None:
    from generators.cof_hcmttpp_v2_execution_backend import (
        attach_cpu_readiness_attempt_003,
    )

    mutations: dict[str, dict[str, Any]] = {}
    for field in (
        "ownership_id",
        "dataset",
        "candidate_id",
        "seed",
        "attempt",
        "authorization_sha256",
    ):
        mutated = json.loads(json.dumps(payload))
        if field == "ownership_id":
            mutated[field] = "f" * 64
        elif field == "dataset":
            mutated["job"][field] = "sparkov"
        elif field == "candidate_id":
            mutated["job"][field] = "H2"
        elif field == "seed":
            mutated["job"][field] = 4002
        elif field == "attempt":
            mutated["job"][field] = "attempt_002"
        else:
            mutated[field] = "e" * 64
        try:
            attach_cpu_readiness_attempt_003(mutated)
        except H1RunnerContractError:
            mutations[field] = {"status": "REJECTED"}
        else:
            mutations[field] = {"status": "ACCEPTED"}
    output_queue.put({"status": "PASS", "mutations": mutations})


def _spawn_and_receive(target: Any, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    context = multiprocessing.get_context("spawn")
    output_queue = context.Queue()
    process = context.Process(target=target, args=(payload, output_queue))
    process.start()
    try:
        message = output_queue.get(timeout=_PROCESS_TIMEOUT_SECONDS)
    except queue.Empty as error:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
        raise H1RunnerContractError(
            "CPU readiness child did not report within the bounded deadline"
        ) from error
    process.join(timeout=5.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
        raise H1RunnerContractError("CPU readiness child did not exit after reporting")
    if process.exitcode != 0 or message.get("status") != "PASS":
        raise H1RunnerContractError(
            "CPU readiness child failed: "
            f"exit={process.exitcode}, message={dict(message)}"
        )
    output_queue.close()
    output_queue.join_thread()
    return message


def _payload_for_dataset(
    *,
    repository_root: Path,
    scratch_root: Path,
    dataset: str,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    job = ExecutionJob(
        dataset=dataset,
        candidate_id=_CANDIDATE,
        seed=_SEED,
        attempt=_ATTEMPT,
    )
    source_paths = (
        repository_root / "experiments/cof_hcmttpp_v2_execution_runner.py",
        repository_root / "generators/cof_hcmttpp_v2_execution_backend.py",
        repository_root / "models/cof_hcmttpp_v2.py",
    )
    provenance_sha256 = _sha256_bytes(
        dataset.encode("utf-8") + b"\0" + b"\0".join(path.read_bytes() for path in source_paths)
    )
    authorization_sha256 = _sha256_bytes(
        json.dumps(
            {
                "scope": _SCOPE,
                "dataset": dataset,
                "candidate_id": _CANDIDATE,
                "seed": _SEED,
                "attempt": _ATTEMPT,
                "provenance_sha256": provenance_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    dataset_scratch = scratch_root / dataset
    store = H1AttemptStore.claim_readiness_attempt_003(
        scratch_root=dataset_scratch,
        job=job,
        authorization_sha256=authorization_sha256,
        provenance_sha256=provenance_sha256,
    )
    ownership = json.loads(store.ownership_path.read_text(encoding="utf-8"))
    attempt_path = store.allocate_readiness_attempt_003(
        {
            "schema_version": "cof-hcmttpp-v2-h1-execution-readiness-manifest-v1",
            "scope": _SCOPE,
            "dataset": dataset,
            "candidate_id": _CANDIDATE,
            "seed": _SEED,
            "attempt": _ATTEMPT,
            "authorization_sha256": authorization_sha256,
            "provenance_sha256": provenance_sha256,
            "ownership_id": ownership["ownership_id"],
            "runtime_execution_authorized": False,
        }
    )
    payload = {
        "repository_root": str(repository_root),
        "runtime_root": str(dataset_scratch.resolve()),
        "ownership_path": str(store.ownership_path),
        "attempt_path": str(attempt_path),
        "ownership_id": ownership["ownership_id"],
        "authorization_sha256": authorization_sha256,
        "job": {
            "dataset": dataset,
            "candidate_id": _CANDIDATE,
            "seed": _SEED,
            "attempt": _ATTEMPT,
        },
        "frozen_model_and_optimizer": dict(frozen),
    }
    # Explicitly prove the entire child payload survives spawn's pickle boundary.
    round_tripped = pickle.loads(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    if round_tripped != payload:
        raise H1RunnerContractError("spawn-compatible payload round trip changed")
    return round_tripped


def run_h1_attempt_003_readiness_gate(
    *, repository_root: Path, scratch_root: Path
) -> Mapping[str, Any]:
    """Run the one source-only integration gate spanning both dataset scopes."""

    repository_root = repository_root.resolve()
    scratch_root = scratch_root.resolve()
    if not repository_root.is_dir() or not scratch_root.is_dir():
        raise H1RunnerContractError("readiness roots must already exist")
    actual_runtime_root = (
        repository_root / "artifacts/cof_hcmttpp_v2/external_validation"
    ).resolve()
    try:
        scratch_root.relative_to(actual_runtime_root)
    except ValueError:
        pass
    else:
        raise H1RunnerContractError("readiness scratch cannot be an actual runtime root")

    frozen = _frozen_model_and_optimizer(repository_root)
    dataset_results: dict[str, Mapping[str, Any]] = {}
    payloads: dict[str, Mapping[str, Any]] = {}
    for dataset in _DATASETS:
        payload = _payload_for_dataset(
            repository_root=repository_root,
            scratch_root=scratch_root,
            dataset=dataset,
            frozen=frozen,
        )
        payloads[dataset] = payload
        dataset_results[dataset] = _spawn_and_receive(
            _child_backend_entry, payload
        )["result"]

    mutation_message = _spawn_and_receive(
        _mutation_probe_entry, payloads["amlsim"]
    )
    mutation_rejections = {
        key: value["status"]
        for key, value in mutation_message["mutations"].items()
    }
    if any(value != "REJECTED" for value in mutation_rejections.values()):
        raise H1RunnerContractError("mutated readiness ownership was accepted")

    return {
        "status": "PASS",
        "scope": {
            "candidate_id": _CANDIDATE,
            "seed": _SEED,
            "attempt": _ATTEMPT,
            "datasets": list(_DATASETS),
        },
        "datasets": dataset_results,
        "mutation_rejections": mutation_rejections,
        "execution_counts": {
            "gpu_queries": 0,
            "cuda_calls": 0,
            "external_data_body_reads": 0,
            "actual_runtime_root_writes": 0,
            "model_fit_calls": 0,
            "model_sample_calls": 0,
            "evaluation_calls": 0,
            "optimizer_steps_per_dataset": 1,
        },
    }
