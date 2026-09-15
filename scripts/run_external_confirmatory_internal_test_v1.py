"""Confirmatory external internal-test runner.

``plan`` and ``dry-run`` inspect only JSON/YAML manifests and byte hashes of
validation artifacts.  They never import a generator, load an NPZ archive, or
query CUDA.  ``execute`` is unavailable without a separately created,
hash-bound authorization and owns exactly one of the eight preregistered jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

import yaml


class ConfirmatoryInternalTestError(RuntimeError):
    """Raised when confirmatory scope or provenance is not exact."""


DATASETS = ("amlsim", "sparkov")
MODELS = (
    "empirical_iid",
    "ctgan_separate_class",
    "tvae_separate_class",
    "cof_seqgen_frozen_non_v3",
)
DEFAULT_CONFIG = Path(
    "configs/benchmark_v2/external_confirmatory_internal_test_v1.yaml"
)
RELEVANT_SOURCE_FILES = (
    "scripts/run_external_confirmatory_internal_test_v1.py",
    "experiments/external_confirmatory_internal_test_runner_v1.py",
    "eval/external_validation_metrics_v1.py",
    "eval/external_validation_protocol_v1.py",
    "generators/empirical_conditional_iid.py",
    "generators/conditional_ctgan.py",
    "generators/conditional_tvae.py",
    "generators/checkpointable_tabular_v2_5.py",
    "generators/cof_seqgen_adapter.py",
    "models/cof_seqgen.py",
    "models/seq_denoiser.py",
)
ZERO_INSTRUMENTATION = {
    "gpu_inventory_queries": 0,
    "cuda_calls": 0,
    "model_import_calls": 0,
    "model_fit_calls": 0,
    "model_sample_calls": 0,
    "internal_test_npz_body_reads": 0,
    "evaluation_calls": 0,
    "authorization_artifacts_created": 0,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryInternalTestError(
            f"cannot read JSON provenance: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise ConfirmatoryInternalTestError(f"JSON provenance is not an object: {path}")
    return value


def _read_config(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfirmatoryInternalTestError(
            f"cannot read confirmatory config: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise ConfirmatoryInternalTestError("confirmatory config must be a mapping")
    return value


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


def resolve_confirmatory_path(
    path_text: str,
    *,
    resolver: Callable[[Path], Path] | None = None,
) -> Path:
    """Resolve an allowed path, rejecting Sparkov fraudTest first.

    The string check intentionally occurs before constructing or resolving a
    ``Path`` so a forbidden public-test path cannot trigger filesystem access.
    """

    normalized = str(path_text).replace("_", "").replace("-", "").casefold()
    if "fraudtest" in normalized:
        raise ConfirmatoryInternalTestError(
            "Sparkov fraudTest is forbidden before path resolution"
        )
    path = Path(path_text)
    return (resolver or (lambda candidate: candidate.resolve()))(path)


def _require_exact_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "external-confirmatory-internal-test-v1":
        raise ConfirmatoryInternalTestError("confirmatory schema changed")
    if tuple(config.get("models", ())) != MODELS:
        raise ConfirmatoryInternalTestError("confirmatory model list must be exact")
    datasets = config.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != set(DATASETS):
        raise ConfirmatoryInternalTestError("confirmatory datasets must be exact")
    execution = config.get("execution_plan", {})
    expected_execution = {
        "job_count": 8,
        "attempts": "attempt_001_only",
        "one_execution_per_dataset_model": True,
        "retries": "forbidden",
        "early_stopping": "forbidden",
        "tuning_or_sweep": "forbidden",
        "artifact_root": "artifacts/external_confirmatory_internal_test_v1",
        "append_only": True,
        "authorization_required": True,
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise ConfirmatoryInternalTestError(
                f"confirmatory execution contract changed: {key}"
            )
    frozen = config.get("frozen_contract", {})
    for key in (
        "model_training_calls",
        "transform_fit_calls",
        "threshold_fit_calls",
        "validation_based_selection_calls",
    ):
        if frozen.get(key) != 0:
            raise ConfirmatoryInternalTestError(f"frozen zero-call contract changed: {key}")
    for key in (
        "seed_changes_allowed",
        "model_or_hyperparameter_changes_allowed",
        "evaluator_or_formula_changes_allowed",
    ):
        if frozen.get(key) is not False:
            raise ConfirmatoryInternalTestError(f"frozen prohibition changed: {key}")
    evaluation = config.get("evaluation", {})
    if (
        evaluation.get("split") != "internal_test"
        or int(evaluation.get("sample_seed_offset", -1)) != 1
        or evaluation.get("all_models_reported_without_test_selection") is not True
        or evaluation.get("partial_or_invalid_result_exclusion") != "forbidden"
    ):
        raise ConfirmatoryInternalTestError("confirmatory evaluation contract changed")
    reuse = config.get("validation_attempt_reuse")
    if not isinstance(reuse, Mapping) or set(reuse) != set(DATASETS):
        raise ConfirmatoryInternalTestError("validation reuse mapping is incomplete")
    for dataset in DATASETS:
        if set(reuse[dataset]) != set(MODELS):
            raise ConfirmatoryInternalTestError(
                f"validation reuse model mapping is incomplete: {dataset}"
            )


def _verify_indexed_file(
    attempt_root: Path,
    index: Mapping[str, Any],
    relative: str,
) -> str:
    path = attempt_root / relative
    if not path.is_file():
        raise ConfirmatoryInternalTestError(
            f"validation artifact is missing: {path}"
        )
    actual = _sha256_file(path)
    if index.get(relative) != actual:
        raise ConfirmatoryInternalTestError(
            f"validation artifact index mismatch: {path}"
        )
    return actual


def _bundle_manifest(
    repo_root: Path, dataset: str
) -> tuple[Path, Mapping[str, Any], str]:
    path = (
        repo_root
        / "artifacts/external_sequence_protocol_v1/materialization"
        / dataset
        / "attempt_001/checksum_manifest.json"
    )
    manifest = _read_json(path)
    files = manifest.get("data_files")
    required = {
        "train.npz",
        "internal_test.npz",
        "train_transform_state.json",
        "summary.json",
    }
    if (
        manifest.get("dataset") != dataset
        or not isinstance(files, Mapping)
        or not required.issubset(files)
    ):
        raise ConfirmatoryInternalTestError(
            f"frozen bundle checksum manifest is incomplete: {dataset}"
        )
    return path, manifest, _sha256_file(path)


def build_confirmatory_plan(
    repo_root: Path,
    config_path: Path,
    *,
    source_head: str | None = None,
) -> Mapping[str, Any]:
    """Build the exact eight-job plan without loading data arrays or models."""

    repository = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repository / config_file
    config = _read_config(config_file)
    _require_exact_config(config)
    config_hash = _sha256_file(config_file)
    source_inventory = {
        relative: _sha256_file(repository / relative)
        for relative in RELEVANT_SOURCE_FILES
        if (repository / relative).is_file()
    }
    if source_head is None and set(source_inventory) != set(RELEVANT_SOURCE_FILES):
        raise ConfirmatoryInternalTestError(
            "confirmatory relevant source inventory is incomplete"
        )
    frozen_cof_fingerprint = {
        "frozen_cof_adapter_sha256": source_inventory.get(
            "generators/cof_seqgen_adapter.py"
        ),
        "frozen_cof_model_sha256": source_inventory.get("models/cof_seqgen.py"),
        "frozen_cof_denoiser_sha256": source_inventory.get(
            "models/seq_denoiser.py"
        ),
        "frozen_cof_config_sha256": (
            _sha256_file(repository / "configs/benchmark_v2/full_v2_5.yaml")
            if (repository / "configs/benchmark_v2/full_v2_5.yaml").is_file()
            else None
        ),
        "frozen_cof_source_commit": "99a445f6dc893a8c2240d950de4f92877cc07f8a",
    }
    jobs: list[Mapping[str, Any]] = []
    datasets: dict[str, Mapping[str, Any]] = {}
    for dataset in DATASETS:
        dataset_config = config["datasets"][dataset]
        if dataset == "sparkov" and dataset_config.get(
            "sparkov_fraudTest_access"
        ) != "forbidden_before_path_resolution":
            raise ConfirmatoryInternalTestError(
                "Sparkov fraudTest pre-resolution prohibition changed"
            )
        test_relative = str(dataset_config["test_input"])
        test_path = resolve_confirmatory_path(
            str(repository / test_relative), resolver=lambda value: value.resolve()
        )
        expected_test = (
            repository
            / "data/external_sequence_protocol_v1/frozen"
            / dataset
            / "attempt_001/internal_test.npz"
        ).resolve()
        if test_path != expected_test or not test_path.is_file():
            raise ConfirmatoryInternalTestError(
                f"internal-test input is not the frozen protocol path: {dataset}"
            )
        checksum_path, checksum_manifest, checksum_hash = _bundle_manifest(
            repository, dataset
        )
        checksums = checksum_manifest["data_files"]
        bundle = expected_test.parent
        transform_path = bundle / "train_transform_state.json"
        summary_path = bundle / "summary.json"
        if (
            _sha256_file(transform_path) != checksums["train_transform_state.json"]
            or _sha256_file(summary_path) != checksums["summary.json"]
        ):
            raise ConfirmatoryInternalTestError(
                f"frozen train-only state hash mismatch: {dataset}"
            )
        # Plan/dry-run deliberately does not open either NPZ body.  Their hashes
        # are inherited from the append-only materialization checksum manifest.
        datasets[dataset] = {
            "seed": int(dataset_config["seed"]),
            "bundle_checksum_manifest_path": str(checksum_path),
            "bundle_checksum_manifest_sha256": checksum_hash,
            "internal_test_path": str(test_path),
            "internal_test_sha256": checksums["internal_test.npz"],
            "internal_test_hash_source": "append_only_materialization_checksum_manifest",
            "internal_test_npz_body_read": False,
            "train_pool_path": str(bundle / "train.npz"),
            "train_pool_sha256": checksums["train.npz"],
            "transform_path": str(transform_path),
            "transform_sha256": checksums["train_transform_state.json"],
            "summary_path": str(summary_path),
            "summary_sha256": checksums["summary.json"],
        }
        expected_threshold = str(dataset_config["train_only_thresholds_sha256"])
        expected_validation_plan = str(
            dataset_config["validation_sampling_plan_sha256"]
        )
        for model in MODELS:
            validation_attempt = str(
                config["validation_attempt_reuse"][dataset][model]
            )
            validation_root = (
                repository
                / "artifacts/external_validation_v1"
                / dataset
                / model
                / validation_attempt
            )
            terminal_path = validation_root / "COMPLETE.json"
            terminal = _read_json(terminal_path)
            if (
                terminal.get("status") != "COMPLETE"
                or terminal.get("dataset") != dataset
                or terminal.get("model") != model
            ):
                raise ConfirmatoryInternalTestError(
                    f"validation terminal is not COMPLETE: {dataset}/{model}"
                )
            index_path = validation_root / "artifact_index.json"
            index = _read_json(index_path)
            manifest_hash = _verify_indexed_file(
                validation_root, index, "manifest.json"
            )
            validation_manifest = _read_json(validation_root / "manifest.json")
            if (
                validation_manifest.get("dataset") != dataset
                or validation_manifest.get("model") != model
                or int(validation_manifest.get("seed", -1))
                != int(dataset_config["seed"])
                or validation_manifest.get("retry_allowed") is not False
                or validation_manifest.get("fit_split") != "train"
                or validation_manifest.get("evaluation_split") != "validation"
                or validation_manifest.get("data_hashes", {}).get("train")
                != checksums["train.npz"]
            ):
                raise ConfirmatoryInternalTestError(
                    f"validation manifest scope mismatch: {dataset}/{model}"
                )
            if source_head is None and any(
                validation_manifest.get(key) != value
                for key, value in frozen_cof_fingerprint.items()
            ):
                raise ConfirmatoryInternalTestError(
                    f"frozen non-v3 CoF fingerprint mismatch: {dataset}/{model}"
                )
            threshold_hash = _verify_indexed_file(
                validation_root, index, "train_bootstrap_thresholds.json"
            )
            plan_hash = _verify_indexed_file(
                validation_root, index, "validation_sampling_plan.npz"
            )
            if threshold_hash != expected_threshold:
                raise ConfirmatoryInternalTestError(
                    f"train-only threshold hash mismatch: {dataset}/{model}"
                )
            if plan_hash != expected_validation_plan:
                raise ConfirmatoryInternalTestError(
                    f"validation conditioning plan hash mismatch: {dataset}/{model}"
                )
            checkpoint_path: str | None = None
            checkpoint_hash: str | None = None
            if model == "empirical_iid":
                state_kind = "hash_bound_frozen_train_pool_no_fit"
                state_path = str(bundle / "train.npz")
                state_hash = checksums["train.npz"]
            else:
                state_kind = "validation_final_checkpoint_read_only"
                checkpoint_hash = _verify_indexed_file(
                    validation_root, index, "checkpoints/final.pt"
                )
                checkpoint_path = str(validation_root / "checkpoints/final.pt")
                state_path = checkpoint_path
                state_hash = checkpoint_hash
            job: dict[str, Any] = {
                "job_id": f"{dataset}/{model}",
                "dataset": dataset,
                "model": model,
                "seed": int(dataset_config["seed"]),
                "sample_seed_offset": int(config["evaluation"]["sample_seed_offset"]),
                "attempt": "attempt_001",
                "attempt_path": str(
                    repository
                    / config["execution_plan"]["artifact_root"]
                    / dataset
                    / model
                    / "attempt_001"
                ),
                "validation_attempt": validation_attempt,
                "validation_root": str(validation_root),
                "validation_terminal_sha256": _sha256_file(terminal_path),
                "validation_artifact_index_sha256": _sha256_file(index_path),
                "validation_manifest_sha256": manifest_hash,
                "train_only_thresholds_path": str(
                    validation_root / "train_bootstrap_thresholds.json"
                ),
                "train_only_thresholds_sha256": threshold_hash,
                "validation_sampling_plan_path": str(
                    validation_root / "validation_sampling_plan.npz"
                ),
                "validation_sampling_plan_sha256": plan_hash,
                "internal_test_path": str(test_path),
                "internal_test_sha256": checksums["internal_test.npz"],
                "train_pool_path": str(bundle / "train.npz"),
                "train_pool_sha256": checksums["train.npz"],
                "transform_path": str(transform_path),
                "transform_sha256": checksums["train_transform_state.json"],
                "summary_path": str(summary_path),
                "summary_sha256": checksums["summary.json"],
                "restore": {
                    "state_kind": state_kind,
                    "state_path": state_path,
                    "state_sha256": state_hash,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_sha256": checkpoint_hash,
                    "training_calls": 0,
                    "refit_calls": 0,
                    "checkpoint_writes": 0,
                },
                "sample_calls": 1,
                "evaluation_calls": 1,
                "fit_calls": 0,
                "refit_calls": 0,
                "test_time_selection_calls": 0,
                "retry_allowed": False,
            }
            job["authorization_binding_sha256"] = _canonical_sha256(job)
            jobs.append(job)
    if len(jobs) != 8 or len({job["job_id"] for job in jobs}) != 8:
        raise ConfirmatoryInternalTestError("confirmatory plan is not exactly eight jobs")
    plan: dict[str, Any] = {
        "schema_version": "external-confirmatory-internal-test-plan-v1",
        "status": "PLAN_VALIDATED_NOT_AUTHORIZED",
        "source_head": source_head or _git_head(repository),
        "config_path": str(config_file),
        "config_sha256": config_hash,
        "config_source_base_head": config.get("source_base_head"),
        "relevant_source_files": source_inventory,
        "relevant_source_sha256": _canonical_sha256(source_inventory),
        "frozen_non_v3_cof_fingerprint": frozen_cof_fingerprint,
        "datasets": datasets,
        "jobs": jobs,
        "job_count": 8,
        "authorization_required": True,
        "execution_authorized": False,
        "test_time_selection": "forbidden",
        "aggregate_policy": "all_eight_terminal_no_partial_exclusion",
        "instrumentation": dict(ZERO_INSTRUMENTATION),
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def validate_confirmatory_authorization(
    plan: Mapping[str, Any], authorization: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    """Validate a future authorization without creating one."""

    if not isinstance(authorization, Mapping):
        raise ConfirmatoryInternalTestError(
            "a separate confirmatory execution authorization is required"
        )
    expected_jobs = [job["job_id"] for job in plan["jobs"]]
    if (
        authorization.get("schema_version")
        != "external-confirmatory-internal-test-authorization-v1"
        or authorization.get("explicit_user_approval") is not True
        or authorization.get("source_commit") != plan["source_head"]
        or authorization.get("config_sha256") != plan["config_sha256"]
        or authorization.get("plan_sha256") != plan["plan_sha256"]
    ):
        raise ConfirmatoryInternalTestError("authorization provenance mismatch")
    if list(authorization.get("allowed_jobs", ())) != expected_jobs:
        raise ConfirmatoryInternalTestError(
            "authorization must contain exactly eight ordered jobs"
        )
    expected_bindings = {
        job["job_id"]: job["authorization_binding_sha256"]
        for job in plan["jobs"]
    }
    if authorization.get("job_provenance") != expected_bindings:
        raise ConfirmatoryInternalTestError("authorization job provenance mismatch")
    conditioning_hashes = authorization.get("conditioning_plan_sha256")
    if not isinstance(conditioning_hashes, Mapping) or set(conditioning_hashes) != set(
        DATASETS
    ):
        raise ConfirmatoryInternalTestError(
            "authorization must bind both internal-test conditioning plans"
        )
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in conditioning_hashes.values()
    ):
        raise ConfirmatoryInternalTestError(
            "authorization conditioning-plan hash is invalid"
        )
    expected_values = {
        "attempt": "attempt_001",
        "sample_calls_per_job": 1,
        "evaluation_calls_per_job": 1,
        "training_calls": 0,
        "refit_calls": 0,
        "retries_allowed": False,
        "sweeps_allowed": False,
        "test_time_selection_allowed": False,
        "sparkov_fraudTest_access_allowed": False,
    }
    for key, expected in expected_values.items():
        if authorization.get(key) != expected:
            raise ConfirmatoryInternalTestError(
                f"authorization scope mismatch: {key}"
            )
    return authorization


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("plan", "dry-run", "execute"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    plan = build_confirmatory_plan(repo_root, args.config)
    if args.mode in {"plan", "dry-run"}:
        output = dict(plan)
        output["mode"] = args.mode
        output["dry_run_checks"] = {
            "exact_eight_jobs": True,
            "checkpoint_or_empirical_state_hash_bound": True,
            "train_transform_and_threshold_hash_bound": True,
            "internal_test_body_loaded": False,
            "models_imported": False,
            "gpu_or_cuda_queried": False,
            "authorization_created": False,
        }
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.authorization is None or args.dataset is None or args.model is None:
        raise ConfirmatoryInternalTestError(
            "execute requires authorization, dataset, and model"
        )
    authorization = _read_json(args.authorization)
    validate_confirmatory_authorization(plan, authorization)
    job_id = f"{args.dataset}/{args.model}"
    job = next((value for value in plan["jobs"] if value["job_id"] == job_id), None)
    if job is None:
        raise ConfirmatoryInternalTestError("requested job is not preregistered")
    # Execution-only dependencies are intentionally unreachable from plan/dry-run.
    from experiments.external_confirmatory_internal_test_runner_v1 import (
        execute_confirmatory_job,
    )

    result = execute_confirmatory_job(
        job=job,
        plan=plan,
        authorization=authorization,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
