from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.run_external_confirmatory_internal_test_v1 import (
    ConfirmatoryInternalTestError,
    build_confirmatory_plan,
    resolve_confirmatory_path,
    validate_confirmatory_authorization,
)


DATASETS = ("amlsim", "sparkov")
MODELS = (
    "empirical_iid",
    "ctgan_separate_class",
    "tvae_separate_class",
    "cof_seqgen_frozen_non_v3",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture_repository(tmp_path: Path) -> tuple[Path, Path]:
    config = {
        "schema_version": "external-confirmatory-internal-test-v1",
        "status": "source_only_not_authorized",
        "source_base_head": "a" * 40,
        "datasets": {},
        "models": list(MODELS),
        "validation_attempt_reuse": {
            dataset: {model: "attempt_001" for model in MODELS}
            for dataset in DATASETS
        },
        "frozen_contract": {
            "model_training_calls": 0,
            "transform_fit_calls": 0,
            "threshold_fit_calls": 0,
            "validation_based_selection_calls": 0,
            "seed_changes_allowed": False,
            "model_or_hyperparameter_changes_allowed": False,
            "evaluator_or_formula_changes_allowed": False,
            "train_only_transform_reuse_required": True,
            "train_only_threshold_reuse_required": True,
            "validation_terminal_and_checkpoint_hash_reuse_required": True,
            "conditioning_plan": {
                "source": "exact_internal_test_y_and_valid_lengths",
                "derivation": "deterministic_order_preserving_no_fit",
                "shared_by_all_four_models": True,
                "hash_before_first_job": "required",
            },
        },
        "evaluation": {
            "split": "internal_test",
            "sample_seed_offset": 1,
            "all_models_reported_without_test_selection": True,
            "partial_or_invalid_result_exclusion": "forbidden",
        },
        "execution_plan": {
            "job_count": 8,
            "attempts": "attempt_001_only",
            "one_execution_per_dataset_model": True,
            "retries": "forbidden",
            "early_stopping": "forbidden",
            "tuning_or_sweep": "forbidden",
            "artifact_root": "artifacts/external_confirmatory_internal_test_v1",
            "append_only": True,
            "authorization_required": True,
        },
    }
    for number, dataset in enumerate(DATASETS, start=1):
        bundle = (
            tmp_path
            / "data/external_sequence_protocol_v1/frozen"
            / dataset
            / "attempt_001"
        )
        bundle.mkdir(parents=True)
        files = {
            "train.npz": f"{dataset}-train".encode(),
            "internal_test.npz": f"{dataset}-test".encode(),
            "train_transform_state.json": b"{}",
            "summary.json": b"{}",
        }
        for name, payload in files.items():
            (bundle / name).write_bytes(payload)
        checksums = {name: _sha(bundle / name) for name in files}
        materialization = (
            tmp_path
            / "artifacts/external_sequence_protocol_v1/materialization"
            / dataset
            / "attempt_001"
        )
        _write_json(
            materialization / "checksum_manifest.json",
            {"dataset": dataset, "data_files": checksums},
        )
        threshold_hash = ""
        sampling_plan_hash = ""
        for model in MODELS:
            attempt = (
                tmp_path
                / "artifacts/external_validation_v1"
                / dataset
                / model
                / "attempt_001"
            )
            attempt.mkdir(parents=True)
            _write_json(
                attempt / "manifest.json",
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": 1000 + number,
                    "data_hashes": {"train": checksums["train.npz"]},
                    "retry_allowed": False,
                    "fit_split": "train",
                    "evaluation_split": "validation",
                    "job_hard_cap_seconds": 100,
                },
            )
            _write_json(
                attempt / "train_bootstrap_thresholds.json",
                {
                    "schema_version": "external-train-bootstrap-thresholds-v1",
                    "source_split": "train",
                },
            )
            (attempt / "validation_sampling_plan.npz").write_bytes(
                f"{dataset}-plan".encode()
            )
            if model != "empirical_iid":
                checkpoint = attempt / "checkpoints/final.pt"
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(f"{dataset}-{model}-checkpoint".encode())
            indexed = {
                "manifest.json": _sha(attempt / "manifest.json"),
                "train_bootstrap_thresholds.json": _sha(
                    attempt / "train_bootstrap_thresholds.json"
                ),
                "validation_sampling_plan.npz": _sha(
                    attempt / "validation_sampling_plan.npz"
                ),
            }
            if model != "empirical_iid":
                indexed["checkpoints/final.pt"] = _sha(
                    attempt / "checkpoints/final.pt"
                )
            _write_json(attempt / "artifact_index.json", indexed)
            _write_json(
                attempt / "COMPLETE.json",
                {"status": "COMPLETE", "dataset": dataset, "model": model},
            )
            threshold_hash = indexed["train_bootstrap_thresholds.json"]
            sampling_plan_hash = indexed["validation_sampling_plan.npz"]
        config["datasets"][dataset] = {
            "seed": 1000 + number,
            "test_input": str(
                Path("data/external_sequence_protocol_v1/frozen")
                / dataset
                / "attempt_001/internal_test.npz"
            ),
            "validation_sampling_plan_sha256": sampling_plan_hash,
            "train_only_thresholds_sha256": threshold_hash,
            (
                "sparkov_fraudTest_access"
                if dataset == "sparkov"
                else "public_external_test"
            ): (
                "forbidden_before_path_resolution"
                if dataset == "sparkov"
                else "not_applicable"
            ),
        }
    config_path = tmp_path / "confirmatory.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return tmp_path, config_path


def test_plan_has_exactly_eight_hash_bound_restore_only_jobs(tmp_path):
    repository, config = _fixture_repository(tmp_path)
    plan = build_confirmatory_plan(repository, config, source_head="b" * 40)

    assert len(plan["jobs"]) == 8
    assert {(job["dataset"], job["model"]) for job in plan["jobs"]} == {
        (dataset, model) for dataset in DATASETS for model in MODELS
    }
    assert all(job["attempt"] == "attempt_001" for job in plan["jobs"])
    assert all(job["sample_calls"] == 1 for job in plan["jobs"])
    assert all(job["evaluation_calls"] == 1 for job in plan["jobs"])
    assert all(job["fit_calls"] == 0 for job in plan["jobs"])
    assert all(job["refit_calls"] == 0 for job in plan["jobs"])
    assert all(job["test_time_selection_calls"] == 0 for job in plan["jobs"])
    iid = [job for job in plan["jobs"] if job["model"] == "empirical_iid"]
    assert {job["restore"]["state_kind"] for job in iid} == {
        "hash_bound_frozen_train_pool_no_fit"
    }
    assert all(job["restore"]["checkpoint_path"] is None for job in iid)
    learned = [job for job in plan["jobs"] if job["model"] != "empirical_iid"]
    assert all(job["restore"]["checkpoint_sha256"] for job in learned)
    assert plan["instrumentation"] == {
        "gpu_inventory_queries": 0,
        "cuda_calls": 0,
        "model_import_calls": 0,
        "model_fit_calls": 0,
        "model_sample_calls": 0,
        "internal_test_npz_body_reads": 0,
        "evaluation_calls": 0,
        "authorization_artifacts_created": 0,
    }


def test_plan_fails_closed_on_checkpoint_or_threshold_hash_mismatch(tmp_path):
    repository, config = _fixture_repository(tmp_path)
    checkpoint = (
        repository
        / "artifacts/external_validation_v1/amlsim/ctgan_separate_class/attempt_001"
        / "checkpoints/final.pt"
    )
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(ConfirmatoryInternalTestError, match="artifact index"):
        build_confirmatory_plan(repository, config, source_head="b" * 40)

    repository, config = _fixture_repository(tmp_path / "threshold")
    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    loaded["datasets"]["amlsim"]["train_only_thresholds_sha256"] = "0" * 64
    config.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfirmatoryInternalTestError, match="threshold"):
        build_confirmatory_plan(repository, config, source_head="b" * 40)


def test_sparkov_fraudtest_is_rejected_before_path_resolution():
    calls = []

    def resolver(path):
        calls.append(path)
        return Path(path)

    with pytest.raises(ConfirmatoryInternalTestError, match="fraudTest"):
        resolve_confirmatory_path(
            "data/sparkov/fraudTest.csv", resolver=resolver
        )
    assert calls == []


def test_authorization_is_exact_and_authorizationless_execution_fails(tmp_path):
    repository, config = _fixture_repository(tmp_path)
    plan = build_confirmatory_plan(repository, config, source_head="b" * 40)

    with pytest.raises(ConfirmatoryInternalTestError, match="authorization"):
        validate_confirmatory_authorization(plan, None)

    authorization = {
        "schema_version": "external-confirmatory-internal-test-authorization-v1",
        "explicit_user_approval": True,
        "source_commit": plan["source_head"],
        "config_sha256": plan["config_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "allowed_jobs": [job["job_id"] for job in plan["jobs"]],
        "attempt": "attempt_001",
        "sample_calls_per_job": 1,
        "evaluation_calls_per_job": 1,
        "training_calls": 0,
        "refit_calls": 0,
        "retries_allowed": False,
        "sweeps_allowed": False,
        "test_time_selection_allowed": False,
        "sparkov_fraudTest_access_allowed": False,
        "job_provenance": {
            job["job_id"]: job["authorization_binding_sha256"]
            for job in plan["jobs"]
        },
        "conditioning_plan_sha256": {
            "amlsim": "1" * 64,
            "sparkov": "2" * 64,
        },
    }
    assert validate_confirmatory_authorization(plan, authorization)[
        "explicit_user_approval"
    ]
    broken = dict(authorization)
    broken["allowed_jobs"] = broken["allowed_jobs"][:-1]
    with pytest.raises(ConfirmatoryInternalTestError, match="exactly eight"):
        validate_confirmatory_authorization(plan, broken)


def test_core_executes_one_restore_sample_evaluation_and_never_fits(tmp_path):
    from experiments.external_confirmatory_internal_test_runner_v1 import (
        ConfirmatoryExecutionHooks,
        execute_confirmatory_job_core,
    )

    calls = {"load": 0, "restore": 0, "fit": 0, "sample": 0, "evaluate": 0}

    def load(_job):
        calls["load"] += 1
        return "internal-test-batch", "shared-plan"

    def restore(_job, _device):
        calls["restore"] += 1
        return "frozen-state"

    def sample(state, plan, seed):
        calls["sample"] += 1
        assert (state, plan, seed) == ("frozen-state", "shared-plan", 1002)
        return "synthetic"

    def evaluate(real, synthetic, _job):
        calls["evaluate"] += 1
        assert (real, synthetic) == ("internal-test-batch", "synthetic")
        return {"hard_validity": {"mask": True}, "metrics": {"fidelity": {}}}

    hooks = ConfirmatoryExecutionHooks(
        load_internal_test=load,
        restore_state=restore,
        sample=sample,
        evaluate=evaluate,
    )
    job = {
        "job_id": "amlsim/empirical_iid",
        "dataset": "amlsim",
        "model": "empirical_iid",
        "seed": 1001,
        "sample_seed_offset": 1,
        "attempt": "attempt_001",
        "attempt_path": str(tmp_path / "attempt_001"),
        "sample_calls": 1,
        "evaluation_calls": 1,
        "fit_calls": 0,
        "refit_calls": 0,
        "test_time_selection_calls": 0,
    }
    result = execute_confirmatory_job_core(job=job, device="cpu", hooks=hooks)
    assert result["status"] == "COMPLETE"
    assert calls == {"load": 1, "restore": 1, "fit": 0, "sample": 1, "evaluate": 1}
    assert (tmp_path / "attempt_001/COMPLETE.json").is_file()
    with pytest.raises(ConfirmatoryInternalTestError, match="already exists"):
        execute_confirmatory_job_core(job=job, device="cpu", hooks=hooks)
    assert calls["sample"] == 1


def test_execution_source_has_no_fit_selection_retry_or_checkpoint_write_path():
    source = (
        Path(__file__).parents[1]
        / "experiments/external_confirmatory_internal_test_runner_v1.py"
    ).read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "select_external_candidate" not in source
    assert "save_training_checkpoint" not in source
    assert "attempt_002" not in source
