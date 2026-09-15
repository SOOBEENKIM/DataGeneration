import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml

from scripts.prepare_external_validation_authorization_v1 import (
    build_authorization_preparation_plan,
    build_external_validation_launch_argv,
    build_external_validation_wave_launch_plan,
    main as prepare_authorization_main,
)

from scripts.run_external_validation_v1 import (
    ExternalValidationError,
    audit_frozen_leakage_manifests,
    build_external_adapter_contract,
    build_external_validation_batch_authorization,
    build_external_execution_manifest,
    build_external_validation_plan,
    claim_external_validation_attempt,
    main,
    resolve_external_bundle_access,
    select_batch_authorized_job,
    validate_external_input_and_frozen_cof_contract,
    validate_external_continuation_preservation,
    validate_external_launch_schedule,
    validate_external_wave_barrier,
    validate_external_validation_authorization,
    validate_external_validation_batch_authorization,
    write_external_validation_authorization,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/benchmark_v2/external_validation_v1.yaml"


def _config_with_isolated_runtime(tmp_path: Path) -> Path:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["runtime"]["root"] = str(tmp_path / "external_validation_runtime")
    isolated = tmp_path / "external_validation_v1.yaml"
    isolated.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return isolated


def test_sparkov_cof_launch_argv_keeps_model_flag_and_id_separate():
    argv = build_external_validation_launch_argv(
        python_executable="/opt/cofseq/bin/python3",
        authorization_path=Path("authorization_replacement.json"),
        dataset="sparkov",
        model="cof_seqgen_frozen_non_v3",
        device="cuda:0",
    )

    model_flag_index = argv.index("--model")
    assert argv[model_flag_index + 1] == "cof_seqgen_frozen_non_v3"
    assert "--modelcof_seqgen_frozen_non_v3" not in argv


def test_plan_and_dry_run_verify_frozen_bundles_without_data_or_model_execution():
    modules_before = set(sys.modules)
    plan = build_external_validation_plan(
        repo_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
        mode="plan",
    )
    dry_run = build_external_validation_plan(
        repo_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
        mode="dry-run",
    )

    assert set(plan["datasets"]) == {"amlsim", "sparkov"}
    assert plan["models"] == [
        "empirical_iid",
        "ctgan_separate_class",
        "tvae_separate_class",
        "cof_seqgen_frozen_non_v3",
    ]
    assert plan["job_count"] == 8
    assert plan["runtime_artifacts_created"] is False
    assert plan["execution_authorized"] is False
    assert all(value == 0 for value in plan["execution_counts"].values())
    assert all(value == 0 for value in dry_run["execution_counts"].values())
    newly_imported = set(sys.modules) - modules_before
    assert "experiments.external_validation_runner_v1" not in newly_imported
    assert "generators.conditional_ctgan" not in newly_imported
    assert "generators.conditional_tvae" not in newly_imported
    assert "generators.cof_seqgen_adapter" not in newly_imported
    assert dry_run["access_counts"] == {
        "manifest_json_reads": 16,
        "bundle_hash_reads": 6,
        "frozen_body_hash_reads": 0,
        "npz_load_calls": 0,
        "raw_csv_hash_reads": 0,
        "raw_csv_header_reads": 0,
        "raw_csv_body_reads": 0,
        "internal_test_npz_load_calls": 0,
        "sparkov_fraud_test_accesses": 0,
    }
    assert dry_run["bundles"]["amlsim"]["status"] == "PASS"
    assert dry_run["bundles"]["sparkov"]["status"] == "PASS"
    assert dry_run["bundles"]["amlsim"]["leakage_audit"]["status"] == "PASS"
    assert dry_run["bundles"]["sparkov"]["leakage_audit"]["status"] == "PASS"
    assert dry_run["frozen_cof_contracts"]["amlsim"]["status"] == "PASS"
    assert dry_run["frozen_cof_contracts"]["sparkov"]["status"] == "PASS"
    assert dry_run["adapter_contracts"]["amlsim"]["model_import_calls"] == 0
    assert dry_run["bundles"]["amlsim"]["tree_sha256"] == (
        "cc8f4c9c6ff415ccd4f78ec193122100145e10084751b971dcc0ce5e705fa024"
    )
    assert dry_run["bundles"]["sparkov"]["tree_sha256"] == (
        "62fefda6207911104a9bff0ebb41d5edf44c20a4c78dccef5875d9768be6addc"
    )
    for dataset, dataset_plan in plan["datasets"].items():
        assert dataset_plan["internal_test_access"] == "fail_closed"
        assert dataset_plan["sparkov_fraud_test_access"] == "fail_closed"
        for model, job in dataset_plan["jobs"].items():
            assert dataset in job["attempt_path"]
            assert model in job["attempt_path"]
            expected_attempt = (
                "attempt_002" if model == "empirical_iid" else "attempt_001"
            )
            assert job["attempt"] == expected_attempt
            assert job["attempt_path"].endswith(expected_attempt)
            assert job["attempt_exists"] is Path(job["attempt_path"]).exists()
    assert plan["all_target_attempts_absent"] is all(
        not job["attempt_exists"]
        for dataset in plan["datasets"].values()
        for job in dataset["jobs"].values()
    )
    preservation = plan["continuation_preservation"]
    assert preservation["status"] == "PASS"
    assert preservation["artifacts"]["amlsim_empirical_iid_attempt_001"][
        "tree_sha256"
    ] == "e0d2bf55ca9eaf6404294168a309b66d0c1c34ce63c247f15af853a4b37e7810"
    assert preservation["artifacts"]["sparkov_empirical_iid_attempt_001"][
        "tree_sha256"
    ] == "64abdd9ec13d52b25f8f9f79de15e6fd95d81bb321058a90d42b77e10b4a8592"
    assert preservation["artifacts"]["prior_authorization_history"][
        "tree_sha256"
    ] == "281372d002df2af1c42dc941410cb25348ab875d078da396c77c2c867e4b7246"
    assert preservation["artifacts"]["prior_continuation_authorization_history"][
        "tree_sha256"
    ] == "22d82f8512ee1c7acab130947a360f938710417130891cd114408a4bdd99a824"
    assert preservation["artifacts"]["prior_wave_scheduled_authorization_history"][
        "tree_sha256"
    ] == "75f066a680681ad5a917b225c99f6ff02bd1f40eea7fc432089c355982d841c1"


def test_launch_schedule_separates_ctgan_and_tvae_heavy_transforms():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    schedule = validate_external_launch_schedule(config)

    assert schedule["schema_version"] == (
        "external-validation-memory-safe-five-wave-schedule-v1"
    )
    assert schedule["scheduling_only"] is True
    assert [wave["wave_id"] for wave in schedule["waves"]] == [
        "cpu_wave",
        "amlsim_gpu_wave_1",
        "amlsim_gpu_wave_2",
        "sparkov_gpu_wave_1",
        "sparkov_gpu_wave_2",
    ]
    assert [len(wave["jobs"]) for wave in schedule["waves"]] == [2, 2, 1, 2, 1]
    assert all(wave["parallel"] is True for wave in schedule["waves"])
    assert all(wave["terminal_barrier_after"] is True for wave in schedule["waves"])

    cpu_wave, amlsim_wave_1, amlsim_wave_2, sparkov_wave_1, sparkov_wave_2 = schedule[
        "waves"
    ]
    assert {(job["dataset"], job["model"]) for job in cpu_wave["jobs"]} == {
        ("amlsim", "empirical_iid"),
        ("sparkov", "empirical_iid"),
    }
    for wave, dataset in (
        (amlsim_wave_1, "amlsim"),
        (amlsim_wave_2, "amlsim"),
        (sparkov_wave_1, "sparkov"),
        (sparkov_wave_2, "sparkov"),
    ):
        assert {job["dataset"] for job in wave["jobs"]} == {dataset}
        assert all(job["runner_device"] == "cuda:0" for job in wave["jobs"])
        assert all(job["cuda_visible_devices_count"] == 1 for job in wave["jobs"])
        heavy = {
            job["model"]
            for job in wave["jobs"]
            if job["model"] in {"ctgan_separate_class", "tvae_separate_class"}
        }
        assert len(heavy) <= 1


def test_wave_launch_plan_binds_one_visible_gpu_per_worker_and_never_executes():
    result = build_external_validation_wave_launch_plan(
        python_executable="/opt/cofseq/bin/python3",
        authorization_path=Path("scheduled_authorization.json"),
        config_path=CONFIG_PATH,
    )

    assert [len(wave["jobs"]) for wave in result["waves"]] == [2, 2, 1, 2, 1]
    assert all(wave["terminal_barrier_after"] for wave in result["waves"])
    for wave in result["waves"]:
        for job in wave["jobs"]:
            device_index = job["argv"].index("--device")
            assert job["argv"][device_index + 1] == job["runner_device"]
            if wave["device_class"] == "gpu":
                assert job["cuda_visible_devices_env"] in {
                    "EXTV1_GPU_A",
                    "EXTV1_GPU_B",
                    "EXTV1_GPU_C",
                }
                assert job["runner_device"] == "cuda:0"
            else:
                assert job["cuda_visible_devices_env"] is None
                assert job["runner_device"] == "cpu"
    assert all(value == 0 for value in result["execution_counts"].values())


def test_execution_wave_barrier_is_fail_closed_until_all_prior_jobs_terminal(tmp_path):
    config_path = _config_with_isolated_runtime(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runtime_root = Path(config["runtime"]["root"])

    assert validate_external_wave_barrier(
        repo_root=REPOSITORY_ROOT,
        config_path=config_path,
        dataset="amlsim",
        model="empirical_iid",
    )["prior_terminal_count"] == 0
    with pytest.raises(ExternalValidationError, match="terminal barrier"):
        validate_external_wave_barrier(
            repo_root=REPOSITORY_ROOT,
            config_path=config_path,
            dataset="amlsim",
            model="ctgan_separate_class",
        )

    for dataset in ("amlsim", "sparkov"):
        path = runtime_root / dataset / "empirical_iid" / "attempt_002"
        path.mkdir(parents=True)
        (path / "COMPLETE.json").write_text("{}\n", encoding="utf-8")
    wave_1 = validate_external_wave_barrier(
        repo_root=REPOSITORY_ROOT,
        config_path=config_path,
        dataset="amlsim",
        model="ctgan_separate_class",
    )
    assert wave_1["prior_terminal_count"] == 2

    with pytest.raises(ExternalValidationError, match="terminal barrier"):
        validate_external_wave_barrier(
            repo_root=REPOSITORY_ROOT,
            config_path=config_path,
            dataset="sparkov",
            model="ctgan_separate_class",
        )
    for model in (
        "ctgan_separate_class",
        "tvae_separate_class",
        "cof_seqgen_frozen_non_v3",
    ):
        path = runtime_root / "amlsim" / model / "attempt_001"
        path.mkdir(parents=True)
        (path / "INVALID.json").write_text("{}\n", encoding="utf-8")
    wave_2 = validate_external_wave_barrier(
        repo_root=REPOSITORY_ROOT,
        config_path=config_path,
        dataset="sparkov",
        model="ctgan_separate_class",
    )
    assert wave_2["prior_terminal_count"] == 5


def test_bundle_access_is_train_fit_and_validation_evaluation_only():
    train = resolve_external_bundle_access(
        repo_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
        dataset="amlsim",
        split="train",
        purpose="model_fit",
    )
    validation = resolve_external_bundle_access(
        repo_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
        dataset="sparkov",
        split="validation",
        purpose="candidate_generation_and_evaluation",
    )
    assert train.name == "train.npz"
    assert validation.name == "validation.npz"

    with pytest.raises(ExternalValidationError, match="validation cannot fit"):
        resolve_external_bundle_access(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            dataset="amlsim",
            split="validation",
            purpose="model_fit",
        )
    with pytest.raises(ExternalValidationError, match="internal_test is locked"):
        resolve_external_bundle_access(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            dataset="sparkov",
            split="internal_test",
            purpose="candidate_generation_and_evaluation",
        )


def test_frozen_entity_window_and_transaction_leakage_evidence_is_revalidated():
    for dataset, expected_windows in (("amlsim", 43251), ("sparkov", 40513)):
        audit = audit_frozen_leakage_manifests(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            dataset=dataset,
        )
        assert audit["status"] == "PASS"
        assert audit["entity_overlap_count"] == 0
        assert audit["window_overlap_count"] == 0
        assert audit["transaction_overlap_count"] == 0
        assert audit["window_count"] == expected_windows
        assert audit["duplicate_window_ids"] == 0
        assert audit["duplicate_window_membership_hashes"] == 0
        assert audit["window_entities_missing_from_split"] == 0
        assert audit["transaction_evidence"] == (
            "hash_bound_materializer_independent_audit"
        )


def test_pad_unk_dynamic_receiver_cardinality_and_frozen_cof_fingerprint_contract():
    expected = {"amlsim": 9656, "sparkov": 695}
    for dataset, cardinality in expected.items():
        contract = validate_external_input_and_frozen_cof_contract(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            dataset=dataset,
        )
        assert contract["status"] == "PASS"
        assert contract["pad_code"] == 0
        assert contract["unk_code"] == 1
        assert contract["receiver_vocabulary_cardinality"] == cardinality
        assert contract["receiver_cardinality_role"] == (
            "train_only_input_vocabulary_not_architecture_change"
        )
        assert contract["amount_input_width"] == 1
        assert contract["gap_bin_cardinality"] == 16
        assert contract["sequence_length"] == 32
        assert contract["architecture_change_required"] is False
        assert contract["frozen_source_commit"] == (
            "99a445f6dc893a8c2240d950de4f92877cc07f8a"
        )
        assert contract["frozen_config_sha256"] == (
            "81bc16416f6dad3c23eaf1990c3cb76e5449e74f1ddc634f42d3a897cd3c5bb3"
        )
        assert contract["model_source_sha256"] == (
            "caf5fdb3baf367a28c6081a8ba5f4a89e59dacbba2377636acba5c051ae7006e"
        )
        assert contract["adapter_source_sha256"] == (
            "48e9e1280bc23a27abf0b5f6447f6682e3b846d0d1773fb96d99abcf407dd7f3"
        )


def test_append_only_attempt_claim_has_exclusive_dataset_model_ownership(tmp_path):
    amlsim_attempt = (
        tmp_path / "external_validation_v1/amlsim/cof_seqgen_frozen_non_v3/attempt_001"
    )
    sparkov_attempt = (
        tmp_path / "external_validation_v1/sparkov/cof_seqgen_frozen_non_v3/attempt_001"
    )
    manifest = {
        "dataset": "amlsim",
        "model": "cof_seqgen_frozen_non_v3",
        "source_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "train_sha256": "c" * 64,
        "validation_sha256": "d" * 64,
    }
    first = claim_external_validation_attempt(amlsim_attempt, manifest)
    assert first["status"] == "RUNNING"
    assert json.loads((amlsim_attempt / "OWNERSHIP.json").read_text())["manifest_sha256"]
    tree_before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in amlsim_attempt.iterdir()
    }
    with pytest.raises(ExternalValidationError, match="already exists"):
        claim_external_validation_attempt(amlsim_attempt, manifest)
    tree_after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in amlsim_attempt.iterdir()
    }
    assert tree_after == tree_before

    sparkov_manifest = {**manifest, "dataset": "sparkov"}
    second = claim_external_validation_attempt(sparkov_attempt, sparkov_manifest)
    assert second["status"] == "RUNNING"
    assert amlsim_attempt != sparkov_attempt


def test_adapter_plan_reuses_frozen_budgets_and_only_binds_external_input_state():
    contracts = build_external_adapter_contract(
        repo_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
        dataset="amlsim",
    )

    assert contracts["model_import_calls"] == 0
    assert contracts["frozen_config_mutated"] is False
    assert contracts["empirical_iid"]["requested_steps"] == 0
    assert contracts["ctgan_separate_class"]["requested_steps"] == 10000
    assert contracts["ctgan_separate_class"]["class_wall_seconds"] == {
        "0": 3600,
        "1": 3600,
    }
    assert contracts["tvae_separate_class"]["requested_steps"] == 20000
    cof = contracts["cof_seqgen_frozen_non_v3"]
    assert cof["requested_steps"] == 20000
    assert cof["architecture_override"] == {}
    assert cof["runtime_data_bindings"] == {
        "tau": "train_transform_state.gap_tau",
        "receiver_cardinality": 9656,
        "receiver_cardinality_role": "input_vocabulary_only",
        "amount_input_width": 1,
        "gap_bin_cardinality": 16,
        "sequence_length": 32,
    }


def test_execution_authorization_is_one_dataset_model_and_hash_bound(tmp_path):
    config_path = _config_with_isolated_runtime(tmp_path)
    batch = build_external_validation_batch_authorization(
        repo_root=REPOSITORY_ROOT,
        config_path=config_path,
        approval_text="explicit performance-corrective continuation approval",
    )
    authorization = select_batch_authorized_job(
        repo_root=REPOSITORY_ROOT,
        config_path=config_path,
        authorization=batch,
        dataset="amlsim",
        model="cof_seqgen_frozen_non_v3",
    )
    validated = validate_external_validation_authorization(
        repo_root=REPOSITORY_ROOT,
        config_path=config_path,
        authorization=authorization,
    )
    assert validated["status"] == "PASS"
    assert validated["dataset"] == "amlsim"
    assert validated["model"] == "cof_seqgen_frozen_non_v3"
    assert validated["npz_load_calls"] == 0
    assert validated["model_import_calls"] == 0

    manifest = build_external_execution_manifest(
        repo_root=REPOSITORY_ROOT,
        config_path=config_path,
        authorization=authorization,
    )
    assert manifest["dataset"] == "amlsim"
    assert manifest["model"] == "cof_seqgen_frozen_non_v3"
    assert manifest["seed"] == 31001
    assert set(manifest["data_hashes"]) == {"train", "validation"}
    assert "internal_test" not in json.dumps(manifest)
    assert manifest["fit_split"] == "train"
    assert manifest["evaluation_split"] == "validation"
    assert manifest["attempt"] == "attempt_001"
    assert manifest["metric_source_sha256"] == (
        "e91de63a11cf34e537738cf0fbbfe3c76bbde9c1a383b26c2cae963d5fbcd933"
    )
    assert manifest["runtime_artifacts_created"] is False

    authorization["validation_sha256"] = "0" * 64
    with pytest.raises(ExternalValidationError, match="authorization provenance"):
        validate_external_validation_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=config_path,
            authorization=authorization,
        )
    with pytest.raises(ExternalValidationError, match="fraudTest is locked"):
        resolve_external_bundle_access(
            repo_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
            dataset="sparkov",
            split="fraudTest",
            purpose="candidate_generation_and_evaluation",
        )


def test_execute_without_separate_authorization_fails_before_runtime_or_models(tmp_path):
    with pytest.raises(ExternalValidationError, match="requires a separate authorization"):
        main(
            [
                "--repo-root",
                str(REPOSITORY_ROOT),
                "--config",
                str(CONFIG_PATH),
                "--mode",
                "execute",
                "--dataset",
                "amlsim",
                "--model",
                "empirical_iid",
                "--device",
                "cpu",
            ]
        )
    assert not (tmp_path / "artifacts").exists()


def test_execution_source_freezes_train_fit_before_validation_array_load():
    source = (
        REPOSITORY_ROOT / "experiments/external_validation_runner_v1.py"
    ).read_text(encoding="utf-8")
    train_load = source.index("train = _load_batch(train_path)")
    fit = source.index("adapter.fit(train")
    validation_load = source.index("validation = _load_batch(validation_path)")
    assert train_load < fit < validation_load
    assert 'split="internal_test"' not in source
    assert "fraudTest" not in source


def test_batch_authorization_is_exactly_eight_hash_bound_jobs_and_append_only(tmp_path):
    config_path = _config_with_isolated_runtime(tmp_path)
    authorization = build_external_validation_batch_authorization(
        repo_root=REPOSITORY_ROOT,
        config_path=config_path,
        approval_text="explicit external validation v1 approval",
    )
    validated = validate_external_validation_batch_authorization(
        repo_root=REPOSITORY_ROOT,
        config_path=config_path,
        authorization=authorization,
    )

    assert validated["status"] == "PASS"
    assert validated["job_count"] == 8
    assert validated["execution_counts"] == {
        "npz_load_calls": 0,
        "raw_csv_accesses": 0,
        "model_import_calls": 0,
        "gpu_inventory_queries": 0,
    }
    expected = {
        (dataset, model)
        for dataset in ("amlsim", "sparkov")
        for model in (
            "empirical_iid",
            "ctgan_separate_class",
            "tvae_separate_class",
            "cof_seqgen_frozen_non_v3",
        )
    }
    assert {(job["dataset"], job["model"]) for job in authorization["jobs"]} == expected
    assert authorization["schema_version"] == (
        "external-validation-performance-continuation-authorization-v1"
    )
    assert authorization["scientific_definition_unchanged"] is True
    assert authorization["seed_conditioning_selection_unchanged"] is True
    assert authorization["job_cap_changed"] is False
    assert [
        (wave["wave_id"], len(wave["jobs"]))
        for wave in authorization["launch_schedule"]["waves"]
    ] == [
        ("cpu_wave", 2),
        ("amlsim_gpu_wave_1", 2),
        ("amlsim_gpu_wave_2", 1),
        ("sparkov_gpu_wave_1", 2),
        ("sparkov_gpu_wave_2", 1),
    ]
    for job in authorization["jobs"]:
        assert job["attempt"] == (
            "attempt_002" if job["model"] == "empirical_iid" else "attempt_001"
        )
        assert job["retry_allowed"] is False
        assert job["early_stopping_allowed"] is False
        assert job["sweep_allowed"] is False
        if job["model"] == "empirical_iid":
            assert job["wave_id"] == "cpu_wave"
            assert job["runner_device"] == "cpu"
            assert job["cuda_visible_devices_count"] == 0
            assert job["training_hard_cap_seconds"] == 0
            assert job["job_hard_cap_seconds"] == 7200
            assert job["expected_gpu_hours_upper"] == 0.0
        else:
            assert job["wave_id"] == (
                f"{job['dataset']}_gpu_wave_2"
                if job["model"] == "tvae_separate_class"
                else f"{job['dataset']}_gpu_wave_1"
            )
            assert job["runner_device"] == "cuda:0"
            assert job["cuda_visible_devices_count"] == 1
            if job["model"] in {"ctgan_separate_class", "tvae_separate_class"}:
                assert job["data_transformer_n_jobs"] == 1
                assert job["heavy_transform_exclusion_group"] == (
                    "external_tabular_transform"
                )
            assert job["training_hard_cap_seconds"] == 7200
            assert job["job_hard_cap_seconds"] == 10800
            assert job["expected_gpu_hours_upper"] == 2.0

    selected = select_batch_authorized_job(
        repo_root=REPOSITORY_ROOT,
        config_path=config_path,
        authorization=authorization,
        dataset="sparkov",
        model="tvae_separate_class",
    )
    assert selected["dataset"] == "sparkov"
    assert selected["model"] == "tvae_separate_class"
    assert len(selected["parent_authorization_sha256"]) == 64

    sparkov_iid = select_batch_authorized_job(
        repo_root=REPOSITORY_ROOT,
        config_path=config_path,
        authorization=authorization,
        dataset="sparkov",
        model="empirical_iid",
    )
    assert sparkov_iid["attempt"] == "attempt_002"
    iid_manifest = build_external_execution_manifest(
        repo_root=REPOSITORY_ROOT,
        config_path=config_path,
        authorization=sparkov_iid,
    )
    assert iid_manifest["attempt"] == "attempt_002"
    assert iid_manifest["attempt_path"].endswith(
        "sparkov/empirical_iid/attempt_002"
    )
    assert iid_manifest["job_hard_cap_seconds"] == 7200

    preservation = validate_external_continuation_preservation(
        repo_root=REPOSITORY_ROOT,
        config=yaml.safe_load(config_path.read_text(encoding="utf-8")),
    )
    corrupted = {
        **authorization,
        "continuation_preservation": {
            **preservation,
            "preservation_sha256": "0" * 64,
        },
    }
    with pytest.raises(ExternalValidationError, match="preservation mismatch"):
        validate_external_validation_batch_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=config_path,
            authorization=corrupted,
        )

    path = tmp_path / "authorization_attempt_001.json"
    written = write_external_validation_authorization(path, authorization)
    assert written == path
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_external_validation_authorization(path, authorization)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before

    incomplete = {**authorization, "jobs": authorization["jobs"][:-1]}
    with pytest.raises(ExternalValidationError, match="exactly eight"):
        validate_external_validation_batch_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=config_path,
            authorization=incomplete,
        )

    wrong_schedule = {
        **authorization,
        "launch_schedule": {
            **authorization["launch_schedule"],
            "waves": authorization["launch_schedule"]["waves"][:2],
        },
    }
    with pytest.raises(ExternalValidationError, match="launch schedule"):
        validate_external_validation_batch_authorization(
            repo_root=REPOSITORY_ROOT,
            config_path=config_path,
            authorization=wrong_schedule,
        )


def test_authorization_plan_and_dry_run_are_manifest_only_and_create_nothing(tmp_path):
    config_path = _config_with_isolated_runtime(tmp_path)
    for mode in ("plan", "dry-run"):
        result = build_authorization_preparation_plan(
            repo_root=REPOSITORY_ROOT,
            config_path=config_path,
            approval_text="explicit external validation v1 approval",
            mode=mode,
        )
        assert result["status"] == "PASS"
        assert result["job_count"] == 8
        assert result["authorization_created"] is False
        assert all(value == 0 for value in result["access_counts"].values())
        assert all(value == 0 for value in result["execution_counts"].values())
        assert len(result["jobs"]) == 8
        assert result["authorization_root"].endswith(
            "artifacts/external_validation_v1/"
            "memory_corrective_authorization_history"
        )
        assert [
            len(wave["jobs"]) for wave in result["launch_schedule"]["waves"]
        ] == [2, 2, 1, 2, 1]
        assert result["memory_safety"]["data_transformer"] == {
            "models": ["ctgan_separate_class", "tvae_separate_class"],
            "fixed_n_jobs": 1,
            "unbounded_n_jobs_forbidden": True,
            "execution_mode": "synchronous_column_transform",
            "max_concurrent_heavy_transforms": 1,
            "algorithm_seed_conditioning_metrics_unchanged": True,
        }
        assert result["completed_result_reuse"]["ctgan_separate_class"][
            "status"
        ] == "REUSE_ELIGIBLE"

    outside = tmp_path / "not_the_authorized_history" / "authorization.json"
    with pytest.raises(ExternalValidationError, match="outside its append-only root"):
        prepare_authorization_main(
            [
                "--repo-root",
                str(REPOSITORY_ROOT),
                "--config",
                str(config_path),
                "--mode",
                "create",
                "--approval-text",
                "explicit performance-corrective continuation approval",
                "--output",
                str(outside),
            ]
        )
    assert not outside.exists()


def test_execute_cli_uses_runner_owned_whole_job_watchdog():
    runner_source = (
        REPOSITORY_ROOT / "scripts/run_external_validation_v1.py"
    ).read_text(encoding="utf-8")
    execution_source = (
        REPOSITORY_ROOT / "experiments/external_validation_runner_v1.py"
    ).read_text(encoding="utf-8")
    assert "execute_external_validation_job_bounded" in runner_source
    assert 'authorization.get("job_hard_cap_seconds", 0)' in execution_source
    assert "process.join(max_wall_seconds)" in execution_source
    assert "process.terminate()" in execution_source
    assert "process.kill()" in execution_source
    assert '"failure_class": "wall_cap"' in execution_source
    assert "require_all_attempts_absent=False" in runner_source
