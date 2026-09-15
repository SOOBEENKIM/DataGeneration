import json
from pathlib import Path
import time

import numpy as np
import pytest
import yaml

from benchmarks.types import SequenceBatch
from eval.validation_selection_v2_6 import read_yaml_mapping, sha256_file
from experiments.candidate_runner_v2_6 import (
    CandidateRunnerContractError,
    build_model_continuation_plan,
    build_checkpoint_bundle,
    build_candidate_result_manifest,
    build_model_trajectory_plan,
    build_trajectory_plan,
    claim_model_continuation,
    claim_next_attempt,
    claim_model_worker,
    dispatch_continuation_plan,
    finalize_model_worker,
    load_train_only_candidate_context,
    preserved_attempt_tree_sha256,
    run_bounded_process,
    validate_candidate_continuation_authorization,
    validate_candidate_aggregate_readiness,
    validate_candidate_io_path,
    write_selection_artifact_bundle,
)
from generators.candidate_adapters_v2_6 import (
    CandidateAdapterContractError,
    TrainOnlyZScore,
)
from experiments.provenance_v2_5 import hash_batch
from scripts.run_candidate_training_v2_6 import parse_args
from scripts.aggregate_candidate_selection_v2_6 import (
    parse_args as parse_aggregate_args,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def train_fixture() -> SequenceBatch:
    lengths = np.asarray([2, 1, 2, 1], dtype=np.int64)
    valid = np.arange(2)[None, :] < lengths[:, None]
    amount = np.zeros((4, 2, 1), dtype=np.float32)
    amount[..., 0][valid] = np.asarray(
        [1, 2, 3, 4, 5, 6],
        dtype=np.float32,
    )
    gap = np.zeros((4, 2), dtype=np.int64)
    gap[valid] = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int64)
    receiver = np.zeros((4, 2, 1), dtype=np.int64)
    receiver[..., 0][valid] = np.asarray(
        [0, 1, 2, 1, 0, 2],
        dtype=np.int64,
    )
    return SequenceBatch(
        x_num=amount,
        dt_bin=gap,
        x_cat=receiver,
        valid_mask=valid,
        y_entity=np.asarray([0, 0, 1, 1], dtype=np.int64),
        lengths=lengths,
        entity_ids=np.asarray(["a", "b", "c", "d"]),
    )


def save_batch(path: Path, batch: SequenceBatch) -> None:
    np.savez_compressed(
        path,
        **{
            field: getattr(batch, field)
            for field in SequenceBatch.__dataclass_fields__
        },
    )


def hang_fixture() -> None:
    time.sleep(10)


def no_op_fixture() -> None:
    return None


def invalid_candidate_fixture() -> None:
    raise CandidateAdapterContractError("fixture invalid candidate")


def infrastructure_failure_fixture() -> None:
    raise RuntimeError("fixture infrastructure failure")


def test_real_config_builds_12_trajectories_and_16_evaluation_candidates():
    config = read_yaml_mapping(
        REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
    )

    plan = build_trajectory_plan(config)

    assert len(plan.trajectories) == 12
    assert sum(len(item.candidates) for item in plan.trajectories) == 16
    for model_id in config["model_order"]:
        model_trajectories = [
            item for item in plan.trajectories if item.model_id == model_id
        ]
        assert len(model_trajectories) == 3
        native = [
            item
            for item in model_trajectories
            if {candidate.candidate_id for candidate in item.candidates}
            == {
                "c00_native_checkpoint_10000",
                "c01_native_checkpoint_20000",
            }
        ]
        assert len(native) == 1
        assert native[0].requested_updates == 20_000
        assert native[0].max_wall_seconds == 7_200


@pytest.mark.parametrize(
    "model_id",
    [
        "ctgan_separate_class",
        "tvae_separate_class",
        "neural_sequence",
        "cof_seqgen",
    ],
)
def test_model_worker_plan_owns_exactly_three_trajectories_and_four_candidates(
    model_id,
):
    config = read_yaml_mapping(
        REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
    )

    plan = build_model_trajectory_plan(config, model_id=model_id)

    assert len(plan.trajectories) == 3
    assert {item.model_id for item in plan.trajectories} == {model_id}
    assert sum(len(item.candidates) for item in plan.trajectories) == 4
    native = next(
        item
        for item in plan.trajectories
        if {
            candidate.candidate_id for candidate in item.candidates
        }
        == {
            "c00_native_checkpoint_10000",
            "c01_native_checkpoint_20000",
        }
    )
    assert native.requested_updates == 20_000
    assert {
        candidate.sampled_checkpoint_step
        for candidate in native.candidates
    } == {10_000, 20_000}


@pytest.mark.parametrize(
    "model_id",
    ["ctgan_separate_class", "tvae_separate_class"],
)
def test_continuation_plan_reuses_native_and_trains_only_remaining_trajectories(
    model_id,
):
    config = read_yaml_mapping(
        REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
    )

    plan = build_model_continuation_plan(config, model_id=model_id)

    assert plan.model_id == model_id
    assert {
        candidate.candidate_id
        for candidate in plan.evaluation_only_candidates
    } == {
        "c00_native_checkpoint_10000",
        "c01_native_checkpoint_20000",
    }
    assert {
        candidate.candidate_id
        for trajectory in plan.training_trajectories
        for candidate in trajectory.candidates
    } == {
        "c02_zscore_checkpoint_20000",
        (
            "c03_zscore_tempered_checkpoint_20000"
            if model_id == "ctgan_separate_class"
            else "c03_zscore_weighted_tempered_checkpoint_20000"
        ),
    }
    assert len(plan.training_trajectories) == 2
    assert all(
        "native" not in trajectory.shared_trajectory_id
        for trajectory in plan.training_trajectories
    )
    assert plan.native_training_trajectory_count == 0


def test_continuation_authorization_rejects_corrupted_native_checkpoint(
    tmp_path,
):
    repository = tmp_path / "repository"
    candidate_root = (
        repository
        / "artifacts/benchmark_v2_6/selection/candidates"
    )
    model_id = "ctgan_separate_class"
    worker_root = candidate_root / "_workers" / model_id
    worker_attempt = worker_root / "attempt_001"
    worker_attempt.mkdir(parents=True)
    ownership = worker_root / "ownership.lock"
    ownership.write_text('{"immutable": true}\n', encoding="utf-8")
    failed = worker_attempt / "WORKER_FAILED.json"
    failed.write_text('{"status": "FAILED"}\n', encoding="utf-8")
    native = (
        candidate_root
        / "_trajectories"
        / model_id
        / "ctgan_native_seed_2601"
        / "seed_2601"
        / "attempt_001"
    )
    checkpoints = native / "checkpoints"
    checkpoints.mkdir(parents=True)
    manifest = native / "manifest.json"
    complete = native / "TRAJECTORY_COMPLETE.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "source_commit": "f" * 40,
                "relevant_source_sha256": "1" * 64,
                "selection_config_sha256": "c" * 64,
                "development_manifest_sha256": "d" * 64,
                "selection_plan_sha256": "e" * 64,
                "model_id": model_id,
                "shared_trajectory_id": "ctgan_native_seed_2601",
            }
        ),
        encoding="utf-8",
    )
    complete.write_text('{"status": "COMPLETE"}\n', encoding="utf-8")
    checkpoint_paths = {
        "c00_native_checkpoint_10000": checkpoints / "step_10000.pt",
        "c01_native_checkpoint_20000": checkpoints / "step_20000.pt",
    }
    for candidate_id, path in checkpoint_paths.items():
        path.write_bytes(candidate_id.encode())
    authorization = {
        "schema_version": (
            "benchmark-v2.6-candidate-continuation-authorization-v1"
        ),
        "candidate_continuation_authorized": True,
        "source_commit": "a" * 40,
        "relevant_source_sha256": "b" * 64,
        "selection_config_sha256": "c" * 64,
        "development_manifest_sha256": "d" * 64,
        "selection_plan_sha256": "e" * 64,
        "worker_model_ids": [model_id, "tvae_separate_class"],
        "test_access_authorized": False,
        "fresh_test_authorized": False,
        "five_seed_full_experiment_authorized": False,
        "validation_selection_execution_authorized": False,
        "approval_text": "fixture continuation only",
        "continuation_scope": {
            model_id: {
                "worker_attempt": "attempt_002",
                "prior_ownership_lock_sha256": sha256_file(ownership),
                "prior_worker_terminal_path": str(
                    failed.relative_to(repository)
                ),
                "prior_worker_terminal_sha256": sha256_file(failed),
                "attempt_001_preservation_sha256": (
                    preserved_attempt_tree_sha256(
                        repository_root=repository,
                        candidate_root=candidate_root,
                        model_id=model_id,
                        native_trajectory_id="ctgan_native_seed_2601",
                        selection_seed=2601,
                    )
                ),
                "native_training_source_commit": "f" * 40,
                "native_training_relevant_source_sha256": "1" * 64,
                "native_trajectory_manifest_path": str(
                    manifest.relative_to(repository)
                ),
                "native_trajectory_manifest_sha256": sha256_file(manifest),
                "native_trajectory_complete_path": str(
                    complete.relative_to(repository)
                ),
                "native_trajectory_complete_sha256": sha256_file(complete),
                "native_checkpoints": {
                    candidate_id: {
                        "path": str(path.relative_to(repository)),
                        "sha256": sha256_file(path),
                    }
                    for candidate_id, path in checkpoint_paths.items()
                },
                "evaluation_only_candidate_ids": list(checkpoint_paths),
                "training_only_candidate_ids": [
                    "c02_zscore_checkpoint_20000",
                    "c03_zscore_tempered_checkpoint_20000",
                ],
                "native_trajectory_retraining_authorized": False,
            }
        },
    }
    authorization_path = (
        repository
        / "artifacts/benchmark_v2_6/selection/authorization_history/"
        "continuation.json"
    )
    authorization_path.parent.mkdir(parents=True)
    authorization_path.write_text(
        json.dumps(authorization),
        encoding="utf-8",
    )

    validated = validate_candidate_continuation_authorization(
        repository_root=repository,
        candidate_root=candidate_root,
        authorization_path=authorization_path,
        source_commit="a" * 40,
        source_hash="b" * 64,
        selection_config_sha256="c" * 64,
        development_manifest_sha256="d" * 64,
        selection_plan_sha256="e" * 64,
        model_id=model_id,
        plan=build_model_continuation_plan(
            read_yaml_mapping(
                REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
            ),
            model_id=model_id,
        ),
    )
    assert validated["candidate_continuation_authorized"] is True

    authorization["source_commit"] = "9" * 40
    authorization_path.write_text(
        json.dumps(authorization),
        encoding="utf-8",
    )
    with pytest.raises(
        CandidateRunnerContractError,
        match="authorization mismatch",
    ):
        validate_candidate_continuation_authorization(
            repository_root=repository,
            candidate_root=candidate_root,
            authorization_path=authorization_path,
            source_commit="a" * 40,
            source_hash="b" * 64,
            selection_config_sha256="c" * 64,
            development_manifest_sha256="d" * 64,
            selection_plan_sha256="e" * 64,
            model_id=model_id,
            plan=build_model_continuation_plan(
                read_yaml_mapping(
                    REPOSITORY
                    / "configs/benchmark_v2/selection_v2_6.yaml"
                ),
                model_id=model_id,
            ),
        )
    authorization["source_commit"] = "a" * 40
    authorization_path.write_text(
        json.dumps(authorization),
        encoding="utf-8",
    )
    checkpoint_paths["c00_native_checkpoint_10000"].write_bytes(b"corrupt")
    with pytest.raises(
        CandidateRunnerContractError,
        match="checkpoint",
    ):
        validate_candidate_continuation_authorization(
            repository_root=repository,
            candidate_root=candidate_root,
            authorization_path=authorization_path,
            source_commit="a" * 40,
            source_hash="b" * 64,
            selection_config_sha256="c" * 64,
            development_manifest_sha256="d" * 64,
            selection_plan_sha256="e" * 64,
            model_id=model_id,
            plan=build_model_continuation_plan(
                read_yaml_mapping(
                    REPOSITORY
                    / "configs/benchmark_v2/selection_v2_6.yaml"
                ),
                model_id=model_id,
            ),
        )


def test_continuation_claim_preserves_attempt_001_lock_and_tree(tmp_path):
    repository = tmp_path / "repository"
    candidate_root = (
        repository
        / "artifacts/benchmark_v2_6/selection/candidates"
    )
    model_id = "tvae_separate_class"
    worker_root = candidate_root / "_workers" / model_id
    prior_worker = worker_root / "attempt_001"
    prior_worker.mkdir(parents=True)
    ownership = worker_root / "ownership.lock"
    ownership.write_bytes(b'{"original":true}\n')
    (prior_worker / "WORKER_FAILED.json").write_bytes(
        b'{"status":"FAILED"}\n'
    )
    native = (
        candidate_root
        / "_trajectories"
        / model_id
        / "tvae_native_seed_2601"
        / "seed_2601"
        / "attempt_001"
    )
    native.mkdir(parents=True)
    (native / "manifest.json").write_bytes(b'{"original":true}\n')
    before_lock = ownership.read_bytes()
    before_tree = preserved_attempt_tree_sha256(
        repository_root=repository,
        candidate_root=candidate_root,
        model_id=model_id,
        native_trajectory_id="tvae_native_seed_2601",
        selection_seed=2601,
    )
    plan = build_model_continuation_plan(
        read_yaml_mapping(
            REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
        ),
        model_id=model_id,
    )

    attempt = claim_model_continuation(
        repository_root=repository,
        candidate_root=candidate_root,
        plan=plan,
        authorization_sha256="a" * 64,
        prior_ownership_lock_sha256=sha256_file(ownership),
        attempt_001_preservation_sha256=before_tree,
        provenance={"source_commit": "b" * 40},
    )

    assert attempt.name == "attempt_002"
    assert ownership.read_bytes() == before_lock
    assert (
        preserved_attempt_tree_sha256(
            repository_root=repository,
            candidate_root=candidate_root,
            model_id=model_id,
            native_trajectory_id="tvae_native_seed_2601",
            selection_seed=2601,
        )
        == before_tree
    )
    assert (attempt / "worker_manifest.json").is_file()
    assert (attempt / "RUNNING.json").is_file()
    with pytest.raises(
        CandidateRunnerContractError,
        match="attempt_002",
    ):
        claim_model_continuation(
            repository_root=repository,
            candidate_root=candidate_root,
            plan=plan,
            authorization_sha256="a" * 64,
            prior_ownership_lock_sha256=sha256_file(ownership),
            attempt_001_preservation_sha256=before_tree,
            provenance={"source_commit": "b" * 40},
        )


@pytest.mark.parametrize(
    "model_id",
    ["ctgan_separate_class", "tvae_separate_class"],
)
def test_continuation_dispatch_never_retrains_native_trajectory(model_id):
    plan = build_model_continuation_plan(
        read_yaml_mapping(
            REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
        ),
        model_id=model_id,
    )
    calls = {"evaluation": [], "training": []}

    results = dispatch_continuation_plan(
        plan,
        evaluate_native=lambda candidates: calls["evaluation"].append(
            tuple(candidate.candidate_id for candidate in candidates)
        )
        or {"status": "COMPLETE", "operation": "evaluation_only"},
        train_trajectory=lambda trajectory: calls["training"].append(
            trajectory.shared_trajectory_id
        )
        or {"status": "COMPLETE", "operation": "training_only"},
    )

    assert calls["evaluation"] == [
        (
            "c00_native_checkpoint_10000",
            "c01_native_checkpoint_20000",
        )
    ]
    assert len(calls["training"]) == 2
    assert all("native" not in value for value in calls["training"])
    assert [result["operation"] for result in results] == [
        "evaluation_only",
        "training_only",
        "training_only",
    ]


def test_worker_cli_requires_one_frozen_model_and_rejects_unknown_model():
    required = [
        "--authorization",
        "authorization.json",
        "--source-commit",
        "a" * 40,
        "--device",
        "cuda:0",
    ]

    parsed = parse_args(
        [
            *required,
            "--model",
            "cof_seqgen",
        ]
    )

    assert parsed.model == "cof_seqgen"
    with pytest.raises(SystemExit):
        parse_args(required)
    with pytest.raises(SystemExit):
        parse_args([*required, "--model", "unknown"])


@pytest.mark.parametrize(
    "model_id",
    ["ctgan_separate_class", "tvae_separate_class"],
)
def test_worker_cli_continuation_is_limited_to_failed_tabular_workers(
    model_id,
):
    required = [
        "--authorization",
        "authorization.json",
        "--source-commit",
        "a" * 40,
        "--device",
        "cuda:0",
        "--continuation",
    ]

    parsed = parse_args([*required, "--model", model_id])

    assert parsed.continuation is True
    with pytest.raises(SystemExit):
        parse_args([*required, "--model", "cof_seqgen"])
    with pytest.raises(SystemExit):
        parse_args([*required, "--model", "neural_sequence"])


def test_model_worker_ownership_rejects_duplicate_but_not_other_models(
    tmp_path,
):
    candidate_root = tmp_path / "artifacts/benchmark_v2_6/selection/candidates"
    provenance = {
        "source_commit": "a" * 40,
        "relevant_source_sha256": "b" * 64,
        "selection_config_sha256": "c" * 64,
        "development_manifest_sha256": "d" * 64,
        "selection_plan_sha256": "e" * 64,
    }

    ctgan = claim_model_worker(
        candidate_root=candidate_root,
        model_id="ctgan_separate_class",
        selection_seed=2601,
        trajectory_ids=("native", "zscore", "tempered"),
        candidate_ids=("c00", "c01", "c02", "c03"),
        provenance=provenance,
    )

    assert ctgan.name == "attempt_001"
    assert (ctgan / "worker_manifest.json").is_file()
    assert (ctgan / "RUNNING.json").is_file()
    assert (
        candidate_root
        / "_workers/ctgan_separate_class/ownership.lock"
    ).is_file()
    with pytest.raises(
        CandidateRunnerContractError,
        match="already owned",
    ):
        claim_model_worker(
            candidate_root=candidate_root,
            model_id="ctgan_separate_class",
            selection_seed=2601,
            trajectory_ids=("native", "zscore", "tempered"),
            candidate_ids=("c00", "c01", "c02", "c03"),
            provenance=provenance,
        )

    tvae = claim_model_worker(
        candidate_root=candidate_root,
        model_id="tvae_separate_class",
        selection_seed=2601,
        trajectory_ids=("native", "zscore", "tempered"),
        candidate_ids=("c00", "c01", "c02", "c03"),
        provenance=provenance,
    )

    assert tvae.name == "attempt_001"
    assert ctgan.parent.name == "ctgan_separate_class"
    assert tvae.parent.name == "tvae_separate_class"
    assert not (candidate_root.parent / "selection_report.json").exists()


def test_aggregate_readiness_rejects_before_all_four_workers_are_complete(
    tmp_path,
):
    candidate_root = tmp_path / "candidates"
    candidate_root.mkdir()
    config = read_yaml_mapping(
        REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
    )

    with pytest.raises(
        CandidateRunnerContractError,
        match="four model workers",
    ):
        validate_candidate_aggregate_readiness(
            candidate_root=candidate_root,
            config=config,
            source_commit="a" * 40,
            relevant_source_sha256="b" * 64,
            selection_config_sha256="c" * 64,
            development_manifest_sha256="d" * 64,
            selection_plan_sha256="e" * 64,
        )

    assert not (candidate_root.parent / "selection_report.json").exists()


def test_aggregate_readiness_accepts_only_complete_four_worker_fixture(
    tmp_path,
    monkeypatch,
):
    candidate_root = tmp_path / "candidates"
    config = read_yaml_mapping(
        REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
    )
    provenance = {
        "source_commit": "a" * 40,
        "relevant_source_sha256": "b" * 64,
        "selection_config_sha256": "c" * 64,
        "development_manifest_sha256": "d" * 64,
        "selection_plan_sha256": "e" * 64,
    }
    calls = {
        "adapter": 0,
        "fit": 0,
        "sample": 0,
        "dgp": 0,
        "cuda": 0,
    }

    def forbidden_adapter(*args, **kwargs):
        calls["adapter"] += 1
        raise AssertionError("aggregate planning must not build an adapter")

    monkeypatch.setattr(
        "experiments.candidate_runner_v2_6.build_candidate_adapter",
        forbidden_adapter,
    )
    for model_id in config["model_order"]:
        plan = build_model_trajectory_plan(config, model_id=model_id)
        trajectory_ids = tuple(
            item.shared_trajectory_id for item in plan.trajectories
        )
        candidate_ids = tuple(
            candidate.candidate_id
            for item in plan.trajectories
            for candidate in item.candidates
        )
        worker = claim_model_worker(
            candidate_root=candidate_root,
            model_id=model_id,
            selection_seed=plan.selection_seed,
            trajectory_ids=trajectory_ids,
            candidate_ids=candidate_ids,
            provenance=provenance,
        )
        for candidate_id in candidate_ids:
            attempt = (
                candidate_root
                / model_id
                / candidate_id
                / f"seed_{plan.selection_seed}"
                / "attempt_001"
            )
            attempt.mkdir(parents=True)
            (attempt / "candidate_result.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "model_id": model_id,
                        "candidate_id": candidate_id,
                        **provenance,
                    }
                ),
                encoding="utf-8",
            )
            (attempt / "COMPLETE.json").write_text(
                json.dumps({"status": "COMPLETE"}),
                encoding="utf-8",
            )
        terminal = finalize_model_worker(
            candidate_root=candidate_root,
            worker_attempt=worker,
            plan=plan,
            provenance=provenance,
            trajectory_results=(
                {"status": "COMPLETE"},
                {"status": "COMPLETE"},
                {"status": "COMPLETE"},
            ),
        )
        assert terminal["status"] == "COMPLETE"
        assert terminal["selection_aggregate_created"] is False

    readiness = validate_candidate_aggregate_readiness(
        candidate_root=candidate_root,
        config=config,
        source_commit=provenance["source_commit"],
        relevant_source_sha256=provenance[
            "relevant_source_sha256"
        ],
        selection_config_sha256=provenance[
            "selection_config_sha256"
        ],
        development_manifest_sha256=provenance[
            "development_manifest_sha256"
        ],
        selection_plan_sha256=provenance["selection_plan_sha256"],
    )

    assert readiness["status"] == "READY_FOR_VALIDATION_AGGREGATE"
    assert readiness["worker_count"] == 4
    assert readiness["trajectory_count"] == 12
    assert readiness["candidate_count"] == 16
    assert calls == {
        "adapter": 0,
        "fit": 0,
        "sample": 0,
        "dgp": 0,
        "cuda": 0,
    }


def test_aggregate_readiness_accepts_authorized_mixed_worker_attempts(
    tmp_path,
):
    candidate_root = tmp_path / "candidates"
    config = read_yaml_mapping(
        REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
    )
    plan_hash = "e" * 64
    config_hash = "c" * 64
    development_hash = "d" * 64
    attempt_by_model = {
        "cof_seqgen": "attempt_001",
        "ctgan_separate_class": "attempt_002",
        "tvae_separate_class": "attempt_002",
        "neural_sequence": "attempt_001",
    }
    source_by_model = {
        "cof_seqgen": ("1" * 40, "2" * 64),
        "ctgan_separate_class": ("3" * 40, "4" * 64),
        "tvae_separate_class": ("3" * 40, "4" * 64),
        "neural_sequence": ("3" * 40, "4" * 64),
    }
    approved = {}
    for model_id in config["model_order"]:
        plan = build_model_trajectory_plan(config, model_id=model_id)
        _, trajectory_ids, candidate_ids = (
            model_id,
            tuple(
                trajectory.shared_trajectory_id
                for trajectory in plan.trajectories
            ),
            tuple(
                candidate.candidate_id
                for trajectory in plan.trajectories
                for candidate in trajectory.candidates
            ),
        )
        candidate_records = {}
        for candidate_id in candidate_ids:
            candidate_attempt = (
                candidate_root
                / model_id
                / candidate_id
                / f"seed_{plan.selection_seed}"
                / "attempt_001"
            )
            candidate_attempt.mkdir(parents=True)
            result_path = candidate_attempt / "candidate_result.json"
            complete_path = candidate_attempt / "COMPLETE.json"
            result_path.write_text(
                json.dumps({"status": "COMPLETE"}),
                encoding="utf-8",
            )
            complete_path.write_text(
                json.dumps({"status": "COMPLETE"}),
                encoding="utf-8",
            )
            candidate_records[candidate_id] = {
                "candidate_result_path": str(
                    result_path.relative_to(candidate_root)
                ),
                "candidate_result_sha256": sha256_file(result_path),
                "complete_path": str(
                    complete_path.relative_to(candidate_root)
                ),
                "complete_sha256": sha256_file(complete_path),
            }
        source_commit, source_hash = source_by_model[model_id]
        attempt_name = attempt_by_model[model_id]
        schema = (
            "benchmark-v2.6-model-worker-continuation-v1"
            if attempt_name == "attempt_002"
            else "benchmark-v2.6-model-worker-v1"
        )
        terminal_path = (
            candidate_root
            / "_workers"
            / model_id
            / attempt_name
            / "WORKER_COMPLETE.json"
        )
        terminal_path.parent.mkdir(parents=True)
        terminal = {
            "schema_version": schema,
            "status": "COMPLETE",
            "model_id": model_id,
            "selection_seed": plan.selection_seed,
            "trajectory_ids": list(trajectory_ids),
            "candidate_ids": list(candidate_ids),
            "trajectory_count": 3,
            "candidate_count": 4,
            "selection_aggregate_created": False,
            "provenance": {
                "source_commit": source_commit,
                "relevant_source_sha256": source_hash,
                "selection_config_sha256": config_hash,
                "development_manifest_sha256": development_hash,
                "selection_plan_sha256": plan_hash,
            },
            "candidate_artifacts": candidate_records,
        }
        if attempt_name == "attempt_002":
            terminal["native_trajectory_retrained"] = False
        terminal_path.write_text(
            json.dumps(terminal),
            encoding="utf-8",
        )
        approved[model_id] = {
            "attempt": attempt_name,
            "schema_version": schema,
            "terminal_sha256": sha256_file(terminal_path),
            "source_commit": source_commit,
            "relevant_source_sha256": source_hash,
        }

    readiness = validate_candidate_aggregate_readiness(
        candidate_root=candidate_root,
        config=config,
        source_commit="3" * 40,
        relevant_source_sha256="4" * 64,
        selection_config_sha256=config_hash,
        development_manifest_sha256=development_hash,
        selection_plan_sha256=plan_hash,
        approved_worker_terminals=approved,
    )

    assert readiness["status"] == "READY_FOR_VALIDATION_AGGREGATE"
    assert {
        model_id: worker["attempt"]
        for model_id, worker in readiness["workers"].items()
    } == attempt_by_model
    assert readiness["mixed_worker_attempts"] is True
    assert readiness["mixed_source_provenance"] is True

    primary_only = validate_candidate_aggregate_readiness(
        candidate_root=candidate_root,
        config=config,
        source_commit="3" * 40,
        relevant_source_sha256="4" * 64,
        selection_config_sha256=config_hash,
        development_manifest_sha256=development_hash,
        selection_plan_sha256=plan_hash,
        approved_worker_terminals={
            model_id: record
            for model_id, record in approved.items()
            if model_id != "neural_sequence"
        },
        unavailable_model_ids=("neural_sequence",),
    )

    assert primary_only["status"] == "READY_FOR_VALIDATION_AGGREGATE"
    assert primary_only["worker_count"] == 3
    assert primary_only["trajectory_count"] == 9
    assert primary_only["candidate_count"] == 12
    assert primary_only["unavailable_model_ids"] == ["neural_sequence"]


def test_aggregate_only_cli_requires_authorization_and_source_commit():
    required = [
        "--authorization",
        "authorization.json",
        "--source-commit",
        "a" * 40,
    ]

    parsed = parse_aggregate_args([*required, "--mode", "plan"])

    assert parsed.authorization == Path("authorization.json")
    assert parsed.source_commit == "a" * 40
    assert parsed.mode == "plan"
    assert parse_aggregate_args(required).mode == "execute"
    with pytest.raises(SystemExit):
        parse_aggregate_args([])


def test_selection_bundle_is_append_only_and_indexes_final_hashes(tmp_path):
    output_root = tmp_path / "aggregate_attempt_001"
    authorization = tmp_path / "authorization.json"
    authorization.write_text('{"authorized":true}\n', encoding="utf-8")

    terminal = write_selection_artifact_bundle(
        output_root=output_root,
        authorization_path=authorization,
        readiness={"status": "READY", "worker_count": 3},
        report={"status": "PRIMARY_SELECTION_PASS"},
        selection_manifest={
            "status": "FROZEN_AWAITING_SEPARATE_AUTHORIZATION"
        },
        source_commit="a" * 40,
        relevant_source_sha256="b" * 64,
        worker_count=3,
        trajectory_count=9,
        candidate_count=12,
        unavailable_model_ids=("neural_sequence",),
        primary_c2_selection_ready=True,
    )

    assert terminal["status"] == "COMPLETE"
    assert terminal["artifact_index_sha256"] == sha256_file(
        output_root / "artifact_index.json"
    )
    index = json.loads(
        (output_root / "artifact_index.json").read_text()
    )
    assert {
        record["path"] for record in index["artifacts"]
    } == {
        "aggregate_readiness.json",
        "selection_report.json",
        "selection_manifest.json",
        "checksum_manifest.json",
    }
    for record in index["artifacts"]:
        assert sha256_file(output_root / record["path"]) == record["sha256"]
    assert (output_root / "AGGREGATE_COMPLETE.json").is_file()
    with pytest.raises(
        CandidateRunnerContractError,
        match="append-only aggregate attempt exists",
    ):
        write_selection_artifact_bundle(
            output_root=output_root,
            authorization_path=authorization,
            readiness={"status": "READY"},
            report={"status": "PRIMARY_SELECTION_PASS"},
            selection_manifest={"status": "FROZEN"},
            source_commit="a" * 40,
            relevant_source_sha256="b" * 64,
            worker_count=3,
            trajectory_count=9,
            candidate_count=12,
            unavailable_model_ids=("neural_sequence",),
            primary_c2_selection_ready=True,
        )


def test_train_context_never_opens_validation_and_plan_is_train_fitted(
    tmp_path,
    monkeypatch,
):
    repository = tmp_path / "repository"
    frozen = repository / "data/benchmark_v2_5/frozen/s/v"
    frozen.mkdir(parents=True)
    train = train_fixture()
    train_path = frozen / "train.npz"
    save_batch(train_path, train)
    validation_path = frozen / "validation.npz"
    validation_path.write_bytes(b"must not be opened")
    from generators.sampling_plan import SamplingPlan

    plan = SamplingPlan.from_train_policy(
        train,
        entity_count=5,
        seed=26001,
    )
    manifest = {
        "schema_version": "benchmark-v2.6-development-data-v1",
        "status": "FROZEN_FOR_SELECTION",
        "test_split_access": "FORBIDDEN",
        "splits": {
            "train": {
                "path": str(train_path.relative_to(repository)),
                "file_sha256": sha256_file(train_path),
                "content_sha256": hash_batch(train),
                "entities": 4,
            },
            "validation": {
                "path": str(validation_path.relative_to(repository)),
                "file_sha256": "f" * 64,
                "content_sha256": "e" * 64,
                "entities": 5,
            },
        },
        "selection_sampling_plan": {
            "fit_split": "train",
            "entity_count": 5,
            "seed": 26001,
            "content_sha256": plan.plan_hash,
        },
        "tau": [0.5, 2.0],
    }
    manifest_path = repository / "configs/development.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest),
        encoding="utf-8",
    )
    seen: list[Path] = []
    real_load = np.load

    def traced_load(path, *args, **kwargs):
        seen.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(
        "experiments.candidate_runner_v2_6.np.load",
        traced_load,
    )

    context = load_train_only_candidate_context(
        repository_root=repository,
        development_manifest_path=manifest_path,
    )

    assert seen == [train_path]
    assert context.plan.plan_hash == plan.plan_hash
    assert context.validation_file_sha256 == "f" * 64
    assert context.validation_content_sha256 == "e" * 64


@pytest.mark.parametrize(
    "relative",
    [
        "data/benchmark_v2_5/frozen/s/v/test.npz",
        "data/benchmark_v2_6/fresh_test/test.npz",
        "artifacts/benchmark_v2_5/full/runtime.json",
        "artifacts/benchmark_v2_6/full/output.json",
    ],
)
def test_candidate_runner_rejects_test_fresh_full_and_v25_runtime_paths(
    tmp_path,
    relative,
):
    with pytest.raises(CandidateRunnerContractError, match="forbidden"):
        validate_candidate_io_path(
            repository_root=tmp_path,
            path=tmp_path / relative,
            role="fixture",
            access="read",
        )


def test_attempt_claim_is_append_only(tmp_path):
    root = tmp_path / "trajectory"
    first = claim_next_attempt(root)
    (first / "manifest.json").write_text("{}", encoding="utf-8")

    second = claim_next_attempt(root)

    assert first.name == "attempt_001"
    assert second.name == "attempt_002"
    assert (first / "manifest.json").read_text() == "{}"


def test_wall_cap_terminates_only_owned_child_and_fails_closed(tmp_path):
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()
    started = time.monotonic()

    result = run_bounded_process(
        target=hang_fixture,
        attempt_path=attempt,
        max_wall_seconds=0.2,
        termination_grace_seconds=0.1,
    )

    assert time.monotonic() - started < 2
    assert result["status"] == "FAILED"
    assert result["failure_class"] == "wall_cap"
    failed = json.loads((attempt / "FAILED.json").read_text())
    assert failed["failure_class"] == "wall_cap"
    assert failed["actual_elapsed_seconds"] >= 0.2
    assert not (attempt / "COMPLETE.json").exists()


def test_candidate_result_manifest_records_complete_shared_provenance():
    value = build_candidate_result_manifest(
        model_id="cof_seqgen",
        candidate_id="c00_native_checkpoint_10000",
        shared_trajectory_id="cof_native_seed_2601",
        selection_seed=2601,
        requested_updates=20_000,
        actual_updates=20_000,
        sampled_checkpoint_step=10_000,
        source_commit="a" * 40,
        relevant_source_sha256="b" * 64,
        selection_config_sha256="c" * 64,
        development_manifest_sha256="d" * 64,
        train_file_sha256="e" * 64,
        train_content_sha256="f" * 64,
        validation_file_sha256="1" * 64,
        validation_content_sha256="2" * 64,
        selection_plan_sha256="3" * 64,
        trajectory_manifest_path="trajectory/manifest.json",
        trajectory_manifest_sha256="4" * 64,
        trajectory_complete_path="trajectory/TRAJECTORY_COMPLETE.json",
        trajectory_complete_sha256="8" * 64,
        checkpoint_path="trajectory/checkpoints/step_10000.pt",
        checkpoint_sha256="5" * 64,
        validation_sample_path="candidate/validation_sample.npz",
        validation_sample_sha256="6" * 64,
        candidate_definition_sha256="7" * 64,
        actual_wall_seconds=12.5,
    )

    assert value["status"] == "COMPLETE"
    assert value["requested_updates"] == 20_000
    assert value["actual_updates"] == 20_000
    assert value["sampled_checkpoint_step"] == 10_000
    assert value["shared_trajectory_id"] == "cof_native_seed_2601"
    assert value["trajectory_completed_requested_updates"] is True
    assert value["wall_cap_reached"] is False
    assert value["checkpoint_sha256"] == "5" * 64
    assert value["validation_sample_sha256"] == "6" * 64


def test_checkpoint_bundle_contains_reversible_train_only_zscore_state():
    transform = TrainOnlyZScore.fit(train_fixture())
    backend = object()

    bundle = build_checkpoint_bundle(
        backend=backend,
        train_only_zscore=transform,
        shared_trajectory_id="neural_zscore_seed_2601",
        sampled_checkpoint_step=20_000,
        selection_plan_sha256="a" * 64,
    )

    assert bundle["backend"] is backend
    assert bundle["train_only_zscore"] is transform
    assert bundle["train_only_zscore_state"]["fit_split"] == "train"
    assert bundle["train_only_zscore_state"]["mean"] == [3.5]
    assert bundle["train_only_zscore_state"]["std"] == pytest.approx(
        [np.std([1, 2, 3, 4, 5, 6])]
    )


def test_no_op_watchdog_fixture_calls_no_model_or_gpu(tmp_path):
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()

    result = run_bounded_process(
        target=no_op_fixture,
        attempt_path=attempt,
        max_wall_seconds=1.0,
        termination_grace_seconds=0.1,
    )

    assert result["status"] == "COMPLETE"
    assert (attempt / "COMPLETE.json").exists()
    assert not (attempt / "FAILED.json").exists()


@pytest.mark.parametrize(
    ("target", "expected_class"),
    [
        (invalid_candidate_fixture, "invalid_candidate"),
        (infrastructure_failure_fixture, "infrastructure_or_code"),
    ],
)
def test_invalid_and_infrastructure_fail_closed(
    tmp_path,
    target,
    expected_class,
):
    attempt = tmp_path / target.__name__
    attempt.mkdir()

    result = run_bounded_process(
        target=target,
        attempt_path=attempt,
        max_wall_seconds=1.0,
        termination_grace_seconds=0.1,
    )

    assert result["status"] == "FAILED"
    assert result["failure_class"] == expected_class
    assert (attempt / "FAILED.json").exists()
    assert not (attempt / "COMPLETE.json").exists()


def test_plan_fixture_calls_no_model_fit_sample_cuda_or_dgp(monkeypatch):
    calls = {
        "adapter": 0,
        "cuda": 0,
        "dgp": 0,
        "fit": 0,
        "sample": 0,
    }

    def forbidden_adapter(*args, **kwargs):
        calls["adapter"] += 1
        raise AssertionError("planning must not construct an adapter")

    monkeypatch.setattr(
        "experiments.candidate_runner_v2_6.build_candidate_adapter",
        forbidden_adapter,
    )
    config = read_yaml_mapping(
        REPOSITORY / "configs/benchmark_v2/selection_v2_6.yaml"
    )

    plan = build_trajectory_plan(config)

    assert len(plan.trajectories) == 12
    assert calls == {
        "adapter": 0,
        "cuda": 0,
        "dgp": 0,
        "fit": 0,
        "sample": 0,
    }
