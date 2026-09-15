"""Execution-only implementation for external confirmatory internal tests.

The module is imported only after the source-only runner validates a separate
authorization.  It restores validation-frozen generator state and performs one
sample/evaluation call.  There is deliberately no fit, refit, selection,
checkpoint-write, retry, or aggregate path in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np
import torch
import yaml

from benchmarks.types import SequenceBatch, SyntheticBatch
from eval.external_validation_metrics_v1 import (
    compute_external_validation_metrics,
    validate_external_hard_contract,
)
from eval.external_validation_protocol_v1 import (
    COHERENCE_METRICS,
    FIDELITY_METRICS,
    validate_train_bootstrap_thresholds,
)
from generators.sampling_plan import SamplingPlan
from scripts.run_external_confirmatory_internal_test_v1 import (
    ConfirmatoryInternalTestError,
    validate_confirmatory_authorization,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
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


def _exclusive_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
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


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ConfirmatoryInternalTestError(f"JSON evidence is not an object: {path}")
    return value


def _load_batch_hash_checked(path: Path, expected_sha256: str) -> SequenceBatch:
    if _sha256_file(path) != expected_sha256:
        raise ConfirmatoryInternalTestError(f"frozen NPZ hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        fields = {
            name: archive[name]
            for name in SequenceBatch.__dataclass_fields__
        }
    return SequenceBatch(**fields)


class _FrozenEmpiricalIIDState:
    """Exact empirical IID row pool restored without invoking ``fit``."""

    def __init__(self, train: SequenceBatch):
        self.pools = {
            "x_num": train.x_num[train.valid_mask],
            "dt_bin": train.dt_bin[train.valid_mask],
            "x_cat": train.x_cat[train.valid_mask],
        }
        self.row_y = np.repeat(train.y_entity, train.lengths)

    def sample(self, plan: SamplingPlan, *, seed: int) -> SyntheticBatch:
        rng = np.random.default_rng(seed)
        count, length = plan.valid_mask.shape
        x_num = np.zeros(
            (count, length, self.pools["x_num"].shape[-1]), dtype=np.float32
        )
        dt_bin = np.zeros((count, length), dtype=np.int64)
        x_cat = np.zeros(
            (count, length, self.pools["x_cat"].shape[-1]), dtype=np.int64
        )
        for label in (0, 1):
            positions = np.argwhere(
                plan.valid_mask & (plan.y_entity[:, None] == label)
            )
            pool = np.flatnonzero(self.row_y == label)
            if not len(pool):
                raise ConfirmatoryInternalTestError(
                    f"frozen empirical pool has no rows for label {label}"
                )
            selected = rng.choice(pool, len(positions), replace=True)
            x_num[positions[:, 0], positions[:, 1]] = self.pools["x_num"][selected]
            dt_bin[positions[:, 0], positions[:, 1]] = self.pools["dt_bin"][selected]
            x_cat[positions[:, 0], positions[:, 1]] = self.pools["x_cat"][selected]
        return SyntheticBatch(
            x_num=x_num,
            dt_bin=dt_bin,
            x_cat=x_cat,
            valid_mask=plan.valid_mask.copy(),
            y_entity=plan.y_entity.copy(),
            lengths=plan.lengths.copy(),
        )


def _restore_tabular_adapter(
    checkpoint: Path,
    *,
    model: str,
    device: str,
    train_pool_path: Path,
    train_pool_sha256: str,
) -> Any:
    import ctgan.synthesizers.base as ctgan_base

    original_set_device = ctgan_base._set_device
    ctgan_base._set_device = lambda enable_gpu: torch.device("cpu")
    try:
        adapter = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
    finally:
        ctgan_base._set_device = original_set_device
    expected_name = {
        "ctgan_separate_class": "conditional_ctgan",
        "tvae_separate_class": "conditional_tvae",
    }[model]
    if getattr(adapter, "name", None) != expected_name:
        raise ConfirmatoryInternalTestError(
            f"restored tabular checkpoint type mismatch: {model}"
        )
    class_models = getattr(adapter, "models", None)
    if not isinstance(class_models, Mapping) or set(class_models) != {0, 1}:
        raise ConfirmatoryInternalTestError(
            "restored tabular checkpoint has no exact class pair"
        )
    from ctgan import CTGAN, TVAE

    sampling_set_device = CTGAN.set_device if model == "ctgan_separate_class" else TVAE.set_device
    for synthesizer in class_models.values():
        # Invoke the upstream sampling-only device method explicitly. The
        # checkpointable subclasses also move optimizer/discriminator state,
        # which is unnecessary and prohibited for this evaluation-only path.
        sampling_set_device(synthesizer, device)
    if model == "ctgan_separate_class":
        # CheckpointableCTGAN deliberately omits the reconstructible DataSampler
        # from pickle. Rebuild it from the already-fitted transformer and the
        # exact hash-bound train rows. No transformer/model fit occurs.
        import pandas as pd
        from ctgan.data_sampler import DataSampler
        from experiments.external_validation_runner_v1 import (
            bounded_data_transformer_policy,
        )
        from generators.conditional_ctgan import _frame_hash

        train = _load_batch_hash_checked(train_pool_path, train_pool_sha256)
        row_labels = np.repeat(train.y_entity, train.lengths)
        frame = pd.DataFrame(
            {
                "amount_log": train.x_num[train.valid_mask, 0],
                "dt_bin": train.dt_bin[train.valid_mask],
                "receiver": train.x_cat[train.valid_mask, 0],
            }
        )
        with bounded_data_transformer_policy(n_jobs=1):
            for label, synthesizer in class_models.items():
                class_frame = frame.loc[row_labels == label]
                if adapter.training_data_hashes_by_class.get(label) != _frame_hash(
                    class_frame
                ):
                    raise ConfirmatoryInternalTestError(
                        "CTGAN frozen train rows differ from checkpoint"
                    )
                transformed = synthesizer._transformer.transform(class_frame)
                synthesizer._data_sampler = DataSampler(
                    transformed,
                    synthesizer._transformer.output_info_list,
                    synthesizer._log_frequency,
                )
    return adapter


def _restore_cof_adapter(
    checkpoint: Path,
    *,
    device: str,
    transform: Mapping[str, Any],
    summary: Mapping[str, Any],
    repository: Path,
) -> Any:
    from generators.cof_seqgen_adapter import CoFSeqGenAdapter
    from models.cof_seqgen import CoFSeqGen
    from models.seq_denoiser import SeqDenoiser

    config_path = repository / "configs/benchmark_v2/full_v2_5.yaml"
    frozen = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = frozen["baselines"]["cof_seqgen"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        not isinstance(state, Mapping)
        or state.get("schema_version") != "benchmark-v2.5-cof-seqgen"
        or "model" not in state
        or "sample_config" not in state
    ):
        raise ConfirmatoryInternalTestError("frozen CoF checkpoint schema mismatch")
    gap_tau = np.asarray(transform["gap_tau"], dtype=np.float32)
    receiver_cardinality = int(summary["receiver_vocabulary_cardinality"])
    sequence_length = int(summary["splits"]["train"]["length_max"])
    denoiser = SeqDenoiser(
        d_num=1,
        Bbins=len(gap_tau),
        n_cat_classes=[receiver_cardinality],
        d_model=int(source["d_model"]),
        n_layers=int(source["n_layers"]),
        L_max=sequence_length,
    )
    model = CoFSeqGen(
        denoiser,
        torch.as_tensor(gap_tau),
        float(frozen["data"]["window_width"]),
        float(frozen["data"]["soft_g_temperature"]),
        float(source["coherence_lambda"]),
        [receiver_cardinality],
        cfg_dropout=float(source["cfg_dropout"]),
    )
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    adapter = CoFSeqGenAdapter()
    adapter.model = model
    adapter.device = device
    adapter.sample_config = dict(state["sample_config"])
    adapter.train = SimpleNamespace(
        x_num=np.empty((0, sequence_length, 1), dtype=np.float32),
        x_cat=np.empty((0, sequence_length, 1), dtype=np.int64),
    )
    return adapter


def restore_frozen_validation_state(job: Mapping[str, Any], device: str) -> Any:
    """Restore one validation-frozen generator without fit or checkpoint write."""

    state_path = Path(job["restore"]["state_path"])
    if _sha256_file(state_path) != job["restore"]["state_sha256"]:
        raise ConfirmatoryInternalTestError("generator state hash mismatch")
    model = str(job["model"])
    if model == "empirical_iid":
        if device != "cpu":
            raise ConfirmatoryInternalTestError("empirical IID restore requires CPU")
        train = _load_batch_hash_checked(
            state_path, str(job["restore"]["state_sha256"])
        )
        return _FrozenEmpiricalIIDState(train)
    if not device.startswith("cuda:"):
        raise ConfirmatoryInternalTestError(
            "learned confirmatory restore requires an explicit CUDA device"
        )
    if model in {"ctgan_separate_class", "tvae_separate_class"}:
        return _restore_tabular_adapter(
            state_path,
            model=model,
            device=device,
            train_pool_path=Path(job["train_pool_path"]),
            train_pool_sha256=str(job["train_pool_sha256"]),
        )
    if model == "cof_seqgen_frozen_non_v3":
        transform = _read_json(Path(job["transform_path"]))
        summary = _read_json(Path(job["summary_path"]))
        repository = Path(__file__).resolve().parents[1]
        return _restore_cof_adapter(
            state_path,
            device=device,
            transform=transform,
            summary=summary,
            repository=repository,
        )
    raise ConfirmatoryInternalTestError(f"unsupported confirmatory model: {model}")


def _score_without_selection(
    *,
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_train_bootstrap_thresholds(thresholds)
    ratios: dict[str, float] = {}
    passes: dict[str, bool] = {}
    for family, names in (
        ("fidelity", FIDELITY_METRICS),
        ("coherence", COHERENCE_METRICS),
    ):
        values = metrics.get(family)
        if not isinstance(values, Mapping) or set(values) != set(names):
            raise ConfirmatoryInternalTestError(
                f"confirmatory {family} metrics are incomplete"
            )
        for name in names:
            ratio = float(values[name]) / float(thresholds[family][name])
            if not np.isfinite(ratio) or ratio < 0:
                raise ConfirmatoryInternalTestError(
                    f"invalid confirmatory metric ratio: {name}"
                )
            ratios[name] = ratio
            passes[name] = ratio <= 1.0
    fidelity_max = max(ratios[name] for name in FIDELITY_METRICS)
    coherence_max = max(ratios[name] for name in COHERENCE_METRICS)
    return {
        "schema_version": "external-confirmatory-descriptive-score-v1",
        "fidelity_max_ratio": fidelity_max,
        "coherence_max_ratio": coherence_max,
        "combined_score": 0.5 * fidelity_max + 0.5 * coherence_max,
        "metric_ratios": ratios,
        "metric_pass": passes,
        "test_time_selection_performed": False,
    }


@dataclass(frozen=True)
class ConfirmatoryExecutionHooks:
    load_internal_test: Callable[[Mapping[str, Any]], tuple[Any, Any]]
    restore_state: Callable[[Mapping[str, Any], str], Any]
    sample: Callable[[Any, Any, int], Any]
    evaluate: Callable[[Any, Any, Mapping[str, Any]], Mapping[str, Any]]
    persist_sample: Callable[[Path, Any], None] | None = None


def execute_confirmatory_job_core(
    *,
    job: Mapping[str, Any],
    device: str,
    hooks: ConfirmatoryExecutionHooks,
) -> Mapping[str, Any]:
    """Execute the irreversible one-call seam after authorization validation."""

    expected = {
        "attempt": "attempt_001",
        "sample_calls": 1,
        "evaluation_calls": 1,
        "fit_calls": 0,
        "refit_calls": 0,
        "test_time_selection_calls": 0,
    }
    for key, value in expected.items():
        if job.get(key) != value:
            raise ConfirmatoryInternalTestError(f"job scope changed: {key}")
    attempt = Path(job["attempt_path"])
    if attempt.exists():
        raise ConfirmatoryInternalTestError(
            f"confirmatory attempt already exists: {attempt}"
        )
    attempt.parent.mkdir(parents=True, exist_ok=True)
    try:
        attempt.mkdir()
    except FileExistsError as error:
        raise ConfirmatoryInternalTestError(
            f"confirmatory attempt already exists: {attempt}"
        ) from error
    started = time.monotonic()
    manifest = {
        "schema_version": "external-confirmatory-attempt-v1",
        "job": dict(job),
        "device": device,
        "append_only": True,
        "training_calls": 0,
        "refit_calls": 0,
        "test_time_selection_calls": 0,
    }
    _exclusive_json(attempt / "OWNERSHIP.json", manifest)
    _exclusive_json(attempt / "manifest.json", manifest)
    _exclusive_json(
        attempt / "RUNNING.json",
        {"status": "RUNNING", "job_id": job["job_id"]},
    )
    try:
        real, plan = hooks.load_internal_test(job)
        state = hooks.restore_state(job, device)
        synthetic = hooks.sample(
            state, plan, int(job["seed"]) + int(job["sample_seed_offset"])
        )
        if hooks.persist_sample is not None:
            hooks.persist_sample(attempt / "sample.npz", synthetic)
        evidence = hooks.evaluate(real, synthetic, job)
        if not isinstance(evidence, Mapping):
            raise ConfirmatoryInternalTestError("evaluation evidence is not a mapping")
        _exclusive_json(attempt / "evaluation.json", dict(evidence))
        metrics = evidence.get("metrics", {})
        _exclusive_json(
            attempt / "metrics.json",
            metrics if isinstance(metrics, Mapping) else {},
        )
        _exclusive_json(
            attempt / "runtime.json",
            {
                "elapsed_seconds": time.monotonic() - started,
                "training_calls": 0,
                "refit_calls": 0,
                "sample_calls": 1,
                "evaluation_calls": 1,
                "test_time_selection_calls": 0,
                "checkpoint_writes": 0,
            },
        )
        index = {
            str(path.relative_to(attempt)): _sha256_file(path)
            for path in sorted(attempt.rglob("*"))
            if path.is_file()
            and path.name not in {"artifact_index.json", "COMPLETE.json"}
        }
        _exclusive_json(attempt / "artifact_index.json", index)
        complete = {
            "status": "COMPLETE",
            "job_id": job["job_id"],
            "artifact_index_sha256": _sha256_file(
                attempt / "artifact_index.json"
            ),
            "test_time_selection_performed": False,
            "retry_allowed": False,
        }
        _exclusive_json(attempt / "COMPLETE.json", complete)
        return complete
    except BaseException as error:
        terminal = attempt / "FAILED.json"
        if not terminal.exists():
            _exclusive_json(
                terminal,
                {
                    "status": "FAILED",
                    "job_id": job["job_id"],
                    "failure_class": type(error).__name__,
                    "message": str(error),
                    "elapsed_seconds": time.monotonic() - started,
                    "retry_allowed": False,
                },
            )
        raise


def _production_hooks() -> ConfirmatoryExecutionHooks:
    def load(job: Mapping[str, Any]) -> tuple[SequenceBatch, SamplingPlan]:
        real = _load_batch_hash_checked(
            Path(job["internal_test_path"]), str(job["internal_test_sha256"])
        )
        plan = SamplingPlan.from_batch(real)
        if plan.plan_hash != job.get("authorized_conditioning_plan_sha256"):
            raise ConfirmatoryInternalTestError(
                "internal-test conditioning plan hash differs from authorization"
            )
        return real, plan

    def restore(job: Mapping[str, Any], device: str) -> Any:
        return restore_frozen_validation_state(job, device)

    def sample(state: Any, plan: SamplingPlan, seed: int) -> SyntheticBatch:
        return state.sample(plan, seed=seed)

    def evaluate(
        real: SequenceBatch,
        synthetic: SyntheticBatch,
        job: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        thresholds_path = Path(job["train_only_thresholds_path"])
        if _sha256_file(thresholds_path) != job["train_only_thresholds_sha256"]:
            raise ConfirmatoryInternalTestError("train-only threshold hash changed")
        transform_path = Path(job["transform_path"])
        summary_path = Path(job["summary_path"])
        if (
            _sha256_file(transform_path) != job["transform_sha256"]
            or _sha256_file(summary_path) != job["summary_sha256"]
        ):
            raise ConfirmatoryInternalTestError("train-only transform state changed")
        thresholds = _read_json(thresholds_path)
        transform = _read_json(transform_path)
        summary = _read_json(summary_path)
        hard = validate_external_hard_contract(
            real_validation=real,
            synthetic=synthetic,
            gap_cardinality=len(transform["gap_tau"]),
            receiver_cardinality=int(summary["receiver_vocabulary_cardinality"]),
        )
        metrics = compute_external_validation_metrics(
            real=real,
            synthetic=synthetic,
            gap_tau=np.asarray(transform["gap_tau"], dtype=float),
            short_gap_threshold=float(thresholds["short_gap_threshold"]),
        )
        return {
            "schema_version": "external-confirmatory-internal-test-evaluation-v1",
            "dataset": job["dataset"],
            "model": job["model"],
            "hard_validity": hard,
            "metrics": metrics,
            "descriptive_score": _score_without_selection(
                metrics=metrics, thresholds=thresholds
            ),
            "thresholds_sha256": job["train_only_thresholds_sha256"],
            "internal_test_sha256": job["internal_test_sha256"],
            "test_time_selection_performed": False,
        }

    def persist(path: Path, synthetic: SyntheticBatch) -> None:
        _exclusive_npz(
            path,
            {
                "x_num": synthetic.x_num,
                "dt_bin": synthetic.dt_bin,
                "x_cat": synthetic.x_cat,
                "valid_mask": synthetic.valid_mask,
                "y_entity": synthetic.y_entity,
                "lengths": synthetic.lengths,
            },
        )

    return ConfirmatoryExecutionHooks(load, restore, sample, evaluate, persist)


def execute_confirmatory_job(
    *,
    job: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    device: str,
) -> Mapping[str, Any]:
    validate_confirmatory_authorization(plan, authorization)
    expected = next(
        (candidate for candidate in plan["jobs"] if candidate["job_id"] == job["job_id"]),
        None,
    )
    if expected != job:
        raise ConfirmatoryInternalTestError("execution job differs from authorized plan")
    authorized_job = dict(job)
    authorized_job["authorized_conditioning_plan_sha256"] = authorization[
        "conditioning_plan_sha256"
    ][job["dataset"]]
    return execute_confirmatory_job_core(
        job=authorized_job,
        device=device,
        hooks=_production_hooks(),
    )
