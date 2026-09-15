"""Fit an external baseline on train and evaluate generated data on validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, Mapping, Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from benchmarks.cof_seqgen_saf_metrics import fit_metric_state
from data.cof_seqgen_saf_tensorizer import load_canonical_dataset
from experiments.cof_seqgen_saf_validation import (
    _evaluate_candidate,
    _plan_sha256,
    _sha256,
)
from generators.cof_seqgen_saf_baselines import build_shared_generation_plan
from generators.cof_seqgen_saf_external_baselines import (
    EmpiricalSequenceSampler,
    REaLTabFormerWrapper,
    SDVCPARWrapper,
    SDVFlatControlWrapper,
    TabularARGNWrapper,
)


class ExternalValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExternalValidationConfig:
    dataset_dir: str
    output_dir: str
    baseline_id: str
    fit_seed: int = 20260826
    generation_seed: int = 20261826
    generation_entities: Optional[int] = None
    train_entity_limit: Optional[int] = None
    model_kwargs: Optional[Mapping[str, Any]] = None
    fit_kwargs: Optional[Mapping[str, Any]] = None
    fit_device: str = "cuda"
    sample_device: str = "cuda"
    metric_audit_path: str = (
        "artifacts/cof_seqgen_saf/metric_validity_audit_train_only.json"
    )


def _seed_everything(seed: int) -> None:
    """Reset every RNG used by the executable baseline adapters.

    Fit and sampling use separate seeds so changing the generation plan does
    not silently change the fitted baseline.  Deterministic algorithms are a
    hard gate: an implementation that cannot satisfy them must fail visibly.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _sha256_tree(path: Path) -> str:
    """Content hash for either one model file or a persisted model directory."""

    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise ExternalValidationError(f"model artifact does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ExternalValidationError(f"model artifact directory is empty: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fit_wrapper(config: ExternalValidationConfig, dataset: Any, output_dir: Path) -> Any:
    kwargs = dict(config.model_kwargs or {})
    if config.baseline_id == "empirical_sequence_sampler":
        return EmpiricalSequenceSampler().fit(dataset)
    if config.baseline_id == "cpar":
        return SDVCPARWrapper(**kwargs).fit(dataset)
    if config.baseline_id in SDVFlatControlWrapper.ALLOWED:
        return SDVFlatControlWrapper(config.baseline_id, **kwargs).fit(dataset)
    if config.baseline_id == "realtabformer":
        wrapper = REaLTabFormerWrapper(
            workspace_dir=output_dir / "workspace",
            **kwargs,
        )
        return wrapper.fit(
            dataset,
            device=config.fit_device,
            fit_kwargs=config.fit_kwargs,
        )
    if config.baseline_id == "tabularargn":
        wrapper = TabularARGNWrapper(
            workspace_dir=output_dir / "workspace",
            device=config.fit_device,
            **kwargs,
        )
        return wrapper.fit(dataset)
    raise ExternalValidationError(f"unsupported executable baseline {config.baseline_id}")


def _sample(wrapper: Any, config: ExternalValidationConfig, plan: Any) -> pd.DataFrame:
    if config.baseline_id == "realtabformer":
        return wrapper.sample(plan, device=config.sample_device)
    return wrapper.sample(plan)


def run_external_validation(config: ExternalValidationConfig) -> Dict[str, Any]:
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_canonical_dataset(
        config.dataset_dir,
        allowed_splits=("train", "validation"),
    )
    train_ids = dataset.entity_ids_for_split("train")
    if config.train_entity_limit is not None:
        # A pilot subset is explicit in the report and never presented as a
        # full comparator. Constructing a smaller canonical object is avoided;
        # wrappers therefore require full train for confirmatory runs.
        raise ExternalValidationError(
            "train_entity_limit is not implemented for external fits; use full train"
        )
    n_entities = config.generation_entities or len(
        dataset.entity_ids_for_split("validation")
    )
    plan = build_shared_generation_plan(
        dataset,
        n_entities=n_entities,
        seed=config.generation_seed,
        eligible_entity_ids=train_ids,
    )
    plan_frame = plan.static_context.copy()
    plan_frame.insert(1, "source_train_entity_id", plan.source_train_entity_ids)
    plan_frame.insert(2, "planned_length", plan.lengths)
    plan_path = output_dir / "shared_generation_plan.parquet"
    plan_frame.to_parquet(plan_path, index=False)

    _seed_everything(config.fit_seed)
    fit_start = time.perf_counter()
    wrapper = _fit_wrapper(config, dataset, output_dir)
    fit_seconds = time.perf_counter() - fit_start
    model_path = None
    if getattr(wrapper, "model", None) is not None and hasattr(wrapper.model, "save"):
        model_path = output_dir / "model.pkl"
        wrapper.model.save(model_path)
    model_artifacts = {
        name: {
            "path": str(Path(path).resolve()),
            "sha256": _sha256_tree(Path(path)),
        }
        for name, path in dict(getattr(wrapper, "model_artifacts", {})).items()
    }
    if model_path is not None:
        model_artifacts["model"] = {
            "path": str(model_path),
            "sha256": _sha256(model_path),
        }

    _seed_everything(config.generation_seed)
    sample_start = time.perf_counter()
    generated = _sample(wrapper, config, plan)
    sample_seconds = time.perf_counter() - sample_start
    generated_path = output_dir / "validation_synthetic.parquet"
    generated.to_parquet(generated_path, index=False)

    train_set = set(train_ids)
    validation_set = set(dataset.entity_ids_for_split("validation"))
    train_events = dataset.events[dataset.events["entity_id"].isin(train_set)]
    validation_events = dataset.events[
        dataset.events["entity_id"].isin(validation_set)
    ]
    metric_state = fit_metric_state(train_events)
    audit = json.loads(Path(config.metric_audit_path).read_text(encoding="utf-8"))
    margins = audit["datasets"][dataset.schema.dataset_id][
        "noninferiority_margins_q95"
    ]
    evaluation = _evaluate_candidate(
        train_events=train_events,
        validation_events=validation_events,
        generated=generated,
        metric_state=metric_state,
        seed=config.generation_seed,
        margins=margins,
    )
    fit_record = getattr(wrapper, "fit_record", None)
    report = {
        "schema_version": "cof-seqgen-saf-external-validation-v1",
        "baseline_id": config.baseline_id,
        "dataset_id": dataset.schema.dataset_id,
        "schema_sha256": dataset.schema.schema_sha256,
        "split_assignment_sha256": dataset.split_assignment_sha256,
        "shared_generation_plan_sha256": _plan_sha256(plan),
        "generation_entities": n_entities,
        "fit_seed": config.fit_seed,
        "generation_seed": config.generation_seed,
        "determinism": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "fit_entity_count": len(train_ids),
        "fit_record": asdict(fit_record) if fit_record is not None else None,
        "fit_seconds": fit_seconds,
        "sample_seconds": sample_seconds,
        "model_path": str(model_path) if model_path else None,
        "model_sha256": _sha256(model_path) if model_path else None,
        "model_artifacts": model_artifacts,
        "generated_path": str(generated_path),
        "generated_sha256": _sha256(generated_path),
        "evaluation": evaluation,
        "loaded_content_splits": ["train", "validation"],
        "test_accessed": False,
        "claim_status": "DEVELOPMENT_VALIDATION_ONLY",
    }
    report_path = output_dir / "external_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    report["report_sha256"] = _sha256(report_path)
    return report
