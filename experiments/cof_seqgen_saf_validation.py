"""Validation-only generation and frozen-metric evaluation for SAF checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from benchmarks.cof_seqgen_saf_metrics import (
    evaluate_metric_suite,
    evaluate_next_event_utility,
    evaluate_privacy_exposure,
    fit_metric_state,
)
from data.cof_seqgen_saf_tensorizer import (
    SAFTensorizer,
    SAFTensorizerState,
    load_canonical_dataset,
)
from generators.cof_seqgen_saf_baselines import (
    SharedGenerationPlan,
    build_shared_generation_plan,
    validate_raw_generated_events,
)
from generators.cof_seqgen_saf_external_baselines import EmpiricalSequenceSampler
from models.cof_seqgen_saf import (
    MODEL_IMPLEMENTATION_VERSION,
    CoFSeqGenSAF,
    SAFModelConfig,
)


class SAFValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidationConfig:
    dataset_dir: str
    output_dir: str
    checkpoints: Mapping[str, str]
    device: str = "cuda:0"
    generation_seed: int = 20261826
    sample_batch_size: int = 128
    generation_entities: Optional[int] = None
    metric_audit_path: str = (
        "artifacts/cof_seqgen_saf/metric_validity_audit_train_only.json"
    )

    def __post_init__(self) -> None:
        if not self.checkpoints:
            raise SAFValidationError("at least one checkpoint is required")
        if self.sample_batch_size < 1:
            raise SAFValidationError("sample_batch_size must be positive")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise SAFValidationError("CUDA validation requested but unavailable")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _plan_sha256(plan: SharedGenerationPlan) -> str:
    digest = hashlib.sha256()
    for synthetic, source, length in zip(
        plan.entity_ids,
        plan.source_train_entity_ids,
        plan.lengths,
    ):
        digest.update(
            json.dumps(
                [
                    type(synthetic).__name__,
                    str(synthetic),
                    type(source).__name__,
                    str(source),
                    int(length),
                ],
                separators=(",", ":"),
            ).encode()
        )
    digest.update(
        pd.util.hash_pandas_object(plan.static_context, index=False)
        .to_numpy(np.uint64)
        .tobytes()
    )
    return digest.hexdigest()


def _sample_checkpoint(
    *,
    checkpoint_path: Path,
    dataset: Any,
    plan: SharedGenerationPlan,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("test_accessed", True):
        raise SAFValidationError("checkpoint does not certify sealed test content")
    if checkpoint.get("model_implementation_version") != MODEL_IMPLEMENTATION_VERSION:
        raise SAFValidationError(
            "checkpoint implementation version does not match the validation code"
        )
    state = SAFTensorizerState.from_dict(checkpoint["tensorizer_state"])
    tensorizer = SAFTensorizer(state)
    tensorizer._validate_dataset(dataset)
    model_config = SAFModelConfig(**checkpoint["model_config"])
    model = CoFSeqGenSAF(model_config, state.gap_support)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    train_sequences = tensorizer.transform_split(dataset, "train")
    sequence_by_id = {sequence.entity_id: sequence for sequence in train_sequences}
    generated_parts = []
    _seed(seed)
    for start in range(0, len(plan.entity_ids), batch_size):
        stop = min(len(plan.entity_ids), start + batch_size)
        source = plan.source_train_entity_ids[start:stop]
        sequences = [sequence_by_id[entity_id] for entity_id in source]
        static = torch.from_numpy(
            np.stack([sequence.static_numeric for sequence in sequences])
        ).to(device)
        static_categorical = tuple(
            torch.as_tensor(
                [sequence.static_categorical[field] for sequence in sequences],
                dtype=torch.long,
                device=device,
            )
            for field in range(len(sequences[0].static_categorical))
        )
        lengths = plan.lengths[start:stop]
        sampled = model.sample_fixed_lengths(
            lengths,
            static=static,
            static_categorical=static_categorical,
            device=device,
        )
        generated_parts.append(
            tensorizer.decode_generated(
                entity_ids=plan.entity_ids[start:stop],
                gap=sampled["gap"],
                receiver=sampled["receiver"],
                numeric_value=sampled["numeric_value"],
                auxiliary_categorical=sampled["auxiliary_categorical"],
                auxiliary_numeric=sampled["auxiliary_numeric"],
                lengths=lengths,
            )
        )
    generated = pd.concat(generated_parts, ignore_index=True)
    validate_raw_generated_events(generated, plan)
    metadata = {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "candidate_id": model_config.candidate_id,
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "best_epoch": checkpoint["best_epoch"],
        "best_validation_loss": checkpoint["best_validation"],
        "test_accessed": False,
        "loaded_content_splits": ["train", "validation"],
    }
    return generated, metadata


def _evaluate_candidate(
    *,
    train_events: pd.DataFrame,
    validation_events: pd.DataFrame,
    generated: pd.DataFrame,
    metric_state: Any,
    seed: int,
    margins: Mapping[str, float],
) -> Dict[str, Any]:
    metrics = evaluate_metric_suite(
        validation_events,
        generated,
        metric_state,
        include_privacy=False,
    )
    privacy = evaluate_privacy_exposure(train_events, generated)
    utility = evaluate_next_event_utility(
        generated,
        validation_events,
        metric_state,
        seed=seed,
    )
    decisions = {
        endpoint: bool(metrics[endpoint] <= margin)
        for endpoint, margin in margins.items()
        if endpoint in metrics
    }
    return {
        "validation_fidelity": metrics,
        "train_copy_exposure": privacy,
        "validation_tstr_utility": utility,
        "train_only_noninferiority_margins_q95": dict(margins),
        "noninferiority_pass_by_endpoint": decisions,
        "all_calibrated_marginal_structural_endpoints_pass": bool(
            decisions and all(decisions.values())
        ),
    }


def run_validation_comparison(config: ValidationConfig) -> Dict[str, Any]:
    """Sample fixed train-only plans and evaluate only against validation events."""

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_canonical_dataset(
        config.dataset_dir,
        allowed_splits=("train", "validation"),
    )
    checkpoints = {
        candidate: Path(path).resolve()
        for candidate, path in config.checkpoints.items()
    }
    loaded = {
        candidate: torch.load(path, map_location="cpu")
        for candidate, path in checkpoints.items()
    }
    limits = {
        value["training_config"].get("train_entity_limit")
        for value in loaded.values()
    }
    if len(limits) != 1:
        raise SAFValidationError("all ablations must use the same train entity scope")
    train_limit = limits.pop()
    eligible = dataset.entity_ids_for_split("train")
    if train_limit is not None:
        eligible = eligible[: int(train_limit)]
    validation_count = len(dataset.entity_ids_for_split("validation"))
    n_entities = config.generation_entities or validation_count
    plan = build_shared_generation_plan(
        dataset,
        n_entities=n_entities,
        seed=config.generation_seed,
        eligible_entity_ids=eligible,
    )
    plan_hash = _plan_sha256(plan)
    plan_frame = plan.static_context.copy()
    plan_frame.insert(1, "source_train_entity_id", plan.source_train_entity_ids)
    plan_frame.insert(2, "planned_length", plan.lengths)
    plan_frame.to_parquet(output_dir / "shared_generation_plan.parquet", index=False)

    train_ids = set(dataset.entity_ids_for_split("train"))
    validation_ids = set(dataset.entity_ids_for_split("validation"))
    train_events = dataset.events[dataset.events["entity_id"].isin(train_ids)].copy()
    validation_events = dataset.events[
        dataset.events["entity_id"].isin(validation_ids)
    ].copy()
    metric_state = fit_metric_state(train_events)
    audit = json.loads(Path(config.metric_audit_path).read_text(encoding="utf-8"))
    margins = audit["datasets"][dataset.schema.dataset_id][
        "noninferiority_margins_q95"
    ]

    results: Dict[str, Any] = {}
    device = torch.device(config.device)
    for offset, (candidate, checkpoint_path) in enumerate(checkpoints.items()):
        generated, metadata = _sample_checkpoint(
            checkpoint_path=checkpoint_path,
            dataset=dataset,
            plan=plan,
            device=device,
            batch_size=config.sample_batch_size,
            seed=config.generation_seed + 100 + offset,
        )
        generated_path = output_dir / f"{candidate.lower()}_validation_synthetic.parquet"
        generated.to_parquet(generated_path, index=False)
        results[candidate] = {
            **metadata,
            "generated_path": str(generated_path),
            "generated_sha256": _sha256(generated_path),
            **_evaluate_candidate(
                train_events=train_events,
                validation_events=validation_events,
                generated=generated,
                metric_state=metric_state,
                seed=config.generation_seed,
                margins=margins,
            ),
        }

    empirical = EmpiricalSequenceSampler().fit(dataset).sample(plan)
    empirical_path = output_dir / "empirical_sequence_sampler_validation_synthetic.parquet"
    empirical.to_parquet(empirical_path, index=False)
    results["empirical_sequence_sampler"] = {
        "generated_path": str(empirical_path),
        "generated_sha256": _sha256(empirical_path),
        **_evaluate_candidate(
            train_events=train_events,
            validation_events=validation_events,
            generated=empirical,
            metric_state=metric_state,
            seed=config.generation_seed,
            margins=margins,
        ),
    }
    real_upper = evaluate_next_event_utility(
        train_events,
        validation_events,
        metric_state,
        seed=config.generation_seed,
    )
    report = {
        "schema_version": "cof-seqgen-saf-validation-comparison-v1",
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "dataset_id": dataset.schema.dataset_id,
        "schema_sha256": dataset.schema.schema_sha256,
        "split_assignment_sha256": dataset.split_assignment_sha256,
        "shared_generation_plan_sha256": plan_hash,
        "generation_entities": n_entities,
        "generation_source_train_entities": len(eligible),
        "metric_state_fit_split": "train",
        "candidate_selection_split": "validation",
        "loaded_content_splits": ["train", "validation"],
        "test_accessed": False,
        "claim_status": "DEVELOPMENT_VALIDATION_ONLY_NO_TEST_OR_SOTA_CLAIM",
        "real_train_to_validation_utility_upper_benchmark": real_upper,
        "results": results,
    }
    report_path = output_dir / "validation_comparison.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    report["report_sha256"] = _sha256(report_path)
    return report
