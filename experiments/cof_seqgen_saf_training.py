"""Deterministic checkpointed training for CoFSeqGen-SAF development roles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data.cof_seqgen_saf_tensorizer import (
    SAFTensorizer,
    TensorizedSequence,
    load_canonical_dataset,
)
from models.cof_seqgen_saf import (
    MODEL_IMPLEMENTATION_VERSION,
    PAD_CODE,
    CoFSeqGenSAF,
    SAFModelConfig,
)


class SAFTrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: str
    output_dir: str
    candidate_id: str = "SAF-O1"
    seed: int = 20260826
    hidden_dim: int = 128
    gap_embedding_dim: int = 32
    mark_embedding_dim: int = 32
    context_window: int = 128
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    patience: int = 6
    max_positive_gap_states: int = 31
    train_entity_limit: Optional[int] = None
    validation_entity_limit: Optional[int] = None
    device: str = "cpu"
    amp: bool = False
    resume: bool = True

    def __post_init__(self) -> None:
        if self.context_window < 2 or self.batch_size < 1 or self.epochs < 1:
            raise SAFTrainingError("invalid context, batch, or epoch setting")
        if self.learning_rate <= 0 or self.patience < 1:
            raise SAFTrainingError("invalid optimizer or patience setting")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise SAFTrainingError("CUDA was requested but is unavailable")


@dataclass(frozen=True)
class WindowRecord:
    sequence_index: int
    start: int
    end: int
    target_start: int


class SequenceWindowDataset(Dataset):
    """Non-overlapping targets with one-event warm-up across window boundaries."""

    def __init__(
        self,
        sequences: Sequence[TensorizedSequence],
        *,
        context_window: int,
    ) -> None:
        self.sequences = tuple(sequences)
        self.records: list[WindowRecord] = []
        for sequence_index, sequence in enumerate(self.sequences):
            target_start = 0
            while target_start < sequence.length:
                if target_start == 0:
                    start = 0
                    end = min(sequence.length, context_window)
                    local_target_start = 0
                else:
                    start = target_start - 1
                    end = min(sequence.length, target_start + context_window - 1)
                    local_target_start = 1
                self.records.append(
                    WindowRecord(
                        sequence_index,
                        start,
                        end,
                        local_target_start,
                    )
                )
                target_start = end
        if not self.records:
            raise SAFTrainingError("window dataset is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        sequence = self.sequences[record.sequence_index]
        section = slice(record.start, record.end)
        target_mask = np.ones(record.end - record.start, dtype=bool)
        target_mask[: record.target_start] = False
        return {
            "gap": sequence.gap[section],
            "receiver": sequence.receiver[section],
            "numeric_value": sequence.numeric_value[section],
            "auxiliary_categorical": tuple(
                values[section] for values in sequence.auxiliary_categorical
            ),
            "auxiliary_numeric": sequence.auxiliary_numeric[section],
            "static_categorical": sequence.static_categorical,
            "static": sequence.static_numeric,
            "target_mask": target_mask,
        }


def collate_windows(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    batch = len(items)
    steps = max(len(item["gap"]) for item in items)
    auxiliary_categorical_count = len(items[0]["auxiliary_categorical"])
    auxiliary_numeric_dim = items[0]["auxiliary_numeric"].shape[1]
    static_categorical_count = len(items[0]["static_categorical"])
    static_dim = len(items[0]["static"])
    gap = torch.full((batch, steps), float("nan"), dtype=torch.float32)
    receiver = torch.full((batch, steps), PAD_CODE, dtype=torch.long)
    numeric = torch.zeros((batch, steps), dtype=torch.float32)
    valid = torch.zeros((batch, steps), dtype=torch.bool)
    target = torch.zeros((batch, steps), dtype=torch.bool)
    auxiliary_categorical = tuple(
        torch.full((batch, steps), PAD_CODE, dtype=torch.long)
        for _ in range(auxiliary_categorical_count)
    )
    auxiliary_numeric = torch.zeros(
        batch,
        steps,
        auxiliary_numeric_dim,
        dtype=torch.float32,
    )
    static_categorical = tuple(
        torch.zeros(batch, dtype=torch.long)
        for _ in range(static_categorical_count)
    )
    static = torch.zeros(batch, static_dim, dtype=torch.float32)
    for row, item in enumerate(items):
        length = len(item["gap"])
        gap[row, :length] = torch.from_numpy(item["gap"])
        receiver[row, :length] = torch.from_numpy(item["receiver"])
        numeric[row, :length] = torch.from_numpy(item["numeric_value"])
        valid[row, :length] = True
        target[row, :length] = torch.from_numpy(item["target_mask"])
        for field, values in enumerate(item["auxiliary_categorical"]):
            auxiliary_categorical[field][row, :length] = torch.from_numpy(values)
        if auxiliary_numeric_dim:
            auxiliary_numeric[row, :length] = torch.from_numpy(
                item["auxiliary_numeric"]
            )
        for field, value in enumerate(item["static_categorical"]):
            static_categorical[field][row] = int(value)
        if static_dim:
            static[row] = torch.from_numpy(item["static"])
    return {
        "gap": gap,
        "receiver": receiver,
        "numeric_value": numeric,
        "valid_mask": valid,
        "target_mask": target,
        "auxiliary_categorical": auxiliary_categorical,
        "auxiliary_numeric": auxiliary_numeric,
        "static_categorical": static_categorical,
        "static": static,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: (
            tuple(value_item.to(device) for value_item in value)
            if isinstance(value, tuple)
            else value.to(device)
        )
        for key, value in batch.items()
    }


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _evaluate(
    model: CoFSeqGenSAF,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    target_count = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            losses = model.compute_loss(**batch)
            weight = int(batch["target_mask"].sum().item())
            target_count += weight
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value.item()) * weight
    if not target_count:
        raise SAFTrainingError("validation has no target events")
    return {name: value / target_count for name, value in sums.items()}


def train_development_model(config: TrainingConfig) -> Dict[str, Any]:
    """Fit on train and select by validation NLL; test is never transformed."""

    _seed_everything(config.seed)
    dataset_dir = Path(config.dataset_dir).resolve()
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_canonical_dataset(
        dataset_dir,
        allowed_splits=("train", "validation"),
    )
    tensorizer = SAFTensorizer.fit(
        dataset,
        max_positive_gap_states=config.max_positive_gap_states,
    )
    train_sequences = tensorizer.transform_split(
        dataset,
        "train",
        entity_limit=config.train_entity_limit,
    )
    validation_sequences = tensorizer.transform_split(
        dataset,
        "validation",
        entity_limit=config.validation_entity_limit,
    )
    train_windows = SequenceWindowDataset(
        train_sequences,
        context_window=config.context_window,
    )
    validation_windows = SequenceWindowDataset(
        validation_sequences,
        context_window=config.context_window,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_windows,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_windows,
    )
    validation_loader = DataLoader(
        validation_windows,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_windows,
    )
    model_config = SAFModelConfig(
        candidate_id=config.candidate_id,
        hidden_dim=config.hidden_dim,
        gap_embedding_dim=config.gap_embedding_dim,
        mark_embedding_dim=config.mark_embedding_dim,
        context_window=config.context_window,
        max_gap=max(tensorizer.state.gap_support.representatives),
        **tensorizer.state.model_config_kwargs(),
    )
    device = torch.device(config.device)
    model = CoFSeqGenSAF(
        model_config,
        tensorizer.state.gap_support,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, config.patience // 2),
    )
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    checkpoint_path = output_dir / "checkpoint_best.pt"
    latest_path = output_dir / "checkpoint_latest.pt"
    start_epoch = 0
    best_validation = float("inf")
    best_epoch = -1
    history: list[Dict[str, Any]] = []
    stale = 0
    if config.resume and latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device)
        if (
            checkpoint.get("schema_version") != "cof-seqgen-saf-checkpoint-v3"
            or checkpoint.get("model_implementation_version")
            != MODEL_IMPLEMENTATION_VERSION
        ):
            raise SAFTrainingError(
                "checkpoint predates the current model implementation; "
                "resume into a new output directory"
            )
        stored_config = dict(checkpoint["training_config"])
        current_config = asdict(config)
        stored_epochs = int(stored_config.pop("epochs"))
        current_epochs = int(current_config.pop("epochs"))
        if stored_config != current_config or current_epochs < stored_epochs:
            raise SAFTrainingError("resume config differs from checkpoint")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])
        best_epoch = int(checkpoint["best_epoch"])
        history = list(checkpoint["history"])
        stale = int(checkpoint.get("stale", 0))
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        if "loader_generator_state" in checkpoint:
            generator.set_state(checkpoint["loader_generator_state"].cpu())
            random.setstate(checkpoint["python_random_state"])
            np.random.set_state(checkpoint["numpy_random_state"])
            torch.set_rng_state(checkpoint["torch_random_state"].cpu())
            if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in checkpoint["cuda_random_state"]]
                )
    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_sum = 0.0
        target_count = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                losses = model.compute_loss(**batch)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            weight = int(batch["target_mask"].sum().item())
            train_sum += float(losses["loss"].detach().item()) * weight
            target_count += weight
        validation = _evaluate(model, validation_loader, device)
        train_loss = train_sum / max(target_count, 1)
        scheduler.step(validation["loss"])
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        improved = validation["loss"] < best_validation - 1e-8
        if improved:
            best_validation = validation["loss"]
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        checkpoint = {
            "schema_version": "cof-seqgen-saf-checkpoint-v3",
            "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
            "training_config": asdict(config),
            "model_config": asdict(model_config),
            "tensorizer_state": tensorizer.state.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_validation": best_validation,
            "best_epoch": best_epoch,
            "history": history,
            "stale": stale,
            "scaler_state": scaler.state_dict(),
            "loader_generator_state": generator.get_state(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "test_accessed": False,
            "loaded_content_splits": ("train", "validation"),
        }
        _atomic_torch_save(checkpoint, latest_path)
        if improved:
            _atomic_torch_save(checkpoint, checkpoint_path)
        if stale >= config.patience:
            break

    if not checkpoint_path.is_file():
        raise SAFTrainingError("training produced no best checkpoint")
    report = {
        "schema_version": "cof-seqgen-saf-training-report-v1",
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "training_config": asdict(config),
        "model_config": asdict(model_config),
        "dataset_id": dataset.schema.dataset_id,
        "schema_sha256": dataset.schema.schema_sha256,
        "split_assignment_sha256": dataset.split_assignment_sha256,
        "train_entities": len(train_sequences),
        "validation_entities": len(validation_sequences),
        "train_windows": len(train_windows),
        "validation_windows": len(validation_windows),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "history": history,
        "resumed_from_epoch": start_epoch if start_epoch else None,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "test_accessed": False,
        "loaded_content_splits": ["train", "validation"],
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "determinism": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
        },
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "tensorizer_state.json").write_text(
        json.dumps(
            tensorizer.state.to_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report
