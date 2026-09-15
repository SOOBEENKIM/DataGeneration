"""Executable external-baseline adapters for the CoFSeqGen-SAF protocol.

Every adapter fits canonical *training entities only*. Identity, event index,
and cumulative timestamp are protocol structure rather than generated model
features. The modelled temporal variable is the nonnegative inter-event gap;
timestamp is its deterministic cumulative representation with origin zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from data.cof_seqgen_saf_contract import (
    CanonicalEntitySequenceDataset,
    TrainOnlyFitProvenance,
    make_train_only_fit_provenance,
)
from generators.cof_seqgen_saf_baselines import (
    BaselineCompatibilityError,
    SharedGenerationPlan,
    validate_raw_generated_events,
)


PLANNED_LENGTH_COLUMN = "__saf_planned_length"
FIRST_GAP_MODEL_PLACEHOLDER = 0.0


def _save_realtabformer_024(model: Any, path: Path) -> Path:
    """Persist 0.2.4 while containing its Path JSON-serialization defect.

    Version 0.2.4 converts ``checkpoints_dir`` and ``samples_save_dir`` before
    JSON encoding but overlooks ``full_save_dir``.  Temporarily serializing
    that one field as text preserves the upstream artifact format without
    monkey-patching the installed package.
    """

    original_full_save_dir = model.full_save_dir
    model.full_save_dir = str(original_full_save_dir)
    try:
        model.save(path, allow_overwrite=True)
    finally:
        model.full_save_dir = original_full_save_dir
    return path / model.experiment_id


def _train_frames(
    dataset: CanonicalEntitySequenceDataset,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids = set(dataset.entity_ids_for_split("train"))
    parent = dataset.static_context[
        dataset.static_context["entity_id"].isin(train_ids)
    ].copy()
    lengths = dataset.events.groupby("entity_id", sort=False).size()
    parent[PLANNED_LENGTH_COLUMN] = parent["entity_id"].map(lengths).astype(int)
    event_columns = [
        "entity_id",
        "gap",
        "receiver_or_mark",
        "amount_or_numeric_value",
        *dataset.schema.auxiliary_numeric_columns,
        *dataset.schema.auxiliary_categorical_columns,
    ]
    train_events = dataset.events[dataset.events["entity_id"].isin(train_ids)]
    child = train_events[event_columns].copy()
    # Third-party generators generally do not expose a masked likelihood for
    # the structurally undefined first gap. A finite compatibility placeholder
    # prevents them from learning source missingness as an event outcome. It is
    # never written to canonical data and the generated first gap is projected
    # back to missing before evaluation.
    child.loc[
        train_events["event_index"].to_numpy(int) == 0,
        "gap",
    ] = FIRST_GAP_MODEL_PLACEHOLDER
    return parent, child


def _planned_parent(plan: SharedGenerationPlan) -> pd.DataFrame:
    parent = plan.static_context.copy()
    parent[PLANNED_LENGTH_COLUMN] = np.asarray(plan.lengths, dtype=int)
    return parent


def _canonicalize_generated_rows(
    rows: pd.DataFrame,
    *,
    plan: SharedGenerationPlan,
    dataset: CanonicalEntitySequenceDataset,
    require_exact_lengths: bool,
) -> pd.DataFrame:
    """Map model fields to the canonical event view without content repair."""

    modeled = [
        "gap",
        "receiver_or_mark",
        "amount_or_numeric_value",
        *dataset.schema.auxiliary_numeric_columns,
        *dataset.schema.auxiliary_categorical_columns,
    ]
    missing = set(modeled) - set(rows.columns)
    if missing:
        raise BaselineCompatibilityError(
            f"generated baseline rows miss modeled columns: {sorted(missing)}"
        )
    if "entity_id" not in rows:
        raise BaselineCompatibilityError("generated baseline rows miss entity_id")
    unknown_ids = set(rows["entity_id"]) - set(plan.entity_ids)
    if unknown_ids:
        raise BaselineCompatibilityError("generated rows contain unplanned entities")

    output = []
    grouped = {key: value for key, value in rows.groupby("entity_id", sort=False)}
    for entity_id, planned_length in zip(plan.entity_ids, plan.lengths):
        if entity_id not in grouped:
            raise BaselineCompatibilityError(f"baseline omitted entity {entity_id}")
        group = grouped[entity_id].reset_index(drop=True)
        if require_exact_lengths and len(group) != planned_length:
            raise BaselineCompatibilityError(
                f"baseline emitted {len(group)} rows for planned length {planned_length}"
            )
        gap = pd.to_numeric(group["gap"], errors="coerce").to_numpy(float)
        if not len(gap):
            raise BaselineCompatibilityError("zero-length sequence is outside this plan")
        gap[0] = np.nan
        if len(gap) > 1 and (not np.isfinite(gap[1:]).all() or (gap[1:] < 0).any()):
            raise BaselineCompatibilityError("baseline generated invalid non-first gaps")
        timestamp = np.zeros(len(group), dtype=float)
        if len(group) > 1:
            timestamp[1:] = np.cumsum(gap[1:])
        group = group[modeled].copy()
        group.insert(0, "timestamp", timestamp)
        group.insert(0, "event_index", np.arange(len(group), dtype=int))
        group.insert(0, "event_id", [f"{entity_id}-{i}" for i in range(len(group))])
        group.insert(0, "entity_id", entity_id)
        group["gap"] = gap
        output.append(group)
    generated = pd.concat(output, ignore_index=True)
    if require_exact_lengths:
        validate_raw_generated_events(generated, plan)
    return generated


@dataclass(frozen=True)
class BaselineFitRecord:
    baseline_id: str
    provenance: TrainOnlyFitProvenance
    length_mode: str
    package_version: Optional[str]


class EmpiricalSequenceSampler:
    """Train-only whole-trajectory resampling; intentionally exposes copying."""

    baseline_id = "empirical_sequence_sampler"

    def __init__(self) -> None:
        self.dataset: Optional[CanonicalEntitySequenceDataset] = None
        self.fit_record: Optional[BaselineFitRecord] = None

    def fit(self, dataset: CanonicalEntitySequenceDataset) -> "EmpiricalSequenceSampler":
        self.dataset = dataset
        self.fit_record = BaselineFitRecord(
            self.baseline_id,
            make_train_only_fit_provenance(dataset),
            "exact_from_selected_train_trajectory",
            None,
        )
        return self

    def sample(self, plan: SharedGenerationPlan) -> pd.DataFrame:
        if self.dataset is None:
            raise BaselineCompatibilityError("empirical sampler is not fitted")
        train_ids = set(self.dataset.entity_ids_for_split("train"))
        if not set(plan.source_train_entity_ids) <= train_ids:
            raise BaselineCompatibilityError("empirical plan references a non-train entity")
        rows = []
        grouped = {
            key: value
            for key, value in self.dataset.events.groupby("entity_id", sort=False)
        }
        for synthetic_id, source_id in zip(
            plan.entity_ids, plan.source_train_entity_ids
        ):
            group = grouped[source_id].copy().reset_index(drop=True)
            group["entity_id"] = synthetic_id
            group["event_id"] = [f"{synthetic_id}-{i}" for i in range(len(group))]
            rows.append(group)
        generated = pd.concat(rows, ignore_index=True)
        validate_raw_generated_events(generated, plan)
        return generated


class SDVCPARWrapper:
    """SDV CPAR with explicit plan context and per-length sampling.

    SDV's public ``sample`` method samples its own contexts. Under the pinned
    SDV API, ``_sample`` is used deliberately after transforming the supplied
    plan context. This private-API dependency is locked and smoke-tested.
    """

    baseline_id = "cpar"

    def __init__(
        self,
        *,
        epochs: int = 128,
        sample_size: int = 1,
        cuda: bool = True,
        verbose: bool = False,
    ) -> None:
        self.epochs = epochs
        self.sample_size = sample_size
        self.cuda = cuda
        self.verbose = verbose
        self.dataset: Optional[CanonicalEntitySequenceDataset] = None
        self.model: Any = None
        self.fit_record: Optional[BaselineFitRecord] = None
        self._columns: list[str] = []
        self._context_columns: list[str] = []

    def fit(self, dataset: CanonicalEntitySequenceDataset) -> "SDVCPARWrapper":
        import sdv
        from sdv.metadata import SingleTableMetadata
        from sdv.sequential import PARSynthesizer

        parent, child = _train_frames(dataset)
        data = child.merge(parent, on="entity_id", validate="many_to_one")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        metadata.update_column("entity_id", sdtype="id")
        metadata.set_sequence_key("entity_id")
        context_columns = [column for column in parent if column != "entity_id"]
        model = PARSynthesizer(
            metadata,
            context_columns=context_columns,
            epochs=self.epochs,
            sample_size=self.sample_size,
            cuda=self.cuda,
            verbose=self.verbose,
        )
        model.fit(data)
        self.dataset = dataset
        self.model = model
        self._columns = list(data.columns)
        self._context_columns = context_columns
        self.fit_record = BaselineFitRecord(
            self.baseline_id,
            make_train_only_fit_provenance(dataset),
            "exact_private_pinned_api",
            sdv.__version__,
        )
        return self

    def _processed_context(self, parent: pd.DataFrame) -> pd.DataFrame:
        stub = pd.DataFrame(index=parent.index, columns=self._columns)
        for column in parent:
            stub[column] = parent[column].to_numpy()
        processed = self.model._data_processor.transform(stub)
        return processed[["entity_id", *self._context_columns]]

    def sample(self, plan: SharedGenerationPlan) -> pd.DataFrame:
        if self.model is None or self.dataset is None:
            raise BaselineCompatibilityError("CPAR wrapper is not fitted")
        parent = _planned_parent(plan)
        processed = self._processed_context(parent)
        pieces = []
        for length in sorted(set(plan.lengths)):
            mask = np.asarray(plan.lengths) == length
            pieces.append(self.model._sample(processed.loc[mask], int(length)))
        raw = pd.concat(pieces, ignore_index=True)
        return _canonicalize_generated_rows(
            raw,
            plan=plan,
            dataset=self.dataset,
            require_exact_lengths=True,
        )


class SDVFlatControlWrapper:
    """CTGAN, TVAE, or GaussianCopula as explicitly flattened controls."""

    ALLOWED = {"ctgan", "tvae", "gaussian_copula"}

    def __init__(self, baseline_id: str, **model_kwargs: Any) -> None:
        if baseline_id not in self.ALLOWED:
            raise BaselineCompatibilityError(f"unknown SDV flat control {baseline_id}")
        self.baseline_id = baseline_id
        self.model_kwargs = dict(model_kwargs)
        self.dataset: Optional[CanonicalEntitySequenceDataset] = None
        self.model: Any = None
        self.fit_record: Optional[BaselineFitRecord] = None
        self._model_columns: list[str] = []

    def fit(self, dataset: CanonicalEntitySequenceDataset) -> "SDVFlatControlWrapper":
        import sdv
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import (
            CTGANSynthesizer,
            GaussianCopulaSynthesizer,
            TVAESynthesizer,
        )

        _, child = _train_frames(dataset)
        data = child.drop(columns=["entity_id"]).copy()
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        classes = {
            "ctgan": CTGANSynthesizer,
            "tvae": TVAESynthesizer,
            "gaussian_copula": GaussianCopulaSynthesizer,
        }
        model = classes[self.baseline_id](metadata, **self.model_kwargs)
        model.fit(data)
        self.dataset = dataset
        self.model = model
        self._model_columns = list(data.columns)
        self.fit_record = BaselineFitRecord(
            self.baseline_id,
            make_train_only_fit_provenance(dataset),
            "protocol_partition_not_learned",
            sdv.__version__,
        )
        return self

    def sample(self, plan: SharedGenerationPlan) -> pd.DataFrame:
        if self.model is None or self.dataset is None:
            raise BaselineCompatibilityError("flat wrapper is not fitted")
        raw = self.model.sample(num_rows=int(sum(plan.lengths))).copy()
        raw = raw[self._model_columns]
        raw.insert(
            0,
            "entity_id",
            np.repeat(np.asarray(plan.entity_ids, dtype=object), plan.lengths),
        )
        return _canonicalize_generated_rows(
            raw,
            plan=plan,
            dataset=self.dataset,
            require_exact_lengths=True,
        )


class REaLTabFormerWrapper:
    """Pinned REaLTabFormer parent/child wrapper with exact related counts."""

    baseline_id = "realtabformer"

    def __init__(
        self,
        *,
        workspace_dir: str | Path,
        epochs: int = 100,
        batch_size: int = 8,
        seed: int = 20260826,
        output_max_length: Optional[int] = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.output_max_length = output_max_length
        self.dataset: Optional[CanonicalEntitySequenceDataset] = None
        self.parent_model: Any = None
        self.child_model: Any = None
        self.fit_record: Optional[BaselineFitRecord] = None
        self.model_artifacts: Dict[str, Path] = {}

    def fit(
        self,
        dataset: CanonicalEntitySequenceDataset,
        *,
        device: str = "cuda",
        fit_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> "REaLTabFormerWrapper":
        import realtabformer
        from realtabformer import REaLTabFormer

        parent, child = _train_frames(dataset)
        parent_features = parent.drop(columns=["entity_id"])
        common = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.seed,
            # External logging must never make a benchmark run interactive or
            # introduce an untracked network-side dependency.
            "report_to": [],
        }
        parent_model = REaLTabFormer(
            model_type="tabular",
            checkpoints_dir=str(self.workspace_dir / "parent_checkpoints"),
            samples_save_dir=str(self.workspace_dir / "parent_samples"),
            full_save_dir=str(self.workspace_dir / "parent_full"),
            **common,
        )
        parent_model.fit(parent_features, device=device, **dict(fit_kwargs or {}))
        parent_root = self.workspace_dir / "parent_model"
        parent_path = _save_realtabformer_024(parent_model, parent_root)
        child_model = REaLTabFormer(
            model_type="relational",
            parent_realtabformer_path=parent_path,
            output_max_length=self.output_max_length,
            checkpoints_dir=str(self.workspace_dir / "child_checkpoints"),
            samples_save_dir=str(self.workspace_dir / "child_samples"),
            full_save_dir=str(self.workspace_dir / "child_full"),
            **common,
        )
        child_model.fit(
            df=child,
            in_df=parent,
            join_on="entity_id",
            device=device,
            **dict(fit_kwargs or {}),
        )
        child_root = self.workspace_dir / "child_model"
        child_path = _save_realtabformer_024(child_model, child_root)
        self.dataset = dataset
        self.parent_model = parent_model
        self.child_model = child_model
        self.model_artifacts = {
            "parent": parent_path,
            "child": child_path,
        }
        self.fit_record = BaselineFitRecord(
            self.baseline_id,
            make_train_only_fit_provenance(dataset),
            "exact_related_num_context_column",
            realtabformer.__version__,
        )
        return self

    def sample(
        self,
        plan: SharedGenerationPlan,
        *,
        device: str = "cuda",
        gen_batch: int = 64,
    ) -> pd.DataFrame:
        if self.child_model is None or self.dataset is None:
            raise BaselineCompatibilityError("REaLTabFormer wrapper is not fitted")
        parent = _planned_parent(plan)
        raw = self.child_model.sample(
            input_unique_ids=parent["entity_id"],
            input_df=parent.drop(columns=["entity_id"]),
            related_num=PLANNED_LENGTH_COLUMN,
            gen_batch=gen_batch,
            device=device,
        )
        # REaLTabFormer 0.2.4 returns the relational join key as its repeated
        # DataFrame index rather than as a column.  Promote only a fully
        # recognized planned-ID index; unknown identifiers still fail closed.
        if "entity_id" not in raw.columns:
            index_values = set(raw.index.tolist())
            if not index_values or not index_values <= set(plan.entity_ids):
                raise BaselineCompatibilityError(
                    "REaLTabFormer output has neither an entity_id column nor "
                    "a recognized planned-entity index"
                )
            raw = raw.rename_axis("entity_id").reset_index()
        return _canonicalize_generated_rows(
            raw,
            plan=plan,
            dataset=self.dataset,
            require_exact_lengths=True,
        )


class TabularARGNWrapper:
    """MOSTLY AI TabularARGN wrapper; native length is never silently repaired."""

    baseline_id = "tabularargn"

    def __init__(
        self,
        *,
        workspace_dir: str | Path,
        max_epochs: int = 100,
        seed: int = 20260826,
        device: str = "cuda",
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.max_epochs = max_epochs
        self.seed = seed
        self.device = device
        self.dataset: Optional[CanonicalEntitySequenceDataset] = None
        self.model: Any = None
        self.fit_record: Optional[BaselineFitRecord] = None
        self.model_artifacts: Dict[str, Path] = {}

    def fit(self, dataset: CanonicalEntitySequenceDataset) -> "TabularARGNWrapper":
        import mostlyai.engine
        from mostlyai.engine import TabularARGN

        parent, child = _train_frames(dataset)
        model = TabularARGN(
            tgt_context_key="entity_id",
            ctx_primary_key="entity_id",
            ctx_data=parent,
            max_epochs=self.max_epochs,
            workspace_dir=self.workspace_dir,
            random_state=self.seed,
            device=self.device,
            verbose=1,
        )
        model.fit(child)
        self.dataset = dataset
        self.model = model
        self.model_artifacts = {"workspace": self.workspace_dir}
        self.fit_record = BaselineFitRecord(
            self.baseline_id,
            make_train_only_fit_provenance(dataset),
            "native_conditioned_on_planned_length_not_forced",
            getattr(mostlyai.engine, "__version__", "2.6.2"),
        )
        return self

    def sample(self, plan: SharedGenerationPlan) -> pd.DataFrame:
        if self.model is None or self.dataset is None:
            raise BaselineCompatibilityError("TabularARGN wrapper is not fitted")
        parent = _planned_parent(plan)
        raw = self.model.sample(ctx_data=parent, device=self.device)
        return _canonicalize_generated_rows(
            raw,
            plan=plan,
            dataset=self.dataset,
            require_exact_lengths=False,
        )


class TabDiTUnavailable:
    baseline_id = "tabdit"

    def fit(self, dataset: CanonicalEntitySequenceDataset) -> None:
        del dataset
        raise BaselineCompatibilityError(
            "The pinned official TabDiT repository contains evaluation code but "
            "no generator architecture, training loop, or sampling implementation. "
            "It is reported-only until executable upstream code is obtained."
        )
