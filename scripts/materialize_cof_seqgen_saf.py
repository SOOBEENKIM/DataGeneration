"""Materialize canonical SAF datasets, entity splits, transforms, and gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from data.cof_seqgen_saf_adapters import ADAPTERS, AdapterResult, stable_hash_rank
from data.cof_seqgen_saf_contract import (
    CanonicalEntitySequenceDataset,
    CanonicalSchema,
    RESERVED_CODES,
    make_train_only_fit_provenance,
)


class MaterializationError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _typed_records(values: Iterable[Any]) -> list[dict[str, str]]:
    return [
        {"type": type(value).__name__, "value": str(value)}
        for value in sorted(values, key=lambda value: (type(value).__name__, str(value)))
    ]


def _typed_hash(values: Iterable[Any]) -> str:
    payload = json.dumps(
        _typed_records(values), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fit_train_transforms(
    canonical: CanonicalEntitySequenceDataset,
    output_dir: Path,
) -> dict[str, Any]:
    train_ids = set(canonical.entity_ids_for_split("train"))
    train = canonical.events[canonical.events["entity_id"].isin(train_ids)].copy()
    nontrain = canonical.events[~canonical.events["entity_id"].isin(train_ids)]
    categorical_columns = [
        "receiver_or_mark",
        *canonical.schema.auxiliary_categorical_columns,
    ]
    vocabularies: dict[str, Any] = {}
    unseen: dict[str, Any] = {}
    for column in categorical_columns:
        values = train[column].dropna().unique().tolist()
        ordered = sorted(values, key=lambda value: (type(value).__name__, str(value)))
        records = [
            {"code": index + RESERVED_CODES.first_learned_code, "type": type(value).__name__, "value": str(value)}
            for index, value in enumerate(ordered)
        ]
        vocabulary_keys = {(item["type"], item["value"]) for item in records}
        nontrain_observed = nontrain[column].dropna().tolist()
        unseen_count = sum(
            (type(value).__name__, str(value)) not in vocabulary_keys
            for value in nontrain_observed
        )
        vocabularies[column] = {
            "fit_split": "train",
            "reserved_codes": asdict(RESERVED_CODES),
            "learned_count": len(records),
            "learned_values": records,
        }
        unseen[column] = {
            "nontrain_observed_count": len(nontrain_observed),
            "mapped_to_unk_count": int(unseen_count),
        }
    _write_json(output_dir / "vocabularies.json", vocabularies)

    train_gap = pd.to_numeric(train["gap"], errors="coerce").dropna()
    gap_counts = train_gap.value_counts(dropna=False).sort_index().rename_axis("gap").reset_index(name="count")
    _write_parquet(output_dir / "gap_support_counts.parquet", gap_counts)
    probabilities = np.asarray([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    quantiles = np.quantile(train_gap.to_numpy(float), probabilities)
    gap_state = {
        "fit_split": "train",
        "observed_count": int(len(train_gap)),
        "unique_support_count": int(train_gap.nunique()),
        "minimum": float(train_gap.min()),
        "maximum": float(train_gap.max()),
        "zero_count": int((train_gap == 0).sum()),
        "quantiles": {str(probability): float(value) for probability, value in zip(probabilities, quantiles)},
        "support_counts_sha256": sha256_file(output_dir / "gap_support_counts.parquet"),
    }
    _write_json(output_dir / "gap_state.json", gap_state)

    numeric_columns = [
        "amount_or_numeric_value",
        *canonical.schema.auxiliary_numeric_columns,
    ]
    numeric_state: dict[str, Any] = {"fit_split": "train", "columns": {}}
    for column in numeric_columns:
        values = pd.to_numeric(train[column], errors="coerce").dropna().to_numpy(float)
        numeric_state["columns"][column] = {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "std_population": float(values.std(ddof=0)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "quantiles": {
                str(probability): float(value)
                for probability, value in zip(probabilities, np.quantile(values, probabilities))
            },
        }
    _write_json(output_dir / "numeric_state.json", numeric_state)
    provenance = asdict(make_train_only_fit_provenance(canonical))
    _write_json(output_dir / "train_only_provenance.json", provenance)
    return {
        "fit_split": "train",
        "train_event_count": int(len(train)),
        "vocabulary_columns": categorical_columns,
        "unseen_nontrain": unseen,
        "gap_state": gap_state,
        "numeric_columns": numeric_columns,
        "provenance": provenance,
    }


def _sequence_report(canonical: CanonicalEntitySequenceDataset) -> dict[str, Any]:
    split_map = canonical.entity_splits.set_index("entity_id")["split"]
    lengths = canonical.events.groupby("entity_id", sort=True).size().rename("length")
    frame = lengths.to_frame().assign(split=lengths.index.map(split_map))
    report: dict[str, Any] = {
        "entity_count": int(len(lengths)),
        "event_count": int(len(canonical.events)),
        "split_assignment_sha256": canonical.split_assignment_sha256,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        values = frame.loc[frame["split"] == split, "length"]
        report["splits"][split] = {
            "entity_count": int(len(values)),
            "event_count": int(values.sum()),
            "length_min": int(values.min()),
            "length_mean": float(values.mean()),
            "length_median": float(values.median()),
            "length_max": int(values.max()),
        }
    gaps = pd.to_numeric(canonical.events["gap"], errors="coerce")
    report["gap"] = {
        "missing_count": int(gaps.isna().sum()),
        "zero_count": int((gaps == 0).sum()),
        "negative_count": int((gaps.dropna() < 0).sum()),
        "minimum_nonmissing": float(gaps.min()),
        "maximum_nonmissing": float(gaps.max()),
    }
    return report


def _write_split_ids(canonical: CanonicalEntitySequenceDataset, output_dir: Path) -> dict[str, Any]:
    result = {}
    for split in ("train", "validation", "test"):
        values = canonical.entity_ids_for_split(split)
        records = _typed_records(values)
        _write_json(output_dir / f"{split}.json", {"entities": records})
        result[split] = {"count": len(values), "typed_identity_sha256": _typed_hash(values)}
    return result


def _schema_dict(schema: CanonicalSchema) -> dict[str, Any]:
    return {
        "schema_version": schema.schema_version,
        "schema_sha256": schema.schema_sha256,
        "dataset_id": schema.dataset_id,
        "time_representation": schema.time_representation,
        "timestamp_unit": schema.timestamp_unit,
        "static_context_columns": list(schema.static_context_columns),
        "auxiliary_numeric_columns": list(schema.auxiliary_numeric_columns),
        "auxiliary_categorical_columns": list(schema.auxiliary_categorical_columns),
        "reserved_codes": asdict(RESERVED_CODES),
    }


def _materialize_result(
    result: AdapterResult,
    output_dir: Path,
    *,
    source_manifest: Path,
    adapter_source: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = result.canonical
    _write_parquet(output_dir / "static_context.parquet", canonical.static_context)
    _write_parquet(output_dir / "events.parquet", canonical.events)
    _write_parquet(output_dir / "entity_splits.parquet", canonical.entity_splits)
    oracle_gate = True
    if result.oracle_events is not None:
        oracle = result.oracle_events
        required = {"entity_id", "event_id", "event_index"}
        if not required.issubset(oracle.columns):
            raise MaterializationError(
                f"{result.dataset_id} oracle events lack identity columns"
            )
        identity_columns = ["entity_id", "event_id", "event_index"]
        oracle_gate = len(oracle) == len(canonical.events) and oracle[
            identity_columns
        ].reset_index(drop=True).equals(
            canonical.events[identity_columns].reset_index(drop=True)
        )
        if not oracle_gate:
            raise MaterializationError(
                f"{result.dataset_id} oracle/canonical event identities differ"
            )
        _write_parquet(output_dir / "oracle_latents.parquet", oracle)
    mark_context_gate = True
    if result.mark_context is not None:
        mark_context = result.mark_context
        if (
            "receiver_or_mark" not in mark_context.columns
            or mark_context["receiver_or_mark"].isna().any()
            or mark_context["receiver_or_mark"].duplicated().any()
        ):
            raise MaterializationError(
                f"{result.dataset_id} mark context identity is invalid"
            )
        observed_marks = set(canonical.events["receiver_or_mark"].dropna().tolist())
        context_marks = set(mark_context["receiver_or_mark"].tolist())
        mark_context_gate = observed_marks.issubset(context_marks)
        if not mark_context_gate:
            raise MaterializationError(
                f"{result.dataset_id} mark context misses observed event marks"
            )
        _write_parquet(output_dir / "mark_context.parquet", mark_context)
    _write_json(output_dir / "schema.json", _schema_dict(canonical.schema))
    split_ids = _write_split_ids(canonical, output_dir / "split_ids")
    transform = _fit_train_transforms(canonical, output_dir / "train_transforms")
    sequences = _sequence_report(canonical)
    gates = {
        "entity_leakage_zero": result.audit["entity_leakage"] == 0,
        "timestamp_parsing_error_zero": result.audit["timestamp_parse_errors"] == 0,
        "invalid_negative_gap_zero": result.audit["negative_gap_count"] == 0,
        "first_gap_missing_exactly_once_per_entity": (
            result.audit["first_gap_missing_count"] == result.audit["canonical_entities"]
        ),
        "train_only_vocabulary_verified": transform["fit_split"] == "train",
        "sequence_report_generated": True,
        "source_manifest_present": source_manifest.is_file(),
        "oracle_identity_alignment_verified": oracle_gate,
        "mark_context_coverage_verified": mark_context_gate,
    }
    if not all(gates.values()):
        raise MaterializationError(f"{result.dataset_id} failed gates: {gates}")
    report = {
        "schema_version": "cof-seqgen-saf-dataset-report-v1",
        "dataset": result.dataset_id,
        "audit": result.audit,
        "stratification": result.stratification,
        "split_ids": split_ids,
        "sequence_report": sequences,
        "train_transform_summary": transform,
        "gates": gates,
    }
    _write_json(output_dir / "dataset_report.json", report)
    relative_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "canonical_manifest.json"
    )
    manifest = {
        "schema_version": "cof-seqgen-saf-canonical-manifest-v1",
        "dataset": result.dataset_id,
        "source_acquisition_manifest_sha256": sha256_file(source_manifest),
        "adapter_source_sha256": sha256_file(adapter_source),
        "materializer_source_sha256": sha256_file(Path(__file__).resolve()),
        "canonical_contract_source_sha256": sha256_file(
            Path(__file__).resolve().parents[1]
            / "data"
            / "cof_seqgen_saf_contract.py"
        ),
        "files": {
            str(path.relative_to(output_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in relative_files
        },
    }
    _write_json(output_dir / "canonical_manifest.json", manifest)
    return report


def _nested_berka_900(
    result: AdapterResult,
    scope: dict[str, Any],
) -> AdapterResult:
    selected: list[Any] = []
    for split, count in scope["per_split_counts"].items():
        candidates = list(result.canonical.entity_ids_for_split(split))
        ranked = stable_hash_rank(
            candidates,
            seed=int(scope["selection_seed"]),
            namespace=f"cofseqgen-saf-berka-900-{split}",
        )
        selected.extend(ranked[: int(count)])
    selected_set = set(selected)
    static = result.canonical.static_context[
        result.canonical.static_context["entity_id"].isin(selected_set)
    ].reset_index(drop=True)
    events = result.canonical.events[
        result.canonical.events["entity_id"].isin(selected_set)
    ].reset_index(drop=True)
    splits = result.canonical.entity_splits[
        result.canonical.entity_splits["entity_id"].isin(selected_set)
    ].reset_index(drop=True)
    parent = result.canonical.schema
    schema = CanonicalSchema(
        dataset_id="berka_nested_900",
        time_representation=parent.time_representation,
        timestamp_unit=parent.timestamp_unit,
        static_context_columns=parent.static_context_columns,
        auxiliary_numeric_columns=parent.auxiliary_numeric_columns,
        auxiliary_categorical_columns=parent.auxiliary_categorical_columns,
    )
    canonical = CanonicalEntitySequenceDataset(
        schema=schema,
        static_context=static,
        events=events,
        entity_splits=splits,
    )
    audit = dict(result.audit)
    audit.update(
        {
            "canonical_entities": len(static),
            "canonical_events": len(events),
            "first_gap_missing_count": int(
                events.loc[events["event_index"] == 0, "gap"].isna().sum()
            ),
            "split_assignment_sha256": canonical.split_assignment_sha256,
            "selection_identity_status": scope["identity_status"],
            "seq2synth_comparison_status": scope["seq2synth_comparison_status"],
            "selected_entity_ids_sha256": _typed_hash(selected),
            "seq2synth_reported_entity_count": int(
                scope["seq2synth_reported_entity_count"]
            ),
            "seq2synth_reported_child_rows": int(
                scope["seq2synth_reported_child_rows"]
            ),
            "tabdit_tau_max": int(scope["tabdit_tau_max"]),
            "exact_tabdit_test_entity_ids_publicly_available": False,
            "sequence_truncation_applied": False,
        }
    )
    stratification = {
        "strategy": "stable_hash_nested_within_primary_split",
        "selection_seed": int(scope["selection_seed"]),
        "per_split_counts": {
            key: int(value) for key, value in scope["per_split_counts"].items()
        },
        "parent_split_assignment_sha256": result.canonical.split_assignment_sha256,
    }
    return AdapterResult("berka_nested_900", canonical, audit, stratification)


def materialize(
    repository_root: Path,
    config_path: Path,
    dataset_ids: Iterable[str],
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "cof-seqgen-saf-dataset-materialization-v1":
        raise MaterializationError("unsupported dataset materialization config")
    requested = list(dataset_ids)
    unknown = sorted(set(requested) - set(config["datasets"]))
    if unknown:
        raise MaterializationError(f"unknown datasets: {unknown}")
    runtime_root = repository_root / config["runtime_root"]
    source_manifest_root = runtime_root / "manifests" / "acquisition"
    canonical_root = runtime_root / "canonical"
    adapter_source = repository_root / "data" / "cof_seqgen_saf_adapters.py"
    reports = {}
    for dataset_id in requested:
        spec = config["datasets"][dataset_id]
        raw_dir = repository_root / spec["raw_dir"]
        adapter = ADAPTERS[spec["adapter"]]
        if dataset_id == "controlled_coupling_dgp":
            generator_config_path = repository_root / spec["generator_config"]
            generator_config = yaml.safe_load(
                generator_config_path.read_text(encoding="utf-8")
            )
            for cell in spec["cells"]:
                result = adapter(
                    raw_dir,
                    generator_config=generator_config,
                    scenario=cell["scenario"],
                    kappa=float(cell["kappa"]),
                    n_entities=int(spec["n_entities"]),
                    generation_seed=int(spec["generation_seed"]),
                    split_seed=int(config["split"]["seed"]),
                    cell_id=cell["id"],
                )
                reports[result.dataset_id] = _materialize_result(
                    result,
                    canonical_root / result.dataset_id,
                    source_manifest=source_manifest_root
                    / "controlled_coupling_dgp.json",
                    adapter_source=adapter_source,
                )
            continue
        kwargs = {"split_seed": int(config["split"]["seed"])}
        if dataset_id == "hm":
            kwargs.update(
                {
                    "sample_seed": int(spec["sample_seed"]),
                    "customer_count": int(spec["customer_count"]),
                }
            )
        result = adapter(raw_dir, **kwargs)
        reports[dataset_id] = _materialize_result(
            result,
            canonical_root / dataset_id,
            source_manifest=source_manifest_root / f"{dataset_id}.json",
            adapter_source=adapter_source,
        )
        if dataset_id == "berka":
            scope = spec["auxiliary_scopes"]["deterministic_nested_900"]
            subset = _nested_berka_900(result, scope)
            reports[subset.dataset_id] = _materialize_result(
                subset,
                canonical_root / subset.dataset_id,
                source_manifest=source_manifest_root / "berka.json",
                adapter_source=adapter_source,
            )
    persisted_reports: dict[str, Any] = {}
    for report_path in sorted(canonical_root.glob("*/dataset_report.json")):
        persisted = json.loads(report_path.read_text(encoding="utf-8"))
        persisted_reports[persisted["dataset"]] = persisted
    acquisition_summary_path = (
        runtime_root / "manifests" / "acquisition" / "acquisition_summary.json"
    )
    acquisition_status = {}
    if acquisition_summary_path.is_file():
        acquisition_summary = json.loads(
            acquisition_summary_path.read_text(encoding="utf-8")
        )
        acquisition_status = {
            key: value["status"]
            for key, value in acquisition_summary.get("datasets", {}).items()
        }
    summary = {
        "schema_version": config["schema_version"],
        "config_sha256": sha256_file(config_path),
        "requested_this_run": requested,
        "datasets_materialized": sorted(persisted_reports),
        "acquisition_status": acquisition_status,
        "all_materialized_gates_pass": all(
            all(report["gates"].values()) for report in persisted_reports.values()
        ),
        "reports": persisted_reports,
    }
    _write_json(runtime_root / "manifests" / "materialization_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark_v2/cof_seqgen_saf_datasets.yaml"),
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    summary = materialize(root, (root / args.config).resolve(), args.datasets)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
